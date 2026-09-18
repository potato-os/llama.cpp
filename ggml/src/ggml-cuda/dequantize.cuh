#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q4_sym16(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_sym16 * x = (const block_q4_sym16 *) vx;

    // pair (elem iqs, elem iqs+16): group 0 and group 1 of the same block
    const int  b  = iqs & 7;
    const bool hi = iqs >= 8;
    const uint8_t b0 = x[ib].qs[b];
    const uint8_t b1 = x[ib].qs[8 + b];
    const int q0 = hi ? (b0 >> 4) : (b0 & 0x0F);
    const int q1 = hi ? (b1 >> 4) : (b1 & 0x0F);

    v.x = ggml_cuda_ue4m3_to_fp32(x[ib].d[0]) * (q0 - 7.5f);
    v.y = ggml_cuda_ue4m3_to_fp32(x[ib].d[1]) * (q1 - 7.5f);
}

static __device__ __forceinline__ void dequantize_q2_sym32k4(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_sym32k4 * x = (const block_q2_sym32k4 *) vx;

    // pair (elem iqs, elem iqs+128), iqs in [0,128); byte j/4 holds elem j at shift 2*(j%4)
    const float d = __half2float(x[ib].d);
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int j  = iqs + 128*p;
        const int s  = j / QK2_SYM32K_SUB;
        const uint8_t sc = (s & 1) ? (x[ib].scales[s >> 1] >> 4) : (x[ib].scales[s >> 1] & 0x0F);
        const float ds = d * (float) sc;
        const int c = (x[ib].qs[j/4] >> (2*(j%4))) & 3;
        const float w = ((float) c - 1.5f) * ds;
        if (p == 0) { v.x = w; } else { v.y = w; }
    }
}

static __device__ __forceinline__ void dequantize_q1_sym32k(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_sym32k * x = (const block_q1_sym32k *) vx;

    // pair (elem iqs, elem iqs+128), iqs in [0,128); bit j of qs[j/8], LSB first
    const float d = __half2float(x[ib].d);
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int j  = iqs + 128*p;
        const int s  = j / QK1_SYM32K_SUB;
        const uint8_t sc = (s & 1) ? (x[ib].scales[s >> 1] >> 4) : (x[ib].scales[s >> 1] & 0x0F);
        const float h = 0.5f * d * (float) sc;
        const float w = ((x[ib].qs[j/8] >> (j%8)) & 1) ? h : -h;
        if (p == 0) { v.x = w; } else { v.y = w; }
    }
}

static __device__ __forceinline__ void dequantize_q4_sym16k(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_sym16k * x = (const block_q4_sym16k *) vx;

    // pair (elem iqs, elem iqs+64), iqs in [0,64); PAIRWISE nibbles: byte j
    // of a sub-block holds elems 2j (low) and 2j+1 (high)
    const float d = __half2float(x[ib].d);
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int j  = iqs + 64*p;
        const int s  = j / QK4_SYM16K_SUB;
        const int jj = j % QK4_SYM16K_SUB;
        const uint8_t b = x[ib].qs[s*(QK4_SYM16K_SUB/2) + jj/2];
        const int q = (jj & 1) ? (b >> 4) : (b & 0x0F);
        const float w = ((float) q - 7.5f) * d * (float) x[ib].scales[s];
        if (p == 0) { v.x = w; } else { v.y = w; }
    }
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}
