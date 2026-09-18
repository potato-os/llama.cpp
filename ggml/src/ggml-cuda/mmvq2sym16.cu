#include "ggml.h"
#include "common.cuh"
#include "mmvq2sym16.cuh"

// Fused mul_mat_vec kernel for Q2_SYM16. Unpack and dequantize codes inside
// the dot-product loop against f32 activations, without materializing dense
// weights. Row/channel/sample indexing follows mmvf.cu's mul_mat_vec_f.
// One physical warp computes each output row; ncols_dst=1 is supported.

static __device__ __forceinline__ float ggml_ue4m3_to_fp32_device(uint8_t x) {
    // Exact copy of ggml_ue4m3_to_fp32 (ggml-impl.h), duplicated here for
    // the same reason as convert.cu's dequantize_block_q2_sym16: that
    // header's version is host-only static inline, not device-callable
    // without risking a shared header used by the CPU build too.
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

template <int block_size>
static __global__ void mul_mat_vec_q2sym16(
        const block_q2_sym16 * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols_x, const uint3 nchannels_y, const int stride_row_blocks, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride) {
    const int row         = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int tid          = threadIdx.x;
    constexpr int warp_size = 32;

    const int token_idx  = ids ? blockIdx.z                                : 0;
    const int channel_x  = ids ? ids[channel_dst + token_idx * ids_stride] : fastdiv((uint32_t) channel_dst, channel_ratio);
    const int channel_y  = ids ? fastmodulo(channel_dst, nchannels_y)      : channel_dst;
    const int sample_dst = ids ? 0                                        : blockIdx.z;
    const int sample_x   = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y   = sample_dst;

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + int64_t(row)*stride_row_blocks;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if (ids) {
        y   += int64_t(token_idx) * 0; // ncols_dst==1: nothing to add, kept for clarity vs mmvf's token stride
        dst += token_idx*stride_col_dst;
    }

    const int n_blocks = ncols_x / QK2_SYM16;
    float sumf = 0.0f;
    for (int ib = tid; ib < n_blocks; ib += block_size) {
        const block_q2_sym16 blk = x[ib];
        const float d = ggml_ue4m3_to_fp32_device(blk.d);
        const float * yb = y + ib*QK2_SYM16;
#pragma unroll
        for (int j = 0; j < QK2_SYM16; ++j) {
            const int code = (blk.qs[j/4] >> ((j%4)*2)) & 3;
            sumf += d * ((float) code - 1.5f) * yb[j];
        }
    }

    sumf = warp_reduce_sum(sumf);

    if constexpr (block_size > warp_size) {
        __shared__ float buf[block_size / warp_size];
        if (tid % warp_size == 0) {
            buf[tid / warp_size] = sumf;
        }
        __syncthreads();
        if (tid < warp_size) {
            sumf = (tid < block_size / warp_size) ? buf[tid] : 0.0f;
            sumf = warp_reduce_sum(sumf);
        }
    }

    if (tid == 0) {
        dst[row] = sumf;
    }
}

bool ggml_cuda_should_use_mmvq2sym16(const int64_t * src0_ne, int64_t ne11) {
    // Keep the plain MUL_MAT broadcast path on the dequantize-plus-cuBLAS
    // fallback: this kernel does not support all channel/broadcast layouts.
    // Routed MUL_MAT_ID dispatch is handled separately.
    GGML_UNUSED(src0_ne);
    GGML_UNUSED(ne11);
    return false;
}

void ggml_cuda_mul_mat_vec_q2sym16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q2_SYM16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int64_t ncols_x  = ne00;
    const int64_t nrows_x  = ne01;
    const int64_t ncols_dst = ids ? ne2 : ne11;
    GGML_ASSERT(ncols_dst == 1);

    const int64_t stride_row_blocks = nb01 / sizeof(block_q2_sym16);
    const int64_t stride_col_y      = ids ? ne10 : ne10; // src1 rows are contiguous f32
    GGML_UNUSED(stride_col_y);

    const int64_t nchannels_x = ids ? ne02 : ne12;
    const int64_t nchannels_y = ids ? ne11 : ne12;
    const int64_t nchannels_dst = ids ? ids->ne[0] : ne2;
    const uint3 channel_ratio = ids ? make_uint3(0,0,0) : init_fastdiv_values((uint32_t)(nchannels_dst / nchannels_x));
    const uint3 nchannels_y_fd = init_fastdiv_values((uint32_t) nchannels_y);

    const int64_t stride_channel_x = ids ? nb02 / sizeof(block_q2_sym16) : nb02 / sizeof(block_q2_sym16);
    const int64_t stride_channel_y = ids ? ne10 : ne10*ne11;
    const int64_t stride_channel_dst = ne0;

    const int64_t nsamples_x = ne03;
    const int64_t nsamples_dst = ids ? 1 : ne3;
    const uint3 sample_ratio = init_fastdiv_values((uint32_t)(nsamples_dst / nsamples_x));
    const int64_t stride_sample_x = nb03 / sizeof(block_q2_sym16);
    const int64_t stride_sample_y = ids ? ne10*ne11 : ne10*ne11*ne12;
    const int64_t stride_sample_dst = ids ? 0 : ne0*ne1;

    const int64_t ids_stride = ids ? ids->nb[1] / sizeof(int32_t) : 0;

    const block_q2_sym16 * src0_d = (const block_q2_sym16 *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const int32_t * ids_d = ids ? (const int32_t *) ids->data : nullptr;
    float * dst_d = (float *) dst->data;

    // Block size scales with K (n_blocks = ncols_x/16): more warps per row
    // gives more memory-level parallelism for larger reduction dimensions,
    // at the cost of a cross-warp shared-memory reduction step.
    const int64_t n_blocks = ncols_x / QK2_SYM16;
    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
    cudaStream_t stream = ctx.stream();

    if (n_blocks >= 64) {
        mul_mat_vec_q2sym16<128><<<block_nums, dim3(128,1,1), 0, stream>>>(
            src0_d, src1_d, ids_d, dst_d,
            ncols_x, nchannels_y_fd, stride_row_blocks, (int) stride_channel_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
            ids_stride);
    } else {
        mul_mat_vec_q2sym16<32><<<block_nums, dim3(32,1,1), 0, stream>>>(
            src0_d, src1_d, ids_d, dst_d,
            ncols_x, nchannels_y_fd, stride_row_blocks, (int) stride_channel_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
            ids_stride);
    }

    GGML_UNUSED_VARS(ncols_dst);
}
