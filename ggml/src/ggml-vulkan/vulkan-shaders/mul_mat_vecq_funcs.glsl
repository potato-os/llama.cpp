#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(data_a[ib].d);
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
FLOAT_TYPEV2 get_dm(uint ib) {
    return FLOAT_TYPEV2(data_a_packed32[ib].dm);
}
#endif

#if defined(DATA_A_MXFP4)
FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e));
}
#endif

#if defined(DATA_A_Q2_K)
FLOAT_TYPEV2 get_dm(uint ib) {
    const uint ib_k = ib / 8;
    return FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
}
#endif

// Each iqs value maps to a 32-bit integer
#if defined(DATA_A_Q4_0)
// 2-byte loads for Q4_0 blocks (18 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - (8 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q4_1)
// 4-byte loads for Q4_1 blocks (20 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q5_0)
// 2-byte loads for Q5_0 blocks (22 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t((uint32_t(data_a_packed16[ib].qh[1]) << 16 | data_a_packed16[ib].qh[0]) >> (4 * iqs));
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - (16 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q5_1)
// 4-byte loads for Q5_1 blocks (24 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t(data_a_packed32[ib].qh >> (4 * iqs));
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
int32_t repack(uint ib, uint iqs) {
    return pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                          data_a_packed16[ib].qs[iqs * 2 + 1]));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(float(q_sum) * da * dsb.x);
}
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                      data_a[ib].qs[iqs * 4 + 1],
                                      data_a[ib].qs[iqs * 4 + 2],
                                      data_a[ib].qs[iqs * 4 + 3]));

    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

    return i32vec2(pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w])),
                   pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w])));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return FLOAT_TYPE(da * dsb.x * float(q_sum) * 0.5);
}
#endif

#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;
#if QUANT_R == 2
    const i32vec2 data_a_qs = repack(ib_a, iqs);
    q_sum += dotPacked4x8EXT(data_a_qs.x,
                             cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(data_a_qs.y,
                             cache_b_qs[1]);
#else
    int32_t data_a_qs = repack(ib_a, iqs * 2);
    q_sum += dotPacked4x8EXT(data_a_qs,
                             cache_b_qs[0]);
    data_a_qs = repack(ib_a, iqs * 2 + 1);
    q_sum += dotPacked4x8EXT(data_a_qs,
                             cache_b_qs[1]);
#endif

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, get_dm(ib_a), cache_b_ds, 4);
}
#endif

#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    return i32vec4((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303,
                   (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303);
}

uint8_t get_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    return data_a[ib_k].scales[iqs_k / 4];
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const uint8_t scale = get_scale(ib_a, iqs * 4);
    const vec2 dm = vec2(get_dm(ib_a));
    const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.

    sum_d += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[0]);

    sum_d += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[1]);

    sum_d += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[2]);

    sum_d += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]) * (scale & 0xF);
    sum_m += dotPacked4x8EXT(scale_m, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * (float(dm.x) * float(sum_d) - float(dm.y) * float(sum_m)));
}
#endif

// ---- Millie symmetric K-scale types. A 32-element sub-block is exactly one
// q8_1 block, so ib indexes 32-wide blocks and iqs (0/1) picks the 16-element
// half, as for the k-quants. Blocks are 2-byte aligned: codes are loaded through
// the packed16 view and rebuilt into 32-bit words.
//
// Helpers: spread 2-bit / 1-bit / 4-bit fields of a small integer into the four
// bytes of a word, then turn unsigned codes c into the odd integers 2c - m
// per byte with a borrow-free (SWAR) subtraction. This keeps the -1.5 / -0.5 /
// -7.5 offsets integer-exact inside the int8 dot, exactly as the CUDA kernels do.
#if defined(DATA_A_SYM)
uint sym_spread2(uint b) {      // 4 x 2-bit fields of a byte -> bytes
    b = (b | (b << 12)) & 0x000F000F;
    return (b | (b << 6)) & 0x03030303;
}
uint sym_spread1(uint n) {      // 4 bits of a nibble -> bytes
    return (n * 0x00204081) & 0x01010101;
}
uint sym_spread4(uint u) {      // 4 nibbles of a 16-bit value -> bytes
    u = (u | (u << 8)) & 0x00FF00FF;
    return (u | (u << 4)) & 0x0F0F0F0F;
}
int32_t sym_signed(uint v, uint m) { // per byte: 2*v - m, bytes of 2*v must be < 128
    return int32_t((((v << 1) | 0x80808080) - m) ^ 0x80808080);
}
#endif // DATA_A_SYM helpers

#if defined(DATA_A_Q2_SYM32K4)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib_k = ib_a / 8;
    const uint s    = ib_a % 8;
    const uint w16  = 4 * s + 2 * iqs;   // packed16 index of this 16-element half
    const uint w = uint(data_a_packed16[ib_k].qs[w16]) | (uint(data_a_packed16[ib_k].qs[w16 + 1]) << 16);

    int32_t sum = 0;
    [[unroll]] for (uint l = 0; l < 4; ++l) {
        // byte l of w holds elems 4l..4l+3 at shifts 0/2/4/6
        const int32_t q = sym_signed(sym_spread2((w >> (8 * l)) & 0xFF), 0x03030303); // 2c - 3
        sum += dotPacked4x8EXT(q, cache_b_qs[l]);
    }

    const uint scb = uint(data_a[ib_k].scales[s >> 1]);
    const float sc = float((s & 1) != 0 ? (scb >> 4) : (scb & 0xF));
    return FLOAT_TYPE(0.5f * float(data_a[ib_k].d) * sc * float(cache_b_ds.x) * float(sum));
}
#endif

#if defined(DATA_A_QT_MS32K4)
// Ternary mask+sign, K_PER_ITER == 32: one call covers a whole 32-weight
// sub-block (= one q8_1 block; iqs is always 0). The sign index runs across
// the whole 256-weight block: its start is the popcount of all preceding
// mask words, and the signs are scattered onto the mask bits one nibble at a
// time with the shared-memory table from types.glsl.
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib_k = ib_a / 8;
    const uint sb   = ib_a % 8;

    // sign start = popcount of all mask words before this sub-block
    uint start = 0;
    [[unroll]] for (uint k = 0; k < 16; ++k) {
        if (k < 2 * sb) {
            start += bitCount(uint(data_a_packed16[ib_k].mask[k]));
        }
    }
    const uint m0 = uint(data_a_packed16[ib_k].mask[2 * sb    ]);
    const uint m1 = uint(data_a_packed16[ib_k].mask[2 * sb + 1]);

    // up to 32 sign bits from bit `start` of the 128-bit sign field
    const uint i  = start >> 4;
    const uint sh = start & 15;
    const uint lo = uint(data_a_packed16[ib_k].signs[min(i,      7u)])
                 | (uint(data_a_packed16[ib_k].signs[min(i + 1u, 7u)]) << 16);
    const uint hi =  uint(data_a_packed16[ib_k].signs[min(i + 2u, 7u)]);
    uint sw = sh == 0 ? lo : ((lo >> sh) | (hi << (32 - sh)));

    // scatter the signs onto the mask positions, nibble by nibble
    uint sx0 = 0, sx1 = 0;
    [[unroll]] for (uint nib = 0; nib < 4; ++nib) {
        const uint mn = (m0 >> (4 * nib)) & 0xF;
        sx0 |= tern_scatter4[(mn << 4) | (sw & 0xF)] << (4 * nib);
        sw >>= bitCount(mn);
    }
    [[unroll]] for (uint nib = 0; nib < 4; ++nib) {
        const uint mn = (m1 >> (4 * nib)) & 0xF;
        sx1 |= tern_scatter4[(mn << 4) | (sw & 0xF)] << (4 * nib);
        sw >>= bitCount(mn);
    }

    int32_t sum = 0;
    [[unroll]] for (uint l = 0; l < 4; ++l) {
        // bytes: +1 at each non-zero, then -2 where negative => {-1, 0, +1}
        const uint mm0 = sym_spread1((m0  >> (4 * l)) & 0xF);
        const uint ss0 = sym_spread1((sx0 >> (4 * l)) & 0xF);
        sum += dotPacked4x8EXT(int32_t(((mm0 | 0x80808080u) - (ss0 + ss0)) ^ 0x80808080u), cache_b_qs[l]);
        const uint mm1 = sym_spread1((m1  >> (4 * l)) & 0xF);
        const uint ss1 = sym_spread1((sx1 >> (4 * l)) & 0xF);
        sum += dotPacked4x8EXT(int32_t(((mm1 | 0x80808080u) - (ss1 + ss1)) ^ 0x80808080u), cache_b_qs[4 + l]);
    }

    const uint scb = uint(data_a[ib_k].scales[sb >> 1]);
    const float sc = float((sb & 1) != 0 ? (scb >> 4) : (scb & 0xF));
    return FLOAT_TYPE(float(data_a[ib_k].d) * sc * float(cache_b_ds.x) * float(sum));
}
#endif

#if defined(DATA_A_Q1_SYM32K)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib_k = ib_a / 8;
    const uint s    = ib_a % 8;
    const uint q16  = uint(data_a_packed16[ib_k].qs[2 * s + iqs]); // 16 bits = 16 elems, LSB first

    int32_t sum = 0;
    [[unroll]] for (uint l = 0; l < 4; ++l) {
        const int32_t q = sym_signed(sym_spread1((q16 >> (4 * l)) & 0xF), 0x01010101); // 2b - 1
        sum += dotPacked4x8EXT(q, cache_b_qs[l]);
    }

    const uint scb = uint(data_a[ib_k].scales[s >> 1]);
    const float sc = float((s & 1) != 0 ? (scb >> 4) : (scb & 0xF));
    return FLOAT_TYPE(0.5f * float(data_a[ib_k].d) * sc * float(cache_b_ds.x) * float(sum));
}
#endif

#if defined(DATA_A_Q4_SYM16K)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    // 128-element blocks: 32-wide block ib_a covers sub-blocks 2*(ib_a%4) and +1,
    // and the 16-element half iqs is exactly one sub-block with its own scale.
    const uint ib_k = ib_a / 4;
    const uint s    = 2 * (ib_a % 4) + iqs;

    int32_t sum = 0;
    [[unroll]] for (uint l = 0; l < 4; ++l) {
        // pairwise nibbles: uint16 l of the sub-block = elems 4l..4l+3 in order
        const uint u = uint(data_a_packed16[ib_k].qs[4 * s + l]);
        const int32_t q = sym_signed(sym_spread4(u), 0x0F0F0F0F); // 2c - 15
        sum += dotPacked4x8EXT(q, cache_b_qs[l]);
    }

    const float sc = float(data_a[ib_k].scales[s]);
    return FLOAT_TYPE(0.5f * float(data_a[ib_k].d) * sc * float(cache_b_ds.x) * float(sum));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // bitwise OR to add 4 if hmask is set, subtract later
    const i8vec2 vals00 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2    ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2    ] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals01 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 1] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 1] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals10 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 2] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 2] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals11 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 3] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 3] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals20 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 4] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 4] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals21 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 5] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 5] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals30 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 6] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 6] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals31 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx  * 2 + 7] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[iqs * 2 + 7] >> hm_shift) & uint16_t(0x0101)) << 2));

    return i32vec4(pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y) - int8_t(4)),
                   pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y) - int8_t(4)),
                   pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y) - int8_t(4)),
                   pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y) - int8_t(4)));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const int8_t scale = int8_t(((data_a[ib_k].scales[is % 8      ] >> (4 * (is / 8))) & 0x0F0F) |
                               (((data_a[ib_k].scales[8 + (is % 4)] >> (2 * (is / 4))) & 0x0303) << 4));
    return float(data_a[ib_k].d) * float(scale - 32);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const float d_scale = get_d_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * d_scale * float(q_sum));
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F;

    return i32vec4(vals0, vals1, vals2, vals3);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs;
    const uint qh_shift = iqs_k / 8;

    return i32vec4(((data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx    ] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 1] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 2] >> qh_shift) & 0x01010101) << 4),
                   ((data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F) |
                  (((data_a_packed32[ib_k].qh[qh_idx + 3] >> qh_shift) & 0x01010101) << 4));
#endif
}

vec2 get_dm_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 8;

    const uvec3 scales = uvec3(data_a_packed32[ib_k].scales[0],
                               data_a_packed32[ib_k].scales[1],
                               data_a_packed32[ib_k].scales[2]);
    const uint scalesoffs = (is & 3) * 8;

    const uint scidx0 = (is < 4) ? 0 : 2;
    const uint scidxshift0 = scalesoffs;
    const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
    const uint mbidx0 = (is < 4) ? 1 : 2;
    const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
    const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

    const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
    const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
    u8vec2 scale_dm = u8vec2(sc, mbyte);

    return FLOAT_TYPEV2(data_a_packed32[ib_k].dm) * FLOAT_TYPEV2(scale_dm);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const vec2 dm_scale = get_dm_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(dm_scale.x) * float(q_sum) - float(dm_scale.y) * float(cache_b_ds.y / 2));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals10 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 2] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 2] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals11 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 3] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 3] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals20 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 4] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 4] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals21 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 5] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 5] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals30 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 6] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 6] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals31 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 7] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 7] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);

    return i32vec4(pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y)),
                   pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y)),
                   pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y)),
                   pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y)));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    return float(data_a[ib_k].d) * float(data_a[ib_k].scales[iqs_k / 4]);
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const i32vec4 qs_a = repack4(ib_a, iqs * 4);
    const float d_scale = get_d_scale(ib_a, iqs * 4);

    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(d_scale) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ1_S)
void repack8(uint ib, uint iqs, out i32vec4 out0, out i32vec4 out1) {
    const uint ib32 = iqs / 32;

    const uint qh = data_a[ib].qh[ib32];

    const uint qs16_0 = data_a_packed16[ib].qs[(4 * ib32 + 0) / 2];
    const uint qs16_1 = data_a_packed16[ib].qs[(4 * ib32 + 2) / 2];

    const uint qs0 = qs16_0 & 0xFF;
    const uint qs1 = qs16_0 >> 8;
    const uint qs2 = qs16_1 & 0xFF;
    const uint qs3 = qs16_1 >> 8;

    const uint hi0 = bitfieldExtract(qh, 3 * int(0), 3);
    const uint hi1 = bitfieldExtract(qh, 3 * int(1), 3);
    const uint hi2 = bitfieldExtract(qh, 3 * int(2), 3);
    const uint hi3 = bitfieldExtract(qh, 3 * int(3), 3);

    const int32_t grid0 = int32_t(iq1s_grid_gpu[qs0 | (hi0 << 8)]);
    const int32_t grid1 = int32_t(iq1s_grid_gpu[qs1 | (hi1 << 8)]);
    const int32_t grid2 = int32_t(iq1s_grid_gpu[qs2 | (hi2 << 8)]);
    const int32_t grid3 = int32_t(iq1s_grid_gpu[qs3 | (hi3 << 8)]);

    out0 = i32vec4((grid0 >> 0) & 0x0F0F0F0F,
                   (grid0 >> 4) & 0x0F0F0F0F,
                   (grid1 >> 0) & 0x0F0F0F0F,
                   (grid1 >> 4) & 0x0F0F0F0F);
    out1 = i32vec4((grid2 >> 0) & 0x0F0F0F0F,
                   (grid2 >> 4) & 0x0F0F0F0F,
                   (grid3 >> 0) & 0x0F0F0F0F,
                   (grid3 >> 4) & 0x0F0F0F0F);
}

vec2 get_dm(uint ib, uint iqs) {
    const uint ib32 = iqs / 32;

    const uint qh = data_a[ib].qh[ib32];
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;

    const float d = float(data_a[ib].d);
    const float dl = d * float(2 * bitfieldExtract(qh, 12, 3) + 1);

    // the -1 cancels out the bias in iq1s_grid_gpu
    return FLOAT_TYPEV2(dl, dl * (delta - 1));
}

FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    int32_t q_sum = 0;

    const uint ib_k = ib_a / 8;
    const uint iqs_k = (ib_a % 8) * 32 + iqs * 32;

    i32vec4 qs_a0;
    i32vec4 qs_a1;
    repack8(ib_k, iqs_k, qs_a0, qs_a1);

    const vec2 dm = get_dm(ib_k, iqs_k);

    q_sum += dotPacked4x8EXT(qs_a0.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a0.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a0.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a0.w, cache_b_qs[3]);
    q_sum += dotPacked4x8EXT(qs_a1.x, cache_b_qs[4]);
    q_sum += dotPacked4x8EXT(qs_a1.y, cache_b_qs[5]);
    q_sum += dotPacked4x8EXT(qs_a1.z, cache_b_qs[6]);
    q_sum += dotPacked4x8EXT(qs_a1.w, cache_b_qs[7]);

    return FLOAT_TYPE(float(cache_b_ds.x) * float(dm.x) * float(q_sum) + float(dm.y) * float(cache_b_ds.y));
}
#endif

#if defined(DATA_A_IQ1_M)
FLOAT_TYPE mmvq_dot_product(const uint ib_a, const uint iqs) {
    const uint ib_k = ib_a / 8;
    const uint iqs_k = (ib_a % 8) * 32 + iqs * 32;

    const uint ib32 = iqs_k / 32;
    const uint ib64 = ib32 / 2;

    const uint16_t[4] scales = data_a[ib_k].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);

    const uint qs32 = data_a_packed32[ib_k].qs[ib32];
    const uint qh16 = data_a_packed16[ib_k].qh[ib32];

    float sum = 0;
    const uint sc = data_a[ib_k].scales[ib64];
    [[unroll]] for (int l = 0; l < 4; ++l) {
        const uint ib16 = 2 * ib32 + l / 2;
        const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
        const uint qh = qh16 >> (4 * l);
        const uint qs = (qs32 >> (8 * l)) & 0xFF;
        const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;

        const int32_t grid = int32_t(iq1s_grid_gpu[qs | ((qh & 7) << 8)]);

        int32_t q_sum = 0;
        q_sum += dotPacked4x8EXT((grid >> 0) & 0x0F0F0F0F, cache_b_qs[2 * l + 0]);
        q_sum += dotPacked4x8EXT((grid >> 4) & 0x0F0F0F0F, cache_b_qs[2 * l + 1]);

        int32_t y_sum = 0;
        y_sum += dotPacked4x8EXT(int(0x01010101), cache_b_qs[2 * l + 0]);
        y_sum += dotPacked4x8EXT(int(0x01010101), cache_b_qs[2 * l + 1]);

        // the -1 cancels out the bias in iq1s_grid_gpu
        sum += dl * (q_sum + y_sum * (delta - 1));
    }
    sum *= float(cache_b_ds.x);

    return sum;
}
#endif
