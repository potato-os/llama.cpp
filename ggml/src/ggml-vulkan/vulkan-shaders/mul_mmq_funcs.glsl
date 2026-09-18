#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// Each iqs value maps to a 32-bit integer

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
// 2-byte loads for Q4_0 blocks (18 bytes)
// 4-byte loads for Q4_1 blocks (20 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q4_0
    buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
#else // DATA_A_Q4_1
    buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib].dm);
    }
#endif
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const i32vec2 qs_a = i32vec2( vui       & 0x0F0F0F0F,
                                     (vui >> 4) & 0x0F0F0F0F);

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += dotPacked4x8EXT(qs_a.x, qs_b0);
        q_sum += dotPacked4x8EXT(qs_a.y, qs_b1);
    }

#ifdef DATA_A_Q4_0
    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - 8.0 * float(cache_b.ds.y)));
#else // DATA_A_Q4_1
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm.x) * float(cache_b.ds.x) + float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
#endif
}
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
// 2-byte loads for Q5_0 blocks (22 bytes)
// 4-byte loads for Q5_1 blocks (24 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q5_0
    buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
        buf_a[buf_ib].qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
    }
#else // DATA_A_Q5_1
    buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib].dm);
        buf_a[buf_ib].qh = data_a_packed32[ib].qh;
    }
#endif
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].qh = buf_a[buf_ib].qh;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const int32_t qh = int32_t(cache_a[ib_a].qh >> (4 * iqs));
        const int32_t qs_a0 = int32_t(vui & 0x0F0F0F0F)
                         | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)
        const int32_t qs_a1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                         | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += dotPacked4x8EXT(qs_a0, qs_b0);
        q_sum += dotPacked4x8EXT(qs_a1, qs_b1);
    }

#ifdef DATA_A_Q5_0
    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - 16.0 * float(cache_b.ds.y)));
#else // DATA_A_Q5_1
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm.x) * float(cache_b.ds.x) + float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
#endif
}
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    buf_a[buf_ib].qs[iqs] = pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
        const int32_t qs_b = cache_b.qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, qs_b);
    }

    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm) * float(cache_b.ds.x));
}
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                      data_a[ib].qs[iqs * 4 + 1],
                                      data_a[ib].qs[iqs * 4 + 2],
                                      data_a[ib].qs[iqs * 4 + 3]));

    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

    buf_a[buf_ib].qs[iqs    ] = pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w]));
    buf_a[buf_ib].qs[iqs + 4] = pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w]));

    if (iqs == 0) {
        buf_a[buf_ib].d = FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e) * 0.5);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

// For k-quants, ib and iqs still assume 32-wide blocks, but k-quants are 256-wide
// iqs still refers to a 32-bit integer, meaning 0..7 for 32-wide quants
#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    // Repack 4x4 quants into one int
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 2) | (vals2 << 4) | (vals3 << 6);

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
        buf_a[buf_ib].scales = unpack8(uint32_t(data_a_packed16[ib_k].scales[iqs_k / 8])).xy; // vec4 used due to #12147
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].scales = buf_a[buf_ib].scales;

    [[unroll]] for (uint iqs = 0; iqs < 2; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const uint8_t scale = cache_a[ib_a].scales[iqs / 4];
        const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x03030303);

        sum_d += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]) * (scale & 0xF);
        sum_m += dotPacked4x8EXT(scale_m, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].dm.x) * float(sum_d) - float(cache_a[ib_a].dm.y) * float(sum_m)));
}
#endif

// ---- Millie symmetric K-scale types (see mul_mat_vecq_funcs.glsl for the
// bit tricks). ib indexes 32-wide blocks; iqs picks the 16-element half. ----
#if defined(DATA_A_SYM)
uint sym_spread2(uint b) {
    b = (b | (b << 12)) & 0x000F000F;
    return (b | (b << 6)) & 0x03030303;
}
uint sym_spread1(uint n) {
    return (n * 0x00204081) & 0x01010101;
}
uint sym_spread4(uint u) {
    u = (u | (u << 8)) & 0x00FF00FF;
    return (u | (u << 4)) & 0x0F0F0F0F;
}
int32_t sym_signed(uint v, uint m) {
    return int32_t((((v << 1) | 0x80808080) - m) ^ 0x80808080);
}
#endif

#if defined(DATA_A_Q2_SYM32K4)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint s    = ib % 8;
    const uint w16  = 4 * s + 2 * iqs;
    const uint w = uint(data_a_packed16[ib_k].qs[w16]) | (uint(data_a_packed16[ib_k].qs[w16 + 1]) << 16);

    // sequential words for elems 4l..4l+3, then q2_k-style packing so the dot
    // product can peel them off with a shift and a mask
    const uint v0 = sym_spread2((w >>  0) & 0xFF);
    const uint v1 = sym_spread2((w >>  8) & 0xFF);
    const uint v2 = sym_spread2((w >> 16) & 0xFF);
    const uint v3 = sym_spread2((w >> 24) & 0xFF);
    buf_a[buf_ib].qs[iqs] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);

    if (iqs == 0) {
        const uint scb = uint(data_a[ib_k].scales[s >> 1]);
        const float sc = float((s & 1) != 0 ? (scb >> 4) : (scb & 0xF));
        buf_a[buf_ib].dsc = FLOAT_TYPE(0.5f * float(data_a[ib_k].d) * sc);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dsc = buf_a[buf_ib].dsc;
    cache_a[reg_ib].qs[0] = buf_a[buf_ib].qs[0];
    cache_a[reg_ib].qs[1] = buf_a[buf_ib].qs[1];
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum = 0;
    [[unroll]] for (uint k = 0; k < 8; ++k) {
        const uint v = (cache_a[ib_a].qs[k / 4] >> ((k % 4) * 2)) & 0x03030303;
        sum += dotPacked4x8EXT(sym_signed(v, 0x03030303), cache_b.qs[k]);
    }
    return ACC_TYPE(float(cache_b.ds.x) * float(cache_a[ib_a].dsc) * float(sum));
}
#endif

#if defined(DATA_A_QT_MS32K4)
// One thread per 32-weight sub-block (QUANT_R_MMQ == 8): prefix popcount,
// sign window and scatter run once per tile load; the dot product is 8 plain
// dp4a against precomputed {-1,0,+1} bytes. tern_scatter4 lives in shared
// memory (types.glsl).
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint sb   = ib % 8;

    uint start = 0;
    for (uint k = 0; k < 2 * sb; ++k) {
        start += bitCount(uint(data_a_packed16[ib_k].mask[k]));
    }
    const uint m0 = uint(data_a_packed16[ib_k].mask[2 * sb    ]);
    const uint m1 = uint(data_a_packed16[ib_k].mask[2 * sb + 1]);

    const uint i  = start >> 4;
    const uint sh = start & 15;
    const uint lo = uint(data_a_packed16[ib_k].signs[min(i,      7u)])
                 | (uint(data_a_packed16[ib_k].signs[min(i + 1u, 7u)]) << 16);
    const uint hi =  uint(data_a_packed16[ib_k].signs[min(i + 2u, 7u)]);
    uint sw = sh == 0 ? lo : ((lo >> sh) | (hi << (32 - sh)));

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

    [[unroll]] for (uint l = 0; l < 4; ++l) {
        const uint mm0 = sym_spread1((m0  >> (4 * l)) & 0xF);
        const uint ss0 = sym_spread1((sx0 >> (4 * l)) & 0xF);
        buf_a[buf_ib].qs[l]     = int32_t(((mm0 | 0x80808080u) - (ss0 + ss0)) ^ 0x80808080u);
        const uint mm1 = sym_spread1((m1  >> (4 * l)) & 0xF);
        const uint ss1 = sym_spread1((sx1 >> (4 * l)) & 0xF);
        buf_a[buf_ib].qs[4 + l] = int32_t(((mm1 | 0x80808080u) - (ss1 + ss1)) ^ 0x80808080u);
    }

    const uint scb = uint(data_a[ib_k].scales[sb >> 1]);
    const float sc = float((sb & 1) != 0 ? (scb >> 4) : (scb & 0xF));
    buf_a[buf_ib].dsc = FLOAT_TYPE(float(data_a[ib_k].d) * sc);
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dsc = buf_a[buf_ib].dsc;
    [[unroll]] for (uint k = 0; k < 8; ++k) {
        cache_a[reg_ib].qs[k] = buf_a[buf_ib].qs[k];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum = 0;
    [[unroll]] for (uint k = 0; k < 8; ++k) {
        sum += dotPacked4x8EXT(cache_a[ib_a].qs[k], cache_b.qs[k]);
    }
    return ACC_TYPE(float(cache_b.ds.x) * float(cache_a[ib_a].dsc) * float(sum));
}
#endif

#if defined(DATA_A_Q1_SYM32K)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    // one thread per 32-wide block (QUANT_R_MMQ == 8)
    const uint ib_k = ib / 8;
    const uint s    = ib % 8;
    buf_a[buf_ib].qs = uint(data_a_packed16[ib_k].qs[2 * s]) | (uint(data_a_packed16[ib_k].qs[2 * s + 1]) << 16);

    const uint scb = uint(data_a[ib_k].scales[s >> 1]);
    const float sc = float((s & 1) != 0 ? (scb >> 4) : (scb & 0xF));
    buf_a[buf_ib].dsc = FLOAT_TYPE(0.5f * float(data_a[ib_k].d) * sc);
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dsc = buf_a[buf_ib].dsc;
    cache_a[reg_ib].qs  = buf_a[buf_ib].qs;
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum = 0;
    [[unroll]] for (uint k = 0; k < 8; ++k) {
        const int32_t q = sym_signed(sym_spread1((cache_a[ib_a].qs >> (4 * k)) & 0xF), 0x01010101);
        sum += dotPacked4x8EXT(q, cache_b.qs[k]);
    }
    return ACC_TYPE(float(cache_b.ds.x) * float(cache_a[ib_a].dsc) * float(sum));
}
#endif

#if defined(DATA_A_Q4_SYM16K)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    // 128-element blocks; the 16-element half iqs is sub-block s with its own scale
    const uint ib_k = ib / 4;
    const uint s    = 2 * (ib % 4) + iqs;
    buf_a[buf_ib].qs[2 * iqs    ] = uint(data_a_packed16[ib_k].qs[4 * s    ]) | (uint(data_a_packed16[ib_k].qs[4 * s + 1]) << 16);
    buf_a[buf_ib].qs[2 * iqs + 1] = uint(data_a_packed16[ib_k].qs[4 * s + 2]) | (uint(data_a_packed16[ib_k].qs[4 * s + 3]) << 16);

    if (iqs == 0) {
        const float d = float(data_a[ib_k].d);
        buf_a[buf_ib].dsc = FLOAT_TYPEV2(0.5f * d * float(data_a[ib_k].scales[s]),
                                         0.5f * d * float(data_a[ib_k].scales[s + 1]));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dsc = buf_a[buf_ib].dsc;
    [[unroll]] for (uint k = 0; k < 4; ++k) {
        cache_a[reg_ib].qs[k] = buf_a[buf_ib].qs[k];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0f;
    [[unroll]] for (uint h = 0; h < 2; ++h) {
        int32_t sum = 0;
        [[unroll]] for (uint l = 0; l < 2; ++l) {
            const uint u = cache_a[ib_a].qs[2 * h + l];
            sum += dotPacked4x8EXT(sym_signed(sym_spread4(u & 0xFFFF), 0x0F0F0F0F), cache_b.qs[4 * h + 2 * l    ]);
            sum += dotPacked4x8EXT(sym_signed(sym_spread4(u >> 16),    0x0F0F0F0F), cache_b.qs[4 * h + 2 * l + 1]);
        }
        result += float(cache_a[ib_a].dsc[h]) * float(sum);
    }
    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint hm_idx = iqs * QUANT_R_MMQ;
    const uint iqs_k = (ib % 8) * 8 + hm_idx;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // Repack 2x4 quants into one int
    // Add the 3rd bit instead of subtracting it to allow packing the quants
    // vec4 for unpack8 used due to #12147
    const i8vec2 vals00 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2        ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2    ] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals01 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 1    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 1] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals10 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 2    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 2] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    const i8vec2 vals11 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 3    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                          unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 3] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
    buf_a[buf_ib].qs[iqs] = pack32(u8vec4(vals00.x, vals00.y, vals01.x, vals01.y)) |
                           (pack32(u8vec4(vals10.x, vals10.y, vals11.x, vals11.y)) << 4);

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = i8vec2(unpack8(uint32_t(((data_a_packed16[ib_k].scales[(is % 8      ) / 2] >> (4 * (is / 8))) & 0x0F0F) |
                                                     (((data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2] >> (2 * (is / 4))) & 0x0303) << 4))).xy); // vec4 used due to #12147

        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales - 32));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        // Subtract 4 from the quants to correct the 3rd bit offset
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    // Repack 2x4 quants into one int
#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 4);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs * QUANT_R_MMQ;
    const uint qh_shift = iqs_k / 8;

    buf_a[buf_ib].qs[iqs] = int32_t(((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F) |
                                   (((data_a_packed32[ib_k].qh[qh_idx] >> qh_shift) & 0x01010101) << 4));
#endif

    if (iqs == 0) {
        // Scale index
        const uint is = iqs_k / 8;
        u8vec2 scale_dm;
        if (is < 4) {
            scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
        } else {
            scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
                              (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
        }

        buf_a[buf_ib].dm = FLOAT_TYPEV2(vec2(data_a_packed32[ib_k].dm) * vec2(scale_dm));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8 / QUANT_R_MMQ; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
#if defined(DATA_A_Q4_K)
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
#else // defined(DATA_A_Q5_K)
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
#endif

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * float(cache_a[ib_a].dm.x) * float(q_sum) - float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))).xy |
                          unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4)).xy) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))).xy |
                          unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4)).xy) - int8_t(32);
    buf_a[buf_ib].qs[iqs] = pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y));

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = unpack8(int32_t(data_a_packed16[ib_k].scales[is / 2])).xy;

        buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales));
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_outer = ib / 4;
        const uint ib_inner = ib % 4;

        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(data_b[ib_outer].ds[ib_inner]);
        }

        const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
        buf_b[buf_ib].qs[iqs * 4    ] = values.x;
        buf_b[buf_ib].qs[iqs * 4 + 1] = values.y;
        buf_b[buf_ib].qs[iqs * 4 + 2] = values.z;
        buf_b[buf_ib].qs[iqs * 4 + 3] = values.w;
    } else {
        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(0.0f);
        }

        buf_b[buf_ib].qs[iqs * 4    ] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 1] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 2] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 3] = 0;
    }
}

void block_b_to_registers(const uint ib) {
    cache_b.ds = buf_b[ib].ds;
    [[unroll]] for (uint iqs = 0; iqs < BK / 4; iqs++) {
        cache_b.qs[iqs] = buf_b[ib].qs[iqs];
    }
}
