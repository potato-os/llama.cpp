#if defined(DATA_A_Q4_0)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_Q4_1)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q5_0)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    uint32_t qh;
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_Q5_1)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    uint32_t qh;
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q8_0)
#define QUANT_R_MMQ 1
// AMD likes 4, Intel likes 1 and Nvidia likes 2
// #define BK_STEP 1
struct block_a_cache {
    int32_t qs[32/4];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_IQ4_NL)
#define QUANT_R_MMQ 2
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_MXFP4)
#define QUANT_R_MMQ 2
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPE d;
};
#elif defined(DATA_A_Q2_K)
#define QUANT_R_MMQ 4
struct block_a_cache {
    uint32_t qs[2];
    u8vec2 scales;
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q2_SYM32K4)
#define QUANT_R_MMQ 4
struct block_a_cache {
    uint32_t qs[2];      // q2_k-style packing: bits 2k..2k+1 of byte m = elem 4k+m
    FLOAT_TYPE dsc;      // 0.5 * d * sub-scale
};
#elif defined(DATA_A_QT_MS32K4)
#define QUANT_R_MMQ 8
struct block_a_cache {
    int32_t qs[8];       // precomputed {-1,0,+1} bytes: scatter cost paid once
                         // per tile load instead of per dot product
    FLOAT_TYPE dsc;      // d * sub-scale
};
#elif defined(DATA_A_Q1_SYM32K)
#define QUANT_R_MMQ 8
struct block_a_cache {
    uint32_t qs;         // raw 32 bits, LSB first
    FLOAT_TYPE dsc;      // 0.5 * d * sub-scale
};
#elif defined(DATA_A_Q4_SYM16K)
#define QUANT_R_MMQ 4
struct block_a_cache {
    uint32_t qs[4];      // raw pairwise nibbles: nibble n of word l = elem 8l+n
    FLOAT_TYPEV2 dsc;    // 0.5 * d * sub-scale for the two 16-element sub-blocks
};
#elif defined(DATA_A_Q3_K)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[4];
    FLOAT_TYPEV2 d_scales;
};
#elif defined(DATA_A_Q4_K)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[4];
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q5_K)
#define QUANT_R_MMQ 1
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q6_K)
#define QUANT_R_MMQ 1
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPEV2 d_scales;
};
#endif

struct block_b_cache
{
    int32_t qs[8];
    FLOAT_TYPEV2 ds;
};
