#include "gemm-fallback.cuh"

// Generic float matmul used when the library is built without cuBLAS
// (GGML_CUDA_NO_CUBLAS). Handles any shape, any byte strides, and the usual
// ne2/ne3 broadcast of src0 over src1. Plain shared-memory tiling with fp32
// accumulation -- correctness and coverage over speed. The fast float paths
// (mmv, mmvf, mmf) are still preferred by the dispatcher; this kernel only
// picks up the shapes they reject.

#define GEMM_FB_TILE 16

template <typename T>
static __device__ __forceinline__ float gemm_fb_load(const T * x) {
    return float(*x);
}

template <typename src0_t, typename src1_t>
static __global__ void gemm_fallback_kernel(
        const char * __restrict__ src0, const char * __restrict__ src1, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne11,
        const size_t  nb00, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t  nb10, const size_t nb11, const size_t nb12, const size_t nb13,
        const int64_t ne0,  const int64_t ne1,
        const size_t  nb2_dst_elts, const size_t nb3_dst_elts,
        const int64_t r2,   const int64_t r3,
        const int64_t ne12) {
    const int64_t i13 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const char * s0 = src0 + i02*nb02 + i03*nb03;
    const char * s1 = src1 + i12*nb12 + i13*nb13;
    float      * d  = dst  + i12*nb2_dst_elts + i13*nb3_dst_elts;

    const int64_t row = (int64_t) blockIdx.y*GEMM_FB_TILE + threadIdx.y; // index into ne01 (dst ne0)
    const int64_t col = (int64_t) blockIdx.x*GEMM_FB_TILE + threadIdx.x; // index into ne11 (dst ne1)

    __shared__ float tile_a[GEMM_FB_TILE][GEMM_FB_TILE + 1];
    __shared__ float tile_b[GEMM_FB_TILE][GEMM_FB_TILE + 1];

    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < ne00; k0 += GEMM_FB_TILE) {
        const int64_t ka = k0 + threadIdx.x;
        const int64_t kb = k0 + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < ne01 && ka < ne00)
            ? gemm_fb_load((const src0_t *)(s0 + row*nb01 + ka*nb00)) : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (col < ne11 && kb < ne00)
            ? gemm_fb_load((const src1_t *)(s1 + col*nb11 + kb*nb10)) : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < GEMM_FB_TILE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < ne0 && col < ne1) {
        d[col*ne0 + row] = acc;
    }
}


// Thin outputs (few dst elements, long K) collapse the tiled kernel's
// occupancy -- a [K,1]x[K,16] projection would occupy two blocks. One warp
// per output element with a K-parallel reduction keeps the GPU busy on
// exactly those shapes; the tiled kernel above remains better once the
// output is large enough to fill the grid.
template <typename src0_t, typename src1_t>
static __global__ void gemm_fallback_dot_kernel(
        const char * __restrict__ src0, const char * __restrict__ src1, float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne11,
        const size_t  nb00, const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t  nb10, const size_t nb11, const size_t nb12, const size_t nb13,
        const int64_t ne0,
        const size_t  nb2_dst_elts, const size_t nb3_dst_elts,
        const int64_t r2,   const int64_t r3,
        const int64_t ne12) {
    const int64_t i13 = blockIdx.z / ne12;
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const int64_t out_idx = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int64_t row = out_idx % ne01;
    const int64_t col = out_idx / ne01;
    if (col >= ne11) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const char * s0 = src0 + i02*nb02 + i03*nb03 + row*nb01;
    const char * s1 = src1 + i12*nb12 + i13*nb13 + col*nb11;

    float acc = 0.0f;
    for (int64_t k = lane; k < ne00; k += WARP_SIZE) {
        acc += float(*(const src0_t *)(s0 + k*nb00)) * float(*(const src1_t *)(s1 + k*nb10));
    }
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        (dst + i12*nb2_dst_elts + i13*nb3_dst_elts)[col*ne0 + row] = acc;
    }
}

void ggml_cuda_mul_mat_gemm_fallback(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(dst));

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];

    const dim3 block(GEMM_FB_TILE, GEMM_FB_TILE, 1);
    const dim3 grid(
        (unsigned)((src1->ne[1] + GEMM_FB_TILE - 1) / GEMM_FB_TILE),
        (unsigned)((src0->ne[1] + GEMM_FB_TILE - 1) / GEMM_FB_TILE),
        (unsigned)(src1->ne[2] * src1->ne[3]));

    cudaStream_t stream = ctx.stream();
    const size_t nb2d = dst->nb[2] / sizeof(float);
    const size_t nb3d = dst->nb[3] / sizeof(float);

    // few output elements + long reduction: use the warp-per-output kernel
    const int64_t n_out = src0->ne[1]*src1->ne[1];
    const bool thin = n_out <= 4096 && src0->ne[0] >= 256;

    auto launch = [&](auto s0_tag, auto s1_tag) {
        using S0 = decltype(s0_tag);
        using S1 = decltype(s1_tag);
        if (thin) {
            constexpr int warps_per_block = 8;
            const dim3 dot_block(warps_per_block*WARP_SIZE, 1, 1);
            const dim3 dot_grid(
                (unsigned)((n_out + warps_per_block - 1)/warps_per_block), 1,
                (unsigned)(src1->ne[2]*src1->ne[3]));
            gemm_fallback_dot_kernel<S0, S1><<<dot_grid, dot_block, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (float *) dst->data,
                src0->ne[0], src0->ne[1], src1->ne[1],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], nb2d, nb3d, r2, r3, src1->ne[2]);
            return;
        }
        gemm_fallback_kernel<S0, S1><<<grid, block, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (float *) dst->data,
            src0->ne[0], src0->ne[1], src1->ne[1],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->ne[0], dst->ne[1], nb2d, nb3d, r2, r3, src1->ne[2]);
    };

    auto with_src1 = [&](auto s0_tag) {
        switch (src1->type) {
            case GGML_TYPE_F32:  launch(s0_tag, float());       break;
            case GGML_TYPE_F16:  launch(s0_tag, half());        break;
            case GGML_TYPE_BF16: launch(s0_tag, nv_bfloat16()); break;
            default: GGML_ABORT("gemm fallback: unsupported src1 type %s", ggml_type_name(src1->type));
        }
    };

    switch (src0->type) {
        case GGML_TYPE_F32:  with_src1(float());       break;
        case GGML_TYPE_F16:  with_src1(half());        break;
        case GGML_TYPE_BF16: with_src1(nv_bfloat16()); break;
        default: GGML_ABORT("gemm fallback: unsupported src0 type %s", ggml_type_name(src0->type));
    }
}
