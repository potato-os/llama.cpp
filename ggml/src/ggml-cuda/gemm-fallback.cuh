#pragma once

#include "common.cuh"

// Any-shape float matmul (f32/f16/bf16 src0, f32 src1/dst) used by builds
// without cuBLAS. Correctness-first tiled kernel; see gemm-fallback.cu.
void ggml_cuda_mul_mat_gemm_fallback(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
