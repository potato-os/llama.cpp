#pragma once
#include "common.cuh"

void ggml_cuda_mul_mat_vec_q2sym16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_cuda_should_use_mmvq2sym16(const int64_t * src0_ne, int64_t ne11);
