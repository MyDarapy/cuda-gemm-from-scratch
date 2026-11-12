#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

constexpr int ceil_div(int m, int n)
{
    return (m + n - 1) / n;
}

constexpr int WARPSIZE = 32;

// Shared host-side validation for every sgemm_* entry point.
// A: M x K, B: K x N, C: M x N, all fp32, row-major, contiguous, on the same GPU.
inline void check_gemm_inputs(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                              const torch::Tensor &output_matrix)
{
    TORCH_CHECK(matrix_a.is_cuda() && matrix_b.is_cuda() && output_matrix.is_cuda(),
                "A, B and C must be CUDA tensors");
    TORCH_CHECK(matrix_a.scalar_type() == torch::kFloat32 &&
                    matrix_b.scalar_type() == torch::kFloat32 &&
                    output_matrix.scalar_type() == torch::kFloat32,
                "A, B and C must be float32");
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2 && output_matrix.dim() == 2,
                "A, B and C must be 2D");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous() && output_matrix.is_contiguous(),
                "A, B and C must be contiguous (row-major)");
    TORCH_CHECK(matrix_b.size(0) == matrix_a.size(1),
                "A is MxK so B must be KxN, got A ", matrix_a.sizes(), " B ", matrix_b.sizes());
    TORCH_CHECK(output_matrix.size(0) == matrix_a.size(0) && output_matrix.size(1) == matrix_b.size(1),
                "C must be MxN, got ", output_matrix.sizes());
}
