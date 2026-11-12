#pragma once
#include <torch/torch.h>

// Every kernel exposes the same host entry point, declared here:
//     C = alpha * (A @ B) + beta * C
// A: M x K, B: K x N, C: M x N, fp32, row-major, contiguous, on the GPU.
// bench_matmul.py binds each one (see its KERNELS table) and calls it as gemm(A, B, C, alpha, beta).

// 01 - naive: one thread per output element
void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                 torch::Tensor &output_matrix, float alpha, float beta);

// 02 - global memory coalescing: consecutive threads compute consecutive columns of C
void sgemm_coalesced(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                     torch::Tensor &output_matrix, float alpha, float beta);

// 03 - shared memory caching: each block stages block_size x block_size tiles of A and B in shared memory
void sgemm_shared_mem(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &matrix_c, float alpha, float beta);
