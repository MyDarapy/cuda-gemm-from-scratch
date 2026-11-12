#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"



template <const uint block_size>
__global__ void sgemm_gmem_coaleased_kernel(int num_row_a, int num_col_b, int num_col_a,
                                    float alpha, const float* matrix_a, 
                                    const float* matrix_b, float beta, float* output_matrix)
                                    {
                                        // Map 1D threads unto 2D output Each thread is responsible for a portion of the 2D output position                                         
                                        // Global memory coleasing loads. 
                                        const int output_row = blockIdx.y * block_size + (threadIdx.x / block_size);
                                        const int output_col = blockIdx.x * block_size + (threadIdx.x % block_size);

                                        // Boundary check for non multiple of block size output = (num_row_a by num_col_b)
                                        if (output_row < num_row_a && output_col < num_col_b){

                                            float accumulator = 0.0f;
                                            for (int k_idx = 0; k_idx < num_col_a; k_idx++){
                                                accumulator += matrix_a[num_col_a * output_row + k_idx] * matrix_b[k_idx * num_col_b + output_col];
                                            }
                                            
                                                const int output_index = output_row * num_col_b + output_col;
                                                output_matrix[output_index] = alpha*accumulator + beta*output_matrix[output_index];
                                    
                                        }


                                    }



// Host entry point (declared in gemm_kernels.cuh): C = alpha * (A @ B) + beta * C
void sgemm_coalesced(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                     torch::Tensor &output_matrix, float alpha, float beta)
{
    check_gemm_inputs(matrix_a, matrix_b, output_matrix);

    const int num_row_a = static_cast<int>(matrix_a.size(0));
    const int num_col_a = static_cast<int>(matrix_a.size(1));
    const int num_col_b = static_cast<int>(matrix_b.size(1));

    // 1D blocks of block_size * block_size threads; blockIdx.x walks the columns of C,
    // blockIdx.y walks the rows (the kernel reads output_col from blockIdx.x, output_row from blockIdx.y)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(ceil_div(num_col_b, block_size), ceil_div(num_row_a, block_size));

    // Launch on PyTorch's current stream so the benchmark's CUDA events time this kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sgemm_gmem_coaleased_kernel<block_size><<<grid_dim, block_dim, 0, stream>>>(
        num_row_a, num_col_b, num_col_a,
        alpha, matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(),
        beta, output_matrix.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
