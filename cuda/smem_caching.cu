#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"



template <const uint block_size>
__global__ void sgemm_smem_caching_kernel(int num_rows_a, int num_col_b, int num_col_a, 
    float alpha, float beta, const float *matrix_a, const float *matrix_b, float *matrix_c)
    {
        const uint block_row = blockIdx.y;
        const uint block_col = blockIdx.x;

        __shared__ float tile_a[block_size * block_size];
        __shared__ float tile_b[block_size * block_size];

        const uint thread_row = threadIdx.x / block_size;
        const uint thread_col = threadIdx.x  % block_size; // fastest changing index -- the column

        // Calculate the global roow and global column indicies for this thread 
        const uint global_row = block_row * block_size + thread_row;
        const uint global_col = block_col * block_size + thread_col;

        // Move pointers to the starting position of this block
        // address = row * width + col
        matrix_a += block_row * block_size * num_col_a + 0; // row = block_row, col=0
        matrix_b += 0 + block_col * block_size;
        matrix_c += block_row * block_size * num_col_b + block_col * block_size;

        float accumulator = 0.0f;

        // Loop over all the tiles along k dimension 
        for (int tile_idx = 0; tile_idx < num_col_a; tile_idx +=block_size){
            // Load tile from matrix A into shared memory with bounds checking 
            // thread_col is consecutive for coaleased memory accesses
            
            if (global_row < num_rows_a && (tile_idx + thread_col) < num_col_a){
                tile_a[thread_row * block_size + thread_col] = matrix_a[thread_row * num_col_a + thread_col];
            }
            else{
                tile_a[thread_row * block_size + thread_col] = 0.0f;
            }

            // Load tile from matrix B into shared memory with bounds checking
            if ((tile_idx + thread_row) < num_col_a && global_col < num_col_b){
                tile_b[thread_row * block_size + thread_col] = matrix_b[thread_row * num_col_b + thread_col];
            }
            else{
                tile_b[thread_row * block_size + thread_col] = 0.0f;
            }

            // Block threads until cache is fully populated
            __syncthreads();

            // Advance pointers to next tile 
            matrix_a += block_size;
            matrix_b += block_size * num_col_b;
            
            // Compute partial dot product using shared memory
            for (int dot_idx = 0; dot_idx < block_size; ++dot_idx){
                accumulator += tile_a[thread_row * block_size + dot_idx]* tile_b[dot_idx * block_size + thread_col];
            }
            // Sync again to avoid faster threads fetching bext block before slower threads finish
            __syncthreads();
        }
        // Write result to global memory with bounds checking
        if (global_row < num_rows_a && global_col < num_col_b){
            matrix_c[thread_row * num_col_b + thread_col] = alpha * accumulator + beta * matrix_c[thread_row * num_col_b + thread_col];
        }


    }



// Host entry point (declared in gemm_kernels.cuh): C = alpha * (A @ B) + beta * C
void sgemm_shared_mem(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &matrix_c, float alpha, float beta)
{
    check_gemm_inputs(matrix_a, matrix_b, matrix_c);

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_col_a = static_cast<int>(matrix_a.size(1));
    const int num_col_b = static_cast<int>(matrix_b.size(1));

    // 1D blocks of block_size * block_size threads; blockIdx.x walks the columns of C,
    // blockIdx.y walks the rows (the kernel reads block_col = blockIdx.x, block_row = blockIdx.y)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(ceil_div(num_col_b, block_size), ceil_div(num_rows_a, block_size));

    // Launch on PyTorch's current stream so the benchmark's CUDA events time this kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sgemm_smem_caching_kernel<block_size><<<grid_dim, block_dim, 0, stream>>>(
        num_rows_a, num_col_b, num_col_a,
        alpha, beta, matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(),
        matrix_c.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
