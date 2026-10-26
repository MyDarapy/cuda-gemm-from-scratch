#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"



template <const uint >
__global__ void sgemm_naive_kernel(int num_row_a, int num_col_b, int num_col_a,
                                    float alpha, const float* matrix_a, 
                                    const float* matrix_b, float beta, float* output_matrix)
                                    {
                                        // Map 1D threads unto 2D output Each thread is responsible for a portion of the 2D output position                                         
                                        // Global memory coleasing loads. 
                                        const int output_row = blockIdx.y * block_size + (threadIdx.x / block_size)
                                        const int output_col = blockIdx.x * block_size + (threadIdx.x % block_size)

                                        // Boundary check for non multiple of block size output = (num_row_a by num_col_b)
                                        if (output_row < num_row_a && output_col < num_col_b){

                                            float accumulator = 0.0f;
                                            for (int k_idx = 0; k_idx < num_col_a; k_idx++){
                                                accumulator += matrix_a[num_col_a * output_row + k_idx] * matrix_b[k_idx * num_col_b + output_col];
                                            }
                                            
                                                const output_index = output_row * num_col_b + output_col;
                                                output_matrix[output_index] = alpha*accumulator + beta*output_matrix[output_index];
                                    
                                        }


                                    }


template <const uint block_size>
__global__ void sgemm_shared_mem_kernel(int num_rows_a, int num_col_b, int num_col_a, 
    float alpha, float beta, const *matrix_a, const* matrix_b, float* matrix_c)
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
        matrix_b += 0 + block_col * block_size 
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
            if ((tile_ix + thread_row) < num_col_a && global_col < num_col_b){
                tile_b[thread_row * block_size + thread_col] = matrix_b[thread_row * num_col_b + thread_col]        
            }
            else{
                tile_b[thread_row * block_size + thread_col] = 0.0f;
            }

            // Block threads until cache is fully populated
            __syncthread();

            // Advance pointers to next tile 
            matrix_a += block_size;
            matrix_b += block_size * num_col_b
            
            // Compute partial dot product using shared memory
            for int(dot_idx = 0; dot_idx < block_size; ++dot_idx){
                accumulator += tile_a[thread_row * block_size + dot_idx]* tile_b[dot_idx * block_size + thread_col];
            }
            // Sync again to avoid faster threads fetching bext block before slower threads finish
            __syncthreads();
        }
        // Write result to global memory with bounds checking
        if (global_row < num_rows_a && global_col < num_col_b){
            matrix_c[thread_row * num_cols_b + thread_col] = alpha * accumlator + beta * matrix_c[thread_row * num_cols_b + thread_col];
        }


    }




