template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_a, int num_cols_b, float alpha, const float *matrix_a, const float *matrix_b, float beta, float *matrix_c){
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BM * BK]
    __shared__ float tile_b[BK * BN]

    const uint thread_row = threadIdx.x / (BN/TN);
    const uint thread_col = threadIdx.x % (BN/TN);
    const uint num_threads = (BM/TM) * (BN/TN);

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN; 
    matrix_c += block_row * BM + num_cols_b + block_col * BN;

    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a, block_k_idx += BK){
        #pragma unroll 
        // Load tiles from A into shared memory. Each thread loads in an elements of A each pass
        for (uint load_offset = 0; load_offset < BM * BK; load_offset +=num_threads){
            load_idx = threadIdx.x + load_offset; // this thread flat positon in the tile for this pass
            uint a_row = load_idx / BK; // turn flat position into (row, col) cordinate within tile
            uint a_col = load_idx % BK;
            tile_a[load_idx] = matrix_a[a_row * num_col_a + a_col];
        }

        #pragma unroll
        for (uint load_offset = 0; load_offset < num_cols_b; load_offset +=BN){
            load_idx = threadIdx.x + load_offset; 
            uint b_row = load_idx / BN;
            uint b_col = load_idx % BN;
            tile_b[load_idx] = mmatrix_b[b_row * num_col_b +b_col];
        }
        __syncthreads();

        matrix_a += BK;
        matrix_b += BK * num_cols_b // moving down along the rows

        for (uint dot_idx =0; dot_idx < BK; ++dot_idx){
            for (uint i = 0; i < TM; ++i){
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx]
            }
        }
    }

}
