template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(int num_rows_a, int num_col_b, int num_col_a, float alpha, const float *matrix_a, const float *matrix_b, float beta, float *matrix_c){
    
    const uint block_row = blockIdx.x;
    const uint block_col = blockidx.y;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BM];

    const uint thread_row = threadIdx.x / BN; // which TM rows group of C is this thread responsible for ()
    const uint thread_col = threadIdx.x  % BN; // the column of C this thread owns (fastest changing)

    // calculate global row and global column indicies for this thread
    const int global_row = block_row * BM + thread_row * TM;
    const int global_col = block_col * BN + thread_col;

    //Move pointers to the starting position of this block 
    matrix_a += block_row * BM * num_cols_a;
    matrix_b = block_col * BN; 
    matrix_c = block_row * BM * num_col_b * block_col * BN;

    // Allocate thread-local cache for results in register 
    float thread_results[TM] = {0.0f};

    // Loop over all tiles along K dimension 
    for (int tile_idx =0; tile_idx < num_col_a; tile_idx += BK){
        // Load tile from matrix A into shared memory with bounds checking
        const uint a_row = threadIdx.x / BK; // the particular row within a current tile/block
        const uint a_col = threadIdx.x % BK;

        const uint b_row = threadIdx.x / BN;
        const uint b_col = threadIdx.x % BN;

        if ((block_row * BM + a_row) < num_rows_a && (block_col * BN + a_col) < num_col_a){ 
            tile_a[a_row * BK + a_col] = matrix_a[a_row * num_col_a + a_col];
        }
        else {
            tile_a[a_row * BK + a_col] = 0.0f;
        }

        if ((tile_idx + b_row) < num_col_a && (block_col * BN + b_col) < num_col_b){
            tile_b[b_row * BN + b_col] = matrix_b[b_row * num_col_b+b_col];
        }
        else {
            tile_b[b_row * BN + b_col] = 0.0f;
        }

        __syncthreads();

        // Advance pointers to the next tile
        matrix_a += BK; // moves BK chunks to the right (col-wise)
        matrix_b += BK * num_col_b; // moves downwards (rowwise)

        // Calculate per-thread results
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx){
            float b_tmp = tile_b[dot_idx * BN + thread_col];
            for (uint res_idx =0; res_idx < TM; ++res_idx) {
                thread_results[res_idx] += tile_a[(thread_row * TM + res_idx) * BM + dot_idx] * b_tmp;
            }
        }
        __syncthreads();

    }
    // Write results to global memory;
    for (uint res_idx =0; res_idx < TM; ++res_idx) {
        int row = global_row + res_idx;
        if (row < num_rows_a && global_col < num_col_b) {
            matrix_c[(thread_row * TM + res_idx) * num_col_b + thread_col] = 
            alpha * thread_results[res_idx] + beta * matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col];
        }
    }
}
