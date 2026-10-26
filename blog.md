Every element of of the output matrix is computed by a single thread launched in the kernel. 
threadidx.x  / block_size finds the row of the output tile this thread is responsible for, and threadidx.x % block_size finds the column of the output tile this thread is responsible for. 

each [output_row, output_col] position in the output matrix is computed by a single thread. The thread computes the dot product of the corresponding row of matrix_a and column of matrix_b to produce the output value at that position.

matrix_a = output_row * num_rows_in_a + k_idx 
matrix_b = k_idx * num_rows_in_b + output_col