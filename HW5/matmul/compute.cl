__kernel void mat_mul_compute(
		const __global float* A,
		const __global float* B,
		__global float* C,
		const int ndim)
{
	// Indices
	const int local_i = get_local_id(0);
	const int local_j = get_local_id(1);
	const int global_i = TDIM * get_group_id(0) + local_i;
	const int global_j = TDIM * get_group_id(1) + local_j;

	// Tiles on local memory
	__local float TA[TDIM][TDIM];
	__local float TB[TDIM][TDIM];

	// Accumulator
	float acc = 0.0f;

	// Number of tiles
	const int tile_cnt = ndim / TDIM;

	// Perform calculatons for each tile
	for (int t = 0; t < tile_cnt; t++) {

		const int tile_i = TDIM * t + local_i;
		const int tile_j = TDIM * t + local_j;
		
		TA[local_i][local_j] = A[global_i * ndim + tile_j];
		TB[local_i][local_j] = B[tile_i * ndim + global_j];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < TDIM; k++) {
			acc += TA[local_i][k] * TB[k][local_j];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Save result in C
	C[global_i * ndim + global_j] = acc;

}
