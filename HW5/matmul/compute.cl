__kernel void mat_mul_compute(
		const __global float* A,
		const __global float* B,
		__global float* C,
		const int ndim)
{
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = TDIM * get_group_id(0) + row;
	const int globalCol = TDIM * get_group_id(1) + col;

	__local float Asub[TDIM][TDIM];
	__local float Bsub[TDIM][TDIM];

	float acc = 0.0f;

	const int tile_cnt = ndim / TDIM;

	for (int t = 0; t < tile_cnt; t++) {

		const int tiledRow = TDIM * t + row;
		const int tiledCol = TDIM * t + col;
		Asub[row][col] = A[globalRow * ndim + tiledCol];
		Bsub[row][col] = B[tiledRow * ndim + globalCol];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < TDIM; k++) {
			acc += Asub[row][k] * Bsub[k][col];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[globalRow * ndim + globalCol] = acc;

}
