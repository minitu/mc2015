// Tiled and coalesced version
__kernel void mat_mul_cpu(
		const __global float* A,
		const __global float* B,
		__global float* C,
		const int ndim,	// Matrix dimension
		const int t_size,	// Tile size
		const int t_num) // Number of tiles
{
	// Thread identifiers
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = t_size * get_group_id(0) + row;
	const int globalCol = t_size * get_group_id(1) + col;

	// Local memory to store tiles
	__local float Asub[t_size][t_size];
	__local float Bsub[t_size][t_size];

	float tmp = 0.0f;

	// Loop over tiles
	for (int t = 0; t < t_num; t++) {
		
		// Load 1 tile each from A and B into local memory
		const int tiledRow = t_size * t + row;
		const int tiledCol = t_size * t + col;

		Asub[row][col] = A[tiledRow * ndim + globalCol];
		Bsub[row][col] = B[globalRow * ndim + tiledCol];

		// Synchronize
		barrier(CLK_LOCAL_MEM_FENCE);

		// Compute for single tile
		for (int k = 0; k < t_size; k++) {
			tmp += Asub[row][k] * Bsub[k][col];
		}

		// Synchronize
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[globalRow * ndim + globalCol] = tmp;
}
