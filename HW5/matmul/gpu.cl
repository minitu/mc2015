// Tiled and coalesced version
__kernel void mat_mul_gpu(
		const __global float* A,
		const __global float* B,
		__global float* C,
		__local float* Asub,
		__local float* Bsub,
		const int ndim,	// Matrix dimension
		const int tileSize,
		const int tileNum) // Number of tiles
{
	// Thread identifiers
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = tileSize * get_group_id(0) + row;
	const int globalCol = tileSize * get_group_id(1) + col;

	float tmp = 0.0f;

	// Loop over tiles
	for (int t = 0; t < tileNum; t++) {
		
		// Load 1 tile each from A and B into local memory
		const int tiledRow = tileSize * t + row;
		const int tiledCol = tileSize * t + col;

		Asub[tileSize * row + col] = A[tiledRow * ndim + globalCol];
		Bsub[tileSize * row + col] = B[globalRow * ndim + tiledCol];

		// Synchronize
		barrier(CLK_LOCAL_MEM_FENCE);

		// Compute for single tile
		for (int k = 0; k < tileSize; k++) {
			tmp += Asub[tileSize * row + k] * Bsub[tileSize * k + col];
		}

		// Synchronize
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[globalRow * ndim + globalCol] = tmp;
}
