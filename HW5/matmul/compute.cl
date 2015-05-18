// Version 3: more workload per work-item
__kernel void mat_mul_compute(
		const __global float* A,
		const __global float* B,
		__global float* C,
		__local float* Asub,
		__local float* Bsub,
		const int ndim,	// Matrix dimension
		const int tileSize, // Tile size (of one side, to be exact)
		const int tileNum, // Number of tiles (per dimension)
		const int workload, // Workload per work-item
		const int rts) // tileSize / workload
{
	// Thread identifiers
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = tileSize * get_group_id(0) + row;
	const int globalCol = tileSize * get_group_id(1) + col;

	// Initialize accumulation registers
	float acc[32]; // Set to 32 since workload (or tile size) can't be larger than 32
	for (int w = 0; w < workload; w++) {
		acc[w] = 0.0f;
	}

	// Loop over tiles
	for (int t = 0; t < tileNum; t++) {
		
		// Load 1 tile each from A and B into local memory
		for (int w = 0; w < workload; w++) {
			const int tiledRow = tileSize * t + row;
			const int tiledCol = tileSize * t + col;
			Asub[tileSize * row + col + w * rts] = A[globalRow * ndim + tiledCol + w * rts];
			Bsub[tileSize * row + col + w * rts] = B[tiledRow * ndim + globalCol + w * rts];
		}

		// Synchronize
		barrier(CLK_LOCAL_MEM_FENCE);

		// Compute for single tile
		for (int k = 0; k < tileSize; k++) {
			for (int w = 0; w < workload; w++) {
				acc[w] += Asub[tileSize * row + k] * Bsub[tileSize * k + col + w * rts];
			}
		}

		// Synchronize
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (int w = 0; w < workload; w++) {
		C[globalRow * ndim + globalCol + w * rts] = acc[w];
	}
}
