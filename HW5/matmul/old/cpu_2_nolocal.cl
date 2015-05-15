// Version 2: tiled and coalesced
__kernel void mat_mul_cpu(
		const __global float* A,
		const __global float* B,
		__global float* C,
		const int ndim,	// Matrix dimension
		const int tileSize, // Tile size (of one side, to be exact)
		const int tileNum) // Number of tiles (per dimension)
{
	// Thread identifiers
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = tileSize * get_group_id(0) + row;
	const int globalCol = tileSize * get_group_id(1) + col;

	float tmp = 0.0f;

	// Loop over tiles
	for (int t = 0; t < tileNum; t++) {
		// Compute for single tile
		for (int k = 0; k < tileSize; k++) {
			tmp += A[globalRow * ndim + t * tileSize + k] * B[globalCol + (t * tileSize + k) * ndim];
		}
	}

	C[globalRow * ndim + globalCol] = tmp;
}
