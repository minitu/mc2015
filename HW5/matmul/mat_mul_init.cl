__kernel void mat_mul_init(
		__global float* A,
		__global float* B,
		int NDIM,
		int SDIM,
		float startNum)
{
	int ki = get_global_id(0);
	int kj = get_global_id(1);
	int i, j;

	for (i = ki * SDIM; i < ((ki * SDIM + SDIM < NDIM) ? ki * SDIM + SDIM : NDIM); i++) {
		for (j = kj * SDIM; j < ((kj * SDIM + SDIM < NDIM) ? kj * SDIM + SDIM : SDIM); j++) {
			float tmp = startNum + i * NDIM + j;
			int index = i * NDIM + j;
			A[index] = tmp;
			B[index] = tmp;
		}
	}
}
