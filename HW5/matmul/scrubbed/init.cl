__kernel void mat_mul_init(
		__global float* A,
		__global float* B,
		const int NDIM,
		const int SDIM,
		const float startNum,
		const int setNum,
		const int setRows,
		const int use_gpu)
{
	if (!use_gpu) { // CPU
		int ki = get_global_id(0);
		int kj = get_global_id(1);
		int i, j;

		for (i = ki * SDIM; i < ((ki * SDIM + SDIM < NDIM) ? ki * SDIM + SDIM : NDIM); i++) {
			for (j = kj * SDIM; j < ((kj * SDIM + SDIM < NDIM) ? kj + SDIM + SDIM : NDIM); j++) {
				float tmp = startNum + i * NDIM + j;
				int index = i * NDIM + j;
				A[index] = tmp;
				B[index] = tmp;
			}
		}
	}
	else { // GPU
		int id = get_global_id(0);
		int i = setNum * setRows + id;
		int j;
		float tmp = startNum + id * NDIM;

		if (i < NDIM) {
			for (j = 0; j < NDIM; j++) {
				A[id * NDIM + j] = tmp;
				B[id * NDIM + j] = tmp;
				tmp++;
			}
		}
	}
}
