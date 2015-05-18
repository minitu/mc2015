__kernel void mat_mul_compute(
		const __global float* A,
		const __global float* B,
		__global float* C,
		const int ndim)
{
	int A_index = TDIM * TDIM * get_group_id(0) + get_local_id(0);
	int B_index = TDIM * TDIM * get_group_id(1) + get_local_id(1);
	int C_index = A_index + B_index;

	float c[TDIM * TDIM] = {0.0};

	for (int n = 0; n < ndim; n+= TDIM) {
		for (int i = 0; i < TDIM; i++) {
			for (int j = 0; j < TDIM; j++) {
				for (int k = 0; k < TDIM; k++) {
					c[i * TDIM + j] += A[A_index + k + i * TDIM] * B[B_index + k + j * TDIM];
				}
			}
		}
		A_index += TDIM;
		B_index += TDIM;
	}

	for (int i = 0; i < TDIM; i++) {
		for (int j = 0; j < TDIM; j++) {
			C[C_index + i * TDIM + j * TDIM] = c[i * TDIM + j];
		}
	}
}
