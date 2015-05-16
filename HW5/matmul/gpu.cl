// Version 1: naive
__kernel void mat_mul_gpu(
		const __global float* A,
		const __global float* B,
		__global float* C,
		const int ndim)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	int k;

	float tmp = 0.0f;
	for (k = 0; k < ndim; k++) {
		tmp += A[i * ndim + k] * B[k * ndim + j];
	}

	C[i * ndim + j] = tmp;
}
