__kernel void mat_mul_kernel(
		__global float* A,
		__global float* B,
		__global float* C,
		int wA)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k;

	float tmp = 0.0f;
	for (k = 0; k < wA; k++) {
		tmp += A[i * wA + k] * B[k * wA + j];
	}

	C[i * wA + j] = tmp;
}
