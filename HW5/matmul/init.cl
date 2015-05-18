__kernel void mat_mul_init(
		__global float* A,
		__global float* B,
		const int NDIM,
		const float startNum)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	int index = i * NDIM + j;
	float tmp = startNum + index;
		
	A[index] = tmp;
	B[index] = tmp;
}
