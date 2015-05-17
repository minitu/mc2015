__kernel void kmeans_cpu(
		const int iteration_n,
		const int class_n,
		const int data_n,
		float* centroids,
		float* data,
		int* partitioned)
{
	const int i = 
