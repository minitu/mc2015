// Kernel 1: perform first update step
__kernel void kmeans_1(
		const int class_n,
		const int data_n,
		__global float* centroids,
		__global float* data,
		__global int* partitioned,
		const int data_n_wi,
		const int data_n_wg,
		__global float* acc_centroids,
		__global int* acc_count)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

	float p_centroids[2048] = {0.0f};
	int p_count[1024] = {0};

	int i, data_i, class_i;	// Loop indices
	float x, y; // Temporal point

	// Sum up and count data for each class
	for (data_i = group_id * data_n_wg + local_id * data_n_wi; data_i < group_id * data_n_wg + (local_id + 1) * data_n_wi; data_i++) {

		p_centroids[2 * partitioned[data_i]] += data[2 * data_i];
		p_centroids[2 * partitioned[data_i] + 1] += data[2 * data_i + 1];
		p_count[partitioned[data_i]]++;
	}

	// Write back to global accumulation memory
	for (class_i = 0; class_i < class_n; class_i++) {
		acc_centroids[2 * class_n * global_id + 2 * class_i] = p_centroids[2 * class_i];
		acc_centroids[2 * class_n * global_id + 2 * class_i + 1] = p_centroids[2 * class_i + 1];
		acc_count[class_n * global_id + class_i] = p_count[class_i];
	}
}
