__kernel void kmeans_2(
		const int class_n,
		const int data_n,
		__global float* centroids,
		__global float* data,
		__global int* partitioned,
		const int data_n_wi,
		const int data_n_wg,
		__local float* centroids_wg,
		__local float* data_wg,
		__local int* partitioned_wg)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

	int i, data_i, class_i;	// Loop indices
	float x, y; // Temporal point

	// Copy data from global to local memory
	if (local_id == 0) {
		for (class_i = 0; class_i < 2 * class_n; class_i++) {
			centroids_wg[class_i] = centroids[class_i];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Each work-item does the below independently
	for (data_i = group_id * data_n_wg + local_id * data_n_wi; data_i < group_id * data_n_wg + (local_id + 1) * data_n_wi; data_i++) {
		data_wg[2 * data_i] = data[2 * data_i];
		data_wg[2 * data_i + 1] = data[2 * data_i + 1];
		partitioned_wg[data_i] = partitioned[data_i];

		// Assignment
		float min_dist = DBL_MAX;

		for (class_i = 0; class_i < class_n; class_i++) {
			x = data_wg[2 * data_i] - centroids_wg[2 * class_i];
			y = data_wg[2 * data_i + 1] - centroids_wg[2 * class_i + 1];

			float dist = x * x + y * y;

			if (dist < min_dist) {
				partitioned_wg[data_i] = class_i;
				min_dist = dist;
			}
		}

		// Copy data back to global memory
		partitioned[data_i] = partitioned_wg[data_i];
	}
}
