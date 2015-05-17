__kernel void kmeans_gpu(
		const int iteration_n,
		const int class_n,
		const int data_n,
		__global float* centroids,
		__global float* data,
		__global int* partitioned,
		__local int* count,
		const int my_data_n)
{
	const int id = get_global_id(0); // Work-item ID
	int i, data_i, class_i;	// Loop indices
	float x, y; // Temporal point

	// Iterate
	for (i = 0; i < iteration_n; i++) {
		
		// Assignment
		for (data_i = id * my_data_n; data_i < (((id + 1) * my_data_n) < (data_n) ? ((id + 1) * my_data_n) : (data_n)); data_i++) {
			float min_dist = DBL_MAX;

			for (class_i = 0; class_i < class_n; class_i++) {
				x = data[2 * data_i] - centroids[2 * class_i];
				y = data[2 * data_i + 1] - centroids[2 * class_i + 1];
				
				float dist = x * x + y * y;

				if (dist < min_dist) {
					partitioned[data_i] = class_i;
					min_dist = dist;
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// Update
		// Clear sum buffer and class count (for work-item 0)
		if (id == 0) {
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[2 * class_i] = 0.0;
				centroids[2 * class_i + 1] = 0.0;
				count[class_i] = 0;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		// Sum up and count data for each class
		for (data_i = id * my_data_n; data_i < (((id + 1) * my_data_n) < (data_n) ? ((id + 1) * my_data_n) : (data_n)); data_i++) {
			centroids[2 * partitioned[data_i]] += data[2 * data_i];
			centroids[2 * partitioned[data_i] + 1] += data[2 * data_i + 1];
			count[partitioned[data_i]]++;
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		// Divide the sum with number of class for mean point
		if (id == 0) {
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[2 * class_i] /= count[class_i];
				centroids[2 * class_i + 1] /= count[class_i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
}
