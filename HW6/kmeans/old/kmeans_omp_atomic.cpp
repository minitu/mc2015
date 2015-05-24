
#include "kmeans.h"

#include <stdlib.h>
#include <float.h>
#include <omp.h>

#define TNUM	64

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	omp_set_dynamic(0); // Disable dynamic threads
	omp_set_num_threads(TNUM); // Set number of threads

	int i, data_i, class_i;
	int* count = (int*)malloc(sizeof(int) * class_n);
	Point t;

	// Repeat iterations
	for (i = 0; i < iteration_n; i++) {

#pragma omp parallel
		{
			// Assignment step
#pragma omp for private(class_i)
			for (data_i = 0; data_i < data_n; data_i++) {
				float min_dist = DBL_MAX;

				for (class_i = 0; class_i < class_n; class_i++) {
					t.x = data[data_i].x - centroids[class_i].x;
					t.y = data[data_i].y - centroids[class_i].y;

					float dist = t.x * t.x + t.y * t.y;

					if (dist < min_dist) {
						partitioned[data_i] = class_i;
						min_dist = dist;
					}
				}
			}

			// Update step
			// Clear sum buffer and class count
#pragma omp for
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[class_i].x = 0.0;
				centroids[class_i].y = 0.0;
				count[class_i] = 0;
			}

			// Sum up and count data for each class (use reduction)
#pragma omp for
			for (data_i = 0; data_i < data_n; data_i++) {
#pragma omp atomic
				centroids[partitioned[data_i]].x += data[data_i].x;
#pragma omp atomic
				centroids[partitioned[data_i]].y += data[data_i].y;
#pragma omp atomic
				count[partitioned[data_i]]++;
			}

			// Divide the sum with number of class for mean point
#pragma omp for
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[class_i].x /= count[class_i];
				centroids[class_i].y /= count[class_i];
			}
		}
	}
}
