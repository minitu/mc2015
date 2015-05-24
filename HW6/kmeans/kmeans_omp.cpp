
#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>

#define TNUM	64

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	omp_set_dynamic(0); // Disable dynamic threads
	omp_set_num_threads(TNUM); // Set number of threads

	int i, j, data_i, class_i;
	Point* acc_centroids = (Point*)malloc(sizeof(Point) * class_n * TNUM);
	int* acc_count = (int*)malloc(sizeof(int) * class_n * TNUM);
	Point t;

	// Copy centroids
	for (class_i = 0; class_i < class_n; class_i++) {
		acc_centroids[class_i].x = centroids[class_i].x;
		acc_centroids[class_i].y = centroids[class_i].y;
	}

#pragma omp parallel private(i, j, data_i, class_i, t)
	{
		int tid = omp_get_thread_num();
		int ti;

		// Repeat iterations
		for (i = 0; i < iteration_n; i++) {
			// Assignment step
#pragma omp for
			for (data_i = 0; data_i < data_n; data_i++) {
				float min_dist = DBL_MAX;

				for (class_i = 0; class_i < class_n; class_i++) {
					t.x = data[data_i].x - acc_centroids[class_i].x;
					t.y = data[data_i].y - acc_centroids[class_i].y;

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
			for (class_i = 0; class_i < class_n * TNUM; class_i++) {
				acc_centroids[class_i].x = 0.0;
				acc_centroids[class_i].y = 0.0;
				acc_count[class_i] = 0;
			}

			// Sum up and count data for each class in accumulator memory
#pragma omp for
			for (data_i = 0; data_i < data_n; data_i++) {
				ti = class_n * tid + partitioned[data_i];
				acc_centroids[ti].x += data[data_i].x;
				acc_centroids[ti].y += data[data_i].y;
				acc_count[ti]++;
			}

			// Accumulate
#pragma omp for
			for (class_i = 0; class_i < class_n; class_i++) {
				for (j = 1; j < TNUM; j++) {
					acc_centroids[class_i].x += acc_centroids[class_n * j + class_i].x;
					acc_centroids[class_i].y += acc_centroids[class_n * j + class_i].y;
					acc_count[class_i] += acc_count[class_n * j + class_i];
				}
			}

			// Divide the sum with number of class for mean point
#pragma omp for
			for (class_i = 0; class_i < class_n; class_i++) {
				acc_centroids[class_i].x /= acc_count[class_i];
				acc_centroids[class_i].y /= acc_count[class_i];
			}
		}
	}

	// Store back in original memory
	for (class_i = 0; class_i < class_n; class_i++) {
		centroids[class_i].x = acc_centroids[class_i].x;
		centroids[class_i].y = acc_centroids[class_i].y;
	}
}
