
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

	int i, data_i, class_i;
	Point* acc_centroids = (Point*)malloc(sizeof(Point) * class_n * TNUM);
	int* acc_count = (int*)malloc(sizeof(int) * class_n * TNUM);
	Point t;

	// Copy centroids
	for (class_i = 0; class_i < class_n; class_i++) {
		acc_centroids[class_i].x = centroids[class_i].x;
		acc_centroids[class_i].y = centroids[class_i].y;
	}

	// Repeat iterations
	for (i = 0; i < iteration_n; i++) {
		// Assignment step
#pragma omp parallel for private(class_i, t)
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
#pragma omp parallel for
		for (class_i = 0; class_i < class_n * TNUM; class_i++) {
			acc_centroids[class_i].x = 0.0;
			acc_centroids[class_i].y = 0.0;
			acc_count[class_i] = 0;
		}

		// Sum up and count data for each class in accumulator memory
#pragma omp parallel for
		for (data_i = 0; data_i < data_n; data_i++) {
			int ti = class_n * omp_get_thread_num() + partitioned[data_i];
			acc_centroids[ti].x += data[data_i].x;
			acc_centroids[ti].y += data[data_i].y;
			acc_count[ti]++;
		}

		// Accumulate
		for (int t = 0; t < TNUM; t++) {
			for (class_i = 0; class_i < class_n; class_i++) {
				acc_centroids[class_i].x += acc_centroids[class_n * t + class_i].x;
				acc_centroids[class_i].y += acc_centroids[class_n * t + class_i].y;
				acc_count[class_i] += acc_count[class_n * t + class_i];
			}
		}

		// Divide the sum with number of class for mean point
		for (class_i = 0; class_i < class_n; class_i++) {
			acc_centroids[class_i].x /= acc_count[class_i];
			acc_centroids[class_i].y /= acc_count[class_i];
		}
	}

	// Store back in original memory
	for (class_i = 0; class_i < class_n; class_i++) {
		centroids[class_i].x = acc_centroids[class_i].y;
		centroids[class_i].y = acc_centroids[class_i].y;
	}
}
