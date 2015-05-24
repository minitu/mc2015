
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

	int i, data_i, class_i, p_class_i;
	int* count = (int*)malloc(sizeof(int) * class_n);
	Point* p_centroids;
	int* p_count;
	Point t;

#pragma omp parallel private(p_class_i, p_centroids, p_count, t)
	{
		p_centroids = (Point*)malloc(sizeof(Point) * class_n);
		p_count = (int*) malloc(sizeof(int) * class_n);

		// Repeat iterations
		for (i = 0; i < iteration_n; i++) {
			// Assignment step
#pragma omp for
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
#pragma omp single
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[class_i].x = 0.0;
				centroids[class_i].y = 0.0;
				count[class_i] = 0;
			}

			for (p_class_i = 0; p_class_i < class_n; p_class_i++) {
				p_centroids[p_class_i].x = 0.0;
				p_centroids[p_class_i].y = 0.0;
				p_count[p_class_i] = 0;
			}

			// Sum up and count data for each class in thread private memory
#pragma omp for
			for (data_i = 0; data_i < data_n; data_i++) {
				p_centroids[partitioned[data_i]].x += data[data_i].x;
				p_centroids[partitioned[data_i]].y += data[data_i].y;
				p_count[partitioned[data_i]]++;
			}

			// Reduce back to shared memory
			for (p_class_i = 0; p_class_i < class_n; p_class_i++) {
#pragma omp atomic
				centroids[p_class_i].x += p_centroids[p_class_i].x;
#pragma omp atomic
				centroids[p_class_i].y += p_centroids[p_class_i].y;
#pragma omp atomic
				count[p_class_i] += p_count[p_class_i];
			}

			// Divide the sum with number of class for mean point
#pragma omp single
			for (class_i = 0; class_i < class_n; class_i++) {
				centroids[class_i].x /= count[class_i];
				centroids[class_i].y /= count[class_i];
			}
		}

		free(p_centroids);
		free(p_count);
	}
}
