/*
   MPI implementation of KMeans
   */

#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "mpi.h"

#define DATA_DIM 2
#define DEFAULT_ITERATION 1024

#define GET_TIME(T) __asm__ __volatile__ ("rdtsc\n" : "=A" (T))


// Read data from file
unsigned int read_data(FILE* f, float** data_p);
int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);


int main(int argc, char** argv)
{
	// MPI
	int comm_rank, comm_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	// Kmeans
	int class_n, data_n, iteration_n;
	float *acc_centroids, *centroids, *data;
	int *acc_count;
	int *acc_partitioned, *partitioned;
	FILE *io_file;
	struct timespec start, end, spent;

	// Check parameters
	if (argc < 4) {
		fprintf(stderr, "usage: %s <centroid file> <data file> <paritioned result> [<final centroids>] [<iteration number>]\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	if (comm_rank == 0) {
		// Read initial centroid data
		io_file = fopen(argv[1], "rb");
		if (io_file == NULL) {
			fprintf(stderr, "File open error %s\n", argv[1]);
			exit(EXIT_FAILURE);
		}
		class_n = read_data(io_file, &centroids);
		fclose(io_file);

		// Read input data
		io_file = fopen(argv[2], "rb");
		if (io_file == NULL) {
			fprintf(stderr, "File open error %s\n", argv[2]);
			exit(EXIT_FAILURE);
		}
		data_n = read_data(io_file, &data);
		fclose(io_file);

		acc_centroids = (float*)malloc(sizeof(float) * DATA_DIM * class_n);
		acc_count = (int*)malloc(sizeof(int) * class_n);
		acc_partitioned = (int*)malloc(sizeof(int) * data_n);
	}

	// Broadcast class_n & data_n
	MPI_Bcast(&class_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&data_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (data_n % comm_size != 0) {
		printf("Data size should be a multiple of number of processes.\n");
		MPI_Finalize();
		return -1;
	}

	if (comm_rank != 0) {
		centroids = (float*)malloc(sizeof(float) * DATA_DIM * class_n);
		data = (float*)malloc(sizeof(float) * DATA_DIM * data_n);
	}

	iteration_n = argc > 5 ? atoi(argv[5]) : DEFAULT_ITERATION;


	partitioned = (int*)malloc(sizeof(int)*data_n);


	clock_gettime(CLOCK_MONOTONIC, &start);

	// Run Kmeans algorithm
	int i, data_i, class_i;
	int st, ed;
	int *count = (int*)malloc(sizeof(int) * class_n);
	Point t;

	st = comm_rank * (data_n / comm_size);
	ed = (comm_rank + 1) * (data_n / comm_size);

	// Broadcast points
	MPI_Bcast(centroids, DATA_DIM * class_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(data, DATA_DIM * data_n, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Start iterations
	for (i = 0; i < iteration_n; i++) {
		
		// Assignment step
		for (data_i = st; data_i < ed; data_i++) {
			float min_dist = DBL_MAX;

			for (class_i = 0; class_i < class_n; class_i++) {
				t.x = data[2 * data_i] - centroids[2 * class_i];
				t.y = data[2 * data_i + 1] - centroids[2 * class_i + 1];

				float dist = t.x * t.x + t.y * t.y;

				if (dist < min_dist) {
					partitioned[data_i] = class_i;
					min_dist = dist;
				}
			}
		}

		// Update step
		// Clear sum buffer and class count
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[2 * class_i] = 0.0;
			centroids[2 * class_i + 1] = 0.0;
			count[class_i] = 0;
		}

		// Sum up and count data for each class
		for (data_i = st; data_i < ed; data_i++) {
			centroids[2 * partitioned[data_i]] += data[2 * data_i];
			centroids[2 * partitioned[data_i] + 1] += data[2 * data_i + 1];
			count[partitioned[data_i]]++;
		}

		// Reduction & broadcast
		MPI_Reduce(centroids, acc_centroids, DATA_DIM * class_n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(count, acc_count, class_n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

		if (comm_rank == 0) {
			// Divide the sum with number of classes for mean point
			for (class_i = 0; class_i < class_n; class_i++) {
				acc_centroids[2 * class_i] /= acc_count[class_i];
				centroids[2 * class_i] = acc_centroids[2 * class_i]; 
				acc_centroids[2 * class_i + 1] /= acc_count[class_i];
				centroids[2 * class_i + 1] = acc_centroids[2 * class_i + 1];
			}
		}

		MPI_Bcast(centroids, DATA_DIM * class_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	// Reduce partitioned
	MPI_Reduce(partitioned, acc_partitioned, data_n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	clock_gettime(CLOCK_MONOTONIC, &end);

	timespec_subtract(&spent, &end, &start);

	if (comm_rank == 0) {
		printf("Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

		// Write classified result
		io_file = fopen(argv[3], "wb");
		fwrite(&data_n, sizeof(data_n), 1, io_file);
		fwrite(acc_partitioned, sizeof(int), data_n, io_file); // Changed to acc_partitioned
		fclose(io_file);


		// Write final centroid data
		if (argc > 4) {
			io_file = fopen(argv[4], "wb");
			fwrite(&class_n, sizeof(class_n), 1, io_file);
			fwrite(centroids, sizeof(Point), class_n, io_file); 
			fclose(io_file);
		}
	}

	// Free allocated buffers
	if (comm_rank == 0) {
		free(acc_partitioned);
		free(acc_centroids);
		free(acc_count);
	}
	free(centroids);
	free(data);
	free(partitioned);
	free(count);

	MPI_Finalize();

	return 0;
}



int timespec_subtract (struct timespec* result, struct timespec *x, struct timespec *y)
{
	/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_nsec < y->tv_nsec) {
		int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
		y->tv_nsec -= 1000000000 * nsec;
		y->tv_sec += nsec;
	}
	if (x->tv_nsec - y->tv_nsec > 1000000000) {
		int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
		y->tv_nsec += 1000000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait.
	   tv_nsec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_nsec = x->tv_nsec - y->tv_nsec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}


unsigned int read_data(FILE* f, float** data_p)
{
	unsigned int size;
	size_t r;

	r = fread(&size, sizeof(size), 1, f);
	if (r < 1) {
		fputs("Error reading file size", stderr);
		exit(EXIT_FAILURE);
	}

	*data_p = (float*)malloc(sizeof(float) * DATA_DIM * size);

	r = fread(*data_p, sizeof(float), DATA_DIM*size, f);
	if (r < DATA_DIM*size) {
		fputs("Error reading data", stderr);
		exit(EXIT_FAILURE);
	}

	return size;
}

