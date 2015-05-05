
#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <pthread.h>
#include <math.h>

#define TNUM		16
#define MIN(x,y)	((x < y) ? (x) : (y))

/* Some global variables */
int t_iteration_n;
int t_class_n;
int t_class_cnt;
int t_data_n;
int t_data_cnt;
Point* t_centroids;
Point* t_data;
int* t_partitioned;
int* count;

/* Pthread barrier & mutex */
pthread_barrier_t my_barrier;
pthread_mutex_t* mutex;

/* Thread data structure */
struct thread_data {
	int tid;
};

struct thread_data tds[TNUM];

void *kmeans_sub(void *t_arg)
{
	struct thread_data *my_data = (struct thread_data *) t_arg;

	// Thread ID
	int tid = my_data->tid;
	// Loop indices
	int i, data_i, class_i;
	// Temporal point value to calculate distance
	Point t;

	// Iterate
	for (i = 0; i < t_iteration_n; i++) {

		// Assignment step
		for (data_i = tid * t_data_cnt; data_i < MIN(tid * t_data_cnt + t_data_cnt, t_data_n); data_i++) {
			float min_dist = DBL_MAX;

			for (class_i = 0; class_i < t_class_n; class_i++) {
				t.x = t_data[data_i].x - t_centroids[class_i].x;
				t.y = t_data[data_i].y - t_centroids[class_i].y;

				float dist = t.x * t.x + t.y * t.y;

				if (dist < min_dist) {
					t_partitioned[data_i] = class_i;
					min_dist = dist;
				}
			}
		}

		// Synchronize with other threads
		pthread_barrier_wait(&my_barrier);

		// Update step
		// Clear sum buffer and class count
		for (class_i = tid * t_class_cnt; class_i < MIN(tid * t_class_cnt + t_class_cnt, t_class_n); class_i++) {
			t_centroids[class_i].x = 0.0;
			t_centroids[class_i].y = 0.0;
			count[class_i] = 0;
		}

		// Synchronize
		pthread_barrier_wait(&my_barrier);

		// Sum up and count data for each class (use mutex here)
		for (data_i = tid * t_data_cnt; data_i < MIN(tid * t_data_cnt + t_data_cnt, t_data_n); data_i++) {
			pthread_mutex_lock(&mutex[t_partitioned[data_i]]); // Lock
			t_centroids[t_partitioned[data_i]].x += t_data[data_i].x;
			t_centroids[t_partitioned[data_i]].y += t_data[data_i].y;
			count[t_partitioned[data_i]]++;
			pthread_mutex_unlock(&mutex[t_partitioned[data_i]]); // Unlock
		}

		// Synchronize
		pthread_barrier_wait(&my_barrier);

		// Divide the sum with the number of classes for mean point
		for (class_i = tid * t_class_cnt; class_i < MIN(tid * t_class_cnt + t_class_cnt, t_class_n); class_i++) {
			t_centroids[class_i].x /= count[class_i];
			t_centroids[class_i].y /= count[class_i];
		}
		
		// All threads must reach here before next iteration
		pthread_barrier_wait(&my_barrier);
	}

	pthread_exit(NULL);
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	/* Set global variables for threads to read */
	t_iteration_n = iteration_n;
	t_class_n = class_n;
	t_data_n = data_n;
	t_centroids = centroids;
	t_data = data;
	t_partitioned = partitioned;

	/* Thread related declarations */
	pthread_t threads[TNUM];
	pthread_attr_t attr;
	int i, t, rc;
	void *status;

	/* Set number of data & classes that each thread is responsible for */
	t_data_cnt = ceil((double)t_data_n / (double)TNUM);
	t_class_cnt = ceil((double)t_class_n / (double)TNUM);

	/* Malloc count & locks & mutex */
	count = (int*) malloc(sizeof(int) * class_n);
	mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t) * class_n);

	/* Initialize mutex */
	for (i = 0; i < class_n; i++) {
		pthread_mutex_init(&mutex[i], NULL);
	}

	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	/* Intialize barrier */
	pthread_barrier_init(&my_barrier, NULL, TNUM);

	/* Create and execute threads */
	for (t = 0; t < TNUM; t++) {
		tds[t].tid = t;
		rc = pthread_create(&threads[t], &attr, kmeans_sub, (void *) &tds[t]);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}
	
	/* Free attribute and wait for the other threads */
	pthread_attr_destroy(&attr);
	for (t = 0; t < TNUM; t++) {
		rc = pthread_join(threads[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}

	/* Destroy barrier */
	pthread_barrier_destroy(&my_barrier);

	/* Destroy mutex */
	for (i = 0; i < class_n; i++) {
		pthread_mutex_destroy(&mutex[i]);
	}

}
