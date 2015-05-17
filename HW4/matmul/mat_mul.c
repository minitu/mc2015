#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "timers.h"

#define NDIM		4096
#define MIN(x,y)	((x < y) ? (x) : (y))

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

int print_matrix = 0;
int validation = 0;

int tnum = 1;
int tsqr = 1;
int bsize = 1;

void mat_mul_sub(void *t)
{
	int tid = (int) t;

	int ii, jj, kk, i, j, k;

	for (ii = ((tid - tid % tsqr) / tsqr) * bsize; ii < MIN(((tid - tid % tsqr) / tsqr) * bsize + bsize, NDIM); ii += bsize) {
		for (jj = (tid % tsqr) * bsize; jj < MIN((tid % tsqr) * bsize + bsize, NDIM); jj += bsize) {
			for (kk = 0; kk < NDIM; kk += bsize) {
				for (i = ii; i < MIN(ii + bsize, NDIM); i++) {
					for (j = jj; j < MIN(jj + bsize, NDIM); j++) {
						for (k = kk; k < MIN(kk + bsize, NDIM); k++) {
							c[i][j] += a[i][k] * b[k][j];
						}
					}
				}
			}
		}
	}

	pthread_exit((void *) t);
}

void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	pthread_t threads[tnum];
	pthread_attr_t attr;
	int t, rc;
	void *status;

	/* Calculate tsqr and bsize */
	tsqr = (int) sqrt(tnum);
	bsize = ceil((double)NDIM / (double)tsqr);

	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/* Create and execute threads */
	for (t = 0; t < tnum; t++) {
		rc = pthread_create(&threads[t], &attr, mat_mul_sub, (void *) t);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	/* Free attribute and wait for the other threads */
	pthread_attr_destroy(&attr);
	for (t = 0; t < tnum; t++) {
		rc = pthread_join(threads[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}
}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	int i, j, k;
	float sum;
	int validated = 1;

	printf("Validating the result..\n");

	// C = AB
	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			sum = 0;
			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i][k] * b[k][j];
			}

			if( c[i][j] != sum )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i][j], sum );
				validated = 0;
			}
		}
	}

	printf("Validation : ");
	if( validated )
		printf("SUCCESSFUL.\n");
	else
		printf("FAILED.\n");
}

void print_mat( float mat[NDIM][NDIM] )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i][j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [-pvh]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -v : validate matrix multiplication.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "pvhikjst:")) != -1 )
	{
		switch(opt)
		{
			case 'p':
				// print matrix data.
				print_matrix = 1;
				break;

			case 'v':
				// validation
				validation = 1;
				break;

			case 't':
				// set number of threads (has to be square numbers)
				tnum = atoi(optarg);
				break;

			case 'h':
			default:
				print_help(argv[0]);
				exit(0);
				break;
		}
	}
}

int main(int argc, char** argv)
{
	int i, j, k = 1;

	parse_opt( argc, argv );

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			a[i][j] = k;
			b[i][j] = k;
			k++;
		}
	}

	timer_start(1);
	mat_mul( c, a, b );
	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));


	if( validation )
		check_mat_mul( c, a, b );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a);

		printf("MATRIX B: \n");
		print_mat(b);

		printf("MATRIX C: \n");
		print_mat(c);
	}

	return 0;
}
