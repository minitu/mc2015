#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "timers.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define NDIM		4096
#define BDIM		512
#define TNUM		32

#define	MIN(x,y)	((x < y) ? (x) : (y))

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

int print_matrix = 0;
int validation = 0;

void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	int ii, jj, kk;
	int i, j, k;
	int iend, jend, kend;

#ifdef _OPENMP
	omp_set_dynamic(0); // Disable dynamic teams
	omp_set_num_threads(TNUM); // Set number of threads
#endif

	/* Naive
#pragma omp parallel for private(j,k)
	for (i = 0; i < NDIM; i++) {
		for (j = 0; j < NDIM; j++) {
			for (k = 0; k < NDIM; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	*/

	// Tiling
#pragma omp parallel for collapse(2) private(kk,i,j,k)
	for(ii = 0; ii < NDIM; ii += BDIM) {
		for (jj = 0; jj < NDIM; jj += BDIM) {
			for (kk = 0; kk < NDIM; kk += BDIM) {
				iend = MIN(ii + BDIM, NDIM);
				for (i = ii; i < iend; i++) {
					jend = MIN(jj + BDIM, NDIM);
					for (j = jj; j < jend; j++) {
						kend = MIN(kk + BDIM, NDIM);
						for (k = kk; k < kend; k++) {
							c[i][j] += a[i][k] * b[k][j];
						}
					}
				}
			}
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

	while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
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
