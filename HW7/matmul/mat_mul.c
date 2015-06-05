#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "timers.h"
#include "mpi.h"

#define NDIM		8192
#define TDIM		8
#define MIN(x,y)	((x < y) ? (x) : (y))

float a[NDIM][NDIM];
float b[NDIM][NDIM];
float c[NDIM][NDIM];

int print_matrix = 0;
int validation = 0;

int st, ed;

void mat_mul( float c[NDIM][NDIM], float a[NDIM][NDIM], float b[NDIM][NDIM] )
{
	int ii, jj, kk, i, j, k;
	int ied, jed, ked;

	// C = AB
	for(ii = st; ii < ed; ii += TDIM) {
		for (jj = 0; jj < NDIM; jj += TDIM) {
			for (kk = 0; kk < NDIM; kk += TDIM) {
				ied = MIN(ii + TDIM, NDIM);
				for (i = ii; i < ied; i++) {
					jed = MIN(jj + TDIM, NDIM);
					for (j = jj; j < jed; j++) {
						ked = MIN(kk + TDIM, NDIM);
						for (k = kk; k < ked; k++) {
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
	int comm_rank, comm_size; // MPI

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

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (NDIM % comm_size != 0) {
		if (comm_rank == 0) printf("Matrix size should be a multiple of number of processes.\n");
		MPI_Finalize();
		return -1;
	}

	st = comm_rank * (NDIM / comm_size);
	ed = (comm_rank + 1) * (NDIM / comm_size);

	mat_mul( c, a, b );

	MPI_Gather(c[st], NDIM * NDIM / comm_size, MPI_FLOAT, c, NDIM * NDIM / comm_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	timer_stop(1);

	if (comm_rank == 0) {
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
	}

	MPI_Finalize();

	return 0;
}
