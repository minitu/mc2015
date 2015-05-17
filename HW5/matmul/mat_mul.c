#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "timers.h"

#define USE_GPU				1

#define MAX_SOURCE_SIZE		0x100000
#define KCNT				3			/* Number of different kernels
										 * (init, cpu, gpu) */

#define NDIM				4
#define BDIM				2
#define GLOBAL_WORK_SIZE	NDIM
#define LOCAL_WORK_SIZE		4			/* Should not exceed 32 on Chundoong CPU,
										 * because MAX_WORK_GROUP_SIZE = 1024 = 32 ^ 2. 
										 * Also, it should be a divisor of GLOBAL_WORK_SIZE. */
#define WORKLOAD			4			/* Maximum workload is equal to tile size,
										 * or LOCAL_WORK_SIZE */
#define MIN(x,y)			((x < y) ? (x) : (y))

int debug = 0;
int print_matrix = 0;

void print_mat( float* mat )
{
	int i;

	for (i = 0; i < NDIM * NDIM; i++) {
		if (i != 0 && i % NDIM == 0)
			printf("\n");
		printf("%8.2lf ", mat[i]);
	}
	printf("\n");
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [-dph]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -d : print debug info.\n");
	printf("  -p : print matrix data.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "dph:")) != -1 )
	{
		switch(opt)
		{
			case 'd':
				// print debug info
				debug = 1;
				break;

			case 'p':
				// print matrix data
				print_matrix = 1;
				break;

			case 'h':
				// print help
			default:
				print_help(argv[0]);
				exit(0);
				break;
		}
	}
}

int main(int argc, char** argv)
{
	parse_opt( argc, argv );

	timer_start(1); // DEBUG

	/* OpenCL variables */
	int err;
	int use_gpu = USE_GPU;

	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program programs[KCNT];
	cl_kernel kernels[KCNT];

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	/* Allocate host memory for matrices */
	unsigned long m_size = (unsigned long)NDIM * (unsigned long)NDIM; // Total matrix size
	float* h_A = (float*) malloc(sizeof(float) * m_size);
	float* h_B = (float*) malloc(sizeof(float) * m_size);
	float* h_C = (float*) malloc(sizeof(float) * m_size);

	/* Gather platform data */
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	/* Get platform info */
	int i;
	size_t info_size;
	char *platform_info;
	const cl_platform_info attrTypes[4] = {
		CL_PLATFORM_PROFILE,
		CL_PLATFORM_VERSION,
		CL_PLATFORM_NAME,
		CL_PLATFORM_VENDOR };

	for (i = 0; i < 4; i++) {
		err = clGetPlatformInfo(platform_ids[0], attrTypes[i], 0, NULL, &info_size);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to get info size.\n");
			return EXIT_FAILURE;
		}
		platform_info = (char*) malloc(info_size);
		err = clGetPlatformInfo(platform_ids[0], attrTypes[i], info_size, platform_info, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to get platform info.\n");
			return EXIT_FAILURE;
		}

		if (debug)
			printf("%s\n", platform_info); // DEBUG

		free(platform_info);
	}


	/* Connect to compute device */
	err = clGetDeviceIDs(platform_ids[0], USE_GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to connect to compute device.\n");
		return EXIT_FAILURE;
	}

	/* Create a compute context */
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		printf("Error: Failed to create a compute context.\n");
		return EXIT_FAILURE;
	}

	/* Create an in-order command queue and attach it to the compute device */
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) {
		printf("Error: Failed to create a command queue.\n");
		return EXIT_FAILURE;
	}

	/* Create init & compute programs from source files */
	FILE *fp;
	char *fileName[KCNT];
	char *src_str;
	size_t src_size;

	for (i = 0; i < KCNT; i++) {
		fileName[i] = (char*) malloc(100 * sizeof(char));
	}

	strcpy(fileName[0], "./init.cl");
	strcpy(fileName[1], "./cpu.cl");
	strcpy(fileName[2], "./gpu.cl");

	for (i = 0; i < KCNT; i++) {
		fp = fopen(fileName[i], "r");
		if (!fp) {
			perror("File read failed");
			return 1;
		}
		src_str = (char*) malloc(MAX_SOURCE_SIZE);
		src_size = fread(src_str, 1, MAX_SOURCE_SIZE, fp);

		programs[i] = clCreateProgramWithSource(context, 1, (const char **)&src_str, (const size_t *)&src_size, &err);

		if (!programs[i]) {
			printf("Error: Failed to create program %d.\n", i);
			return EXIT_FAILURE;
		}

		fclose(fp);
		free(src_str);
		free(fileName[i]);
	}

	/* Build the program executables */
	for (i = 0; i < KCNT; i++) {
		err = clBuildProgram(programs[i], 0, NULL, NULL, NULL, NULL);

		if (err != CL_SUCCESS) {
			size_t len;
			char buffer[2048];
			printf("Error: Failed to build program %d executable.\n", i);
			clGetProgramBuildInfo(programs[i], device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			exit(1);
		}
	}

	/* Create init & compute kernels */
	char kernelName[100];

	for (i = 0; i < KCNT; i++) {
		if (i == 0) strcpy(kernelName, "mat_mul_init");
		else if (i == 1) strcpy(kernelName, "mat_mul_cpu");
		else if (i == 2) strcpy(kernelName, "mat_mul_gpu");

		kernels[i] = clCreateKernel(programs[i], kernelName, &err);
		if (!kernels[i] || err != CL_SUCCESS) {
			printf("Error: Failed to create kernel %d.\n", i);
			exit(1);
		}
	}

	/* Find out maximum work group size & maximum work item sizes */
	size_t max_work_group_size;
	size_t max_work_item_sizes[3];

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to get CL_DEVICE_MAX_WORK_GROUP_SIZE.\n");
		exit(1);
	}

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to get CL_DEVICE_MAX_WORK_ITEM_SIZES.\n");
		exit(1);
	}

	// DEBUG
	if (debug) {
		printf("\n");
		printf("max_work_group_size: %zu\n", max_work_group_size);
		printf("max_work_item_sizes[0]: %zu\n", max_work_item_sizes[0]);
		printf("max_work_item_sizes[1]: %zu\n\n", max_work_item_sizes[1]);
	}

	if (!USE_GPU) {
		/******************** CPU ********************/

		/* Create buffers for matrices in device global memory */
		d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * m_size, h_A, &err);
		d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * m_size, h_B, &err);;
		d_C = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * m_size, h_C, &err);
		if (!d_A || !d_B || !d_C) {
			printf("Error: Failed to allocate device memory.\n");
			exit(1);
		}

		/* Set kernel arguments and launch init */
		size_t initLocalWorkSize[2], initGlobalWorkSize[2];
		size_t localWorkSize[2], globalWorkSize[2];
		int tileSize = LOCAL_WORK_SIZE;
		int tileNum = ceil((double)NDIM / (double)tileSize);
		int ndim = NDIM;
		int sdim = ceil((double)NDIM / (double)GLOBAL_WORK_SIZE);
		int workload = WORKLOAD;
		int rts = tileSize / workload;
		float startNum = 0.0f + 1;

		// Set work sizes
		for (i = 0; i < 2; i++) {
			initLocalWorkSize[i] = LOCAL_WORK_SIZE;
			initGlobalWorkSize[i] = GLOBAL_WORK_SIZE;
		}

		localWorkSize[0] = tileSize;
		localWorkSize[1] = rts;
		globalWorkSize[0] = ndim;
		globalWorkSize[1] = ndim / workload;

		err = clSetKernelArg(kernels[0], 0, sizeof(cl_mem), (void *)&d_A);
		err |= clSetKernelArg(kernels[0], 1, sizeof(cl_mem), (void *)&d_B);
		err |= clSetKernelArg(kernels[0], 2, sizeof(int), (void *)&ndim);
		err |= clSetKernelArg(kernels[0], 3, sizeof(int), (void *)&sdim);
		err |= clSetKernelArg(kernels[0], 4, sizeof(float), (void *)&startNum);
		err |= clSetKernelArg(kernels[0], 5, sizeof(int), (void *)&use_gpu);
		err |= clSetKernelArg(kernels[0], 6, sizeof(int), (void *)&use_gpu);
		err |= clSetKernelArg(kernels[0], 7, sizeof(int), (void *)&use_gpu);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to set init kernel arguments. %d\n", err);
			exit(1);
		}

		err = clEnqueueNDRangeKernel(commands, kernels[0], 2, NULL, initGlobalWorkSize, initLocalWorkSize, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to execute init kernel. %d\n", err);
			exit(1);
		}

		/* Read A and B matrices to host memory */
		err = clEnqueueReadBuffer(commands, d_A, CL_TRUE, 0, sizeof(float) * m_size, h_A, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(commands, d_B, CL_TRUE, 0, sizeof(float) * m_size, h_B, 0, NULL, NULL);

		/* Set kernel arguments and launch compute */
		err = clSetKernelArg(kernels[1], 0, sizeof(cl_mem), (void *)&d_A);
		err |= clSetKernelArg(kernels[1], 1, sizeof(cl_mem), (void *)&d_B);
		err |= clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void *)&d_C);
		err |= clSetKernelArg(kernels[1], 3, sizeof(float) * tileSize * tileSize, NULL);
		err |= clSetKernelArg(kernels[1], 4, sizeof(float) * tileSize * tileSize, NULL);
		err |= clSetKernelArg(kernels[1], 5, sizeof(int), (void *)&ndim);
		err |= clSetKernelArg(kernels[1], 6, sizeof(int), (void *)&tileSize);
		err |= clSetKernelArg(kernels[1], 7, sizeof(int), (void *)&tileNum);
		err |= clSetKernelArg(kernels[1], 8, sizeof(int), (void *)&workload);
		err |= clSetKernelArg(kernels[1], 9, sizeof(int), (void *)&rts);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to set compute kernel arguments. %d\n", err);
			exit(1);
		}

		err = clEnqueueNDRangeKernel(commands, kernels[1], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to execute compute kernel. %d\n", err);
			exit(1);
		}

		/* Retrieve result from device */
		err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, sizeof(float) * m_size, h_C, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to read output array. %d\n", err);
			exit(1);
		}

	}
	else {
		/******************** GPU ********************/

		/* To initialize, divide each matrix into sets of rows */
		int set_rows = NDIM; // Number of rows per set (each row has 100,000 columns max)
		int set_cnt = 1;
		unsigned long set_size;

		if (NDIM > 512) {
			set_rows = 512;
			set_cnt = ceil((double)NDIM / (double)set_rows);
		}

		set_size = (unsigned long)set_rows * (unsigned long)NDIM;

		/* Variable definitions */
		size_t initGlobalWorkSize[1];
		initGlobalWorkSize[0] = set_rows;
		int ndim = NDIM;
		int sdim = 1;
		float startNum = 0.0f + 1;
		int setNum = 0;

		/* Create buffers for matrices in device global memory */
		d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * set_size, NULL, &err);
		d_B = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * set_size, NULL, &err);
		if (!d_A || !d_B) {
			printf("Error: Failed to allocate device memory.\n");
			exit(1);
		}

		/* Iterate init for A and B */
		for (i = 0; i < set_cnt; i++) {
			/* Set kernel arguments and execute init */
			setNum = i;

			err = clSetKernelArg(kernels[0], 0, sizeof(cl_mem), (void *)&d_A);
			err |= clSetKernelArg(kernels[0], 1, sizeof(cl_mem), (void *)&d_B);
			err |= clSetKernelArg(kernels[0], 2, sizeof(int), (void *)&ndim);
			err |= clSetKernelArg(kernels[0], 3, sizeof(int), (void *)&sdim);
			err |= clSetKernelArg(kernels[0], 4, sizeof(float), (void *)&startNum);
			err |= clSetKernelArg(kernels[0], 5, sizeof(int), (void *)&setNum);
			err |= clSetKernelArg(kernels[0], 6, sizeof(int), (void *)&set_rows);
			err |= clSetKernelArg(kernels[0], 7, sizeof(int), (void *)&use_gpu);

			if (err != CL_SUCCESS) {
				printf("Error: Failed to set init kernel arguments. %d\n", err);
				exit(1);
			}

			err = clEnqueueNDRangeKernel(commands, kernels[0], 1, NULL, initGlobalWorkSize, NULL, 0, NULL, NULL);

			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute init kernel. %d\n", err);
				exit(1);
			}

			startNum += (float)set_size;

			size_t copy_size = sizeof(float) * set_size;

			// Special care for last set, if NDIM is not a multiple of the number of rows in a set
			if ((i == set_cnt - 1) && (NDIM % set_rows != 0)) { 
				copy_size = sizeof(float) * (NDIM % set_rows) * NDIM;
			}

			/* Retrieve result from device */
			err = clEnqueueReadBuffer(commands, d_A, CL_TRUE, 0, copy_size, h_A + set_size * i, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(commands, d_B, CL_TRUE, 0, copy_size, h_B + set_size * i, 0, NULL, NULL);

			if (err != CL_SUCCESS) {
				printf("Error: Failed to read arrays. %d\n", err);
				exit(1);
			}
		}

		/* Free buffers */
		clReleaseMemObject(d_A);
		clReleaseMemObject(d_B);

		/* To compute, divide area of C into 4096 X 4096 matrix blocks */
		int blk_dim = NDIM; // Number of rows per set (each row has 100,000 columns max)
		int blk_cnt = 1;
		unsigned long blk_size;

		if (NDIM > BDIM) {
			blk_dim = BDIM;
			blk_cnt = ceil((double)NDIM / (double)blk_dim);
		}

		blk_size = (unsigned long)blk_dim * (unsigned long)blk_dim;

		/* Variable definitions */
		size_t localWorkSize[2], globalWorkSize[2];
		int tileSize = LOCAL_WORK_SIZE;
		int tileNum = ceil((double)blk_dim / (double)tileSize);
		int workload = WORKLOAD;
		int rts = tileSize / workload;

		// Set work sizes
		for (i = 0; i < 2; i++) {
			//localWorkSize[i] = LOCAL_WORK_SIZE;
			globalWorkSize[i] = blk_dim;
		}

		/*
		   localWorkSize[0] = tileSize;
		   localWorkSize[1] = rts;
		   globalWorkSize[0] = ndim;
		   globalWorkSize[1] = ndim / workload;
		   */

		/* Create buffers for matrices in device global memory */
		d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * blk_size, NULL, &err);
		d_B = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * blk_size, NULL, &err);
		d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * blk_size, NULL, &err);
		if (!d_A || !d_B || !d_C) {
			printf("Error: Failed to allocate device memory.\n");
			exit(1);
		}

		int j, k, l;
		int ch = blk_dim; // Height of C block
		int cw = blk_dim; // Width of C block
		const int float_size = sizeof(float);
		const size_t buffer_origin[3] = {0, 0, 0};
		size_t host_origin[3] = {0, 0, 0};
		size_t region[3] = {1, 1, 1};
		const size_t buffer_row_pitch = float_size * blk_dim;
		const size_t host_row_pitch = float_size * NDIM;
		const size_t host_slice_pitch = float_size * blk_size;
		float pattern = 0.0f;
		int leftover = blk_dim;
		float* mid_blks = (float*) malloc(float_size * blk_size * blk_cnt); // Store intermediate C blocks here for later accumulation

		/* Iterate compute loops */
		for (i = 0; i < blk_cnt; i++) { // C block moves down
			if (i == blk_cnt - 1) {
				ch = MIN(blk_dim, NDIM - blk_dim * i); // For last row, height might be smaller
			}
			for (j = 0; j < blk_cnt; j++) { // C block moves right
				if (j == blk_cnt - 1) {
					cw = MIN(blk_dim, NDIM - blk_dim * j); // For last column, width might be smaller
				}
				for (k = 0; k < blk_cnt; k++) { // A and B blocks move

					if (k == blk_cnt - 1) {
						leftover = MIN(blk_dim, NDIM - blk_dim * k);
					}

					// Fill buffers with 0
					err = clEnqueueFillBuffer(commands, d_A, (const void*)&pattern, float_size, 0, float_size * blk_size, 0, NULL, NULL);
					err |= clEnqueueFillBuffer(commands, d_B, (const void*)&pattern, float_size, 0, float_size * blk_size, 0, NULL, NULL);

					if (err != CL_SUCCESS) {
						printf("Error: Failed to fill buffers. %d\n", err);
						exit(1);
					}

					// Host origin for A block
					host_origin[0] = float_size * blk_dim * k;
					host_origin[1] = blk_dim * i;
					host_origin[2] = 0;

					// Region of A block
					if (k == blk_cnt - 1) {
						region[0] = float_size * leftover;
					}
					else {
						region[0] = float_size * blk_dim;
					}
					region[1] = ch;

					// Copy A block into buffer
					err = clEnqueueWriteBufferRect(commands, d_A, CL_TRUE, buffer_origin, host_origin, region, buffer_row_pitch, 0, host_row_pitch, 0, h_A, 0, NULL, NULL);

					if (err != CL_SUCCESS) {
						printf("Error: Failed to write A block into buffer. %d\n", err);
						exit(1);
					}
					
					// Host origin for B block
					host_origin[0] = float_size * blk_dim * j;
					host_origin[1] = blk_dim * k;
					host_origin[2] = 0;

					// Region of B block
					region[0] = float_size * cw;
					if (k == blk_cnt - 1) {
						region[1] = leftover;
					}
					else {
						region[1] = blk_dim;
					}

					// Copy B block into buffer
					err = clEnqueueWriteBufferRect(commands, d_B, CL_TRUE, buffer_origin, host_origin, region, buffer_row_pitch, 0, host_row_pitch, 0, h_B, 0, NULL, NULL);

					if (err != CL_SUCCESS) {
						printf("Error: Failed to write B block into buffer. %d\n", err);
						exit(1);
					}

					// Set kernel arguments and launch compute
					err = clSetKernelArg(kernels[2], 0, sizeof(cl_mem), (void *)&d_A);
					err |= clSetKernelArg(kernels[2], 1, sizeof(cl_mem), (void *)&d_B);
					err |= clSetKernelArg(kernels[2], 2, sizeof(cl_mem), (void *)&d_C);
					//err |= clSetKernelArg(kernels[2], 3, sizeof(float) * tileSize * tileSize, NULL);
					//err |= clSetKernelArg(kernels[2], 4, sizeof(float) * tileSize * tileSize, NULL);
					err |= clSetKernelArg(kernels[2], 3, sizeof(int), (void *)&blk_dim);
					//err |= clSetKernelArg(kernels[2], 6, sizeof(int), (void *)&tileSize);
					//err |= clSetKernelArg(kernels[2], 7, sizeof(int), (void *)&tileNum);
					//err |= clSetKernelArg(kernels[2], 8, sizeof(int), (void *)&workload);
					//err |= clSetKernelArg(kernels[2], 9, sizeof(int), (void *)&rts);

					if (err != CL_SUCCESS) {
						printf("Error: Failed to set compute kernel arguments. %d\n", err);
						exit(1);
					}

					err = clEnqueueNDRangeKernel(commands, kernels[2], 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

					if (err != CL_SUCCESS) {
						printf("Error: Failed to execute compute kernel. %d\n", err);
						exit(1);
					}

					// Host origin for C block
					host_origin[0] = 0;
					host_origin[1] = 0;
					host_origin[2] = k;

					printf("C host_origin: %zu\n", host_origin[2] * host_slice_pitch);
					
					// Region of C block
					region[0] = float_size * blk_dim;
					region[1] = blk_dim;

					// Retrieve C block from buffer
					err = clEnqueueReadBufferRect(commands, d_C, CL_TRUE, buffer_origin, host_origin, region, buffer_row_pitch, 0, 0, host_slice_pitch, mid_blks, 0, NULL, NULL);

					//DEBUG
						printf("\n--mid_blks--\n");
						int m;
						for (m = 0; m < blk_size * 2; m++) {
							printf("%f ", mid_blks[m]);
						}
						printf("\n");

					if (err != CL_SUCCESS) {
						printf("Error: Failed to read C block from buffer. %d\n", err);
						exit(1);
					}
				}
				
				// Finished one C block, now accumulate
				for (k = 1; k < blk_cnt; k++) {
					for (l = 0; l < blk_size; l++) {
						mid_blks[l] += mid_blks[k * blk_size + l];
					}
				}

				// C block result stored in the first block of mid_blks
				int ci = blk_dim * i;
				int cj = blk_dim * j;
				int ti, tj;

				for (ti = 0; ti < ch; ti++) {
					for (tj = 0; tj < cw; tj++) {
						h_C[(ci + ti) * NDIM + cj + tj] = mid_blks[ti * blk_dim + tj];
					}
				}
			}
		}

		free(mid_blks);
	}

	timer_stop(1); // DEBUG

	printf("Time elapsed : %lf sec\n", timer_read(1));

	if( print_matrix )
	{
		printf("\nMATRIX A: \n");
		print_mat(h_A);

		printf("MATRIX B: \n");
		print_mat(h_B);

		printf("MATRIX C: \n");
		print_mat(h_C);
	}

	/* Cleanup */
	free(h_A);
	free(h_B);
	free(h_C);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	for (i = 0; i < KCNT; i++) {
		clReleaseProgram(programs[i]);
		clReleaseKernel(kernels[i]);
	}
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
