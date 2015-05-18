#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "timers.h"

// Set if debug
#define DEBUG				1

// 0 for CPU, 1 for GPU
#define USE_GPU				1

// Maximum kernel source code size
#define MAX_SOURCE_SIZE		0x100000

// Number of different kernels (init, compute)
#define KCNT				2

// Matrix dimension
#define NDIM				10000

// Tile dimension, equal to dimension of work group.
// Should not exceed 32 on CPU and 16 on GPU,
// and should be a divisor of NDIM.
#define TDIM				16

// Minimum function
#define MIN(x,y)			((x < y) ? (x) : (y))

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
	printf("Usage: %s [-ph]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "ph:")) != -1 )
	{
		switch(opt)
		{
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

	/* OpenCL variables */
	int err;

	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program programs[KCNT];
	cl_kernel kernels[KCNT];

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

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

	if (DEBUG)
		printf("*** Platform Information***\n");

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

		if (DEBUG)
			printf("%s\n", platform_info);

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
	strcpy(fileName[1], "./compute.cl");

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
	char define_str[10] = "-DTDIM=";
	char tdim_str[10];
	sprintf(tdim_str, "%d", TDIM);
	char *build_options = malloc(sizeof(char) * 100);
	build_options = strcat(define_str, tdim_str);
	if (DEBUG) {
		printf("*** Kernel Build Options ***\n");
		printf("%s\n", build_options);
	}

	for (i = 0; i < KCNT; i++) {
		if (i == 0) {
			err = clBuildProgram(programs[i], 0, NULL, NULL, NULL, NULL);
		}
		else {
			err = clBuildProgram(programs[i], 0, NULL, build_options, NULL, NULL);
		}

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
		else if (i == 1) strcpy(kernelName, "mat_mul_compute");

		kernels[i] = clCreateKernel(programs[i], kernelName, &err);
		if (!kernels[i] || err != CL_SUCCESS) {
			printf("Error: Failed to create kernel %d.\n", i);
			exit(1);
		}
	}

	/* Query for device specific info */
	size_t max_work_group_size;

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to get device info.\n");
		exit(1);
	}

	if (DEBUG) {
		printf("*** Device Information ***\n");
		printf("Maximum work group size: %zu\n", max_work_group_size);
	}

	/* Allocate host memory for matrices */
	unsigned long m_size = (unsigned long)NDIM * (unsigned long)NDIM; // Total matrix size
	float* h_A = (float*) malloc(sizeof(float) * m_size);
	float* h_B = (float*) malloc(sizeof(float) * m_size);
	float* h_C = (float*) malloc(sizeof(float) * m_size);

	/* Create buffers for matrices in device global memory */
	d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * m_size, h_A, &err);
	d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * m_size, h_B, &err);;
	d_C = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * m_size, h_C, &err);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to allocate device memory.\n");
		exit(1);
	}

	// Kernel variables
	size_t initGlobalWorkSize[2];
	size_t globalWorkSize[2], localWorkSize[2];
	int ndim = NDIM;
	float startNum = 0.0f + 1;

	// Set work sizes
	for (i = 0; i < 2; i++) {
		initGlobalWorkSize[i] = NDIM;
		globalWorkSize[i] = NDIM;
		localWorkSize[i] = TDIM;
	}

	/* Launch init */
	err = clSetKernelArg(kernels[0], 0, sizeof(cl_mem), (void *)&d_A);
	err |= clSetKernelArg(kernels[0], 1, sizeof(cl_mem), (void *)&d_B);
	err |= clSetKernelArg(kernels[0], 2, sizeof(int), (void *)&ndim);
	err |= clSetKernelArg(kernels[0], 3, sizeof(float), (void *)&startNum);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set init kernel arguments. %d\n", err);
		exit(1);
	}

	err = clEnqueueNDRangeKernel(commands, kernels[0], 2, NULL, initGlobalWorkSize, NULL, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to execute init kernel. %d\n", err);
		exit(1);
	}

	/* Read A and B matrices to host memory */
	err = clEnqueueReadBuffer(commands, d_A, CL_TRUE, 0, sizeof(float) * m_size, h_A, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(commands, d_B, CL_TRUE, 0, sizeof(float) * m_size, h_B, 0, NULL, NULL);

	/* Launch compute */
	err = clSetKernelArg(kernels[1], 0, sizeof(cl_mem), (void *)&d_A);
	err |= clSetKernelArg(kernels[1], 1, sizeof(cl_mem), (void *)&d_B);
	err |= clSetKernelArg(kernels[1], 2, sizeof(cl_mem), (void *)&d_C);
	err |= clSetKernelArg(kernels[1], 3, sizeof(int), (void *)&ndim);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set compute kernel arguments. %d\n", err);
		exit(1);
	}

	// Start timing
	timer_start(1);
	
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

	// End timing
	timer_stop(1);

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
