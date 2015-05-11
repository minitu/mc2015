#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <stdbool.h>
#include <math.h>
#include "timers.h"

#define USE_GPU				0

#define MAX_SOURCE_SIZE		0x100000

#define NDIM				1024
#define GLOBAL_WORK_SIZE	1024
#define LOCAL_WORK_SIZE		1			// Should not exceed 32 on Chundoong CPU,
										// because MAX_WORK_GROUP_SIZE = 1024 = 32 ^ 2.

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
	printf("Usage: %s [-pvh]\n", prog_name );
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
			// print matrix data.
			print_matrix = 1;
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
	parse_opt( argc, argv );

	/* OpenCL variables */
	int err;
	
	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program init_program, compute_program;
	cl_kernel init, compute;

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	/* Allocate host memory for matrices */
	unsigned long m_size = (unsigned long)NDIM * (unsigned long)NDIM;
	float* h_A = (float*) malloc(m_size * sizeof(float));
	float* h_B = (float*) malloc(m_size * sizeof(float));
	float* h_C = (float*) malloc(m_size * sizeof(float));

	timer_start(1); // DEBUG

	/* Gather platform data */
	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	/* Connect to a compute device */
	err = clGetDeviceIDs(platform_ids[0], USE_GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group.\n");
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

	/* Create init & compute program from source files */
	FILE *fp1, *fp2;
	char initFileName[] = "./mat_mul_init.cl";
	char computeFileName[] = "./mat_mul_compute.cl";
	char *src_str1, *src_str2;
	size_t src_size1, src_size2;

	fp1 = fopen(initFileName, "r");
	fp2 = fopen(computeFileName, "r");
	if (!fp1 || !fp2) {
		perror("File read failed");
		return 1;
	}
	src_str1 = (char*) malloc(MAX_SOURCE_SIZE);
	src_str2 = (char*) malloc(MAX_SOURCE_SIZE);
	src_size1 = fread(src_str1, 1, MAX_SOURCE_SIZE, fp1);
	src_size2 = fread(src_str2, 1, MAX_SOURCE_SIZE, fp2);

	init_program = clCreateProgramWithSource(context, 1, (const char **)&src_str1, (const size_t *)&src_size1, &err);
	if (!init_program) {
		printf("Error: Failed to create compute program.\n");
		return EXIT_FAILURE;
	}

	compute_program = clCreateProgramWithSource(context, 1, (const char **)&src_str2, (const size_t *)&src_size2, &err);
	if (!compute_program) {
		printf("Error: Failed to create compute program.\n");
		return EXIT_FAILURE;
	}

	fclose(fp1);
	fclose(fp2);

	/* Build the program executable */
	err = clBuildProgram(init_program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable.\n");
		clGetProgramBuildInfo(init_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	err = clBuildProgram(compute_program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable.\n");
		clGetProgramBuildInfo(compute_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	/* Create init & compute kernel */
	init = clCreateKernel(init_program, "mat_mul_init", &err);
	if (!init || err != CL_SUCCESS) {
		printf("Error: Failed to create init kernel.\n");
		exit(1);
	}
	compute = clCreateKernel(compute_program, "mat_mul_compute", &err);
	if (!compute || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel.\n");
		exit(1);
	}

	/* Create buffers for matrices in device global memory */
	if (!USE_GPU) { // If using CPU, just use the host memory
		d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_size, h_A, &err);
		d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_size, h_B, &err);;
		d_C = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_size, h_C, &err);
		if (!d_A || !d_B || !d_C) {
			printf("Error: Failed to allocate device memory.\n");
			exit(1);
		}
	}
	else { // If using GPU, matrices need to be split up to fit into 3GB
		d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, m_size, h_A, &err);
		d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, m_size, h_B, &err);;
		d_C = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, m_size, h_C, &err);
		if (!d_A || !d_B || !d_C) {
			printf("Error: Failed to allocate device memory.\n");
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
	printf("max_work_group_size: %d\n", max_work_group_size);
	printf("max_work_item_sizes[0]: %d\n", max_work_item_sizes[0]);
	printf("max_work_item_sizes[1]: %d\n", max_work_item_sizes[1]);

	/* Set kernel arguments and launch init */
	size_t localWorkSize[2], globalWorkSize[2];
	int i;
	int ndim = NDIM;
	int sdim = ceil((double)NDIM / (double)GLOBAL_WORK_SIZE);
	float startNum = 0.0f + 1;

	for (i = 0; i < 2; i++) {
		localWorkSize[i] = LOCAL_WORK_SIZE;
		globalWorkSize[i] = GLOBAL_WORK_SIZE;
	}

	if (!USE_GPU) { // CPU
		err = clSetKernelArg(init, 0, sizeof(cl_mem), (void *)&d_A);
		err |= clSetKernelArg(init, 1, sizeof(cl_mem), (void *)&d_B);
		err |= clSetKernelArg(init, 2, sizeof(int), (void *)&ndim);
		err |= clSetKernelArg(init, 3, sizeof(int), (void *)&sdim);
		err |= clSetKernelArg(init, 4, sizeof(float), (void *)&startNum);
	}
	else { // GPU
		// Do something
	}

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set init kernel arguments. %d\n", err);
		exit(1);
	}

	err = clEnqueueNDRangeKernel(commands, init, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to execute init kernel. %d\n", err);
		exit(1);
	}

	/* Set kernel arguments and launch compute */
	if (!USE_GPU) { // CPU
		err = clSetKernelArg(compute, 0, sizeof(cl_mem), (void *)&d_A);
		err |= clSetKernelArg(compute, 1, sizeof(cl_mem), (void *)&d_B);
		err |= clSetKernelArg(compute, 2, sizeof(cl_mem), (void *)&d_C);
		err |= clSetKernelArg(compute, 3, sizeof(int), (void *)&ndim);
	}
	else { // GPU
		// Do something
	}

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set compute kernel arguments. %d\n", err);
		exit(1);
	}

	err = clEnqueueNDRangeKernel(commands, compute, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to execute compute kernel. %d\n", err);
		exit(1);
	}

	/* Retrieve result from device */
	err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, m_size, h_C, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array. %d\n", err);
		exit(1);
	}

	// DEBUG
	clEnqueueReadBuffer(commands, d_A, CL_TRUE, 0, m_size, h_A, 0, NULL, NULL);
	clEnqueueReadBuffer(commands, d_B, CL_TRUE, 0, m_size, h_B, 0, NULL, NULL);

	timer_stop(1); // DEBUG

	printf("Time elapsed : %lf sec\n", timer_read(1));

	if( print_matrix )
	{
		/*
		printf("MATRIX A: \n");
		print_mat(h_A);

		printf("MATRIX B: \n");
		print_mat(h_B);
		*/

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

	clReleaseProgram(init_program);
	clReleaseProgram(compute_program);
	clReleaseKernel(init);
	clReleaseKernel(compute);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
