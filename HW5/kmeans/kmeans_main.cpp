#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <CL/opencl.h>

#define USE_GPU				0
#define DEBUG				1

#define MAX_SOURCE_SIZE		0x100000
#define KCNT				2

#define GLOBAL_WORK_SIZE	32
#define LOCAL_WORK_SIZE		32

#define DATA_DIM			2
#define DEFAULT_ITERATION	1024

#define GET_TIME(T) __asm__ __volatile__ ("rdtsc\n" : "=A" (T))


// Read data from file
unsigned int read_data(FILE* f, float** data_p);
int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);


int main(int argc, char** argv)
{
    int class_n, data_n, iteration_n;
    float *centroids, *data;
    int* partitioned;
    FILE *io_file;
    struct timespec start, end, spent;

    // Check parameters
    if (argc < 4) {
        fprintf(stderr, "usage: %s <centroid file> <data file> <paritioned result> [<final centroids>] [<iteration number>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

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

    iteration_n = argc > 5 ? atoi(argv[5]) : DEFAULT_ITERATION;
        

    partitioned = (int*)malloc(sizeof(int)*data_n);

	/* OpenCL */
	int err;

	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program programs[KCNT];
	cl_kernel kernels[KCNT];

	cl_mem d_centroids;
	cl_mem d_data;
	cl_mem d_partitioned;

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

		if (DEBUG)
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

	strcpy(fileName[0], "./cpu.cl");
	strcpy(fileName[1], "./gpu.cl");

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
		if (i == 0) strcpy(kernelName, "kmeans_cpu");
		else if (i == 1) strcpy(kernelName, "kmeans_gpu");

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
	if (DEBUG) {
		printf("\n");
		printf("max_work_group_size: %zu\n", max_work_group_size);
		printf("max_work_item_sizes[0]: %zu\n", max_work_item_sizes[0]);
		printf("max_work_item_sizes[1]: %zu\n\n", max_work_item_sizes[1]);
	}

    clock_gettime(CLOCK_MONOTONIC, &start);
    
	// Run Kmeans algorithm
	int my_data_n = ceil((double)data_n / (double)GLOBAL_WORK_SIZE);
	
	size_t globalWorkSize, localWorkSize;

	if (!USE_GPU) {
		/******************** CPU ********************/

		// Allocate buffers
		d_centroids = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * DATA_DIM * class_n, centroids, &err);
		d_data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * DATA_DIM * data_n, data, &err);
		d_partitioned = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(int) * data_n, partitioned, &err);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to allocate device memory. %d\n", err);
			exit(1);
		}

		// Set work sizes (all global work items are in a single work group)
		globalWorkSize = GLOBAL_WORK_SIZE;
		localWorkSize = GLOBAL_WORK_SIZE;

		// Set kernel arguments
		err = clSetKernelArg(kernels[0], 0, sizeof(int), (void*) &iteration_n);
		err |= clSetKernelArg(kernels[0], 1, sizeof(int), (void*) &class_n);
		err |= clSetKernelArg(kernels[0], 2, sizeof(int), (void*) &data_n);
		err |= clSetKernelArg(kernels[0], 3, sizeof(cl_mem), (void*) &d_centroids);
		err |= clSetKernelArg(kernels[0], 4, sizeof(cl_mem), (void*) &d_data);
		err |= clSetKernelArg(kernels[0], 5, sizeof(cl_mem), (void*) &d_partitioned);
		err |= clSetKernelArg(kernels[0], 6, sizeof(int) * class_n, NULL);
		err |= clSetKernelArg(kernels[0], 7, sizeof(int), (void*) &my_data_n);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to set kernel arguments. %d\n", err);
			exit(1);
		}

		// Execute kernel
		err = clEnqueueNDRangeKernel(commands, kernels[0], 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to execute kernel. %d\n", err);
			exit(1);
		}

		// Read data back to host memory
		err = clEnqueueReadBuffer(commands, d_centroids, CL_TRUE, 0, sizeof(float) * DATA_DIM * class_n, centroids, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(commands, d_data, CL_TRUE, 0, sizeof(float) * DATA_DIM * data_n, data, 0, NULL, NULL);
		err |= clEnqueueReadBuffer(commands, d_partitioned, CL_TRUE, 0, sizeof(int) * data_n, partitioned, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: Failed to execute kernel. %d\n", err);
			exit(1);
		}
	}
	else {
		/******************** GPU ********************/
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

    timespec_subtract(&spent, &end, &start);
    printf("Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

    // Write classified result
    io_file = fopen(argv[3], "wb");
    fwrite(&data_n, sizeof(data_n), 1, io_file);
    fwrite(partitioned, sizeof(int), data_n, io_file); 
    fclose(io_file);


    // Write final centroid data
    if (argc > 4) {
        io_file = fopen(argv[4], "wb");
        fwrite(&class_n, sizeof(class_n), 1, io_file);
        fwrite(centroids, sizeof(Point), class_n, io_file); 
        fclose(io_file);
    }


    // Free allocated buffers
    free(centroids);
    free(data);
    free(partitioned);

	/* Cleanup */
	clReleaseMemObject(d_centroids);
	clReleaseMemObject(d_data);
	clReleaseMemObject(d_partitioned);

	for (i = 0; i < KCNT; i++) {
		clReleaseProgram(programs[i]);
		clReleaseKernel(kernels[i]);
	}
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

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

