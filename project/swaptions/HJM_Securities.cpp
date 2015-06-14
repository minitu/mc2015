//HJM_Securities.cpp
//Routines to compute various security prices using HJM framework (via Simulation).
//Authors: Mark Broadie, Jatin Dewanwala
//Collaborator: Mikhail Smelyanskiy, Jike Chong, Intel

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <time.h>

#include "nr_routines.h"
#include "HJM.h"
#include "HJM_Securities.h"
#include "HJM_type.h"

#ifdef ENABLE_THREADS
#include <pthread.h>
#define MAX_THREAD 1024

#ifdef TBB_VERSION
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/cache_aligned_allocator.h"
tbb::cache_aligned_allocator<FTYPE> memory_ftype;
tbb::cache_aligned_allocator<parm> memory_parm;
#define TBB_GRAINSIZE 1
#endif // TBB_VERSION
#endif //ENABLE_THREADS

#if (defined(USE_CPU) || defined(USE_GPU)) || (defined(USE_MPI) || defined(USE_SNUCL))
#include <CL/opencl.h>
#ifdef _OPENMP
#include <omp.h>
#endif // OpenMP
#ifdef USE_MPI
#include "mpi.h"
#endif // MPI
#define MAX_SOURCE_SIZE 0x100000
#define KCNT 2
#define GLOBAL_WORK_SIZE 1024
using namespace std;
#endif // USE_CPU || USE_GPU || USE_MPI || USE_SNUCL

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int nThreads = 1;
int nSwaptions = 1;
int iN = 11; 
FTYPE dYears = 5.5; 
int iFactors = 3; 
parm *swaptions;

// =================================================
FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;
int chunksize;

int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);


#ifdef TBB_VERSION
struct Worker {
	Worker(){}
	void operator()(const tbb::blocked_range<int> &range) const {
		FTYPE pdSwaptionPrice[2];
		int begin = range.begin();
		int end   = range.end();

		for(int i=begin; i!=end; i++) {
			int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
					swaptions[i].dCompounding, swaptions[i].dMaturity, 
					swaptions[i].dTenor, swaptions[i].dPaymentInterval,
					swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
					swaptions[i].pdYield, swaptions[i].ppdFactors,
					100, NUM_TRIALS, BLOCK_SIZE, 0);
			assert(iSuccess == 1);
			swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
			swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];

		}



	}
};

#endif //TBB_VERSION


void * worker(void *arg){
	int tid = *((int *)arg);
	FTYPE pdSwaptionPrice[2];

	int chunksize = nSwaptions/nThreads;
	int beg = tid*chunksize;
	int end = (tid+1)*chunksize;
	if(tid == nThreads -1 )
		end = nSwaptions;

	for(int i=beg; i < end; i++) {
		int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
				swaptions[i].dCompounding, swaptions[i].dMaturity, 
				swaptions[i].dTenor, swaptions[i].dPaymentInterval,
				swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
				swaptions[i].pdYield, swaptions[i].ppdFactors,
				100, NUM_TRIALS, BLOCK_SIZE, 0);
		assert(iSuccess == 1);
		swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
		swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
	}

	return NULL;    
}




//Please note: Whenever we type-cast to (int), we add 0.5 to ensure that the value is rounded to the correct number. 
//For instance, if X/Y = 0.999 then (int) (X/Y) will equal 0 and not 1 (as (int) rounds down).
//Adding 0.5 ensures that this does not happen. Therefore we use (int) (X/Y + 0.5); instead of (int) (X/Y);

int main(int argc, char *argv[])
{
	int iSuccess = 0;
	int i,j;

	struct timespec start, end, spent;

	FTYPE **factors=NULL;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
	printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n"); 
	fflush(NULL);
#else
	printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_swaptions);
#endif

	if(argc == 1)
	{
		fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
		exit(1);
	}

	for (int j=1; j<argc; j++) {
		if (!strcmp("-sm", argv[j])) {NUM_TRIALS = atoi(argv[++j]);}
		else if (!strcmp("-nt", argv[j])) {nThreads = atoi(argv[++j]);} 
		else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);} 
		else {
			fprintf(stderr," usage: \n\t-ns [number of swaptions (should be > number of threads]\n\t-sm [number of simulations]\n\t-nt [number of threads]\n"); 
		}
	}

	if(nSwaptions < nThreads) {
		nSwaptions = nThreads; 
	}

	printf("Number of Simulations: %d,  Number of threads: %d Number of swaptions: %d\n", NUM_TRIALS, nThreads, nSwaptions);

#ifdef ENABLE_THREADS

#ifdef TBB_VERSION
	tbb::task_scheduler_init init(nThreads);
#else
	pthread_t      *threads;
	pthread_attr_t  pthread_custom_attr;

	if ((nThreads < 1) || (nThreads > MAX_THREAD))
	{
		fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
		exit(1);
	}
	threads = (pthread_t *) malloc(nThreads * sizeof(pthread_t));
	pthread_attr_init(&pthread_custom_attr);

#endif // TBB_VERSION

	if ((nThreads < 1) || (nThreads > MAX_THREAD))
	{
		fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
		exit(1);
	}

#else
	if (nThreads != 1)
	{
		fprintf(stderr,"Number of threads must be 1 (serial version)\n");
		exit(1);
	}
#endif //ENABLE_THREADS

	// initialize input dataset
	factors = dmatrix(0, iFactors-1, 0, iN-2);
	//the three rows store vol data for the three factors
	factors[0][0]= .01;
	factors[0][1]= .01;
	factors[0][2]= .01;
	factors[0][3]= .01;
	factors[0][4]= .01;
	factors[0][5]= .01;
	factors[0][6]= .01;
	factors[0][7]= .01;
	factors[0][8]= .01;
	factors[0][9]= .01;

	factors[1][0]= .009048;
	factors[1][1]= .008187;
	factors[1][2]= .007408;
	factors[1][3]= .006703;
	factors[1][4]= .006065;
	factors[1][5]= .005488;
	factors[1][6]= .004966;
	factors[1][7]= .004493;
	factors[1][8]= .004066;
	factors[1][9]= .003679;

	factors[2][0]= .001000;
	factors[2][1]= .000750;
	factors[2][2]= .000500;
	factors[2][3]= .000250;
	factors[2][4]= .000000;
	factors[2][5]= -.000250;
	factors[2][6]= -.000500;
	factors[2][7]= -.000750;
	factors[2][8]= -.001000;
	factors[2][9]= -.001250;

	// setting up multiple swaptions
	swaptions = 
#ifdef TBB_VERSION
		(parm *)memory_parm.allocate(sizeof(parm)*nSwaptions, NULL);
#else
	(parm *)malloc(sizeof(parm)*nSwaptions);
#endif

	int k;
	for (i = 0; i < nSwaptions; i++) {
		swaptions[i].Id = i;
		swaptions[i].iN = iN;
		swaptions[i].iFactors = iFactors;
		swaptions[i].dYears = dYears;

		swaptions[i].dStrike =  (double)i / (double)nSwaptions; 
		swaptions[i].dCompounding =  0;
		swaptions[i].dMaturity =  1;
		swaptions[i].dTenor =  2.0;
		swaptions[i].dPaymentInterval =  1.0;

		swaptions[i].pdYield = dvector(0,iN-1);;
		swaptions[i].pdYield[0] = .1;
		for(j=1;j<=swaptions[i].iN-1;++j)
			swaptions[i].pdYield[j] = swaptions[i].pdYield[j-1]+.005;

		swaptions[i].ppdFactors = dmatrix(0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
		for(k=0;k<=swaptions[i].iFactors-1;++k)
			for(j=0;j<=swaptions[i].iN-2;++j)
				swaptions[i].ppdFactors[k][j] = factors[k][j];
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

#if (defined(USE_CPU) || defined(USE_GPU)) || (defined(USE_MPI) || defined(USE_SNUCL))

	// ******************** OpenCL ********************

#ifdef USE_MPI
	// ***** MPI Setup *****
	int comm_rank, comm_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif // MPI

	// ***** Preparation *****

	int err;

	cl_device_id device_ids[100];
	cl_context context;
	cl_command_queue commands[100];
	cl_program programs[KCNT];
	cl_kernel kernels[KCNT];

	// Gather platform data
	cl_uint plat_cnt = 0;
	clGetPlatformIDs(0, 0, &plat_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(plat_cnt, platform_ids, NULL);

	size_t info_size;
	char *platform_info;
	const cl_platform_info attrTypes[4] = {
		CL_PLATFORM_PROFILE,
		CL_PLATFORM_VERSION,
		CL_PLATFORM_NAME,
		CL_PLATFORM_VENDOR };

#ifdef DEBUG
	printf("\n[ Platform Information ]\n\n");

	for (i = 0; i < 4; i++) {
		err = clGetPlatformInfo(platform_ids[0], attrTypes[i], 0, NULL, &info_size);
		if (err != CL_SUCCESS) {
			printf("Error: failed to get platform info size. %d\n", err);
			return EXIT_FAILURE;
		}
		platform_info = (char*) malloc(info_size);
		err = clGetPlatformInfo(platform_ids[0], attrTypes[i], info_size, platform_info, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: failed to get platform info. %d\n", err);
			return EXIT_FAILURE;
		}

		printf("%s\n", platform_info);

		free(platform_info);
	}

	printf("\n");
#endif

	// Connect to compute devices
	cl_uint dev_cnt;
#ifdef USE_CPU
	err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_CPU, 100, device_ids, &dev_cnt);
#elif USE_GPU
	err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 100, device_ids, &dev_cnt);
#endif // Compute devices

#ifdef DEBUG
	printf("[ Device Information ]\n\n");
	printf("# of devices: %u\n", dev_cnt);

	size_t max_work_group_size;
	cl_ulong local_mem_size;
	size_t max_param_size;

	err = clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	err |= clGetDeviceInfo(device_ids[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	err |= clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &max_param_size, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: failed to get device info. %d\n", err);
		return EXIT_FAILURE;
	}

	printf("Max work group size: %zu\n", max_work_group_size);
	printf("Local memory size: %lu\n\n", local_mem_size);
	printf("Max parameter size: %zu\n", max_param_size);
#endif

	// Create compute context
	context = clCreateContext(0, dev_cnt, device_ids, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error: failed to create compute context. %d\n", err);
		return EXIT_FAILURE;
	}

	// Create command queues (in-order)
	for (i = 0; i < dev_cnt; i++) {
		commands[i] = clCreateCommandQueue(context, device_ids[i], 0, &err);
		if (err != CL_SUCCESS) {
			printf("Error: failed to create command queue %d. %d\n", i, err);
			return EXIT_FAILURE;
		}
	}

	// Create programs
	FILE *fp;
	char *fileName[KCNT];
	char *src_str;
	size_t src_size;

	for (i = 0; i < KCNT; i++) {
		fileName[i] = (char*)malloc(sizeof(char) * 100);
	}

	// KERNEL
	strcpy(fileName[0], "./RanGen.cl");
	strcpy(fileName[1], "./sim.cl");
	//

	for (i = 0; i < KCNT; i++) {
		fp = fopen(fileName[i], "r");
		if (!fp) {
			printf("Error: file read failed.\n");
			return EXIT_FAILURE;
		}
		src_str = (char*)malloc(MAX_SOURCE_SIZE);
		src_size = fread(src_str, 1, MAX_SOURCE_SIZE, fp);

		programs[i] = clCreateProgramWithSource(context, 1, (const char **)&src_str, (const size_t *)&src_size, &err);

		if (err != CL_SUCCESS) {
			printf("Error: failed to create program %d. %d\n", i, err);
			return EXIT_FAILURE;
		}

		fclose(fp);
		free(src_str);
		free(fileName[i]);
	}

	// Build programs
	string build_str;
	char *build_options;

#ifdef DEBUG
	printf("[ Kernel Build Options ]\n\n");
#endif

	for (i = 0; i < KCNT; i++) {
		// Set build options (KERNEL)
		build_str = "-DFTYPE=double";
		build_options = const_cast<char*>(build_str.c_str());
#ifdef DEBUG
		printf("Kernel %d: %s\n", i, build_options);
#endif
		err = clBuildProgram(programs[i], 0, NULL, build_options, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: failed to build program %d. %d\n", i, err);
			size_t log_size;
			clGetProgramBuildInfo(programs[i], device_ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			char *log = (char*) malloc(sizeof(char) * log_size);
			clGetProgramBuildInfo(programs[i], device_ids[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			printf("%s\n", log);
			return EXIT_FAILURE;
		}
	}

#ifdef DEBUG
	printf("\n");
#endif

	// Create kernels
	string kernel_str;
	char *kernelName;

	for (i = 0; i < KCNT; i++) {
		// Set kernel name (KERNEL)
		if (i == 0) kernel_str = "swaption_RanGen";
		else if (i == 1) kernel_str = "swaption_sim";
		kernelName = const_cast<char*>(kernel_str.c_str());
		kernels[i] = clCreateKernel(programs[i], kernelName, &err);

		if (err != CL_SUCCESS) {
			printf("Error: failed to create kernel %d. %d\n", i, err);
			return EXIT_FAILURE;
		}
	}

	// Set work size
	size_t globalWorkSize = GLOBAL_WORK_SIZE;


	// ***** Pre-computations in HJM_Swaption_Blocking *****

	FTYPE dCompounding = swaptions[0].dCompounding;
	FTYPE dMaturity = swaptions[0].dMaturity;
	FTYPE dTenor = swaptions[0].dTenor;
	FTYPE dPaymentInterval = swaptions[0].dPaymentInterval;
	FTYPE ddelt = (FTYPE)(dYears/iN);
	FTYPE sqrt_ddelt = sqrt(ddelt);
	int iFreqRatio = (int)(dPaymentInterval/ddelt + 0.5);
	FTYPE *dStrikeCont = (FTYPE*)malloc(sizeof(FTYPE) * nSwaptions); // Differ
	int iSwapVectorLength = (int)(iN - dMaturity/ddelt + 0.5);
	int iSwapStartTimeIndex = (int)(dMaturity/ddelt + 0.5);
	int iSwapTimePoints = (int)(dTenor/ddelt + 0.5);
	FTYPE dSwapVectorYears = (FTYPE)(iSwapVectorLength*ddelt);

//#pragma omp parallel for
	for (i = 0; i < nSwaptions; i++) {
		if (swaptions[i].dCompounding == 0) {
			dStrikeCont[i] = swaptions[i].dStrike;
		} else {
			dStrikeCont[i] = (1/swaptions[i].dCompounding) * log(1+swaptions[i].dStrike*swaptions[i].dCompounding);
		}
	}

	FTYPE *pdSwapPayoffs = (FTYPE*) malloc(sizeof(FTYPE) * iSwapVectorLength * nSwaptions);
	FTYPE *pdForward = (FTYPE*) malloc(sizeof(FTYPE) * iN * nSwaptions);
	FTYPE *pdTotalDrift = (FTYPE*) malloc(sizeof(FTYPE) * (iN-1) * nSwaptions);
	FTYPE **ppdDrifts; // Temporary so can be kept a matrix
	int l;
	FTYPE dSumVol;

	for (i = 0; i < nSwaptions; i++) {

		// Store swap payoffs
		for (j = 0; j <= iSwapVectorLength-1; j++)
			pdSwapPayoffs[iSwapVectorLength * i + j] = 0.0;
		for (j = iFreqRatio; j <= iSwapTimePoints; j+=iFreqRatio) {
			if (j != iSwapTimePoints)
				pdSwapPayoffs[iSwapVectorLength * i + j] = exp(dStrikeCont[i]*dPaymentInterval) - 1;
			if (j == iSwapTimePoints)
				pdSwapPayoffs[iSwapVectorLength * i + j] = exp(dStrikeCont[i]*dPaymentInterval);
		}

		// Generate forward curve
		pdForward[iN * i] = swaptions[i].pdYield[0];
		for (j = 1; j <= iN-1; j++) {
			pdForward[iN * i + j] = (j+1)*swaptions[i].pdYield[j] - j*swaptions[i].pdYield[j-1];
		}

		// Compute drifts
		ppdDrifts = dmatrix(0, iFactors-1, 0, iN-2);

		for (j = 0; j <= iFactors-1; j++)
			ppdDrifts[j][0] = 0.5*ddelt*(swaptions[i].ppdFactors[j][0])*(swaptions[i].ppdFactors[j][0]);

		for (j = 0; j <= iFactors-1; j++)
			for (k = 1; k <= iN-2; k++) {
				ppdDrifts[j][k] = 0;
				for (l = 0; l <= k-1; l++)
					ppdDrifts[j][k] -= ppdDrifts[j][l];
				dSumVol = 0;
				for (l = 0; l <= k; l++)
					dSumVol += swaptions[i].ppdFactors[j][l];
				ppdDrifts[j][k] += 0.5*ddelt*(dSumVol)*(dSumVol);
			}

		for (j = 0; j <= iN-2; j++) {
			pdTotalDrift[(iN-1) * i + j] = 0;
			for (k = 0; k <= iFactors-1; k++)
				pdTotalDrift[(iN-1) * i + j] += ppdDrifts[k][j];
		}

		free_dmatrix(ppdDrifts, 0, iFactors-1, 0, iN-2);
	}

	// ***** Calculate some constants *****

	// Calculate # of swaptions per device
	int *swp_dev = (int*) malloc(sizeof(int) * dev_cnt);
	int leftover = nSwaptions % dev_cnt;
	int tmp_cnt = floor((double)nSwaptions / (double)dev_cnt);

	for (i = 0; i < dev_cnt; i++) {
		if (i < leftover)
			swp_dev[i] = tmp_cnt + 1;
		else
			swp_dev[i] = tmp_cnt;
	}

	// Calculate # of simulation iterations per work item
	unsigned int *iter_wi = (unsigned int*) malloc(sizeof(unsigned int) * GLOBAL_WORK_SIZE);
	unsigned int iter_tot = ceil((double)NUM_TRIALS / (double)BLOCK_SIZE);
	leftover = iter_tot % GLOBAL_WORK_SIZE;
	tmp_cnt = floor((double)iter_tot / (double)GLOBAL_WORK_SIZE);

//#pragma omp parallel for
	for (i = 0; i < GLOBAL_WORK_SIZE; i++) {
		if (i < leftover)
			iter_wi[i] = tmp_cnt + 1;
		else
			iter_wi[i] = tmp_cnt;
	}

	// Calculate simulation iteration indices per work item
	unsigned int *iter_wi_sti = (unsigned int*) malloc(sizeof(unsigned int) * GLOBAL_WORK_SIZE);
	unsigned int *iter_wi_edi = (unsigned int*) malloc(sizeof(unsigned int) * GLOBAL_WORK_SIZE);

	unsigned int stIndex1 = 0;
	for (i = 0; i < GLOBAL_WORK_SIZE; i++) {
		iter_wi_sti[i] = stIndex1;
		stIndex1 += iter_wi[i];
		iter_wi_edi[i] = stIndex1 - 1;
	}

	// For generating random numbers
	// (same across all swaptions)
	long lRndSeed = 100;
	unsigned int ranCnt = iFactors * iN * BLOCK_SIZE * iter_tot;
	FTYPE *pdZ = (FTYPE*) malloc(sizeof(FTYPE) * ranCnt);

	// Calculate # of random numbers per device
	unsigned int *ran_dev = (unsigned int*) malloc(sizeof(unsigned int) * dev_cnt);
	leftover = ranCnt % dev_cnt;
	tmp_cnt = floor((double)ranCnt / (double)dev_cnt);

	for (i = 0; i < dev_cnt; i++) {
		if (i < leftover)
			ran_dev[i] = tmp_cnt + 1;
		else
			ran_dev[i] = tmp_cnt;
	}

	// Calculate # of random numbers per work-item
	unsigned int *ran_wi = (unsigned int*) malloc(sizeof(unsigned int) * GLOBAL_WORK_SIZE * dev_cnt);
	for (i = 0; i < dev_cnt; i++) {
		leftover = ran_dev[i] % GLOBAL_WORK_SIZE;
		tmp_cnt = floor((double)ran_dev[i] / (double)GLOBAL_WORK_SIZE);

//#pragma omp parallel for
		for (j = 0; j < GLOBAL_WORK_SIZE; j++) {
			if (j < leftover)
				ran_wi[GLOBAL_WORK_SIZE * i + j] = tmp_cnt + 1;
			else
				ran_wi[GLOBAL_WORK_SIZE * i + j] = tmp_cnt;
		}
	}

	// Calculate start & end index of pdZ per work-item
	unsigned int *ran_wi_sti = (unsigned int*) malloc(sizeof(unsigned int) * dev_cnt * GLOBAL_WORK_SIZE);
	unsigned int *ran_wi_edi = (unsigned int*) malloc(sizeof(unsigned int) * dev_cnt * GLOBAL_WORK_SIZE);

	stIndex1 = 0;
	unsigned int stIndex2;
	for (i = 0; i < dev_cnt; i++) {
		stIndex2 = stIndex1;
		for (j = 0; j < GLOBAL_WORK_SIZE; j++) {
			ran_wi_sti[GLOBAL_WORK_SIZE * i + j] = stIndex2;
			stIndex2 += ran_wi[GLOBAL_WORK_SIZE * i + j];
			ran_wi_edi[GLOBAL_WORK_SIZE * i + j] = stIndex2 - 1;
		}
		stIndex1 += ran_dev[i];
	}

	// ***** RanUnif & CumNormalInv *****

	// Prepare memory
	cl_mem cl_pdZ[dev_cnt];
	cl_mem cl_sti[dev_cnt];
	cl_mem cl_edi[dev_cnt];

	for (i = 0; i < dev_cnt; i++) {
		cl_pdZ[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * ranCnt, NULL, &err);
#ifdef USE_CPU
		cl_sti[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * dev_cnt * GLOBAL_WORK_SIZE, ran_wi_sti, &err);
		cl_edi[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * dev_cnt * GLOBAL_WORK_SIZE, ran_wi_edi, &err);
#elif USE_GPU
		cl_sti[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * dev_cnt * GLOBAL_WORK_SIZE, ran_wi_sti, &err);
		cl_edi[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * dev_cnt * GLOBAL_WORK_SIZE, ran_wi_edi, &err);
#endif
	}

	if (err != CL_SUCCESS) {
		printf("Error: failed to allocate device memory. %d\n", err);
		return EXIT_FAILURE;
	}

	// Random number generation with all OpenCL devices
	unsigned int buf_ofs = 0; // Offset in buffer
	unsigned int host_ofs = 0; // Offset in host memory
	for (i = 0; i < dev_cnt; i++) {
		// Set kernel arguments
		err = clSetKernelArg(kernels[0], 0, sizeof(int), (void*) &globalWorkSize);
		err |= clSetKernelArg(kernels[0], 1, sizeof(int), (void*) &i);
		err |= clSetKernelArg(kernels[0], 2, sizeof(long), (void*) &lRndSeed);
		err |= clSetKernelArg(kernels[0], 3, sizeof(cl_mem), (void*) &cl_pdZ[i]);
		err |= clSetKernelArg(kernels[0], 4, sizeof(cl_mem), (void*) &cl_sti[i]);
		err |= clSetKernelArg(kernels[0], 5, sizeof(cl_mem), (void*) &cl_edi[i]);

		if (err != CL_SUCCESS) {
			printf("Error: failed to set kernel arguments. %d\n", err);
			return EXIT_FAILURE;
		}

		// Enqueue kernel
		err = clEnqueueNDRangeKernel(commands[i], kernels[0], 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: failed to enqueue kernel. %d\n", err);
			return EXIT_FAILURE;
		}

		// Read pdZ back to host memory
		err = clEnqueueReadBuffer(commands[i], cl_pdZ[i], CL_FALSE, (size_t) buf_ofs, (size_t) (ran_dev[i] * sizeof(FTYPE)), pdZ + host_ofs, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error: failed to read buffer. %d\n", err);
			return EXIT_FAILURE;
		}

		buf_ofs += ran_dev[i] * sizeof(FTYPE);
		host_ofs += ran_dev[i];
	}

	// Ensure kernel completion
	for (i = 0; i < dev_cnt; i++) {
		clFinish(commands[i]);
	}

	// ***** Simulation *****

	// Convert ppdFactors into vectors
	// (use swaption 0's, because they are same across all swaptions)
	FTYPE *gppdFactors = (FTYPE*) malloc(sizeof(FTYPE) * iFactors * (iN-1));

	for (i = 0; i < iFactors; i++) {
		for (j = 0; j < iN-1; j++) {
			gppdFactors[(iN-1) * i + j] = swaptions[0].ppdFactors[i][j];
		}
	}

	// Device memory objects
	cl_mem cl_ppdHJMPath[nSwaptions];
	cl_mem cl_pdDiscountingRatePath[nSwaptions];
	cl_mem cl_pdPayoffDiscountFactors[nSwaptions];
	cl_mem cl_pdexpRes[nSwaptions];
	cl_mem cl_pdSwapRatePath[nSwaptions];
	cl_mem cl_pdSwapDiscountFactors[nSwaptions];

	cl_mem cl_pdForward[nSwaptions];
	cl_mem cl_pdTotalDrift[nSwaptions];
	cl_mem cl_ppdFactors;
	cl_mem cl_gpdZ;
	cl_mem cl_pdSwapPayoffs[nSwaptions];
	cl_mem cl_dSumSimSwaptionPrice[nSwaptions];
	cl_mem cl_dSumSquareSimSwaptionPrice[nSwaptions];

	cl_mem cl_iter_wi_sti;
	cl_mem cl_iter_wi_edi;

	FTYPE *acc_dSumSimSwaptionPrice = (FTYPE*) malloc(sizeof(FTYPE) * GLOBAL_WORK_SIZE * nSwaptions);
	FTYPE *acc_dSumSquareSimSwaptionPrice = (FTYPE*) malloc(sizeof(FTYPE) * GLOBAL_WORK_SIZE * nSwaptions);

	// Create buffers (common across all swaptions)
#ifdef USE_CPU
	cl_ppdFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * iFactors * (iN-1), gppdFactors, &err);
	cl_gpdZ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * ranCnt, pdZ, &err);
	cl_iter_wi_sti = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * GLOBAL_WORK_SIZE, iter_wi_sti, &err);
	cl_iter_wi_edi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int) * GLOBAL_WORK_SIZE, iter_wi_edi, &err);
#elif USE_GPU
	cl_ppdFactors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * iFactors * (iN-1), gppdFactors, &err);
	cl_gpdZ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * ranCnt, pdZ, &err);
	cl_iter_wi_sti = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * GLOBAL_WORK_SIZE, iter_wi_sti, &err);
	cl_iter_wi_edi = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * GLOBAL_WORK_SIZE, iter_wi_edi, &err);
#endif

	if (err != CL_SUCCESS) {
		printf("Error: failed to create buffer. %d\n", err);
		return EXIT_FAILURE;
	}
	
	// Enqueue kernels for each swaption
	int blk_size = BLOCK_SIZE;
	int swp_cnt;
	int cur_swp = 0;

	for (i = 0; i < dev_cnt; i++) {
		for (swp_cnt = 0; swp_cnt < swp_dev[i]; swp_cnt++) {

			// Create buffers (unique per swaption)
			cl_ppdHJMPath[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * iN * iN * BLOCK_SIZE * GLOBAL_WORK_SIZE, NULL, &err);
			cl_pdDiscountingRatePath[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * iN * BLOCK_SIZE * GLOBAL_WORK_SIZE, NULL, &err);
			cl_pdPayoffDiscountFactors[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * iN * BLOCK_SIZE * GLOBAL_WORK_SIZE, NULL, &err);
			cl_pdexpRes[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * (iN-1) * BLOCK_SIZE * GLOBAL_WORK_SIZE, NULL, &err);
			cl_pdSwapRatePath[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * iSwapVectorLength * BLOCK_SIZE * GLOBAL_WORK_SIZE, NULL, &err);
			cl_pdSwapDiscountFactors[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(FTYPE) * iSwapVectorLength * BLOCK_SIZE * GLOBAL_WORK_SIZE, NULL, &err);

#ifdef USE_CPU
			cl_pdForward[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * iN, pdForward + iN * cur_swp, &err);
			cl_pdTotalDrift[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * (iN-1), pdTotalDrift + (iN-1) * cur_swp, &err);
			cl_pdSwapPayoffs[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * iSwapVectorLength, pdSwapPayoffs + iSwapVectorLength * cur_swp, &err);
			cl_dSumSimSwaptionPrice[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * GLOBAL_WORK_SIZE, acc_dSumSimSwaptionPrice + GLOBAL_WORK_SIZE * cur_swp, &err);
			cl_dSumSquareSimSwaptionPrice[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(FTYPE) * GLOBAL_WORK_SIZE, acc_dSumSquareSimSwaptionPrice + GLOBAL_WORK_SIZE * cur_swp, &err);
#elif USE_GPU
			cl_pdForward[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * iN, pdForward + iN * cur_swp, &err);
			cl_pdTotalDrift[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * (iN-1), pdTotalDrift + (iN-1) * cur_swp, &err);
			cl_pdSwapPayoffs[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * iSwapVectorLength, pdSwapPayoffs + iSwapVectorLength * cur_swp, &err);
			cl_dSumSimSwaptionPrice[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * GLOBAL_WORK_SIZE, acc_dSumSimSwaptionPrice + GLOBAL_WORK_SIZE * cur_swp, &err);
			cl_dSumSquareSimSwaptionPrice[cur_swp] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(FTYPE) * GLOBAL_WORK_SIZE, acc_dSumSquareSimSwaptionPrice + GLOBAL_WORK_SIZE * cur_swp, &err);
#endif

			if (err != CL_SUCCESS) {
				printf("Error: failed to create buffer. %d\n", err);
				return EXIT_FAILURE;
			}

			// Set kernel arguments (unique per swaption)
			err = clSetKernelArg(kernels[1], 0, sizeof(cl_mem), (void*) &cl_ppdHJMPath[cur_swp]);
			err |= clSetKernelArg(kernels[1], 1, sizeof(int), (void*) &iN);
			err |= clSetKernelArg(kernels[1], 2, sizeof(int), (void*) &iFactors);
			err |= clSetKernelArg(kernels[1], 3, sizeof(FTYPE), (void*) &dYears);
			err |= clSetKernelArg(kernels[1], 4, sizeof(int), (void*) &blk_size);
			err |= clSetKernelArg(kernels[1], 5, sizeof(FTYPE), (void*) &ddelt);
			err |= clSetKernelArg(kernels[1], 6, sizeof(FTYPE), (void*) &sqrt_ddelt);
			err |= clSetKernelArg(kernels[1], 7, sizeof(int), (void*) &iSwapVectorLength);
			err |= clSetKernelArg(kernels[1], 8, sizeof(int), (void*) &iSwapStartTimeIndex);
			err |= clSetKernelArg(kernels[1], 9, sizeof(FTYPE), (void*) &dSwapVectorYears);
			err |= clSetKernelArg(kernels[1], 10, sizeof(cl_mem), (void*) &cl_pdForward[cur_swp]);
			err |= clSetKernelArg(kernels[1], 11, sizeof(cl_mem), (void*) &cl_pdTotalDrift[cur_swp]);
			err |= clSetKernelArg(kernels[1], 12, sizeof(cl_mem), (void*) &cl_ppdFactors);
			err |= clSetKernelArg(kernels[1], 13, sizeof(cl_mem), (void*) &cl_gpdZ);
			err |= clSetKernelArg(kernels[1], 14, sizeof(cl_mem), (void*) &cl_pdDiscountingRatePath[cur_swp]);
			err |= clSetKernelArg(kernels[1], 15, sizeof(cl_mem), (void*) &cl_pdPayoffDiscountFactors[cur_swp]);
			err |= clSetKernelArg(kernels[1], 16, sizeof(cl_mem), (void*) &cl_pdexpRes[cur_swp]);
			err |= clSetKernelArg(kernels[1], 17, sizeof(cl_mem), (void*) &cl_pdSwapRatePath[cur_swp]);
			err |= clSetKernelArg(kernels[1], 18, sizeof(cl_mem), (void*) &cl_pdSwapDiscountFactors[cur_swp]);
			err |= clSetKernelArg(kernels[1], 19, sizeof(cl_mem), (void*) &cl_pdSwapPayoffs[cur_swp]);
			err |= clSetKernelArg(kernels[1], 20, sizeof(cl_mem), (void*) &cl_dSumSimSwaptionPrice[cur_swp]);
			err |= clSetKernelArg(kernels[1], 21, sizeof(cl_mem), (void*) &cl_dSumSquareSimSwaptionPrice[cur_swp]);
			err |= clSetKernelArg(kernels[1], 22, sizeof(cl_mem), (void*) &cl_iter_wi_sti);
			err |= clSetKernelArg(kernels[1], 23, sizeof(cl_mem), (void*) &cl_iter_wi_edi);

			if (err != CL_SUCCESS) {
				printf("Error: failed to set kernel arguments. %d\n", err);
				return EXIT_FAILURE;
			}

			// Enqueue kernel
			err = clEnqueueNDRangeKernel(commands[i], kernels[1], 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

			if (err != CL_SUCCESS) {
				printf("Error: failed to enqueue kernel. %d\n", err);
				return EXIT_FAILURE;
			}

			// Read prices back to host memory
			err = clEnqueueReadBuffer(commands[i], cl_dSumSimSwaptionPrice[cur_swp], CL_FALSE, 0, (size_t) (sizeof(FTYPE) * GLOBAL_WORK_SIZE), acc_dSumSimSwaptionPrice + GLOBAL_WORK_SIZE * cur_swp, 0, NULL, NULL);
			err |= clEnqueueReadBuffer(commands[i], cl_dSumSquareSimSwaptionPrice[cur_swp], CL_FALSE, 0, (size_t) (sizeof(FTYPE) * GLOBAL_WORK_SIZE), acc_dSumSquareSimSwaptionPrice + GLOBAL_WORK_SIZE * cur_swp, 0, NULL, NULL);

			if (err != CL_SUCCESS) {
				printf("Error: failed to read buffer. %d\n", err);
				return EXIT_FAILURE;
			}

			// Next swaption
			cur_swp++;
		}
	}

	// Ensure kernel completion
	for (i = 0; i < dev_cnt; i++) {
		clFinish(commands[i]);
	}
	
	// Reduce prices TODO: too heavy?
	FTYPE fin_dSumSimSwaptionPrice[nSwaptions];
	FTYPE fin_dSumSquareSimSwaptionPrice[nSwaptions];

//#pragma omp parallel for private(j)
	for (i = 0; i < nSwaptions; i++) {
		fin_dSumSimSwaptionPrice[i] = 0.0;
		fin_dSumSquareSimSwaptionPrice[i] = 0.0;
		for (j = 0; j < GLOBAL_WORK_SIZE; j++) {
			fin_dSumSimSwaptionPrice[i] += acc_dSumSimSwaptionPrice[GLOBAL_WORK_SIZE * i + j];
			fin_dSumSquareSimSwaptionPrice[i] += acc_dSumSquareSimSwaptionPrice[GLOBAL_WORK_SIZE * i + j];
		}
		swaptions[i].dSimSwaptionMeanPrice = fin_dSumSimSwaptionPrice[i] / NUM_TRIALS;
		swaptions[i].dSimSwaptionStdError = sqrt((fin_dSumSquareSimSwaptionPrice[i]-fin_dSumSimSwaptionPrice[i]*fin_dSumSimSwaptionPrice[i]/NUM_TRIALS)/(NUM_TRIALS-1.0))/sqrt((FTYPE)NUM_TRIALS);
	}

	// ***** Free *****
	free(pdForward);
	free(pdTotalDrift);
	free(pdSwapPayoffs);
	free(gppdFactors);
	free(dStrikeCont);
	free(swp_dev);
	free(iter_wi);
	free(iter_wi_sti);
	free(iter_wi_edi);
	free(pdZ);
	free(ran_dev);
	free(ran_wi);
	free(ran_wi_sti);
	free(ran_wi_edi);

#endif // OpenCL

	// **********Calling the Swaption Pricing Routine*****************
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS

#ifdef TBB_VERSION
	Worker w;
	tbb::parallel_for(tbb::blocked_range<int>(0,nSwaptions,TBB_GRAINSIZE),w);
#else

	int threadIDs[nThreads];
	for (i = 0; i < nThreads; i++) {
		threadIDs[i] = i;
		pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
	}
	for (i = 0; i < nThreads; i++) {
		pthread_join(threads[i], NULL);
	}

	free(threads);

#endif // TBB_VERSION	

#elif USE_CPU

#elif USE_GPU

#else
	int threadID=0;
	worker(&threadID);

#endif //ENABLE_THREADS

	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

	for (i = 0; i < nSwaptions; i++) {
		fprintf(stderr,"Swaption%d: [SwaptionPrice: %.10lf StdError: %.10lf] \n", 
				i, swaptions[i].dSimSwaptionMeanPrice, swaptions[i].dSimSwaptionStdError);

	}

	for (i = 0; i < nSwaptions; i++) {
		free_dvector(swaptions[i].pdYield, 0, swaptions[i].iN-1);
		free_dmatrix(swaptions[i].ppdFactors, 0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
	}


#ifdef TBB_VERSION
	memory_parm.deallocate(swaptions, sizeof(parm));
#else
	free(swaptions);
#endif // TBB_VERSION

	//***********************************************************

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	return iSuccess;
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
	 * tv_nsec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_nsec = x->tv_nsec - y->tv_nsec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}
