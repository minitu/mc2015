__kernel void swaption_sim(
		__global FTYPE *g_ppdHJMPath,
		int iN,
		int iFactors,
		FTYPE dYears,
		int BLOCKSIZE,
		FTYPE ddelt,
		FTYPE sqrt_ddelt,
		int iSwapVectorLength,
		int iSwapStartTimeIndex,
		FTYPE dSwapVectorYears,
		__global FTYPE *g_pdForward,
		__global FTYPE *g_pdTotalDrift,
		__global FTYPE *g_ppdFactors,
		__global FTYPE *pdZ,
		__global FTYPE *g_pdDiscountingRatePath,
		__global FTYPE *g_pdPayoffDiscountFactors,
		__global FTYPE *g_pdexpRes,
		__global FTYPE *g_pdSwapRatePath,
		__global FTYPE *g_pdSwapDiscountFactors,
		__global FTYPE *g_pdSwapPayoffs,
		__global unsigned int *iter_wi)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

	// Calculate device global memory address allocated to this work item
	__global FTYPE *ppdHJMPath = g_ppdHJMPath + iN * iN * BLOCKSIZE * global_id;
	__global FTYPE *pdDiscountingRatePath = g_pdDiscountingRatePath + iN * BLOCKSIZE * global_id;
	__global FTYPE *pdPayoffDiscountFactors = g_pdPayoffDiscountFactors + iN * BLOCKSIZE * global_id;
	__global FTYPE *pdexpRes = g_pdexpRes + (iN-1) * BLOCKSIZE * global_id;
	__global FTYPE *pdSwapRatePath = g_pdSwapRatePath + iSwapVectorLength * BLOCKSIZE * global_id;
	__global FTYPE *pdSwapDiscountFactors = g_pdSwapDiscountFactors + iSwapVectorLength * BLOCKSIZE * global_id;

	// Copy into local memory

	// Simulation loops
	int my_iter = iter_wi[global_id];
	int i, j, k, l;

	for (i = 0; i < my_iter; i++) {
		
		// HJM_SimPath_Forward_Blocking
		FTYPE dTotalShock;
		int b;

		for (b = 0; b < BLOCKSIZE; b++) {
			for (j = 0; j <= iN-1; j++) {
				ppdHJMPath[BLOCKSIZE * j + b] = g_pdForward[j];

				for (i = 1; i <= iN-1; i++) {
					ppdHJMPath[iN * BLOCKSIZE * i + BLOCKSIZE * j + b] = 0;
				}
			}
		}
	}
}
