__kernel void swaption_sim(
		__global FTYPE *ppdHJMPath,
		int iN,
		int iFactors,
		FTYPE dYears,
		int BLOCKSIZE,
		FTYPE ddelt,
		FTYPE sqrt_ddelt,
		int iSwapVectorLength,
		int iSwapStartTimeIndex,
		FTYPE dSwapVectorYears,
		__global FTYPE *pdForward,
		__global FTYPE *pdTotalDrift,
		__global FTYPE *ppdFactors,
		__global FTYPE *pdZ,
		__global FTYPE *pdDiscountingRatePath,
		__global FTYPE *pdPayoffDiscountFactors,
		__global FTYPE *pdexpRes,
		__global FTYPE *pdSwapRatePath,
		__global FTYPE *pdSwapDiscountFactors,
		__global FTYPE *pdSwapPayoffs)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

}
