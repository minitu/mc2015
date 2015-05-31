__kernel void swaption_sim1(
		__global FTYPE *ppdHJMPath,
		int iN,
		int iFactors,
		FTYPE dYears,
		__global FTYPE *pdForward,
		__global FTYPE *pdTotalDrift,
		__global FTYPE *ppdFactors,
		__global long *iRndSeed,
		int BLOCKSIZE,
		FTYPE ddelt,
		FTYPE sqrt_ddelt)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

	int i, j, l;

	FTYPE pdz[MATRIX_SIZE];
	FTYPE randz[MATRIX_SIZE];

}
