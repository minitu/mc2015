__kernel void swaption_init(
		__global FTYPE *pdSwaptionPrice,
		FTYPE dStrike,
		FTYPE dCompounding,
		FTYPE dMaturity,
		FTYPE dTenor,
		FTYPE dPaymentInterval,
		int iN,
		int iFactors,
		FTYPE dYears,
		__global FTYPE *pdYield,
		__global FTYPE *ppdFactors,
		long iRndSeed,
		long lTrials,
		int BLOCKSIZE,
		__global FTYPE *xddelt,
		__global int *xiFreqRatio,
		__global FTYPE *xdStrikeCont)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

	// Only 1 work item executes
	if (global_id == 0) {

		int i, b;
		long l;

		*xddelt = (FTYPE) (dYears/iN);
		*xiFreqRatio = (int) (dPaymentInterval/(*xddelt) + 0.5);

		FTYPE p_dStrike = dStrike;
		FTYPE p_dCompounding = dCompounding;

		if (p_dCompounding == 0) {
			*xdStrikeCont = p_dStrike;
		}
		else {
			*xdStrikeCont = (1/p_dCompounding) * log(1+p_dStrike*p_dCompounding);
		}

	}
}
