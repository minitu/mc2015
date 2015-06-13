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
		__global FTYPE *pdForward,
		__global FTYPE *pdTotalDrift,
		__global FTYPE *ppdFactors,
		__global FTYPE *g_pdZ,
		__global FTYPE *g_pdDiscountingRatePath,
		__global FTYPE *g_pdPayoffDiscountFactors,
		__global FTYPE *g_pdexpRes,
		__global FTYPE *g_pdSwapRatePath,
		__global FTYPE *g_pdSwapDiscountFactors,
		__global FTYPE *pdSwapPayoffs,
		__global FTYPE *g_dSumSimSwaptionPrice,
		__global FTYPE *g_dSumSquareSimSwaptionPrice,
		__global unsigned int *iter_wi_sti,
		__global unsigned int *iter_wi_edi)
{
	const int global_id = get_global_id(0);
	const int local_id = get_local_id(0);
	const int group_id = get_group_id(0);

	// Calculate device global memory address allocated to this work item
	__global FTYPE *ppdHJMPath = g_ppdHJMPath + iN * iN * BLOCKSIZE * global_id;
	__global FTYPE *pdZ;
	__global FTYPE *pdDiscountingRatePath = g_pdDiscountingRatePath + iN * BLOCKSIZE * global_id;
	__global FTYPE *pdPayoffDiscountFactors = g_pdPayoffDiscountFactors + iN * BLOCKSIZE * global_id;
	__global FTYPE *pdexpRes = g_pdexpRes + (iN-1) * BLOCKSIZE * global_id;
	__global FTYPE *pdSwapRatePath = g_pdSwapRatePath + iSwapVectorLength * BLOCKSIZE * global_id;
	__global FTYPE *pdSwapDiscountFactors = g_pdSwapDiscountFactors + iSwapVectorLength * BLOCKSIZE * global_id;

	// Copy into local memory

	// Simulation loops
	int my_iter_index = 0;
	int i, j, k, l, ii;

	FTYPE dTotalShock;
	int b;
	int pdZi;

	int n_iN = iSwapVectorLength;
	FTYPE n_ddelt = (FTYPE) ((FTYPE)dSwapVectorYears / n_iN);

	FTYPE dSwaptionPayoff;
	FTYPE dDiscSwaptionPayoff;
	FTYPE dFixedLegValue;

	FTYPE dSumSimSwaptionPrice = 0.0;
	FTYPE dSumSquareSimSwaptionPrice = 0.0;

	for (ii = iter_wi_sti; ii <= iter_wi_edi; ii++) {
		pdZ = g_pdZ + iFactors * iN * BLOCKSIZE * ii;
		pdZi = 0;
		
		// Rest of HJM_SimPath_Forward_Blocking
		for (b = 0; b < BLOCKSIZE; b++) {
			for (j = 0; j <= iN-1; j++) {
				ppdHJMPath[BLOCKSIZE * j + b] = pdForward[j];

				for (i = 1; i <= iN-1; i++) {
					ppdHJMPath[iN * BLOCKSIZE * i + BLOCKSIZE * j + b] = 0;
				}
			}
		}

		for (b = 0; b < BLOCKSIZE; b++) {
			for (j = 1; j <= iN-1; j++) {
				for (l = 0; l <= iN-(j+1); l++) {
					dTotalShock = 0;

					for (i = 0; i <= iFactors-1; i++) {
						dTotalShock += ppdFactors[(iN-1) * i + l] * pdZ[pdZi++];
					}

					ppdHJMPath[iN * BLOCKSIZE * j + BLOCKSIZE * l + b] = ppdHJMPath[iN * BLOCKSIZE * (j-1) + BLOCKSIZE * (l+1) + b] + pdTotalDrift[l] * ddelt + sqrt_ddelt * dTotalShock;
				}
			}
		}

		// Compute discount factor vector
		for (i = 0; i <= iN-1; i++) {
			for (b = 0; b <= BLOCKSIZE-1; b++) {
				pdDiscountingRatePath[BLOCKSIZE*i + b] = ppdHJMPath[iN * BLOCKSIZE * i + b];
			}
		}

		// Discount_Factors_Blocking
		for (j = 0; j <= (iN-1)*BLOCKSIZE-1; j++) pdexpRes[j] = -pdDiscountingRatePath[j]*ddelt;
		for (j = 0; j <= (iN-1)*BLOCKSIZE-1; j++) pdexpRes[j] = exp(pdexpRes[j]);

		for (i = 0; i <= iN*BLOCKSIZE; i++)
			pdPayoffDiscountFactors[i] = 1.0;

		for (i = 1; i <= iN-1; i++) {
			for (b = 0; b < BLOCKSIZE; b++) {
				for (j = 0; j <= i-1; j++) {
					pdPayoffDiscountFactors[i*BLOCKSIZE + b] *= pdexpRes[j*BLOCKSIZE + b];
				}
			}
		}

		// Compute discount factors along the swap path
		for (i = 0; i <= iSwapVectorLength-1; i++) {
			for (b = 0; b < BLOCKSIZE; b++) {
				pdSwapRatePath[i * BLOCKSIZE + b] = ppdHJMPath[iN * BLOCKSIZE * iSwapStartTimeIndex + i * BLOCKSIZE + b];
			}
		}

		// Discount_Factors_Blocking
		for (j = 0; j <= (n_iN-1) * BLOCKSIZE-1; j++) pdexpRes[j] = -pdSwapRatePath[j]*n_ddelt;
		for (j = 0; j <= (n_iN-1) * BLOCKSIZE-1; j++) pdexpRes[j] = exp(pdexpRes[j]);

		for (i = 0; i < n_iN*BLOCKSIZE; i++)
			pdSwapDiscountFactors[i] = 1.0;

		for (i = 1; i <= n_iN-1; i++) {
			for (b = 0; b < BLOCKSIZE; b++) {
				for (j = 0; j <= i-1; j++) {
					pdSwapDiscountFactors[i*BLOCKSIZE + b] *= pdexpRes[j*BLOCKSIZE + b];
				}
			}
		}

		// Simulation
		for (b = 0; b < BLOCKSIZE; b++) {
			dFixedLegValue = 0.0;
			for (i = 0; i <= iSwapVectorLength-1; i++) {
				dFixedLegValue += pdSwapPayoffs[i]*pdSwapDiscountFactors[i*BLOCKSIZE + b];
			}
			dSwaptionPayoff = ((dFixedLegValue - 1.0) > 0) ? (dFixedLegValue - 1.0) : 0;
			dDiscSwaptionPayoff = dSwaptionPayoff*pdPayoffDiscountFactors[iSwapStartTimeIndex*BLOCKSIZE + b];

			// Accumulate
			dSumSimSwaptionPrice += dDiscSwaptionPayoff;
			dSumSquareSimSwaptionPrice += dDiscSwaptionPayoff*dDiscSwaptionPayoff;
		}
	}

	// Store partial simulation results in global memory
	g_dSumSimSwaptionPrice[global_id] = dSumSimSwaptionPrice;
	g_dSumSquareSimSwaptionPrice[global_id] = dSumSquareSimSwaptionPrice;
}
