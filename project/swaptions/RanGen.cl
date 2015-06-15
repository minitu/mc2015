__kernel void swaption_RanGen(
		int globalWorkSize,
		int dev_i,
		long lRndSeed,
		__global FTYPE *pdZ,
		__global unsigned int *ran_wi_sti,
		__global unsigned int *ran_wi_edi)
{
	const int global_id = get_global_id(0);

	FTYPE a[4] = {
		2.50662823884,
		-18.61500062529,
		41.39119773534,
		-25.44106049637
	};

	FTYPE b[4] = {
		-8.47351093090,
		23.08336743743,
		-21.06224101826,
		3.13082909833
	};

	FTYPE c[9] = {
		0.3374754822726147,
		0.9761690190917186,
		0.1607979714918209,
		0.0276438810333863,
		0.0038405729373609,
		0.0003951896511919,
		0.0000321767881768,
		0.0000002888167364,
		0.0000003960315187
	};

	unsigned int i;
	long s = lRndSeed;
	long ix, k1;
	FTYPE x, r, u;
	
	// Get start & end indices of pdZ
	unsigned int stIndex = ran_wi_sti[globalWorkSize * dev_i + global_id];
	unsigned int edIndex = ran_wi_edi[globalWorkSize * dev_i + global_id];

	for (i = stIndex; i <= edIndex; i++) {

		// RanUnif
		ix = s + (long)i;
		ix *= 1513517L;
		ix %= 2147483647L;
		k1 = ix/127773L;
		ix = 16807L*( ix - k1*127773L ) - k1 * 2836L;
		if (ix < 0) ix = ix + 2147483647L;
		u = (ix * 4.656612875e-10);

		// CumNormalInv
		x = u - 0.5;
		if (fabs(x) < 0.42) {
			r = x * x;
			r = x * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) /
				((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
		}
		else {
			r = u;
			if (x > 0.0) r = 1.0 - u;
			r = log(-log(r));
			r = c[0] + r * (c[1] + r *
				(c[2] + r * (c[3] + r *
				(c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))));
			if (x < 0.0) r = -r;
		}

		pdZ[i] = r;
	}
}
