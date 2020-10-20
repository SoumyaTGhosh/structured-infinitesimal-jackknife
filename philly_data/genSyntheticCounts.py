def genSyntheticCounts(z, lam_hi, lam_lo):
    N = z.size
    counts = np.empty(N, dtype = np.int)
    for i in range(N):
        if (z[i] == 1):
            tmp_lam = lam_hi
        if(z[i] == 0): 
            tmp_lam = lam_lo
        counts[i] = np.random.poisson(lam = tmp_lam)
    data = {}
    data['counts'] = counts
    return data
