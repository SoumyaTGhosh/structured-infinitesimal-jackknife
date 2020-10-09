import numpy as np

def get_indices_in_held_out_fold(T, pct_to_drop, contiguous=False):
    """
    :param T: length of the sequence
    :param pct_to_drop: % of T in the held out fold
    :param contiguous: if True generate a block of indices to drop else generate indices by iid sampling
    :return: o (the set of indices in the fold)
    """
    if contiguous:
        l = np.floor(pct_to_drop / 100. * T)
        anchor = np.random.choice(np.arange(l+1, T))
        o = np.arange(anchor-l, anchor).astype(int)
    else:
        # i.i.d LWCV
        o = np.random.choice(T-2, size=np.int(pct_to_drop / 100. * T), replace=False) + 1
    return o


def genSyntheticDataset(K, T, N, D, sigma0=None, seed=1234, varainces_of_mean=1.0,
                        diagonal_upweight=False):
    np.random.seed(seed)
    if sigma0 is None:
        sigma0 = np.eye(D)

    A = np.random.dirichlet(alpha=np.ones(K), size=K)
    if diagonal_upweight:
        A = A + 5 * np.eye(K) # add 5 to the diagonal and renormalize
        A = A / A.sum(axis=1)

    pi0 = np.random.dirichlet(alpha=np.ones(K))
    mus = np.random.normal(size=(K, D), scale=np.sqrt(varainces_of_mean))
    zs = np.empty((N, T), dtype=np.int)
    X = np.empty((N, T, D))

    for n in range(N):
        zs[n, 0] = int(np.random.choice(np.arange(K), p=pi0))
        X[n, 0] = np.random.multivariate_normal(mean=mus[zs[n,0]], cov=sigma0)
        for t in range(1, T):
            zs[n, t] = int(np.random.choice(np.arange(K), p=A[zs[n,t-1],:]))
            X[n, t] = np.random.multivariate_normal(mean=mus[zs[n,t]], cov=sigma0)

    return X, zs, A, pi0, mus