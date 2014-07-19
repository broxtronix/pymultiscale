import numpy as np

def universal_threshold(X):
    '''
    Universal threshold based on empirical noise estimation in the
    smallest wavelet coefficients.  See Sparse Signal and
    Image Processing p. 164 by Starck for details.

    Returns: (data_median, data_threshold)
    '''
    N = np.prod(X.shape)
    med = np.median(X)
    sigma_est = 1.4826 * np.median(np.abs(X - med))
    tau = 1.0 # 2*np.sqrt(2)
    return (med, tau*np.sqrt(2*np.log(N))*sigma_est)

def mad_threshold(X, alpha = 0.99):
    '''
    The median absolute deviation (MAD)-based confidence interval is
    similar to the universal threshold, but less conservative and
    user-adjustable.

    Returns: (data_median, data_threshold)
    '''
    print '**', alpha
    import scipy.stats
    multiplier = scipy.stats.norm.interval(alpha, loc=0, scale=1)[1]
    med = np.median(X)
    return (med, multiplier * 1.4826 * np.median(np.abs(X - med)))

