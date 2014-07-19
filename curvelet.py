# -----------------------------------------------------------------------------
#                       SUPPORT FUNCTIONS FOR CURVLETS
# -----------------------------------------------------------------------------

def curvelet_transform(vol, num_bands):
    import pyct
  
    ct3 = pyct.fdct3( n = vol.shape, 
                      nbs = num_bands,   # Number of bands
                      nba = 8,           # Number of discrete angles
                      ac = True,         # Return curvelets at the finest detail level
                      vec = False,       # Return results as nested python vectors
                      cpx = False)       # Disable complex-valued curvelets
    result = ct3.fwd(vol)
    del ct3
    return result

def inverse_curvelet_transform(coefs, vol_shape):
    import pyct

    num_bands = len(coefs)
    ct3 = pyct.fdct3( n = vol_shape, 
                      nbs = num_bands,     # Number of bands
                      nba = 8,             # Number of discrete angles
                      ac = True,
                      vec = False,
                      cpx = False)
    result = ct3.inv(coefs)
    del ct3
    return result

def curvelet_mean(coefs):
    accum = []
    for coef in coefs:
        accum.append(np.hstack( [ block.ravel() for block in coef ] ))
    return np.hstack(accum).mean()

def modify_curvelet_coefs(coefs, update, update_rate):

  delta_sqrsum = 0.0
  new_coefs = []
  for band in xrange(len(coefs)):
      new_band = []
      for angle in xrange(len(coefs[band])):
          delta = update_rate * update[band][angle]
          new_band.append(coefs[band][angle] + delta)
          delta_sqrsum += np.square(delta).sum()
      new_coefs.append(new_band)
  update_norm = np.sqrt(delta_sqrsum)
  return (new_coefs, update_norm)

def autotune_curvelet_thresholds(coefs, confidence_interval):
    def find_threshold(X, alpha = 0.99):
        import scipy.stats
        multiplier = scipy.stats.norm.interval(alpha, loc=0, scale=1)[1]
        med = np.median(X)
        return med + multiplier * 1.4826 * np.median(np.abs(X - med))
    
    thresholds = []
    for band in coefs:
        tmp = np.hstack( [ b.ravel() for b in band ] )
        thresholds.append( find_threshold(tmp, confidence_interval) )
        #thresholds.append( 1e-2 )

    # Disable thresholds for coarsest scale
    thresholds[0] = 0.0

    return thresholds

def threshold_curvelet_coefs(coefs, threshold):

    # If a single threshold was supplied, turn it into a list 
    # with an identical threshold for each band
    #if not isinstance(threshold, list):
    #    raise NotImplementedError("Single thresholds not implements for curvelet.")
        #    threshold = threshold * np.ones((len(coefs))) 

    assert len(threshold) == len(coefs)

    new_coefs = []
    for band_num, band in enumerate(coefs):
        new_band = []
        for curvelet_array in band:
            temp = curvelet_array.copy()
            temp[np.where(np.abs(temp) < threshold[band_num])] = 0.0
            new_band.append(temp)
        new_coefs.append(new_band)
    return new_coefs

