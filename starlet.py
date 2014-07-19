# -----------------------------------------------------------------------------
#                 STARLET (aka Undecimated Isotropic Wavelets)
# -----------------------------------------------------------------------------

def bspline_star(x, step):
    ndim = len(x.shape)
    C1 = 1./16.
    C2 = 1./4.
    C3 = 3./8.
    KSize = 4*step+1
    KS2 = KSize/2
    kernel = np.zeros((KSize), dtype = np.float32)
    kernel[0] = C1
    kernel[KSize-1] = C1
    kernel[KS2+step] = C2
    kernel[KS2-step] = C2
    kernel[KS2] = C3

    result = x
    import scipy.ndimage
    for dim in xrange(ndim):
        result = scipy.ndimage.filters.convolve1d(result, kernel, axis = dim, mode='reflect', cval = 0.0)
    return result


def starlet_transform(input_image, num_bands, gen2 = True):
    '''
    Computes the starlet transform of an image (i.e. undecimated 
    isotropic wavelet transform). 
    
    The output is a python list containing the sub-bands. If the keyword Gen2 is set, 
    then it is the 2nd generation starlet transform which is computed: i.e. g = Id - h*h 
    instead of g = Id - h.
    
    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
        Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition 
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the STAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html
    '''
    
    ndim = len(input_image.shape)
    
    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    WT = []
    
    for band in xrange(num_bands):
        im_out = bspline_star(im_in, step_trou)
        if gen2:  # Gen2 starlet applies smoothing twice
            WT.append(im_in - bspline_star(im_out, step_trou))
        else:
            WT.append(im_in - im_out)
        im_in = im_out
        step_trou *= 2
    
    WT.append(im_out)
    return WT
  

def inverse_starlet_transform(coefs, gen2 = True):
    '''
    Computes the inverse starlet transform of an image (i.e. undecimated 
    isotropic wavelet transform). 
    
    The input is a python list containing the sub-bands. If the keyword Gen2 is set, 
    then it is the 2nd generation starlet transform which is computed: i.e. g = Id - h*h 
    instead of g = Id - h.
    
    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
        Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition 
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the ISTAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html
    '''
  
    # Gen1 starlet can be reconstructed simply by summing the coefficients at each scale.
    if not gen2: 
        recon_img = np.zeros_like(coefs[0])
        for i in xrange(len(coefs)):
            recon_img += coefs[i]
        
    # Gen2 starlet requires more careful reconstruction.
    else:  
        num_bands = len(coefs)-1
        recon_img = coefs[-1]
        step_trou = np.power(2, num_bands - 1)
        
        for i in reversed(range(num_bands)):
            im_temp = bspline_star(recon_img, step_trou)
            recon_img = im_temp + coefs[i]
            step_trou /= 2

    return recon_img


def threshold_starlet_coefs(coefs, threshold, gen2 = True):

    # If a single threshold was supplied, turn it into a list 
    # with an identical threshold for each band
    #if not isinstance(threshold, list):
    #    raise NotImplementedError("Single thresholds not implements for starlet.")
        # threshold = threshold * np.ones((len(coefs)-1))                        

    assert len(threshold) == (len(coefs)-1)
    
    # Apply threshold
    result_coefs = []
    for i in xrange(len(coefs)-1):
        temp = coefs[i].copy()
        temp[np.where(np.abs(temp) < threshold[i])] = 0.0
            
        # Due to their special properties, Gen2 wavelets can be forced 
        # to have a strictly positive reconstruction if we zero out all 
        # negative coefficients.
        if gen2:
            temp[np.where(temp < 0)] = 0.0

        result_coefs.append(temp)

    # Pass the low frequency coefficient image through un-touched
    result_coefs.append(coefs[-1])

    # DEBUGGING CODE
    for i in range(len(threshold)):
      new_num_coefs = np.nonzero(result_coefs[i])[0].shape[0]
      old_num_coefs = np.nonzero(coefs[i])[0].shape[0] 
      print '\t             Band %d threshold = %0.2g    Retained %d / %d   ( %0.2f%% )' % (i, threshold[i], new_num_coefs, old_num_coefs, 100 * float( float(new_num_coefs)/old_num_coefs))
    
    return result_coefs


def threshold_starlet_coefs(coefs, threshold, gen2 = True):

    # If a single threshold was supplied, turn it into a list 
    # with an identical threshold for each band
    #if not isinstance(threshold, list):
    #    raise NotImplementedError("Single thresholds not implements for starlet.")
        # threshold = threshold * np.ones((len(coefs)-1))                        

    assert len(threshold) == (len(coefs)-1)
    
    # Apply threshold
    result_coefs = []
    for i in xrange(len(coefs)-1):
        temp = coefs[i].copy()
        temp[np.where(np.abs(temp) < threshold[i])] = 0.0
            
        # Due to their special properties, Gen2 wavelets can be forced 
        # to have a strictly positive reconstruction if we zero out all 
        # negative coefficients.
        if gen2:
            temp[np.where(temp < 0)] = 0.0

        result_coefs.append(temp)

    # Pass the low frequency coefficient image through un-touched
    result_coefs.append(coefs[-1])

    # DEBUGGING CODE
    for i in range(len(threshold)):
      new_num_coefs = np.nonzero(result_coefs[i])[0].shape[0]
      old_num_coefs = np.nonzero(coefs[i])[0].shape[0] 
      print '\t             Band %d threshold = %0.2g    Retained %d / %d   ( %0.2f%% )' % (i, threshold[i], new_num_coefs, old_num_coefs, 100 * float( float(new_num_coefs)/old_num_coefs))
    
    return result_coefs

    
def update_starlet_coefs(coefs, update, update_rate):

  assert len(update) == len(coefs)

  delta_sqrsum = 0.0
  new_coefs = []
  for i in xrange(len(coefs)):
    delta = update_rate * update[i]
    new_coefs.append(coefs[i] + delta)
    delta_sqrsum += np.square(delta).sum()

  update_norm = np.sqrt(delta_sqrsum)
  return (new_coefs, update_norm)

