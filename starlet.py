import numpy as np

# ----------------------------  UTIILITY FUNCTIONS ---------------------------

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


# -----------------------------------------------------------------------------
#                            FUNCTION API
# -----------------------------------------------------------------------------

def starlet_transform(input_image, num_bands = None, gen2 = True):
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

    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) )

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


# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class StarletTransform(object):

    def __init__(self, gen2 = True):
        self.gen2 = gen2

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data, num_bands = None):
        return starlet_transform(data, num_bands, self.gen2)

    def inv(self, coefs):
        return inverse_starlet_transform(coefs, self.gen2)

    # --------------------- Utility methods -------------------------

    def num_bands(self, coefs):
        return len(coefs)-1

    def num_coefficients(self, coefs):
        return len(coefs) * np.prod(coefs[0].shape)

    def num_nonzero_coefficients(self, coefs):
        return sum([ band.nonzero()[0].shape[0] for band in coefs ])

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''

        assert len(update) == len(coefs)

        update_squared_sum = 0.0;
        for b in xrange(len(coefs)):
            delta = alpha * update[b]
            coefs[b] += delta

        update_norm = np.sqrt(update_squared_sum)
        return (coefs, update_norm)

    def mean(self, coefs):
        '''
        Compute the average over all starlet coefficients.
        '''
        n        = sum( [ np.prod(coef.shape) for coef in coefs] )
        coef_sum = sum( [ coef.sum()          for coef in coefs] )
        return  coef_sum / n

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, skip_bands = []):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        Note that the low frequency band is left untouched.

        For the sake of speed and memory efficiency, updates to the
        coefficients are performed in-place.
        '''

        num_bands = len(coefs)

        for b in xrange(num_bands-1):

            # Skip band?
            if b in skip_bands:
                continue

            # Compute the center and threshold.  
            (band_center, band_threshold) = threshold_func(coefs[b])

            # Zero out any coefficients that are more than
            # band_threshold units away from band_center.
            idxs = np.where( np.abs( coefs[b] - band_center ) < band_threshold )
            coefs[b][idxs] = 0.0

        return coefs
