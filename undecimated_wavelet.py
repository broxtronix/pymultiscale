# This is a Python port of portions of the waveslim package for R.
# 
#   http://cran.r-project.org/web/packages/waveslim/index.html
#
# Waveslim was written by Brandon Whitcher <bwhitcher@gmail.com>.
# This Python port was written by Michael Broxton
# <broxton@stanford.edu>.
#
# This code and is licensed under the GPL (v2 or above).
#
# At the moment only the 3D undecimated wavelet transform and its
# inverse have been wrapped in Python.  However, it would be easy to
# wrap the the 1D, 2D, and 3D DWT, as well as the 1D, and 2D UDWT.
# The C code for doing so is already compiled as part of this module,
# and the code below could serve as a guide for wrapping these other
# functions.
import numpy as np

# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class UndecimatedWaveletTransform(object):

    def __init__(self, wavelet_type):
        '''
        A class for performing the maximal overlap discrete wavelet
        transform (MODWT), which is very closely related to the 3D
        undecimated (i.e. stationary) wavelet transform.

        Arguments:

           wavelet_type - A string referring to one of the wavelets defined
                          in filters.py. To see the complete list, run.

                            from wavelets/filters import list_filters
                            list_filters()
        '''

        # Store wavelet type
        self.wavelet_type = wavelet_type

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data, num_bands = None):
        '''
        Perform a maximal overlap discrete wavelet transform (MODWT),
        which is very closely related to the 3D undecimated
        (i.e. stationary) wavelet transform.

        Arguments:

            data - A 1D, 2D, or 3D numpy array to be transformed.


        num_bands - Sets the number of bands to compute in the decomposition.
                    If 'None' is provided, then num_bands is automatically
                    set to:

                          int( ceil( log2( min(data.shape) ) ) )

        Returns:

            coefs - A python list containing (7 * num_bands + 1) entries.
                    Each successive set of 7 entries contain the
                    directional wavelet coefficient (HHH, HLL, LHL, LLH,
                    HHL, HLH, LHH) for increasingly coarse wavelet bands.
                    The final entry contains the final low-pass image (LLL)
                    at the end of the filter bank.

        '''

        ndims = len(data.shape)

        if data.dtype != np.float64:
            data = data.astype(np.float64, order = 'F')
        
        if ndims == 1:
            raise NotImplementedError("1D UDWT not yet implemented.")
        elif ndims == 2:
            raise NotImplementedError("2D UDWT not yet implemented.")
        elif ndims == 3:
            from lflib.wavelets.dwt import modwt3
            return modwt3(data, self.wavelet_type, num_bands)
        else:
            raise NotImplementedError("UDWT not supported for %dD data." % (len(data.shape)))

    def inv(self, coefs):

        ndims = len(coefs[0].shape)
        if ndims == 1:
            raise NotImplementedError("1D Inverse UDWT not yet implemented.")
        elif ndims == 2:
            raise NotImplementedError("2D Inverse UDWT not yet implemented.")
        elif ndims == 3:
            from lflib.wavelets.dwt import imodwt3
            return imodwt3(coefs, self.wavelet_type)
        else:
            raise NotImplementedError("UDWT not supported for %dD data." % (len(data.shape)))
    
    # --------------------- Utility methods -------------------------

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''

        # Check arguments
        assert len(coefs) == len(update)
        
        update_squared_sum = 0.0;
        for p in zip(coefs, update):
            delta = alpha * p[1]
            p[0] += delta
            update_squared_sum += np.square(delta).sum()

        update_norm = np.sqrt(update_squared_sum)
        return (coefs, update_norm)

    def mean(self, coefs):
        '''
        Compute the average over all wavelet coefficients.
        '''
        n        = sum( [ np.prod(coef.shape) for coef in coefs] )
        coef_sum = sum( [ coef.sum()          for coef in coefs] )
        return  coef_sum / n

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, omit_bands = []):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        Note that the low frequency band is left untouched.

        For the sake of speed and memory efficiency, updates to the
        coefficients are performed in-place.
        '''

        # Store number of dimensions
        ndims = len(coefs[0].shape)
        if ndims == 1:
            ndirections = 1
        elif ndims == 2:
            ndirections = 3
        elif ndims == 3:
            ndirections = 7
        else:
            raise NotImplementedError("UDWT not supported for %dD data." % (ndims))

        # Compute the number of bands
        num_bands = (len(coefs)-1)/ndirections

        # Combine all the directional coefficients from each band, and
        # pass them into the threshold_func() to determine a per-band
        # threshold.
        coefs_by_band = [ np.hstack([ np.array(coef) for coef in coefs[ndirections*band_num :
                                                                       ndirections*band_num + ndirections] ])
                          for band_num in xrange(num_bands) ]
        
        for b in xrange(num_bands):

            # Skip band?
            if b in omit_bands:
                continue

            # Compute the center and threshold.  
            (band_center, band_threshold) = threshold_func(coefs_by_band[b])

            # Zero out any coefficients that are more than
            # band_threshold units away from band_center.
            for j in xrange(ndims):
                idxs = np.where( np.abs( coefs[b*ndirections + j] - band_center ) < band_threshold )
                coefs[b*ndirections + j][idxs] = 0.0
        return coefs

          

