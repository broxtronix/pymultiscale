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

    def __init__(self, wavelet_type, num_bands = None):
        '''
        A class for performing the maximal overlap discrete wavelet
        transform (MODWT), which is very closely related to the 3D
        undecimated (i.e. stationary) wavelet transform.

        Arguments:

           wavelet_type - A string referring to one of the wavelets defined
                          in filters.py. To see the complete list, run.

                            from wavelets/filters import list_filters
                            list_filters()

              num_bands - Sets the number of bands to compute in the decomposition.
                          If 'None' is provided, then num_bands is automatically
                          set to:

                             int( ceil( log2( min(data.shape) ) ) - 3)
        '''

        # Store wavelet type
        self.wavelet_type = wavelet_type
        self.num_bands = num_bands

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data):
        '''
        Perform a maximal overlap discrete wavelet transform (MODWT),
        which is very closely related to the 3D undecimated
        (i.e. stationary) wavelet transform.

        Arguments:

            data - A 1D, 2D, or 3D numpy array to be transformed.

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

        if self.num_bands == None:
            num_bands = int(np.ceil(np.log2(np.min(data.shape))) - 3)
            assert num_bands > 0
        else:
            num_bands = self.num_bands

        if ndims == 1:
            raise NotImplementedError("1D UDWT not yet implemented.")
        elif ndims == 2:
            from lflib.wavelets.dwt import modwt2
            return modwt2(data, self.wavelet_type, num_bands)
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
            from lflib.wavelets.dwt import imodwt2
            return imodwt2(coefs, self.wavelet_type)
        elif ndims == 3:
            from lflib.wavelets.dwt import imodwt3
            return imodwt3(coefs, self.wavelet_type)
        else:
            raise NotImplementedError("UDWT not supported for %dD data." % (len(data.shape)))

    # --------------------- Utility methods -------------------------

    def num_bands(self, coefs):
        if len(coefs[0].shape) == 2:
            return (len(coefs)-1)/3
        elif len(coefs[0].shape) == 2:
            return (len(coefs)-1)/7
        else:
            raise NotImplementedError("UDWT num_bands() not supported for %dD data." % (len(data.shape)))

    def num_coefficients(self, coefs):
        return len(coefs) * np.prod(coefs[0].shape)

    def num_nonzero_coefficients(self, coefs):
        return sum([ band.nonzero()[0].shape[0] for band in coefs ])

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''

        # Check arguments
        assert len(coefs) == len(update)

        update_squared_sum = 0.0;
        for b in xrange(len(coefs)):
            delta = alpha * update[b]
            coefs[b] += delta
            update_squared_sum += np.square(delta).sum()

        update_norm = np.sqrt(update_squared_sum)
        return (coefs, update_norm)

    def multiplicative_update(self, coefs, numerator, denominator, normalization, alpha):
        '''
        Multiplies the update to each set of coefficients, updating
        them in place.
        '''

        # Check arguments
        assert len(coefs) == len(numerator) == len(denominator)
        for b in xrange(len(coefs)):
            coefs[b] = ((coefs[b] * numerator[b]) / normalization[b]) / (denominator[b] + alpha)
        return coefs

    def set_coefs(self, coefs, value):
        for b in xrange(len(coefs)):
            coefs[b].fill(value)

    def mean(self, coefs):
        '''
        Compute the average over all wavelet coefficients.
        '''
        n        = sum( [ np.prod(coef.shape) for coef in coefs] )
        coef_sum = sum( [ coef.sum()          for coef in coefs] )
        return  coef_sum / n

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, skip_bands = [], within_axis = None):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        If you want to threshold within a particular plane within a band,
        set within_axis to that plane.

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

        # Compute the number of bands, but skip the final LLL image.
        for b in xrange(len(coefs) - 1):

            # There are seven directions per band level
            band_level = int(np.floor(b/7))

            # Skip band?
            if band_level in skip_bands:
                continue

            if within_axis != None:
                num_planes = coefs[b].shape[within_axis]
                for p in xrange(num_planes):
                    if within_axis == 0:
                        A = coefs[b][p,:,:]
                    elif within_axis == 1:
                        A = coefs[b][:,p,:]
                    else:
                        A = coefs[b][:,:,p]

                    (band_center, band_threshold) = threshold_func(A)

                    # Zero out any coefficients that are more than
                    # band_threshold units away from band_center.
                    idxs = np.where( np.abs( A - band_center ) < band_threshold )
                    A[idxs] = 0.0

            else:
                (band_center, band_threshold) = threshold_func(coefs[b])

                # Zero out any coefficients that are more than
                # band_threshold units away from band_center.
                idxs = np.where( np.abs( coefs[b] - band_center ) < band_threshold )
                coefs[b][idxs] = 0.0

        return coefs


    def low_pass_spatial_filter(self, coefs, within_axis = 2, range = (0, 0), max_band = 1 ):

        # Compute the number of bands, but skip the final LLL image.
        num_bands = len(coefs) - 1
        for b in xrange(num_bands):

            # There are seven directions per band level
            band_level = int(np.floor(b/7))
            num_planes = coefs[b].shape[within_axis]

            for p in xrange(num_planes):

                if within_axis == 0:
                    A = coefs[b][p,:,:]
                elif within_axis == 1:
                    A = coefs[b][:,p,:]
                else:
                    A = coefs[b][:,:,p]

                if p > range[0] and p < range[1] and band_level < max_band:
                    A[:,:] = 0
        return coefs

