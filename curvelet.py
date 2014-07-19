import numpy as np

# This file requires Curvelab and the PyCurveLab packages be installed on your system.
try:
    import pyct
except ImportError:
    raise NotImplementedError("Use of curvelets requires installation of CurveLab and the PyCurveLab package.\nSee: http://curvelet.org/  and  https://www.slim.eos.ubc.ca/SoftwareLicensed/")

# -----------------------------------------------------------------------------
#                             FUNCTION API
# -----------------------------------------------------------------------------

def curvelet_transform(vol, num_bands, num_angles = 8, all_curvelets = True, as_complex = False):
    ct3 = pyct.fdct3( n = vol.shape, 
                      nbs = num_bands,   # Number of bands
                      nba = num_angles,  # Number of discrete angles
                      ac = all_curvelets,# Return curvelets at the finest detail level
                      vec = False,       # Return results as nested python vectors
                      cpx = as_complex)  # Disable complex-valued curvelets
    result = ct3.fwd(vol)
    del ct3
    return result

def inverse_curvelet_transform(coefs, vol_shape, num_bands, num_angles, all_curvelets, as_complex):
    ct3 = pyct.fdct3( n = vol_shape, 
                      nbs = num_bands,     # Number of bands
                      nba = num_angles,
                      ac = all_curvelets,
                      vec = False,
                      cpx = as_complex)
    result = ct3.inv(coefs)
    del ct3
    return result

# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class CurveletTransform(object):

    def __init__(self, vol_shape, num_bands = None, num_angles = 8, all_curvelets = True, as_complex = False):

        if num_bands == None:
            self.num_bands = int(np.ceil(np.log2(np.max(vol_shape)) - 1))
        else:
            self.num_bands = num_bands

        self.vol_shape = vol_shape
        self.num_angles = num_angles
        self.all_curvelets = all_curvelets
        self.as_complex = as_complex

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data):

        # Check argument
        assert data.shape == self.vol_shape
        
        ndims = len(self.vol_shape)
        if ndims == 1:
            raise NotImplementedError("1D curvelet transform not yet implemented.")
        elif ndims == 2:
            raise NotImplementedError("2D curvelet transform not yet implemented.")
        elif ndims == 3:
            return curvelet_transform(data, self.num_bands, self.num_angles, self.all_curvelets, self.as_complex)
        else:
            raise NotImplementedError("Curveletes not supported for %dD data." % (len(data.shape)))

    def inv(self, coefs):

        ndims = len(self.vol_shape)
        if ndims == 1:
            raise NotImplementedError("1D Inverse curvelet transform not yet implemented.")
        elif ndims == 2:
            raise NotImplementedError("2D Inverse curvelet transform not yet implemented.")
        elif ndims == 3:
            return inverse_curvelet_transform(coefs, self.vol_shape, self.num_bands, self.num_angles,
                                              self.all_curvelets, self.as_complex)
        else:
            raise NotImplementedError("Curvelets not supported for %dD data." % (len(data.shape)))

    # --------------------- Utility methods -------------------------

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''

        delta_sqrsum = 0.0
        for band in xrange(len(coefs)):
            for angle in xrange(len(coefs[band])):
                delta = update_rate * update[band][angle]
                coefs[band][angle] += delta
                delta_sqrsum += np.square(delta).sum()
        update_norm = np.sqrt(delta_sqrsum)
        return (new_coefs, update_norm)

    def mean(self, coefs):
        '''
        Compute the average over all wavelet coefficients.
        '''
        accum = []
        for coef in coefs:
            accum.append(np.hstack( [ block.ravel() for block in coef ] ))
        return np.hstack(accum).mean()

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, omit_bands = []):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        For the sake of speed and memory efficiency, updates to the coefficients
        are performed in-place.
        '''
        for b in xrange(len(coefs)):

            # Skip band?
            if b in omit_bands:
                continue

            # Compute the center and threshold.  
            tmp = np.hstack( [ angle.ravel() for angle in coefs[b] ] )
            (band_center, band_threshold) = threshold_func(tmp)

            for angle in coefs[b]:
                idxs = np.where( np.abs( angle - band_center ) < band_threshold )
                angle[idxs] = 0.0

        return coefs

