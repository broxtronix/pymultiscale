import numpy as np

# -----------------------------------------------------------------------------
#                             FUNCTION API
# -----------------------------------------------------------------------------

def curvelet_transform(x, num_bands, num_angles = 8, all_curvelets = True, as_complex = False):
    ndims = len(x.shape)

    # This file requires Curvelab and the PyCurveLab packages be installed on your system.
    try:
        import pyct
    except ImportError:
        raise NotImplementedError("Use of curvelets requires installation of CurveLab and the PyCurveLab package.\nSee: http://curvelet.org/  and  https://www.slim.eos.ubc.ca/SoftwareLicensed/")

    if ndims == 2:
        ct = pyct.fdct2( n = x.shape,
                         nbs = num_bands,   # Number of bands
                         nba = num_angles,  # Number of discrete angles
                         ac = all_curvelets,# Return curvelets at the finest detail level
                         vec = False,       # Return results as nested python vectors
                         cpx = as_complex)  # Disable complex-valued curvelets
    elif ndims == 3:
        ct = pyct.fdct3( n = x.shape,
                         nbs = num_bands,   # Number of bands
                         nba = num_angles,  # Number of discrete angles
                         ac = all_curvelets,# Return curvelets at the finest detail level
                         vec = False,       # Return results as nested python vectors
                         cpx = as_complex)  # Disable complex-valued curvelets
    else:
        raise NotImplementedError("%dD Curvelets are not supported." % (ndims))
    result = ct.fwd(x)
    del ct
    return result

def inverse_curvelet_transform(coefs, x_shape, num_bands, num_angles, all_curvelets, as_complex):
    # This file requires Curvelab and the PyCurveLab packages be installed on your system.
    try:
        import pyct
    except ImportError:
        raise NotImplementedError("Use of curvelets requires installation of CurveLab and the PyCurveLab package.\nSee: http://curvelet.org/  and  https://www.slim.eos.ubc.ca/SoftwareLicensed/")

    if len(x_shape) == 2:
        ct = pyct.fdct2( n = x_shape,
                          nbs = num_bands,     # Number of bands
                          nba = num_angles,
                          ac = all_curvelets,
                          vec = False,
                          cpx = as_complex)
    else:
        ct = pyct.fdct3( n = x_shape,
                          nbs = num_bands,     # Number of bands
                          nba = num_angles,
                          ac = all_curvelets,
                          vec = False,
                          cpx = as_complex)
    result = ct.inv(coefs)
    del ct
    return result

# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class CurveletTransform(object):

    def __init__(self, x_shape, num_bands = None, num_angles = 8, all_curvelets = True, as_complex = False):
        if num_bands == None:
            self._num_bands = int(np.ceil(np.log2(np.min(x_shape)) - 3))
        else:
            self._num_bands = num_bands

        self.x_shape = x_shape
        self.num_angles = num_angles
        self.all_curvelets = all_curvelets
        self.as_complex = as_complex

        self.example_coefs = self.fwd(np.zeros(x_shape))


    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data, num_bands = None):
        '''
        Curvelets must have the num_bands initialized in the
        constructor, but for uniformity with the API for forward
        transforms, we allow the user to supply a num_bands
        argument.  If the supplied num_bands does not match the
        num_bands used in the constructor, an error is generated.
        '''

        if num_bands != None:
            assert self._num_bands == num_bands

        # Check argument
        assert data.shape == self.x_shape

        ndims = len(self.x_shape)
        if ndims == 1:
            raise NotImplementedError("1D curvelet transform not yet implemented.")
        elif ndims == 2:
            return curvelet_transform(data, self._num_bands, self.num_angles, self.all_curvelets, self.as_complex)
        elif ndims == 3:
            return curvelet_transform(data, self._num_bands, self.num_angles, self.all_curvelets, self.as_complex)
        else:
            raise NotImplementedError("Curveletes not supported for %dD data." % (len(data.shape)))

    def inv(self, coefs):

        ndims = len(self.x_shape)
        if ndims == 1:
            raise NotImplementedError("1D Inverse curvelet transform not yet implemented.")
        elif ndims == 2:
            return inverse_curvelet_transform(coefs, self.x_shape, self._num_bands, self.num_angles,
                                              self.all_curvelets, self.as_complex)
        elif ndims == 3:
            return inverse_curvelet_transform(coefs, self.x_shape, self._num_bands, self.num_angles,
                                              self.all_curvelets, self.as_complex)
        else:
            raise NotImplementedError("Curvelets not supported for %dD data." % (len(data.shape)))

    # --------------------- Utility methods -------------------------

    def num_bands(self, coefs):
        return self._num_bands

    def num_coefficients(self):
        total = 0
        for band in self.example_coefs:
            total += sum([ np.prod(angle.shape) for angle in band ] )
        return total

    def num_nonzero_coefficients(self, coefs):
        total = 0
        for band in coefs:
            total += sum([ angle.nonzero()[0].shape[0] for angle in band ] )
        return total


    def coefs_to_vec(self, coefs):
        return np.hstack([np.hstack([ angle.ravel(order = 'c') for angle in band ]) for band in coefs])

    def vec_to_coefs(self, coef_vec):
        base_idx = 0
        coefs = []
        for band in self.example_coefs:
            angle_list = []
            for angle in band:
                angle_size = np.prod(angle.shape)
                angle_list.append(np.reshape(coef_vec[base_idx:base_idx+angle_size], angle.shape, order = 'c'))
                base_idx += angle_size
            coefs.append(angle_list)
        return coefs

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''

        delta_sqrsum = 0.0
        for band in xrange(len(coefs)):
            for angle in xrange(len(coefs[band])):
                delta = alpha * update[band][angle]
                coefs[band][angle] += delta
                delta_sqrsum += np.square(delta).sum()
        update_norm = np.sqrt(delta_sqrsum)
        return (coefs, update_norm)

    def mean(self, coefs):
        '''
        Compute the average over all wavelet coefficients.
        '''
        accum = []
        for coef in coefs:
            accum.append(np.hstack( [ block.ravel() for block in coef ] ))
        return np.hstack(accum).mean()

    # ------------------ Thresholding methods -----------------------
    def _estimate_noise(self):
        '''
        Helper function for the thresholding function below.

        Adapted from the fdct_osfft_demo_denoise.m file in CurveLab.
        '''
        E_coefs = self.fwd(np.random.randn(*self.x_shape))
        E_thresholds = []
        for band in xrange(len(E_coefs)):
            angle_thresholds = []
            for angle in xrange(len(E_coefs[band])):
                A = E_coefs[band][angle]
                angle_thresholds.append( np.median(np.abs(A - np.median(A)))/0.6745 )
            E_thresholds.append(angle_thresholds)
        return E_thresholds

    def threshold_by_band(self, coefs, threshold_func, skip_bands = [], within_axis = None, scaling_factor = None):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        For the sake of speed and memory efficiency, updates to the coefficients
        are performed in-place.
        '''

        # Skip the lowest frequency band
        for b in xrange(1, len(coefs)):
            num_removed = 0
            num_total = 0

            # Skip band?
            if b in skip_bands:
                continue

            # print num_total - num_removed, num_total
            # Compute the center and threshold.
            tmp = np.hstack( [ angle.ravel() for angle in coefs[b] ] )
            (band_center, band_threshold) = threshold_func(tmp, b, None)
            # print '\t\t****', b, band_center, band_threshold
            #if scaling_factor != None:
            #    band_threshold /= scaling_factor

            for a in xrange(len(coefs[b])):

                # Soft threshold the coefficients
                idxs = np.where( coefs[b][a] > band_threshold )
                coefs[b][a][idxs] -= band_threshold
                idxs = np.where( np.abs(coefs[b][a]) <= band_threshold )
                coefs[b][a][idxs] = 0.0
                idxs = np.where( coefs[b][a] < -band_threshold )
                coefs[b][a][idxs] += band_threshold

                #idxs = np.where( np.abs( coefs[b][a] - band_center ) < band_threshold )#  * scaling_factor[b][a] / max_scale )

                #num_removed += idxs[0].shape[0]
                #num_total += np.prod(coefs[b][a].shape)
                #coefs[b][a][idxs] = 0.0

            #print 'Retained %0.2f -- ( %g / %g )' % (100.0*(num_total - num_removed)/float(num_total),
            #                                         num_total - num_removed, num_total)
        return coefs
