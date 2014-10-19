import numpy as np

class WaveletOperator(object):
    def __init__(self, vol_shape, transform_type = 'starlet', wavelet_type = 'la8', num_bands = 4):
        self.vol_shape = vol_shape
        self.transform_type = transform_type
        self.wavelet_type = wavelet_type

        # Create the wavelet transform object.
        # Set up the transform object, or use the one supplied by the user.
        if transform_type == 'udwt':
            from lflib.wavelets.undecimated_wavelet import UndecimatedWaveletTransform
            self.wt = UndecimatedWaveletTransform(vol_shape, wavelet_type, num_bands = num_bands)
        elif transform_type == 'starlet':
            from lflib.wavelets.starlet import StarletTransform
            self.wt = StarletTransform(vol_shape, gen2 = True, num_bands = num_bands)
        elif transform_type == 'curvelet':
            from lflib.wavelets.curvelet import CurveletTransform
            self.wt = CurveletTransform(vol_shape)
        elif transform_type == 'udwt+starlet':
            from lflib.wavelets import UndecimatedWaveletTransform, StarletTransform, CombinedTransform
            wt1 = UndecimatedWaveletTransform(vol_shape, wavelet_type, num_bands = num_bands)
            wt2 = StarletTransform(vol_shape, gen2 = True, num_bands = num_bands)
            self.wt = CombinedTransform( (wt1, wt2) )
        elif transform_type == 'udwt+starlet+curvelet':
            from lflib.wavelets import UndecimatedWaveletTransform, StarletTransform, CurveletTransform, CombinedTransform
            wt1 = UndecimatedWaveletTransform(vol_shape, wavelet_type, num_bands = num_bands		)
            wt2 = StarletTransform(vol_shape, gen2 = True, num_bands = num_bands)
            wt3 = CurveletTransform(vol_shape)
            self.wt = CombinedTransform( (wt1, wt2, wt3) )
        else:
            raise NotImplementedError("Unknown wavelet transform type: " + transform_type)

        self.num_pixels = np.prod(self.vol_shape)
        self.num_coefs = self.wt.num_coefficients()
        self.shape = (self.num_pixels, self.num_coefs)

    def matvec(self, coef_vec ):
        '''
        Take a vector containing wavelet coefficients, and reconstitute
        them, and apply the fwd wavelet transform.
        '''
        assert coef_vec.shape[0] == self.num_coefs

        coefs = self.wt.vec_to_coefs(coef_vec)
        return np.reshape(self.wt.inv(coefs), np.prod(self.vol_shape))

    def rmatvec(self, vol_vec ):
        '''
        Take a vector containing pixels or voxels, and return the
        wavelet coefficients produced by the wavelet transform.  The
        coefficients are returned in one long vector.
        '''
        assert vol_vec.shape[0] == self.num_pixels
        coefs = self.wt.fwd(np.reshape(vol_vec, self.vol_shape))
        return self.wt.coefs_to_vec(coefs)

    def as_linear_operator(self):
        from scipy.sparse.linalg.interface import LinearOperator
        return LinearOperator((self.num_pixels, self.num_coefs),
                              matvec=self.matvec,
                              rmatvec=self.rmatvec,
                              dtype='float')

    def as_volume(self, vec):
        return np.reshape(vec, self.vol_shape)

    def as_vector(self, vol):
        return np.reshape(vol, np.prod(self.vol_shape))

    def num_nonzero_coefficients(self, coef_vec):
        assert coef_vec.shape[0] == self.num_coefs
        coefs = self.wt.vec_to_coefs(coef_vec)
        return self.wt.num_nonzero_coefficients(coefs)

    def threshold_by_band(self, coef_vec, confidence_interval, scaling_factor, within_axis = None):
        assert coef_vec.shape[0] == self.num_coefs

        from lflib.wavelets.threshold import mad_threshold
        coefs = self.wt.vec_to_coefs(coef_vec)
        coefs = self.wt.threshold_by_band(coefs, lambda x: mad_threshold(x, confidence_interval),
                                          skip_bands = [], within_axis = within_axis, scaling_factor = scaling_factor)
        return self.wt.coefs_to_vec(coefs)

    def smooth_native_plane(self, coef_vec, band_attenuation):
        coefs = self.wt.vec_to_coefs(coef_vec)
        coefs_smoothed = self.wt.smooth_native_plane(coefs, band_attenuation)
        return self.wt.coefs_to_vec(coefs_smoothed)
