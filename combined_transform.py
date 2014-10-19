import numpy as np

# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class CombinedTransform(object):

    def __init__(self, transform_objects):
        '''
        transform_objects is a tuple containing several
        (pre-initialized) wavelet, starlet, or curvelet transform
        objects.
        '''
        self.transform_objects = transform_objects
        self.subvec_sizes = [ wt.num_coefficients() for wt in transform_objects ]

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data):
        '''

        In this transform, num_bands can be a list containing the
        number of bands desired for each sub-transform.  You can
        supply 'None' at any position in this list to force a
        sub-tranform to choose its num_bands automatically.
        '''
        return tuple( trans.fwd(data) for trans in self.transform_objects )

    def inv(self, coefs):
        '''
        Inversion of a morphologically diverse transform.  The
        resulting transform is the average of the component
        transforms.
        '''

        sub_itransforms = tuple( p[0].inv(p[1]) for p in zip(self.transform_objects, coefs) )
        return sum(sub_itransforms) / float(len(sub_itransforms))

    # --------------------- Utility methods -------------------------

    def num_bands(self, coefs):
        return tuple( p[0].num_bands(p[1]) for p in zip(self.transform_objects, coefs) )

    def num_coefficients(self):
        return sum([ p.num_coefficients() for p in self.transform_objects ])

    def num_nonzero_coefficients(self, coefs):
        return sum(tuple( p[0].num_nonzero_coefficients(p[1]) for p in zip(self.transform_objects, coefs) ))

    def coefs_to_vec(self, coefs):
        return np.hstack([ p[0].coefs_to_vec(p[1]) for p in zip(self.transform_objects, coefs) ])

    def vec_to_coefs(self, coef_vec):
        split_locs = np.cumsum(self.subvec_sizes)[:-1]
        return tuple( self.transform_objects[i].vec_to_coefs(subvec) for i, subvec in enumerate(np.split(coef_vec, split_locs)) )

    def update(self, coefs, update, alpha):
        '''
        Adds the update (multiplied by alpha) to each set of
        coefficients.
        '''
        update_norm_total = 0.0
        result_ceofs = []
        for p in zip(self.transform_objects, coefs, update):
            (result_ceof, update_norm) = p[0].update( p[1], p[2], alpha )
            update_norm_total += np.square(update_norm_total)
            result_ceofs.append(result_ceof)
        return (result_ceofs, np.sqrt(update_norm_total))

    def mean(self, coefs):
        '''
        Compute the average over all starlet coefficients.
        '''
        subtransform_means = tuple( p[0].mean(p[1]) for p in zip(self.transform_objects, coefs) )

        # !! This is not exactly the combined mean.  We should be careful !!
        return sum(subtransform_means) / float(len(subtransform_means))

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, skip_bands = [], within_axis = None, scaling_factor = None):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        Note that the low frequency band is left untouched.

        For the sake of speed and memory efficiency, updates to the
        coefficients are performed in-place.
        '''
        for p in zip(self.transform_objects, coefs):
            p[0].threshold_by_band(p[1], threshold_func, skip_bands, within_axis, scaling_factor)

        return coefs
