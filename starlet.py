import numpy as np

# ----------------------------  UTIILITY FUNCTIONS ---------------------------

def bspline_star(x, step):
    ndim = len(x.shape)
    C1 = 1./16.
    C2 = 4./16.
    C3 = 6./16.
    KSize = 4*step+1
    KS2 = KSize/2
    kernel = np.zeros((KSize), dtype = np.float32)
    if KSize == 1:
        kernel[0] = 1.0
    else:
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
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) - 3)
        assert num_bands > 0

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
            test = im_in - im_out
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

# ----------------- MS-VST Starlet --------------------------


def msvst(im, band):
    ndim = len(im.shape)

    def compute_tau(level, ndim):
        kernel_size = 4*(level+1)+1
        if ndim == 1:
            h_accum = np.zeros(kernel_size)
            h_accum[kernel_size/2] = 1.0   # Create an impulse
        elif ndim == 2:
            h_accum = np.zeros((kernel_size, kernel_size))
            h_accum[kernel_size/2, kernel_size/2] = 1.0   # Create an impulse
        elif ndim == 3:
            h_accum = np.zeros((kernel_size, kernel_size, kernel_size))
            h_accum[kernel_size/2, kernel_size/2, kernel_size/2] = 1.0   # Create an impulse

        step_trou = 1
        for i in range(level):
            h_accum = bspline_star(h_accum.copy(), step_trou)
            step_trou *= 2

        return ( np.sum(h_accum), np.sum(np.power(h_accum,2)), np.sum(np.power(h_accum,3)) )

    tau1, tau2, tau3  = compute_tau(band, ndim)
    #print 'band = ', band, '   tau = ', tau1, tau2, tau3
    b = np.sign(tau1) / np.sqrt(np.abs(tau1))
    #    b = 2.0 * np.sqrt(tau1/tau2)
    e = 7.0 * tau2 / (8.0 * tau1) - tau3 / (2.0 * tau2)
    return b * np.sign( im + e ) * np.sqrt( np.abs( im + e ) )


def multiscale_vst_stabilize(input_image, num_bands = None):
    ndim = len(input_image.shape)

    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) )

    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    coefs = []

    for band in xrange(num_bands):
        im_out = bspline_star(im_in, step_trou)
        coefs.append(msvst(im_in, band) - msvst(im_out, band+1))
        im_in = im_out
        step_trou *= 2

    coefs.append(im_out)
    return sum(coefs[:-1]) + msvst(coefs[-1], len(coefs)-1)


def msvst_starlet_transform(input_image, num_bands = None, gen2 = True):
    '''
    '''
    ndim = len(input_image.shape)

    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) -3 )
        assert num_bands > 0

    im_in = input_image.astype(np.float32)
    step_trou = 1
    im_out = None
    WT = []

    for band in xrange(num_bands):
        im_out = bspline_star(im_in, step_trou)
        if gen2:  # Gen2 starlet applies smoothing twice
            raise NotImplementedError("Gen2 Starlet with MS-VST not yet implemented.")
            # WT.append(msvst(im_in) - (bspline_star(im_out, step_trou))
        else:
            WT.append(msvst(im_in, band) - msvst(im_out, band+1))
            #print ''
            # WT.append((im_in) - (im_out))
        im_in = im_out.copy()
        step_trou *= 2

    WT.append(im_out)
    return WT


def inverse_msvst_starlet_transform(coefs, gen2 = True):
    '''
    '''
    # Gen1 starlet can be reconstructed simply by summing the coefficients at each scale.
    if not gen2:

        # Reconstruct the image
        recon_img = sum(coefs[:-1]) + msvst(coefs[-1], len(coefs)-1)

        # Apply the normal inverse Anscombe transform to the reconstructed image
        b0 = 1.0
        e0 = 3.0/8.0
        recon_img = np.square(recon_img / b0) - e0
        # print recon_img.min(), recon_img.max()

    # Gen2 starlet requires more careful reconstruction.
    else:
        raise NotImplementedError("Inverse MS-VST Starlet transform not yet implemented.")

    return recon_img


# -----------------------------------------------------------------------------
#                         OBJECT-ORIENTED API
# -----------------------------------------------------------------------------

class StarletTransform(object):

    def __init__(self, gen2 = True, num_bands = None):
        self.gen2 = gen2
        self.num_bands = num_bands

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data):
        return starlet_transform(data, self.num_bands, self.gen2)

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
            update_squared_sum += np.square(delta).sum()

        update_norm = np.sqrt(update_squared_sum)
        return (coefs, update_norm)

    def multiplicative_update(self, coefs, numerator, normalization, alpha):
        '''
        Multiplies the update to each set of coefficients, updating
        them in place.
        '''

        # Check arguments
        assert len(coefs) == len(numerator) == len(normalization)
        for b in xrange(len(coefs)):
            coefs[b] = (coefs[b] * numerator[b]) / (normalization[b] + alpha)
        return coefs

    def set_coefs(self, coefs, value):
        for b in xrange(len(coefs)):
            coefs[b].fill(value)

    def mean(self, coefs):
        '''
        Compute the average over all starlet coefficients.
        '''
        return np.hstack(coefs[:-1]).mean()
#        n        = sum( [ np.prod(coef.shape) for coef in coefs] )
#        coef_sum = sum( [ coef.sum()          for coef in coefs] )
#        return  coef_sum / n

    # ------------------ Thresholding methods -----------------------

    def threshold_by_band(self, coefs, threshold_func, skip_bands = [], within_axis = None):
        '''
        Threshold each band individually.  The threshold_func() should
        take an array of coefficients (which may be 1d or 2d or 3d),
        and return a tuple: (band_center, band_threshold)

        Note that the low frequency band is left untouched.

        For the sake of speed and memory efficiency, updates to the
        coefficients are performed in-place.
        '''

        for b in xrange(len(coefs)-1):

            # Skip band?
            if b in skip_bands:
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
                # Compute the center and threshold.
                (band_center, band_threshold) = threshold_func(coefs[b])

                # Zero out any coefficients that are more than
                # band_threshold units away from band_center.
                idxs = np.where( np.abs( coefs[b] - band_center ) < band_threshold )
                coefs[b][idxs] = 0.0

                # Due to their special properties, Gen2 wavelets can be forced 
                # to have a strictly positive reconstruction if we zero out all 
                # negative coefficients.
                #if self.gen2:
                #    coefs[b][np.where(coefs[b] < 0)] = 0.0

        return coefs

    def low_pass_spatial_filter(self, coefs, within_axis = 2, range = (0, 0), max_band = 1 ):

        # Compute the number of bands, but skip the final LLL image.
        for b in xrange(len(coefs) - 1):

            # There are seven directions per band level
            num_planes = coefs[b].shape[within_axis]

            for p in xrange(num_planes):

                if within_axis == 0:
                    A = coefs[b][p,:,:]
                elif within_axis == 1:
                    A = coefs[b][:,p,:]
                else:
                    A = coefs[b][:,:,p]

                if p > range[0] and p < range[1] and b < max_band:
                    A[:,:] = 0
        return coefs


class MsvstStarletTransform(StarletTransform):

    def __init__(self):
        super(MsvstStarletTransform, self).__init__(gen2 = False)

    # ------------- Forward and inverse transforms ------------------

    def fwd(self, data, num_bands = None):
        return msvst_starlet_transform(data, num_bands, gen2 = False)

    def inv(self, coefs):
        return inverse_msvst_starlet_transform(coefs, gen2 = False)
