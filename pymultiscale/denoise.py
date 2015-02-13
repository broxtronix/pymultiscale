import numpy as np

def wavelet_denoise(vol, transform_type = 'udwt', confidence_interval = None ):
    """
    Denoise a volume by thresholding in wavelet band space.
    threshold: overrides alpha
    """

    # Set up the transform object, or use the one supplied by the user.
    if transform_type == 'udwt':
        from lflib.wavelets.undecimated_wavelet import UndecimatedWaveletTransform
        wt = UndecimatedWaveletTransform('d8')
    elif transform_type == 'starlet':
        from lflib.wavelets.starlet import StarletTransform
        wt = StarletTransform(gen2 = True)
    elif transform_type == 'curvelet':
        from lflib.wavelets.curvelet import CurveletTransform
        wt = CurveletTransform(vol.shape)
    elif transform_type == 'udwt+starlet':
        from lflib.wavelets import UndecimatedWaveletTransform, StarletTransform, CombinedTransform
        wt1 = UndecimatedWaveletTransform('d8')
        wt2 = StarletTransform(gen2 = True)
        wt = CombinedTransform( (wt1, wt2) )
    elif transform_type == 'udwt+starlet+curvelet':
        from lflib.wavelets import UndecimatedWaveletTransform, StarletTransform, CurveletTransform, CombinedTransform
        wt1 = UndecimatedWaveletTransform('d8')
        wt2 = StarletTransform(gen2 = True)
        wt3 = CurveletTransform(vol.shape)
        wt = CombinedTransform( (wt1, wt2, wt3) )
    else:
        wt = transform_type

    # Transform the volume into wavelet space.
    coefs = wt.fwd(vol)

    # Print out debugging info
    # print 'Decomposed into %d bands.' % (wt.num_bands(coefs))

    # Threshold & invert the transform.
    from lflib.wavelets.threshold import mad_threshold, universal_threshold
    coefs = wt.threshold_by_band(coefs, lambda x: mad_threshold(x, confidence_interval), skip_bands = [], within_axis = 2)
    # coefs = wt.low_pass_spatial_filter(coefs, within_axis = 2, range = (15, 35), max_band = 0)
    #    coefs = wt.threshold_by_band(coefs, lambda x: universal_threshold(x), skip_bands = [])

    # Inverse transform
    result = wt.inv(coefs)

    num_coefs = wt.num_coefficients(coefs)
    num_nonzero = wt.num_nonzero_coefficients(coefs)
    print 'Retained %d / %d coefficients  ( %0.2f %%)' % (num_nonzero, num_coefs,
                                                          100.0 * float(num_nonzero) / num_coefs)

    return result[:vol.shape[0],:vol.shape[1],:vol.shape[2]]
