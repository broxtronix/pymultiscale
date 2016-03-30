# PyMultiscale

PyMultiscale is a collection of 1D, 2D, and 3D wavelet transforms for Python.  This code is unsupported, and a work in progress.  Still, you may find it useful.  Enjoy! 

The following families of transforms are supported:

* MODWT - Maximal Overlap Discrete Wavelet Transform

This is a Python port of portions of the [http://cran.r-project.org/web/packages/waveslim/index.html](waveslim) package for R by Brandon Whitcher <bwhitcher@gmail.com>.

* Starlet Transform (i.e. undecimated isotropic wavelet transform) [1, 2]

* Curvelet Transform

This is a thin wrapper around PyCurvelab, which must be installed seperately.  First install Curvelab, and the PyCurvelab.

http://www.curvelet.org/
https://www.slim.eos.ubc.ca/SoftwareLicensed/

## References

    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
    Astronomy and Astrophysics, 288, pp-343-348, 1994.
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

