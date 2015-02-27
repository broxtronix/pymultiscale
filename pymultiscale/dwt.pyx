
# Import C standard library objects for use in Cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
import cython
from cython.operator cimport dereference as deref, preincrement as inc

# Import numpy
import numpy as np
cimport numpy as np


# ---------------------------------- C DECLARATIONS -----------------------------------

cdef extern from "dwt.h":
    void three_D_modwt(double *X, int *NX, int *NY, int *NZ, int *J, int *L,
		   double *h, double *g, double *LLL, double *HLL,
		   double *LHL, double *LLH, double *HHL, double *HLH,
		   double *LHH, double *HHH)
    void three_D_imodwt(double *LLL, double *HLL, double *LHL, double *LLH,
                        double *HHL, double *HLH, double *LHH, double *HHH,
                        int *NX, int *NY, int *NZ, int *J, int *L, double *h,
                        double *g, double *image)

    void two_D_modwt(double *X, int *M, int *N, int *J, int *L, double *h,
                     double *g, double *LL, double *LH, double *HL, double *HH)
    void two_D_imodwt(double *LL, double *LH, double *HL, double *HH, int *M,
  		      int *N, int *J, int *L, double *h, double *g, double *image)

    void modwt(double *Vin, int *N, int *j, int *L, double *ht, double *gt, 
                     double *Wout, double *Vout)

    void imodwt(double *Win, double *Vin, int *N, int *j, int *L, 
                      double *ht, double *gt, double *Vout)


# ------------------------------- UTILITY FUNCTIONS ----------------------------------

def zapsmall(x, digits = 7):
    '''
    zapsmall determines a digits argument dr for calling round(x,
    digits = dr) such that values close to zero (compared with the
    maximal absolute value) are ‘zapped’, i.e., treated as 0.

    This mimics the zapsmall() method in the R standard library,
    although it has fewer checks for NaNs.
    '''
    mx = np.max(np.abs(x))
    if(mx > 0):
        return np.round(x, int(max(0, digits - np.log10(mx))))
    else:
        return np.round(x, int(digits))


# -----------------------------------------------------------------------------
#                   1D UNDECIMATED WAVELET TRANSFORM
# -----------------------------------------------------------------------------

def modwt1(np.ndarray[np.double_t,ndim=1] img, wavelet_type, num_bands = None):
    '''
    Perform a maximal overlap discrete wavelet transform (MODWT),
    which is very closely related to the 1D undecimated
    (i.e. stationary) wavelet transform.

    Arguments:

            x - A 1D numpy array to be transformed.

    wavelet_type - A string referring to one of the wavelets defined
                   in filters.py. To see the complete list, run.

                      from wavelets/filters import list_filters
                      list_filters()

        num_bands - Sets the number of bands to compute in the decomposition.
                    If 'None' is provided, then num_bands is automatically
                    set to:

                          int( ceil( log2( min(vol.shape) ) ) - 3)

    Returns:

          coefs - A python list containing (num_bands + 1)
                  entries.  Bands contain increasingly coarse wavelet 
                  bands.  The final entry contains the final low-pass 
                  residual.

    '''

    # Determine volume shape and num bands
    img_shape = np.array(img).shape
    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(img_shape))) - 3)
        assert num_bands > 0

    # Ensure that the data is in fortran (column-major) order.
    cdef np.ndarray[np.double_t, ndim=1] x
    if not img.flags['F_CONTIGUOUS']:
        x = np.asfortranarray(img)
    else:
        x = img

    cdef int nx = img.shape[0]

    # Access wavelet info
    from pymultiscale.filters import wavelet_filter
    cdef int L
    cdef np.ndarray[np.double_t, ndim=1] g
    cdef np.ndarray[np.double_t, ndim=1] h
    (L, g, h) = wavelet_filter(wavelet_type)

    # Rescale for UDWT
    g /= np.sqrt(2.0)
    h /= np.sqrt(2.0)

    # Declare arrays that will serve as working memory & a place for results.
    # The 1D wavelet decompostion is performed with only one filter orientation.
    cdef np.ndarray[np.double_t, ndim=1] LL, HH

    coefs = []
    cdef int J
    for j in range(num_bands):
        J = j+1

        # Zero out working memory
        LL = np.zeros_like(x);  HH = np.zeros_like(x);

        # Call the modwt C function
        modwt( <double*> x.data, &nx, &J, &L, <double*> h.data, <double*> g.data,
                     <double*> LL.data, <double*> HH.data)

        # Update the input image with the low-pass filter output
        x = LL.copy(order = 'F')

        # Store high pass filter channels, copying data to fresh
        # arrays so that working memory can be zero'd and re-used on
        # the next iteration.
        coefs.append(HH.copy(order = 'F'))

        # On the final iteration, store the low-pass filter image.
        if j == num_bands-1:
            coefs.append(LL.copy(order = 'F'))

    return coefs


def imodwt1(coefs, wavelet_type):
    '''
    Perform the inverse maximal overlap discrete wavelet transform
    (MODWT), which is very closely related to the 1D undecimated
    (i.e. stationary) wavelet transform.

    Arguments:

             coefs - A python list of coefficients like those produced using modwt1().

       wavelet_type - A string referring to one of the wavelets defined
                      in filters.py. To see the complete list, run.

                          from wavelets/filters import list_filters
                          list_filters()
    Returns:

                x - A 1D numpy array containing the reconstruction.
    '''

    # Extract info
    num_bands = len(coefs)-1
    img_shape = coefs[0].shape

    # Access wavelet info
    from pymultiscale.filters import wavelet_filter
    cdef int L
    cdef np.ndarray[np.double_t, ndim=1] g
    cdef np.ndarray[np.double_t, ndim=1] h
    (L, g, h) = wavelet_filter(wavelet_type)

    g /= np.sqrt(2)
    h /= np.sqrt(2)

    cdef int nx = img_shape[0]

    # We start the reconstruction using the low-pass image
    cdef np.ndarray[np.double_t, ndim=1] Yin = coefs[-1]  # LLL

    # Create working memory
    cdef np.ndarray[np.double_t, ndim=1] HH

    # Create the result buffer
    cdef np.ndarray[np.double_t, ndim=1] img = np.zeros( (nx,), dtype = np.float64, order = 'F' )

    cdef int J
    for j in np.arange(num_bands)[::-1]:
        J = j+1

        HH = coefs[j]

        imodwt( <double *> Yin.data, <double *> HH.data, &nx, &J, &L, <double *> h.data,
                      <double *> g.data, <double *> img.data)

        # The output of this iteration is the input to the next
        Yin = img.copy(order = 'F')

    return zapsmall(img)
    

# -----------------------------------------------------------------------------
#                   2D UNDECIMATED WAVELET TRANSFORM
# -----------------------------------------------------------------------------

def modwt2(np.ndarray[np.double_t,ndim=2] img, wavelet_type, num_bands = None):
    '''
    Perform a maximal overlap discrete wavelet transform (MODWT),
    which is very closely related to the 2D undecimated
    (i.e. stationary) wavelet transform.

    Arguments:

            vol - A 2D numpy array to be transformed.

    wavelet_type - A string referring to one of the wavelets defined
                   in filters.py. To see the complete list, run.

                      from wavelets/filters import list_filters
                      list_filters()

        num_bands - Sets the number of bands to compute in the decomposition.
                    If 'None' is provided, then num_bands is automatically
                    set to:

                          int( ceil( log2( min(vol.shape) ) ) - 3)

    Returns:

          coefs - A python list containing (3 * num_bands + 1)
                  entries.  Each successive set of 5 entries contain
                  the directional wavelet coefficient (HH, HL, LH) for
                  increasingly coarse wavelet bands.  The final entry
                  contains the final low-pass image (LL) at the end
                  of the filter bank.

    '''

    # Determine volume shape and num bands
    img_shape = np.array(img).shape
    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(img_shape))) - 3)
        assert num_bands > 0

    # Ensure that the data is in fortran (column-major) order.
    cdef np.ndarray[np.double_t, ndim=2] x
    if not img.flags['F_CONTIGUOUS']:
        x = np.asfortranarray(img)
    else:
        x = img

    cdef int nx = img.shape[0]
    cdef int ny = img.shape[1]

    # Access wavelet info
    from pymultiscale.filters import wavelet_filter
    cdef int L
    cdef np.ndarray[np.double_t, ndim=1] g
    cdef np.ndarray[np.double_t, ndim=1] h
    (L, g, h) = wavelet_filter(wavelet_type)

    # Rescale for UDWT
    g /= np.sqrt(2.0)
    h /= np.sqrt(2.0)

    # Declare arrays that will serve as working memory & a place for
    # results.  The 2d wavelet compostion is performed across 3
    # different orientations.
    cdef np.ndarray[np.double_t, ndim=2] LL, HH, HL, LH

    coefs = []
    cdef int J
    for j in range(num_bands):
        J = j+1

        # Zero out working memory
        LL = np.zeros_like(x);  HH = np.zeros_like(x);
        HL = np.zeros_like(x);  LH = np.zeros_like(x);

        # Call the modwt C function
        two_D_modwt( <double*> x.data, &nx, &ny, &J, &L, <double*> h.data, <double*> g.data,
                     <double*> LL.data, <double*> LH.data, <double*> HL.data, <double*> HH.data)

        # Update the input image with the low-pass filter output
        x = LL.copy(order = 'F')

        # Store high pass filter channels, copying data to fresh
        # arrays so that working memory can be zero'd and re-used on
        # the next iteration.
        coefs.append(HL.copy(order = 'F'))
        coefs.append(LH.copy(order = 'F'))
        coefs.append(HH.copy(order = 'F'))

        # On the final iteration, store the low-pass filter image.
        if j == num_bands-1:
            coefs.append(LL.copy(order = 'F'))

    return coefs


def imodwt2(coefs, wavelet_type):
    '''
    Perform the inverse maximal overlap discrete wavelet transform
    (MODWT), which is very closely related to the 2D undecimated
    (i.e. stationary) wavelet transform.

    Arguments:

             coefs - A python list of coefficients like those produced using modwt2().

       wavelet_type - A string referring to one of the wavelets defined
                      in filters.py. To see the complete list, run.

                          from wavelets/filters import list_filters
                          list_filters()
    Returns:

                x - A 2D numpy array containing the reconstruction.
    '''

    # Extract info
    num_bands = (len(coefs)-1)/3
    img_shape = coefs[0].shape

    # Access wavelet info
    from pymultiscale.filters import wavelet_filter
    cdef int L
    cdef np.ndarray[np.double_t, ndim=1] g
    cdef np.ndarray[np.double_t, ndim=1] h
    (L, g, h) = wavelet_filter(wavelet_type)

    g /= np.sqrt(2)
    h /= np.sqrt(2)

    cdef int nx = img_shape[0]
    cdef int ny = img_shape[1]

    # We start the reconstruction using the low-pass image
    cdef np.ndarray[np.double_t, ndim=2] Yin = coefs[-1]  # LLL

    # Create working memory
    cdef np.ndarray[np.double_t, ndim=2] HL, LH, HH

    # Create the result buffer
    cdef np.ndarray[np.double_t, ndim=2] img = np.zeros( (nx, ny), dtype = np.float64, order = 'F' )

    cdef int J
    for j in np.arange(num_bands)[::-1]:
        J = j+1

        HL = coefs[j*3  ]
        LH = coefs[j*3+1]
        HH = coefs[j*3+2]

        two_D_imodwt( <double *> Yin.data, <double *> LH.data, <double *> HL.data,
                      <double *> HH.data, &nx, &ny, &J, &L, <double *> h.data,
                      <double *> g.data, <double *> img.data)

        # The output of this iteration is the input to the next
        Yin = img.copy(order = 'F')

    return zapsmall(img)

# -----------------------------------------------------------------------------
#                   3D UNDECIMATED WAVELET TRANSFORM
# -----------------------------------------------------------------------------

def modwt3(np.ndarray[np.double_t,ndim=3] vol, wavelet_type, num_bands = None):
    '''
    Perform a maximal overlap discrete wavelet transform (MODWT),
    which is very closely related to the 3D undecimated
    (i.e. stationary) wavelet transform.

    Arguments:

            vol - A 3D numpy array to be transformed.

    wavelet_type - A string referring to one of the wavelets defined
                   in filters.py. To see the complete list, run.

                      from wavelets/filters import list_filters
                      list_filters()

        num_bands - Sets the number of bands to compute in the decomposition.
                    If 'None' is provided, then num_bands is automatically
                    set to:

                          int( ceil( log2( min(vol.shape) ) ) - 3 )

    Returns:

          coefs - A python list containing (7 * num_bands + 1) entries.
                  Each successive set of 7 entries contain the
                  directional wavelet coefficient (HHH, HLL, LHL, LLH,
                  HHL, HLH, LHH) for increasingly coarse wavelet bands.
                  The final entry contains the final low-pass image (LLL)
                  at the end of the filter bank.

    '''

    # Determine volume shape and num bands
    vol_shape = np.array(vol).shape
    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(vol_shape))) - 3)
        assert num_bands > 0


    # Ensure that the data is in fortran (column-major) order.
    cdef np.ndarray[np.double_t, ndim=3] x
    if not vol.flags['F_CONTIGUOUS']:
        x = np.asfortranarray(vol)
    else:
        x = vol

    cdef int nx = vol.shape[0]
    cdef int ny = vol.shape[1]
    cdef int nz = vol.shape[2]

    # Access wavelet info
    from pymultiscale.filters import wavelet_filter
    cdef int L
    cdef np.ndarray[np.double_t, ndim=1] g
    cdef np.ndarray[np.double_t, ndim=1] h
    (L, g, h) = wavelet_filter(wavelet_type)

    # Rescale for UDWT
    g /= np.sqrt(2.0)
    h /= np.sqrt(2.0)

    # Declare arrays that will serve as working memory & a place for
    # results.  The 3d wavelet compostion is performed acress 8
    # different directions.
    cdef np.ndarray[np.double_t, ndim=3] LLL, HLL, LHL, LLH, HHL, HLH, LHH, HHH

    coefs = []
    cdef int J
    for j in range(num_bands):
        J = j+1

        # Zero out working memory
        LLL = np.zeros_like(x);  HHH = np.zeros_like(x);
        HLL = np.zeros_like(x);  LHL = np.zeros_like(x);  LLH = np.zeros_like(x)
        HHL = np.zeros_like(x);  HLH = np.zeros_like(x);  LHH = np.zeros_like(x)

        # Call the modwt C function
        three_D_modwt( <double*> x.data, &nx, &ny, &nz, &J, &L, <double*> h.data, <double*> g.data,
                       <double*> LLL.data, <double*> HLL.data, <double*> LHL.data, <double*> LLH.data,
                       <double*> HHL.data, <double*> HLH.data, <double*> LHH.data, <double*> HHH.data )

        # Update the input volume with the low-pass filter output
        x = LLL.copy(order = 'F')

        # Store high pass filter channels, copying data to fresh
        # arrays so that working memory can be zero'd and re-used on
        # the next iteration.
        coefs.append(HLL.copy(order = 'F'))
        coefs.append(LHL.copy(order = 'F'))
        coefs.append(LLH.copy(order = 'F'))
        coefs.append(HHL.copy(order = 'F'))
        coefs.append(HLH.copy(order = 'F'))
        coefs.append(LHH.copy(order = 'F'))
        coefs.append(HHH.copy(order = 'F'))

        # On the final iteration, store the low-pass filter image.
        if j == num_bands-1:
            coefs.append(LLL.copy(order = 'F'))

    return coefs


def imodwt3(coefs, wavelet_type):
    '''
    Perform the inverse maximal overlap discrete wavelet transform
    (MODWT), which is very closely related to the 3D undecimated
    (i.e. stationary) wavelet transform.

    Arguments:

             coefs - A python list of coefficients like those produced using modwt3().

       wavelet_type - A string referring to one of the wavelets defined
                      in filters.py. To see the complete list, run.

                          from wavelets/filters import list_filters
                          list_filters()
    Returns:

                vol - A 3D numpy array containing the reconstruction.
    '''

    # Extract info
    num_bands = (len(coefs)-1)/7
    vol_shape = coefs[0].shape

    # Access wavelet info
    from pymultiscale.filters import wavelet_filter
    cdef int L
    cdef np.ndarray[np.double_t, ndim=1] g
    cdef np.ndarray[np.double_t, ndim=1] h
    (L, g, h) = wavelet_filter(wavelet_type)

    g /= np.sqrt(2)
    h /= np.sqrt(2)

    cdef int nx = vol_shape[0]
    cdef int ny = vol_shape[1]
    cdef int nz = vol_shape[2]

    # We start the reconstruction using the low-pass image
    cdef np.ndarray[np.double_t, ndim=3] Yin = coefs[-1]  # LLL

    # Create working memory
    cdef np.ndarray[np.double_t, ndim=3] HLL, LHL, LLH, HHL, HLH, LHH, HHH

    # Create the result buffer
    cdef np.ndarray[np.double_t, ndim=3] vol = np.zeros( (nx, ny, nz), dtype = np.float64, order = 'F' )

    cdef int J
    for j in np.arange(num_bands)[::-1]:
        J = j+1

        HLL = coefs[j*7  ]
        LHL = coefs[j*7+1]
        LLH = coefs[j*7+2]
        HHL = coefs[j*7+3]
        HLH = coefs[j*7+4]
        LHH = coefs[j*7+5]
        HHH = coefs[j*7+6]

        three_D_imodwt( <double *> Yin.data, <double *> HLL.data, <double *> LHL.data,
                        <double *> LLH.data, <double *> HHL.data, <double *> HLH.data,
                        <double *> LHH.data, <double *> HHH.data, &nx, &ny, &nz, &J, &L,
                        <double *> h.data, <double *> g.data, <double *> vol.data)

        # The output of this iteration is the input to the next
        Yin = vol.copy(order = 'F')

    return zapsmall(vol)
