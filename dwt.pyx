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


# ------------------------------------ CYTHON CODE -----------------------------------

def qmf(g):
    '''
    Quadrature mirror filter for computing wavelet coefficients.
    '''
    L = g.shape[0]
    return np.power(-1.0, np.arange(0.0, L)) * g[::-1] 

def zapsmall(x, digits = 7):
    '''
    zapsmall determines a digits argument dr for calling round(x,
    digits = dr) such that values close to zero (compared with the
    maximal absolute value) are ‘zapped’, i.e., treated as 0.
    
    This mimics the zapsmall() method in the R standard library,
    although it has fewer checks for NaNs.
    '''
    mx = np.max(np.abs(x))
    print mx
    print int(max(0, digits - np.log10(mx)))
    if(mx > 0):
        return np.round(x, int(max(0, digits - np.log10(mx))))
    else:
        return np.round(x, int(digits))



def modwt3(np.ndarray[np.double_t,ndim=3] vol, num_bands):
    '''
    3D Undecimated (i.e. Stationary) Wavelet Transform
    '''

    # Ensure that the data is in fortran (column-major) order.
    cdef np.ndarray[np.double_t, ndim=3] x
    if not vol.flags['F_CONTIGUOUS']:
        x = np.asfortranarray(vol)
    else:
        x = vol

    cdef int nx = vol.shape[0]
    cdef int ny = vol.shape[1]
    cdef int nz = vol.shape[2]

    # Hard coded Haar for now
    cdef int L = 2
    cdef np.ndarray[np.double_t, ndim=1] g = np.array([0.7071067811865475, 0.7071067811865475])
    cdef np.ndarray[np.double_t, ndim=1] h = qmf(g)
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


def imodwt3(coefs, vol_shape):
    '''
    Inverse 3D Undecimated (i.e. Stationary) Wavelet Transform
    '''

    num_bands = (len(coefs)-1)/7 

    # Hard coded Haar for now
    cdef int L = 2
    cdef np.ndarray[np.double_t, ndim=1] g = np.array([0.7071067811865475, 0.7071067811865475])
    cdef np.ndarray[np.double_t, ndim=1] h = qmf(g)
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
