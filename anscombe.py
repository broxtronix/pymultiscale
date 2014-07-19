import numpy as np

#-------- Variance stabilizing transforms -----------

def anscombe(x):
    '''
    Compute the anscombe variance stabilizing transform.

    Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
    binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
    '''
    return 2.0*np.sqrt(x + 3.0/8.0)

def inverse_anscombe(z):
    '''
    Compute the inverse transform using an approximation of the exact
    unbiased inverse.

    Reference: Makitalo, M., & Foi, A. (2011). A closed-form
    approximation of the exact unbiased inverse of the Anscombe
    variance-stabilizing transformation. Image Processing.
    '''
    return (1.0/4.0 * np.power(z, 2) +
            1.0/4.0 * np.sqrt(3.0/2.0) * np.power(z, -1) -
            11.0/8.0 * np.power(z, -2) + 
            5.0/8.0 * np.sqrt(3.0/2.0) * np.power(z, -3) - 1.0 / 8.0)

def generalized_anscombe(x,mu,sigma,gain=1.0):
    return (2.0/gain)*np.sqrt(gain*x + (gain**2)*3.0/8.0 + sigma**2 - gain*mu)

def inverse_generalized_anscombe(z,mu,sigma,gain=1.0):
    return (1.0/gain)*(gain*y/2.0)**2 - gain*3.0/8.0 - (sigma**2)/gain + mu
