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
    #return (z/2.0)**2 - 3.0/8.0
    return (1.0/4.0 * np.power(z, 2) +
            1.0/4.0 * np.sqrt(3.0/2.0) * np.power(z, -1.0) -
            11.0/8.0 * np.power(z, -2.0) + 
            5.0/8.0 * np.sqrt(3.0/2.0) * np.power(z, -3.0) - 1.0 / 8.0)

def generalized_anscombe(x, mu, sigma, gain=1.0):
    '''
    Compute the generalized anscombe variance stabilizing transform,
    which assumes that the data provided to it is a mixture of poisson
    and gaussian noise.

    We assume that x contains only positive values.  Values that are
    less than or equal to 0 are ignored by the transform.

    Note, this transform will show some bias for counts less than
    about 20.
    '''
    result = x.copy()

    positive_idxs = np.where(x > 0.0)
    y = gain*x[positive_idxs] + (gain**2)*3.0/8.0 + sigma**2 - gain*mu

    # Clamp to zero before taking the square root.
    y[y < 0.0] = 0.0
    y_stabilized = (2.0/gain)*np.sqrt(y)

    # Replace values in original array and return result
    result[positive_idxs] = y_stabilized
    return result

def inverse_generalized_anscombe(x, mu, sigma, gain=1.0):
    result = x.copy()

    positive_idxs = np.where(x > 0.0)
    y = (1.0/gain)*(gain*x[positive_idxs]/2.0)**2 - gain*3.0/8.0 - (sigma**2)/gain + mu

    # Replace values in original array and return result
    result[positive_idxs] = y
    return result
