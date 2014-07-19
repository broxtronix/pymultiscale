
#-------- Variance stabilizing transforms -----------

def anscombe(x):
    return 2.0*np.sqrt(x + 3.0/8.0)

def inverse_anscombe(z):
    return (z/2.0)**2 - 3.0/8.0

def generalized_anscombe(x,mu,sigma,gain=1.0):
    return (2.0/gain)*np.sqrt(gain*x + (gain**2)*3.0/8.0 + sigma**2 - gain*mu)

def inverse_generalized_anscombe(z,mu,sigma,gain=1.0):
    return (1.0/gain)*(gain*y/2.0)**2 - gain*3.0/8.0 - (sigma**2)/gain + mu
