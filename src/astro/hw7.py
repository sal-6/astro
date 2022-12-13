import astro
import numpy as np
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def clohessy_wiltshire(x0, y0, z0, dx0, dy0, dz0, w, t):
    
    print(w)
    
    x = 4*x0 + 2*dy0/w + dx0/w * np.sin(w*t) - (2*dy0/w + 3*x0) * np.cos(w*t)
    y = 2*dx0/w * np.cos(w*t) + (4*dy0/w + 6*x0) * np.sin(w*t) + (-6*w*x0 - 3*dy0) * t + y0 - 2*dx0/w
    z = z0 * np.cos(w*t) + dz0/w * np.sin(w*t)
    
    
    return x, y, z