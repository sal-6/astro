import astro
import numpy as np
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def clohessy_wiltshire(x0, y0, z0, dx0, dy0, dz0, w, t):
    
    x = 4*x0 + 2*dy0/w + dx0/w * np.sin(w*t) - (2*dy0/w + 3*x0) * np.cos(w*t)
    y = 2*dx0/w * np.cos(w*t) + (4*dy0/w + 6*x0) * np.sin(w*t) + (-6*w*x0 - 3*dy0) * t + y0 - 2*dx0/w
    z = z0 * np.cos(w*t) + dz0/w * np.sin(w*t)
    
    
    return x, y, z
    
## are those phis or Q's ?!?! 
def docking(r0, v0, w, t):
    
    Qrr = np.array([[4-3*np.cos(w*t), 0, 0],
                    [6*(np.sin(w*t) - w*t), 1, 0],
                    [0, 0, np.cos(w*t)]])
                    
    Qrv = np.array([[np.sin(w*t)/w, (2/w) * (1 - np.cos(w*t)), 0],
                    [(-2/w) * (1 - np.cos(w*t)), (4/w) * np.sin(w*t) - 3*t, 0],
                    [0, 0, np.sin(w*t)/w]])
                    
    Qvr = np.array([[3*w*np.sin(w*t), 0, 0],
                    [6*w*(np.cos(w*t) - 1), 0, 0],
                    [0, 0, -w*np.sin(w*t)]])
                    
    Qvv = np.array([[np.cos(w*t), 2*np.sin(w*t), 0],
                    [-2*np.sin(w*t), 4*np.cos(w*t) - 3, 0],
                    [0, 0, np.cos(w*t)]])
    
    _v_t0 = - np.linalg.inv(Qrv) @ Qrr @ r0
    _v_t2 = (Qvr - Qvv @ np.linalg.inv(Qrv) @ Qrr) @ r0
    
    _delta_V1 = _v_t0 - v0
    _delta_V2 = - _v_t2
    
    return _delta_V1, _delta_V2
    
    