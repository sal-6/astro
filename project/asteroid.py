import astro
import numpy as np
from interplanetary import NBodySolarSailGeneralDirection
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from matplotlib.patches import Circle


def asteroid_miss_distance():
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    sc_init_state = np.array([149597870700, 0, 0,
                              0, 30300, 0])
                                
                                
    masses = np.array([1.989e30, 5*10**9])
    
    diameter = 170 # m
    
    area = np.pi * (diameter / 2) ** 2
    
    reflectivity = 1
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 5
    
    def apply_at_45_degree_to_sun(spacecraft_state, sun_state, curr_time):
        _r_sun = sun_state[0:3]
        _v_sun = sun_state[3:6]
        
        _r_sc = spacecraft_state[0:3]
        _v_sc = spacecraft_state[3:6]
        
        _r_sun_to_sc = _r_sc - _r_sun
        
        _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
        
        _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _momentum_dir, -np.pi/4)

        # normalize the force direction
        _force_dir = _force_dir / np.linalg.norm(_force_dir)
        
        return _force_dir
        
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, AREA_SS, REFLECTIVITY, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    #nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    plt.figure()
    ax = plt.axes()
    
    
    
    