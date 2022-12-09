import astro
import numpy as np
from interplanetary import NBodySolarSailGeneralDirection
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def inclination_change(mass_sc=50, ss_area=14*14, reflectivity=1):
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) 
    
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
    
    
    sc_init_state = np.array([149597870700, 0, 0,
                              0, 30300, 0])
                                
                                
    masses = np.array([1.989e30, mass_sc])
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 15
    
    def apply_cranking_manoeuvre(spacecraft_state, sun_state, t):
        
        _r_sun = sun_state[0:3]
        _v_sun = sun_state[3:6]
        
        _r_sc = spacecraft_state[0:3]
        _v_sc = spacecraft_state[3:6]
        
        _r_sun_to_sc = _r_sc - _r_sun
        
        _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
        
        _sun_tangent = np.cross(_momentum_dir, _r_sun_to_sc)
        
        
        # if the spacecraft is in the ecliptic plane (z = 0) or it is above the ecliptic plane (z > 0)
        # apply the force orthogonal spacecraft velocity and radial vector from the sun is maximum
        if spacecraft_state[2] >= 0:
            _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _sun_tangent, - np.pi / 4)
            
        else:
            _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _sun_tangent, np.pi / 4)


        # get angle between the force direction and the velocity vector of the spacecraft
        _angle = np.arccos(np.dot(_force_dir, _v_sc) / (np.linalg.norm(_force_dir) * np.linalg.norm(_v_sc)))
        #print(f"Angle: {np.rad2deg(_angle)}")
        return np.linalg.norm(_force_dir)
        
    def apply_cranking_manoeuvre2(spacecraft_state, sun_state, t):
        
        _r_sun = sun_state[0:3]
        _v_sun = sun_state[3:6]
        
        _r_sc = spacecraft_state[0:3]
        _v_sc = spacecraft_state[3:6]
        
        _r_sun_to_sc = _r_sc - _r_sun
        
        _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
        _sun_tangent = np.cross(_momentum_dir, _r_sun_to_sc)
        
        _out = np.cross(_v_sc, _momentum_dir)
        
        
        # if the spacecraft is in the ecliptic plane (z = 0) or it is above the ecliptic plane (z > 0)
        # apply the force orthogonal spacecraft velocity and radial vector from the sun is maximum
        if spacecraft_state[2] >= 0:
            _force_dir = astro.rodrigues_rotation_formula(_momentum_dir, _v_sc,  np.pi / 4)
            
        else:
            _force_dir = astro.rodrigues_rotation_formula(_momentum_dir, _v_sc,  np.pi / 4)
            
            

        # get angle between the force direction and the velocity vector of the spacecraft
        _angle = np.arccos(np.dot(_force_dir, _v_sc) / (np.linalg.norm(_force_dir) * np.linalg.norm(_v_sc)))
        #print(f"Angle: {np.rad2deg(_angle)}")
        return np.linalg.norm(_force_dir)
        
        
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, ss_area, reflectivity, apply_cranking_manoeuvre2), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    #nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the results in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(solver.y[6], solver.y[7], solver.y[8], linewidth=1, label="Trajectory")
    ax.set_xlabel('x (DU)', size=10)
    ax.set_ylabel('y (DU)', size=10)
    ax.set_zlabel('z (DU)', size=10)
    ax.set_title("Trajectory of Spacecraft")
    
    # plot sun at 0 0 0
    ax.scatter3D(0, 0, 0, color='orange', label="Sun")
    
    plt.show()
    

if __name__ == "__main__":
    inclination_change()