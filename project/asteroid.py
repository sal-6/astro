import astro
import numpy as np
from interplanetary import NBodySolarSailGeneralDirection
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from matplotlib.patches import Circle



def NBodySolarSailGeneralDirection(t, state, masses, ss_area, ss_reflectivity, direction_vector_handle):
    """Calculates the state derivative for the NBody problem.
    masses in kg
    state in m and m/s

    Args:
        t (float): Time
        state (arr): State vector with form:
            [r_1_x, r_1_y, r_1_z, v_1_x, v_1_y, v_1_z, r_2_x, r_2_y, r_2_z, v_2_x, v_2_y, v_2_z, ...]

            In order to perform the necessary calculations, the state vector must be given with the suns states listed first,
            followed by the spacecraft states, followed by any additional bodies in the system. It shall be structured as follows:

            [r_sun_x, r_sun_y, r_sun_z, v_sun_x, v_sun_y, v_sun_z, r_sc_x, r_sc_y, r_sc_z, v_sc_x, v_sc_y, v_sc_z, ...]
        masses (arr): Masses of the bodies in the same order as the state vector:
            [m_1, m_2, ...]
        ss_area (float): Area of the solar sail in m^2
        ss_reflectivity (float): Reflectivity of the solar sail
        sun_sc_angle (float): Angle between the sun and the spacecraft in radians
        direction_vector_handle (function): Function that takes in the spacecraft state, suns state, and returns a unit vector. 
            Shall be implemented as: direction_vector_handle(spacecraft_state, sun_state, curr_time)
    
    """
    
    # perform typical n body calculations for each body included in the spacecraft
    state = state.reshape((len(masses), 6))
    state_der = np.zeros((len(masses), 6))
    for i, body in enumerate(state):
        _r_body = np.array([state[i][0], state[i][1], state[i][2]])
        _v_body = np.array([state[i][3], state[i][4], state[i][5]])
        _F = np.array([0., 0., 0.])
        for j, other_body in enumerate(state):
            if i != j:
                _r_other_body = np.array([state[j][0], state[j][1], state[j][2]])
                _v_other_body = np.array([state[j][3], state[j][4], state[j][5]])
                _r_to_body = _r_body - _r_other_body
                force = -astro.G * masses[j] * masses[i] / np.linalg.norm(_r_to_body) ** 3 * _r_to_body
                _F = np.add(_F, force)

        # if the body has the solar sail, then add the force of the solar sail
        if i == 1:
            
            # distance from sun to spacecraft
            _r_sun = np.array([state[0][0], state[0][1], state[0][2]])
            _r_sc = np.array([state[1][0], state[1][1], state[1][2]])
            _r_sun_to_sc = _r_sc - _r_sun
            
            # distance from sun to spacecraft
            distance_from_sun = np.linalg.norm(_r_sun_to_sc)
            
            # sun state
            _sun_state = np.array([state[0][0], state[0][1], state[0][2], state[0][3], state[0][4], state[0][5]])
            
            # sc state
            _sc_state = np.array([state[1][0], state[1][1], state[1][2], state[1][3], state[1][4], state[1][5]])

            # get the direction vector
            _force_dir = direction_vector_handle(_sc_state, _sun_state, t)
            
            # get sun-sc angle
            phi = np.arccos(np.dot(_r_sun_to_sc, _force_dir) / (distance_from_sun * np.linalg.norm(_force_dir)))
            sun_sc_angle = np.pi - phi
            
            # normalize the force direction
            _force_dir = _force_dir / np.linalg.norm(_force_dir)
            
            # calculate the force (via SMAD)
            F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(sun_sc_angle) ** 2 / (distance_from_sun / astro.AU_meters) ** 2
            
            # apply the force
            _F = np.add(_F, F_srp * _force_dir)

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()


def asteroid_miss_distance():
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    earth_init_state = np.array([149597870700, 0, 0,
                              0, 30300, 0])
    
    asteroid_init_state = np.array([149597870700, 0, 0,
                                   0, -30300, 0])                       
                                
    masses = np.array([1.989e30, 5*10**9]) # dimorphos
    
    diameter = 170 * 25  # m
    
    area = np.pi * (diameter / 2) ** 2
    
    reflectivity = 1
    
    earth_sun = np.array([sun_init_state, earth_init_state])
    asteroid_sun = np.array([sun_init_state, asteroid_init_state])
    
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
    
    solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), asteroid_sun.flatten(), args=(masses, area, reflectivity, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    nominal = solve_ivp(astro.NBodyODE, (0, T), earth_sun.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    plt.figure()
    ax = plt.axes()
    
    ax.plot(solver.y[6], solver.y[7], label="Solar Sail")
    # plot initial and final positions
    #ax.plot(solver.y[6][0], solver.y[7][0], 'o', color='green')
    #ax.plot(solver.y[6][-1], solver.y[7][-1], 'x', color='red')
    ax.plot(nominal.y[6], nominal.y[7], label="Earth", linestyle="--")
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Asteroid w/ Solar Sail on Earth trajectory")
    
    circle = Circle((149597870700, 0), 6378000, label="Earth", fill=True, color='blue')
    ax.add_patch(circle)
    circle = Circle((149597870700, 0), 42164000, label="GEO", fill=False, color='red', linestyle="--")
    ax.add_patch(circle)
    circle = Circle((149597870700, 0), 924000000, label="SOI", fill=False, color='green', linestyle="--")
    ax.add_patch(circle)
        
    ax.legend()
    
    # set the aspect ratio
    ax.set_aspect('equal')
    
    plt.show()
    
    
def test_various_diameters():
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    earth_init_state = np.array([149597870700, 0, 0,
                              0, 30300, 0])
    
    asteroid_init_state = np.array([149597870700, 0, 0,
                                   0, -30300, 0])                       
                                
    masses = np.array([1.989e30, 5*10**9]) 
    
    diameter = 170 * 250  # m
    
    area = np.pi * (diameter / 2) ** 2
    area = 100 * 100
    
    reflectivity = 1
    
    earth_sun = np.array([sun_init_state, earth_init_state])
    asteroid_sun = np.array([sun_init_state, asteroid_init_state])
    
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
    
    solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), asteroid_sun.flatten(), args=(masses, area, reflectivity, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    nominal = solve_ivp(astro.NBodyODE, (0, T), earth_sun.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    plt.figure()
    ax = plt.axes()
    
    ax.plot(solver.y[6], solver.y[7], label="Solar Sail")
    # plot initial and final positions
    ax.plot(solver.y[6][0], solver.y[7][0], 'o', color='green')
    ax.plot(solver.y[6][-1], solver.y[7][-1], 'x', color='red')
    ax.plot(nominal.y[6], nominal.y[7], label="Earth", linestyle="--")
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Asteroid w/ Solar Sail on Earth trajectory")
    
    circle = Circle((149597870700, 0), 924000000, label="SOI", fill=False, color='green', linestyle="--")
    ax.add_patch(circle)
    
    ax.legend()
    
    # set the aspect ratio
    ax.set_aspect('equal')
    
    plt.show()

if __name__ == "__main__":
    asteroid_miss_distance()
    
    
    
    
    