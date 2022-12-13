import astro
import numpy as np
#from interplanetary import NBodySolarSailGeneralDirection
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


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
            sun_sc_angle = np.pi / 2 - phi
            
            # normalize the force direction
            _force_dir = _force_dir / np.linalg.norm(_force_dir)
            
            # calculate the force (via SMAD)
            F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(sun_sc_angle) ** 2 / (distance_from_sun / astro.AU_meters) ** 2
            
            durr_in = t % (365 * 24 * 60 * 60)
                    
            mult = np.cos(2 * np.pi * durr_in / (365 * 24 * 60 * 60))
            
            # apply the force
            _F = np.add(_F, F_srp * mult* _force_dir)
            
            

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()


def inclination_change(mass_sc=50, ss_area=14*14, reflectivity=1):
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) 
    
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
    
    
    sc_init_state = np.array([149597870700, 0, 0,
                              0, 30300, 0]) # 1 AU circular
                                
    """ sc_init_state = np.array([5.9e11, 0, 0,
                              0, 29780, 0]) # .5 AU circular """
                    
                                
    masses = np.array([1.989e30, mass_sc])
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 15     
    
        
        
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
    
    axisEqual3D(ax)
    
    plt.show()
    
    
    
def inclination_change_1AU():    
    sc_mass = 10
    area = 14*14
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    
    sc_init_state = np.array([149597870700, 0, 0,
                              0, 30300, 1]) # 1 AU circular
                              
    
                              
    masses = np.array([1.989e30, sc_mass])
    
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 6
    
            
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(inclinationTesting, (0, T), bodies.flatten(), args=(masses, area, 1, np.pi/4), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, z, linewidth=0.5, label="Solar Sail")
    
    # plot the spacecraft nominal
    x = nominal.y[6]
    y = nominal.y[7]
    z = nominal.y[8]
    ax.plot(x, y, z, linewidth=0.5, label="Nominal")
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, 0, color='red')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Inclination Change at 1 AU (T = 6 years)")
    ax.legend(loc="best")
    
    axisEqual3D(ax)
    
    # calculate standard orbital parameters
    incs = []
    for i in range(len(solver.y[0])):
        _r = np.array([solver.y[0][i], solver.y[1][i], solver.y[2][i]])
        _v = np.array([solver.y[3][i], solver.y[4][i], solver.y[5][i]])
        params = astro.cartesian_to_standard(_r, _v, mu=astro.MU_SUN)
        
        incs.append(params[2])
        
    return incs, solver.t
    
    #plt.show()


def inclination_change_halfAU():
    sc_mass = 10
    area = 14*14
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
    
    sc_init_state = np.array([5.9e11, 0, 0,
                              0, 29780, 0]) # .5 au
                              
    sc_init_state = np.array([7.48e10, 0, 0,
                              0, 42120, 1]) # .5 au
                              
    masses = np.array([1.989e30, sc_mass])
    
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 6
    
            
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(inclinationTestingHalf, (0, T), bodies.flatten(), args=(masses, area, 1, np.pi/4), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, z, linewidth=0.5, label="Solar Sail")
    
    # plot the spacecraft nominal
    x = nominal.y[6]
    y = nominal.y[7]
    z = nominal.y[8]
    ax.plot(x, y, z, linewidth=0.5, label="Nominal")
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, 0, color='red')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Inclination Change at 0.5 AU (T = 6 years)")

    ax.legend(loc="best")
    
    axisEqual3D(ax)
    
    # calculate standard orbital parameters
    incs = []
    for i in range(len(solver.y[0])):
        _r = np.array([solver.y[0][i], solver.y[1][i], solver.y[2][i]])
        _v = np.array([solver.y[3][i], solver.y[4][i], solver.y[5][i]])
        params = astro.cartesian_to_standard(_r, _v, mu=astro.MU_SUN)
        
        incs.append(params[2])
        
    return incs, solver.t
    
def inclinationTesting(t, state, masses, ss_area, ss_reflectivity, sun_sc_angle):
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
            Shall be implemented as: direction_vector_handle(spacecraft_state, sun_state)
    
    """
    
    # perform typical n body calculations for each body included in the spacecraft
    #print(t / (365 * 24 * 60 * 60 * 10))
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
                    # calculate r cross v
                    _r_cross_v = np.cross(_r_sun_to_sc, _v_body)
                    # magnitude 
                    _r_cross_v_mag = np.linalg.norm(_r_cross_v)
                    # normalize
                    _force_dir = _r_cross_v / _r_cross_v_mag
                    
                    # calculate the force (via SMAD)
                    F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(sun_sc_angle) ** 2 / (distance_from_sun / astro.AU_meters) ** 2
                    
                    durr_in = t % (31559675.2)
                    
                    mult = np.cos(2 * np.pi * durr_in / (31559675.2))
                    
                    # apply the force
                    _F = np.add(_F, F_srp * mult * _force_dir)

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()


def inclinationTestingHalf(t, state, masses, ss_area, ss_reflectivity, sun_sc_angle):
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
            Shall be implemented as: direction_vector_handle(spacecraft_state, sun_state)
    
    """
    
    # perform typical n body calculations for each body included in the spacecraft
    #print(t / (365 * 24 * 60 * 60 * 10))
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
                    # calculate r cross v
                    _r_cross_v = np.cross(_r_sun_to_sc, _v_body)
                    # magnitude 
                    _r_cross_v_mag = np.linalg.norm(_r_cross_v)
                    # normalize
                    _force_dir = _r_cross_v / _r_cross_v_mag
                    
                    # calculate the force (via SMAD)
                    F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(sun_sc_angle) ** 2 / (distance_from_sun / astro.AU_meters) ** 2
                    
                    durr_in = t % (11158268.4)
                    
                    mult = np.cos(2 * np.pi * durr_in / (11158268.4))
                    
                    # apply the force
                    _F = np.add(_F, F_srp * mult * _force_dir)

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()

# To help with plotting
# https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":
    #inclination_change()
    inc1, t1 = inclination_change_1AU()
    inc2, t2 = inclination_change_halfAU()
    fig = plt.figure()
    ax = plt.axes()
    
    ax.plot(t1, inc1, label="1 AU")
    ax.plot(t2, inc2, label="0.5 AU")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inclination (deg)")
    ax.set_title("Inclination Change Over Time")
    ax.legend()
    
    plt.show()
