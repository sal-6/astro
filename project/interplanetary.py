import astro
import numpy as np


import astro
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.integrate import solve_ivp

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




def nominal_vs_solarsail_out():
    
    MASS_SC = 10 # kg
    AREA_SS = 14 * 14 # m^2
    REFLECTIVITY = 1
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
    #sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
    #                         -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
    
    # earth orbit with no z components
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
    
    masses = np.array([1.989e30, MASS_SC])
    
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 200
    
    def apply_at_45_degree_to_sun(spacecraft_state, sun_state, curr_time):
        _r_sun = sun_state[0:3]
        _v_sun = sun_state[3:6]
        
        _r_sc = spacecraft_state[0:3]
        _v_sc = spacecraft_state[3:6]
        
        _r_sun_to_sc = _r_sc - _r_sun
        
        _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
        
        _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _momentum_dir, np.pi/4)

        # normalize the force direction
        _force_dir = _force_dir / np.linalg.norm(_force_dir)
        
        return _force_dir
            
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, AREA_SS, REFLECTIVITY, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
    #nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, linewidth=.5, label="Spacecraft", color='black')
    
    # plot the spacecraft nominal
    """ x = nominal.y[6]
    y = nominal.y[7]
    z = nominal.y[8]
    ax.plot(x, y, linewidth=1, label="Earth Orbit", linestyle="--", color='blue') """
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, color='orange')
    
    # plot point at initial position of spacecraft
    ax.scatter(sc_init_state[0], sc_init_state[1], color='green')
    
    
    # plot planets
    circle = Circle((0, 0), 230892583680, label="Mars", fill=False, color='red', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 149600000000, label="Earth", fill=False, color='blue', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 63062144640, label="Mercury", fill=False, color='green', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 1.088512e11, label="Venus", fill=False, color='yellow', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 740829323520, label="Jupiter", fill=False, color='purple', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 1.470280585e12, label="Saturn", fill=False, color='brown', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 2.871e12, label="Uranus", fill=False, color='pink', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 4.495e12, label="Neptune", fill=False, color='gray', linestyle="--")
    ax.add_patch(circle)
    
    
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    #ax.set_zlabel('z (m)')
    
    # set aspect ratio to 1
    ax.set_aspect('equal')
    ax.set_title("Trajectory of the Spacecraft")

    ax.legend(loc="best")
    
    #axisEqual3D(ax)
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, linewidth=.5, label="Spacecraft", color='black')
    
    # plot planets
    circle = Circle((0, 0), 230892583680, label="Mars", fill=False, color='red', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 149600000000, label="Earth", fill=False, color='blue', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 63062144640, label="Mercury", fill=False, color='green', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 1.088512e11, label="Venus", fill=False, color='yellow', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 740829323520, label="Jupiter", fill=False, color='purple', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 1.470280585e12, label="Saturn", fill=False, color='brown', linestyle="--")
    ax.add_patch(circle)
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, color='orange')
    
    # plot point at initial position of spacecraft
    ax.scatter(sc_init_state[0], sc_init_state[1], color='green')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    #ax.set_zlabel('z (m)')
    
    # set aspect ratio to 1
    ax.set_aspect('equal')
    ax.set_title("Trajectory of the Spacecraft")
    ax.legend(loc="best")
    
    # calculate at what time the spacecraft reaches a radius of 230892583680
    
    for i in range(len(solver.t)):
        if np.linalg.norm(solver.y[6:9, i]) >= 230892583680:
            time_to_mars = solver.t[i]
            print("The spacecraft reaches Mars at time t = ", solver.t[i] / 60 / 60 / 24, " days")
            break
        
    energy_deviation = []
    # use astro.calculate_kinetic_energy to calculate the energy deviation passing it the position and velocity vectors
    init_energy = astro.calculate_orbital_energy(np.linalg.norm(solver.y[6:9, 0]), np.linalg.norm(solver.y[9:12, 0]), mu=1.32712440018e20)
    print(init_energy)
    for i in range(len(solver.t)):
        energy_deviation.append(astro.calculate_orbital_energy(np.linalg.norm(solver.y[6:9, i]), np.linalg.norm(solver.y[9:12, i]), mu=1.32712440018e20) - init_energy)
    
    # plot the energy deviation
    plt.figure()
    plt.plot(solver.t, energy_deviation)
    plt.xlabel("Time (s)")
    plt.ylabel("Specific Energy Deviation (J/kg)")
    plt.title("Specific Energy Deviation of the Spacecraft")
    
    plt.show()
    
    

def nominal_vs_solarsail_in():
    
    MASS_SC = 10 # kg
    AREA_SS = 14 * 14 # m^2
    REFLECTIVITY = 1
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
    #sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
    #                         -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
    
    # earth orbit with no z components
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
    
    masses = np.array([1.989e30, MASS_SC])
    
    
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
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, linewidth=.5, label="Spacecraft", color='black')
    
    # plot the spacecraft nominal
    """ x = nominal.y[6]
    y = nominal.y[7]
    z = nominal.y[8]
    ax.plot(x, y, linewidth=1, label="Earth Orbit", linestyle="--", color='blue') """
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, color='orange')
    
    # plot point at initial position of spacecraft
    ax.scatter(sc_init_state[0], sc_init_state[1], color='green')
    
    
    # plot planets
    circle = Circle((0, 0), 230892583680, label="Mars", fill=False, color='red', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 149600000000, label="Earth", fill=False, color='blue', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 1.088512e11, label="Venus", fill=False, color='yellow', linestyle="--")
    ax.add_patch(circle)
    
    circle = Circle((0, 0), 63062144640, label="Mercury", fill=False, color='green', linestyle="--")
    ax.add_patch(circle)
    

    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    #ax.set_zlabel('z (m)')
    
    # set aspect ratio to 1
    ax.set_aspect('equal')
    ax.set_title("Trajectory of the Spacecraft")

    ax.legend(loc="best")
    
    #axisEqual3D(ax)
    
    # calculate at what time the spacecraft reaches a radius of 63062144640
    
    for i in range(len(solver.t)):
        if np.linalg.norm(solver.y[6:9, i]) <= 63062144640:
            time_to_mars = solver.t[i]
            print("The spacecraft reaches Mercury at time t = ", solver.t[i] / 60 / 60 / 24, " days")
            break
        
    # calculate at what time the spacecraft reaches a radius of 1.088512e11
    
    for i in range(len(solver.t)):
        if np.linalg.norm(solver.y[6:9, i]) <= 1.088512e11:
            time_to_venus = solver.t[i]
            print("The spacecraft reaches Venus at time t = ", solver.t[i] / 60 / 60 / 24, " days")
            break
        
    energy_deviation = []
    # use astro.calculate_kinetic_energy to calculate the energy deviation passing it the position and velocity vectors
    init_energy = astro.calculate_orbital_energy(np.linalg.norm(solver.y[6:9, 0]), np.linalg.norm(solver.y[9:12, 0]), mu=1.32712440018e20)
    print(init_energy)
    for i in range(len(solver.t)):
        energy_deviation.append(astro.calculate_orbital_energy(np.linalg.norm(solver.y[6:9, i]), np.linalg.norm(solver.y[9:12, i]), mu=1.32712440018e20) - init_energy)
    
    # plot the energy deviation
    plt.figure()
    plt.plot(solver.t, energy_deviation)
    plt.xlabel("Time (s)")
    plt.ylabel("Specific Energy Deviation (J/kg)")
    plt.title("Specific Energy Deviation of the Spacecraft")
    

    plt.show()
    
def mass_sensitivity_analysis():
    
    masses = [.1, 1, 10, 30, 50, 100, 500]
    AREA_SS = 14 * 14 # m^2
    REFLECTIVITY = 1
    
    mass_vs_distance = []
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    
    for mass in masses:
        sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
        #sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
        #                         -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
        
        # earth orbit with no z components
        sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                                -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
        
        masses = np.array([1.989e30, mass])
        
        
        bodies = np.array([sun_init_state, sc_init_state])
        
        tol = 1e-13
        T = 365 * 24 * 60 * 60 * 20 # 20 years
        
        def apply_at_45_degree_to_sun(spacecraft_state, sun_state, curr_time):
            _r_sun = sun_state[0:3]
            _v_sun = sun_state[3:6]
            
            _r_sc = spacecraft_state[0:3]
            _v_sc = spacecraft_state[3:6]
            
            _r_sun_to_sc = _r_sc - _r_sun
            
            _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
            
            _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _momentum_dir, np.pi/4)

            # normalize the force direction
            _force_dir = _force_dir / np.linalg.norm(_force_dir)
            
            return _force_dir
                
        print("Solving with RK45")
        t_start = time.time()
        
        solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, AREA_SS, REFLECTIVITY, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
        #nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
        
        print(f"Time to Propgate: {time.time() - t_start}")
        
        
        
        # plot the spacecraft
        x = solver.y[6]
        y = solver.y[7]
        z = solver.y[8]
        ax.plot(x, y, linewidth=.5, label=f"{mass} kg")
        
        mass_vs_distance.append([mass, np.linalg.norm(solver.y[6:9, -1])])
        
    # plot a point at 0 0 0
    ax.scatter(0, 0, color='orange')
    
    # plot point at initial position of spacecraft
    ax.scatter(sc_init_state[0], sc_init_state[1], color='green')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # set aspect ratio to 1
    ax.set_aspect('equal')
    ax.set_title("Trajectory of the Spacecraft for Various Masses")
    plt.legend(loc="best")
    
    
    fig = plt.figure()
    ax = plt.axes()
    
    
    ax.plot([mass_vs_distance[i][0] for i in range(len(mass_vs_distance))], [mass_vs_distance[i][1] for i in range(len(mass_vs_distance))])
    ax.set_xlabel("Mass (kg)")
    ax.set_ylabel("Distance from Sun (m)")
    ax.set_title("Distance from Sun vs Mass (t = 20 years)")
    
    # make the x axis log scale
    ax.set_xscale('log')
    
    plt.show()
    
    
def solar_sail_area_sensitivity_analysis():
    
    AREA_SS = 14 * 14 # m^2
    REFLECTIVITY = 1
    
    areas = [2*2, 4*4, 6*6, 8*8, 10*10, 12*12, 14*14, 16*16, 18*18, 20*20, 50*50]
    #areas = [2*2, 4*4, 6*6] #, 8*8, 10*10, 12*12, 14*14, 16*16, 18*18, 20*20, 50*50]
    mass = 10 # kg
    
    area_vs_distance = []
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    
    for area in areas:
        sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
        #sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
        #                         -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
        
        # earth orbit with no z components
        sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                                -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
        
        masses = np.array([1.989e30, mass])
        
        
        bodies = np.array([sun_init_state, sc_init_state])
        
        tol = 1e-13
        T = 365 * 24 * 60 * 60 * 20 # 20 years
        
        def apply_at_45_degree_to_sun(spacecraft_state, sun_state, curr_time):
            _r_sun = sun_state[0:3]
            _v_sun = sun_state[3:6]
            
            _r_sc = spacecraft_state[0:3]
            _v_sc = spacecraft_state[3:6]
            
            _r_sun_to_sc = _r_sc - _r_sun
            
            _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
            
            _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _momentum_dir, np.pi/4)

            # normalize the force direction
            _force_dir = _force_dir / np.linalg.norm(_force_dir)
            
            return _force_dir
                
        print("Solving with RK45")
        t_start = time.time()
        
        solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, area, REFLECTIVITY, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
        #nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
        
        print(f"Time to Propgate: {time.time() - t_start}")
        
        
        
        # plot the spacecraft
        x = solver.y[6]
        y = solver.y[7]
        z = solver.y[8]
        p = ax.plot(x, y, linewidth=.5, label=f"{area} m^2")
        
        # plot a red x at the final position of the spacecraft
        ax.scatter(x[-1], y[-1], color=p[0].get_color(), marker='x')
        
        area_vs_distance.append([area, np.linalg.norm(solver.y[6:9, -1])])
        
    # plot a point at 0 0 0
    ax.scatter(0, 0, color='orange')
    
    # plot point at initial position of spacecraft
    ax.scatter(sc_init_state[0], sc_init_state[1], color='green')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # set aspect ratio to 1
    ax.set_aspect('equal')
    ax.set_title("Trajectory of the Spacecraft for Various Areas (t = 20 years)")
    plt.legend(loc="best")
    
    
    fig = plt.figure()
    ax = plt.axes()
    
    
    ax.plot([area_vs_distance[i][0] for i in range(len(area_vs_distance))], [area_vs_distance[i][1] for i in range(len(area_vs_distance))])
    ax.set_xlabel("Mass (kg)")
    ax.set_ylabel("Distance from Sun (m)")
    ax.set_title("Area vs Distance from Sun (t = 20 years)")
    
    # make the x axis log scale
    ax.set_xscale('log')
    
    plt.show()
    

def reflectivity_sensitivity_analysis():
    
    reflectivities = [.2, .4, .6, .8, 1]
    #areas = [2*2, 4*4, 6*6] #, 8*8, 10*10, 12*12, 14*14, 16*16, 18*18, 20*20, 50*50]
    mass = 10 # kg
    area = 14*14 # m^2
    
    ref_vs_distance = []
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes()
    
    for ref in reflectivities:
        sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
        #sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
        #                         -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
        
        # earth orbit with no z components
        sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, 0,
                                -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, 0])
        
        masses = np.array([1.989e30, mass])
        
        bodies = np.array([sun_init_state, sc_init_state])
        
        tol = 1e-13
        T = 365 * 24 * 60 * 60 * 20 # 20 years
        
        def apply_at_45_degree_to_sun(spacecraft_state, sun_state, curr_time):
            _r_sun = sun_state[0:3]
            _v_sun = sun_state[3:6]
            
            _r_sc = spacecraft_state[0:3]
            _v_sc = spacecraft_state[3:6]
            
            _r_sun_to_sc = _r_sc - _r_sun
            
            _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
            
            _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _momentum_dir, np.pi/4)

            # normalize the force direction
            _force_dir = _force_dir / np.linalg.norm(_force_dir)
            
            return _force_dir
                
        print("Solving with RK45")
        t_start = time.time()
        
        solver = solve_ivp(NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, area, ref, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 10000))
        #nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
        
        print(f"Time to Propgate: {time.time() - t_start}")
        
        # plot the spacecraft
        x = solver.y[6]
        y = solver.y[7]
        z = solver.y[8]
        p = ax.plot(x, y, linewidth=.5, label=f"{area} m^2")
        
        # plot a red x at the final position of the spacecraft
        ax.scatter(x[-1], y[-1], color=p[0].get_color(), marker='x')
        
        ref_vs_distance.append([ref, np.linalg.norm(solver.y[6:9, -1])])
        
    # plot a point at 0 0 0
    ax.scatter(0, 0, color='orange')
    
    # plot point at initial position of spacecraft
    ax.scatter(sc_init_state[0], sc_init_state[1], color='green')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # set aspect ratio to 1
    ax.set_aspect('equal')
    ax.set_title("Trajectory of the Spacecraft for Various Reflectivities (t = 20 years)")
    plt.legend(loc="best")
    
    fig = plt.figure()
    ax = plt.axes()
    
    ax.plot([ref_vs_distance[i][0] for i in range(len(ref_vs_distance))], [ref_vs_distance[i][1] for i in range(len(ref_vs_distance))])
    ax.set_xlabel("Reflectance (kg)")
    ax.set_ylabel("Distance from Sun (m)")
    ax.set_title("Reflectance vs Distance from Sun (t = 20 years)")
    
    # set x axis from 0 to 1
    ax.set_xlim(0, 1)
    
    plt.show()
    
    
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
    #nominal_vs_solarsail_out()
    nominal_vs_solarsail_in()
    
    #mass_sensitivity_analysis()
    #solar_sail_area_sensitivity_analysis()
    #reflectivity_sensitivity_analysis()