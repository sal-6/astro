#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: HW2 assignment problems

import time
import astro
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def num_1():
    bodies = [
        astro.Body(10 ** 24, np.array([2 * 10 ** 6, 0., 0.]), np.array([0., 5000., 0.]), name="Body 1"),
        astro.Body(10 ** 24, np.array([-2 * 10 ** 6, 0., 0.]), np.array([0., -5000., 0.]), name="Body 2"),
        astro.Body(10 ** 24, np.array([4 * 10 ** 6, 0., 0.]), np.array([0., -5000., 3000.]), name="Body 3"),
        astro.Body(10 ** 24, np.array([-4 * 10 ** 6, 0., 0.]), np.array([0., 5000., -3000.]), name="Body 4")
    ]

    sim = astro.NBody(bodies, 10000, .5)
    sim.run()

    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for body in sim.bodies:
        x = [i[0] for i in body._r_history]
        y = [i[1] for i in body._r_history]
        z = [i[2] for i in body._r_history]

        ax.plot(x, y, z, label=body.name)

    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Paths of the Bodies")
    ax.legend(loc="best")

    position_comps_fig, axes = plt.subplots(3, 1)

    # plot the x, y, and z components of the position of the bodies in subplots
    for i, axis in enumerate(['x', 'y', 'z']):
        axes[i].set_title(f'Position of bodies in {axis} direction')
        axes[i].set_xlabel('time (s)')
        axes[i].set_ylabel(f'{axis} (m)')

        for body in sim.bodies:
            axes[i].plot(body.time, [j[i] for j in body._r_history])
    
    plt.tight_layout()

    mom_fig = plt.figure()
    ax = plt.axes()
    ax.plot(bodies[0].time, sim.calculate_angular_momentum_mag())

    plt.show()


def num_1_ode():
    bodies = [
        astro.Body(10 ** 24, np.array([2 * 10 ** 6, 0., 0.]), np.array([0., 5000., 0.]), name="Body 1"),
        astro.Body(10 ** 24, np.array([-2 * 10 ** 6, 0., 0.]), np.array([0., -5000., 0.]), name="Body 2"),
        astro.Body(10 ** 24, np.array([4 * 10 ** 6, 0., 0.]), np.array([0., -5000., 3000.]), name="Body 3"),
        astro.Body(10 ** 24, np.array([-4 * 10 ** 6, 0., 0.]), np.array([0., 5000., -3000.]), name="Body 4")
    ]

    init_state = np.array([body.get_state() for body in bodies])
    masses = [body.mass for body in bodies] 
    tol = 10**-13
    T = 20000
    
    print("Solving with RK45")
    t_start = time.time()
    solver = solve_ivp(astro.NBodyODE, (0, T), init_state.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol)
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(0, len(solver.y), 6):
        x = solver.y[i]
        y = solver.y[i+1]
        z = solver.y[i+2]

        ax.plot(x, y, z, label=f"Body {i//6 + 1}")

    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Paths of the Bodies")
    ax.legend(loc="best")

    fig, axes = plt.subplots(3, 1)
    # plot x, y, and z components of the position of the bodies in subplots
    for i, axis in enumerate(['x', 'y', 'z']):
        axes[i].set_title(f'Position of bodies in {axis} direction')
        axes[i].set_xlabel('time (s)')
        axes[i].set_ylabel(f'{axis} (m)')

        for j in range(i, len(solver.y), 6):
            axes[i].plot(solver.t, solver.y[j], label=f"Body {j//6 + 1}")
            axes[i].legend(loc="best")

    plt.tight_layout()

    # calculate the magnitude of the angular momentum of the system at each time step
    mom = []
    for i in range(len(solver.t)):
        mom_step = 0
        for j in range(0, len(solver.y), 6):
            _r = np.array([solver.y[j][i], solver.y[j+1][i], solver.y[j+2][i]])
            _v = np.array([solver.y[j+3][i], solver.y[j+4][i], solver.y[j+5][i]])
            mom_step += np.linalg.norm(astro.calculate_angular_momentum(_r, _v))
        mom.append(mom_step)

    # calculate deviation on momentum from initial value at each step
    mom_dev = []
    for i in mom:
        mom_dev.append(mom[0] - i)
    
    fig, axes = plt.subplots(2, 1)

    axes[0].plot(solver.t, mom)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Specific Angular Momentum (m^2/s/kg")
    axes[0].set_title("Total Angular Momentum of System Vs. Time")
    axes[1].plot(solver.t, mom_dev)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Specific Angular Momentum (m^2/s/kg")
    axes[1].set_title("Deviation in Total Angular Momentum of System Vs. Time")

    plt.tight_layout()

    # calculate the total energy of the system at each time step.
    # Needs clarification on what the total energy is
    """ energy = []
    for timestep in range(len(solver.t)):
        energy_step = 0
        for i in range(0, len(solver.y), 6):
            _r1 = np.array([solver.y[i][timestep], solver.y[i+1][timestep], solver.y[i+2][timestep]])
            _v1 = np.array([solver.y[i+3][timestep], solver.y[i+4][timestep], solver.y[i+5][timestep]])
            potential_energy = 0
            for j in range(0, len(solver.y), 6):
                if i != j:
                    _r2 = np.array([solver.y[j][timestep], solver.y[j+1][timestep], solver.y[j+2][timestep]])
                    _v2 = np.array([solver.y[j+3][timestep], solver.y[j+4][timestep], solver.y[j+5][timestep]])
                    U = astro.calculate_potential_energy(masses[i], masses[j], _r1, _r2)
                    potential_energy += U
            kinetic_energy = astro.calculate_kinetic_energy(masses[i], _v1)
            total_energy = potential_energy + kinetic_energy
            energy_step += kinetic_energy + potential_energy
        energy.append(energy_step) """

    plt.show()


def num_5():

    final_states = {}
    
    # define ellipse elements
    r_p = 7378 * 1000
    e = 0.5
    i = 0
    raan = 0
    w = 0
    u = 32 # deg 

    a = astro.calculate_semi_major_axis_from_perigee(r_p, e)

    # define t_f
    P = astro.calculate_orbital_period(a)
    t_f = P / 3

    # get inital r and v from orbital elements
    _r, _v = astro.standard_to_cartesian(a, e, i, raan, w, u)
    
    init_state = np.array([_r[0], _r[1], _r[2], _v[0], _v[1], _v[2]])
    tol = 10**-13
    orbit = solve_ivp(astro.propogate_2BP, [0, P], init_state, method="RK45", atol=tol, rtol=tol, t_eval=np.linspace(0, int(P), int(P)+1))

    # define tolerance
    tol = 10 ** -7

    # perform initial guess for x
    x = astro.inital_guess_kepler(t_f, 0, a, _r, _v)
    t_curr = astro.calculate_time(x, a, _r, _v)

    while abs(t_f - t_curr) > tol:
        r = astro.calculate_radius(x, a, _r, _v)
        x = x + (t_f - t_curr) / (r / math.sqrt(astro.MU_EARTH))
        t_curr = astro.calculate_time(x, a, _r, _v)

    _r_f, _v_f = astro.calculate_final_state_kepler(x, a, _r, _v, t_curr)

    pos = []
    vel = []
    acc = []
    for i in range(orbit.t.shape[0]):
        pos.append(math.sqrt(orbit.y[0][i] ** 2 + orbit.y[1][i] ** 2 + orbit.y[2][i] ** 2))
        vel.append(math.sqrt(orbit.y[3][i] ** 2 + orbit.y[4][i] ** 2 + orbit.y[5][i] ** 2))
        a_x = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[0][i]
        a_y = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[1][i]
        a_z = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[2][i]
        acc.append(math.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2))

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(orbit.t, pos)
    ax.set_xlabel('Time (s)', size=10)
    ax.set_ylabel('Position (m)', size=10)
    ax.set_title("Postion Vs Time of Ellipse")

    ax.plot([0], [np.linalg.norm(_r)], "bo")
    ax.plot([t_curr], [np.linalg.norm(_r_f)], "ro")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2])
    ax.plot3D([_r[0]], [_r[1]], [_r[2]], "bo")
    ax.plot3D([_r_f[0]], [_r_f[1]], [_r_f[2]], "ro") 
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Ellipse")

    _e = astro.calculate_eccentricity_vector(_r_f, _v_f)
    u_f = astro.calculate_true_anomoly(_e, _r_f, _v_f)

    final_states["ellipse"] = {
        "_r": _r_f,
        "_v": _v_f,
        "u": np.degrees(u_f)
    }

    # define hyperbola elements
    r_p = 7378 * 1000
    e = 2
    i = 0
    raan = 0
    w = 0
    u = 0 # deg 

    t_f = 1000 # sec

    P = t_f

    a = astro.calculate_semi_major_axis_from_perigee(r_p, e)

    # get inital r and v from orbital elements
    _r, _v = astro.standard_to_cartesian(a, e, i, raan, w, u)

    init_state = np.array([_r[0], _r[1], _r[2], _v[0], _v[1], _v[2]])
    tol = 10**-13
    orbit = solve_ivp(astro.propogate_2BP, [0, P], init_state, method="RK45", atol=tol, rtol=tol, t_eval=np.linspace(0, int(P), int(P)+1))    

    # perform initial guess for x
    x = astro.inital_guess_kepler(t_f, 0, a, _r, _v)
    t_curr = astro.calculate_time(x, a, _r, _v)

    tol = 10**-10
    while abs(t_f - t_curr) > tol:
        r = astro.calculate_radius(x, a, _r, _v)
        x = x + (t_f - t_curr) / (r / math.sqrt(astro.MU_EARTH))
        t_curr = astro.calculate_time(x, a, _r, _v)

    _r_f, _v_f = astro.calculate_final_state_kepler(x, a, _r, _v, t_curr)

    pos = []
    vel = []
    acc = []
    for i in range(orbit.t.shape[0]):
        pos.append(math.sqrt(orbit.y[0][i] ** 2 + orbit.y[1][i] ** 2 + orbit.y[2][i] ** 2))
        vel.append(math.sqrt(orbit.y[3][i] ** 2 + orbit.y[4][i] ** 2 + orbit.y[5][i] ** 2))
        a_x = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[0][i]
        a_y = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[1][i]
        a_z = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[2][i]
        acc.append(math.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2))

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(orbit.t, pos)
    ax.set_xlabel('Time (s)', size=10)
    ax.set_ylabel('Position (m)', size=10)
    ax.set_title("Postion Vs Time of Hyperbola")

    ax.plot([0], [np.linalg.norm(_r)], "bo")
    ax.plot([t_curr], [np.linalg.norm(_r_f)], "ro")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2])
    ax.plot3D([_r[0]], [_r[1]], [_r[2]], "bo")
    ax.plot3D([_r_f[0]], [_r_f[1]], [_r_f[2]], "ro")
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Hyperbola")

    _e = astro.calculate_eccentricity_vector(_r_f, _v_f)
    u_f = astro.calculate_true_anomoly(_e, _r_f, _v_f)

    final_states["hyperbola"] = {
        "_r": _r_f,
        "_v": _v_f,
        "u": np.degrees(u_f)
    }

    print(final_states["ellipse"])
    print(final_states["hyperbola"])

    plt.show()


if __name__ == "__main__":
    num_1_ode()