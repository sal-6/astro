#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: HW2 assignment problems

import astro
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def num_1():
    pass


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
    ax.set_title("Postion Vs Time of Orbit")

    ax.plot([0, t_curr], [np.linalg.norm(_r), np.linalg.norm(_r_f)], "ro")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2]) 
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path")

    final_states["ellipse"] = {
        "_r": _r_f,
        "_v": _v_f
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
    print("gere")
    

    # perform initial guess for x
    x = astro.inital_guess_kepler(t_f, 0, a, _r, _v)
    t_curr = astro.calculate_time(x, a, _r, _v)

    tol = 10**-10
    while abs(t_f - t_curr) > tol:
        print(t_f - t_curr)
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
    ax.set_title("Postion Vs Time of Orbit")

    ax.plot([0, t_curr], [np.linalg.norm(_r), np.linalg.norm(_r_f)], "ro")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2]) 
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path")

    final_states["hyperbola"] = {
        "_r": _r_f,
        "_v": _v_f
    }

    plt.show()


if __name__ == "__main__":
    num_5()