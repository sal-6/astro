#############################################################################
# Sal Aslam
# ENAE691 - UMD
# Description: HW5 Orbital Mechanics assignment problem

import astro
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import time
import math

# Part 2 - Consrtuct an acceleration function for the given perturbations
def two_body_accel(state, mu=astro.MU_EARTH):
    """Compute the two-body acceleration

    Args:
        state (np.array): state vector

    Returns:
        np.array: acceleration vector
    """
    
    # Unpack the state vector
    _r = state[0:3]
    _v = state[3:6]
    
    # Two-body acceleration
    _a_two_body = - mu / (np.linalg.norm(_r) ** 3) * _r
    
    return _a_two_body


def j2_accel(state, j2_term=0.0010826269, radius_body=astro.RADIUS_EARTH, mu=astro.MU_EARTH):
    _r = state[0:3]
    _v = state[3:6]
    
    r = np.linalg.norm(_r)
    r_x = _r[0]
    r_y = _r[1]
    r_z = _r[2]
    
    a_x = -3 * j2_term * mu * radius_body ** 2 * r_x / (2 * r ** 5) * (1 - 5 * r_z ** 2 / r ** 2)
    a_y = -3 * j2_term * mu * radius_body ** 2 * r_y / (2 * r ** 5) * (1 - 5 * r_z ** 2 / r ** 2)
    a_z = -3 * j2_term * mu * radius_body ** 2 * r_z / (2 * r ** 5) * (3 - 5 * r_z ** 2 / r ** 2)
    
    return np.array([a_x, a_y, a_z])


# state in m and m/s
def atmospheric_drag_accel(state, density, c_d, A, mass):
    earth_rate = 7.9292115 * 10 ** -5
    
    _v_rel = np.array([
        state[3] / 1000 - earth_rate * state[1] / 1000,
        state[4] / 1000 + earth_rate * state[0] / 1000,
        state[5] / 1000
    ])
    
    v_rel = np.linalg.norm(_v_rel)
    
    _a_x = -0.5 * density * c_d * A * v_rel * _v_rel[0] / mass
    _a_y = -0.5 * density * c_d * A * v_rel * _v_rel[1] / mass
    _a_z = -0.5 * density * c_d * A * v_rel * _v_rel[2] / mass
    
    _a = np.array([_a_x * 1000, _a_y * 1000, _a_z * 1000])
    return _a 


def propogate_with_perturbations(t, y, mu=astro.MU_EARTH):
    """Propogate the orbit of a satellite with the given perturbations

    Args:
        t (float): time
        y (np.array): state vector
        mu (float): gravitational parameter

    Returns:
        np.array: derivative of state vector
    """
    
    # Unpack the state vector
    _r = y[0:3]
    _v = y[3:6]
    
    _a_two_body = two_body_accel(y, mu)
    _a_j2 = j2_accel(y)
    _a_drag = atmospheric_drag_accel(y, 5*10**-3, 2.2, 10*10**-6, 1000)
    _a = _a_two_body + _a_drag + _a_j2
    #print(_a - _a_two_body)
    
    return np.array([_v[0], _v[1], _v[2], _a[0], _a[1], _a[2]])
    

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


def delta_v(r1, r2, mu=astro.MU_EARTH):
    
    del_v_1 = math.sqrt(mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
    del_v_2 = math.sqrt(mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
    
    return del_v_1 + del_v_2


def main():
    a = ((600 * 10 ** 3) + astro.RADIUS_EARTH)
    e = 0
    i = 98
    raan = 0
    w = 0
    u = 0
    
    # Part 1 - Compute the initial cartesian state vectors
    _r, _v = astro.standard_to_cartesian(a, e, i, raan, w, u)

    print(f"Initial position vector (m): {_r}")
    print(f"Initial velocity vector (m/s): {_v}")
    
    initial_state = np.concatenate((_r, _v))    
    
    num_days = 20
    period = astro.calculate_orbital_period(a)
    num_periods = num_days * 24 * 60 * 60 / period
    T = period * num_periods
    
    a = time.time()
    orbit = solve_ivp(propogate_with_perturbations, [0, T], initial_state, method="RK45", atol=10**-13, rtol=10**-13, t_eval=np.linspace(0, T, 1000000))
    print(f"Propogation took: {time.time() - a} seconds.")
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2]) 
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Orbit A")
    axisEqual3D(ax)
    
    rad_dev = [0]
    for i in range(1, len(orbit.y[0])):
        rad_dev.append(np.linalg.norm(orbit.y[0:3, i]) - np.linalg.norm(orbit.y[0:3, 0]))
    
    # plot of radius vs time
    fig = plt.figure()
    ax = plt.axes()
    
    ax.plot(orbit.t, rad_dev)
    ax.set_xlabel('Time (s)', size=10)
    ax.set_ylabel('Radius (m)', size=10)
    ax.set_title("Radius vs Time of Orbit")
    
    # plot a horizontal line at -10000
    ax.axhline(y=-10000, color='r', linestyle='-')
    
    print(f"Delta V: {delta_v((590 * 10 ** 3) + astro.RADIUS_EARTH, (600 * 10 ** 3) + astro.RADIUS_EARTH)}")
    
    
    plt.show()


if __name__ == "__main__":
    main()    