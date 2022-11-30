#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: HW6 assignment problems

import numpy as np
import astro
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sym
import time


def get_partials():
    
    mu = 3.986004e14 # m^3/s^2
    R_earth = 6378.137e3 # m
    J2 = 0.00108248
    
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    z = sym.Symbol('z')
    
    r = sym.sqrt(x**2 + y**2 + z**2)
    
    U = (mu / r) * (1 - J2 * ((R_earth / r) ** 2) * (1.5 * (z / r)**2 - 0.5));

    dU_dx = sym.diff(U, x)
    dU_dy = sym.diff(U, y)
    dU_dz = sym.diff(U, z)
    
    print("dU_dx:")
    print(sym.simplify(dU_dx))
    
    print("dU_dy:")
    print(sym.simplify(dU_dy))
    
    print("dU_dz:")
    print(sym.simplify(dU_dz))
    
    return dU_dx, dU_dy, dU_dz
    
    
def U(x, y, z):
    
    r = np.sqrt(x**2 + y**2 + z**2)
    mu = 3.986004e14 # m^3/s^2
    R_earth = 6378.137e3 # m
    J2 = 0.00108248
    
    U = (mu / r) * (1 - J2 * ((R_earth / r) ** 2) * (1.5 * (z / r)**2 - 0.5));
    
    return U

def num_1():
    
    mu = 3.986004e14 # m^3/s^2
    R_earth = 6378.137e3 # m
    J2 = 0.00108248
    
    # initial orbital elements
    r_p = R_earth + 1000e3 # m
    e = 0.15
    i = 40 # deg
    omega = 25 # deg
    w = 15 # deg
    nu = 20 # deg
    
    a = r_p / (1 - e)
    
    initial_orbital_elements = {
        "a": a,
        "e": e,
        "i": i,
        "omega": omega,
        "w": w,
        "nu": nu
    }
    
    # convert to cartesian
    _r_init, _v_init = astro.standard_to_cartesian(a, e, i, omega, w, nu, mu)
    
    # initial state
    state_init = np.array([_r_init[0], _r_init[1], _r_init[2], _v_init[0], _v_init[1], _v_init[2]])
    
    # define time span of 3 days
    t_span = 3 * 24 * 60 * 60 # s
    
    tol = 1e-13
    
    print("Solving with J2 perturbations...")
    a = time.time()
    sol = solve_ivp(astro.propogate_2BP_with_J2_perturbations, [0, t_span], [_r_init[0], _r_init[1], _r_init[2], _v_init[0], _v_init[1], _v_init[2]], method='RK45', args=(mu,), atol=tol, rtol=tol)
    print("Done.")
    b = time.time()
    print("Took: ", b - a)
    
    # plot orbit
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    # set title
    ax.set_title('Orbit with J2 perturbations')
        
    axisEqual3D(ax)
    
    
    # convert to orbital elements and plot them over time
    a = np.zeros(len(sol.t))
    e = np.zeros(len(sol.t))
    inc = np.zeros(len(sol.t))
    omega = np.zeros(len(sol.t))
    w = np.zeros(len(sol.t))
    nu = np.zeros(len(sol.t))
    
    for i in range(len(sol.t)):
        _r = np.array([sol.y[0][i], sol.y[1][i], sol.y[2][i]])
        _v = np.array([sol.y[3][i], sol.y[4][i], sol.y[5][i]])
        a[i], e[i], inc[i], omega[i], w[i], nu[i] = astro.cartesian_to_standard(_r, _v, mu)
        
    
    
    # plot a
    plt.figure()
    plt.plot(sol.t, a)
    plt.xlabel('Time [s]')
    plt.ylabel('Semi-major axis [m]')
    plt.title('Semi-major axis over time')
    
    # plot e
    plt.figure()
    plt.plot(sol.t, e)
    plt.xlabel('Time [s]')
    plt.ylabel('Eccentricity')
    plt.title('Eccentricity over time')
    
    # plot i
    plt.figure()
    plt.plot(sol.t, inc)
    plt.xlabel('Time [s]')
    plt.ylabel('Inclination [deg]')
    plt.title('Inclination over time')
    
    # plot omega
    plt.figure()
    plt.plot(sol.t, omega, label='Numerical')
    plt.xlabel('Time [s]')
    plt.ylabel('RAAN [deg]')
    plt.title('RAAN over time')
    
    # theoretical RAAN over time
    n = np.sqrt(mu / initial_orbital_elements["a"]**3)
    p = initial_orbital_elements['a'] * (1 - initial_orbital_elements["e"]**2) 
    draandt = np.rad2deg(-3 * n * J2 * R_earth**2 * np.cos(np.deg2rad(initial_orbital_elements["i"])) / (2 * p ** 2))
    
    omega_theoretical = omega[0] + draandt * sol.t
    plt.plot(sol.t, omega_theoretical, '--', label='Analytical')
    
    plt.legend()
    
    # plot w
    plt.figure()
    plt.plot(sol.t, w, label='Numerical')
    plt.xlabel('Time [s]')
    plt.ylabel('Argument of Perigee [deg]')
    plt.title('Argument of Perigee over time')
    
    # theoretical argument of perigee over time
    
    dwdt = np.rad2deg( (3 * n * J2 * (R_earth)**2) / (4 * p ** 2) * (4 - 5 * np.sin(np.deg2rad(initial_orbital_elements["i"]))**2) )
    
    w_theoretical = w[0] + dwdt * sol.t
    plt.plot(sol.t, w_theoretical, '--', label='Analytical')
    
    plt.legend()
    
    # plot nu
    plt.figure()
    plt.plot(sol.t, nu)
    plt.xlabel('Time [s]')
    plt.ylabel('True Anomaly [deg]')
    plt.title('True Anomaly over time')
    
    # initial angular momentum
    h_init = np.cross(_r_init, _v_init)
    
    # angular momentum deviation over time
    h_dev_x = np.zeros(len(sol.t))
    h_dev_y = np.zeros(len(sol.t))
    h_dev_z = np.zeros(len(sol.t))
    for i in range(len(sol.t)):
        _r = np.array([sol.y[0][i], sol.y[1][i], sol.y[2][i]])
        _v = np.array([sol.y[3][i], sol.y[4][i], sol.y[5][i]])
        h = np.cross(_r, _v)
        h_dev_x[i] = h[0] - h_init[0]
        h_dev_y[i] = h[1] - h_init[1]
        h_dev_z[i] = h[2] - h_init[2]
        
        
    # plot angular momentum deviation
    plt.figure()
    plt.plot(sol.t, h_dev_x, label='x')
    plt.plot(sol.t, h_dev_y, label='y')
    plt.plot(sol.t, h_dev_z, label='z')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular momentum deviation [m^2/s]')
    plt.title('Angular momentum deviation over time')
    plt.legend()
    
    
    # initial energy
    E_init = 0.5 * np.linalg.norm(_v_init)**2 - U(_r_init[0], _r_init[1], _r_init[2])
    
    # energy deviation over time
    E_dev = np.zeros(len(sol.t))
    for i in range(len(sol.t)):
        _r = np.array([sol.y[0][i], sol.y[1][i], sol.y[2][i]])
        _v = np.array([sol.y[3][i], sol.y[4][i], sol.y[5][i]])
        E = 0.5 * np.linalg.norm(_v)**2 - U(_r[0], _r[1], _r[2])
        E_dev[i] = E - E_init
        
    # plot energy deviation
    plt.figure()
    plt.plot(sol.t, E_dev)
    plt.xlabel('Time [s]')
    plt.ylabel('Energy deviation [m^2/s^2]')
    plt.title('Energy deviation over time')
    
    
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
    num_1()
