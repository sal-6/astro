#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: HW3 assignment problems

import astro
import numpy as np
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def num_1():
    a = 7500 * 10 ** 3
    e = 0
    i = 90
    w = 0
    raan = 0
    v = 0

    a_t = 10 ** -3

    _r, _v = astro.standard_to_cartesian(a, e, i, w, raan, v)

    # if any element of _v is really small, set it to 0
    for i in range(len(_v)):
        if abs(_v[i]) < 10 ** -10:
            _v[i] = 0

    r_0 = np.linalg.norm(_r)
    v_0 = np.linalg.norm(_v)

    t_escape = v_0 / a_t * (1 - ((20 * a_t ** 2 * r_0 **2) / (v_0 ** 4)) ** (1/8))
    
    print(f"Initial r: {_r}")
    print(f"Initial v: {_v}")
    print(f"Escape time: {t_escape}")

    init_state = np.array([_r[0], _r[1], _r[2], _v[0], _v[1], _v[2]])
    tol = 10**-13

    a = time.time()
    orbit = solve_ivp(astro.propogate_2BP_low_thrust, [0, t_escape*2], init_state, args=(a_t,), method="RK45", atol=tol, rtol=tol, t_eval=np.linspace(0, t_escape*2, int(t_escape*2)), dense_output=True)
    print(f"Propogation took: {time.time() - a} seconds.")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2], linewidth=.2)
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Satellite")
    ax.set_zlim(-5*10**8, 5*10**8) # due to doing the math in meters

    # get the final velocity vector
    _v_f = orbit.y[3:, -1]

    print(f"Velocity at final epoch (t={orbit.t[-1]}): {_v_f}")

    r_escape = (r_0 * v_0) / (20 * a_t ** 2 * r_0 ** 2) ** (1/4)
    
    # calculate when the satellite will escape
    for i in range(len(orbit.t)):
        if np.linalg.norm(orbit.y[:3, i]) >= r_escape:
            t_escape_true = orbit.t[i]
            print(f"Satellite escaped at t={orbit.t[i]} with radius {np.linalg.norm(orbit.y[:3, i])}")
            print(f"Overshot r_escape by {np.linalg.norm(orbit.y[:3, i]) - r_escape}")
            break
    
    # calculate error in escape time and percent relative error
    print(f"Error in escape time: {t_escape - t_escape_true}")
    print(f"Percent relative error: {abs(t_escape - t_escape_true) / t_escape * 100}%")

    # calculate when the satellite will reach a radius of 8000 km
    for i in range(len(orbit.t)):
        if np.linalg.norm(orbit.y[:3, i]) >= 8000 * 10 ** 3:
            t_8000 = orbit.t[i]
            print(f"Satellite reached 8000 km at t={orbit.t[i]} with radius {np.linalg.norm(orbit.y[:3, i])}")
            print(f"Overshot 8000 km by {np.linalg.norm(orbit.y[:3, i]) - 8000 * 10 ** 3}")
            break
    
    plt.show()


def num_2():

    mu_sun = 132712440018

    r_mars = np.array([-128169484.29, -190592298.12, -844880.03])
    v_mars = np.array([21.02, -11.45, -0.76])

    r_jupiter = np.array([483382929.98, -587464623.05, -8381282.40])
    v_jupiter = np.array([9.93, 8.92, -0.26])

    _v0, _vf = astro.lamberts(r_mars, r_jupiter, 830 * 24 * 60 * 60, 1, mu_sun)

    print(f"Initial velocity: {_v0}")
    print(f"Final velocity: {_vf}")


def num_3():

    mu_sun = 132712440042 * 10 ** 9

    r_mars = np.array([-128169484.29 * 10 ** 3, -190592298.12 * 10 ** 3, -844880.03 * 10 ** 3])
    v_mars = np.array([21.02 * 10 ** 3, -11.45 * 10 ** 3, -0.76 * 10 ** 3])

    r_jupiter = np.array([483382929.98 * 10 ** 3, -587464623.05 * 10 ** 3, -8381282.40 * 10 ** 3])
    v_jupiter = np.array([9.93 * 10 ** 3, 8.92 * 10 ** 3, -0.26 * 10 ** 3])

    _v0, _vf = astro.lamberts(r_mars, r_jupiter, 830 * 24 * 60 * 60, 1, mu_sun)

    # initial state for spacecraft leaving mars
    # postition of mars
    # velocity from lamberts
    init_state = np.array([r_mars[0], r_mars[1], r_mars[2], _v0[0], _v0[1], _v0[2]])

    # propogate the orbit
    orbit = solve_ivp(astro.propogate_2BP, [0, 830 * 24 * 60 * 60], init_state, args=(mu_sun,), method="RK45", atol=10**-13, rtol=10**-13, t_eval=np.linspace(0, 830 * 24 * 60 * 60, 830 * 24 * 60 * 60), dense_output=True)

    # compare the final position to the position of jupiter
    print("Short way:")
    print(f"Final position from 2BP: {orbit.y[:3, -1]}")
    print(f"Jupiter position: {r_jupiter}")
    print()
    print(f"Error in position: {orbit.y[:3, -1] - r_jupiter}")
    print(f"Percent relative error: {abs(orbit.y[:3, -1] - r_jupiter) / r_jupiter * 100}%")
    print()
    print("-----------------------------------------")
    print()
    # velocity of spacecraft at jupiter
    print(f"Final velocity from 2BP: {orbit.y[3:, -1]}")
    # compare the final velocity to the velocity of lamberts
    print(f"Lamberts velocity at Jupiter: {_vf}")
    
    print()
    
    # calculate the error in velocity
    print(f"Error in velocity: {orbit.y[3:, -1] - _vf}")
    print(f"Percent relative error: {abs(orbit.y[3:, -1] - _vf) / _vf * 100}%")

    
    _v0_long, _vf_long = astro.lamberts(r_mars, r_jupiter, 830 * 24 * 60 * 60, -1, mu_sun)
    
    # initial state for spacecraft leaving mars
    # postition of mars
    # velocity from lamberts
    init_state = np.array([r_mars[0], r_mars[1], r_mars[2], _v0_long[0], _v0_long[1], _v0_long[2]])
    
    # propogate the orbit
    orbit = solve_ivp(astro.propogate_2BP, [0, 830 * 24 * 60 * 60], init_state, args=(mu_sun,), method="RK45", atol=10**-13, rtol=10**-13, t_eval=np.linspace(0, 830 * 24 * 60 * 60, 830 * 24 * 60 * 60), dense_output=True)
    
    # compare the final position to the position of jupiter
    print()
    print("Long way:")
    print(f"Final position from 2BP: {orbit.y[:3, -1]}")
    print(f"Jupiter position: {r_jupiter}")
    print()
    print(f"Error in position: {orbit.y[:3, -1] - r_jupiter}")
    print(f"Percent relative error: {abs(orbit.y[:3, -1] - r_jupiter) / r_jupiter * 100}%")
    print()
    print("-----------------------------------------")
    print()
    # velocity of spacecraft at jupiter
    print(f"Final velocity from 2BP: {orbit.y[3:, -1]}")
    # compare the final velocity to the velocity of lamberts
    print(f"Lamberts velocity at Jupiter: {_vf_long}")
    
    print()
    
    # calculate the error in velocity
    print(f"Error in velocity: {orbit.y[3:, -1] - _vf_long}")
    print(f"Percent relative error: {abs(orbit.y[3:, -1] - _vf_long) / _vf_long * 100}%")
    
    
    
def num_4():

    mu_sun = 132712440042 * 10 ** 9

    r_mars = np.array([-128169484.29 * 10 ** 3, -190592298.12 * 10 ** 3, -844880.03 * 10 ** 3])
    v_mars = np.array([21.02 * 10 ** 3, -11.45 * 10 ** 3, -0.76 * 10 ** 3])

    r_jupiter = np.array([483382929.98 * 10 ** 3, -587464623.05 * 10 ** 3, -8381282.40 * 10 ** 3])
    v_jupiter = np.array([9.93 * 10 ** 3, 8.92 * 10 ** 3, -0.26 * 10 ** 3])

    
    tof = []
    long_way = []
    short_way = []
    
    # loop over times of flight from 1 year to 3 years in 1 day increments and store the delta v for each
    # for both the long way and short way
    # also store the time of flight
    for i in range(365, 1095, 1):
        _v0, _vf = astro.lamberts(r_mars, r_jupiter, i * 24 * 60 * 60, 1, mu_sun)
        _v0_long, _vf_long = astro.lamberts(r_mars, r_jupiter, i * 24 * 60 * 60, -1, mu_sun)
        
        tof.append(i)
        long_way.append(np.linalg.norm(_vf_long - v_jupiter) + np.linalg.norm(_v0_long - v_mars))
        short_way.append(np.linalg.norm(_vf - v_jupiter) + np.linalg.norm(_v0 - v_mars))
        
        # progress bar
        print(f"{i - 365} / {1095 - 365}", end="\r")
        
        
    # plot the results
    plt.plot(tof, long_way, label="Long Way")
    plt.plot(tof, short_way, label="Short Way")
    plt.xlabel("Time of Flight (days)")
    plt.ylabel("Delta V (m/s)")
    plt.title("Delta V vs Time of Flight")
    plt.legend()
    plt.show()
    
    
    

if __name__ == "__main__":
    num_1()
    num_2()
    num_3()
    num_4()