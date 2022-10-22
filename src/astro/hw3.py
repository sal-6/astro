#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Utilities written for homewaork 3

import astro
import numpy as np
import math

def propogate_2BP_low_thrust(t, state, specfic_thrust):
    """State space representation of Newton's Law of Gravitation. Only implemented for Earth.
        Selected state variables are [r_x, r_y, r_z, v_x, v_y, v_z]

        This implementation considers a specfic thrust (acceleration) that is tangential to 
        the velocity of the spacecraft at the given timestep.

    Args:
        t (float): Current time step
        state (arr): State vector of form [r_x, r_y, r_z, v_x, v_y, v_z]

    Returns:
        Value of velocity and acceleration at the given time and state in form:
            [v_x, v_y, v_z, a_x, a_y, a_z]
    """

    _r = state[0:3]
    _v = state[3:]

    _v_dir = _v / np.linalg.norm(_v)

    # a = - mu / norm(r) ^ 3 * r
    _a = - astro.MU_EARTH / (np.linalg.norm(_r) ** 3) * _r + specfic_thrust * _v_dir

    val = np.array([_v[0], _v[1], _v[2], _a[0], _a[1], _a[2]])
    return val


def calculate_lamberts_constants(phi):
    """ Calculates constants c2 and c3 for lamberts solver"""

    if phi > 10 ** -6:
        c2 = (1 - np.cos(np.sqrt(phi))) / phi
        c3 = (np.sqrt(phi) - np.sin(np.sqrt(phi))) / np.sqrt(phi ** 3)

    else:

        if phi < -10 ** -6:
            c2 = (1 - np.cosh(np.sqrt(-phi))) / phi
            c3 = (np.sinh(np.sqrt(-phi)) - np.sqrt(-phi)) / np.sqrt((-phi) ** 3)
        
        else:
            c2 = 1 / 2
            c3 = 1 / 6

    return c2, c3


def lamberts(_r0, _rf, del_t, tm):
    """
    Lamberts solver

    Args:
        _r0 (arr): Initial position vector
        _rf (arr): Final position vector
        tof (float): Time of flight
        dom (int): Direction of motion (1 or -1)

    Returns:
        (arr): initial velocity vector
        (arr): final velocity vector
    """

    cos_del_nu = np.dot(_r0, _rf) / (np.linalg.norm(_r0) * np.linalg.norm(_rf))
    sin_del_nu = tm * math.sqrt(1 - cos_del_nu ** 2)
    A = tm * math.sqrt(np.linalg.norm(_r0) * np.linalg.norm(_rf) * (1 + cos_del_nu))

    if A == 0:
        return None, None

    phi_n = 0.0
    c_2, c_3 = calculate_lamberts_constants(phi_n)
    
    phi_up = 4 * math.pi ** 2
    phi_low = - 4 * math.pi

    while True:
        y_n = np.linalg.norm(_r0) + np.linalg.norm(_rf) + (A * (phi_n * c_3 - 1)) / (math.sqrt(c_2))

        if A > 0 and y < 0:
            return 
