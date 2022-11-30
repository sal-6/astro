#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Utilities writen for homewark 1

import math
import numpy as np
import astro

def cartesian_to_standard(r, v, mu=astro.MU_EARTH):
    """Converts Cartian orbital elements to standard orbital elements.

    Args:
        r (ndarray): Position vector (in meters)
        v (ndarray): Velocity vector (in meters)
        mu (float): Specific gravity of celestial body (in m^3 s^-2)

    Returns:
        A list containing the elements: [a, e, i, omega, w, u]
    """
    r = r.astype('float64')
    v = v.astype('float64')

    pos = np.linalg.norm(r)
    vel = np.linalg.norm(v)

    _momentum = np.cross(r, v)
    _line_of_nodes = np.cross(astro.DIR_Z, _momentum)
    _eccentricity = 1 / mu * ((vel ** 2 - mu / pos) * r - (np.dot(r, v)) * v)

    e = np.linalg.norm(_eccentricity)
    
    energy = vel ** 2 / 2 - mu / pos
    a = - mu / (2 * energy)

    cos_i = np.dot(_momentum, astro.DIR_Z) / np.linalg.norm(_momentum)
    i = np.degrees(np.arccos(cos_i))

    cos_omega = np.dot(_line_of_nodes, astro.DIR_X) / np.linalg.norm(_line_of_nodes)
    omega = np.degrees(np.arccos(cos_omega))

    if np.dot(_line_of_nodes, astro.DIR_Y) < 0:
        omega = 360 - omega

    cos_w = np.dot(_line_of_nodes, _eccentricity) / (np.linalg.norm(_line_of_nodes) * e)
    w = np.degrees(np.arccos(cos_w))

    if np.dot(_eccentricity, astro.DIR_Z) < 0:
        w = 360 - w
    
    cos_u = np.dot(_eccentricity, r) / (e * pos)
    u = np.degrees(np.arccos(cos_u))

    if np.dot(r, v) < 0:
        u = 360 - u
    
    return [a, e, i, omega, w, u]


def standard_to_cartesian(a, e, i, raan, w, u, mu=astro.MU_EARTH):
    """_summary_

    Args:
        a (float): Semi-major axis (m)
        e (float): Eccentricity ()
        i (float): Inclination (deg)
        raan (float): RAAN (deg)
        w (float): Argument of periapsis (deg)
        u (float): True anamoly (deg)

    Returns:
        r
        v
    """
    
    r = a * (1 - e ** 2) / (1 + e * np.cos(np.radians(u)))

    _r = np.array([r * np.cos(np.radians(u)), r * np.sin(np.radians(u)), 0])
    _v = math.sqrt(mu / (a * (1 - e ** 2))) * np.array([-np.sin(np.radians(u)), e + np.cos(np.radians(u)), 0])

    rad_raan = np.radians(raan)
    rad_i = np.radians(i)
    rad_w = np.radians(w)

    R_3_raan = np.array([[np.cos(rad_raan) , np.sin(rad_raan), 0],
                         [-np.sin(rad_raan), np.cos(rad_raan) , 0],
                         [0                , 0               , 1]])

    R_1_i = np.array([[1, 0, 0],
                      [0, np.cos(rad_i), np.sin(rad_i)],
                      [0, -np.sin(rad_i), np.cos(rad_i)]])

    R_3_w = np.array([[np.cos(rad_w) , np.sin(rad_w), 0],
                      [-np.sin(rad_w), np.cos(rad_w), 0],
                      [0             , 0         , 1]])

    R_eff = R_3_w @ R_1_i @ R_3_raan
    R_eff = np.matrix.transpose(R_eff)
    
    _pos = R_eff @ _r
    _vel = R_eff @ _v
    return [_pos, _vel]


def propogate_2BP(t, state, mu=astro.MU_EARTH):
    """State space representation of Newton's Law of Gravitation. Only implemented for Earth.
        Selected state variables are [r_x, r_y, r_z, v_x, v_y, v_z]

    Args:
        t (float): Current time step
        state (arr): State vector of form [r_x, r_y, r_z, v_x, v_y, v_z]

    Returns:
        Value of velocity and acceleration at the given time and state in form:
            [v_x, v_y, v_z, a_x, a_y, a_z]
    """

    _r = state[0:3]
    _v = state[3:]

    # a = - mu / norm(r) ^ 3 * r
    _a = - mu / (np.linalg.norm(_r) ** 3) * _r

    val = np.array([_v[0], _v[1], _v[2], _a[0], _a[1], _a[2]])
    return val


def calculate_orbital_period(a, mu=astro.MU_EARTH):
    """Calculates orbital period of an orbit.

    Args:
        a (float): Semi-major axis of orbit
        mu (float, optional): Gravitional constant for body. Defaults to astro.MU_EARTH.

    Returns:
        float: Orbital peroid in seconds
    """
    return 2 * math.pi * math.sqrt(a ** 3 / mu )


def calculate_orbital_energy(r, v, mu=astro.MU_EARTH):
    """Calculates orbital energy

    Args:
        r (float): Radius magnitude
        v (float): Velocity magnitude
    
    Returns:
        float: Energy of orbit
    """
    return v ** 2 / 2 - mu / r

def calculate_angular_momentum(_r, _v):
    """Calculates angular momentum

    Args:
        _r (arr): Position vector
        _v (arr): Velocity vector

    Returns:
        arr: Angular Momentum
    """
    return np.cross(_r, _v)


def calculate_true_anomoly(_e, _r, _v):
    """Calciulate true anamoly

    Args:
        _e (_type_): _description_
        _r (_type_): Radius vector
        _v (_type_): Velocity vectore

    Returns:
        _type_: _description_
    """
    
    v = np.arccos(np.dot(_e, _r) / (np.linalg.norm(_e) * np.linalg.norm(_r)))

    if np.dot(_r, _v) < 0:
        v = 2*math.pi - v
    
    return v

