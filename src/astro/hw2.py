#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Utilities writen for homewark 2

import math
import numpy as np
import astro


def calculate_semi_major_axis_from_perigee(r_p, e):
    return r_p / (1 - e)


def inital_guess_kepler(t, t_0, a, _r_0, _v_0, mu=astro.MU_EARTH):
    # consider hyperbola and ellipse
    if a > 0:
        x = (t - t_0) * math.sqrt(mu) / a
    elif a < 0:
        sign = (t - t_0) / abs(t - t_0)
        x = sign * math.sqrt(-a) * np.log((-2 * mu * (t - t_0) * (a * ((np.dot(_r_0, _v_0)) + sign * math.sqrt(-mu * a) * (1 - np.linalg.norm(_r_0)/ a))) ** -1))
    return x
    

def calculate_kepler_coeffecients(x, a):
    z = (x ** 2) / a

    if z > 10 ** -6:
        c = (1 - np.cos(math.sqrt(z))) / z
        s = (math.sqrt(z) - np.sin(math.sqrt(z))) / math.sqrt(z ** 3)

    elif z >= -10 ** -6 and z <= 10 ** -6:
        c = 1 / 2
        s = 1 / 6

    elif z < -10 ** -6: 
        c = (1 - np.cosh(math.sqrt(-z))) / z
        s = (np.sinh(math.sqrt(-z)) - math.sqrt(-z)) / math.sqrt((-z) ** 3)

    return c, s, z


def calculate_time(x, a, _r_0, _v_0, mu=astro.MU_EARTH):
    c, s, z = calculate_kepler_coeffecients(x, a)
    t = (1 / math.sqrt(mu)) * ((x ** 3) * s + np.dot(_r_0, _v_0) / math.sqrt(mu) * (x ** 2) * c + np.linalg.norm(_r_0) * x * (1 - z * s))
    return t


def calculate_radius(x, a, _r_0, _v_0, mu=astro.MU_EARTH):
    c, s, z = calculate_kepler_coeffecients(x, a)
    r = x ** 2 * c + np.dot(_r_0, _v_0) / math.sqrt(mu) * x * ( 1 - z * s) + np.linalg.norm(_r_0) * (1 - z * c)
    return r


def calculate_final_state_kepler(x, a, _r_0, _v_0, t, mu=astro.MU_EARTH):
    c, s, z = calculate_kepler_coeffecients(x, a)

    f = 1 - (x ** 2) / np.linalg.norm(_r_0) * c
    g = t - (x ** 3) / math.sqrt(mu) * s

    r = calculate_radius(x, a, _r_0, _v_0, mu)

    f_dot = math.sqrt(mu) / (np.linalg.norm(_r_0) * r) * x * (z * s - 1)
    g_dot = 1 - x ** 2 / r * c

    _r = f * _r_0 + g * _v_0
    _v = f_dot * _r_0 + g_dot * _v_0

    return _r, _v