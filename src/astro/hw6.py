#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Utilities written for homewaork 6


import numpy as np
import astro


# These were generated using the get_params function in assignments/hw6.py. 
# These function definitions and calls below were opted for as they are much more
# performant than performing a substitution of values into a sympy expression.
# *******************************************************************************
def dudx(x, y, z):
    return x*(-2.63291329350302e+25*x**2 - 2.63291329350302e+25*y**2 + 1.05316531740121e+26*z**2 - 398600400000000.0*(x**2 + y**2 + z**2)**2)/(x**2 + y**2 + z**2)**(7/2)

def dudy(x, y, z):
    return y*(-2.63291329350302e+25*x**2 - 2.63291329350302e+25*y**2 + 1.05316531740121e+26*z**2 - 398600400000000.0*(x**2 + y**2 + z**2)**2)/(x**2 + y**2 + z**2)**(7/2)
    
def dudz(x, y, z):
    return z*(-7.89873988050907e+25*x**2 - 7.89873988050907e+25*y**2 + 5.26582658700605e+25*z**2 - 398600400000000.0*(x**2 + y**2 + z**2)**2)/(x**2 + y**2 + z**2)**(7/2)
# *******************************************************************************


def propogate_2BP_with_J2_perturbations(t, state, mu):

    _r = state[0:3]
    _v = state[3:]
    
    _a = np.array([0., 0., 0.])
    
    # J2 perturbation
    _a[0] = dudx(_r[0], _r[1], _r[2])
    _a[1] = dudy(_r[0], _r[1], _r[2])
    _a[2] = dudz(_r[0], _r[1], _r[2])

    val = np.array([_v[0], _v[1], _v[2], _a[0], _a[1], _a[2]])
    return val