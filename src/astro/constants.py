#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Constants that are useful in astrodynamics

import numpy as np


# Natural Basis
DIR_X = np.array([1, 0, 0])
DIR_Y = np.array([0, 1, 0])
DIR_Z = np.array([0, 0, 1])


# Specific Gravitation Constants
MU_EARTH = 3.986004418 * 10 ** 14


RADIUS_EARTH = 6378000 # m

G = 6.67408 * 10 ** -11 # m^3 kg^-1 s^-2