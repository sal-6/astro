#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Utilities written for the final project

import astro
import numpy as np

def NBodySolarSail(t, state, masses, ss_area, ss_tilt_angle, ss_reflectivity):
    """Calculates the state derivative for the NBody problem.
    masses in kg
    state in m and m/s

    
    Args:
        t (float): Time
        state (arr): State vector with form:
            [r_1_x, r_1_y, r_1_z, v_1_x, v_1_y, v_1_z, r_2_x, r_2_y, r_2_z, v_2_x, v_2_y, v_2_z, ...]

            In order to perform the necessary calculations, the state vector must be given with the suns states listed first,
            followed by the spacecraft states, followed by any additional bodies in the system. It shall be structured as follows:

            [r_sun_x, r_sun_y, r_sun_z, v_sun_x, v_sun_y, v_sun_z, r_sc_x, r_sc_y, r_sc_z, v_sc_x, v_sc_y, v_sc_z, ...]

            
        masses (arr): Masses of the bodies in the same order as the state vector:
            [m_1, m_2, ...]
    
    """
    
    # perform typical n body calculations for each body included in the spacecraft
    state = state.reshape((len(masses), 6))
    state_der = np.zeros((len(masses), 6))
    for i, body in enumerate(state):
        _r_body = np.array([state[i][0], state[i][1], state[i][2]])
        _v_body = np.array([state[i][3], state[i][4], state[i][5]])
        _F = np.array([0., 0., 0.])
        for j, other_body in enumerate(state):
            if i != j:
                _r_other_body = np.array([state[j][0], state[j][1], state[j][2]])
                _v_other_body = np.array([state[j][3], state[j][4], state[j][5]])
                _r_to_body = _r_body - _r_other_body
                force = -astro.G * masses[j] * masses[i] / np.linalg.norm(_r_to_body) ** 3 * _r_to_body
                _F = np.add(_F, force)

                # if the body has the solar sail, then add the force of the solar sail
                if i == 1:
                    

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    
    


    return state_der.flatten()


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
