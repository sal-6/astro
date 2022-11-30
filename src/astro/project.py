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
                    pass

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()
    
def NBodySolarSailRadial(t, state, masses, ss_area, ss_reflectivity):
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
                    _r_sun = np.array([state[0][0], state[0][1], state[0][2]])
                    _r_sc = np.array([state[1][0], state[1][1], state[1][2]])
                    _r_to_sun = _r_sun - _r_sc
                    distance_from_sun = np.linalg.norm(_r_to_sun)
                    
                    F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(90) ** 2 / distance_from_sun ** 2 * _r_to_sun
                                         
                    unit_vector_sun_to_sc = -1 * _r_to_sun / distance_from_sun
                    _F_srp = F_srp * unit_vector_sun_to_sc
                    
                    _F = np.add(_F, _F_srp)    

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()


def NBodySolarSail45(t, state, masses, ss_area, ss_reflectivity):
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
                    _r_sun = np.array([state[0][0], state[0][1], state[0][2]])
                    _r_sc = np.array([state[1][0], state[1][1], state[1][2]])
                    _r_to_sun = _r_sun - _r_sc
                    _r_sun_to_sc = -1 * _r_to_sun
                    distance_from_sun = np.linalg.norm(_r_to_sun)
                    
                    F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(np.pi / 4) ** 2 / (distance_from_sun / astro.AU_meters) ** 2
                    unit_vector_sun_to_sc = -1 * _r_to_sun / distance_from_sun
                    dir_moving = _v_body / np.linalg.norm(_v_body)
                    
                    _momentum_dir = np.cross(unit_vector_sun_to_sc, dir_moving)
                    
                    # apply for 45 degree angle to sun
                    _force_dir = astro.rodrigues_rotation_formula(unit_vector_sun_to_sc, _momentum_dir, np.pi / 4)
                    _F = np.add(_F, F_srp * _force_dir)
                        

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()

def NBodySolarSailGeneralDirection(t, state, masses, ss_area, ss_reflectivity, sun_sc_angle, direction_vector_handle):
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
        ss_area (float): Area of the solar sail in m^2
        ss_reflectivity (float): Reflectivity of the solar sail
        sun_sc_angle (float): Angle between the sun and the spacecraft in radians
        direction_vector_handle (function): Function that takes in the spacecraft state, suns state, and returns a unit vector. 
            Shall be implemented as: direction_vector_handle(spacecraft_state, sun_state)
    
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
                    
                    # distance from sun to spacecraft
                    _r_sun = np.array([state[0][0], state[0][1], state[0][2]])
                    _r_sc = np.array([state[1][0], state[1][1], state[1][2]])
                    _r_sun_to_sc = _r_sc - _r_sun
                    
                    # distance from sun to spacecraft
                    distance_from_sun = np.linalg.norm(_r_sun_to_sc)
                    
                    # sun state
                    _sun_state = np.array([state[0][0], state[0][1], state[0][2], state[0][3], state[0][4], state[0][5]])
                    
                    # sc state
                    _sc_state = np.array([state[1][0], state[1][1], state[1][2], state[1][3], state[1][4], state[1][5]])

                    # get the direction vector
                    _force_dir = direction_vector_handle(_sc_state, _sun_state)
                    
                    # normalize the force direction
                    _force_dir = _force_dir / np.linalg.norm(_force_dir)
                    
                    # calculate the force (via SMAD)
                    F_srp = 9.1113 * 10 ** -6 * ss_area * ss_reflectivity * np.sin(sun_sc_angle) ** 2 / (distance_from_sun / astro.AU_meters) ** 2
                    
                    # apply the force
                    _F = np.add(_F, F_srp * _force_dir)

        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()

def rodrigues_rotation_formula(v, k, theta):
    """Rotates a vector v about a vector k by an angle theta.
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    
    Args:
        v (arr): Vector to be rotated
        k (arr): Vector about which to rotate
        theta (float): Angle to rotate by
    
    Returns:
        arr: Rotated vector
    """
    
    # normalize k
    k = k / np.linalg.norm(k)
    
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
    
