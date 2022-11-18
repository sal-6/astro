#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: Utilities written for homewaork 5

import numpy as np

def propogate_CRTBP(t, state, mu):
    """
    State space representation of CRTBP.
    Selected state variables are [r_x, r_y, r_z, v_x, v_y, v_z]

    Args:
        t (float): Current time step
        state (arr): State vector of form [r_x, r_y, r_z, v_x, v_y, v_z]
        mu (float): Mass ratio of the two bodies

    Returns:
        Value of velocity and acceleration at the given time and state in form:
            [v_x, v_y, v_z, a_x, a_y, a_z]
    """

    x_0_dot = state[3]
    x_1_dot = state[4]
    x_2_dot = state[5]

    x_3_dot = 2 * state[4] + state[0] - (mu * (state[0] - 1 + mu)) / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** (3 / 2) - ((1 - mu) * (state[0] + mu)) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** (3 / 2)
    x_4_dot = -2 * state[3] + state[1] - (mu * state[1]) / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** (3 / 2) - ((1 - mu) * state[1]) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** (3 / 2)
    x_5_dot = - (mu * state[2]) / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** (3 / 2) - ((1 - mu) * state[2]) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** (3 / 2)

    val = np.array([x_0_dot, x_1_dot, x_2_dot, x_3_dot, x_4_dot, x_5_dot])
    return val


def rotating_to_inertial(x_arr, y_arr, z_arr, time):
    """
    Rotates the given arrays of position vectors from rotating frame to inertial frame.

    Args:
        x_arr (arr): Array of x positions
        y_arr (arr): Array of y positions
        z_arr (arr): Array of z positions
        time (arr): Array of times

    Returns:
        Tuple of arrays of position vectors in inertial frame
    """

    # loop over each time step and rotate the position vector
    x_arr_inertial = []
    y_arr_inertial = []
    z_arr_inertial = []

    for i in range(len(time)):
        # get the rotation matrix
        R = np.array([[np.cos(time[i]), -np.sin(time[i]), 0], [np.sin(time[i]), np.cos(time[i]), 0], [0, 0, 1]])

        # get the position vector
        pos_vec = np.array([x_arr[i], y_arr[i], z_arr[i]])

        # rotate the position vector
        pos_vec_rotated = np.dot(R, pos_vec)

        # append to the arrays
        x_arr_inertial.append(pos_vec_rotated[0])
        y_arr_inertial.append(pos_vec_rotated[1])
        z_arr_inertial.append(pos_vec_rotated[2])

    return np.array(x_arr_inertial), np.array(y_arr_inertial), np.array(z_arr_inertial)


def rotating_to_inertial_for_const(x, y, z, time):

    x_arr_inertial = []
    y_arr_inertial = []
    z_arr_inertial = []

    for i in range(len(time)):
        # get the rotation matrix
        R = np.array([[np.cos(time[i]), -np.sin(time[i]), 0], [np.sin(time[i]), np.cos(time[i]), 0], [0, 0, 1]])

        # get the position vector
        pos_vec = np.array([x, y, z])

        # rotate the position vector
        pos_vec_rotated = np.dot(R, pos_vec)

        # append to the arrays
        x_arr_inertial.append(pos_vec_rotated[0])
        y_arr_inertial.append(pos_vec_rotated[1])
        z_arr_inertial.append(pos_vec_rotated[2])

    return np.array(x_arr_inertial), np.array(y_arr_inertial), np.array(z_arr_inertial)

    