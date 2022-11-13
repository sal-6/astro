#############################################################################
# Sal Aslam
# ENAE601 - UMD
# Description: HW5 assignment problems

import pandas as pd
import numpy as np
import astro
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def num_2():

    x = 1.2
    y = 0
    z = 0
    vx = 0
    vy = -1.049357509830343
    vz = 0

    state = np.array([x, y, z, vx, vy, vz])
    mu = 0.012150585609624
    T = 6.192169331319632
    tol = 10**-13

    orbit = solve_ivp(astro.propogate_CRTBP, [0, T], state, args=(mu,), rtol=tol, atol=tol)

    # create a data frame out of intial and final states
    df = pd.DataFrame(orbit.y[:, [0, -1]].T, columns=["x", "y", "z", "v_x", "v_y", "v_z"], index=["Initial", "Final"])
    print(df)

    # plot the orbit
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2], linewidth=1, label="Orbit")
    ax.set_xlabel('x (DU)', size=10)
    ax.set_ylabel('y (DU)', size=10)
    ax.set_zlabel('z (DU)', size=10)
    ax.set_title("Orbit A Trajectory")

    # plot primary and secondary bodies at -mu and 1-mu
    ax.scatter(-mu, 0, 0, color="orange", s=5, label="Primary Body")
    ax.scatter(1-mu, 0, 0, color="blue", s=1, label="Secondary Body")

    ax.legend()
    plt.show()

def num_3():

    orbits = [
        {
            "id": "B",
            "initial_state": np.array([-0.08, -0.03, 0.01, 3.5, -3.1, -0.1]),
            "time": 26
        },
        {
            "id": "C",
            "initial_state": np.array([0.05, -0.05, 0, 4.0, 2.6, 0]),
            "time": 25
        },
        {
            "id": "D",
            "initial_state": np.array([0.83, 0, 0.114062816271683, 0, 0.229389507175582, 0]),
            "time": 15
        },
        {
            "id": "E",
            "initial_state": np.array([-0.05, -0.02, 0, 4.09, -5.27, 0]),
            "time": 15
        }
    ]

    mu = 0.012150585609624
    tol = 10**-13

    for orbit in orbits:
        print("Propogating Orbit " + orbit["id"] + "...")

        trajectory = solve_ivp(astro.propogate_CRTBP, [0, orbit["time"]], orbit["initial_state"], args=(mu,), rtol=tol, atol=tol)


        # convert from rotational to inertial frame
        x = trajectory.y[0]
        y = trajectory.y[1]
        z = trajectory.y[2]
        vx = trajectory.y[3]
        vy = trajectory.y[4]
        vz = trajectory.y[5]

        inert_x, inert_y, inert_z = astro.rotating_to_inertial(x, y, z, trajectory.t)
        inert_primary_x, inert_primary_y, inert_primary_z = astro.rotating_to_inertial_for_const(-mu, 0, 0, trajectory.t)
        inert_secondary_x, inert_secondary_y, inert_secondary_z = astro.rotating_to_inertial_for_const(1-mu, 0, 0, trajectory.t)

        # plot the intertial frame trajectory
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(inert_x, inert_y, inert_z, linewidth=1, label="Orbit")
        ax.plot3D(inert_primary_x, inert_primary_y, inert_primary_z, linewidth=1, label="Primary Body")
        ax.plot3D(inert_secondary_x, inert_secondary_y, inert_secondary_z, linewidth=1, label="Secondary Body")
        ax.set_xlabel('x (DU)', size=10)
        ax.set_ylabel('y (DU)', size=10)
        ax.set_zlabel('z (DU)', size=10)
        ax.set_title("Orbit {} Trajectory (Intertial Frame)".format(orbit["id"]))
        ax.legend()

        axisEqual3D(ax)


        # plot the orbit
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(trajectory.y[0], trajectory.y[1], trajectory.y[2], linewidth=1, label="Orbit")
        ax.set_xlabel('x (DU)', size=10)
        ax.set_ylabel('y (DU)', size=10)
        ax.set_zlabel('z (DU)', size=10)
        ax.set_title("Orbit " + orbit["id"] + " Trajectory (Rotating Frame)")

        # plot primary and secondary bodies at -mu and 1-mu
        ax.scatter(-mu, 0, 0, color="orange", s=5, label="Primary Body")
        ax.scatter(1-mu, 0, 0, color="blue", s=1, label="Secondary Body")

        axisEqual3D(ax)

        ax.legend()

        print("Done Propogating Orbit " + orbit["id"] + "!")
    
    plt.show()

def num_4():
    orbits = [
        {
            "id": "B",
            "initial_state": np.array([-0.08, -0.03, 0.01, 3.5, -3.1, -0.1]),
            "time": 26
        },
        {
            "id": "D",
            "initial_state": np.array([0.83, 0, 0.114062816271683, 0, 0.229389507175582, 0]),
            "time": 15
        }
    ]

    mu = 0.012150585609624
    tol = 10**-13


    for orbit in orbits:
        print("Propogating Orbit " + orbit["id"] + "...")

        trajectory = solve_ivp(astro.propogate_CRTBP, [0, orbit["time"]], orbit["initial_state"], args=(mu,), rtol=tol, atol=tol)

        # plot in rotating frame
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(trajectory.y[0], trajectory.y[1], trajectory.y[2], linewidth=1, label="Orbit")
        ax.set_xlabel('x (DU)', size=10)
        ax.set_ylabel('y (DU)', size=10)
        ax.set_zlabel('z (DU)', size=10)
        ax.set_title("Orbit " + orbit["id"] + " Trajectory (Rotating Frame)")

        # plot primary and secondary bodies at -mu and 1-mu
        ax.scatter(-mu, 0, 0, color="orange", s=5, label="Primary Body")
        ax.scatter(1-mu, 0, 0, color="blue", s=1, label="Secondary Body")

        axisEqual3D(ax)

        ax.legend()

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
    num_4()


