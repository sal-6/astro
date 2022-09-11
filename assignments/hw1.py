import astro
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import math
from scipy.signal import find_peaks

def num_1():

    a = (15000 * 10 ** 3) + astro.RADIUS_EARTH
    e = 0.3
    i = 10
    raan = 0
    w = 10
    u = 0

    P = astro.calculate_orbital_period(a)
    print(P)

    _r, _v = astro.standard_to_cartesian(a, e, i, raan, w, u)
    init_state = np.array([_r[0], _r[1], _r[2], _v[0], _v[1], _v[2]])
    tol = 10**-13

    a = time.time()
    orbit = solve_ivp(astro.propogate_2BP, [0, P*2], init_state, method="RK45", atol=tol, rtol=tol, t_eval=np.linspace(0, 2*int(P), 2*int(P)+1))
    print(f"Propogation took: {time.time() - a} seconds.")

    #print(orbit.t.shape)
    #print(orbit.y.shape)

    pos = []
    vel = []
    acc = []
    for i in range(orbit.t.shape[0]):
        pos.append(math.sqrt(orbit.y[0][i] ** 2 + orbit.y[1][i] ** 2 + orbit.y[2][i] ** 2))
        vel.append(math.sqrt(orbit.y[3][i] ** 2 + orbit.y[4][i] ** 2 + orbit.y[5][i] ** 2))
        a_x = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[0][i]
        a_y = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[1][i]
        a_z = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[2][i]
        acc.append(math.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2))

    #plt.style.use('_mpl-gallery')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(orbit.t, pos)
    ax1.set_xlabel('Time (s)', size=10)
    ax1.set_ylabel('Position (m)', size=10)
    ax1.set_title("Postion Vs Time of Orbit A (t = 2P)")

    ax2.plot(orbit.t, vel)
    ax2.set_xlabel('Time (s)', size=10)
    ax2.set_ylabel('Velocity (m/s)', size=10)
    ax2.set_title("Velocity Vs Time of Orbit A (t = 2P)")


    ax3.plot(orbit.t, acc)
    ax3.set_xlabel('Time (s)', size=10)
    ax3.set_ylabel('Acceleration (m/s^2)', size=10)
    ax3.set_title("Acceleration Vs Time of Orbit A (t = 2P)")

    plt.tight_layout()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2]) 
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Orbit A")
    

    plt.tight_layout()


    # label periapsis
    mins, _ = find_peaks(np.array(pos)*-1)

    min_inds = [0, *mins, len(pos) - 1]
    t_s = [orbit.t[i] for i in min_inds]
    y_s = [pos[i] for i in min_inds]

    ax1.plot(t_s, y_s, "ro")


    init_energy = vel[0] ** 2 / 2 - astro.MU_EARTH / pos[0]
    energy_deviation = [astro.calculate_orbital_energy(pos[i], vel[i]) - init_energy for i in range(len(orbit.t))]


    fig2 = plt.figure()
    ax4 = plt.axes()

    ax4.plot(orbit.t, energy_deviation)
    ax4.set_xlabel('Time (s)', size=10)
    ax4.set_ylabel('Deviation from Initial Orbital Energy (J/kg)', size=10)
    ax4.set_title("Deviation from Initial Orbital Energy Vs Time of Orbit A (t = 2P)")

    _rad = []
    _vel = []
    _angular_momentum = []
    for i in range(len(orbit.t)):
        _rad_c = np.array([orbit.y[0][i], orbit.y[1][i], orbit.y[2][i]])
        _vel_c = np.array([orbit.y[3][i], orbit.y[4][i], orbit.y[5][i]])
        _rad.append(_rad_c)
        _vel.append(_vel_c)
        _angular_momentum.append(astro.calculate_angular_momentum(_rad_c, _vel_c))

    init_momentum = np.linalg.norm(_angular_momentum[0])
    momentum_deviation = []
    for item in _angular_momentum:
        momentum_deviation.append(np.linalg.norm(item) - init_momentum)

    
    fig3, momentum_axes = plt.subplots(3, 1)

    for i, ax_symbol in enumerate(["x", "y", "z"]):
        momentum_axes[i].plot(orbit.t, [j[i] for j in _angular_momentum])
        momentum_axes[i].set_xlabel('Time (s)', size=10)
        momentum_axes[i].set_ylabel(f'Angular Momentum (m^2/s)', size=10)
        momentum_axes[i].set_title(f'{ax_symbol.upper()}-Component of Angular Momentum Vs Time')


    plt.tight_layout()
    
    fig4 = plt.figure()
    ang_dev_axes = plt.axes()
    ang_dev_axes.plot(orbit.t, momentum_deviation)
    
    ang_dev_axes.set_xlabel('Time (s)', size=10)
    ang_dev_axes.set_ylabel(f'Deviation of Angular Momentum (m^2/s)', size=10)
    ang_dev_axes.set_title('Deviation of Angular Momentum from Initial Value Vs Time')

    plt.tight_layout()

    true_anomoly = []
    for i in range(len(orbit.t)):
        _e = 1 / astro.MU_EARTH * ((vel[i] ** 2 - astro.MU_EARTH / pos[i]) * _rad[i] - (np.dot(_rad[i], _vel[i])) * _vel[i])
        true_anomoly.append(astro.calculate_true_anomoly(_e, _rad[i], _vel[i]))

    fig5 = plt.figure()
    t_anom_axes = plt.axes()

    t_anom_axes.plot(orbit.t, true_anomoly)
    t_anom_axes.set_xlabel('Time (s)', size=10)
    t_anom_axes.set_ylabel(f'True Anomoly (rads)', size=10)
    t_anom_axes.set_title('True Anomoly Vs Time')

    plt.show()


def num_2():

    a = (-30000 * 10 ** 3) 
    e = 1.2
    i = 80
    raan = 180
    w = 10

    # solve for true anomoly such that r = R_earth
    u_rad = np.arccos((((a * (1 - e ** 2)) / astro.RADIUS_EARTH) - 1) / e)
    u = np.degrees(u_rad)

    print(u)
    print(u_rad)

    P = 10000

    _r, _v = astro.standard_to_cartesian(a, e, i, raan, w, u)
    init_state = np.array([_r[0], _r[1], _r[2], _v[0], _v[1], _v[2]])
    tol = 10**-13

    a = time.time()
    orbit = solve_ivp(astro.propogate_2BP, [0, P], init_state, method="RK45", atol=tol, rtol=tol, t_eval=np.linspace(0, P, P+1))
    print(f"Propogation took: {time.time() - a} seconds.")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(orbit.y[0], orbit.y[1], orbit.y[2]) 
    ax.set_xlabel('x (m)', size=10)
    ax.set_ylabel('y (m)', size=10)
    ax.set_zlabel('z (m)', size=10)
    ax.set_title("Orbital Path of Orbit B")
    
    plt.tight_layout()

    pos = []
    vel = []
    acc = []
    for i in range(orbit.t.shape[0]):
        pos.append(math.sqrt(orbit.y[0][i] ** 2 + orbit.y[1][i] ** 2 + orbit.y[2][i] ** 2))
        vel.append(math.sqrt(orbit.y[3][i] ** 2 + orbit.y[4][i] ** 2 + orbit.y[5][i] ** 2))
        a_x = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[0][i]
        a_y = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[1][i]
        a_z = - astro.MU_EARTH / (pos[-1] ** 3) * orbit.y[2][i]
        acc.append(math.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2))

    init_energy = vel[0] ** 2 / 2 - astro.MU_EARTH / pos[0]
    energy_deviation = [astro.calculate_orbital_energy(pos[i], vel[i]) - init_energy for i in range(len(orbit.t))]


    fig2 = plt.figure()
    ax4 = plt.axes()

    ax4.plot(orbit.t, energy_deviation)
    ax4.set_xlabel('Time (s)', size=10)
    ax4.set_ylabel('Deviation from Initial Orbital Energy (J/kg)', size=10)
    ax4.set_title("Deviation from Initial Orbital Energy Vs Time of Orbit B (t = 10000 s)")

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    num_2()
