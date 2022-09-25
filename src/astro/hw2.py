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


def calculate_eccentricity_vector(_r, _v, mu=astro.MU_EARTH):
    """Calculates eccentricity vector

    Args:
        _r (arr): Radius vector
        _v (arr): Velocity vector
        mu (float, optional): Specific gravity. Defaults to astro.MU_EARTH.
    
    Returns:
        arr: Eccentricity vector
    """

    _e = 1 / mu * ((np.linalg.norm(_v) ** 2 - mu / np.linalg.norm(_r)) * _r - (np.dot(_r, _v)) * _v)
    return _e


class Body():
    def __init__(self, mass, _r_0, _v_0, name=""):
        self.name = name
        self.mass = mass
        self._r_0 = _r_0
        self._v_0 = _v_0
        
        self._r_history = [_r_0]
        self._v_history = [_v_0]
        self.time = [0]

    def get_state(self):
        return self._r_history[-1].tolist() + self._v_history[-1].tolist()

    def step_states(self, _F, dt):
        _a = _F / self.mass
        self._v_history.append(self._v_history[-1] + _a * dt)
        self._r_history.append(self._r_history[-1] + self._v_history[-1] * dt)
        self.time.append(self.time[-1] + dt)


# ! IMPORTANT: This does not work. Need to figure out why ...
class NBody():
    def __init__(self, bodies, t_prop, dt=1):
        self.bodies = bodies
        self.t_prop = t_prop
        self.dt = dt
        
        self.timesteps = []

    def run(self):
        
        t = 0
        while t < self.t_prop:
            for body in self.bodies:
                # calculate net force due to other bodies
                _F = np.array([0., 0., 0.])
                for other_body in self.bodies:
                    if other_body != body:
                        _r_to_body = body._r_history[-1] - other_body._r_history[-1]
                        force = -astro.G * other_body.mass * body.mass / np.linalg.norm(_r_to_body) ** 3 * _r_to_body
                        _F = np.add(_F, force)
                        if t == 0:
                            print(f"{body.name} is being pulled by {other_body.name} with a force of {force}")

                if t == 0:
                    print(_F)
                    print()
                body.step_states(_F, self.dt)

            # progress bar of the simulation
            percent = t / self.t_prop * 100
            print(f"\r{percent:.2f}%", end="")
            
            self.timesteps.append(t)
            t += self.dt
            
        print()

    
    def calculate_angular_momentum_mag(self):

        h = []
        for i in range(len(self.bodies[0].time)):
            h_step = 0
            for body in self.bodies:
                r = body._r_history[i].astype(np.float64)
                v = body._v_history[i].astype(np.float64)
                h_step += np.linalg.norm(astro.calculate_angular_momentum(r, v))
            h.append(h_step)
        
        return h


def NBodyODE(t, state, masses):
    """Calculates the state derivative for the NBody problem
    
    Args:
        t (float): Time
        state (arr): State vector with form:
            [r_1_x, r_1_y, r_1_z, v_1_x, v_1_y, v_1_z, r_2_x, r_2_y, r_2_z, v_2_x, v_2_y, v_2_z, ...]
        masses (arr): Masses of the bodies in the same order as the state vector:
            [m_1, m_2, ...]
    
    """
    
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
        _a = _F / masses[i]
        state_der[i][0] = _v_body[0]
        state_der[i][1] = _v_body[1]
        state_der[i][2] = _v_body[2]
        state_der[i][3] = _a[0]
        state_der[i][4] = _a[1]
        state_der[i][5] = _a[2]

    return state_der.flatten()

    
def calculate_potential_energy(mass1, mass2, _r1, _r2):
    return -1 * astro.G * mass1 * mass2 / np.linalg.norm(_r1 - _r2)


def calculate_kinetic_energy(mass, _v):
    return 0.5 * mass * np.linalg.norm(_v) ** 2


