import astro
import numpy as np
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

def num_1():
    r_ijk = np.array([-2981784, 5207055, 3161595])
    v_ijk = np.array([-3384, -4887, 4843])

    print(astro.cartesian_to_standard(r_ijk, v_ijk))


def num_2():
    a = 6786230
    ecc = .01
    incl = 52
    RAAN = 95
    argp = 93
    nu = 300

    print(astro.standard_to_cartesian(a, ecc, incl, RAAN, argp, nu))


def test_ode_solve():

    def ode(t, y):
        return 2*t
    sol = solve_ivp(ode, [0, 5], [0], method="RK45", t_eval=np.linspace(0, 5, 100))
    #sol = odeint(ode, 0, np.linspace(0, 25, 25), full_output=True)

    print(sol)


    #plt.style.use('_mpl-gallery')
    ax = plt.axes(projection='3d')
    ax.plot(sol.t, sol.y[0], linewidth=2.0, )

    plt.show()
    

test_ode_solve()