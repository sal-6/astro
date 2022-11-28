import astro
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.integrate import solve_ivp

MASS_SC = 3 # kg
AREA_SS = 14 * 14 # m^2

def nominal_vs_solarsail():
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
    
    masses = np.array([1.989e30, MASS_SC])
    
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 40000000
    
    print("Solving with RK45")
    t_start = time.time()
    solver = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 100))
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(0, len(solver.y), 6):
        x = solver.y[i]
        y = solver.y[i+1]
        z = solver.y[i+2]
    
        ax.plot(x, y, z, label=f"Body {i//6 + 1}")
        
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Paths of the Bodies")
    ax.legend(loc="best")
    
    plt.show()
    
def nominal_vs_solarsail_1():
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
    
    masses = np.array([1.989e30, MASS_SC])
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 40000000 * 5
    
    print("Solving with RK45")
    t_start = time.time()
    nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 100))
    solver = solve_ivp(astro.NBodySolarSail45, (0, T), bodies.flatten(), args=(masses, 80, 1), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 100))
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    """ for i in range(0, len(solver.y), 6):
        x = solver.y[i]
        y = solver.y[i+1]
        z = solver.y[i+2]
    
        ax.plot(x, y, z, label=f"Body {i//6 + 1}", linewidth=0.5) """
        
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, z, linewidth=0.5)
    
    # plot the spacecraft nominal
    x = nominal.y[6]
    y = nominal.y[7]
    z = nominal.y[8]
    ax.plot(x, y, z, linewidth=0.5)    
        
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Paths of the Bodies")
    ax.legend(loc="best")
    
    plt.show()
    
    
if __name__ == "__main__":
    nominal_vs_solarsail_1()

        
    
    