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
    T = 40000000 * 5
    
    def apply_at_45_degree_to_sun(spacecraft_state, sun_state):
        _r_sun = sun_state[0:3]
        _v_sun = sun_state[3:6]
        
        _r_sc = spacecraft_state[0:3]
        _v_sc = spacecraft_state[3:6]
        
        _r_sun_to_sc = _r_sc - _r_sun
        
        _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
        
        _force_dir = astro.rodrigues_rotation_formula(_r_sun_to_sc, _momentum_dir, np.pi/4)

        # normalize the force direction
        _force_dir = _force_dir / np.linalg.norm(_force_dir)
        
        return _force_dir
            
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(astro.NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, 80, 1, np.pi/4, apply_at_45_degree_to_sun), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
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
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, 0, color='red')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Paths of the Bodies")
    ax.legend(loc="best")
    
    axisEqual3D(ax)
    
    plt.show()
    

def crank_up():
    
    sun_init_state = np.array([0, 0, 0, 0, 0, 0])
                             
    sc_init_state = np.array([1.500933757963485 * 10 ** 11, 1.641110930187314 * 10 ** 9, -1.297659862625296 * 10 ** 6,
                             -8.061144129033480 * 10 ** 2, 2.968583925292633 * 10 ** 4, -1.022503317926748 * 10 ** 0]) # using earth orbit as initial state
    
    masses = np.array([1.989e30, MASS_SC])
    
    
    bodies = np.array([sun_init_state, sc_init_state])
    
    tol = 1e-13
    T = 365 * 24 * 60 * 60 * 25
    
    def apply_up(spacecraft_state, sun_state):
        _r_sun = sun_state[0:3]
        _v_sun = sun_state[3:6]
        
        _r_sc = spacecraft_state[0:3]
        _v_sc = spacecraft_state[3:6]
        
        _r_sun_to_sc = _r_sc - _r_sun
        
        _momentum_dir = np.cross(_r_sun_to_sc, _v_sc)
        
        _force_dir = _momentum_dir / np.linalg.norm(_momentum_dir)
        
        return _force_dir
            
    print("Solving with RK45")
    t_start = time.time()
    
    solver = solve_ivp(astro.NBodySolarSailGeneralDirection, (0, T), bodies.flatten(), args=(masses, 80, 1, np.pi/4, apply_up), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    nominal = solve_ivp(astro.NBodyODE, (0, T), bodies.flatten(), args=(masses,), method='RK45', atol=tol, rtol=tol, t_eval=np.arange(0, T, 1000))
    
    print(f"Time to Propgate: {time.time() - t_start}")
    
    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # plot the spacecraft
    x = solver.y[6]
    y = solver.y[7]
    z = solver.y[8]
    ax.plot(x, y, z, linewidth=0.5, label="Solar Sail")
    
    # plot the spacecraft nominal
    x = nominal.y[6]
    y = nominal.y[7]
    z = nominal.y[8]
    ax.plot(x, y, z, linewidth=0.5, label="Nominal")
    
    # plot a point at 0 0 0
    ax.scatter(0, 0, 0, color='red')
    
    # label the axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title("Paths of the Bodies")
    ax.legend(loc="best")
    
    axisEqual3D(ax)
    
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
    crank_up()