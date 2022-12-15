import astro
import numpy as np
import time
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

def num_1_and_2():
    
    target_r_p = astro.RADIUS_EARTH + 1010 * 1000
    interceptor_r_p = astro.RADIUS_EARTH + 1000 * 1000
    
    target_orbital_elements = {
        'a': target_r_p / (1 - 0),
        'e' : 0,
        'i' : 40,
        'RAAN' : 25,
        'w' : 15,
        'nu' : 0
    }
    
    interceptor_orbital_elements = {
        'a': interceptor_r_p / (1 - 0.15),
        'e' : 0.15,
        'i' : 40,
        'RAAN' : 25,
        'w' : 15,
        'nu' : 0
    }
    
    target_state_r, target_v = astro.standard_to_cartesian(target_orbital_elements['a'], target_orbital_elements['e'], target_orbital_elements['i'], target_orbital_elements['RAAN'], target_orbital_elements['w'], target_orbital_elements['nu'])
    interceptor_state_r, interceptor_v = astro.standard_to_cartesian(interceptor_orbital_elements['a'], interceptor_orbital_elements['e'], interceptor_orbital_elements['i'], interceptor_orbital_elements['RAAN'], interceptor_orbital_elements['w'], interceptor_orbital_elements['nu'])
    
    target_state = np.concatenate((target_state_r, target_v))
    interceptor_state = np.concatenate((interceptor_state_r, interceptor_v))
    
    tol = 10**-13
    
    target_period = astro.calculate_orbital_period(target_orbital_elements['a'], astro.MU_EARTH)
    T = round(target_period)
    
    print("Solving with RK45")
    a = time.time()
    
    target_orbit = solve_ivp(astro.propogate_2BP, [0, T], target_state, method='RK45', rtol=tol, atol=tol, t_eval=np.linspace(0, T, T+1))
    interceptor_orbit = solve_ivp(astro.propogate_2BP, [0, T], interceptor_state, method='RK45', rtol=tol, atol=tol, t_eval=np.linspace(0, T, T+1))
    
    b = time.time()
    print("Time: ", b - a)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    ax.plot3D(target_orbit.y[0], target_orbit.y[1], target_orbit.y[2], 'b', label='Target Orbit')
    ax.plot3D(interceptor_orbit.y[0], interceptor_orbit.y[1], interceptor_orbit.y[2], 'r', label='Interceptor Orbit')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.legend()
    
    omega = np.sqrt(astro.MU_EARTH / target_orbital_elements['a']**3)
    
    rel_R = []
    rel_S = []
    rel_W = []
    
    rel_R_v = []
    rel_S_v = []
    rel_W_v = []
    
    cw_x = []
    cw_y = []
    cw_z = []
    
    init_rel_r = None
    init_rel_v = None
    
    timesteps = range(0, round((T+1)/2))
    
    # loop over timesteps
    for i in timesteps:
        
        # calculate RSW unit vectors at current timestep
        r = target_orbit.y[0:3, i] / np.linalg.norm(target_orbit.y[0:3, i])
        s = target_orbit.y[3:6, i] / np.linalg.norm(target_orbit.y[3:6, i])
        w = np.cross(r, s) / np.linalg.norm(np.cross(r, s))
        
        transformation_matrix = np.array([r, s, w])
        
        # calculate relative position and velocity
        rel_r = interceptor_orbit.y[0:3, i] - target_orbit.y[0:3, i]
        rel_v = interceptor_orbit.y[3:6, i] - target_orbit.y[3:6, i]
        
        if i == 0:
            init_rel_r = rel_r
            init_rel_v = rel_v
        
        # transform relative position and velocity to RSW frame
        rel_r_rsw = np.matmul(transformation_matrix, rel_r)
        rel_v_rsw = np.matmul(transformation_matrix, rel_v)
        
        rel_R.append(rel_r_rsw[0])
        rel_S.append(rel_r_rsw[1])
        rel_W.append(rel_r_rsw[2])
        
        rel_R_v.append(rel_v_rsw[0])
        rel_S_v.append(rel_v_rsw[1])
        rel_W_v.append(rel_v_rsw[2])
        
        cw = astro.clohessy_wiltshire(round(rel_R[0]), round(rel_S[0]), round(rel_W[0]), round(rel_R_v[0]), round(rel_S_v[0]), round(rel_W_v[0]), omega, i)
        
        cw_x.append(cw[0])
        cw_y.append(cw[1])
        cw_z.append(cw[2])    
    
    # get round((T_1+1)/2) timesteps from target orbit time
    timesteps = target_orbit.t[0:round((T+1)/2)]
    print(timesteps.size)
    
    # create a 1 by 3 subplot for the relative position and cw components
    fig, axes = plt.subplots(1, 3)
    
    # plot the relative position in RSW frame
    axes[0].plot(timesteps, rel_R)
    axes[0].plot(timesteps, cw_x)
    axes[0].set_title('Relative Position (R Component)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Relative Position (m)')
    axes[0].legend(['Numerical', 'Clohessy-Wiltshire'])
    
    axes[1].plot(timesteps, rel_S)
    axes[1].plot(timesteps, cw_y)
    axes[1].set_title('Relative Position (S Component)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Relative Position (m)')
    axes[1].legend(['Numerical', 'Clohessy-Wiltshire'])
    
    axes[2].plot(timesteps, rel_W)
    axes[2].plot(timesteps, cw_z)
    axes[2].set_title('Relative Position (W Component)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Relative Position (m)')
    axes[2].legend(['Numerical', 'Clohessy-Wiltshire'])
    
    fig.tight_layout()
    
    ind = 0
    while True:
        if abs(cw_x[ind] - rel_R[ind]) / abs(cw_x[ind]) > 1:
            break
        ind += 1
        
    print("T = " + str(timesteps[ind]))
    
    v1, v2 = astro.docking(init_rel_r, init_rel_v, omega, 100)
    
    print("V1 = " + str(v1))
    print("V2 = " + str(v2))
    
        
    plt.show()
    

    
if __name__ == "__main__":
    num_1_and_2()
    