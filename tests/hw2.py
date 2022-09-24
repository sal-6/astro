import astro
import numpy as np
from matplotlib import pyplot as plt

def num_1():

    sun = astro.Body(1 * 10 ** 30, np.array([0., 0., 0.]), np.array([0., 0., 0.]))

    earth = astro.Body(5.972 * 10 ** 24, np.array([1.4710 * 10 ** 11, 0., 0.]), np.array([0., 3.0287 * 10 ** 3, 0.]))


    bodies = [sun, earth]
    
    sim = astro.NBody(bodies, 365 * 24 * 60 * 60 , 60)
    sim.run()

    # get every 0th index of the position array from sun
    x = [i[0] for i in sim.bodies[0]._r_history]
    print(x[0:5])
    print(sun._r_history[0:5])

    # plot the positions of the bodies in 3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for body in sim.bodies:
        x = [i[0] for i in body._r_history]
        y = [i[1] for i in body._r_history]
        z = [i[2] for i in body._r_history]

        ax.plot(x, y, z)
    
    plt.show() 



def np_test():
    _F = np.array([0., 0., 0.])

    _F += np.array([1., 0., 0.])



num_1()