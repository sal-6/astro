import astro
import numpy as np
from matplotlib import pyplot as plt

def num_1():

    sun = astro.Body(
        1988500 * 10 ** 24, 
        np.array([-1.359 * 10 ** 9, 1.4825 * 10 ** 8, 3.04469 * 10 ** 7]), 
        np.array([-4.6449 * 10 ** -1, -1.57316 * 10 ** 1, 1.35459 * 10 ** -1]),
        name="Sun")

    earth = astro.Body(
        5.972 * 10 ** 24, 
        np.array([1.487339 * 10 ** 11, 1.789363 * 10 ** 9, 2.914929 * 10 **7]), 
        np.array([-8.066 * 10 ** 2, 2.96701 * 10 ** 4, -8.87044 * 10 ** -1]),
        name="Earth")

    mars = astro.Body(
        6.4171 * 10 ** 23, 
        np.array([1.7616 * 10 ** 11, 1.2209 * 10 ** 11, -1.76837 * 10 ** 9]), 
        np.array([-1.279 * 10 ** 4, 2.20267 * 10 ** 4, 7.75949 * 10 ** 2]),
        name="Mars")

    venus = astro.Body(
        4.8675 * 10 ** 24,
        np.array([-1.037586154563294* 10 ** 11, 3.251828282527976 * 10 ** 10, 6.383492836090492 * 10 ** 9]),
        np.array([-1.072525999932200 * 10 ** 4, -3.356733922088807 * 10 ** 4, 1.583732102290671 * 10 ** 2]),
        name="Venus")

    bodies = [sun, earth, venus]
    
    sim = astro.NBody(bodies, 365 * 24 * 60 * 60 * 1 , 60 * 60)
    sim.run()

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