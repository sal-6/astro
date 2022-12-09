import astro
import numpy as np
from matplotlib import pyplot as plt


# plot solar force per unit area vs distance from sun
def plot_force():
    
    distances = np.linspace(.1, 11, 1000)
    
    def calc_f_per_area(D, reflectivity=1):
            return 9.1113 * 10 ** -6 * reflectivity  / D ** 2
        
    fs = []
    for d in distances:
        fs.append(calc_f_per_area(d))
        
    plt.plot(distances, fs)
    plt.xlabel("Distance from Sun (AU)")
    plt.ylabel("Force per unit area (N/m^2)")
    
    # create a vertical line at 0 AU labeled as Sun
    plt.axvline(x=0, color='black', linestyle='--')
    plt.text(0.1, 0, "Sun")
    
    # create a vertical line at 0.387 AU labeled as Mercury
    plt.axvline(x=0.387, color='r', linestyle='--')
    plt.text(0.387 + 0.1, 0, "Mercury")
    
    # create a vertical line at 0.723 AU labeled as Venus
    plt.axvline(x=0.723, color='r', linestyle='--')
    plt.text(0.723 + 0.1, 0, "Venus")
    
    # create a vertical line at 1 AU labeled as Earth
    plt.axvline(x=1, color='r', linestyle='--')
    plt.text(1.1, 0, "Earth")
    
    # create a vertical line at 1.54471252752 AU labeled as Mars
    plt.axvline(x=1.54471252752, color='r', linestyle='--')
    plt.text(1.54471252752 + 0.1, 0, "Mars")
    
    # create a vertical line at 4.9520305778 AU labeled as Jupiter
    plt.axvline(x=4.9520305778, color='r', linestyle='--')
    plt.text(4.9520305778 + 0.1, 0, "Jupiter")
    
    # create a vertical line at 9.82800350834 AU labeled as Saturn
    plt.axvline(x=9.82800350834, color='r', linestyle='--')
    plt.text(9.82800350834 + 0.1, 0, "Saturn")
    
    plt.title("Solar Force per Unit Area vs Distance from Sun")
    
    plt.show()
    
if __name__ == "__main__":
    plot_force()