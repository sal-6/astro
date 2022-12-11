import astro
import numpy as np
from matplotlib import pyplot as plt

def calculate_area_for_one_newton_of_force(f_per_area):
    return 1 / f_per_area

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
    plt.text(-.4, 0, "Sun")
    
    # create a vertical line at 0.387 AU labeled as Mercury
    plt.axvline(x=0.387, color='r', linestyle='--')
    plt.text(0.387, 0, "M")
    
    # create a vertical line at 0.723 AU labeled as Venus
    plt.axvline(x=0.723, color='r', linestyle='--')
    plt.text(0.723, 0, "V")
    
    # create a vertical line at 1 AU labeled as Earth
    plt.axvline(x=1, color='r', linestyle='--')
    plt.text(1.1, 0, "E")
    
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
    
    # calculate the value at each planet
    
    # get the value of fs at the index closest to the planet
    # distance
    i = 0
    while distances[i] < 0.387:
        i += 1
    print("Mercury: " + str(fs[i] * 10**6))
    print("Mercury area: " + str(calculate_area_for_one_newton_of_force(fs[i])) + " | Square side length: " + str(np.sqrt(calculate_area_for_one_newton_of_force(fs[i]))))

    
    
    i = 0
    while distances[i] < 0.723:
        i += 1
    print("Venus: " + str(fs[i] * 10**6))
    print("Venus area: " + str(calculate_area_for_one_newton_of_force(fs[i])) + " | Square side length: " + str(np.sqrt(calculate_area_for_one_newton_of_force(fs[i]))))
    
    i = 0
    while distances[i] < 1:
        i += 1
    print("Earth: " + str(fs[i] * 10**6))
    print("Earth area: " + str(calculate_area_for_one_newton_of_force(fs[i])) + " | Square side length: " + str(np.sqrt(calculate_area_for_one_newton_of_force(fs[i]))))
    
    i = 0
    while distances[i] < 1.54471252752:
        i += 1
    print("Mars: " + str(fs[i] * 10**6))
    print("Mars area: " + str(calculate_area_for_one_newton_of_force(fs[i])) + " | Square side length: " + str(np.sqrt(calculate_area_for_one_newton_of_force(fs[i]))))
    
    i = 0
    while distances[i] < 4.9520305778:
        i += 1
        
    print("Jupiter: " + str(fs[i] * 10**6))
    print("Jupiter area: " + str(calculate_area_for_one_newton_of_force(fs[i])) + " | Square side length: " + str(np.sqrt(calculate_area_for_one_newton_of_force(fs[i]))))
    
    i = 0
    while distances[i] < 9.82800350834:
        i += 1
    print("Saturn: " + str(fs[i] * 10**6))        
    print("Saturn area: " + str(calculate_area_for_one_newton_of_force(fs[i])) + " | Square side length: " + str(np.sqrt(calculate_area_for_one_newton_of_force(fs[i]))))
    
        
    plt.show()
    
if __name__ == "__main__":
    plot_force()