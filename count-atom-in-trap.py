# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:40:11 2024

@author: nuttida_ka
"""

"""
Created on Tue May 21 10:33:39 2024

@author: COMSOL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import axes3d    

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
T = 80e-6  # Temperature in Kelvin
m = 1.443e-25  # Mass of Rubidium-87 atom in kg
mu_B = 9.274e-24  # Bohr magneton in J/T
g_j = 2  # Landé g-factor for simplicity
dt = 1e-5  # Time step for simulation in seconds
num_steps = 10000 # Number of time steps

# Read magnetic field data
data = pd.read_csv('C:/Users/nuttida_ka/Desktop/field from comsol/magnet-Iu=40-Ibias=-0.60-A.txt', delim_whitespace=True)
BB = np.zeros([100, 100, 100])
B = data["mfco.normB1"]
for j in range(100):
    for k in range(100):
        for z in range(100):
            BB[z, k, j] = B[z + k * 100 + j * 10000]

# Get the dimensions of the field grid
field_height, field_width, field_long = BB.shape
x = np.linspace(-2.97, 2.97, field_height) * 1e-3
y = np.linspace(-2.97, 2.97, field_width) * 1e-3
z = np.linspace(6.02, 9.66, field_long) * 1e-3

# Interpolator for the magnetic field magnitude
B_interpolator = RegularGridInterpolator((x, y, z), BB, bounds_error=False, fill_value=None)

# Number of atoms to simulate
num_atoms = 10000

# Function to generate random velocities with Maxwell-Boltzmann distribution
def generate_random_velocities(num_atoms, temperature, mass):
    speeds = np.random.normal(0, np.sqrt(k_B * temperature / mass), num_atoms)
    phi = np.random.uniform(0, 2 * np.pi, num_atoms)
    costheta = np.random.uniform(-1, 1, num_atoms)
    sintheta = np.sqrt(1 - costheta**2)
    directions = np.vstack((sintheta * np.cos(phi), sintheta * np.sin(phi), costheta)).T
    velocities = speeds[:, np.newaxis] * directions
    return velocities

# Initialize positions of atoms using Gaussian distribution
center = np.array([0.0, 0.0, 7.02e-3])  # Center of the Gaussian distribution
positions_initial = np.random.normal(center, 5e-4, (num_atoms, 3))  # 1e-3 is the standard deviation

# Define the initial ellipsoidal region to count atoms in (center and radii)
region_radii = np.array([1e-3, 1e-3, 1e-3])

# Generate initial velocities with random directions
velocities_initial = generate_random_velocities(num_atoms, T, m)


# Function to compute the force on each atom due to the magnetic field gradient
def compute_force(positions):
    epsilon = 1e-9  # Small step for numerical gradient
    
    # Compute numerical gradient of U
    grad_U = np.zeros_like(positions)
    for dim in range(3):
        pos_plus = np.copy(positions)
        pos_minus = np.copy(positions)
        pos_plus[:, dim] += epsilon
        pos_minus[:, dim] -= epsilon
        U_plus = g_j * mu_B * B_interpolator(pos_plus)
        U_minus = g_j * mu_B * B_interpolator(pos_minus)
        grad_U[:, dim] = (U_plus - U_minus) / (2 * epsilon)
    
    force = -grad_U
    return force

# Runge-Kutta 4th order method for updating positions and velocities
def rk4_step(positions, velocities, dt):
    def derivatives(pos, vel):
        force = compute_force(pos)
        return vel, force / m
    
    k1_vel, k1_acc = derivatives(positions, velocities)
    k2_vel, k2_acc = derivatives(positions + 0.5 * dt * k1_vel, velocities + 0.5 * dt * k1_acc)
    k3_vel, k3_acc = derivatives(positions + 0.5 * dt * k2_vel, velocities + 0.5 * dt * k2_acc)
    k4_vel, k4_acc = derivatives(positions + dt * k3_vel, velocities + dt * k3_acc)
    
    new_positions = positions + (dt / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
    new_velocities = velocities + (dt / 6.0) * (k1_acc + 2*k2_acc + 2*k3_acc + k4_acc)
    
    return new_positions, new_velocities

def count_atoms_in_region(positions, center, radii):
    distances = np.linalg.norm((positions - center) / radii, axis=1)
    atoms_in_region = np.sum(distances <= 1)
    return atoms_in_region

# Function to run the simulation
def run_simulation(positions, velocities, with_field=True):
    positions_over_time = []
    atoms_in_region_over_time = []

    for step in range(num_steps):
        # center_of_mass = np.mean(positions, axis=0)  # Calculate center of mass at each time step
        positions_over_time.append(np.copy(positions))
        if with_field:
            positions, velocities = rk4_step(positions, velocities, dt)
        else:
            positions += dt * velocities 

        atoms_in_region = count_atoms_in_region(positions, center, region_radii)
        atoms_in_region_over_time.append(atoms_in_region)

    return np.array(positions_over_time), atoms_in_region_over_time

# Run the simulations
positions_with_field, atoms_in_region_with_field = run_simulation(
    np.copy(positions_initial), np.copy(velocities_initial), with_field=True)
positions_free_expansion, atoms_in_region_free_expansion = run_simulation(
    np.copy(positions_initial), np.copy(velocities_initial), with_field=False)
# Plotting the atom positions over time in 2D scatter plots for final time step
fig, axes = plt.subplots(2, 3, figsize=(20, 14), dpi=200)

# XY Plane
axes[0, 0].scatter(positions_with_field[-1, :, 0], positions_with_field[-1, :, 1], s=1)
axes[0, 0].set_xlabel('X Position')
axes[0, 0].set_ylabel('Y Position')
axes[0, 0].set_xlim(-3e-3,3e-3)
axes[0, 0].set_ylim(-3e-3,3e-3)
axes[0, 0].set_title('XY Plane (With Magnetic Field)')

axes[1, 0].scatter(positions_free_expansion[-1, :, 0], positions_free_expansion[-1, :, 1], s=1)
axes[1, 0].set_xlabel('X Position')
axes[1, 0].set_ylabel('Y Position')
axes[1, 0].set_xlim(-3e-3,3e-3)
axes[1, 0].set_ylim(-3e-3,3e-3)
axes[1, 0].set_title('XY Plane (Free Expansion)')

# XZ Plane
axes[0, 1].scatter(positions_with_field[-1, :, 0], positions_with_field[-1, :, 2], s=1)
axes[0, 1].set_xlabel('X Position')
axes[0, 1].set_ylabel('Z Position')
axes[0, 1].set_xlim(-3e-3,3e-3)
axes[0, 1].set_ylim(4e-3,10e-3)
axes[0, 1].set_title('XZ Plane (With Magnetic Field)')

axes[1, 1].scatter(positions_free_expansion[-1, :, 0], positions_free_expansion[-1, :, 2], s=1)
axes[1, 1].set_xlabel('X Position')
axes[1, 1].set_ylabel('Z Position')
axes[1, 1].set_xlim(-3e-3,3e-3)
axes[1, 1].set_ylim(4e-3,10e-3)
axes[1, 1].set_title('XZ Plane (Free Expansion)')

# YZ Plane
axes[0, 2].scatter(positions_with_field[-1, :, 1], positions_with_field[-1, :, 2], s=1)
axes[0, 2].set_xlabel('Y Position')
axes[0, 2].set_ylabel('Z Position')
axes[0, 2].set_xlim(-3e-3,3e-3)
axes[0, 2].set_ylim(4e-3,10e-3)
axes[0, 2].set_title('YZ Plane (With Magnetic Field)')

axes[1, 2].scatter(positions_free_expansion[-1, :, 1], positions_free_expansion[-1, :, 2], s=1)
axes[1, 2].set_xlabel('Y Position')
axes[1, 2].set_ylabel('Z Position')
axes[1, 2].set_xlim(-3e-3,3e-3)
axes[1, 2].set_ylim(4e-3,10e-3)
axes[1, 2].set_title('YZ Plane (Free Expansion)')

plt.suptitle('Comparison of Atom Positions in 2D Planes at Final Time Step')
plt.show()

# Create a figure
fig = plt.figure(figsize=(15, 15), dpi=200)
X, Y, Z = positions_with_field[-1, :, 0], positions_with_field[-1, :, 1], positions_with_field[-1, :, 2]
# Add a 3D scatter plot to the figure
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(X, Y, Z)
ax1.plot(X, Z, 'r+', zdir='y', zs=0.0035,alpha=0.1)
ax1.plot(Y, Z, 'g+', zdir='x', zs=0.0025,alpha=0.1)
ax1.plot(X, Y, 'k+', zdir='z', zs=0.004,alpha=0.1)
ax1.set_title('3D Scatter Plot')
ax1.set_xlabel('X axis')  # Add space between the label and the axis
ax1.set_xlabel('X axis', labelpad=15)
ax1.set_xlim(-3e-3,3e-3)
ax1.set_ylim(-3e-3,3e-3)
ax1.set_zlim(4e-3,10e-3)
ax1.set_ylabel('Y axis', labelpad=15)
ax1.set_zlabel('Z axis', labelpad=10)
plt.show()
# for angle in range(0, 360):
#    ax1.view_init(angle,30)
#    plt.draw()
#    plt.pause(.001)
#    plt.show()



# Plot the number of atoms in the specified region over time
plt.figure(figsize=(10, 6), dpi = 400)
plt.plot(np.arange(num_steps) * dt, atoms_in_region_with_field, label='With Magnetic Field')
plt.plot(np.arange(num_steps) * dt, atoms_in_region_free_expansion, label='Free Expansion')
plt.xlabel('Time (s)')
plt.ylabel('Number of Atoms in Region')
plt.title('Number of Atoms in Specified Ellipsoidal Region Over Time')
plt.legend()
plt.show()