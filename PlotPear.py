import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Define R(θ, φ)
def nuclear_surface(R0, theta, phi, a_lm):
    """
    Calculate nuclear radius as a function of angles using spherical harmonics expansion.
    
    Parameters:
    - R0: base radius
    - theta, phi: angle arrays (in radians)
    - a_lm: dictionary of deformation coefficients, keys are (l, m) tuples

    Returns:
    - R: radius at each (theta, phi)
    """
    R = np.ones_like(theta, dtype=np.complex128)

    for (l, m), a in a_lm.items():
        Y_lm = sph_harm(m, l, phi, theta)  # Note order: (m, l, phi, theta)
        R += a * Y_lm

    return R0 * np.real(R)

# Generate angles
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2*np.pi, 200)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Set deformation coefficients: (λ, μ) : aλμ
a_lm = {
    (0, 0): 0.1,    # monopole
    (1, 0): 0.05,   # dipole z-axis
    (2, 0): 0.2,    # quadrupole
    (3, 0): 0.0     # octupole
}

# Base radius
R0 = 1.0

# Compute radius
R = nuclear_surface(R0, theta_grid, phi_grid, a_lm)

# Convert to Cartesian coordinates for plotting
X = R * np.sin(theta_grid) * np.cos(phi_grid)
Y = R * np.sin(theta_grid) * np.sin(phi_grid)
Z = R * np.cos(theta_grid)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis((R - R.min()) / (R.max() - R.min())),
                rstride=2, cstride=2, antialiased=True, linewidth=0)

ax.set_title("Deformed Nuclear Surface (Monopole–Octupole)", fontsize=14)
ax.set_box_aspect([1,1,1])
ax.axis('off')
plt.show()

