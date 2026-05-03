import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 16  # Number of DFT points

# Generate theta from 0 to 2π
theta = np.linspace(0, 2*np.pi, 2000)

# Mapping: k = floor( N/(2π) * θ )
k = np.floor((N / (2 * np.pi)) * theta)
k = np.clip(k, 0, N - 1)

# Plot
plt.figure()
plt.step(theta, k, where='post', linewidth=2)

# Axes lines
plt.axhline(0)
plt.axvline(0)

# Ticks
plt.xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'$2\pi$'])
plt.yticks(range(0, N))

# Labels
plt.xlabel(r'$\theta$')
plt.ylabel(r'$k$')
plt.title('DFT Mapping: $k$ vs $\\theta$')

# Optional grid for clarity
plt.grid(True)

# Save files
plt.savefig('dft_theta_to_k.png')
plt.savefig('dft_theta_to_k.pdf')

plt.show()
