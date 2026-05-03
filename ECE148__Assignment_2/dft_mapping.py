import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 16          # Number of DFT points
delta_t = 1     # Sampling interval (change if needed)

# Generate omega_x range (multiple periods to show aliasing)
omega_x = np.linspace(-4*np.pi, 4*np.pi, 4000)

# Step 1: map to theta
theta = (omega_x * delta_t) % (2 * np.pi)

# Step 2: map to k
k = np.floor((N / (2 * np.pi)) * theta)
k = np.clip(k, 0, N - 1)

# Plot
plt.figure()
plt.step(omega_x, k, where='post', linewidth=2)

# Axes lines
plt.axhline(0)
plt.axvline(0)

# Labels
plt.xlabel(r'$\omega_x$')
plt.ylabel(r'$k$')
plt.title('DFT Frequency Mapping: $\omega_x vs. k$')

# Show periodicity ticks
omega0 = 2 * np.pi / delta_t
plt.xticks(
    [-2*omega0, -omega0, 0, omega0, 2*omega0],
    [r'$-2\omega_0$', r'$-\omega_0$', '0', r'$\omega_0$', r'$2\omega_0$']
)

plt.yticks(range(0, N))

# Grid for clarity
plt.grid(True)

# Save files
plt.savefig('wx_to_k.png')
plt.savefig('wx_to_k.pdf')

plt.show()
