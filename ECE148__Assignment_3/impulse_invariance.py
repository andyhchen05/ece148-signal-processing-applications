import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 1
dt = 0.2

# Frequency range
theta = np.linspace(-np.pi, np.pi, 2000)

# Frequency response H(e^jθ)
z = np.exp(1j * theta)

H = (
    np.sin(beta * dt) * z**(-1)
) / (
    1
    - 2 * np.cos(beta * dt) * z**(-1)
    + z**(-2)
)

# Impulse response
N = 50
n = np.arange(0, N)

h = np.sin(beta * dt * n)

# Create grouped plots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Magnitude response
axs[0].plot(theta, np.abs(H))
axs[0].set_title('Magnitude Response')
axs[0].set_xlabel(r'$\theta$')
axs[0].set_ylabel(r'$|H(e^{j\theta})|$')
axs[0].grid(True)

# Phase response
axs[1].plot(theta, np.angle(H))
axs[1].set_title('Phase Response')
axs[1].set_xlabel(r'$\theta$')
axs[1].set_ylabel(r'$\angle H(e^{j\theta})$ (radians)')
axs[1].grid(True)

# Impulse response
axs[2].stem(n, h)
axs[2].set_title('Impulse Response')
axs[2].set_xlabel('n')
axs[2].set_ylabel('h[n]')
axs[2].grid(True)

plt.tight_layout()
plt.show()
