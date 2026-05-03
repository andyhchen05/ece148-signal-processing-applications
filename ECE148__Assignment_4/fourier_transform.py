# ============================================================
# Fourier Transform F(jw)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# coefficients
a = np.array([1, 0, 5, 5, 3, 8, 8])

# period
T = 1

# fundamental angular frequency
w0 = 2 * np.pi / T

# harmonic indices
m = np.arange(1, 8)

# frequencies
omega_pos = m * w0
omega_neg = -m * w0

# amplitudes
amps = np.pi * a

# ============================================================
# Plot Fourier Transform Magnitude
# ============================================================

plt.figure(figsize=(12, 5))

# positive frequencies
plt.stem(omega_pos, amps, basefmt=" ")

# negative frequencies
plt.stem(omega_neg, amps, basefmt=" ")

plt.xlabel(r'Angular Frequency $\omega$ (rad/s)')
plt.ylabel(r'$|F(j\omega)|$')

plt.title(r'Fourier Transform of Periodic Signal')

plt.grid(True)

plt.show()
