import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Fourier Series Spectrum Plot
# ============================================================

# Replace these with the actual digits from your 7-digit perm number
# Example only:
a = np.array([1, 0, 5, 5, 3, 8, 8])

# Harmonic indices
m_pos = np.arange(1, 8)
m_neg = -m_pos[::-1]

# Complex Fourier coefficients
F_pos = -1j * a / 2
F_neg =  1j * a[::-1] / 2

# Combine into full spectrum
m = np.concatenate((m_neg, [0], m_pos))

F = np.concatenate((
    F_neg,
    [0],
    F_pos
))

# ============================================================
# Magnitude Spectrum
# ============================================================

plt.figure(figsize=(10, 5))
plt.stem(m, np.abs(F), basefmt=" ")

plt.xlabel("Harmonic Index m")
plt.ylabel(r"$|F_m|$")
plt.title("Magnitude Spectrum of Fourier Series Coefficients")
plt.grid(True)

plt.show()
