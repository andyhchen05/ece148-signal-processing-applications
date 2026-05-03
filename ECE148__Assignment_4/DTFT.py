# ============================================================
# DTFT of the 32-point sequence f[n]
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# coefficients from perm number
a = np.array([1, 0, 5, 5, 3, 8, 8])

N = 32

# ------------------------------------------------------------
# Generate sampled sequence f[n]
# ------------------------------------------------------------

n = np.arange(N)

f_n = np.zeros(N)

for m in range(1, 8):
    f_n += a[m-1] * np.sin(2 * np.pi * m * n / N)

# ------------------------------------------------------------
# Compute DTFT
# ------------------------------------------------------------

Omega = np.linspace(-np.pi, np.pi, 4000)

F_dtft = np.zeros(len(Omega), dtype=complex)

for k in range(N):
    F_dtft += f_n[k] * np.exp(-1j * Omega * k)

# ------------------------------------------------------------
# Plot DTFT Magnitude
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(Omega, np.abs(F_dtft))

plt.xlabel(r'Normalized Frequency $\Omega$ (rad/sample)')
plt.ylabel(r'$|F(e^{j\Omega})|$')

plt.title('DTFT of 32-Point Sequence')

plt.grid(True)

plt.xlim([-np.pi, np.pi])

plt.show()
