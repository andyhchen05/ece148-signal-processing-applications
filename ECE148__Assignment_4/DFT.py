# ============================================================
# Manual computation of 32-point DFT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 0, 5, 5, 3, 8, 8])

N = 32

# ------------------------------------------------------------
# Generate sequence
# ------------------------------------------------------------

n = np.arange(N)

f_n = np.zeros(N)

for m in range(1, 8):
    f_n += a[m-1] * np.sin((2*np.pi*m*n)/N)

# ------------------------------------------------------------
# Manual DFT computation
# ------------------------------------------------------------

F_k = np.zeros(N, dtype=complex)

for k in range(N):

    summation = 0

    for n in range(N):

        summation += (
            f_n[n]
            * np.exp(-1j * 2 * np.pi * k * n / N)
        )

    F_k[k] = summation

# normalize


# magnitude
mag = np.abs(F_k)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(12,5))

plt.stem(np.arange(N), mag, basefmt=" ")

plt.xlabel("DFT Bin k")
plt.ylabel(r"$|F(k)|$")

plt.title("32-Point DFT Magnitude Spectrum")

plt.grid(True)

plt.show()
