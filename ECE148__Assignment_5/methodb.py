import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

N = 16
T = 1.0
dt = T / N

# UCSB perm number coefficients: 1055388
a = np.array([1, 0, 5, 5, 3, 8, 8])

# ============================================================
# Generate sampled signal
# ============================================================

n = np.arange(N)

f = np.zeros(N)

for m in range(1, 8):
    f += a[m-1] * np.cos(2*np.pi*m*n/N)

# ============================================================
# Hilbert Transform using FFT
# ============================================================

F = np.fft.fft(f)

H = np.zeros(N, dtype=complex)

H[0] = 0
H[1:N//2] = -1j
H[N//2] = 0
H[N//2+1:] = 1j

F_hilbert = F * H

f_hat = np.real(np.fft.ifft(F_hilbert))

# ============================================================
# Analytic Signal
# ============================================================

f_analytic = f + 1j*f_hat

F_analytic = np.fft.fft(f_analytic)

# ============================================================
# Plot 1: Original Sequence f(n)
# ============================================================

plt.figure(figsize=(10,4))

plt.stem(n, f)

plt.title('Original Sampled Sequence $f(n)$')
plt.xlabel('Sample Index n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# Plot 2: Hilbert Transform f_hat(n)
# ============================================================

plt.figure(figsize=(10,4))

plt.stem(n, f_hat)

plt.title('Hilbert Transform $\\hat{f}(n)$')
plt.xlabel('Sample Index n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# Plot 3: Orthogonality Visualization
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(n, f, marker='o', label='f(n)')
plt.plot(n, f_hat, marker='s', label='f_hat(n)')

plt.title('Orthogonality of $f(n)$ and $\\hat{f}(n)$')
plt.xlabel('Sample Index n')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# Plot 4: Magnitude Spectrum of Original Signal
# ============================================================

plt.figure(figsize=(10,4))

plt.stem(np.arange(N), np.abs(F))

plt.title('Magnitude Spectrum |F(k)|')
plt.xlabel('Frequency Bin k')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# Plot 5: Magnitude Spectrum of Analytic Signal
# ============================================================

plt.figure(figsize=(10,4))

plt.stem(np.arange(N), np.abs(F_analytic))

plt.title('Magnitude Spectrum of Analytic Signal $|\\tilde{F}(k)|$')
plt.xlabel('Frequency Bin k')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()

