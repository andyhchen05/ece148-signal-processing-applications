import numpy as np
import matplotlib.pyplot as plt

# Harmonic coefficients from perm number 1055388
coeffs = [1, 0, 5, 5, 3, 8, 8]

T = 1

# Continuous-time signal
def signal(t):
    x = np.zeros_like(t, dtype=float)

    for m in range(1, 8):
        x += coeffs[m-1] * np.sin(2*np.pi*m*t/T)

    return x

# ============================================================
# Original 16-point sequence f(n)
# ============================================================

N1 = 16
n1 = np.arange(N1)
t1 = n1 / N1

f = signal(t1)

# ============================================================
# True 64-point sequence g(n)
# ============================================================

N2 = 64
n2 = np.arange(N2)
t2 = n2 / N2

g = signal(t2)

# ============================================================
# FFT interpolation from 16 -> 64
# ============================================================

# Step 1: FFT of f(n)
F = np.fft.fft(f)

# Step 2: Shift FFT to center zero frequency
F_shift = np.fft.fftshift(F)

# Step 3: Zero-pad spectrum
pad = (N2 - N1) // 2

F_padded = np.pad(F_shift, (pad, pad), mode='constant')

# Step 4: Shift back
F_interp = np.fft.ifftshift(F_padded)

# Step 5: Inverse FFT
f_interp = np.fft.ifft(F_interp)

# Step 6: Scale by interpolation factor
f_interp = np.real(f_interp) * (N2 / N1)

# ============================================================
# Plot comparison
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(n2, g, label='Original g(n), N=64', linewidth=2)

plt.plot(n2, f_interp,
         '--',
         label='FFT Interpolated f(n) -> 64',
         linewidth=2)

plt.stem(n1 * 4, f,
         linefmt='gray',
         markerfmt='ko',
         basefmt=' ')

plt.title('FFT Interpolation: 16-point f(n) to 64 points')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.show()
