import numpy as np
import matplotlib.pyplot as plt

# Harmonic coefficients from perm number 1055388
coeffs = [1, 0, 5, 5, 3, 8, 8]

T = 1

# ============================================================
# Continuous-time signal
# ============================================================

def signal(t):

    x = np.zeros_like(t, dtype=float)

    for m in range(1, 8):
        x += coeffs[m-1] * np.sin(2*np.pi*m*t/T)

    return x

# ============================================================
# 16-point sampled sequence f(n)
# ============================================================

N1 = 16

n1 = np.arange(N1)
t1 = n1 / N1

f = signal(t1)

# ============================================================
# True 256-point sequence p(n)
# ============================================================

N2 = 256

n2 = np.arange(N2)
t2 = n2 / N2

p = signal(t2)

# ============================================================
# FFT interpolation: 16 -> 256
# ============================================================

# FFT of original sequence
F = np.fft.fft(f)

# Shift zero frequency to center
F_shift = np.fft.fftshift(F)

# Zero padding amount
pad = (N2 - N1) // 2

# Pad spectrum with zeros
F_padded = np.pad(F_shift,
                  (pad, pad),
                  mode='constant')

# Shift back
F_interp = np.fft.ifftshift(F_padded)

# Inverse FFT
f_interp = np.fft.ifft(F_interp)

# Scale amplitude
f_interp = np.real(f_interp) * (N2 / N1)

# ============================================================
# Plot comparison
# ============================================================

plt.figure(figsize=(14,6))

# True 256-point sequence
plt.plot(n2,
         p,
         label='Original p(n), N=256',
         linewidth=2)

# Interpolated sequence
plt.plot(n2,
         f_interp,
         '--',
         label='FFT Interpolated f(n) → 256',
         linewidth=2)

# Original 16 samples
plt.stem(n1 * 16,
         f,
         linefmt='gray',
         markerfmt='ko',
         basefmt=' ',
         label='Original 16 Samples')

plt.title('FFT Interpolation: 16-point f(n) to 256 points')

plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.grid(True)
plt.legend()

plt.show()

print("Mean squared error:",
      np.mean(error**2))
