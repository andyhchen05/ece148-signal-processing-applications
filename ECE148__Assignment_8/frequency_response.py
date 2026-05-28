import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Transfer Function

b = 0.001376 * np.array([
     1,
    -5,
     10,
    -10,
     5,
    -1
])

a = np.array([
    1,
    2.9422,
    3.7338,
    2.4807,
    0.8541,
    0.1210
])


# Frequency Response

N = 4096

w, h = signal.freqz(b, a, worN=N)

# Proper symmetric extension to [-pi, pi]

w_full = np.concatenate((-np.flip(w[1:]), w))
h_full = np.concatenate((np.flip(h[1:]), h))

mag = np.abs(h_full)
mag_db = 20 * np.log10(mag + 1e-12)


# Frequency Specifications

Omega_p = (5 * np.pi) / 6
Omega_s = (2 * np.pi) / 3


# Plot 1: Amplitude Response

plt.figure(figsize=(12,5))

plt.plot(w_full, mag, linewidth=2)

# Specification frequencies

plt.axvline( Omega_p, linestyle=':')
plt.axvline(-Omega_p, linestyle=':')

plt.axvline( Omega_s, linestyle=':')
plt.axvline(-Omega_s, linestyle=':')

plt.title('Frequency Response Plot (Amplitude)')

plt.xlabel('Frequency (rad/s)')
plt.ylabel(r'$|H(e^{j\Omega})|$')

plt.xlim([-np.pi, np.pi])

plt.grid(True)

plt.show()


# Plot 2: dB Magnitude Response

plt.figure(figsize=(12,5))

plt.plot(w_full, mag_db, linewidth=2)

# Frequency specification lines

plt.axvline( Omega_p, linestyle=':',
             label=r'$\Omega_p=\frac{5\pi}{6}$ rad/s')

plt.axvline(-Omega_p, linestyle=':',)

plt.axvline( Omega_s, linestyle=':',
             label=r'$\Omega_s=\frac{2\pi}{3}$ rad/s')

plt.axvline(-Omega_s, linestyle=':')

# Attenuation specifications

plt.axhline(-0.5, linestyle=':',
            label=r'$\alpha_p=0.5$ dB',
            color='green')

plt.axhline(-20, linestyle=':',
            label=r'$\alpha_s=20$ dB',
            color='red')

plt.title('Frequency Response Plot (Magnitude)')

plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')

plt.xlim([-np.pi, np.pi])
plt.ylim([-80, 5])

plt.grid(True)
plt.legend()

plt.show()


# Verification

_, Hp = signal.freqz(b, a, worN=[Omega_p])
_, Hs = signal.freqz(b, a, worN=[Omega_s])

Ap = -20*np.log10(np.abs(Hp[0]))
As = -20*np.log10(np.abs(Hs[0]))

print("Passband edge frequency Ωp =", Omega_p)
print("Passband attenuation =", Ap, "dB")

print()

print("Stopband edge frequency Ωs =", Omega_s)
print("Stopband attenuation =", As, "dB")
