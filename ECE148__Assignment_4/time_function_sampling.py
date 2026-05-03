import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Time Function and Sampling
# ============================================================

# Fourier coefficients from perm number A0V5C88
a = np.array([1, 0, 5, 5, 3, 8, 8])

# Period
T = 1

# Fundamental angular frequency
w0 = 2 * np.pi / T

# ============================================================
# Continuous-Time Signal f(t)
# ============================================================

# Dense time axis for smooth plot
t = np.linspace(0, T, 1000)

# Construct f(t)
f_t = np.zeros_like(t)

for m in range(1, 8):
    f_t += a[m-1] * np.sin(m * w0 * t)

# Plot one full period
plt.figure(figsize=(12, 5))
plt.plot(t, f_t)

plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("One Period of f(t)")
plt.grid(True)

plt.show()


# ============================================================
# 32-Point Sampling
# ============================================================

N = 32

# Sample spacing
dt = T / N

# Sample indices
n = np.arange(N)

# Sample times
t_samples = n * dt

# Sampled sequence f(n)
f_n = np.zeros(N)

for m in range(1, 8):
    f_n += a[m-1] * np.sin(m * w0 * t_samples)

# ============================================================
# Plot sampled points on top of continuous signal
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(t, f_t, label='Continuous f(t)')
plt.stem(t_samples, f_n, basefmt=" ", label='Samples')

plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("32-Point Sampling of f(t)")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# Print 32-point sample sequence
# ============================================================

print("32-point sample sequence f(n):\n")

for i in range(N):
    print(f"n = {i:2d},  f(n) = {f_n[i]: .6f}")
