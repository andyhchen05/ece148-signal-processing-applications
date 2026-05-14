import numpy as np
import matplotlib.pyplot as plt

# Harmonic coefficients from UCSB perm number 1055388
# a1 through a7
coeffs = [1, 0, 5, 5, 3, 8, 8]

# Define the continuous-time signal
# T normalized to 1
T = 1

def signal(t):
    f = np.zeros_like(t, dtype=float)

    for m in range(1, 8):
        f += coeffs[m-1] * np.sin(2 * np.pi * m * t / T)

    return f

# Sampling sizes
Ns = [16, 64, 256]
labels = ['f(n) : N=16', 'g(n) : N=64', 'p(n) : N=256']

# Create plots
plt.figure(figsize=(12, 10))

for i, N in enumerate(Ns):

    # Sample locations
    n = np.arange(N)
    t = n / N

    # Sampled sequence
    x = signal(t)

    # Plot
    plt.subplot(3, 1, i+1)
    plt.stem(n, x)
    plt.title(labels[i])
    plt.xlabel('Sample Index n')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()