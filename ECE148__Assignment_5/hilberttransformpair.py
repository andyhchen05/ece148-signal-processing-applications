import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0
t = np.linspace(0, T, 1000)

# Example coefficients (you can replace these)
a = np.array([1, 0, 5, 5, 3, 8, 8])

# Construct signals
f = np.zeros_like(t)
f_hat = np.zeros_like(t)

for m in range(1, 8):
    f += a[m-1] * np.cos(2*np.pi*m*t/T)
    f_hat += a[m-1] * np.sin(2*np.pi*m*t/T)

# Plot
plt.figure()
plt.plot(t, f, label='f(t)')
plt.plot(t, f_hat, label='Hilbert Transform f̂(t)', linestyle='--')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.title('Hilbert Transform Pair')
plt.legend()
plt.grid()
plt.show()
