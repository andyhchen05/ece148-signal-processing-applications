import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Parameters
# =====================================================

beta = 1.0
dt_values = [0.1, 0.2, 0.4, 0.8, 1.6]

theta = np.linspace(0, 2*np.pi, 1000)

# Unit circle
unit_circle = np.exp(1j * theta)

plt.figure(figsize=(10, 10))

# =====================================================
# Plot unit circle
# =====================================================

plt.plot(np.real(unit_circle), np.imag(unit_circle), 'k--', label='Unit Circle')

# =====================================================
# Pole locations for different Δt
# =====================================================

colors = ['r', 'g', 'b', 'm', 'orange']

for i, dt in enumerate(dt_values):

    # -----------------------------
    # Equation Conversion Poles
    # -----------------------------
    r_ec = 1 / np.sqrt(1 + (beta*dt)**2)
    theta_ec = np.arctan(beta*dt)

    z_ec_1 = r_ec * np.exp(1j * theta_ec)
    z_ec_2 = r_ec * np.exp(-1j * theta_ec)

    plt.plot(np.real(z_ec_1), np.imag(z_ec_1), 'o', color=colors[i])
    plt.plot(np.real(z_ec_2), np.imag(z_ec_2), 'o', color=colors[i],
             label=f'EC poles Δt={dt}')

    # -----------------------------
    # Impulse Invariance Poles
    # -----------------------------
    z_ii_1 = np.exp(1j * beta * dt)
    z_ii_2 = np.exp(-1j * beta * dt)

    plt.plot(np.real(z_ii_1), np.imag(z_ii_1), 'x', color=colors[i])
    plt.plot(np.real(z_ii_2), np.imag(z_ii_2), 'x', color=colors[i],
             label=f'II poles Δt={dt}')

# =====================================================
# Plot formatting
# =====================================================

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Pole Locations: Equation Conversion vs Impulse Invariance')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
