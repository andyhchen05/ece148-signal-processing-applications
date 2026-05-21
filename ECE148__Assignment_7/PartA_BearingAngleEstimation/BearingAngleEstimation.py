import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate


# Parameters
d = 2.0          
c = 343.6        
tau_max = d / c  


# Load WAV files
fs1, x1 = wavfile.read("hw_7_partA_mic1.wav")
fs2, x2 = wavfile.read("hw_7_partA_mic2.wav")

# Ensure same sampling rate
assert fs1 == fs2

fs = fs1

# Convert to float
x1 = x1.astype(float)
x2 = x2.astype(float)


# Cross-correlation
corr = correlate(x1, x2, mode='full')

# Lag indices
lags = np.arange(-len(x1) + 1, len(x2))

# Convert lag to time delay
time_delays = lags / fs

# Normalize correlation
corr = corr / np.max(np.abs(corr))


# Plot time-delay profile
plt.figure(figsize=(10,5))
plt.plot(time_delays * 1000, corr)

plt.xlim([-tau_max * 1000, tau_max * 1000])

plt.xlabel("Time Delay (ms)")
plt.ylabel("Normalized Correlation")
plt.title("Cross-Correlation Time-Delay Profile")
plt.grid(True)


# Convert time delay to bearing angle
# Physical constraint:
# |c*tau/d| <= 1
valid = np.abs(c * time_delays / d) <= 1

valid_delays = time_delays[valid]
valid_corr = corr[valid]

angles_rad = np.arcsin(c * valid_delays / d)
angles_deg = np.degrees(angles_rad)


# Plot bearing-angle profile
plt.figure(figsize=(10,5))
plt.plot(angles_deg, valid_corr)

plt.xlim([-90, 90])

plt.xlabel("Bearing Angle (degrees)")
plt.ylabel("Normalized Correlation")
plt.title("Bearing-Angle Estimation Profile")
plt.grid(True)


# Estimate peak angle
peak_index = np.argmax(valid_corr)

estimated_angle = angles_deg[peak_index]
estimated_delay = valid_delays[peak_index]

print("Estimated Time Delay:")
print(f"{estimated_delay:.6e} seconds")

print("Estimated Bearing Angle:")
print(f"{estimated_angle:.2f} degrees")

plt.show()