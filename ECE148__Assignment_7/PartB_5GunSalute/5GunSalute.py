import numpy as np
from scipy.io import wavfile


# Constants
c = 343.6          
d = 0.20           

angles = [-60, -30, 0, 30, 60]


# Load mono audio
fs, x = wavfile.read("hw_7_partA_mic1.wav")

x = x.astype(float)

# Normalize
x = x / np.max(np.abs(x))


# Create stereo output
left_total = []
right_total = []

for angle in angles:

    theta = np.radians(angle)

    # ITD delay
    tau = (d * np.sin(theta)) / c

    delay_samples = int(np.round(abs(tau) * fs))

    # Create delayed channels
    if angle < 0:
        # Source on left:
        # delay right ear

        left = x

        right = np.concatenate((
            np.zeros(delay_samples),
            x
        ))

        left = np.concatenate((
            left,
            np.zeros(delay_samples)
        ))

    elif angle > 0:
        # Source on right:
        # delay left ear

        right = x

        left = np.concatenate((
            np.zeros(delay_samples),
            x
        ))

        right = np.concatenate((
            right,
            np.zeros(delay_samples)
        ))

    else:
        # Center position

        left = x
        right = x

    # Store segments
    left_total.append(left)
    right_total.append(right)


# Equalize segment lengths
max_len = max(len(seg) for seg in left_total)

for i in range(len(left_total)):

    left_total[i] = np.pad(
        left_total[i],
        (0, max_len - len(left_total[i]))
    )

    right_total[i] = np.pad(
        right_total[i],
        (0, max_len - len(right_total[i]))
    )


# Concatenate all positions
left_channel = np.concatenate(left_total)
right_channel = np.concatenate(right_total)

# Form stereo signal
stereo = np.column_stack((left_channel, right_channel))

# Normalize
stereo = stereo / np.max(np.abs(stereo))

# Convert to int16
stereo = (32767 * stereo).astype(np.int16)


# Save output
wavfile.write("5GunSalute.wav", fs, stereo)