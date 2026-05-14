import numpy as np
import matplotlib.pyplot as plt

# Example 16x16 data
f = np.random.rand(16,16)

# Step 1: 2D FFT
F = np.fft.fft2(f)

# Step 2: Shift spectrum
F_shift = np.fft.fftshift(F)

# Step 3: Zero-pad to 64x64
F_pad = np.zeros((64,64), dtype=complex)

start = 24
end = 40

F_pad[start:end, start:end] = F_shift

# Step 4: Shift back
F_interp = np.fft.ifftshift(F_pad)

# Step 5: Inverse FFT
g = np.fft.ifft2(F_interp)

# Step 6: Scale
g = np.real(g) * 16

# Display
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(f)
plt.title('Original 16x16')

plt.subplot(1,2,2)
plt.imshow(g)
plt.title('Interpolated 64x64')

plt.show()
