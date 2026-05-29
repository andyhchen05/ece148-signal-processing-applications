import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# Constants

c = 3e8
eps_r = 6.0
v = c / np.sqrt(eps_r)


# Load Data

data = loadmat("gpr_data.mat")

F  = data['F']              # Shape: (200, 128)
f  = data['f'].flatten()
da = data['da'].flatten()

da = da - da.mean()

Nf, Nx = F.shape


# Range Direction

Nfft_range = 1024

# Range compression
range_profiles = np.fft.ifft(F, n=Nfft_range, axis=0)

# Keep first 43 range bins
Nr_keep = 43

range_profiles = range_profiles[:Nr_keep, :]


# Construct depth axis

df = f[1] - f[0]

dt = 1.0 / (df * Nfft_range)

time = np.arange(Nr_keep) * dt

depth = v * time / 2.0


# Interpolation Procedure (Range)

Ninterp_range = 128

range_interp = np.zeros((Ninterp_range, Nx), dtype=complex)

for col in range(Nx):

    x = range_profiles[:, col]

    # FFT of 43-point sequence
    X = np.fft.fftshift(np.fft.fft(x))

    # Zero-padded spectrum
    Xpad = np.zeros(Ninterp_range, dtype=complex)

    start = (Ninterp_range - Nr_keep) // 2

    Xpad[start:start + Nr_keep] = X

    # Inverse FFT
    x_interp = np.fft.ifft(np.fft.ifftshift(Xpad))

    # Scale correction
    x_interp *= Ninterp_range / Nr_keep

    range_interp[:, col] = x_interp


# Interpolation Procedure (Horizontal)

Ninterp_cross = 2048

final_image = np.zeros((Ninterp_range, Ninterp_cross), dtype=complex)

for row in range(Ninterp_range):

    x = range_interp[row, :]

    # FFT across aperture
    X = np.fft.fftshift(np.fft.fft(x))

    # Zero-pad spectrum
    Xpad = np.zeros(Ninterp_cross, dtype=complex)

    start = (Ninterp_cross - Nx) // 2

    Xpad[start:start + Nx] = X

    # Inverse FFT
    x_interp = np.fft.ifft(np.fft.ifftshift(Xpad))

    # Scale correction
    x_interp *= Ninterp_cross / Nx

    final_image[row, :] = x_interp


image = np.abs(final_image)

image /= np.max(image)

image_db = 20 * np.log10(image + 1e-6)


# Axes

x_interp = np.linspace(da[0], da[-1], Ninterp_cross)

depth_interp = np.linspace(depth[0], depth[-1], Ninterp_range)


# Display

plt.figure(figsize=(14, 6))

extent = [
    x_interp[0],
    x_interp[-1],
    depth_interp[-1],
    depth_interp[0]
]

plt.imshow(
    image_db,
    extent=extent,
    aspect='equal',
    cmap='jet',
    vmin=-40,
    vmax=0
)

plt.xlabel("Distance Along Aperture (m)")
plt.ylabel("Depth (m)")

plt.title("Image Reconstruction of Subsurface Profile")

plt.tight_layout()

plt.savefig("RangeEstimation.png", dpi=150)

plt.show()
