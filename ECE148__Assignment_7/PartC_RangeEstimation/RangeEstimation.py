import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# Constants
c = 3e8
eps_r = 6.0
v = c / np.sqrt(eps_r)


# Load data
data = loadmat("gpr_data.mat")
F    = data['F']             
f    = data['f'].flatten()   
da   = data['da'].flatten()  

da = da - da.mean()        

Nfreq, Npos = F.shape


# Frequency parameters
B  = f[-1] - f[0]           
df = f[1]  - f[0]


# Zero-padded IFFT for finer range resolution
Nfft = 2048                  
range_profiles = np.fft.ifft(F, n=Nfft, axis=0)
image = np.abs(range_profiles)


# Depth axis  
dt    = 1.0 / (df * Nfft)    
time  = np.arange(Nfft) * dt
depth = v * time / 2.0       


# Gate out surface reflection
gate_depth   = 0.00         
gate_bins    = np.searchsorted(depth, gate_depth)
image[:gate_bins, :] = 0


# Trim to 30 cm display depth
max_depth  = 0.30
valid      = depth <= max_depth
depth_disp = depth[valid]
image_disp = image[valid, :]


# Normalize & log-compress
image_disp = image_disp / np.max(image_disp)
image_db   = 20 * np.log10(image_disp + 1e-6)


# Clip dB range for display contrast
vmin, vmax = -40, 0     


# Plot
dx     = da[1] - da[0]                        
dz     = depth_disp[1] - depth_disp[0]        

plt.figure(figsize=(12, 5))
extent = [da[0], da[-1], depth_disp[-1], depth_disp[0]]

plt.imshow(image_db, extent=extent, aspect='equal',
           cmap='jet', vmin=vmin, vmax=vmax)

plt.xlabel("Distance Along Aperture (m)")
plt.ylabel("Depth (m)")
plt.title("GPR Image — Broida Hall Walkway")
plt.colorbar(label="Intensity (dB)")
plt.tight_layout()
plt.savefig("gpr_image.png", dpi=150)
plt.show()