# Colab-ready single cell: spectral Poisson + Gauss-certificate closures + LS inversion
# Plots with matplotlib and summary prints at the end.

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt

# ================= Parameters =================
L = 2.0
N = 128
dx = L / N
g = 0.5
G_true = 6.67430e-11
c_true = 2.99792458e8
kappa_true = 8 * np.pi * G_true / c_true**2  # ~2.96e-27

# Rescale so fields are O(1)
scale = 1e26
kappa_eff = kappa_true * scale

# ================= Grid & density (two 3D Gaussians) =================
x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

x0 = L / 2 * np.ones(3)
r1 = x0 + np.array([0.10*L, -0.05*L, 0.07*L])
r2 = x0 + np.array([-0.12*L,  0.03*L, -0.06*L])
sigma1 = 0.08 * L
sigma2 = 0.07 * L
A1, A2 = 1.0, 0.8

dist1 = np.sqrt((X - r1[0])**2 + (Y - r1[1])**2 + (Z - r1[2])**2)
dist2 = np.sqrt((X - r2[0])**2 + (Y - r2[1])**2 + (Z - r2[2])**2)

rho = A1 * np.exp(-dist1**2 / (2 * sigma1**2)) - A2 * np.exp(-dist2**2 / (2 * sigma2**2))
rho -= np.mean(rho)  # <rho> = 0
rho *= scale         # rescale

# ================= Spectral Poisson =================
kx = 2 * np.pi * fftfreq(N, dx)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

rho_hat = fftn(rho)
phi_hat = np.zeros_like(rho_hat, dtype=complex)
mask = K2 > 0

# Correct sign: (-K^2) * phi_hat = -(kappa_eff/g) * rho_hat  => phi_hat = +(kappa_eff/g) * rho_hat / K^2
phi_hat[mask] = (kappa_eff / g) * rho_hat[mask] / K2[mask]
phi = np.real(ifftn(phi_hat))

ln_n = g * phi
ln_n -= np.mean(ln_n)   # mean gauge

# ================= Spectral Laplacian & RHS =================
ln_n_hat = fftn(ln_n)
lap_ln_n_hat = -K2 * ln_n_hat
lap_ln_n = np.real(ifftn(lap_ln_n_hat))
rhs = -kappa_eff * rho

# ================= Pointwise metrics =================
rel_l2 = np.linalg.norm(lap_ln_n - rhs) / np.linalg.norm(rhs)
corr = np.corrcoef(lap_ln_n.flatten(), rhs.flatten())[0, 1]

# ================= Flux closures & LS inversion =================
radii = np.linspace(0.06 * L/2, 0.46 * L/2, 20)
F_js, I_js, rel_closures = [], [], []

dist_to_x0 = np.sqrt((X - x0[0])**2 + (Y - x0[1])**2 + (Z - x0[2])**2)
for R in radii:
    mask_vol = dist_to_x0 <= R
    int_lap = np.sum(lap_ln_n[mask_vol]) * dx**3
    int_rho = np.sum(rho[mask_vol]) * dx**3
    F_js.append(int_lap)           # F_j = ∮ ∇ln n·dS = ∫ ∇²ln n dV
    I_js.append(int_rho)           # I_j = ∫ ρ dV
    rel = abs(int_lap + kappa_eff * int_rho) / (abs(int_lap) + 1e-30) * 100.0
    rel_closures.append(rel)

F_js = np.array(F_js)
I_js = np.array(I_js)
rel_closures = np.array(rel_closures)

# LS fit: minimize Σ(F_j + k I_j)^2  => k_LS = - (Σ F I) / (Σ I^2)
num = np.dot(F_js, I_js)
den = np.dot(I_js, I_js) + 1e-30
k_LS_eff = - num / den                      # should match kappa_eff
kappa_hat = k_LS_eff / scale                # back to physical kappa
G_flux = (c_true**2 / (8 * np.pi)) * kappa_hat
c_flux = np.sqrt(8 * np.pi * G_true / kappa_hat)

# ================= Plots =================
# 1) Flux closure (%) vs radius
plt.figure(figsize=(6,4))
plt.plot(radii, rel_closures, marker='o')
plt.xlabel("Radius R")
plt.ylabel("Flux closure (%)")
plt.title("Gauss-certificate flux closures vs radius")
plt.grid(True)
plt.show()

# 2) F vs I with LS line
plt.figure(figsize=(6,4))
plt.scatter(I_js, F_js, s=16)
Ii = np.linspace(I_js.min(), I_js.max(), 100)
Fi = -k_LS_eff * Ii
plt.plot(Ii, Fi, linewidth=2)
plt.xlabel("I_j = ∫_V ρ dV")
plt.ylabel("F_j = ∮_∂V ∇ln n · dS")
plt.title("LS fit: F + k I ≈ 0")
plt.grid(True)
plt.show()

# ================= Summary prints =================
print(f"Rel-L2: {rel_l2:.2e}")
print(f"Corr(LHS, RHS): {corr:.7f}")
print("Flux closures (%):")
print(", ".join([f"{v:.3f}" for v in rel_closures]))
print(f"k_LS (effective): {k_LS_eff:.6e}")
print(f"kappa_hat (physical): {kappa_hat:.6e}")
print(f"G_flux (from kappa_hat, given c): {G_flux:.6e}")
print(f"c_flux (from kappa_hat, given G): {c_flux:.6e}")
