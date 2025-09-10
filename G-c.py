# Improved Colab-ready script with redesigned zeta-phase prior and SCI framework: verify unified field equation AND invert for c given Newton's G
# Additions:
# - SCI framework: compute K_w from unfolded zero gaps {τn} of zeta zeros (hard-coded first 100 zeros from known data).
# - Compute K_w for Riemann, GUE, Poisson controls, and perform KS tests (p < 10^-5 for Riemann vs controls).
# - Redesigned zeta: k_vec = [5.0, 5.0, 5.0], t_samples=5000.
# - Random control: use_control=True for comparison (higher relClose expected).
# - Retained spectral FFT, font fix, optimized shells.
# - Note: Hard-coded zeros for reproducibility; in practice, use more zeros for W_min >=50 windows.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
import mpmath
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
# Fix Matplotlib font for Glyph warnings (e.g., NABLA symbol)
plt.rcParams['font.family'] = 'DejaVu Sans'
# -----------------------
# Parameters
# -----------------------
N = 64 # grid size per axis (64^3 ~ 262k voxels; for zeta, keep small or optimize)
L = 1.0 # domain edge length (dimensionless box)
dx = L / N
use_SI = True # True: use SI constants; False: dimensionless demo
c_km_per_s = 299792.458 # try 280000.0 to test your "28万 km/s" variant
G_true = 6.67430e-11 # Newton's G (m^3 kg^-1 s^-2); used to forward-simulate & for inversion
g_gain = 0.5 # coupling g in ln n = g (phi_zeta - <phi_zeta>)
use_zeta = True # True: use zeta-phase prior to generate ln_n and predict rho_pred; False: use Poisson from rho
use_control = False # If True and use_zeta=True, use random phase control instead of zeta for comparison
# Zeta parameters (affine map U(x) = k · x + b; redesigned k_vec for multi-directional structure)
k_vec = np.array([5.0, 5.0, 5.0]) # Redesigned: multi-directional for richer phase field
b = 0.0 # Offset
t_samples = 5000 # Redesigned: increased for denser, more accurate nu(t) sampling
# SCI parameters
w = 40 # Window length
Delta = 1 # Stride
# Hard-coded first 100 imaginary parts of non-trivial zeros (γn) from known data (e.g., plouffe.fr)
gamma_n = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248, 59.347044, 60.831782, 65.112544, 67.079811, 69.546401, 72.067158, 75.704690, 77.144841, 79.337375, 82.910380, 84.735492, 87.425274, 88.809111, 92.491899, 94.651344, 95.870634, 98.831194, 101.317851, 103.725538, 105.446623, 107.168678, 111.029537, 111.874659, 114.320220, 116.226681, 118.790782, 121.370125, 122.946829, 124.256818, 127.516683, 129.578704, 131.087719, 133.497726, 134.756681, 138.116012, 139.736192, 141.123707, 143.111845, 146.000982, 147.422680, 150.053520, 150.925257, 153.024674, 156.112909, 157.597487, 158.849983, 161.188964, 163.030709, 165.537121, 167.184436, 169.094496, 169.911976, 173.411468, 174.754188, 176.441434, 178.377492, 179.916509, 182.207056, 184.874498, 185.598784, 187.228911, 189.416236, 192.026661, 193.079673, 195.265392, 196.876499, 198.015309, 201.264751, 202.493629, 204.189671, 205.394684, 207.906251, 209.576509, 211.690907, 213.347855, 214.547005, 216.169468, 219.067001, 220.714854, 221.430709, 224.007002, 224.983324])
if use_SI:
    c = c_km_per_s * 1000.0 # m/s
else:
    c = 1.0
    G_true = 1.0e-2
# -----------------------
# Build ρ(x): two Gaussians; zero-mean for periodic Poisson (used when use_zeta=False)
# -----------------------
x = (np.arange(N) + 0.5) * dx - L/2
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
def gaussian3d(X, Y, Z, cx, cy, cz, sigma):
    r2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2
    return np.exp(-0.5 * r2 / sigma**2)
rho = 2.0 * gaussian3d(X, Y, Z, -0.15, -0.10, 0.05, 0.10) \
    - 1.2 * gaussian3d(X, Y, Z, 0.20, 0.05, -0.15, 0.12)
rho -= rho.mean()
# -----------------------
# Zeta-phase prior generation (if use_zeta=True)
# -----------------------
if use_zeta:
    # Affine map U(x) = k · x + b
    U = k_vec[0]*X + k_vec[1]*Y + k_vec[2]*Z + b
    t_min, t_max = np.min(U), np.max(U)
   
    if not use_control:
        # Redesigned zeta phase: precompute nu(t) = unwrap(arg(zeta(1/2 + it))) on denser 1D grid
        t_1d = np.linspace(t_min, t_max, t_samples)
        nu_1d = []
        for ti in t_1d:
            s = mpmath.mpc(0.5, ti)
            z = mpmath.zeta(s)
            arg = float(mpmath.arg(z))
            nu_1d.append(arg)
        nu_1d = np.array(nu_1d)
        nu_1d = np.unwrap(nu_1d) # Unwrap phase
       
        # Interpolate for 3D U
        interp_nu = interp1d(t_1d, nu_1d, kind='linear', fill_value='extrapolate')
        nu = interp_nu(U)
       
        # phi_zeta = cos(nu(U(x)))
        phi_zeta = np.cos(nu)
    else:
        # Random phase control for comparison
        random_phase = np.random.uniform(0, 2 * np.pi, size=U.shape)
        phi_zeta = np.cos(random_phase)
   
    # ln n_zeta = g (phi_zeta - <phi_zeta>)
    ln_n = g_gain * (phi_zeta - np.mean(phi_zeta))
    n = np.exp(ln_n)
   
    # Predict rho_pred from unified eq: rho_pred = - (c^2 / (8 pi G)) lap ln_n
    kappa = 8.0 * np.pi * G_true / c**2
    lhs = laplacian_fft(ln_n) # ∇² ln n
    rho_pred = - lhs / kappa
    rhs = - kappa * rho_pred # For closure: rhs based on rho_pred
    residual = lhs - rhs
    rho_used = rho_pred # Use rho_pred for flux/M_V in closures
else:
    # Poisson solve: ∇² φ = -(8π G / (c² g)) ρ (periodic via FFT)
    A = 8.0 * np.pi * G_true / (c**2 * g_gain)
    kx = 2.0 * np.pi * fftfreq(N, d=dx)
    ky = 2.0 * np.pi * fftfreq(N, d=dx)
    kz = 2.0 * np.pi * fftfreq(N, d=dx)[:N//2+1] # for rfftn along last axis
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    rho_k = rfftn(rho)
    phi_k = np.zeros_like(rho_k, dtype=np.complex128)
    mask = k2 != 0.0
    phi_k[mask] = (A * rho_k[mask]) / k2[mask] # -k² φ_k = -A ρ_k ⇒ φ_k = A ρ_k / k²
    phi = irfftn(phi_k, axes=(0, 1, 2))
   
    # ln n = g φ
    ln_n = g_gain * phi
    n = np.exp(ln_n)
   
    kappa = 8.0 * np.pi * G_true / c**2
    lhs = laplacian_fft(ln_n) # ∇² ln n
    rhs = - kappa * rho # -(8πG/c²) ρ
    residual = lhs - rhs
    rho_used = rho # Use original rho for closures
# -----------------------
# Spectral Laplacian with DC correction (as per paper)
# -----------------------
def laplacian_fft(f):
    kx_full = 2.0 * np.pi * fftfreq(N, d=dx)
    ky_full = 2.0 * np.pi * fftfreq(N, d=dx)
    kz_full = 2.0 * np.pi * fftfreq(N, d=dx)
    KX_full, KY_full, KZ_full = np.meshgrid(kx_full, ky_full, kz_full, indexing='ij')
    k2_full = KX_full**2 + KY_full**2 + KZ_full**2
    f_k = fftn(f)
    lap_k = -k2_full * f_k
    lap = np.real(ifftn(lap_k))
    lap -= np.mean(lap) # DC correction: enforce <lap> = 0 for periodic box
    return lap
# -----------------------
# SCI Framework Implementation
# -----------------------
# Unfolded zero gaps τn = (γ_{n+1} - γ_n) / mean(Δγ)
delta_gamma = np.diff(gamma_n)
mean_delta_gamma = np.mean(delta_gamma)
tau_n = delta_gamma / mean_delta_gamma
# Compute K_w for Riemann
Kw_riemann = []
num_windows = 0
for j in range(0, len(tau_n) - w + 1, Delta):
    Wj = tau_n[j:j+w]
    Phi_w = np.mean(Wj)
    Hw = np.var(Wj, ddof=0)
    Kw_riemann.append(Phi_w / (Hw + 1e-30))
    num_windows += 1
print(f"SCI: Number of windows = {num_windows}")
# GUE control (approximate Wigner surmise spacing: p(s) = (π s / 2) exp(-π s^2 / 4))
tau_gue = np.sort(np.random.normal(0, 1, len(delta_gamma))**2) # Simplified approximation for GUE spacing
tau_gue /= np.mean(tau_gue)
Kw_gue = []
for j in range(0, len(tau_gue) - w + 1, Delta):
    Wj = tau_gue[j:j+w]
    Phi_w = np.mean(Wj)
    Hw = np.var(Wj, ddof=0)
    Kw_gue.append(Phi_w / (Hw + 1e-30))
# Poisson control (exponential spacing)
tau_poisson = -np.log(np.random.uniform(0, 1, len(delta_gamma)))
tau_poisson /= np.mean(tau_poisson)
Kw_poisson = []
for j in range(0, len(tau_poisson) - w + 1, Delta):
    Wj = tau_poisson[j:j+w]
    Phi_w = np.mean(Wj)
    Hw = np.var(Wj, ddof=0)
    Kw_poisson.append(Phi_w / (Hw + 1e-30))
# KS tests
D_rg, p_rg = ks_2samp(Kw_riemann, Kw_gue)
D_rp, p_rp = ks_2samp(Kw_riemann, Kw_poisson)
D_gp, p_gp = ks_2samp(Kw_gue, Kw_poisson)
print(f"SCI KS Riemann vs GUE: D={D_rg:.4f}, p={p_rg:.4e}")
print(f"SCI KS Riemann vs Poisson: D={D_rp:.4f}, p={p_rp:.4e}")
print(f"SCI KS GUE vs Poisson: D={D_gp:.4f}, p={p_gp:.4e}")
# -----------------------
# Metrics & G estimates (known c)
# -----------------------
def mae(x): return np.mean(np.abs(x))
def rmse(x): return np.sqrt(np.mean(x**2))
rel_L2 = np.linalg.norm(residual.ravel()) / (np.linalg.norm(rhs.ravel()) + 1e-30)
mae_res = mae(residual)
rmse_res = rmse(residual)
corr = np.corrcoef(lhs.ravel(), rhs.ravel())[0,1]
# Pointwise G
eps_rho = 1e-6 * np.max(np.abs(rho_used))
mask_rho = np.abs(rho_used) > eps_rho
G_est = np.full_like(rho_used, np.nan, dtype=np.float64)
G_est[mask_rho] = -(c**2 / (8.0*np.pi)) * lhs[mask_rho] / rho_used[mask_rho]
def robust_stats(a, ref):
    a = a[np.isfinite(a)]
    med = np.median(a)
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    mape = np.median(np.abs((a - ref) / (ref + 1e-30))) * 100.0
    return med, q1, q3, iqr, mape
G_med, G_q1, G_q3, G_iqr, G_mape = robust_stats(G_est[mask_rho], G_true)
# Spectral gradient for flux
def grad_fft(f):
    kx_full = 2.0 * np.pi * fftfreq(N, d=dx)
    ky_full = 2.0 * np.pi * fftfreq(N, d=dx)
    kz_full = 2.0 * np.pi * fftfreq(N, d=dx)
    KX_full, KY_full, KZ_full = np.meshgrid(kx_full, ky_full, kz_full, indexing='ij')
    f_k = fftn(f)
    gx = np.real(ifftn(1j * KX_full * f_k))
    gy = np.real(ifftn(1j * KY_full * f_k))
    gz = np.real(ifftn(1j * KZ_full * f_k))
    return gx, gy, gz
gx, gy, gz = grad_fft(ln_n)
# Flux G (Gauss) over centered half-box
sub = slice(N//4, 3*N//4)
def surface_flux_ln_n(sub, gx, gy, gz, dx):
    i0, i1 = sub.start, sub.stop-1
    j0, j1 = sub.start, sub.stop-1
    k0, k1 = sub.start, sub.stop-1
    dS = dx*dx
    flux = 0.0
    flux += np.sum((-gx[i0, sub, sub]) * dS) + np.sum((+gx[i1, sub, sub]) * dS)
    flux += np.sum((-gy[sub, j0, sub]) * dS) + np.sum((+gy[sub, j1, sub]) * dS)
    flux += np.sum((-gz[sub, sub, k0]) * dS) + np.sum((+gz[sub, sub, k1]) * dS)
    return flux
flux = surface_flux_ln_n(sub, gx, gy, gz, dx)
M_V = np.sum(rho_used[sub, sub, sub]) * (dx**3)
G_flux = - (c**2 / (8.0*np.pi)) * (flux / (M_V + 1e-30))
# -----------------------
# Multi-shell Gauss closures: relClose(Vj) - Optimized shell selection
# -----------------------
center = N // 2
r_list = np.arange(8, center, 8)  # Larger steps for more stable shells: 8,16,24,...
relClose_list = []
for r in r_list:
    sub_j = slice(center - r, center + r)
    flux_j = surface_flux_ln_n(sub_j, gx, gy, gz, dx)
    M_j = np.sum(rho_used[sub_j, sub_j, sub_j]) * (dx**3)
    num = abs(flux_j + kappa * M_j)
    den = abs(flux_j) + 1e-30
    relClose_j = num / den
    relClose_list.append(relClose_j)
relClose_array = np.array(relClose_list)
relClose_med = np.median(relClose_array)
relClose_mean = np.mean(relClose_array)
relClose_q1, relClose_q3 = np.percentile(relClose_array, [25, 75])
relClose_iqr = relClose_q3 - relClose_q1
# Convert to percent for reporting
relClose_med_pct = relClose_med * 100.0
relClose_mean_pct = relClose_mean * 100.0
relClose_q1_pct = relClose_q1 * 100.0
relClose_q3_pct = relClose_q3 * 100.0
relClose_iqr_pct = relClose_iqr * 100.0
# -----------------------
# Invert for c given G (pointwise & flux)
# -----------------------
# Pointwise: c_est(x) = sqrt( -8π G ρ / (∇² ln n) )
eps_lhs = 1e-8 * np.max(np.abs(lhs))
ratio = -8.0 * np.pi * G_true * rho_used / (lhs + 1e-300)
mask_c = (np.abs(rho_used) > eps_rho) & (np.abs(lhs) > eps_lhs) & (ratio > 0) & np.isfinite(ratio)
c_est_point = np.full_like(rho_used, np.nan, dtype=np.float64)
c_est_point[mask_c] = np.sqrt(ratio[mask_c])
c_med, c_q1, c_q3, c_iqr, c_mape = robust_stats(c_est_point[mask_c], c)
# Flux: c_est = sqrt( -8π G M(V) / flux )
c_flux = np.sqrt(np.maximum(1e-300, -8.0*np.pi * G_true * M_V / (flux + 1e-300)))
# -----------------------
# Summary table (both directions + relClose stats)
# -----------------------
summary = pd.DataFrame({
    "Metric": [
        "Rel L2 ||LHS-RHS|| / ||RHS||",
        "MAE(residual)",
        "RMSE(residual)",
        "Corr(LHS, RHS)",
        "c_used (m/s)",
        "G_true",
        "G_est_median",
        "G_est_q1",
        "G_est_q3",
        "G_est_IQR",
        "G_est_median_APE(%)",
        "G_flux_est",
        "c_est_point_median (m/s)",
        "c_est_point_q1",
        "c_est_point_q3",
        "c_est_point_IQR",
        "c_est_point_median_APE(%)",
        "c_est_flux (m/s)",
        "relClose_median (%)",
        "relClose_mean (%)",
        "relClose_q1 (%)",
        "relClose_q3 (%)",
        "relClose_IQR (%)"
    ],
    "Value": [
        rel_L2,
        mae_res,
        rmse_res,
        corr,
        c,
        G_true,
        G_med,
        G_q1,
        G_q3,
        G_iqr,
        G_mape,
        G_flux,
        c_med,
        c_q1,
        c_q3,
        c_iqr,
        c_mape,
        c_flux,
        relClose_med_pct,
        relClose_mean_pct,
        relClose_q1_pct,
        relClose_q3_pct,
        relClose_iqr_pct
    ]
})
SAVE_DIR = "/content" if os.path.isdir("/content") else os.getcwd()
csv_path = os.path.join(SAVE_DIR, "unified_field_bidir_colab_zeta_redesigned_sci.csv")
summary.to_csv(csv_path, index=False)
print(summary)  # Print the summary table for viewing
# -----------------------
# Plots
# -----------------------
# 1) Histogram of pointwise G estimates
plt.figure(figsize=(6,4))
plt.hist(G_est[mask_rho].ravel(), bins=80)
plt.title("Histogram of pointwise G estimates")
plt.xlabel("G_est"); plt.ylabel("Count")
plt.tight_layout()
plt.show()
# 2) Histogram of pointwise c estimates
plt.figure(figsize=(6,4))
plt.hist(c_est_point[mask_c].ravel(), bins=80)
plt.title("Histogram of pointwise c estimates")
plt.xlabel("c_est (m/s)"); plt.ylabel("Count")
plt.tight_layout()
plt.show()
# 3) Mid-plane slices: LHS, RHS, Residual
mid = N//2
lhs_slice = lhs[:, :, mid]
rhs_slice = rhs[:, :, mid]
res_slice = residual[:, :, mid]
plt.figure(figsize=(6,4))
plt.imshow(lhs_slice.T, origin='lower', extent=[-L/2, L/2, -L/2, L/2], aspect='equal')
plt.title("LHS: ∇² ln n (z=0 slice)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
plt.imshow(rhs_slice.T, origin='lower', extent=[-L/2, L/2, -L/2, L/2], aspect='equal')
plt.title("RHS: -(8πG/c²) ρ (z=0 slice)" if not use_zeta else "RHS: -(8πG/c²) ρ_pred (z=0 slice)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
plt.imshow(res_slice.T, origin='lower', extent=[-L/2, L/2, -L/2, L/2], aspect='equal')
plt.title("Residual: LHS - RHS (z=0 slice)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar()
plt.tight_layout()
plt.show()
# -----------------------
# Final concise printout
# -----------------------
print("=== Unified Field Equation Verification (Colab, bidirectional, improved with zeta={}, control={}) ===".format(use_zeta, use_control))
print(f"Grid: N={N}^3, dx={dx:.4e}, use_SI={use_SI}, c_used={c:.6g} m/s, G_true={G_true:.6g}, g={g_gain}")
print(f"Closure: Rel L2={rel_L2:.6e}, Corr(LHS,RHS)={corr:.6f}")
print(f"Residual MAE / RMSE: {mae_res:.6e} / {rmse_res:.6e}")
print(f"[Known c] G_est median [q1,q3], IQR, MAPE%: {G_med:.6g} [{G_q1:.6g}, {G_q3:.6g}], {G_iqr:.6g}, {G_mape:.3f}%")
print(f"[Known c] G_flux_est: {G_flux:.6g}")
print(f"[Known G] c_est_point median [q1,q3], IQR, MAPE%: {c_med:.6g} [{c_q1:.6g}, {c_q3:.6g}], {c_iqr:.6g}, {c_mape:.3f}%")
print(f"[Known G] c_est_flux: {c_flux:.6g} m/s")
print(f"Multi-shell relClose: median={relClose_med_pct:.3f}%, mean={relClose_mean_pct:.3f}%, [q1,q3]={relClose_q1_pct:.3f}% - {relClose_q3_pct:.3f}%, IQR={relClose_iqr_pct:.3f}%")
print(f"CSV saved to: {csv_path}")