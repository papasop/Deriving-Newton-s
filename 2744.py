# ============================================================
#  Abell 2744 (real κ-map) → strong-field refractive test
#  - Downloads CATS v4 kappa FITS
#  - Solves linear vs nonlinear PDE for φ=ln n
#  - Computes deflection α = ∇φ (2D proxy), compares Δα
#  - Nonlinear uses saturated |∇φ|^4 and 2/3 de-aliasing
#  - Shows plots (plt.show) and prints artifact paths at end
# ============================================================
import os, urllib.request, numpy as np, numpy.fft as fft, matplotlib.pyplot as plt
import pandas as pd

# ---------- Paths & I/O ----------
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
FITS_PATH = "/content/kappa_map.fits"
URL = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"

# Astropy install (Colab usually has it; keep fallback)
try:
    from astropy.io import fits
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "astropy"])
    from astropy.io import fits

if not os.path.exists(FITS_PATH):
    print("Downloading:", URL)
    urllib.request.urlretrieve(URL, FITS_PATH)
else:
    print("File exists:", FITS_PATH)

# ---------- Load κ map ----------
with fits.open(FITS_PATH) as hdul:
    # Try to find first image HDU with data
    data = None
    for h in hdul:
        if hasattr(h, "data") and isinstance(h.data, np.ndarray):
            data = h.data.astype(np.float64)
            if data.ndim == 2:
                break
    assert data is not None and data.ndim == 2, "No 2D image in FITS!"

# Replace NaNs/infs
m = np.nanmedian(data)
data = np.nan_to_num(data, nan=m, posinf=m, neginf=m)

# Center crop to power-of-two square for FFT friendliness
H, W = data.shape
S = min(H, W)
# choose a manageable power-of-two (e.g., 512)
target = 512 if S >= 512 else (256 if S >= 256 else 128)
r0 = (H - target)//2; c0 = (W - target)//2
kappa_map = data[r0:r0+target, c0:c0+target].copy()
N = kappa_map.shape[0]
Lx = Ly = 1.0  # normalized box (arcsec scale factor not needed for demo)

# ---------- Spectral operators ----------
kx = 2*np.pi*fft.fftfreq(N, d=Lx/N)
ky = 2*np.pi*fft.fftfreq(N, d=Ly/N)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
K2[0,0] = 1.0  # avoid zero-div in Poisson inversions

def fft2(a):  return fft.fft2(a)
def ifft2(A): return np.real(fft.ifft2(A))

def grad(phi):
    Ph = fft2(phi)
    gx = ifft2(1j*KX*Ph); gy = ifft2(1j*KY*Ph)
    return gx, gy

def lap(phi):
    return ifft2(-K2*fft2(phi))

def invlap(rhs):
    Rh = fft2(rhs); Ph = Rh/(-K2); Ph[0,0] = 0.0
    return ifft2(Ph)

# 2/3 de-aliasing filter (Orszag)
def dealias_23(Ahat):
    nx = N; ny = N
    kcx = int(nx/3); kcy = int(ny/3)
    B = Ahat.copy()
    B[kcx:-kcx,:] = 0.0
    B[:,kcy:-kcy] = 0.0
    return B

def filter_23(phi):
    return ifft2(dealias_23(fft2(phi)))

# ---------- Build ρ from κ (proxy) ----------
# zero-mean source; scale factor absorbed to kappa_phys=1 for demo
rho = kappa_map - np.mean(kappa_map)
rho = rho - np.mean(rho)
kappa_phys = 1.0

# ---------- Linear φ (weak-field baseline) ----------
phi_lin = invlap(-kappa_phys*rho)
phi_lin -= np.mean(phi_lin)

# ---------- Nonlinearity β(x) from gradient proxy + soft saturation ----------
# Build a seed from linear gradient for β map
gx0, gy0 = grad(phi_lin)
g2_seed = gx0*gx0 + gy0*gy0
p95 = np.percentile(g2_seed, 95)
BETA0 = 0.6  # global strength (tunable)
beta_map = BETA0 * (g2_seed / (p95 + 1e-12))
beta_map = np.clip(beta_map, 0.0, BETA0)
# Smooth β by 1-step 5-point average to avoid checkerboarding
beta_map = 0.5*(beta_map + (np.roll(beta_map,1,0)+np.roll(beta_map,-1,0)+np.roll(beta_map,1,1)+np.roll(beta_map,-1,1))/4.0)

# Soft saturation scale for |∇φ|^2
Lambda2 = np.percentile(g2_seed, 98) + 1e-12

def N_soft(g2):
    # saturated (|∇φ|^2)^2 -> (g2/(1+g2/Lambda2))^2
    s = g2/(1.0 + g2/Lambda2)
    return s*s

# ---------- Strong-field solve: ∇²φ + β(x) * N_soft(|∇φ|^2) = -κ_phys ρ ----------
def solve_strong(rho, beta_map, steps=6, iters=250, tol=2e-6, relax=0.62, do_filter=True):
    phi = invlap(-kappa_phys*rho)  # start from linear
    phi -= np.mean(phi)
    residuals = []
    for s in range(1, steps+1):
        w = s/steps
        beta_eff = w*beta_map
        for it in range(iters):
            gx, gy = grad(phi)
            g2 = gx*gx + gy*gy
            rhs = -kappa_phys*rho - beta_eff * N_soft(g2)
            phi_new = invlap(rhs)
            if do_filter:
                phi_new = filter_23(phi_new)
            phi = (1.0 - relax)*phi + relax*phi_new
            phi -= np.mean(phi)
            F = lap(phi) + beta_eff * N_soft(g2) + kappa_phys*rho
            res = float(np.sqrt(np.mean(F*F)))
            residuals.append(res)
            if res < tol:
                break
    return phi, np.array(residuals)

phi_non, res_hist = solve_strong(rho, beta_map, steps=6, iters=260, tol=2e-6, relax=0.6, do_filter=True)

# ---------- Deflection fields & diagnostics ----------
def deflection_xy(phi):
    # In this refractive model demo, take α ≈ ∇φ as 2D proxy
    return grad(phi)

ax_lin_x, ax_lin_y = deflection_xy(phi_lin)
ax_non_x, ax_non_y = deflection_xy(phi_non)

delta_ax = ax_non_x - ax_lin_x
delta_ay = ax_non_y - ax_lin_y
delta_mag = np.sqrt(delta_ax**2 + delta_ay**2)

# line-cut through center
i0 = N//2
b_axis = np.linspace(-0.5, 0.5, N)
cut_lin = ax_lin_x[i0,:]
cut_non = ax_non_x[i0,:]
cut_dlt = cut_non - cut_lin

# ---------- Nonlinear Gauss closure (square shells) ----------
# Flux + mass + nonlinear ≈ 0  (2D proxy)
dx = Lx/N; dy = Ly/N; dA = dx*dy

def square_closure(phi, rho, beta_map, half_sizes):
    # Use φ_non for closure check (with its β)
    gx, gy = grad(phi)
    rels, fluxes, masses, Nterms = [], [], [], []
    def sample(arr, xi, yi):
        # bilinear periodic sampling
        ii = (xi* N / Lx) % N; jj = (yi* N / Ly) % N
        i0 = np.floor(ii).astype(int); j0 = np.floor(jj).astype(int)
        di = ii - i0; dj = jj - j0
        i1 = (i0+1) % N; j1 = (j0+1) % N
        return ((1-di)*(1-dj)*arr[i0,j0] + di*(1-dj)*arr[i1,j0] +
                (1-di)*dj*arr[i0,j1] + di*dj*arr[i1,j1])
    cx = cy = 0.5*Lx
    m = 2048
    Xg = np.linspace(0, Lx, N, endpoint=False); Yg = np.linspace(0, Ly, N, endpoint=False)
    X, Y = np.meshgrid(Xg, Yg, indexing='ij')

    for a in half_sizes:
        t = np.linspace(-a, a, m, endpoint=False)
        xl, yl = (cx - a)*np.ones_like(t), cy + t
        xr, yr = (cx + a)*np.ones_like(t), cy + t
        xb, yb = cx + t, (cy - a)*np.ones_like(t)
        xt, yt = cx + t, (cy + a)*np.ones_like(t)
        gx_l = sample(gx, xl, yl); gy_l = sample(gy, xl, yl)
        gx_r = sample(gx, xr, yr); gy_r = sample(gy, xr, yr)
        gx_b = sample(gx, xb, yb); gy_b = sample(gy, xb, yb)
        gx_t = sample(gx, xt, yt); gy_t = sample(gy, xt, yt)
        # outward normals: left(-x), right(+x), bottom(-y), top(+y)
        flux = (np.trapezoid(-gx_l, t) + np.trapezoid(+gx_r, t) +
                np.trapezoid(-gy_b, t) + np.trapezoid(+gy_t, t))
        # area terms
        mask = (np.abs((X-cx))<=a) & (np.abs((Y-cy))<=a)
        mass = rho[mask].sum()*dA
        gxv, gyv = gx[mask], gy[mask]
        Nterm = (beta_map[mask] * N_soft(gxv*gxv+gyv*gyv)).sum()*dA
        clos = flux + kappa_phys*mass + Nterm
        denom = np.abs(flux)+np.abs(kappa_phys*mass)+np.abs(Nterm)+1e-12
        rels.append(np.abs(clos)/denom)
        fluxes.append(flux); masses.append(mass); Nterms.append(Nterm)
    return (np.array(rels), np.array(fluxes),
            np.array(masses), np.array(Nterms))

half_sizes = np.linspace(0.06*Lx, 0.35*Lx, 24)
rels, Fv, Mv, Nv = square_closure(phi_non, rho, beta_map, half_sizes)

# ---------- Plots ----------
plt.figure(figsize=(6.4,4))
plt.semilogy(res_hist)
plt.xlabel("Iteration"); plt.ylabel("RMS residual")
plt.title("Strong-field residual (Abell 2744)")
plt.tight_layout(); plt.savefig(f"{OUT}/residual_abell2744.png", dpi=160); plt.show()

plt.figure(figsize=(6,5))
plt.imshow(kappa_map.T, origin="lower", cmap="magma")
plt.colorbar(label="κ (CATS v4)")
plt.title("Abell 2744: κ-map (center crop)")
plt.tight_layout(); plt.savefig(f"{OUT}/abell2744_kappa.png", dpi=160); plt.show()

plt.figure(figsize=(6,5))
plt.imshow(delta_mag.T, origin="lower", cmap="viridis")
plt.colorbar(label="|Δα| = |∇φ_NL - ∇φ_Lin|")
plt.title("Strong-field correction map |Δα|")
plt.tight_layout(); plt.savefig(f"{OUT}/delta_alpha_map.png", dpi=160); plt.show()

plt.figure(figsize=(6.4,4))
plt.plot(b_axis, cut_lin, label="α_x (linear)")
plt.plot(b_axis, cut_non, label="α_x (nonlinear)")
plt.plot(b_axis, cut_dlt, label="Δα_x", linestyle="--")
plt.xlabel("normalized impact parameter (center line)")
plt.ylabel("α_x")
plt.title("Line cut through center (x-direction)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/linecut_alpha.png", dpi=160); plt.show()

plt.figure(figsize=(6.4,3.6))
plt.plot(half_sizes, rels)
plt.xlabel("half-size (box)")
plt.ylabel("relative closure")
plt.title("Nonlinear Gauss closure (Abell 2744)")
plt.tight_layout(); plt.savefig(f"{OUT}/closure_curve.png", dpi=160); plt.show()

# ---------- CSVs ----------
pd.DataFrame({
    "half_size": half_sizes,
    "rel_closure": rels,
    "flux": Fv,
    "mass_term": kappa_phys*Mv,
    "nonlinear_term": Nv
}).to_csv(f"{OUT}/closure_abell2744.csv", index=False)

# Aggregate Δα stats
stats = {
    "delta_alpha_mean": float(np.mean(delta_mag)),
    "delta_alpha_median": float(np.median(delta_mag)),
    "delta_alpha_p95": float(np.percentile(delta_mag,95))
}
pd.DataFrame([stats]).to_csv(f"{OUT}/delta_alpha_stats.csv", index=False)

# ---------- Final print ----------
print("Artifacts saved in:", OUT)
print(" - κ map crop:          ", f"{OUT}/abell2744_kappa.png")
print(" - residual history:    ", f"{OUT}/residual_abell2744.png")
print(" - |Δα| map:            ", f"{OUT}/delta_alpha_map.png")
print(" - α line cut:          ", f"{OUT}/linecut_alpha.png")
print(" - closure curve:       ", f"{OUT}/closure_curve.png")
print(" - CSVs:")
print("   • closure:           ", f"{OUT}/closure_abell2744.csv")
print("   • Δα summary stats:  ", f"{OUT}/delta_alpha_stats.csv")

