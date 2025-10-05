# -*- coding: utf-8 -*-
# DAG–UTH unified field — HFF/CATS v4: Abell2744, MACS0416, Abell370
# Certificates: numerical (flux vs Δ5ψ), physical (Δ5ψ ≈ c κ), nonlinear, and SCI K

import os, requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.fft import dctn, idctn
from scipy.optimize import minimize
from scipy.ndimage import binary_erosion, generate_binary_structure, gaussian_filter
rng = np.random.default_rng(42)

# -------------------- Dataset URLs --------------------
DATASETS = [
    ("Abell2744_CATSv4", "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"),
    ("MACS0416_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"),
    ("Abell370_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits")
]

# -------------------- IO helpers --------------------
def download_fits(url, path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1 << 19):
                f.write(chunk)

def load_kappa(path):
    with fits.open(path) as hdul:
        for h in hdul:
            if h.data is None: 
                continue
            arr = np.array(h.data, dtype=float)
            if arr.ndim == 2:
                k = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if np.isfinite(k).any():
                    return k
    raise RuntimeError("No 2D image found in FITS.")

# -------------------- Poisson solver (Neumann DCT) --------------------
def poisson_solve_neumann(rhs):
    """
    Solve Δψ = rhs with homogeneous Neumann BC via DCT-II/III (orthonormal).
    Returns zero-mean ψ.
    """
    ny, nx = rhs.shape
    rhs = rhs - rhs.mean()  # solvability
    rhs_hat = dctn(rhs, type=2, norm='ortho')
    ky = np.arange(ny); kx = np.arange(nx)
    lam_y = 2.0 * (1.0 - np.cos(np.pi * ky / ny))
    lam_x = 2.0 * (1.0 - np.cos(np.pi * kx / nx))
    L = lam_y[:, None] + lam_x[None, :]
    L[0, 0] = 1.0
    psi_hat = rhs_hat / L
    psi_hat[0, 0] = 0.0
    psi = idctn(psi_hat, type=3, norm='ortho')
    return psi

# -------------------- Discrete operators & flux --------------------
def laplacian5(arr):
    """5-point Laplacian with clamped edges (no wrap)."""
    up    = np.vstack([arr[0:1,:], arr[:-1,:]])
    down  = np.vstack([arr[1: ,:], arr[-1:,:]])
    left  = np.hstack([arr[:,0:1], arr[:,:-1]])
    right = np.hstack([arr[:,1: ], arr[:,-1:]])
    return (up + down + left + right - 4.0*arr)

def boundary_flux(phi, box):
    """
    ∮ ∇φ · n dS on a rectangular subwindow using one-sided diffs
    with *outside* neighbor samples (clamped to image borders).
    """
    y0, y1, x0, x1 = box
    H, W = phi.shape
    # top: n=(0,-1) -> -(φ[y0,x]-φ[y0-1,x])
    y_out = max(0, y0 - 1)
    top = -np.sum(phi[y0, x0:x1] - phi[y_out, x0:x1])
    # bottom: n=(0,+1) -> +(φ[y1,x]-φ[y1-1,x])
    y_in = y1 - 1
    y_out = min(H - 1, y1)
    bot = +np.sum(phi[y_out, x0:x1] - phi[y_in, x0:x1])
    # left: n=(-1,0) -> -(φ[y,x0]-φ[y,x0-1])
    x_out = max(0, x0 - 1)
    left = -np.sum(phi[y0:y1, x0] - phi[y0:y1, x_out])
    # right: n=(+1,0) -> +(φ[y,x1]-φ[y,x1-1])
    x_in = x1 - 1
    x_out = min(W - 1, x1)
    right = +np.sum(phi[y0:y1, x_out] - phi[y0:y1, x_in])
    return top + bot + left + right

def area_lap_sum(phi, box):
    y0,y1,x0,x1 = box
    sub = phi[y0:y1, x0:x1]
    lap = laplacian5(sub)
    return float(lap.sum())

# -------------------- Certificates (numerical & physical) --------------------
def line_num_residual(phi, box):
    """Numerical certificate: flux vs ΣΔ5ψ (same stencil)"""
    flux = boundary_flux(phi, box)
    alap = area_lap_sum(phi, box)
    denom = abs(flux) + abs(alap) + 1e-12
    return (flux - alap) / denom, denom

def calibrate_c_via_lap(phi, kappa, boxes):
    """Fit c in ΣΔ5ψ ≈ c Σκ over many windows."""
    num = 0.0; den = 0.0
    for b in boxes:
        alap = area_lap_sum(phi, b)
        y0,y1,x0,x1 = b
        aK = float(kappa[y0:y1, x0:x1].sum())
        num += alap * aK
        den += aK * aK + 1e-30
    c = num/den if den>0 else 1.0
    if not np.isfinite(c) or abs(c) < 1e-12: c = 1.0
    return c

def line_phys_residual(phi, kappa, box, c_scale):
    """Physical certificate: ΣΔ5ψ ≈ c Σκ"""
    alap = area_lap_sum(phi, box)
    y0,y1,x0,x1 = box
    aK  = float(kappa[y0:y1, x0:x1].sum())
    denom = abs(alap) + abs(c_scale*aK) + 1e-12
    return (alap - c_scale*aK)/denom, denom

def nonlinear_residual(phi, kappa, box, c_scale, beta, Lambda):
    """Nonlinear: ΣΔ5ψ + βΣN(|∇ψ|^2) ≈ c Σκ, N(s)=(s/(1+s/Λ^2))^2"""
    alap = area_lap_sum(phi, box)
    y0,y1,x0,x1 = box
    sub = phi[y0:y1, x0:x1]
    gx = np.diff(sub, axis=1, append=sub[:, -1:])
    gy = np.diff(sub, axis=0, append=sub[-1:, :])
    g2 = gx**2 + gy**2
    N = (g2 / (1.0 + g2/(Lambda**2 + 1e-12)))**2
    aK = float(kappa[y0:y1, x0:x1].sum())
    nl = beta * float(N.sum())
    denom = abs(alap) + abs(c_scale*aK) + abs(nl) + 1e-12
    R = (alap + nl - c_scale*aK) / denom
    return R, denom

# -------------------- Sampling windows with mask --------------------
def build_mask(kappa, thr=None):
    m = np.isfinite(kappa)
    if thr is None:
        thr = max(1e-6, np.nanpercentile(np.abs(kappa[m]), 20) * 0.05)
    m &= (np.abs(kappa) > thr)
    st = generate_binary_structure(2, 1)
    m = binary_erosion(m, structure=st, iterations=2, border_value=0)
    return m

def random_boxes_from_mask(mask, win, nbox, margin=16):
    H, W = mask.shape
    inner = mask.copy()
    for _ in range(margin):
        inner = binary_erosion(inner, structure=generate_binary_structure(2,1), border_value=0)
    ys, xs = np.where(inner)
    boxes = []
    tries = 0
    while len(boxes) < nbox and tries < nbox * 30 and len(ys)>0:
        tries += 1
        idx = rng.integers(0, len(ys))
        y = ys[idx]; x = xs[idx]
        y0 = y - win//2; y1 = y0 + win
        x0 = x - win//2; x1 = x0 + win
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W: 
            continue
        subm = mask[y0:y1, x0:x1]
        if subm.mean() < 0.9: 
            continue
        boxes.append((y0, y1, x0, x1))
    return boxes

# -------------------- SCI slope K (smoothed, robust) --------------------
def sci_K(phi, w=96, stride=48, eps=1e-6, sigma=1.0):
    ph = gaussian_filter(phi, sigma=sigma, mode='nearest')
    gx = np.diff(ph, axis=1, append=ph[:, -1:])
    gy = np.diff(ph, axis=0, append=ph[-1:, :])
    H, W = ph.shape

    Phi=[]; Hvar=[]
    for y in range(0, H-w, stride):
        for x in range(0, W-w, stride):
            Gx = gx[y:y+w, x:x+w]; Gy = gy[y:y+w, x:x+w]
            Jxx = (Gx*Gx).mean(); Jyy = (Gy*Gy).mean(); Jxy = (Gx*Gy).mean()
            lam_max = 0.5*((Jxx+Jyy) + np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2))
            if lam_max <= 0: 
                continue
            block = np.hypot(Gx, Gy).ravel()
            m, v = block.mean(), block.var()
            if m>eps and v>eps:
                Phi.append(m); Hvar.append(v)

    if len(Phi) < 10: 
        return np.nan, np.nan
    x = np.log(np.array(Phi)); y = np.log(np.array(Hvar))
    q1,q3 = np.percentile(y, [25,75]); iqr = q3-q1
    mask = (y>=q1-1.5*iqr) & (y<=q3+1.5*iqr)
    X = np.vstack([x[mask], np.ones(mask.sum())]).T; Y = y[mask]
    K, b = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(K), (np.inf if K==0 else float(1.0/K))

# -------------------- Main pipeline --------------------
def run_one(name, url, win_lin=64, nbox_lin=180, win_nl=64, nbox_nl=96, margin=24, plot_quick=False):
    print(f"\n=== {name} ===")
    path = f"{name}.fits"
    if not os.path.exists(path):
        print("Downloading...", url)
        download_fits(url, path)

    kappa = load_kappa(path)
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

    # Build mask and Poisson solve (temporary rhs; c will absorb scale)
    mask = build_mask(kappa)
    psi = poisson_solve_neumann(kappa)
    phi = psi

    # Linear windows
    boxes_lin = random_boxes_from_mask(mask, win_lin, nbox_lin, margin=margin)

    # 1) Numerical certificate (stencil-consistent)
    Rs_num, Ws_num = [], []
    for b in boxes_lin:
        r,w = line_num_residual(phi, b)
        Rs_num.append(abs(r)); Ws_num.append(w)
    num_med = float(np.median(Rs_num))
    num_wavg = float(np.average(Rs_num, weights=np.array(Ws_num)+1e-12))
    print(f"Numerical certificate |R_num| (med/wavg): {num_med:.3f} / {num_wavg:.3f}")

    # 2) Physical certificate: calibrate c in ΣΔ5ψ ≈ c Σκ
    c_scale = calibrate_c_via_lap(phi, kappa, boxes_lin)
    Rs_phys, Ws_phys = [], []
    for b in boxes_lin:
        r,w = line_phys_residual(phi, kappa, b, c_scale)
        Rs_phys.append(abs(r)); Ws_phys.append(w)
    lin_med = float(np.median(Rs_phys))
    lin_wavg = float(np.average(Rs_phys, weights=np.array(Ws_phys)+1e-12))
    print(f"Physical flux residual |R_phys| (med/wavg): {lin_med:.3f} / {lin_wavg:.3f}   [c≈{c_scale:.3g}]")

    # Strong-field selection by grad variance
    g2 = (np.diff(phi, axis=1, append=phi[:, -1:])**2 +
          np.diff(phi, axis=0, append=phi[-1:, :])**2)
    cand = random_boxes_from_mask(mask, win_nl, nbox_nl*6, margin=margin)
    def box_var(b):
        y0,y1,x0,x1 = b
        return float(np.var(g2[y0:y1, x0:x1]))
    cand.sort(key=box_var, reverse=True)
    boxes_nl = cand[:nbox_nl]

    # Fit (beta, Lambda) on strong-field windows
    init_L = np.sqrt(np.percentile(g2[mask], 90) + 1e-9)
    def obj(theta):
        beta, Lam = theta
        if beta < 0 or Lam <= 0: 
            return 1e9
        rs=[]; ws=[]
        for b in boxes_nl:
            r,w = nonlinear_residual(phi, kappa, b, c_scale, beta, Lam)
            rs.append(abs(r)); ws.append(w)
        return float(np.average(rs, weights=np.array(ws)+1e-12))
    res = minimize(obj, x0=np.array([0.05, init_L]), method='Nelder-Mead',
                   options=dict(maxiter=400, xatol=1e-3, fatol=1e-4))
    beta, Lam = map(float, res.x)

    # Evaluate nonlinear residuals
    Rn, Wn = [], []
    for b in boxes_nl:
        r,w = nonlinear_residual(phi, kappa, b, c_scale, beta, Lam)
        Rn.append(abs(r)); Wn.append(w)
    nl_med = float(np.median(Rn))
    nl_wavg = float(np.average(Rn, weights=np.array(Wn)+1e-12))
    print(f"Nonlinear residual |R_nl| (med/wavg): {nl_med:.3f} / {nl_wavg:.3f}   [beta≈{beta:.3f}, Lambda≈{Lam:.3f}]")

    # SCI K
    K, invK = sci_K(phi, w=96, stride=48, sigma=1.0)
    print(f"SCI slope K≈{K:.3f}  ->  1/K≈{invK:.3f}")

    if plot_quick:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.title("kappa"); plt.imshow(kappa, cmap='viridis'); plt.colorbar(fraction=0.046)
        plt.subplot(1,3,2); plt.title("phi (Neumann Poisson)"); plt.imshow(phi, cmap='magma'); plt.colorbar(fraction=0.046)
        plt.subplot(1,3,3); plt.title("|∇phi|"); 
        gm = np.hypot(np.diff(phi, axis=1, append=phi[:, -1:]), np.diff(phi, axis=0, append=phi[-1:, :]))
        plt.imshow(gm, cmap='inferno'); plt.colorbar(fraction=0.046); plt.tight_layout(); plt.show()

    return dict(num_med=num_med, num_wavg=num_wavg,
                lin_med=lin_med, lin_wavg=lin_wavg, c_scale=c_scale,
                nl_med=nl_med, nl_wavg=nl_wavg, beta=beta, Lambda=Lam,
                K=K, invK=invK)

# -------------------- Run all --------------------
all_results = {}
for name, url in DATASETS:
    all_results[name] = run_one(name, url, plot_quick=False)

print("\nSummary:")
for k,v in all_results.items():
    casted = {}
    for kk, vv in v.items():
        if isinstance(vv, (np.floating,)):
            casted[kk] = float(vv)
        else:
            casted[kk] = vv
    print(k, casted)
