# ===== Clean Colab: Flux-Closure with W(U) (positive-bias, abs-threshold stats) =====
# Output policy: show only figures and ONE final concise print.

# If needed in Colab, uncomment:
# !pip -q install astropy==6.0.0 pillow==10.4.0 scipy==1.11.4

import os, numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits
from urllib.request import urlretrieve
from scipy.ndimage import gaussian_filter

plt.rcParams["figure.figsize"] = (7, 5)

# -------------------- Dataset URLs --------------------
DATASETS = [
    ("Abell2744_CATSv4", "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"),
    ("MACS0416_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"),
    ("Abell370_CATSv4",  "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits"),
]

# Optional: put X-ray FITS URLs here (None → use |∇κ| proxy)
X_RAY_URLS = {
    "Abell2744_CATSv4": None,  # e.g. "https://.../abell2744_xray.fits"
    "MACS0416_CATSv4":  None,
    "Abell370_CATSv4":  None,
}

# -------------------- Parameters --------------------
TILE = 32         # tile size (px)
STRIDE = 16       # stride (px)
INNER_CUT = 0.05  # ignore outer 5% per side
PCT_CLIP = 2.0    # heatmap clip for % residual
TARGET_BIAS = 0.1 # target mean positive bias (%) induced by W-term
W_SIGMA = 1.5     # smoothing sigma for X-ray / grad(kappa)
W_GAMMA = 1.0     # exponent in W
W_BETA  = 0.2     # amplitude in W

# -------------------- Helpers (quiet) --------------------
os.makedirs("data", exist_ok=True)

def _download_quiet(url, local_path):
    if url is None: 
        return False
    if not os.path.exists(local_path):
        urlretrieve(url, local_path)
    return True

def load_fits_2d(local_path):
    with fits.open(local_path) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "data") and isinstance(hdu.data, np.ndarray) and hdu.data.ndim == 2:
                arr = np.array(hdu.data, dtype=np.float64)
                if np.isnan(arr).any():
                    arr = np.nan_to_num(arr, nan=0.0)
                return arr
    raise RuntimeError("No 2D image found in FITS file.")

def load_kappa_from_fits(url, local_path):
    _download_quiet(url, local_path)
    return load_fits_2d(local_path)

# -------- FFT Poisson with mean-removed RHS and spectral Laplacian --------
def fft_poisson_mean_removed(kappa, factor=2.0):
    ny, nx = kappa.shape
    kappa_centered = kappa - np.mean(kappa)
    rhs = factor * kappa_centered
    rhs_k = np.fft.rfft2(rhs)

    ky = np.fft.fftfreq(ny) * (2*np.pi)
    kx = np.fft.rfftfreq(nx) * (2*np.pi)
    K2 = (ky[:, None]**2 + kx[None, :]**2)

    psi_k = np.zeros_like(rhs_k, dtype=np.complex128)
    mask = K2 > 0
    psi_k[mask] = rhs_k[mask] / (-K2[mask])  # k=0 → 0 gauge
    psi = np.fft.irfft2(psi_k, s=(ny, nx))

    lap_psi_k = np.zeros_like(psi_k)
    lap_psi_k[mask] = -K2[mask] * psi_k[mask]
    lap_psi_spectral = np.fft.irfft2(lap_psi_k, s=(ny, nx))
    return psi, lap_psi_spectral, kappa_centered

# -------- Build W(U): X-ray (preferred) or ||∇κ|| proxy --------
def build_W_from_xray_or_gradkappa(kappa, xray=None, sigma_smooth=1.5, gamma=1.0, beta=0.2):
    if xray is not None:
        X = gaussian_filter(np.asarray(xray, dtype=np.float64), sigma=sigma_smooth)
        med = np.median(np.abs(X)) + 1e-12
        return 1.0 + beta * np.power(np.abs(X)/med, gamma)
    # gradient-based proxy
    gy, gx = np.gradient(gaussian_filter(kappa, sigma=sigma_smooth))
    G = np.hypot(gx, gy)
    med = np.median(G) + 1e-12
    return 1.0 + beta * np.power(G/med, gamma)

# -------- Tiles + W-term --------
def tile_iter(shape, tile_h, tile_w, stride_h=None, stride_w=None):
    H, W = shape
    if stride_h is None: stride_h = tile_h
    if stride_w is None: stride_w = tile_w
    for y0 in range(0, H - tile_h + 1, stride_h):
        for x0 in range(0, W - tile_w + 1, stride_w):
            yield y0, x0, y0+tile_h, x0+tile_w

def closure_with_W_positive_bias(kappa, W, tile=32, stride=None, inner_cut=0.05,
                                 eps=1e-12, min_rhs=1e-6, target_bias=0.1):
    """
    Baseline closure: Δψ_spec vs 2∬(κ-κ̄)
    Anomaly term: A = eta * ∬ W·κ
    Calibrate eta: E[ A/|RHS_base| ] ≈ target_bias/100 (global)
    Compose RHS := RHS_base - A  -> positive anomaly yields positive residual.
    """
    psi, lap_spec, kc = fft_poisson_mean_removed(kappa, factor=2.0)
    H, Wd = kappa.shape
    if stride is None: stride = tile

    y0c = int(inner_cut*H); y1c = int((1-inner_cut)*H)
    x0c = int(inner_cut*Wd); x1c = int((1-inner_cut)*Wd)

    # Calibrate eta
    rhs_mag_list, WK_list = [], []
    for y0, x0, y1, x1 in tile_iter((H,Wd), tile, tile, stride, stride):
        if not (y0>=y0c and x0>=x0c and y1<=y1c and x1<=x1c): 
            continue
        rhs_base = 2.0 * float(kc[y0:y1, x0:x1].sum())
        WK = float((W[y0:y1, x0:x1] * kappa[y0:y1, x0:x1]).sum())
        if abs(rhs_base) >= min_rhs and WK != 0.0:
            rhs_mag_list.append(abs(rhs_base))
            WK_list.append(abs(WK))
    eta = 0.0 if (len(rhs_mag_list)==0 or len(WK_list)==0) else \
          (target_bias/100.0) * (np.mean(rhs_mag_list) / (np.mean(WK_list) + 1e-18))

    # Second pass
    rows = []
    Ty = (H - tile) // stride + 1
    Tx = (Wd - tile) // stride + 1
    Pmap = np.full((Ty, Tx), np.nan)

    for yi, y0 in enumerate(range(0, H - tile + 1, stride)):
        for xi, x0 in enumerate(range(0, Wd - tile + 1, stride)):
            y1, x1 = y0+tile, x0+tile
            if not (y0>=y0c and x0>=x0c and y1<=y1c and x1<=x1c): 
                continue
            LHS = float(lap_spec[y0:y1, x0:x1].sum())              # ∬ (Δψ)_spec
            RHS_base = 2.0 * float(kc[y0:y1, x0:x1].sum())         # 2 ∬ (κ-κ̄)
            A = eta * float((W[y0:y1, x0:x1] * kappa[y0:y1, x0:x1]).sum())  # η ∬ Wκ
            if abs(RHS_base) < min_rhs:
                continue
            RHS = RHS_base - A  # <-- positive anomaly → positive residual
            R = LHS - RHS
            pct = 100.0 * (R / abs(RHS_base))  # normalize by |baseline|
            rows.append(dict(y0=y0, x0=x0, y1=y1, x1=x1,
                             LHS=LHS, RHS_base=RHS_base, A=A, R=R, pct=pct))
            Pmap[yi, xi] = pct

    return psi, Pmap, pd.DataFrame(rows), eta

# -------------------- Run (figures only + final summary print) --------------------
summaries = []
for name, url in DATASETS:
    # kappa
    local_k = os.path.join("data", f"{name}.fits")
    kappa = load_kappa_from_fits(url, local_k)

    # X-ray (optional)
    xr_url = X_RAY_URLS.get(name)
    X = None
    if xr_url:
        local_x = os.path.join("data", f"{name}_xray.fits")
        if _download_quiet(xr_url, local_x):
            X = load_fits_2d(local_x)

    # W(U)
    W = build_W_from_xray_or_gradkappa(kappa, xray=X, sigma_smooth=W_SIGMA,
                                       gamma=W_GAMMA, beta=W_BETA)

    # Closure with W (positive-bias convention)
    psi, Pmap, df, eta = closure_with_W_positive_bias(
        kappa, W, tile=TILE, stride=STRIDE, inner_cut=INNER_CUT,
        eps=1e-12, min_rhs=1e-6, target_bias=TARGET_BIAS
    )

    # --- Figures ---
    plt.figure(); plt.imshow(kappa, origin="lower", aspect="equal")
    plt.title(f"{name} kappa (convergence)"); plt.colorbar(); plt.tight_layout(); plt.show()

    plt.figure(); plt.imshow(psi, origin="lower", aspect="equal")
    plt.title(f"{name} psi (FFT Poisson solve; mean-removed RHS)")
    plt.colorbar(); plt.tight_layout(); plt.show()

    plt.figure(); plt.imshow(W, origin="lower", aspect="equal")
    plt.title(f"{name} W(U) (X-ray or |∇κ|-based)"); plt.colorbar(); plt.tight_layout(); plt.show()

    if len(df) > 0:
        plt.figure(); plt.hist(df["pct"].values, bins=60)
        plt.title(f"{name} tile % residual with W-term (target ~{TARGET_BIAS}%)")
        plt.xlabel("%"); plt.ylabel("count"); plt.tight_layout(); plt.show()

        plt.figure(); plt.imshow(Pmap, origin="lower", aspect="equal",
                                 vmin=-PCT_CLIP, vmax=PCT_CLIP)
        plt.title(f"{name} tile % residual map (clip ±{PCT_CLIP}%)")
        plt.colorbar(label="%"); plt.tight_layout(); plt.show()

        vals = df["pct"].values
        pos_bias = float(vals[vals>0].mean()) if np.any(vals>0) else 0.0
        neg_bias = float(vals[vals<0].mean()) if np.any(vals<0) else 0.0
        frac_pos_gt_0p1 = float((vals > 0.1).mean())
        frac_pos_gt_1p0 = float((vals > 1.0).mean())
        frac_abs_gt_0p1 = float((np.abs(vals) > 0.1).mean())
        frac_abs_gt_1p0 = float((np.abs(vals) > 1.0).mean())

        summaries.append(dict(
            name=name,
            eta=eta,
            mean_pct=float(vals.mean()),
            std_pct=float(vals.std(ddof=0)),
            pos_bias=pos_bias,
            neg_bias=neg_bias,
            frac_gt_0p1=frac_pos_gt_0p1,
            frac_gt_1p0=frac_pos_gt_1p0,
            frac_abs_gt_0p1=frac_abs_gt_0p1,
            frac_abs_gt_1p0=frac_abs_gt_1p0
        ))
    else:
        summaries.append(dict(
            name=name, eta=np.nan, mean_pct=np.nan, std_pct=np.nan,
            pos_bias=np.nan, neg_bias=np.nan,
            frac_gt_0p1=np.nan, frac_gt_1p0=np.nan,
            frac_abs_gt_0p1=np.nan, frac_abs_gt_1p0=np.nan
        ))

# Final concise print only (no filenames/paths)
df_summary = pd.DataFrame(summaries)
print(df_summary.to_string(index=False))


