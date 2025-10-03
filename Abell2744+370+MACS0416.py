# ==========================================================
# Fast Pre-registered Independent Test for ζ–U Binding (Eq. 5)
# Dataset: Abell 2744 (CATS v4 κ map)
# Deps: numpy, matplotlib, astropy, scipy, mpmath
# Outputs: matplotlib plots + PASS/FAIL summary
# ==========================================================

# ---------- Fixed (PRIMARY) settings ----------
PRIMARY = {
    "EMBEDDING": "radial-elliptic",  # 'radial-elliptic'|'radial-from-peak'|'radial-center'
    "T_LO": 80.0, "T_HI": 6000.0,    # θ sampling t-range
    "THETA_GRID_N": 1000,            # θ grid points (cached)
    "TARGET_N": 512,                  # κ resize N×N
    "SIGMA_SMOOTH": 1.0,              # κ smoothing [px]
    "TUKEY_ALPHA": 0.20,              # window alpha
    "BANDPASS": (0.06, 0.32),         # k-fraction band (θ plane)
    "SLOPE_FIT_FRAC": (0.10, 0.35),   # band for γ² average
    "TOP_N_PEAKS": 120,               # peak list size
    "PEAK_MATCH_RADIUS": 6,           # px
    "BEST_SHIFT_MAX": 6               # ±px for best-shift r
}

# Passing thresholds (pre-registered)
THRESH = {
    "gamma2_min": 0.05,   # mean γ² >= 0.05
    "peak_min":   0.10,   # peak match >= 0.10
    "r_bonus":    0.03    # best-shift r >= 0.03 (bonus)
}

DATASET_NAME = "Abell2744_CATSv4"
DATASET_URL  = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"

# ------------- Imports / installs -------------
import sys, subprocess, warnings
warnings.filterwarnings("ignore")
def _pip(x): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + x)
try:
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects, zoom, gaussian_laplace
    from scipy import fft as spfft
    from scipy.stats import pearsonr
except Exception:
    _pip(["numpy","matplotlib","astropy","scipy"])
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects, zoom, gaussian_laplace
    from scipy import fft as spfft
    from scipy.stats import pearsonr

try:
    import mpmath as mp
except Exception:
    _pip(["mpmath"])
    import mpmath as mp

# ----------------- Utilities -----------------
def zscore(a):
    m = np.mean(a); s = np.std(a) + 1e-12
    return (a - m) / s

def laplacian_2d(f):
    return (-4.0*f + np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1))

def isotropic_ps(f):
    F = spfft.fft2(f)
    P2 = (F*np.conj(F)).real
    ny, nx = f.shape
    ky = spfft.fftfreq(ny)*ny
    kx = spfft.fftfreq(nx)*nx
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).astype(int)
    ps = np.bincount(K.ravel(), P2.ravel())
    cnt= np.bincount(K.ravel())
    ps = ps/np.maximum(cnt,1)
    k  = np.arange(len(ps))
    return k[1:], ps[1:]

def isotropic_cross_ps(f,g):
    F = spfft.fft2(f); G = spfft.fft2(g)
    C = (F*np.conj(G)).real
    ny, nx = f.shape
    ky = spfft.fftfreq(ny)*ny
    kx = spfft.fftfreq(nx)*nx
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).astype(int)
    cs = np.bincount(K.ravel(), C.ravel())
    cnt= np.bincount(K.ravel())
    cs = cs/np.maximum(cnt,1)
    k  = np.arange(len(cs))
    return k[1:], cs[1:]

def mean_gamma2(a,b, lohi=(0.10,0.35)):
    kd, Pd = isotropic_ps(a); kp, Pp = isotropic_ps(b); kx, Px = isotropic_cross_ps(a,b)
    common = min(len(Pd),len(Pp),len(Px))
    Pd,Pp,Px = Pd[:common],Pp[:common],Px[:common]
    g2 = (Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp,1e-20))
    i0=int(common*lohi[0]); i1=int(common*lohi[1])
    return float(np.mean(g2[i0:i1])), g2, common

def jackknife_gamma2(g2, common, lohi=(0.10,0.35), blocks=10):
    i0=int(common*lohi[0]); i1=int(common*lohi[1])
    seg = g2[i0:i1]; n=len(seg)
    if n < blocks:
        mu = float(np.mean(seg))
        return mu, (np.nan, np.nan)
    size = n // blocks
    vals=[]
    for b in range(blocks):
        mask = np.ones(n, dtype=bool)
        mask[b*size:(b+1)*size] = False
        vals.append(float(np.mean(seg[mask])))
    vals = np.array(vals)
    mu = float(np.mean(vals))
    std = np.sqrt((blocks-1)/blocks * np.sum((vals-mu)**2))
    return mu, (mu-1.96*std, mu+1.96*std)

# --- self-implemented Tukey window (no scipy.signal.tukey needed) ---
def tukey_win(M, alpha=0.5):
    if M <= 0: return np.array([])
    if alpha <= 0: return np.ones(M, dtype=float)
    if alpha >= 1: return np.hanning(M)
    n = np.arange(M, dtype=float)
    w = np.ones(M, dtype=float)
    edge = alpha*(M-1)/2.0
    # left taper
    idx = n < edge
    w[idx] = 0.5*(1 + np.cos(np.pi*((2*n[idx])/(alpha*(M-1)) - 1)))
    # right taper
    idx = n >= (M-1-edge)
    w[idx] = 0.5*(1 + np.cos(np.pi*((2*n[idx])/(alpha*(M-1)) - 2/alpha + 1)))
    return w

def tukey2d(shape, alpha=0.25):
    ny, nx = shape
    wy = tukey_win(ny, alpha=alpha)
    wx = tukey_win(nx, alpha=alpha)
    return np.outer(wy, wx)

def bandpass_mask(shape, kmin_frac=0.06, kmax_frac=0.32):
    ny, nx = shape
    ky = spfft.fftfreq(ny)*ny
    kx = spfft.fftfreq(nx)*nx
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    kmax = K.max()
    return (K >= kmin_frac*kmax) & (K <= kmax_frac*kmax)

def bandpass_apply(f, mask):
    F = spfft.fft2(f)
    return spfft.ifft2(F*mask).real

from scipy.ndimage import gaussian_filter as gf
def topN_peaks(arr, N=80, min_sep=5):
    neigh = maximum_filter(arr, size=min_sep)
    peaks = (arr==neigh)
    lab, _ = label(peaks)
    boxes = find_objects(lab)
    centers=[]
    for sl in boxes:
        ys,xs=sl; sub=arr[ys,xs]
        iy,ix = np.unravel_index(np.argmax(sub), sub.shape)
        centers.append((ys.start+iy, xs.start+ix, arr[ys.start+iy, xs.start+ix]))
    centers.sort(key=lambda t:-t[2])
    return centers[:N]

def peaks_multiscale(arr, sigmas=(1.5, 3.0, 6.0), N=120):
    R = np.zeros_like(arr)
    for s in sigmas:
        R = np.maximum(R, -gaussian_laplace(arr, s))
    return topN_peaks(R, N=N, min_sep=5)

def match_rate(peaksA, peaksB, r_pix=6):
    if len(peaksA)==0 or len(peaksB)==0: return 0.0
    A = np.array([(y,x) for y,x,_ in peaksA]); B = np.array([(y,x) for y,x,_ in peaksB])
    d2 = ((A[:,None,:]-B[None,:,:])**2).sum(2)
    dmin = np.sqrt(d2.min(1))
    return float(np.mean(dmin<=r_pix))

def shuffle_phase(f, seed=0):
    F = spfft.fft2(f); A = np.abs(F)
    rng=np.random.default_rng(seed)
    phase = np.exp(1j*2*np.pi*rng.random(f.shape))
    return spfft.ifft2(A*phase).real

def best_shift_corr(A, B, max_shift=6):
    best_r=-2; best=(0,0)
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            Br = np.roll(np.roll(B, dy, axis=0), dx, axis=1)
            r,_ = pearsonr(A.ravel(), Br.ravel())
            if r>best_r:
                best_r=r; best=(dy,dx)
    return best_r, best

# --------- Exact θ(U) with caching ---------
THETA_CACHE = {}
def get_theta_grid(t_lo, t_hi, GRID_N, mp_dps=50):
    key = (float(t_lo), float(t_hi), int(GRID_N), int(mp_dps))
    if key in THETA_CACHE:
        return THETA_CACHE[key]
    mp.mp.dps = mp_dps
    t_grid = np.linspace(t_lo, t_hi, GRID_N)
    vals=[]
    for tv in t_grid:
        z = mp.zeta(0.5+1j*tv)
        vals.append(float(mp.arg(z)))
    theta_grid = np.unwrap(np.array(vals))
    THETA_CACHE[key] = (t_grid, theta_grid)
    return THETA_CACHE[key]

def theta_exact_from_U(U, t_lo=80.0, t_hi=6000.0, GRID_N=1000, mp_dps=50):
    Umin, Umax = float(np.min(U)), float(np.max(U))
    tmap = t_lo + (U-Umin)*(t_hi-t_lo)/(Umax-Umin+1e-12)
    t_grid, theta_grid = get_theta_grid(t_lo, t_hi, GRID_N, mp_dps)
    th = np.interp(tmap, t_grid, theta_grid)
    return th - np.mean(th)

# --- U embeddings ---
def embedding_field(name, xx, yy, kappa_like):
    ny, nx = kappa_like.shape
    if name=="radial-center":
        X0 = (xx - nx/2)/(nx/2); Y0 = (yy - ny/2)/(ny/2)
        return np.sqrt(X0**2 + Y0**2) + 1e-6
    elif name=="radial-from-peak":
        cy, cx = np.unravel_index(np.argmax(kappa_like), kappa_like.shape)
        Xc = (xx - cx)/(nx/2); Yc = (yy - cy)/(ny/2)
        return np.sqrt(Xc**2 + Yc**2) + 1e-6
    elif name=="radial-elliptic":
        y,x=np.mgrid[0:ny,0:nx]; w=np.maximum(kappa_like,0)
        W=w.sum()+1e-12
        y0=(y*w).sum()/W; x0=(x*w).sum()/W
        Y=y-y0; X=x-x0
        Ixx=( (X**2)*w ).sum()/W; Iyy=( (Y**2)*w ).sum()/W; Ixy=( (X*Y)*w ).sum()/W
        t=0.5*np.arctan2(2*Ixy, Ixx-Iyy)
        ct,st=np.cos(t),np.sin(t)
        Xp=(X*ct+Y*st)/(nx/2); Yp=(-X*st+Y*ct)/(ny/2)
        lam1=0.5*((Ixx+Iyy)+np.sqrt((Ixx-Iyy)**2+4*Ixy**2))
        lam2=0.5*((Ixx+Iyy)-np.sqrt((Ixx-Iyy)**2+4*Ixy**2))
        q=np.sqrt(max(lam2,1e-12)/max(lam1,1e-12))
        a=1.0; b=max(q,0.4)
        return np.sqrt( (Xp/a)**2 + (Yp/b)**2 ) + 1e-6
    else:
        raise ValueError("Unknown EMBEDDING")

# ------------ Core runner ------------
def run_primary(name, url, CFG):
    # load κ
    with fits.open(url) as hdul:
        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
        kappa = np.array(data, dtype=np.float64)
    # resize / preprocess
    if kappa.shape != (CFG["TARGET_N"], CFG["TARGET_N"]):
        zy=CFG["TARGET_N"]/kappa.shape[0]; zx=CFG["TARGET_N"]/kappa.shape[1]
        kappa = zoom(kappa, (zy, zx), order=1)
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    if CFG["SIGMA_SMOOTH"]>0:
        kappa = gaussian_filter(kappa, CFG["SIGMA_SMOOTH"])
    kappa = zscore(kappa)

    ny,nx = kappa.shape
    yy,xx = np.mgrid[0:ny,0:nx]
    W2D = tukey2d(kappa.shape, alpha=CFG["TUKEY_ALPHA"])
    BP_MASK = bandpass_mask(kappa.shape, kmin_frac=CFG["BANDPASS"][0], kmax_frac=CFG["BANDPASS"][1])

    # U -> θ -> κ_pred
    U = embedding_field(CFG["EMBEDDING"], xx, yy, kappa)
    theta = theta_exact_from_U(U, t_lo=CFG["T_LO"], t_hi=CFG["T_HI"], GRID_N=CFG["THETA_GRID_N"], mp_dps=50)
    theta_w  = theta * W2D
    theta_bp = bandpass_apply(theta_w, BP_MASK)
    kappa_w  = kappa * W2D
    kpred    = zscore(-laplacian_2d(theta_bp))

    # controls
    theta_sh = zscore(shuffle_phase(theta_bp, seed=42))
    kpred_sh = zscore(-laplacian_2d(theta_sh))
    U_roll = np.roll(np.roll(U, 64, axis=0), 64, axis=1)
    U_rot  = np.rot90(U, 1)
    th_roll = theta_exact_from_U(U_roll, t_lo=CFG["T_LO"], t_hi=CFG["T_HI"], GRID_N=CFG["THETA_GRID_N"], mp_dps=50)
    th_rot  = theta_exact_from_U(U_rot,  t_lo=CFG["T_LO"], t_hi=CFG["T_HI"], GRID_N=CFG["THETA_GRID_N"], mp_dps=50)
    kpred_roll = zscore(-laplacian_2d(bandpass_apply(th_roll*W2D, BP_MASK)))
    kpred_rot  = zscore(-laplacian_2d(bandpass_apply(th_rot *W2D, BP_MASK)))

    # -------- metrics（去趋势 + 多尺度峰） --------
    def metrics(kd, kp):
        kd2 = kd - gf(kd, 8)     # 去趋势
        kp2 = kp - gf(kp, 8)
        r,_  = pearsonr(kd2.ravel(), kp2.ravel())
        gbar, g2, common = mean_gamma2(kd2, kp2, lohi=CFG["SLOPE_FIT_FRAC"])
        gbar_jk, (glo, ghi) = jackknife_gamma2(g2, common, lohi=CFG["SLOPE_FIT_FRAC"], blocks=10)
        peaks_d = peaks_multiscale(kd2, N=CFG["TOP_N_PEAKS"])
        peaks_p = peaks_multiscale(kp2, N=CFG["TOP_N_PEAKS"])
        pmr = match_rate(peaks_d, peaks_p, r_pix=CFG["PEAK_MATCH_RADIUS"])
        r_best, (dy,dx) = best_shift_corr(kd2, kp2, max_shift=CFG["BEST_SHIFT_MAX"])
        return dict(r=r, r_best=r_best, shift=(dy,dx), gamma2=gbar, gamma2_jk=(gbar_jk, glo, ghi), peak=pmr)

    M_best = metrics(kappa_w, kpred)
    M_shuf = metrics(kappa_w, kpred_sh)
    M_roll = metrics(kappa_w, kpred_roll)
    M_rot  = metrics(kappa_w, kpred_rot)

    # ---- plots ----
    plt.figure(figsize=(5.8,4.8)); plt.imshow(kappa_w, origin='lower'); plt.colorbar(); plt.title(f"{name}: κ data (z, windowed)"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(5.8,4.8)); plt.imshow(kpred, origin='lower'); plt.colorbar(); plt.title("κ_pred ∝ −∇²θ(U) (z)"); plt.tight_layout(); plt.show()
    # spectra
    kt,Pt = isotropic_ps(zscore(theta_bp)); kp,Pp = isotropic_ps(kpred)
    def fit_slope(k, P, lo=0.10, hi=0.35):
        n=len(k); i0=int(n*lo); i1=int(n*hi)
        kk = k[i0:i1].astype(float); pp = P[i0:i1].astype(float)
        return float(np.polyfit(np.log(kk+1e-9), np.log(pp+1e-12), 1)[0])
    s_theta = fit_slope(kt,Pt,*PRIMARY["SLOPE_FIT_FRAC"]); s_pred=fit_slope(kp,Pp,*PRIMARY["SLOPE_FIT_FRAC"])
    plt.figure(figsize=(5.8,4.8))
    plt.loglog(kt,Pt,label="P_theta"); plt.loglog(kp,Pp,label="P_kappa_pred"); plt.legend(); plt.title("Isotropic power spectra"); plt.tight_layout(); plt.show()
    # coherence curves
    kd,Pd = isotropic_ps(kappa_w); kp2,Pp2 = isotropic_ps(kpred); kx,Px = isotropic_cross_ps(kappa_w,kpred)
    common = min(len(Pd),len(Pp2),len(Px)); Pd,Pp2,Px = Pd[:common],Pp2[:common],Px[:common]
    gamma2_curve = (Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp2,1e-20))
    kd,Pd = isotropic_ps(kappa_w); kp3,Pp3 = isotropic_ps(kpred_sh); kx,Px = isotropic_cross_ps(kappa_w,kpred_sh)
    common2 = min(len(Pd),len(Pp3),len(Px)); Pd,Pp3,Px = Pd[:common2],Pp3[:common2],Px[:common2]
    gamma2_sh_curve = (Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp3,1e-20))
    plt.figure(figsize=(5.8,4.8))
    plt.semilogy(np.arange(len(gamma2_curve)), gamma2_curve, label="γ²(data×pred)")
    plt.semilogy(np.arange(len(gamma2_sh_curve)), gamma2_sh_curve, label="γ²(shuffled)")
    plt.legend(); plt.title("Coherence vs k"); plt.tight_layout(); plt.show()

    # ---- summary + pass/fail ----
    def pf(m_real, m_ctrl, thresh_key):
        passed = (m_real >= THRESH[thresh_key]) and (m_real > max(m_ctrl, 1e-9)*1.5)
        return passed
    ok_gamma = pf(M_best["gamma2"], max(M_shuf["gamma2"], M_roll["gamma2"], M_rot["gamma2"]), "gamma2_min")
    ok_peak  = pf(M_best["peak"],  max(M_shuf["peak"],  M_roll["peak"],  M_rot["peak"]),  "peak_min")
    bonus_r  = (M_best["r_best"] >= THRESH["r_bonus"])

    print("\n=== INDEPENDENT TEST (PRIMARY, enhanced) ===")
    print(f"Dataset: {name}")
    print(f"Embedding={PRIMARY['EMBEDDING']}, t=({PRIMARY['T_LO']},{PRIMARY['T_HI']}), θgrid={PRIMARY['THETA_GRID_N']}, "
          f"alpha={PRIMARY['TUKEY_ALPHA']}, band={PRIMARY['BANDPASS']}")
    print(f"[BEST]     r={M_best['r']:+.3f}, r_best={M_best['r_best']:+.3f}@{M_best['shift']}, "
          f"γ²={M_best['gamma2']:.3f} [JK {M_best['gamma2_jk'][0]:.3f} CI {M_best['gamma2_jk'][1]:.3f},{M_best['gamma2_jk'][2]:.3f}], "
          f"peak={M_best['peak']:.3f}")
    print(f"[SHUFFLE]  r={M_shuf['r']:+.3f}, r_best={M_shuf['r_best']:+.3f}, γ²={M_shuf['gamma2']:.3f}, peak={M_shuf['peak']:.3f}")
    print(f"[U-ROLL]   r={M_roll['r']:+.3f}, r_best={M_roll['r_best']:+.3f}, γ²={M_roll['gamma2']:.3f}, peak={M_roll['peak']:.3f}")
    print(f"[U-ROT90]  r={M_rot['r']:+.3f}, r_best={M_rot['r_best']:+.3f}, γ²={M_rot['gamma2']:.3f}, peak={M_rot['peak']:.3f}")
    print(f"[Sanity] slope(P_theta)={s_theta:.2f}, slope(P_pred)={s_pred:.2f}, diff={s_pred-s_theta:.2f}")

    print(f"\nPASS (PRIMARY):  γ²>= {THRESH['gamma2_min']} ? {ok_gamma}   |  peak>= {THRESH['peak_min']} ? {ok_peak}   |  bonus best-shift r>= {THRESH['r_bonus']} ? {bonus_r}")
    print("\nDone.")
    return {"best":M_best, "shuffle":M_shuf, "roll":M_roll, "rot":M_rot, "ok_gamma":ok_gamma, "ok_peak":ok_peak, "bonus_r":bonus_r}

# ------------- Run PRIMARY only (fast) -------------
_ = run_primary(DATASET_NAME, DATASET_URL, PRIMARY)
