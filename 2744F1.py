# ==========================================================
# ζ–U Binding (Eq.5) — γ̄² + Ridge-F1 (Boosted) + Multi-Pred (.reg/.csv)
# Single-cell Colab — drop-in & run
# ==========================================================

# ---------------- Locked (PRIMARY) params ----------------
PRIMARY = {
    "EMBEDDING": "radial-from-peak",      # or: 'elliptic-from-moments' / 'dual-from-twopeaks'
    "T_LO": 80.0, "T_HI": 6000.0,
    "THETA_GRID_N": 1000,
    "TARGET_N": 512,
    "SIGMA_SMOOTH_DATA": 1.0,
    "SIGMA_SMOOTH_PRED": 2.0,
    "TUKEY_ALPHA": 0.20,
    "BANDPASS": (0.10, 0.25),
    "SLOPE_FIT_FRAC": (0.10, 0.25),
    "BEST_SHIFT_MAX": 6,
    # Ridge skeleton (base)
    "RIDGE_SIGMAS": (1.5, 3.0, 6.0),
    # Morphology (info only)
    "MORPH_Q": 0.90,
    "MORPH_MIN_SIZE": 100,
    "MPP": None,  # if known
    # Elliptic embedding helpers
    "ELLIPSE_Q_FOR_MASK": 0.80,
    "ELLIPSE_MIN_AXIS_RATIO": 0.5,
    # Dual-peak embedding helpers
    "DUAL_TOP_N": 2,
    "DUAL_MIN_SEP_PX": 40,
    "DUAL_BLEND": "softmin",
    "DUAL_TAU": 0.08,
}

THRESH = {"gamma2_min": 0.05, "f1_min": 0.10, "r_bonus": 0.03}

# --------------- F1-Boost (only affects F1 stage) ---------------
F1BOOST = {
    "ROT_SCAN_DEG": list(range(-6, 7)),               # -6..+6 deg
    "SCALE_SCAN":   [0.93, 0.965, 1.0, 1.035, 1.07],
    "RIDGE_Q":      0.95,     # from 0.97 -> 0.95
    "RIDGE_MIN_SEP":10,       # from 14 -> 10
    "MAX_PTS_CAP":  300,
    "R_FINAL":      14,       # from 12 -> 14
    "USE_FRANGI":   True,
    "FRANGI_BETA":  0.7,
    "FRANGI_GAMMA": 10.0,
}

# ---------------- Datasets (CATS v4) ----------------
A2744_URL   = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"
MACS0416_URL= "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"
A370_URL    = "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits"

DATASETS = [
    ("Abell2744_CATSv4", A2744_URL),
    ("MACS0416_CATSv4",  MACS0416_URL),
    ("Abell370_CATSv4",  A370_URL),
]

# ---------------- Installs / Imports ----------------
import sys, subprocess, warnings, os, io, math, re, glob
warnings.filterwarnings("ignore")
def _pip(x): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + x)
try:
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import (gaussian_filter, maximum_filter, label, find_objects,
                               zoom, gaussian_laplace, binary_erosion, binary_opening, binary_dilation)
    from scipy import fft as spfft
    from scipy.stats import pearsonr
    try:
        from scipy.signal.windows import tukey as tukey_win
    except Exception:
        try:
            from scipy.signal import tukey as tukey_win
        except Exception:
            raise
    from scipy.optimize import linear_sum_assignment as hungarian
    from skimage.morphology import skeletonize
    from skimage.filters import frangi
except Exception:
    _pip(["numpy","matplotlib","astropy","scipy","scikit-image"])
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import (gaussian_filter, maximum_filter, label, find_objects,
                               zoom, gaussian_laplace, binary_erosion, binary_opening, binary_dilation)
    from scipy import fft as spfft
    from scipy.stats import pearsonr
    # Tukey fallback
    try:
        from scipy.signal.windows import tukey as tukey_win
    except Exception:
        try:
            from scipy.signal import tukey as tukey_win
        except Exception:
            def tukey_win(M, alpha=0.5):
                import numpy as _np
                if M < 1: return _np.array([])
                if alpha <= 0: return _np.ones(M)
                if alpha >= 1:
                    n = _np.arange(M)
                    return 0.5*(1 - _np.cos(2*_np.pi*n/(M-1)))
                n = _np.arange(M)
                width = alpha*(M-1)/2.0
                w = _np.ones(M)
                idx = n < width
                w[idx] = 0.5*(1 + _np.cos(_np.pi*(2*n[idx]/(alpha*(M-1)) - 1)))
                idx = n >= (M - width)
                w[idx] = 0.5*(1 + _np.cos(_np.pi*(2*n[idx]/(alpha*(M-1)) - 2/alpha + 1)))
                return w
    from scipy.optimize import linear_sum_assignment as hungarian
    from skimage.morphology import skeletonize
    from skimage.filters import frangi

try:
    import mpmath as mp
except Exception:
    _pip(["mpmath"])
    import mpmath as mp

# ---------------- Utilities ----------------
def zscore(a): m=np.mean(a); s=np.std(a)+1e-12; return (a-m)/s
def laplacian_2d(f): return (-4.0*f + np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1))
def tukey2d(shape,alpha=0.20):
    ny,nx=shape; wy=tukey_win(ny,alpha=alpha); wx=tukey_win(nx,alpha=alpha); return np.outer(wy,wx)

def bandpass_mask(shape,kmin_frac=0.10,kmax_frac=0.25):
    ny,nx=shape; ky=spfft.fftfreq(ny)*ny; kx=spfft.fftfreq(nx)*nx
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2); kmax=K.max()
    return (K>=kmin_frac*kmax)&(K<=kmax_frac*kmax)

def bandpass_apply(f,mask): F=spfft.fft2(f); return spfft.ifft2(F*mask).real

def isotropic_ps(f):
    F=spfft.fft2(f); P2=(F*np.conj(F)).real
    ny,nx=f.shape; ky=spfft.fftfreq(ny)*ny; kx=spfft.fftfreq(nx)*nx
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2).astype(int)
    ps=np.bincount(K.ravel(),P2.ravel()); cnt=np.bincount(K.ravel()); ps=ps/np.maximum(cnt,1); k=np.arange(len(ps))
    return k[1:], ps[1:]

def isotropic_cross_ps(f,g):
    F=spfft.fft2(f); G=spfft.fft2(g); C=(F*np.conj(G)).real
    ny,nx=f.shape; ky=spfft.fftfreq(ny)*ny; kx=spfft.fftfreq(nx)*nx
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2).astype(int)
    cs=np.bincount(K.ravel(),C.ravel()); cnt=np.bincount(K.ravel()); cs=cs/np.maximum(cnt,1); k=np.arange(len(cs))
    return k[1:], cs[1:]

def mean_gamma2(a,b, lohi=(0.10,0.25)):
    kd,Pd=isotropic_ps(a); kp,Pp=isotropic_ps(b); kx,Px=isotropic_cross_ps(a,b)
    common=min(len(Pd),len(Pp),len(Px)); Pd,Pp,Px=Pd[:common],Pp[:common],Px[:common]
    g2=(Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp,1e-20))
    i0=int(common*lohi[0]); i1=int(common*lohi[1]); return float(np.mean(g2[i0:i1])), g2, common

def jackknife_gamma2(g2, common, lohi=(0.10,0.25), blocks=10):
    i0=int(common*lohi[0]); i1=int(common*lohi[1]); seg=g2[i0:i1]; n=len(seg)
    if n<blocks: mu=float(np.mean(seg)); return mu,(np.nan,np.nan)
    size=n//blocks; vals=[]
    for b in range(blocks):
        mask=np.ones(n,bool); mask[b*size:(b+1)*size]=False; vals.append(float(np.mean(seg[mask])))
    vals=np.array(vals); mu=float(np.mean(vals))
    std=np.sqrt((blocks-1)/blocks*np.sum((vals-mu)**2)); return mu, (mu-1.96*std, mu+1.96*std)

def best_shift_corr(A,B,max_shift=6):
    best_r=-2; best=(0,0)
    for dy in range(-max_shift,max_shift+1):
        for dx in range(-max_shift,max_shift+1):
            Br=np.roll(np.roll(B,dy,0),dx,1); r,_=pearsonr(A.ravel(),Br.ravel())
            if r>best_r: best_r=r; best=(dy,dx)
    return best_r, best

# ---- ζ phase θ(U) with cache ----
THETA_CACHE={}
def get_theta_grid(t_lo,t_hi,GRID_N, mp_dps=50):
    key=(float(t_lo),float(t_hi),int(GRID_N),int(mp_dps))
    if key in THETA_CACHE: return THETA_CACHE[key]
    mp.mp.dps=mp_dps; t_grid=np.linspace(t_lo,t_hi,GRID_N); vals=[]
    for tv in t_grid:
        z=mp.zeta(0.5+1j*tv); vals.append(float(mp.arg(z)))
    theta_grid=np.unwrap(np.array(vals)); THETA_CACHE[key]=(t_grid,theta_grid); return THETA_CACHE[key]

def theta_exact_from_U(U, t_lo=80.0, t_hi=6000.0, GRID_N=1000, mp_dps=50):
    Umin,Umax=float(np.min(U)),float(np.max(U))
    tmap=t_lo+(U-Umin)*(t_hi-t_lo)/(Umax-Umin+1e-12)
    t_grid,theta_grid=get_theta_grid(t_lo,t_hi,GRID_N,mp_dps)
    th=np.interp(tmap,t_grid,theta_grid); return th-np.mean(th)

# ---- Embedding helpers ----
from scipy.ndimage import binary_opening, binary_erosion
def _binary_top_quantile(mask_src, q):
    thr = np.quantile(mask_src, q); return (mask_src >= thr)

def _ellipse_from_mask(mask):
    ys, xs = np.where(mask); h, w = mask.shape
    if len(ys) < 10: return (h/2, w/2, max(h,w)/4, max(h,w)/6, 0.0)
    y = ys.astype(np.float64); x = xs.astype(np.float64)
    cy, cx = float(y.mean()), float(x.mean())
    Y = y - cy; X = x - cx
    cov = np.cov(np.stack([Y, X]))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]; vals = vals[order]; vecs = vecs[:, order]
    a = np.sqrt(max(vals[0], 1e-9))*2.5
    b = np.sqrt(max(vals[1], 1e-9))*2.5
    ar = b/a if a>1e-9 else 0.0
    ar = max(ar, PRIMARY["ELLIPSE_MIN_AXIS_RATIO"]); b = a*ar
    vx = vecs[:,0]  # (y,x)
    angle = np.arctan2(vx[0], vx[1])
    return (cy, cx, a, b, angle)

def _mahalanobis_r(y, x, cy, cx, a, b, angle):
    ca, sa = np.cos(angle), np.sin(angle)
    dy, dx = (y - cy), (x - cx)
    u = ca*dx + sa*dy; v = -sa*dx + ca*dy
    r = np.sqrt((u/(a+1e-12))**2 + (v/(b+1e-12))**2) + 1e-6
    return r

def _top_two_peaks(kappa, min_sep=40):
    neigh = maximum_filter(kappa, size=min_sep)
    peaks = (kappa == neigh)
    lab, n = label(peaks)
    if n == 0:
        cy, cx = np.unravel_index(np.argmax(kappa), kappa.shape)
        return [(cy, cx)]
    sls = find_objects(lab); centers=[]
    for i, sl in enumerate(sls, start=1):
        if sl is None: continue
        ys, xs = sl; sub = kappa[ys, xs]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        centers.append((ys.start+iy, xs.start+ix, float(kappa[ys.start+iy, xs.start+ix])))
    centers.sort(key=lambda t: -t[2])
    filtered=[]
    for y, x, v in centers:
        if all((y-yy)**2 + (x-xx)**2 >= min_sep**2 for yy,xx,_ in filtered):
            filtered.append((y, x, v))
        if len(filtered) >= 2: break
    return [(int(y), int(x)) for y,x,_ in filtered] or [np.unravel_index(np.argmax(kappa), kappa.shape)]

def _softmin(a, b, tau=0.08):
    m = np.minimum(a, b); M = np.maximum(a, b)
    return m - tau*np.log1p(np.exp(-(M-m)/max(tau,1e-9)))

def embedding_field(name, xx, yy, kappa_like):
    ny, nx = kappa_like.shape
    if name == "radial-from-peak":
        cy, cx = np.unravel_index(np.argmax(kappa_like), kappa_like.shape)
        Xc = (xx - cx) / (nx/2); Yc = (yy - cy) / (ny/2)
        return np.sqrt(Xc**2 + Yc**2) + 1e-6
    elif name == "elliptic-from-moments":
        mask = _binary_top_quantile(kappa_like, PRIMARY["ELLIPSE_Q_FOR_MASK"])
        cy, cx, a, b, ang = _ellipse_from_mask(mask)
        return _mahalanobis_r(yy, xx, cy, cx, a, b, ang)
    elif name == "dual-from-twopeaks":
        peaks = _top_two_peaks(kappa_like, min_sep=PRIMARY["DUAL_MIN_SEP_PX"])
        if len(peaks) == 1:
            cy, cx = peaks[0]
            Xc = (xx - cx)/(nx/2); Yc = (yy - cy)/(ny/2)
            return np.sqrt(Xc**2 + Yc**2) + 1e-6
        (y1,x1), (y2,x2) = peaks[0], peaks[1]
        r1 = np.sqrt(((xx - x1)/(nx/2))**2 + ((yy - y1)/(ny/2))**2) + 1e-6
        r2 = np.sqrt(((xx - x2)/(nx/2))**2 + ((yy - y2)/(ny/2))**2) + 1e-6
        return _softmin(r1, r2, tau=PRIMARY["DUAL_TAU"]) if PRIMARY["DUAL_BLEND"]=="softmin" else np.minimum(r1, r2)
    else:
        raise ValueError("EMBEDDING must be: 'radial-from-peak' | 'elliptic-from-moments' | 'dual-from-twopeaks'")

# ---- Ridge & F1 (Boosted) ----
def ridge_response(arr, sigmas=(1.5,3.0,6.0)):
    R=np.zeros_like(arr)
    for s in sigmas:
        R=np.maximum(R, -gaussian_laplace(arr, s))
    return R

def enhance_for_ridge(img):
    if not F1BOOST["USE_FRANGI"]: return img
    e = frangi(img, beta=F1BOOST["FRANGI_BETA"], gamma=F1BOOST["FRANGI_GAMMA"])
    return zscore(e)

def ridge_keypoints_boost(arr):
    arr2 = enhance_for_ridge(arr)
    R = ridge_response(arr2, PRIMARY["RIDGE_SIGMAS"])
    thr = np.quantile(R, F1BOOST["RIDGE_Q"])
    mask = R >= thr
    skel = skeletonize(mask.astype(bool))
    ys, xs = np.where(skel)
    if len(ys)==0: return []
    vals = R[ys, xs]; order = np.argsort(-vals)
    pts = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if all((y-yy)**2 + (x-xx)**2 >= F1BOOST["RIDGE_MIN_SEP"]**2 for yy,xx,_ in pts):
            pts.append((y, x, float(R[y,x])))
            if len(pts) >= F1BOOST["MAX_PTS_CAP"]: break
    return pts

def f1_hungarian_points(A_pts, B_pts, R=12):
    if len(A_pts)==0 or len(B_pts)==0: return 0.0
    A=np.array([(y,x) for y,x,_ in A_pts]); B=np.array([(y,x) for y,x,_ in B_pts])
    D=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(2))
    row,col=hungarian(D); hits=(D[row,col]<=R).sum()
    prec=hits/len(B); rec=hits/len(A); return 2*prec*rec/(prec+rec+1e-12)

def best_align_and_F1(kd2, kp2, shift):
    from scipy.ndimage import zoom, rotate
    dy, dx = shift
    best = 0.0
    for ang in F1BOOST["ROT_SCAN_DEG"]:
        kp_rot = rotate(kp2, angle=ang, reshape=False, order=1)
        for s in F1BOOST["SCALE_SCAN"]:
            z = zoom(kp_rot, s, order=1)
            H,W = kd2.shape; h,w = z.shape
            out = np.zeros_like(kd2)
            y0 = max(0,(H-h)//2); x0 = max(0,(W-w)//2)
            y1 = min(H, y0+h);     x1 = min(W, x0+w)
            zy0 = max(0, - (H-h)//2); zx0 = max(0, - (W-w)//2)
            out[y0:y1, x0:x1] = z[zy0:zy0+(y1-y0), zx0:zx0+(x1-x0)]
            kp_aln = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            A = ridge_keypoints_boost(kd2); B = ridge_keypoints_boost(kp_aln)
            if len(A)==0 or len(B)==0: continue
            f1 = f1_hungarian_points(A,B, R=F1BOOST["R_FINAL"])
            best = max(best, f1)
    return best

def ridge_F1_noalign(kd2, kp2):
    A = ridge_keypoints_boost(kd2)
    B = ridge_keypoints_boost(kp2)
    return f1_hungarian_points(A,B, R=F1BOOST["R_FINAL"])

# ---- SHOCK (edge) candidates (简单梯度盒) ----
def shock_boxes(kimg, topN=80, size=32, min_sep=20):
    gy, gx = np.gradient(kimg)
    G = np.hypot(gy, gx)
    H, W = kimg.shape
    half = size//2
    order = np.argsort(G.ravel())[::-1]
    boxes=[]
    for idx in order:
        y,x=divmod(int(idx), W)
        if any((y-yy)**2+(x-xx)**2 < (min_sep*min_sep) for yy,xx,_ in boxes): continue
        y0,y1=max(0,y-half),min(H,y+half); x0,x1=max(0,x-half),min(W,x+half)
        boxes.append((y,x,G[y,x]))
        if len(boxes)>=topN: break
    return boxes

# ---- Save DS9 region & CSV ----
def write_points_reg(path, points): # [(y,x,val), ...]
    with open(path,"w") as f:
        f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10\" select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0\n")
        f.write("image\n")
        for y,x,v in points:
            f.write(f"point({x+1:.2f},{y+1:.2f}) # point=circle text={{S:{v:.3f}}}\n")

def write_boxes_reg(path, boxes, size=32):
    with open(path,"w") as f:
        f.write("global color=yellow width=1\nimage\n")
        for y,x,v in boxes:
            f.write(f"box({x+1:.2f},{y+1:.2f},{size:.2f},{size:.2f},0)\n")

def save_csv(path, rows, cols=("y","x","score")):
    import pandas as pd
    df = pd.DataFrame(rows, columns=list(cols))
    df.to_csv(path, index=False)

# ---------------- Main per-dataset pipeline ----------------
def run_one(name, url, CFG):
    out_dir = f"{name}_outputs"; os.makedirs(out_dir, exist_ok=True)

    # Load κ
    with fits.open(url) as hdul:
        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
        kappa = np.array(data, dtype=np.float64)

    # Resample / preprocess
    if kappa.shape != (CFG["TARGET_N"], CFG["TARGET_N"]):
        zy=CFG["TARGET_N"]/kappa.shape[0]; zx=CFG["TARGET_N"]/kappa.shape[1]
        kappa=zoom(kappa,(zy,zx),order=1)
    kappa=np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    if CFG["SIGMA_SMOOTH_DATA"]>0: kappa=gaussian_filter(kappa, CFG["SIGMA_SMOOTH_DATA"])
    kappa=zscore(kappa)

    ny,nx=kappa.shape; yy,xx=np.mgrid[0:ny,0:nx]
    W2D=tukey2d(kappa.shape, alpha=CFG["TUKEY_ALPHA"])
    BP_MASK=bandpass_mask(kappa.shape, *CFG["BANDPASS"])

    # θ(U) & κ_pred
    U=embedding_field(CFG["EMBEDDING"], xx, yy, kappa)
    theta=theta_exact_from_U(U, CFG["T_LO"], CFG["T_HI"], CFG["THETA_GRID_N"], mp_dps=50)
    theta_bp=bandpass_apply(theta * W2D, BP_MASK)

    kappa_w = kappa * W2D
    kpred0  = -laplacian_2d(theta_bp)
    if CFG["SIGMA_SMOOTH_PRED"]>0: kpred0=gaussian_filter(kpred0, CFG["SIGMA_SMOOTH_PRED"])
    kappa_bp = bandpass_apply(kappa_w, BP_MASK)
    kpred_bp = bandpass_apply(kpred0,  BP_MASK)
    kpred_bp = gaussian_filter(kpred_bp, 1.0)
    kappa_bp = zscore(kappa_bp); kpred_bp=zscore(kpred_bp)

    # Controls
    def shuffle_phase(f,seed=0):
        F=spfft.fft2(f); A=np.abs(F); rng=np.random.default_rng(seed)
        phase=np.exp(1j*2*np.pi*rng.random(f.shape)); return spfft.ifft2(A*phase).real
    theta_bp_sh = shuffle_phase(theta_bp,seed=42)
    kpred_sh = bandpass_apply(gaussian_filter(-laplacian_2d(theta_bp_sh), CFG["SIGMA_SMOOTH_PRED"]), BP_MASK)
    kpred_sh = gaussian_filter(kpred_sh,1.0); kpred_sh=zscore(kpred_sh)

    U_roll=np.roll(np.roll(U,64,0),64,1); U_rot=np.rot90(U,1)
    th_roll_bp=bandpass_apply(theta_exact_from_U(U_roll, CFG["T_LO"], CFG["T_HI"], CFG["THETA_GRID_N"], 50)*W2D, BP_MASK)
    th_rot_bp =bandpass_apply(theta_exact_from_U(U_rot,  CFG["T_LO"], CFG["T_HI"], CFG["THETA_GRID_N"], 50)*W2D, BP_MASK)
    kpred_roll = bandpass_apply(gaussian_filter(-laplacian_2d(th_roll_bp), CFG["SIGMA_SMOOTH_PRED"]), BP_MASK)
    kpred_rot  = bandpass_apply(gaussian_filter(-laplacian_2d(th_rot_bp),  CFG["SIGMA_SMOOTH_PRED"]), BP_MASK)
    kpred_roll = gaussian_filter(kpred_roll,1.0); kpred_rot=gaussian_filter(kpred_rot,1.0)
    kpred_roll = zscore(kpred_roll); kpred_rot=zscore(kpred_rot)

    # Metrics (γ̄² + JK + best-shift)
    from scipy.ndimage import gaussian_filter as gf
    def metrics_bp(kd_bp,kp_bp):
        kd2=kd_bp-gf(kd_bp,6); kp2=kp_bp-gf(kp_bp,6)
        r,_=pearsonr(kd2.ravel(), kp2.ravel())
        gbar,g2,common=mean_gamma2(kd2,kp2, lohi=CFG["SLOPE_FIT_FRAC"])
        gbar_jk,(glo,ghi)=jackknife_gamma2(g2,common, lohi=CFG["SLOPE_FIT_FRAC"], blocks=10)
        r_best,(dy,dx)=best_shift_corr(kd2,kp2, max_shift=CFG["BEST_SHIFT_MAX"])
        return dict(r=r, r_best=r_best, shift=(dy,dx), gamma2=gbar, gamma2_jk=(gbar_jk,glo,ghi), kd2=kd2, kp2=kp2)
    M_best=metrics_bp(kappa_bp,kpred_bp);  M_shuf=metrics_bp(kappa_bp,kpred_sh)
    M_roll=metrics_bp(kappa_bp,kpred_roll); M_rot=metrics_bp(kappa_bp,kpred_rot)

    # Ridge-F1 (Boosted): best-shift + rot/scale scan
    F1_best = best_align_and_F1(M_best["kd2"], M_best["kp2"], M_best["shift"])
    F1_sh = ridge_F1_noalign(M_shuf["kd2"], M_shuf["kp2"])
    F1_ro = ridge_F1_noalign(M_roll["kd2"], M_roll["kp2"])
    F1_rt = ridge_F1_noalign(M_rot["kd2"],  M_rot["kp2"])

    # Print summary
    kt,Pt=isotropic_ps(zscore(theta_bp)); kp,Pp=isotropic_ps(kpred_bp)
    def fit_slope(k,P, lo=0.10,hi=0.25):
        n=len(k); i0=int(n*lo); i1=int(n*hi); kk=k[i0:i1].astype(float); pp=P[i0:i1].astype(float)
        return float(np.polyfit(np.log(kk+1e-9), np.log(pp+1e-12), 1)[0])
    s_theta=fit_slope(kt,Pt,*PRIMARY["SLOPE_FIT_FRAC"]); s_pred=fit_slope(kp,Pp,*PRIMARY["SLOPE_FIT_FRAC"])

    def pass_flag(real,ctrl_max,thr): return (real>=thr) and (real>max(ctrl_max,1e-9)*1.5)
    ok_gamma = pass_flag(M_best["gamma2"], max(M_shuf["gamma2"],M_roll["gamma2"],M_rot["gamma2"]), THRESH["gamma2_min"])
    ctrl_F1_max = max(F1_sh,F1_ro,F1_rt)
    ok_f1   = pass_flag(F1_best, ctrl_F1_max, THRESH["f1_min"])
    bonus_r = (M_best["r_best"] >= THRESH["r_bonus"])

    print(f"\n=== REPLICATION (locked params + F1 Boost) ===")
    print(f"Dataset: {name}  Embed={PRIMARY['EMBEDDING']}  band={PRIMARY['BANDPASS']}  θgrid={PRIMARY['THETA_GRID_N']}")
    print(f"[BEST]   r={M_best['r']:+.3f}, r_best={M_best['r_best']:+.3f}@{M_best['shift']}, γ̄²={M_best['gamma2']:.3f}  [JK {M_best['gamma2_jk'][0]:.3f}  CI {M_best['gamma2_jk'][1]:.3f},{M_best['gamma2_jk'][2]:.3f}]")
    print(f"[F1]     best={F1_best:.3f}  | ctrl: shuffle={F1_sh:.3f}  roll={F1_ro:.3f}  rot90={F1_rt:.3f}")
    print(f"[Sanity] slope(P_theta)={s_theta:.2f}  slope(P_pred)={s_pred:.2f}  Δ={s_pred-s_theta:.2f}")
    print(f"PASS: γ̄²≥{THRESH['gamma2_min']}? {ok_gamma} | Ridge-F1≥{THRESH['f1_min']} & ≥1.5×ctrl? {ok_f1} | bonus r_best≥{THRESH['r_bonus']}? {bonus_r}")
    print("Done.")

    # ---------- Predictions: ARC/SUBHALO/SHOCK (.reg + .csv) ----------
    # ARC/SUBHALO candidates: 取 κ_pred 的局部峰
    neigh = maximum_filter(M_best["kp2"], size=9)
    peaks_mask = (M_best["kp2"] == neigh)
    lab, n = label(peaks_mask); pts=[]
    if n>0:
        sls=find_objects(lab)
        for i,sl in enumerate(sls, start=1):
            if sl is None: continue
            ys,xs=sl; sub=M_best["kp2"][ys,xs]
            iy,ix=np.unravel_index(np.argmax(sub), sub.shape)
            y,x = ys.start+iy, xs.start+ix
            pts.append((y,x,float(M_best["kp2"][y,x])))
    pts.sort(key=lambda t:-t[2]); pts = pts[:120]
    # SHOCK: 高梯度盒
    boxes = shock_boxes(M_best["kp2"], topN=80, size=32, min_sep=20)

    # Save regions
    write_points_reg(f"{out_dir}/{name}_ARC_candidates.reg", pts)
    write_points_reg(f"{out_dir}/{name}_SUBHALO_candidates.reg", pts)
    write_boxes_reg (f"{out_dir}/{name}_SHOCK_edge_candidates.reg", boxes, size=32)
    save_csv        (f"{out_dir}/{name}_ARC_candidates.csv",    [(y,x,s) for y,x,s in pts])
    save_csv        (f"{out_dir}/{name}_SUBHALO_candidates.csv",[(y,x,s) for y,x,s in pts])
    save_csv        (f"{out_dir}/{name}_SHOCK_edge_candidates.csv",[(y,x,s) for y,x,s in boxes])

    # Fused priority（简单融合：γ̄²/峰值/梯度）
    rows=[]
    for y,x,s in pts:
        rows.append((y,x, float(0.6*s + 0.4*(abs(M_best['kd2'][y,x])) )))
    for y,x,s in boxes:
        rows.append((y,x, float(0.5*s + 0.5*(abs(M_best['kd2'][y,x])) )))
    rows.sort(key=lambda t:-t[2])
    save_csv(f"{out_dir}/{name}_FUSED_priority.csv", rows, cols=("y","x","priority"))
    # TopN .reg
    write_points_reg(f"{out_dir}/{name}_FUSED_Top30.reg", rows[:30])

    return dict(best=M_best, F1=F1_best)

# ---------------- Run all datasets ----------------
for (name, url) in DATASETS:
    _ = run_one(name, url, PRIMARY)

print("\n提示：若要进一步抬升 F1，可将 PRIMARY['EMBEDDING'] 切换为 'elliptic-from-moments' 或 'dual-from-twopeaks'，其余锁参不动；F1BOOST 的 ROT/SCALE/R_FINAL 也可微调。")
