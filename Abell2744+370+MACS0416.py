# ==========================================================
# ζ–U Binding (Eq.5) — Cross-coherence (γ̄²) + Ridge-F1 + Morphology
# Full clean, single-cell Colab script
# ==========================================================

# ---------------- Locked (PRIMARY) params: keep fixed for replication ----------------
PRIMARY = {
    # U embedding: "radial-from-peak" | "elliptic-from-moments" | "dual-from-twopeaks"
    "EMBEDDING": "radial-from-peak",      # 建议再试："elliptic-from-moments" / "dual-from-twopeaks"
    "T_LO": 80.0, "T_HI": 6000.0,         # ζ 相位 t-区间
    "THETA_GRID_N": 1000,                 # ζ 相位预计算网格
    "TARGET_N": 512,                      # 统一重采样尺寸
    "SIGMA_SMOOTH_DATA": 1.0,             # κ data 轻平滑
    "SIGMA_SMOOTH_PRED": 2.0,             # κ_pred 轻平滑
    "TUKEY_ALPHA": 0.20,                  # 2D Tukey 窗
    "BANDPASS": (0.10, 0.25),             # 带通区（锁参）
    "SLOPE_FIT_FRAC": (0.10, 0.25),       # 幂谱斜率拟合区
    "BEST_SHIFT_MAX": 6,                  # best-shift 搜索窗口
    # Ridge skeleton settings (robust geometry)
    "RIDGE_SIGMAS": (1.5, 3.0, 6.0),      # LoG 多尺度
    "RIDGE_Q": 0.97,                      # 脊线二值阈值（分位）
    "RIDGE_MIN_SEP": 14,                  # 脊线采样点最小间距（px）
    "F1_MATCH_RADIUS": 12,                # 匹配半径（px）
    # Morphology
    "MORPH_Q": 0.90,
    "MORPH_MIN_SIZE": 100,
    "MPP": None,                          # 可设物理尺（Mpc/px）；未知则 None
    # Elliptic embedding helpers
    "ELLIPSE_Q_FOR_MASK": 0.80,           # 计算二阶矩时取 κ 的上分位阈
    "ELLIPSE_MIN_AXIS_RATIO": 0.5,        # 轴比下限
    # Dual-peak embedding helpers
    "DUAL_TOP_N": 2,
    "DUAL_MIN_SEP_PX": 40,
    "DUAL_BLEND": "softmin",              # "min" or "softmin"
    "DUAL_TAU": 0.08,                     # softmin 温度
    # Affine (scale) scan for Ridge-F1 after best-shift
    "AFFINE_SCALE_SCAN": [0.95, 0.975, 1.0, 1.025, 1.05],
}
THRESH = {"gamma2_min": 0.05, "f1_min": 0.10, "r_bonus": 0.03}

# ---------------- Datasets (CATS v4 kappa FITS; direct links) ----------------
A2744_URL   = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"
MACS0416_URL= "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"
A370_URL    = "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits"

DATASETS = [
    ("Abell2744_CATSv4", A2744_URL),
    ("MACS0416_CATSv4",  MACS0416_URL),
    ("Abell370_CATSv4",  A370_URL),
]

# ---------------- Installs / Imports ----------------
import sys, subprocess, warnings
warnings.filterwarnings("ignore")
def _pip(x): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + x)
try:
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import (gaussian_filter, maximum_filter, label, find_objects,
                               zoom, gaussian_laplace, binary_erosion, binary_opening, binary_dilation)
    from scipy import fft as spfft
    from scipy.stats import pearsonr
    from scipy.signal import tukey
    from scipy.optimize import linear_sum_assignment as hungarian
    from skimage.morphology import skeletonize
except Exception:
    _pip(["numpy","matplotlib","astropy","scipy","scikit-image"])
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import (gaussian_filter, maximum_filter, label, find_objects,
                               zoom, gaussian_laplace, binary_erosion, binary_opening, binary_dilation)
    from scipy import fft as spfft
    from scipy.stats import pearsonr
    from scipy.signal import tukey
    from scipy.optimize import linear_sum_assignment as hungarian
    from skimage.morphology import skeletonize

try:
    import mpmath as mp
except Exception:
    _pip(["mpmath"])
    import mpmath as mp

# ---------------- Utilities ----------------
def zscore(a): m=np.mean(a); s=np.std(a)+1e-12; return (a-m)/s
def laplacian_2d(f): return (-4.0*f + np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1))
def tukey2d(shape,alpha=0.20):
    ny,nx=shape; wy=tukey(ny,alpha=alpha); wx=tukey(nx,alpha=alpha); return np.outer(wy,wx)

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
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(ys) < 10:
        return (h/2, w/2, max(h,w)/4, max(h,w)/6, 0.0)  # fallback
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

# ---- Embedding field (three modes) ----
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

# ---- Ridge & F1 ----
def ridge_response(arr, sigmas=(1.5,3.0,6.0)):
    R=np.zeros_like(arr)
    for s in sigmas:
        R=np.maximum(R, -gaussian_laplace(arr, s))
    return R

def ridge_keypoints(arr, q=0.97, min_sep=14, max_pts=120):
    R = ridge_response(arr, PRIMARY["RIDGE_SIGMAS"])
    thr = np.quantile(R, q)
    mask = R >= thr
    skel = skeletonize(mask.astype(bool))
    ys, xs = np.where(skel)
    if len(ys)==0: return []
    vals = R[ys, xs]; order = np.argsort(-vals)
    pts = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if all((y-yy)**2 + (x-xx)**2 >= min_sep**2 for yy,xx,_ in pts):
            pts.append((y, x, float(R[y,x])))
            if len(pts) >= max_pts: break
    return pts

def f1_hungarian_points(A_pts, B_pts, R=12):
    if len(A_pts)==0 or len(B_pts)==0: return 0.0
    A=np.array([(y,x) for y,x,_ in A_pts]); B=np.array([(y,x) for y,x,_ in B_pts])
    D=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(2))
    row,col=hungarian(D); hits=(D[row,col]<=R).sum()
    prec=hits/len(B); rec=hits/len(A); return 2*prec*rec/(prec+rec+1e-12)

# ---- Morphology helpers ----
def binarize_by_quantile(arr,q=0.90):
    thr=np.quantile(arr,q); return (arr>=thr),thr

def perimeter_length(binary):
    er=binary_erosion(binary); edge=binary ^ er; return float(edge.sum())

def component_keep_mask_and_count(binary, min_size=100):
    lab,n=label(binary); keep=np.zeros_like(binary,bool); count=0
    if n>0:
        sls=find_objects(lab)
        for i,sl in enumerate(sls, start=1):
            if sl is None: continue
            mask=(lab[sl]==i); sz=int(mask.sum())
            if sz>=min_size:
                keep[sl]|=mask; count+=1
    return keep, count

# ---------------- Main per-dataset pipeline ----------------
def run_one(name, url, CFG):
    if not url.strip():
        print(f"[SKIP] {name}: URL not provided.\n"); return None

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

    # θ(U)
    U=embedding_field(CFG["EMBEDDING"], xx, yy, kappa)
    theta=theta_exact_from_U(U, CFG["T_LO"], CFG["T_HI"], CFG["THETA_GRID_N"], mp_dps=50)
    theta_bp=bandpass_apply(theta * W2D, BP_MASK)

    # κ_pred
    kappa_w = kappa * W2D
    kpred0  = -laplacian_2d(theta_bp)
    if CFG["SIGMA_SMOOTH_PRED"]>0: kpred0=gaussian_filter(kpred0, CFG["SIGMA_SMOOTH_PRED"])
    kappa_bp = bandpass_apply(kappa_w, BP_MASK)
    kpred_bp = bandpass_apply(kpred0,  BP_MASK)
    kpred_bp = gaussian_filter(kpred_bp, 1.0)
    kappa_bp = zscore(kappa_bp); kpred_bp=zscore(kpred_bp)

    # Controls (same aperture)
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

    # Metrics: γ² + JK; best-shift; ridge-F1 with scale scan
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

    # Ridge-F1: best-shift + isotropic scale scan
    dy, dx = M_best["shift"]; kd2 = M_best["kd2"]
    def _scale_image(img, s):
        from scipy.ndimage import zoom
        h, w = img.shape
        z = zoom(img, s, order=1)
        hh, ww = z.shape
        out = np.zeros_like(img)
        y0 = max(0, (h - hh)//2); x0 = max(0, (w - ww)//2)
        y1 = min(h, y0 + hh);     x1 = min(w, x0 + ww)
        zy0 = max(0, - (h - hh)//2); zx0 = max(0, - (w - ww)//2)
        out[y0:y1, x0:x1] = z[zy0:zy0+(y1-y0), zx0:zx0+(x1-x0)]
        return out
    best_F1 = 0.0
    for s in PRIMARY["AFFINE_SCALE_SCAN"]:
        kp2_s = _scale_image(M_best["kp2"], s)
        kp2_aligned = np.roll(np.roll(kp2_s, dy, axis=0), dx, axis=1)
        pts_d = ridge_keypoints(kd2, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
        pts_p = ridge_keypoints(kp2_aligned, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
        F1_s  = f1_hungarian_points(pts_d, pts_p, R=PRIMARY["F1_MATCH_RADIUS"])
        best_F1 = max(best_F1, F1_s)
    F1_best = best_F1

    # Control F1s (no alignment)
    def ridge_F1_noalign(kd2, kp2):
        A = ridge_keypoints(kd2, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
        B = ridge_keypoints(kp2, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
        return f1_hungarian_points(A,B, R=PRIMARY["F1_MATCH_RADIUS"])
    F1_sh = ridge_F1_noalign(M_shuf["kd2"], M_shuf["kp2"])
    F1_ro = ridge_F1_noalign(M_roll["kd2"], M_roll["kp2"])
    F1_rt = ridge_F1_noalign(M_rot["kd2"],  M_rot["kp2"])

    # Morphology (same band)
    morphA_raw,thrA=binarize_by_quantile(kappa_bp,q=PRIMARY["MORPH_Q"])
    morphB_raw,thrB=binarize_by_quantile(kpred_bp,q=PRIMARY["MORPH_Q"])
    morphA_raw=binary_opening(morphA_raw, iterations=1); morphB_raw=binary_opening(morphB_raw, iterations=1)
    bA, compA = component_keep_mask_and_count(morphA_raw, PRIMARY["MORPH_MIN_SIZE"])
    bB, compB = component_keep_mask_and_count(morphB_raw, PRIMARY["MORPH_MIN_SIZE"])
    perA=perimeter_length(bA); perB=perimeter_length(bB); areaA=float(bA.sum()); areaB=float(bB.sum())

    # Power spectra (sanity)
    def fit_slope(k,P, lo=0.10,hi=0.25):
        n=len(k); i0=int(n*lo); i1=int(n*hi); kk=k[i0:i1].astype(float); pp=P[i0:i1].astype(float)
        return float(np.polyfit(np.log(kk+1e-9), np.log(pp+1e-12), 1)[0])
    kt,Pt=isotropic_ps(zscore(theta_bp)); kp,Pp=isotropic_ps(kpred_bp)
    s_theta=fit_slope(kt,Pt,*PRIMARY["SLOPE_FIT_FRAC"]); s_pred=fit_slope(kp,Pp,*PRIMARY["SLOPE_FIT_FRAC"])

    # Plots
    plt.figure(figsize=(5.6,4.6)); plt.imshow(kappa_bp, origin='lower'); plt.colorbar(); plt.title(f"{name}: κ (bandpassed,z)"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(5.6,4.6)); plt.imshow(kpred_bp, origin='lower'); plt.colorbar(); plt.title("κ_pred ∝ −∇²θ(U) (bandpassed,z)"); plt.tight_layout(); plt.show()
    kd,Pd=isotropic_ps(kappa_bp); kp2,Pp2=isotropic_ps(kpred_bp); kx,Px=isotropic_cross_ps(kappa_bp,kpred_bp)
    c=min(len(Pd),len(Pp2),len(Px)); Pd,Pp2,Px=Pd[:c],Pp2[:c],Px[:c]; g2_curve=(Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp2,1e-20))
    kd,Pd=isotropic_ps(kappa_bp); kp3,Pp3=isotropic_ps(kpred_sh); kx,Px=isotropic_cross_ps(kappa_bp,kpred_sh)
    c2=min(len(Pd),len(Pp3),len(Px)); Pd,Pp3,Px=Pd[:c2],Pp3[:c2],Px[:c2]; g2_sh_curve=(Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp3,1e-20))
    plt.figure(figsize=(5.6,4.6)); plt.semilogy(np.arange(len(g2_curve)), g2_curve, label="γ²(data×pred)"); plt.semilogy(np.arange(len(g2_sh_curve)), g2_sh_curve, label="γ²(shuffled)"); plt.legend(); plt.title("Coherence vs k"); plt.tight_layout(); plt.show()

    # PASS flags
    def pass_flag(real,ctrl_max,thr): return (real>=thr) and (real>max(ctrl_max,1e-9)*1.5)
    ok_gamma = pass_flag(M_best["gamma2"], max(M_shuf["gamma2"],M_roll["gamma2"],M_rot["gamma2"]), THRESH["gamma2_min"])
    ctrl_F1_max = max(F1_sh,F1_ro,F1_rt)
    ok_f1   = pass_flag(F1_best, ctrl_F1_max, THRESH["f1_min"])
    bonus_r = (M_best["r_best"] >= THRESH["r_bonus"])

    # Print summary
    print("\n=== REPLICATION (locked params) ===")
    print(f"Dataset: {name}")
    print(f"Embed={PRIMARY['EMBEDDING']}  band={PRIMARY['BANDPASS']}  θgrid={PRIMARY['THETA_GRID_N']}  "
          f"smooth(data,pred)=({PRIMARY['SIGMA_SMOOTH_DATA']},{PRIMARY['SIGMA_SMOOTH_PRED']})")
    print(f"[BEST]     r={M_best['r']:+.3f}, r_best={M_best['r_best']:+.3f}@{M_best['shift']}, "
          f"γ̄²={M_best['gamma2']:.3f} [JK {M_best['gamma2_jk'][0]:.3f} CI {M_best['gamma2_jk'][1]:.3f},{M_best['gamma2_jk'][2]:.3f}]")
    print(f"[BEST-aligned] Ridge F1(one-to-one) = {F1_best:.3f}")
    print(f"[CTRL]  γ̄²: shuffle={M_shuf['gamma2']:.3f} roll={M_roll['gamma2']:.3f} rot90={M_rot['gamma2']:.3f}")
    print(f"[CTRL]  F1 : shuffle={F1_sh:.3f} roll={F1_ro:.3f} rot90={F1_rt:.3f}")
    print(f"[Sanity] slope(P_theta)={s_theta:.2f}, slope(P_pred)={s_pred:.2f}, diff={s_pred-s_theta:.2f}")
    print(f"[Morph @q={PRIMARY['MORPH_Q']:.2f}, min_size={PRIMARY['MORPH_MIN_SIZE']}] "
          f"comps: data={compA} pred={compB}; "
          f"area(px): data={areaA:.0f} pred={areaB:.0f}; "
          f"perim(px): data={perA:.0f} pred={perB:.0f}")
    if PRIMARY['MPP'] is not None:
        print(f"          area(Mpc^2): data={areaA*PRIMARY['MPP']**2:.2f} pred={areaB*PRIMARY['MPP']**2:.2f}; "
              f"perim(Mpc): data={perA*PRIMARY['MPP']:.2f} pred={perB*PRIMARY['MPP']:.2f}")
    print(f"\nPASS: γ̄²≥{THRESH['gamma2_min']}? {ok_gamma} | Ridge-F1≥{THRESH['f1_min']} & ≥1.5×ctrl? {ok_f1} | bonus r_best≥{THRESH['r_bonus']}? {bonus_r}")
    print("Done.\n")

    return dict(metrics=(M_best,M_shuf,M_roll,M_rot), F1_best=F1_best, F1_ctrl=(F1_sh,F1_ro,F1_rt))

# ---------------- Run all datasets (locked params) ----------------
for (name, url) in DATASETS:
    _ = run_one(name, url, PRIMARY)

print("提示：仅改 PRIMARY['EMBEDDING'] 为 'elliptic-from-moments' 或 'dual-from-twopeaks'，其余参数不变，再次运行即可完成“不改参跨天区 + 改几何假设”的鲁棒性检验。")

