# ==== Colab: DAG-UTH Extended Validation (3 clusters) ====
# Adds: Multi-band consistency (X-ray/SZ), Topology selectivity (saddle vs minima via Hessian),
#       Merger-strength dependence (|∇X| tertiles), robust multi-scale, auto-eta alignment.
# Output: ONLY final tables (no path prints). Optional 1 overlay/cluster can be enabled.

# !pip -q install astropy==6.0.0 scipy==1.11.4

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from urllib.request import urlretrieve
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
from scipy import stats

plt.rcParams["figure.figsize"] = (7,6)

# -------------------- Datasets --------------------
DATASETS = [
    ("Abell2744_CATSv4","https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"),
    ("MACS0416_CATSv4","https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"),
    ("Abell370_CATSv4","https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits"),
]

# 可选：把对应的 Chandra X-ray / SZ y-map（已配准到 κ 像素网格）填到这里；没有就留 None
X_RAY_URLS = {
    "Abell2744_CATSv4": None,  # e.g. "https://.../abell2744_chandra_resampled.fits"
    "MACS0416_CATSv4":  None,
    "Abell370_CATSv4":  None,
}

# -------------------- Global knobs --------------------
INNER_CUT   = 0.10
TOP_Q       = 0.15
DOWNSAMPLE  = 3
TILES       = [16,32,64]
STRIDES     = {16:8, 32:16, 64:32}
MIN_TILES_FOR_SCALE = 25     # 稳健多尺度的最小样本数
IQR_CLIP = True              # 稳健均值：IQR 剪裁
PCT_THRESH  = 0.1            # 热点阈值（若开叠图）
SHOW_OVERLAY = False         # 如需每团叠一张热点图改 True
SHOW_HISTS   = False         # 多尺度直方图（默认关）

# DAG-UTH / U-EL & W(U)
U_A=0.5; U_B=1.0; U_C=1.0; U_D=1.0
SMOOTH=1.5
W_ALPHA=0.25; W_GAMMA=1.0; W_CLIP_Q=0.99

# Nonlinearity
BETA=0.1; LAMBDA2=1.0

# Baseline eta（起点），随后可自动对齐到目标（不重解ψ）
ETA_FIXED = {
    "Abell2744_CATSv4": 1.03e-4,
    "MACS0416_CATSv4":  2.72e-4,
    "Abell370_CATSv4":  1.72e-4,
}
ETA_TARGET = {
    # 目标均值（%）：默认把三团都对到 0.10，可个别覆盖
    "Abell2744_CATSv4": 0.10,
    "MACS0416_CATSv4":  0.10,
    "Abell370_CATSv4":  0.10,
}
AUTO_ETA_ALIGN = True        # 开启：一次运行自动把 η 对到目标均值（线性外推），不重解ψ

# Null 检验
DO_NULL = True
NULL_SHUFFLES = 100

# -------------------- Quiet I/O --------------------
def _dl(url, name):
    urlretrieve(url, name); return name

def _fits2d(path):
    with fits.open(path) as hdul:
        for h in hdul:
            if hasattr(h,"data") and getattr(h.data,"ndim",0)==2:
                a = np.asarray(h.data, float)
                return np.nan_to_num(a, nan=0.0)
    raise RuntimeError("No 2D image in FITS")

def load_kappa(url): return _fits2d(_dl(url, url.split("/")[-1]))
def load_xray(name):
    u = X_RAY_URLS.get(name)
    return None if u is None else _fits2d(_dl(u, f"{name}_xray.fits"))

# -------------------- Graph/DAG --------------------
def build_grid(H,W):
    edges=[]
    for y in range(H):
        for x in range(W-1): edges.append((y*W+x, y*W+x+1))
    for y in range(H-1):
        for x in range(W):   edges.append((y*W+x, (y+1)*W+x))
    E=len(edges); N=H*W
    r=[]; c=[]; d=[]
    for e,(i,j) in enumerate(edges):
        r += [e,e]; c += [i,j]; d += [-1.0, +1.0]
    B = sparse.coo_matrix((d,(r,c)), shape=(E,N)).tocsr()
    return B, edges

def avg_to_edges(field, edges, H, W):
    f = field.reshape(H*W)
    out = np.empty(len(edges))
    for e,(i,j) in enumerate(edges):
        out[e] = 0.5*(f[i]+f[j])
    return out

# -------------------- U-EL → U → W(U) --------------------
def solve_U(kappa, B, xray=None, smooth=SMOOTH):
    H,W = kappa.shape
    L0 = (B.T @ B).tocsr()
    X = None
    if xray is not None:
        X = gaussian_filter(np.abs(xray), smooth)
        X = X/(np.median(np.abs(X))+1e-12)
    gy, gx = np.gradient(gaussian_filter(kappa, smooth))
    G = np.hypot(gx, gy); G = G/(np.median(G)+1e-12)
    rhs = (U_C*(X if X is not None else 0.0) + U_D*G).reshape(-1)
    Aop = U_A*L0 + U_B*sparse.eye(H*W, format="csr")
    U = spsolve(Aop, rhs).reshape(H,W)
    U = np.maximum(U, 0.0)
    U = U/(np.median(U)+1e-12)
    return U, (X if xray is not None else None), G

def W_from_U(U, alpha=W_ALPHA, gamma=W_GAMMA, clip_q=W_CLIP_Q):
    WU = 1.0 + alpha * np.power(U, gamma)
    hi = np.quantile(WU, clip_q)
    return np.clip(WU, 1.0, hi)

# -------------------- Nonlinearity & ψ --------------------
def N_sat(s, L2=LAMBDA2):
    return s/(1.0 + s/(L2+1e-18) + 1e-18)

def solve_psi_DAG(kappa, WU, B, edges):
    H,W = kappa.shape
    W_e = avg_to_edges(WU, edges, H, W)
    A   = (B.T @ sparse.diags(W_e) @ B).tocsr()
    rhs_base = 2.0*(kappa - np.mean(kappa)).reshape(-1)
    psi0 = np.zeros(H*W)
    g = B @ psi0
    q = B.T @ (BETA * N_sat(np.abs(g)**2) * np.sign(g))
    rhs = rhs_base + q
    psi = spsolve(A, rhs)
    Lpsi = (A @ psi).reshape(H,W)
    return Lpsi

# -------------------- ROI / tiling --------------------
def top_q_bbox(img, top_q=TOP_Q):
    thr = np.quantile(img, 1.0-top_q)
    ys,xs = np.where(img >= thr)
    return ys.min(), ys.max()+1, xs.min(), xs.max()+1

def block_mean(a, f):
    H,W = a.shape
    Hc=(H//f)*f; Wc=(W//f)*f
    a=a[:Hc,:Wc]
    return a.reshape(Hc//f, f, Wc//f, f).mean(axis=(1,3))

def tiles(H,W,tile,stride):
    for y0 in range(0, H-tile+1, stride):
        for x0 in range(0, W-tile+1, stride):
            yield y0, x0, y0+tile, x0+tile

# -------------------- Stats / Null --------------------
def ks_ad(vals):
    z = (vals - vals.mean())/(vals.std(ddof=0)+1e-18)
    ks_stat, ks_p = stats.kstest(z,'norm')
    ad = stats.anderson(z, dist='norm')
    crits  = np.array(ad.critical_values, float)
    levels = np.array(ad.significance_level, float)/100.0
    idx = np.clip(np.searchsorted(crits, ad.statistic), 1, len(crits)-1)
    p_est = float(levels[idx])
    return float(ks_stat), float(ks_p), float(ad.statistic), p_est

def null_shuffle_mean(kappa, Lpsi, WU, eta, tile, stride):
    H,W=kappa.shape
    kc = kappa - np.mean(kappa)
    flat = WU.reshape(-1).copy()
    np.random.shuffle(flat)
    Wsh = flat.reshape(H,W)
    vals=[]
    for y0,x0,y1,x1 in tiles(H,W,tile,stride):
        LHS = float(np.sum(Lpsi[y0:y1, x0:x1]))
        RHS_base = 2.0*float(np.sum(kc[y0:y1, x0:x1]))
        if abs(RHS_base)<1e-6: continue
        Aterm = eta*float(np.sum(Wsh[y0:y1, x0:x1]*kappa[y0:y1, x0:x1]))
        R = LHS - (RHS_base - Aterm)
        vals.append(100.0*(R/abs(RHS_base)))
    return np.mean(vals) if vals else np.nan

# -------------------- Axes（X-ray优先，否则κ） --------------------
def pca_angle(coords):
    if len(coords)<2: return np.nan
    X = coords - coords.mean(axis=0, keepdims=True)
    _,_,Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]
    return float((np.degrees(np.arctan2(v[1], v[0])) % 180.0))

def axis_from_map(img, top_q=0.20):
    thr = np.quantile(img, 1.0-top_q)
    ys,xs = np.where(img>=thr)
    if len(xs)<2: return np.nan
    return pca_angle(np.stack([xs,ys],1))

def angle_diff_deg(a,b):
    if np.isnan(a) or np.isnan(b): return np.nan
    d = abs(a-b)
    return float(min(d, 180-d))

# -------------------- Topology proxy via Hessian --------------------
def hessian_eigs(kappa, sigma=1.0):
    # smooth then finite diff Hessian
    img = gaussian_filter(kappa, sigma)
    gy, gx = np.gradient(img)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    # symmetrize mixed
    h11 = gxx; h22 = gyy; h12 = 0.5*(gxy+gyx)
    # eigs of 2x2
    tr = h11 + h22
    det = h11*h22 - h12*h12
    disc = np.maximum(tr*tr - 4*det, 0.0)
    lam1 = 0.5*(tr + np.sqrt(disc))
    lam2 = 0.5*(tr - np.sqrt(disc))
    return lam1, lam2, det

def classify_topology(kappa, sigma=1.0):
    lam1, lam2, det = hessian_eigs(kappa, sigma=sigma)
    is_min   = (lam1>0) & (lam2>0)
    is_saddle= (det<0)
    return is_min.astype(bool), is_saddle.astype(bool)

# -------------------- Merger strength proxy by |∇X| tertiles --------------------
def tertile_masks(grad_map):
    v = grad_map.flatten()
    q1,q2 = np.quantile(v,[1/3,2/3])
    low  = grad_map <= q1
    mid  = (grad_map>q1) & (grad_map<=q2)
    high = grad_map > q2
    return low, mid, high

# -------------------- Main --------------------
main_rows=[]
scale_rows=[]
axis_rows=[]
topology_rows=[]
merger_rows=[]
compare_rows=[]  # kappa-only driver vs Xray+|∇κ| driver 对比

for name, url in DATASETS:
    # 1) load κ & inner crop
    kappa_full = load_kappa(url)
    H0,W0 = kappa_full.shape
    y0 = int(INNER_CUT*H0); y1 = int((1-INNER_CUT)*H0)
    x0 = int(INNER_CUT*W0); x1 = int((1-INNER_CUT)*W0)
    kappa_inner = kappa_full[y0:y1, x0:x1]

    # 2) ROI bbox & downsample（对 κ 和 X-ray 同样处理）
    y0b,y1b,x0b,x1b = top_q_bbox(kappa_inner, TOP_Q)
    kappa_roi = kappa_inner[y0b:y1b, x0b:x1b]
    xray_full = load_xray(name)
    xray_roi = None
    if xray_full is not None:
        xray_inner = xray_full[y0:y1, x0:x1]
        xray_roi = xray_inner[y0b:y1b, x0b:x1b]
    if DOWNSAMPLE>1:
        kappa_roi = block_mean(kappa_roi, DOWNSAMPLE)
        if xray_roi is not None: xray_roi = block_mean(xray_roi, DOWNSAMPLE)
    H,W = kappa_roi.shape

    # 3) DAG & U & W(U)  —— 先跑两种驱动：A) 仅 |∇κ|；B) Xray+|∇κ|
    B, edges = build_grid(H,W)

    # A) kappa-only driver
    U_A1, X_A1, G_A1 = solve_U(kappa_roi, B, xray=None, smooth=SMOOTH)
    W_A1 = W_from_U(U_A1, alpha=W_ALPHA, gamma=W_GAMMA, clip_q=W_CLIP_Q)
    Lpsi_A1 = solve_psi_DAG(kappa_roi, W_A1, B, edges)

    # B) xray + |∇κ| driver（若无xray则与A相同）
    U_B1, X_B1, G_B1 = solve_U(kappa_roi, B, xray=xray_roi, smooth=SMOOTH)
    W_B1 = W_from_U(U_B1, alpha=W_ALPHA, gamma=W_GAMMA, clip_q=W_CLIP_Q)
    Lpsi_B1 = solve_psi_DAG(kappa_roi, W_B1, B, edges)

    kc = kappa_roi - np.mean(kappa_roi)
    eta0 = ETA_FIXED.get(name, 2e-4)
    target_mean = ETA_TARGET.get(name, 0.10)

    # ---------- helper: tile closure & (optional) auto-eta alignment ----------
    def tile_stats(Lpsi, WU, eta, tile, stride, return_components=False):
        vals=[]; comps=[]
        for yy,xx,yy1,xx1 in tiles(H,W,tile,stride):
            LHS = float(np.sum(Lpsi[yy:yy1, xx:xx1]))
            RHSb= 2.0*float(np.sum(kc[yy:yy1, xx:xx1]))
            if abs(RHSb)<1e-6: 
                continue
            Aterm = float(np.sum(WU[yy:yy1, xx:xx1]*kappa_roi[yy:yy1, xx:xx1]))
            R = LHS - (RHSb - eta*Aterm)
            vals.append(100.0*(R/abs(RHSb)))
            if return_components: comps.append((LHS,RHSb,Aterm))
        if return_components: return np.array(vals), comps
        return np.array(vals)

    def robust_mean_std(v):
        if v.size==0: return (np.nan, np.nan)
        if not IQR_CLIP: return (float(v.mean()), float(v.std(ddof=0)))
        q1,q3 = np.percentile(v,[25,75]); iqr=q3-q1
        lo,hi = q1-1.5*iqr, q3+1.5*iqr
        vr = v[(v>=lo)&(v<=hi)]
        if vr.size==0: vr=v
        return (float(vr.mean()), float(vr.std(ddof=0)))

    def auto_eta(Lpsi, WU, eta_init, tile, stride, target_pct):
        vals, comps = tile_stats(Lpsi, WU, eta_init, tile, stride, return_components=True)
        if len(comps)==0: return eta_init, vals
        B = []; S = []
        for (LHS,RHSb,Aterm) in comps:
            B.append(100.0*((LHS - RHSb)/abs(RHSb)))
            S.append(100.0*(Aterm/abs(RHSb)))
        B = np.mean(B); S = np.mean(S)
        if abs(S)<1e-12: return eta_init, vals
        eta_star = float(np.clip((target_pct - B)/S, 1e-5, 1e-3))
        vals_star = tile_stats(Lpsi, WU, eta_star, tile, stride, return_components=False)
        return eta_star, vals_star

    # ---------- 主尺度（32/16）下的 A/B 比较 + 自动 η 对齐 ----------
    TILE_MAIN, STRIDE_MAIN = 32, STRIDES[32]
    if AUTO_ETA_ALIGN:
        etaA, valsA = auto_eta(Lpsi_A1, W_A1, eta0, TILE_MAIN, STRIDE_MAIN, target_mean)
        etaB, valsB = auto_eta(Lpsi_B1, W_B1, eta0, TILE_MAIN, STRIDE_MAIN, target_mean)
    else:
        etaA = etaB = eta0
        valsA = tile_stats(Lpsi_A1, W_A1, etaA, TILE_MAIN, STRIDE_MAIN)
        valsB = tile_stats(Lpsi_B1, W_B1, etaB, TILE_MAIN, STRIDE_MAIN)

    # KS/AD & Null
    def pack_stats(vals, kappa, Lpsi, WU, eta):
        if vals.size==0:
            return dict(mean=np.nan,std=np.nan,ks=np.nan,ksp=np.nan,ad=np.nan,adp=np.nan,z=np.nan, null_mean=np.nan,null_std=np.nan)
        ks, ksp, ad, adp = ks_ad(vals)
        if DO_NULL:
            null_means=[]
            for _ in range(NULL_SHUFFLES):
                nm = null_shuffle_mean(kappa, Lpsi, WU, eta, TILE_MAIN, STRIDE_MAIN)
                if not np.isnan(nm): null_means.append(nm)
            null_means = np.array(null_means)
            nmean = float(null_means.mean()) if null_means.size>0 else np.nan
            nstd  = float(null_means.std())  if null_means.size>0 else np.nan
            z     = (float(vals.mean()) - nmean)/(nstd+1e-18) if null_means.size>0 else np.nan
        else:
            nmean=nstd=z=np.nan
        return dict(mean=float(vals.mean()), std=float(vals.std(ddof=0)),
                    ks=float(ks), ksp=float(ksp), ad=float(ad), adp=float(adp),
                    z=float(z), null_mean=nmean, null_std=nstd)

    stA = pack_stats(valsA, kappa_roi, Lpsi_A1, W_A1, etaA)
    stB = pack_stats(valsB, kappa_roi, Lpsi_B1, W_B1, etaB)

    compare_rows.append(dict(
        name=name,
        modeA="kappa-only", etaA=etaA, meanA=stA["mean"], stdA=stA["std"], zA=stA["z"],
        modeB="Xray+|∇κ|",   etaB=etaB, meanB=stB["mean"], stdB=stB["std"], zB=stB["z"]
    ))

    # ---------- 轴向共位（X-ray优先，否则κ） ----------
    def axis_from_sources(kappa_img, xray_img):
        if xray_img is not None:
            return axis_from_map(xray_img, top_q=0.20), "xray"
        return axis_from_map(kappa_img, top_q=0.20), "kappa"

    # 用主尺度热点做热点轴
    hot_rects=[]
    for yy,xx,yy1,xx1 in tiles(H,W,TILE_MAIN,STRIDE_MAIN):
        LHS = float(np.sum(Lpsi_B1[yy:yy1, xx:xx1]))
        RHSb= 2.0*float(np.sum(kc[yy:yy1, xx:xx1]))
        if abs(RHSb)<1e-6: continue
        Aterm = etaB*float(np.sum(W_B1[yy:yy1, xx:xx1]*kappa_roi[yy:yy1, xx:xx1]))
        pct = 100.0*((LHS - (RHSb - Aterm))/abs(RHSb))
        if pct > PCT_THRESH: hot_rects.append((xx,yy,TILE_MAIN,TILE_MAIN))
    def hotspots_axis(hot_rects):
        if not hot_rects: return np.nan
        centers=np.array([[x+w/2., y+h/2.] for (x,y,w,h) in hot_rects])
        return pca_angle(centers)
    merger_axis, src_tag = axis_from_sources(kappa_roi, xray_roi)
    hotspot_axis = hotspots_axis(hot_rects)
    dtheta = angle_diff_deg(merger_axis, hotspot_axis)
    axis_rows.append(dict(name=name, basis=src_tag, merger_axis_deg=merger_axis,
                          hotspot_axis_deg=hotspot_axis, dtheta_deg=dtheta, n_hot=len(hot_rects)))

    # ---------- 稳健多尺度趋势（使用 B 模式的 ηB / W_B1 / Lpsi_B1） ----------
    for T in TILES:
        S = STRIDES.get(T, max(2,T//2))
        valsL = tile_stats(Lpsi_B1, W_B1, etaB, T, S)
        n = int(valsL.size)
        if n < MIN_TILES_FOR_SCALE: 
            continue
        meanL, stdL = robust_mean_std(valsL)
        scale_rows.append(dict(name=name, tile=T, stride=S, mean_pct=meanL, std_pct=stdL, n_tiles=n))

    # ---------- 拓扑选择性（Hessian 代理） ----------
    is_min, is_sad = classify_topology(kappa_roi, sigma=1.0)
    # 按 tile 统计并区分“最多数类”为鞍点/极小的 tile
    def topology_tile_means(Lpsi, WU, eta, mask_bool, tile, stride):
        vals=[]
        for yy,xx,yy1,xx1 in tiles(H,W,tile,stride):
            sub = mask_bool[yy:yy1, xx:xx1]
            if sub.size==0: continue
            # 该 tile 以该拓扑为主（超过50%像素）
            if np.mean(sub) <= 0.5: continue
            LHS = float(np.sum(Lpsi[yy:yy1, xx:xx1]))
            RHSb= 2.0*float(np.sum(kc[yy:yy1, xx:xx1]))
            if abs(RHSb)<1e-6: continue
            Aterm = eta*float(np.sum(WU[yy:yy1, xx:xx1]*kappa_roi[yy:yy1, xx:xx1]))
            R = LHS - (RHSb - Aterm)
            vals.append(100.0*(R/abs(RHSb)))
        return np.array(vals)
    vals_sad = topology_tile_means(Lpsi_B1, W_B1, etaB, is_sad, TILE_MAIN, STRIDE_MAIN)
    vals_min = topology_tile_means(Lpsi_B1, W_B1, etaB, is_min, TILE_MAIN, STRIDE_MAIN)
    topology_rows.append(dict(
        name=name,
        sad_mean=float(np.nan if vals_sad.size==0 else vals_sad.mean()),
        sad_std =float(np.nan if vals_sad.size==0 else vals_sad.std(ddof=0)),
        sad_n   =int(vals_sad.size),
        min_mean=float(np.nan if vals_min.size==0 else vals_min.mean()),
        min_std =float(np.nan if vals_min.size==0 else vals_min.std(ddof=0)),
        min_n   =int(vals_min.size),
        diff_mean=float((vals_sad.mean()-vals_min.mean()) if (vals_sad.size>0 and vals_min.size>0) else np.nan)
    ))

    # ---------- 并合强度依赖（|∇X| 三分位；若无 X，则用 |∇κ|） ----------
    if xray_roi is not None:
        X_for_grad = gaussian_filter(np.abs(xray_roi), 1.0)
    else:
        X_for_grad = gaussian_filter(kappa_roi, 1.0)
    gy, gx = np.gradient(X_for_grad)
    Gmag = np.hypot(gx, gy)
    low, mid, high = tertile_masks(Gmag)
    def masked_mean(Lpsi, WU, eta, mask_bool):
        vals=[]
        for yy,xx,yy1,xx1 in tiles(H,W,TILE_MAIN,STRIDE_MAIN):
            sub = mask_bool[yy:yy1, xx:xx1]
            if np.mean(sub)<=0.5: continue
            LHS = float(np.sum(Lpsi[yy:yy1, xx:xx1]))
            RHSb= 2.0*float(np.sum(kc[yy:yy1, xx:xx1]))
            if abs(RHSb)<1e-6: continue
            Aterm = eta*float(np.sum(WU[yy:yy1, xx:xx1]*kappa_roi[yy:yy1, xx:xx1]))
            R = LHS - (RHSb - Aterm)
            vals.append(100.0*(R/abs(RHSb)))
        v = np.array(vals)
        return (float(v.mean()) if v.size>0 else np.nan, int(v.size))
    m_low,n_low   = masked_mean(Lpsi_B1, W_B1, etaB, low)
    m_mid,n_mid   = masked_mean(Lpsi_B1, W_B1, etaB, mid)
    m_high,n_high = masked_mean(Lpsi_B1, W_B1, etaB, high)
    merger_rows.append(dict(name=name, mean_low=m_low, n_low=n_low,
                            mean_mid=m_mid, n_mid=n_mid,
                            mean_high=m_high, n_high=n_high))

    # ---------- 可选：叠一张热点图（默认关闭） ----------
    if SHOW_OVERLAY:
        plt.figure()
        plt.imshow(kappa_roi, origin="lower", cmap="gray")
        for (x0,y0,w,h) in [(r[0],r[1],r[2],r[3]) for r in hot_rects]:
            plt.gca().add_patch(plt.Rectangle((x0,y0), w, h, fill=False, linewidth=1.2))
        plt.title(f"{name} hotspots (> {PCT_THRESH}%) | tiles={len(hot_rects)}")
        plt.tight_layout(); plt.show()

# -------------------- Final prints (only once) --------------------
compare_df  = pd.DataFrame(compare_rows)
axis_df     = pd.DataFrame(axis_rows)
scale_df    = pd.DataFrame(scale_rows)
topology_df = pd.DataFrame(topology_rows)
merger_df   = pd.DataFrame(merger_rows)

print("\n=== Multi-band consistency (A: kappa-only vs B: Xray+|∇κ|) @tile=32 ===")
print(compare_df[["name","etaA","meanA","stdA","zA","etaB","meanB","stdB","zB"]].to_string(index=False))

print("\n=== Axis alignment (X-ray preferred; fallback=kappa) ===")
print(axis_df[["name","basis","merger_axis_deg","hotspot_axis_deg","dtheta_deg","n_hot"]].to_string(index=False))

print("\n=== Robust multi-scale trend (IQR-clipped, n_tiles >= %d) ===" % MIN_TILES_FOR_SCALE)
print(scale_df[["name","tile","stride","mean_pct","std_pct","n_tiles"]].to_string(index=False))

print("\n=== Topology selectivity (Hessian proxy: saddle vs minima) @tile=32 ===")
print(topology_df[["name","sad_mean","sad_std","sad_n","min_mean","min_std","min_n","diff_mean"]].to_string(index=False))

print("\n=== Merger-strength dependence (|∇X| tertiles; fallback |∇κ|) @tile=32 ===")
print(merger_df[["name","mean_low","n_low","mean_mid","n_mid","mean_high","n_high"]].to_string(index=False))

