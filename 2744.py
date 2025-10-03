# ==========================================================
# ζ–U 绑定 (Eq.5) 的跨天区“不改参复现” + 脊线骨架 F1（鲁棒峰对齐）
# 指标：mean γ²（含 JK 95%CI）、best-shift 后的 ridge-F1、一致性形态（同分位+去小斑块）
# 三对照：phase-shuffle(θ)、U-roll、U-rot90（同口径）
# 仅 plt/print 输出；不写文件
# ==========================================================

# ---------------- 锁定（PRIMARY）参数：跨天区完全不改 ----------------
PRIMARY = {
    "EMBEDDING": "radial-from-peak",   # 固定 U 嵌入
    "T_LO": 80.0, "T_HI": 6000.0,      # 真实 ζ 相位采样区间
    "THETA_GRID_N": 1000,              # ζ 相位预计算网格
    "TARGET_N": 512,                   # 统一重采样尺寸
    "SIGMA_SMOOTH_DATA": 1.0,          # κ data 轻平滑
    "SIGMA_SMOOTH_PRED": 2.0,          # κ_pred 初步轻平滑
    "TUKEY_ALPHA": 0.20,               # 边窗
    "BANDPASS": (0.10, 0.25),          # 锁定带宽（稳 γ² & F1）
    "SLOPE_FIT_FRAC": (0.10, 0.25),    # 幂谱斜率拟合区间
    "BEST_SHIFT_MAX": 6,               # best-shift 搜索窗口
    # ---- 脊线骨架设置（用于一对一 F1）----
    "RIDGE_SIGMAS": (1.5, 3.0, 6.0),   # LoG 多尺度
    "RIDGE_Q": 0.97,                   # 脊线二值化分位阈值
    "RIDGE_MIN_SEP": 14,               # 采样点最小间距（像素）
    "F1_MATCH_RADIUS": 12,             # F1 匹配半径（像素）
    # ---- 形态统计（同分位 + 去小斑块）----
    "MORPH_Q": 0.90,
    "MORPH_MIN_SIZE": 100,             # 删除 tiny 斑块（px）
    "MPP": None                        # 可选物理尺（Mpc/px），如 A2744≈0.102
}
THRESH = {"gamma2_min": 0.05, "f1_min": 0.10, "r_bonus": 0.03}

# === Fill replication targets (locked params; DO NOT change anything else) ===
A2744_URL   = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"

# NEW: MACS J0416.1−2403 (CATS v4, kappa FITS)
MACS0416_URL = "https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits"

# NEW: Abell 370 (CATS v4, kappa FITS)
A370_URL     = "https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits"

DATASETS = [
    ("Abell2744_CATSv4", A2744_URL),
    ("MACS0416_CATSv4",  MACS0416_URL),
    ("Abell370_CATSv4",  A370_URL),
]


# ---------------- 安装/导入 ----------------
import sys, subprocess, warnings
warnings.filterwarnings("ignore")
def _pip(x): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + x)
try:
    import numpy as np, matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import (gaussian_filter, maximum_filter, label, find_objects,
                               zoom, gaussian_laplace, binary_erosion, binary_opening)
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
                               zoom, gaussian_laplace, binary_erosion, binary_opening)
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

# ---------------- 工具函数 ----------------
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

# ---- ζ 相位 θ(U)（缓存）----
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

# ---- U 嵌入 ----
def embedding_field(name, xx, yy, kappa_like):
    ny,nx=kappa_like.shape
    if name=="radial-from-peak":
        cy,cx=np.unravel_index(np.argmax(kappa_like),kappa_like.shape)
        Xc=(xx-cx)/(nx/2); Yc=(yy-cy)/(ny/2); return np.sqrt(Xc**2+Yc**2)+1e-6
    elif name=="radial-center":
        X0=(xx-nx/2)/(nx/2); Y0=(yy-ny/2)/(ny/2); return np.sqrt(X0**2+Y0**2)+1e-6
    else: raise ValueError("Use 'radial-from-peak' or 'radial-center'")

# ---- 形态（同分位 + 去小斑块）----
from scipy.ndimage import binary_dilation
def binarize_by_quantile(arr,q=0.90):
    thr=np.quantile(arr,q); return (arr>=thr),thr
def perimeter_length(binary):
    er=binary_erosion(binary); edge=binary ^ er; return float(edge.sum())
def component_keep_mask(binary, min_size=100):
    lab,n=label(binary); keep=np.zeros_like(binary,bool); sizes=[]
    if n>0:
        sls=find_objects(lab)
        for i,sl in enumerate(sls, start=1):
            if sl is None: continue
            mask=(lab[sl]==i); sz=int(mask.sum())
            if sz>=min_size: keep[sl]|=mask; sizes.append(sz)
    return keep, len(sizes), sizes
def morph_compare(arrA, arrB, q=0.90, min_size=100, mpp=None):
    bA_raw,thrA=binarize_by_quantile(arrA,q=q); bB_raw,thrB=binarize_by_quantile(arrB,q=q)
    # 小开运算，去毛刺
    bA_raw=binary_opening(bA_raw, iterations=1); bB_raw=binary_opening(bB_raw, iterations=1)
    bA,nA,_=component_keep_mask(bA_raw,min_size=min_size); bB,nB,_=component_keep_mask(bB_raw,min_size=min_size)
    perA=perimeter_length(bA); perB=perimeter_length(bB); areaA=float(bA.sum()); areaB=float(bB.sum())
    if mpp is not None:
        px2=mpp**2
        return dict(q=q,thrA=thrA,thrB=thrB,min_size=min_size,compA=nA,compB=nB,
                    areaA_px=areaA,areaB_px=areaB,areaA_mpc2=areaA*px2,areaB_mpc2=areaB*px2,
                    perA_px=perA,perB_px=perB,perA_mpc=perA*mpp,perB_mpc=perB*mpp)
    else:
        return dict(q=q,thrA=thrA,thrB=thrB,min_size=min_size,compA=nA,compB=nB,
                    areaA_px=areaA,areaB_px=areaB,perA_px=perA,perB_px=perB)

# ---- 脊线骨架（ridge skeleton）+ 关键点采样 + 一对一 F1 ----
def ridge_response(arr, sigmas=(1.5,3.0,6.0)):
    R=np.zeros_like(arr)
    for s in sigmas:
        R=np.maximum(R, -gaussian_laplace(arr, s))  # LoG ridgeness
    return R

def ridge_keypoints(arr, q=0.97, min_sep=14, max_pts=120):
    R = ridge_response(arr, PRIMARY["RIDGE_SIGMAS"])
    thr = np.quantile(R, q)
    mask = R >= thr
    # skeletonize on mask
    skel = skeletonize(mask.astype(bool))
    # 取骨架上响应值作为权重，做“最远点采样”以保证稀疏、均匀
    ys, xs = np.where(skel)
    if len(ys)==0: return []
    vals = R[ys, xs]
    order = np.argsort(-vals)
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

# ---------------- 单数据集主流程 ----------------
def run_one(name, url, CFG):
    if not url.strip():
        print(f"[SKIP] {name}: URL 未提供，跳过。\n")
        return None

    # 载入 κ
    with fits.open(url) as hdul:
        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
        kappa = np.array(data, dtype=np.float64)

    # 统一尺寸与预处理
    if kappa.shape != (CFG["TARGET_N"], CFG["TARGET_N"]):
        zy=CFG["TARGET_N"]/kappa.shape[0]; zx=CFG["TARGET_N"]/kappa.shape[1]
        kappa=zoom(kappa,(zy,zx),order=1)
    kappa=np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    if CFG["SIGMA_SMOOTH_DATA"]>0: kappa=gaussian_filter(kappa, CFG["SIGMA_SMOOTH_DATA"])
    kappa=zscore(kappa)

    ny,nx=kappa.shape; yy,xx=np.mgrid[0:ny,0:nx]
    W2D=tukey2d(kappa.shape, alpha=CFG["TUKEY_ALPHA"])
    BP_MASK=bandpass_mask(kappa.shape, *CFG["BANDPASS"])

    # θ(U)（精确 ζ 相位）
    U=embedding_field(CFG["EMBEDDING"], xx, yy, kappa)
    theta=theta_exact_from_U(U, CFG["T_LO"], CFG["T_HI"], CFG["THETA_GRID_N"], mp_dps=50)
    theta_bp=bandpass_apply(theta * W2D, BP_MASK)

    # κ_pred 管线（window→laplacian→smooth→bandpass→轻抑碎→zscore）
    kappa_w = kappa * W2D
    kpred0  = -laplacian_2d(theta_bp)
    if CFG["SIGMA_SMOOTH_PRED"]>0: kpred0=gaussian_filter(kpred0, CFG["SIGMA_SMOOTH_PRED"])
    kappa_bp = bandpass_apply(kappa_w, BP_MASK)
    kpred_bp = bandpass_apply(kpred0,  BP_MASK)
    kpred_bp = gaussian_filter(kpred_bp, 1.0)   # 轻抑碎
    kappa_bp = zscore(kappa_bp); kpred_bp=zscore(kpred_bp)

    # 三个严格对照（同口径）
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

    # 指标：γ² + JK；best-shift；脊线骨架 F1
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

    # BEST：best-shift 后脊线骨架一对一 F1
    dy,dx=M_best["shift"]
    kd2 = M_best["kd2"]
    kp2_aligned=np.roll(np.roll(M_best["kp2"],dy,axis=0),dx,axis=1)
    pts_d = ridge_keypoints(kd2, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
    pts_p = ridge_keypoints(kp2_aligned, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
    F1_best = f1_hungarian_points(pts_d, pts_p, R=PRIMARY["F1_MATCH_RADIUS"])

    # 对照：不做 best-shift（严格）
    def ridge_F1_noalign(kd2, kp2):
        A = ridge_keypoints(kd2, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
        B = ridge_keypoints(kp2, q=PRIMARY["RIDGE_Q"], min_sep=PRIMARY["RIDGE_MIN_SEP"])
        return f1_hungarian_points(A,B, R=PRIMARY["F1_MATCH_RADIUS"])
    F1_sh = ridge_F1_noalign(M_shuf["kd2"], M_shuf["kp2"])
    F1_ro = ridge_F1_noalign(M_roll["kd2"], M_roll["kp2"])
    F1_rt = ridge_F1_noalign(M_rot["kd2"],  M_rot["kp2"])

    # 形态（同带宽）
    morph=morph_compare(kappa_bp, kpred_bp, q=PRIMARY["MORPH_Q"], min_size=PRIMARY["MORPH_MIN_SIZE"], mpp=PRIMARY["MPP"])

    # 幂谱（θ_bp vs kpred_bp）
    def fit_slope(k,P, lo=0.10,hi=0.25):
        n=len(k); i0=int(n*lo); i1=int(n*hi); kk=k[i0:i1].astype(float); pp=P[i0:i1].astype(float)
        return float(np.polyfit(np.log(kk+1e-9), np.log(pp+1e-12), 1)[0])
    kt,Pt=isotropic_ps(zscore(theta_bp)); kp,Pp=isotropic_ps(kpred_bp)
    s_theta=fit_slope(kt,Pt,*PRIMARY["SLOPE_FIT_FRAC"]); s_pred=fit_slope(kp,Pp,*PRIMARY["SLOPE_FIT_FRAC"])

    # 简要图
    plt.figure(figsize=(5.6,4.6)); plt.imshow(kappa_bp, origin='lower'); plt.colorbar(); plt.title(f"{name}: κ (bandpassed,z)"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(5.6,4.6)); plt.imshow(kpred_bp, origin='lower'); plt.colorbar(); plt.title("κ_pred ∝ −∇²θ(U) (bandpassed,z)"); plt.tight_layout(); plt.show()

    # 相干曲线（BEST vs shuffle）
    kd,Pd=isotropic_ps(kappa_bp); kp2,Pp2=isotropic_ps(kpred_bp); kx,Px=isotropic_cross_ps(kappa_bp,kpred_bp)
    c=min(len(Pd),len(Pp2),len(Px)); Pd,Pp2,Px=Pd[:c],Pp2[:c],Px[:c]; g2_curve=(Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp2,1e-20))
    kd,Pd=isotropic_ps(kappa_bp); kp3,Pp3=isotropic_ps(kpred_sh); kx,Px=isotropic_cross_ps(kappa_bp,kpred_sh)
    c2=min(len(Pd),len(Pp3),len(Px)); Pd,Pp3,Px=Pd[:c2],Pp3[:c2],Px[:c2]; g2_sh_curve=(Px**2)/(np.maximum(Pd,1e-20)*np.maximum(Pp3,1e-20))
    plt.figure(figsize=(5.6,4.6)); plt.semilogy(np.arange(len(g2_curve)), g2_curve, label="γ²(data×pred)"); plt.semilogy(np.arange(len(g2_sh_curve)), g2_sh_curve, label="γ²(shuffled)"); plt.legend(); plt.title("Coherence vs k"); plt.tight_layout(); plt.show()

    # 判据
    def pass_flag(real,ctrl_max,thr): return (real>=thr) and (real>max(ctrl_max,1e-9)*1.5)
    ok_gamma = pass_flag(M_best["gamma2"], max(M_shuf["gamma2"],M_roll["gamma2"],M_rot["gamma2"]), THRESH["gamma2_min"])
    ctrl_F1_max = max(F1_sh,F1_ro,F1_rt)
    ok_f1   = pass_flag(F1_best, ctrl_F1_max, THRESH["f1_min"])
    bonus_r = (M_best["r_best"] >= THRESH["r_bonus"])

    # 输出
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
          f"comps: data={morph['compA']} pred={morph['compB']}; "
          f"area(px): data={morph['areaA_px']:.0f} pred={morph['areaB_px']:.0f}; "
          f"perim(px): data={morph['perA_px']:.0f} pred={morph['perB_px']:.0f}")
    if PRIMARY['MPP'] is not None:
        print(f"          area(Mpc^2): data={morph['areaA_mpc2']:.2f} pred={morph['areaB_mpc2']:.2f}; "
              f"perim(Mpc): data={morph['perA_mpc']:.2f} pred={morph['perB_mpc']:.2f}")
    print(f"\nPASS: γ̄²≥{THRESH['gamma2_min']}? {ok_gamma} | Ridge-F1≥{THRESH['f1_min']} & ≥1.5×ctrl? {ok_f1} | bonus r_best≥{THRESH['r_bonus']}? {bonus_r}")
    print("Done.\n")

    return dict(metrics=(M_best,M_shuf,M_roll,M_rot), morph=morph, F1_best=F1_best, F1_ctrl=(F1_sh,F1_ro,F1_rt))

# ---------------- 逐数据集运行（完全不改参） ----------------
for (name, url) in DATASETS:
    _ = run_one(name, url, PRIMARY)

print("提示：将 MACS0416_URL / A370_URL 填上官方 CATS v4 κ FITS 直链后，重新运行本单元即可完成“不改参跨天区复现”。")
