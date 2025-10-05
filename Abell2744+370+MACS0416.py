# @title DAG–UTH Cosmology (v3.0-compact) — HFF CATSv4 κ-maps, Rayleigh certificate + stable dynamics
# @markdown 直接运行：会下载 Abell2744 / MACS0416 / Abell370 三张 κ 图，计算 t=0 的 Rayleigh 证书（Jmax、PD）并做 DAG–UTH 动力学的稳定扫描（相对化 ⟨K_t⟩ 轨迹）。
# ========= Installs & imports =========
import sys, subprocess, warnings, io, math, numpy as np
warnings.filterwarnings("ignore")
def _pip(x): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + x)
try:
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy import fft as spfft
    from scipy.ndimage import zoom, gaussian_filter, gaussian_laplace, maximum_filter, label, find_objects
    from scipy.stats import pearsonr
    from skimage.morphology import skeletonize
except Exception:
    _pip(["matplotlib","astropy","scipy","scikit-image"])
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy import fft as spfft
    from scipy.ndimage import zoom, gaussian_filter, gaussian_laplace, maximum_filter, label, find_objects
    from scipy.stats import pearsonr
    from skimage.morphology import skeletonize
try:
    import mpmath as mp
except Exception:
    _pip(["mpmath"]); import mpmath as mp

# ========= Config =========
URLS = dict(
    Abell2744="https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits",
    MACS0416 ="https://archive.stsci.edu/pub/hlsp/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits",
    Abell370 ="https://archive.stsci.edu/pub/hlsp/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits",
)
TARGET_N = 500
SMOOTH_DATA = 1.0
BANDPASS = (0.10, 0.25)
T_LO, T_HI = 80.0, 4000.0
GRID_THETA = 700
MP_DPS = 50
EMBEDDING = "radial"     # "radial"|"elliptic"|"dual"
ALPHA_BAND_SMOOTH = 1.0

# DAG–UTH params (stable defaults)
T_STEPS = 120
MU_LIST  = [0.02, 0.08, 0.12]
GAM_LIST = [0.50, 0.70, 0.90]
STEP_LIST= [3, 4]
ALPHA_NL = 0.6         # coefficient for +alpha * Phi^2 term (soft-saturated)
CFL_SAFE = 0.18        # CFL factor for explicit Laplacian
LP_FRAC  = 0.40        # small spectral low-pass (fraction of Nyquist)
CLIP_K   = (0.0, 10.0) # clip for reported <K_t> (avoid blow-ups in display)

# ========= Helpers =========
def zscore(a): m=a.mean(); s=a.std()+1e-12; return (a-m)/s
def tukey2d(shape, alpha=0.20):
    ny,nx=shape; y=np.arange(ny); x=np.arange(nx)
    def tukey(M, a=0.5):
        if a<=0: return np.ones(M)
        if a>=1: n=np.arange(M); return 0.5*(1-np.cos(2*np.pi*n/(M-1)))
        n=np.arange(M); w=np.ones(M); w[:int(a*(M-1)/2)] = 0.5*(1+np.cos(np.pi*(2*n[:int(a*(M-1)/2)]/(a*(M-1)) - 1)))
        w[int(M - a*(M-1)/2):] = 0.5*(1+np.cos(np.pi*(2*n[int(M - a*(M-1)/2):]/(a*(M-1)) - 2/a + 1)))
        return w
    return np.outer(tukey(ny,alpha), tukey(nx,alpha))

def bandpass_mask(shape, kmin_frac, kmax_frac):
    ny,nx=shape; ky=spfft.fftfreq(ny)*ny; kx=spfft.fftfreq(nx)*nx
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2); kmax=K.max()
    return (K>=kmin_frac*kmax)&(K<=kmax_frac*kmax)

def fft_apply(f, mask): return spfft.ifft2(spfft.fft2(f)*mask).real

def laplace2(f):
    return -4*f + np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1)

# θ(U) from Riemann ζ phase (cache)
THETA_CACHE={}
def theta_from_U(U, t_lo=T_LO, t_hi=T_HI, N=GRID_THETA, mpdps=MP_DPS):
    key=(float(U.min()),float(U.max()),t_lo,t_hi,N,mpdps, U.shape)
    if key in THETA_CACHE: return THETA_CACHE[key]
    mp.mp.dps=mpdps
    # map U→t
    Umin,Umax=float(U.min()), float(U.max()); t = t_lo + (U-Umin)*(t_hi-t_lo)/(Umax-Umin+1e-12)
    # precompute θ(t) on grid
    grid = np.linspace(t_lo, t_hi, N); vals=[]
    for tv in grid:
        z = mp.zeta(0.5+1j*tv); vals.append(float(mp.arg(z)))
    theta_grid = np.unwrap(np.array(vals))
    # interp
    th = np.interp(t, grid, theta_grid)
    th = th - th.mean()
    THETA_CACHE[key]=th; return th

def embedding_field(kind, xx, yy, kappa):
    ny,nx=kappa.shape
    if kind=="radial":
        cy,cx = np.unravel_index(np.argmax(kappa), kappa.shape)
        X=(xx-cx)/(nx/2); Y=(yy-cy)/(ny/2); return np.sqrt(X**2+Y**2)+1e-6
    elif kind=="elliptic":
        thr=np.quantile(kappa,0.8); mask=(kappa>=thr)
        ys,xs=np.where(mask)
        if len(ys)<10:
            cy,cx = np.unravel_index(np.argmax(kappa), kappa.shape)
            X=(xx-cx)/(nx/2); Y=(yy-cy)/(ny/2); return np.sqrt(X**2+Y**2)+1e-6
        cy, cx = ys.mean(), xs.mean()
        Y = ys - cy; X = xs - cx
        cov=np.cov(np.vstack([Y,X])); w,v=np.linalg.eigh(cov); order=np.argsort(w)[::-1]; v=v[:,order]; w=w[order]
        a=max(np.sqrt(max(w[0],1e-9))*2.5, 1.0); b=max(np.sqrt(max(w[1],1e-9))*2.5, 1.0)
        ang=np.arctan2(v[0,0], v[1,0])
        ca,sa=np.cos(ang),np.sin(ang)
        du=ca*(xx-cx)+sa*(yy-cy); dv=-sa*(xx-cx)+ca*(yy-cy)
        return np.sqrt((du/(a*nx/500))**2 + (dv/(b*ny/500))**2)+1e-6
    elif kind=="dual":
        # pick top-2 peaks with separation
        neigh=maximum_filter(kappa, size=40); peaks=(kappa==neigh); lab,n=label(peaks)
        if n==0:
            cy,cx=np.unravel_index(np.argmax(kappa),kappa.shape)
            X=(xx-cx)/(nx/2); Y=(yy-cy)/(ny/2); return np.sqrt(X**2+Y**2)+1e-6
        centers=[]
        for i,sl in enumerate(find_objects(lab), start=1):
            if sl is None: continue
            sub=kappa[sl]; iy,ix=np.unravel_index(np.argmax(sub), sub.shape)
            y,x=sl[0].start+iy, sl[1].start+ix; centers.append((y,x,kappa[y,x]))
        centers.sort(key=lambda t:-t[2]); pts=[(int(y),int(x)) for y,x,_ in centers[:2]]
        if len(pts)<2:
            cy,cx=np.unravel_index(np.argmax(kappa),kappa.shape)
            X=(xx-cx)/(nx/2); Y=(yy-cy)/(ny/2); return np.sqrt(X**2+Y**2)+1e-6
        (y1,x1),(y2,x2)=pts
        r1=np.sqrt(((xx-x1)/(nx/2))**2+((yy-y1)/(ny/2))**2)+1e-6
        r2=np.sqrt(((xx-x2)/(nx/2))**2+((yy-y2)/(ny/2))**2)+1e-6
        tau=0.08; m=np.minimum(r1,r2); M=np.maximum(r1,r2); return m - tau*np.log1p(np.exp(-(M-m)/tau))
    else:
        raise ValueError("EMBEDDING must be 'radial'|'elliptic'|'dual'")

def omega_from_theta(theta):
    # simple finite diff with smooth
    return gaussian_filter(np.gradient(theta, axis=0), 1.0)  # treat "time" as one axis proxy in local window

def rayleigh_Jmax(omega, q=None):
    if q is None: q = np.ones_like(omega)
    return float(np.sum((omega**2)/(q+1e-12)))

def PD_from_J(Jmax, alpha=0.01):
    # GLRT under H0: N(0,1), under H1: mean sqrt(Jmax). threshold z_{1-alpha}
    import math
    from math import erf, sqrt
    # inverse CDF approx for 1-alpha
    import mpmath as mp2
    z = float(mp2.sqrt(2)*mp2.erfinv(2*(1-alpha)-1))
    mu = math.sqrt(max(Jmax,0.0))
    # PD = 1 - Phi(z - mu)
    Phi=lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
    return 1.0 - Phi(z - mu)

def isotropic_ps(f):
    F=spfft.fft2(f); P=(F*np.conj(F)).real
    ny,nx=f.shape; ky=spfft.fftfreq(ny)*ny; kx=spfft.fftfreq(nx)*nx
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2).astype(int)
    ps=np.bincount(K.ravel(),P.ravel()); cnt=np.bincount(K.ravel()); ps/=np.maximum(cnt,1); k=np.arange(len(ps))
    return k[1:], ps[1:]

def gamma2_band(a,b, lo=0.10, hi=0.25):
    kd,Pd=isotropic_ps(a); kp,Pp=isotropic_ps(b); kx,Px=isotropic_ps((a+b)/2)  # a small proxy
    m=min(len(Pd),len(Pp)); Pd,Pp=Pd[:m],Pp[:m]
    # cross approx (keep simple here): use correlation in band
    n=len(Pd); i0=int(n*lo); i1=int(n*hi)
    A=np.log(Pd[i0:i1]+1e-18); B=np.log(Pp[i0:i1]+1e-18)
    r=np.corrcoef(A,B)[0,1]; return max(r,0.0)

def ridge_F1(imA, imB, q=0.97, min_sep=14, R=12):
    def ridge_pts(arr):
        Rimg = np.maximum.reduce([ -gaussian_laplace(arr, s) for s in (1.5,3.0,6.0) ])
        thr=np.quantile(Rimg, q); sk = skeletonize((Rimg>=thr).astype(bool))
        ys,xs=np.where(sk); vals=Rimg[ys,xs]
        if len(ys)==0: return []
        order=np.argsort(-vals); pts=[]
        for idx in order:
            y,x=int(ys[idx]),int(xs[idx])
            if all((y-yy)**2+(x-xx)**2>=min_sep**2 for yy,xx,_ in pts):
                pts.append((y,x,float(vals[idx])))
                if len(pts)>=120: break
        return [(y,x) for y,x,_ in pts]
    A=ridge_pts(imA); B=ridge_pts(imB)
    if len(A)==0 or len(B)==0: return 0.0
    A=np.array(A); B=np.array(B)
    D=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(2))
    # greedy match
    hits=0; usedB=set()
    for i in range(len(A)):
        j=int(np.argmin(D[i])); 
        if D[i,j]<=R and j not in usedB:
            hits+=1; usedB.add(j)
    prec=hits/max(len(B),1); rec=hits/max(len(A),1)
    return 2*prec*rec/(prec+rec+1e-12)

# spectral low-pass
def lowpass_fraction(f, frac=LP_FRAC):
    ny,nx=f.shape; ky=spfft.fftfreq(ny)*ny; kx=spfft.fftfreq(nx)*nx
    KX,KY=np.meshgrid(kx,ky); K=np.sqrt(KX**2+KY**2); kmax=K.max()
    mask = (K <= frac*kmax)
    return fft_apply(f, mask)

# ========= Load & preprocess κ =========
def load_kappa(url):
    with fits.open(url) as hdul:
        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
        kappa = np.array(data, dtype=np.float64)
    # resize
    zy=TARGET_N/kappa.shape[0]; zx=TARGET_N/kappa.shape[1]
    kappa=zoom(kappa,(zy,zx),order=1)
    kappa=np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    if SMOOTH_DATA>0: kappa=gaussian_filter(kappa, SMOOTH_DATA)
    return zscore(kappa)

# ========= Rayleigh t=0 certificate =========
def t0_certificate(kappa, embedding=EMBEDDING):
    ny,nx=kappa.shape; yy,xx=np.mgrid[0:ny,0:nx]
    U=embedding_field(embedding, xx, yy, kappa)
    theta = theta_from_U(U)
    # bandpass θ
    W = tukey2d(kappa.shape, 0.2)
    mask = bandpass_mask(kappa.shape, *BANDPASS)
    theta_bp = fft_apply(theta*W, mask)
    omega = gaussian_filter(np.gradient(theta_bp, axis=0), 1.0)
    J = rayleigh_Jmax(omega)
    PD = PD_from_J(J, alpha=0.01)
    return dict(Jmax=J, PD=PD, theta_bp=theta_bp)

# ========= Stable DAG–UTH dynamics =========
def run_dynamics(kappa, mu, gamma, steps=T_STEPS):
    # init Phi,H （用 κ 的正/负部分做一个无偏起点）
    Phi = zscore(np.maximum(kappa, 0.0))
    H   = zscore(np.maximum(-kappa,0.0)+1e-6)

    # adaptive dt by CFL (explicit Laplacian in 2D)
    # Δt ≤ CFL / (4 * max(μ,γ))
    dt = CFL_SAFE / (4.0 * max(mu, gamma, 1e-9))

    # small spectral LP for stability
    def evolve(Phi, H):
        # Laplacian
        Lp = laplace2(Phi); Lh = laplace2(H)
        # soft-saturated nonlinearities
        Phi_nl = - gamma * Phi * np.tanh(H)
        H_src  =  ALPHA_NL * np.tanh(Phi) * Phi
        # explicit Euler
        Phi_new = Phi + dt*( mu*Lp + Phi_nl )
        H_new   = H   + dt*( gamma*Lh + H_src )
        # spectral low-pass + light smoothing
        Phi_new = lowpass_fraction(Phi_new, LP_FRAC)
        H_new   = lowpass_fraction(H_new, LP_FRAC)
        Phi_new = gaussian_filter(Phi_new, 0.6)
        H_new   = gaussian_filter(H_new, 0.6)
        # renormalize to keep scales comparable
        Phi_new = zscore(Phi_new)
        H_new   = zscore(H_new + 1e-6)
        return Phi_new, H_new

    K_mean = []
    for t in range(steps):
        K = Phi/(H+1e-6)
        K_mean.append(float(np.mean(K)))
        Phi, H = evolve(Phi, H)

    K_mean = np.array(K_mean)
    # relative & clipped for reporting
    base = max(abs(K_mean[0]), 1e-6)
    K_rel = np.clip(K_mean/base, CLIP_K[0], CLIP_K[1])
    emerge = np.any(K_rel > 1.02)
    return dict(K_rel=K_rel, emerge=emerge, dt=dt)

# ========= Main run =========
def run_all():
    results=[]
    for name,url in URLS.items():
        print(f"\n=== {name} ===")
        kappa = load_kappa(url)
        # t=0 Rayleigh
        cert = t0_certificate(kappa)
        print(f"t=0 Rayleigh: Jmax={cert['Jmax']:.3f}, PD={cert['PD']:.3f}")

        # Eq.(5) morphology proxies（bandpassed/normalized）
        mask = bandpass_mask(kappa.shape, *BANDPASS)
        W = tukey2d(kappa.shape, 0.2)
        kappab = fft_apply(kappa*W, mask)
        # a crude κ_pred ∝ −∇²θ(U) for geometry proxy
        ny,nx=kappa.shape; yy,xx=np.mgrid[0:ny,0:nx]
        U = embedding_field(EMBEDDING, xx, yy, kappa)
        theta = theta_from_U(U)
        thetab = fft_apply(theta*W, mask)
        kpred  = -laplace2(thetab)
        kpred  = gaussian_filter(kpred, ALPHA_BAND_SMOOTH)
        kappab = zscore(kappab); kpred=zscore(kpred)
        g2 = gamma2_band(kappab, kpred)
        F1 = ridge_F1(kappab, kpred)
        print(f"Eq.(5) proxies: gamma2~{g2:.3f}, Ridge-F1={F1:.3f}")

        # dynamics scan
        best=None; curves=[]
        for mu in MU_LIST:
            for ga in GAM_LIST:
                for s in STEP_LIST:
                    dyn = run_dynamics(kappa, mu, ga, steps=40+20*s)
                    curves.append((mu,ga,s,dyn))
                    if best is None or dyn['K_rel'][-1] > best['K_rel'][-1]:
                        best = dict(mu=mu,gamma=ga,s=s, **dyn)
                    print(f"  μ={mu:.2f}, γ={ga:.2f}, s={s} | end K_rel={dyn['K_rel'][-1]:.3f}, emerge={dyn['emerge']} (dt={dyn['dt']:.3e})")

        # plot K_rel
        plt.figure(figsize=(6,4))
        for mu,ga,s,dyn in curves:
            plt.plot(dyn['K_rel'], alpha=0.6, label=f"μ={mu},γ={ga},s={s}")
        plt.axhline(1.0, ls='--', lw=1, color='k')
        plt.title(f"{name}: ⟨K_t⟩ (relative, clipped) — emergence if >1")
        plt.xlabel("step"); plt.ylabel("⟨K_t⟩ / ⟨K_0⟩")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.show()

        print(f"⇒ Best (by final K_rel): μ={best['mu']:.2f}, γ={best['gamma']:.2f}, s={best['s']} | "
              f"K_rel_end={best['K_rel'][-1]:.3f}, emerge={best['emerge']}, dt={best['dt']:.3e}")

        results.append((name, cert, g2, F1, best))
    # summary
    print("\n=== Summary (t=0 + best dynamics) ===")
    for name, cert, g2, F1, best in results:
        print(f"{name}: Jmax={cert['Jmax']:.3f}, PD={cert['PD']:.3f} | "
              f"Eq5[g2~{g2:.3f}, F1={F1:.3f}] | "
              f"Best μ={best['mu']:.2f},γ={best['gamma']:.2f},s={best['s']}, "
              f"K_rel_end={best['K_rel'][-1]:.3f}, emerge={best['emerge']}")
    print("\nTip: 改 EMBEDDING='elliptic' 或 'dual' 可做几何敏感性对照；改 LP_FRAC / CFL_SAFE 可检验数值稳健性。")

run_all()

