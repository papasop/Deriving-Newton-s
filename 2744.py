# ============================================================
#  Riemann ζ  → refractive strong-field test (Colab-ready)
#  - Single cell, no external deps beyond NumPy/Matplotlib
#  - Save plots/CSVs to /content, also plt.show() everything
# ============================================================
import os, math
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 0) Config
# -------------------------------
OUTDIR = "/content"
os.makedirs(OUTDIR, exist_ok=True)

# 若你有高 t 的真实零点（1D float 数组 t_n），把它上传到 /content/zeros.npy 并设 False
USE_BUILTIN_ZEROS = True

# 强场非线性强度（可调）
BETA0 = 0.7
GRID_N = 224   # 提高到 384/512 会更稳（强非线性时）
SEED = 0

np.random.seed(SEED)

# -------------------------------
# 1) Load zeros
# -------------------------------
if USE_BUILTIN_ZEROS:
    t_zeros = np.array([
        14.134725141, 21.022039639, 25.010857580, 30.424876125, 32.935061588,
        37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
        52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
        67.079810529, 69.546401710, 72.067157674, 75.704690699, 77.144840069,
        79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
        92.491899271, 94.651344041, 95.870634228, 98.831194218
    ], dtype=float)
else:
    path = "/content/zeros.npy"
    assert os.path.exists(path), "请先把你的真实零点数组（1D float）上传到 /content/zeros.npy"
    t_zeros = np.load(path).astype(float)

tmin, tmax = float(t_zeros.min()), float(t_zeros.max())

# -------------------------------
# 2) Phase surrogate ω, θ  (Hadamard-inspired)
# -------------------------------
def omega_from_zeros(t_grid, t_zeros):
    # ω(t) = 1/2 log(t/2π) + Σ (t - tn)/((t-tn)^2 + 1/4)
    w = 0.5*np.log(t_grid/(2*np.pi))
    for tn in t_zeros:
        dt = t_grid - tn
        w += dt/(dt*dt + 0.25)
    return w

Ng = max(3000, min(20000, int(40*len(t_zeros))))
t_grid = np.linspace(tmin, tmax, Ng)
omega = omega_from_zeros(t_grid, t_zeros)

# θ(t) = ∫ ω dt （梯形积分），并去均值
theta = np.cumsum((omega[:-1] + omega[1:]) * (t_grid[1] - t_grid[0]) * 0.5)
theta = np.concatenate([[0.0], theta])
theta -= theta.mean()

# -------------------------------
# 3) Windowed SCI → K → g = 1/K with asymptotic shrinkage
# -------------------------------
def window_stats(t_grid, omega, W):
    centers, Phi, H = [], [], []
    for i in range(len(t_grid) - W + 1):
        sl = slice(i, i+W)
        tg = t_grid[sl]; om = omega[sl]
        centers.append(tg.mean()); Phi.append(np.mean(om)); H.append(np.var(om))
    return np.array(centers), np.array(Phi), np.array(H)

def estimate_K(centers, Phi_w, H_w, span=51):
    lnPhi = np.log(np.maximum(Phi_w, 1e-12))
    lnH   = np.log(np.maximum(H_w,  1e-18))
    K = np.zeros_like(lnPhi)
    half = span // 2
    for i in range(len(lnPhi)):
        a = max(0, i-half); b = min(len(lnPhi), i+half+1)
        x = lnPhi[a:b]; y = lnH[a:b]
        A = np.vstack([x, np.ones_like(x)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        K[i] = sol[0]
    c_trim = centers[half:len(centers)-half]
    return c_trim, K[half:len(K)-half]

W = max(60, len(t_grid)//80)  # ~1.25% 窗宽
centers, Phi_w, H_w = window_stats(t_grid, omega, W)
cK, K_est = estimate_K(centers, Phi_w, H_w, span=51)

# 渐近修正： g_asym = 1/2 + π / [6 log(t/2π)]
L = np.log(np.maximum(cK/(2*np.pi), 2.0))
g_asym = 0.5 + np.pi/(6.0*L)
g_emp  = 1.0 / np.maximum(K_est, 1e-6)

# 低 t 用收缩（λ 大），高 t 让 1/K 主导（λ 小）
lam   = np.minimum(1.0, 1.0/np.maximum(L, 1.0))
g_hat = lam*g_asym + (1.0 - lam)*g_emp

# g(t) 图
plt.figure(figsize=(6.4,4))
plt.plot(cK, g_emp,  label="1/K (emp)", alpha=0.6)
plt.plot(cK, g_asym, label="asym 1/2 + π/[6 log(t/2π)]", alpha=0.8)
plt.plot(cK, g_hat,  label="ĝ (shrinkage)", lw=2)
plt.xlabel("t (window center)"); plt.ylabel("g(t)"); plt.title("g(t) from real ζ zeros")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTDIR}/g_of_t_realzeros.png", dpi=160)
plt.show()

# CSV
pd.DataFrame({"t_center": cK, "g_emp": g_emp, "g_asym": g_asym, "g_hat": g_hat, "lambda": lam})\
  .to_csv(f"{OUTDIR}/g_of_t_realzeros.csv", index=False)

# -------------------------------
# 4) Embed to space: U(x)=k·x+b → φ_seed = g(U)[θ(U)-⟨θ⟩]
# -------------------------------
N = GRID_N
Lx = Ly = 2.0
x = np.linspace(0, Lx, N, endpoint=False)
y = np.linspace(0, Ly, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

U = tmin + (tmax - tmin) * (X / Lx)  # 单调沿 x
theta_U = np.interp(U, t_grid, theta)
g_U     = np.interp(U, cK, g_hat, left=g_asym[0], right=g_asym[-1])

phi_seed = g_U * (theta_U - theta_U.mean())
phi_seed -= phi_seed.mean()

plt.figure(figsize=(6,5))
plt.imshow(phi_seed.T, origin='lower', extent=[0,Lx,0,Ly], aspect='equal')
plt.colorbar(); plt.title(r"$\phi_{\rm seed}=g(U)[\theta(U)-\langle\theta\rangle]$ (real zeros)")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.savefig(f"{OUTDIR}/phi_seed_realzeros.png", dpi=160)
plt.show()

# -------------------------------
# 5) Spectral ops & fields
# -------------------------------
kx = 2*np.pi*fft.fftfreq(N, d=Lx/N)
ky = 2*np.pi*fft.fftfreq(N, d=Ly/N)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
K2[0,0] = 1.0  # avoid 0-div

def grad(phi):
    ph = fft.fft2(phi)
    gx = np.real(fft.ifft2(1j*KX*ph))
    gy = np.real(fft.ifft2(1j*KY*ph))
    return gx, gy

def lap(phi):
    ph = fft.fft2(phi)
    return np.real(fft.ifft2(-K2*ph))

def invlap(rhs):
    Rh = fft.fft2(rhs)
    Ph = Rh / (-K2)
    Ph[0,0] = 0.0
    return np.real(fft.ifft2(Ph))

kappa = 1.0
rho = -lap(phi_seed)
rho -= rho.mean()  # periodic DC 修正

# -------------------------------
# 6) β(x) from gradient proxy & Strong-field solve
# PDE: ∇²φ + β(x) |∇φ|^4 = -κρ
# -------------------------------
gx0, gy0 = grad(phi_seed)
g2_0 = gx0*gx0 + gy0*gy0
beta_map = BETA0 * (g2_0 / (np.percentile(g2_0, 90) + 1e-12))   # 归一化
beta_map = np.clip(beta_map, 0.0, BETA0)
# 光滑一下 β，避免棋盘格
beta_map = 0.5 * (beta_map + (np.roll(beta_map,1,0)+np.roll(beta_map,-1,0)+
                               np.roll(beta_map,1,1)+np.roll(beta_map,-1,1))/4.0)

def solve_strong(rho, beta_map, steps=6, iters=240, tol=2e-6, relax=0.62):
    phi = invlap(-kappa*rho); phi -= phi.mean()
    res_hist = []
    for s in range(1, steps+1):
        w = s / steps
        beta_eff = w * beta_map
        for it in range(iters):
            gx, gy = grad(phi)
            g2 = gx*gx + gy*gy
            rhs = -kappa*rho - beta_eff*(g2**2)
            phi_new = invlap(rhs)
            phi = (1.0-relax)*phi + relax*phi_new
            phi -= phi.mean()
            F = lap(phi) + beta_eff*(g2**2) + kappa*rho
            res = float(np.sqrt(np.mean(F**2)))
            res_hist.append(res)
            if res < tol:
                break
    return phi, np.array(res_hist)

phi_lin = invlap(-kappa*rho); phi_lin -= phi_lin.mean()
phi_non, hist = solve_strong(rho, beta_map)

plt.figure(figsize=(6.4,4))
plt.semilogy(hist)
plt.xlabel("Iteration"); plt.ylabel("RMS residual")
plt.title("Strong-field residual (real zeros)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/residual_realzeros.png", dpi=160)
plt.show()

# -------------------------------
# 7) Observables: deflection α(b) & Δα(b)
# -------------------------------
def deflection(phi):
    gx, gy = grad(phi)
    # Born-like: integrate ∂x φ along y
    return np.trapz(gx, y, axis=1)

b = x
alpha_lin = deflection(phi_lin)
alpha_non = deflection(phi_non)
delta_alpha = alpha_non - alpha_lin

plt.figure(figsize=(6.4,4))
plt.plot(b, alpha_lin, label="linear")
plt.plot(b, alpha_non, label="nonlinear")
plt.xlabel("impact parameter b"); plt.ylabel("deflection α(b)")
plt.title("Deflection vs b (real zeros)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTDIR}/deflection_realzeros_full.png", dpi=160)
plt.show()

plt.figure(figsize=(6.4,3.6))
plt.plot(b, delta_alpha)
plt.axhline(0, color='k', lw=0.8)
plt.xlabel("b"); plt.ylabel("Δα(b)")
plt.title("Δα(b) = α_NL - α_Lin (real zeros)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/delta_alpha_realzeros_full.png", dpi=160)
plt.show()

# CSV
pd.DataFrame({"b": b, "alpha_linear": alpha_lin, "alpha_nonlinear": alpha_non, "delta_alpha": delta_alpha})\
  .to_csv(f"{OUTDIR}/deflection_realzeros_full.csv", index=False)

# -------------------------------
# 8) Nonlinear Gauss closure (square shells)
#  ∮∂V ∇φ·n dS + κ∫V ρ dA + ∫V β|∇φ|^4 dA ≈ 0
# -------------------------------
def gauss_square_closure(phi, rho, beta_map, center, half_sizes):
    gx, gy = grad(phi)
    dA = (Lx/N) * (Ly/N)
    rels, fluxes, masses, Nterms = [], [], [], []

    def sample(arr, xi, yi):
        i = (xi/Lx)*N; j = (yi/Ly)*N
        i0 = np.floor(i).astype(int)%N; j0 = np.floor(j).astype(int)%N
        di = i - i0; dj = j - j0
        i1 = (i0+1)%N; j1 = (j0+1)%N
        return ((1-di)*(1-dj)*arr[i0,j0] + di*(1-dj)*arr[i1,j0] +
                (1-di)*dj*arr[i0,j1] + di*dj*arr[i1,j1])

    cx, cy = center; m = 2048
    for a in np.asarray(half_sizes):
        t = np.linspace(-a, a, m, endpoint=False)
        xl, yl = (cx - a)*np.ones_like(t), cy + t
        xr, yr = (cx + a)*np.ones_like(t), cy + t
        xb, yb = cx + t, (cy - a)*np.ones_like(t)
        xt, yt = cx + t, (cy + a)*np.ones_like(t)
        gx_l = sample(gx, xl, yl); gy_l = sample(gy, xl, yl)
        gx_r = sample(gx, xr, yr); gy_r = sample(gy, xr, yr)
        gx_b = sample(gx, xb, yb); gy_b = sample(gy, xb, yb)
        gx_t = sample(gx, xt, yt); gy_t = sample(gy, xt, yt)
        flux = (np.trapz(-gx_l, t) + np.trapz(+gx_r, t) +
                np.trapz(-gy_b, t) + np.trapz(+gy_t, t))

        mask = (np.abs((X-cx))<=a) & (np.abs((Y-cy))<=a)
        mass = rho[mask].sum()*dA
        gxv, gyv = gx[mask], gy[mask]
        Nterm = (beta_map[mask]*((gxv*gxv + gyv*gyv)**2)).sum()*dA

        clos = flux + kappa*mass + Nterm
        denom = np.abs(flux) + np.abs(kappa*mass) + np.abs(Nterm) + 1e-12
        rels.append(np.abs(clos)/denom)
        fluxes.append(flux); masses.append(mass); Nterms.append(Nterm)

    return np.array(rels), np.array(fluxes), np.array(masses), np.array(Nterms)

half_sizes = np.linspace(0.06*Lx, 0.35*Lx, 24)
rels, Fv, Mv, Nv = gauss_square_closure(phi_non, rho, beta_map, (0.5*Lx, 0.5*Ly), half_sizes)

plt.figure(figsize=(6.4,3.6))
plt.plot(half_sizes, rels)
plt.xlabel("half-size"); plt.ylabel("relative closure")
plt.title("Nonlinear Gauss closure (real zeros)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/closure_realzeros.png", dpi=160)
plt.show()

pd.DataFrame({"half_size": half_sizes, "rel_closure": rels,
              "flux": Fv, "mass_term": kappa*Mv, "nonlinear_term": Nv})\
  .to_csv(f"{OUTDIR}/closure_realzeros.csv", index=False)

# -------------------------------
# 9) Paths print (as requested)
# -------------------------------
print("Artifacts saved in", OUTDIR)
print("- g(t):            ", f"{OUTDIR}/g_of_t_realzeros.png")
print("- φ_seed map:      ", f"{OUTDIR}/phi_seed_realzeros.png")
print("- residual history:", f"{OUTDIR}/residual_realzeros.png")
print("- deflection:      ", f"{OUTDIR}/deflection_realzeros_full.png")
print("- delta-alpha:     ", f"{OUTDIR}/delta_alpha_realzeros_full.png")
print("- closure:         ", f"{OUTDIR}/closure_realzeros.png")
print("- CSVs:")
print("  •", f"{OUTDIR}/g_of_t_realzeros.csv")
print("  •", f"{OUTDIR}/deflection_realzeros_full.csv")
print("  •", f"{OUTDIR}/closure_realzeros.csv")
