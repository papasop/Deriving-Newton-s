# ============================================================
# Unified Field — Gaussian Validation (Version 2.1)
# - No quadpy; robust Fibonacci sphere quadrature
# - Spectral Poisson, DC correction
# - Flux (surface) & Volume closures
# - Tricubic sampler for flux (default), switchable to linear
# - Optional finite-difference Laplacian for informative volume closure
# - plt figures + final print summary (incl. G_flux_est / c_flux_est)
# ============================================================

import numpy as np
import numpy.fft as fft
from math import pi
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

print("Running Unified Field Gaussian Validation — Version 2.1")

# ------------------------ Quadrature ------------------------
def sphere_quadrature(n_points: int = 8192):
    """Fibonacci sphere with equal weights summing to 4π (no external deps)."""
    M = int(n_points)
    i = np.arange(M, dtype=float)
    phi = (1 + 5**0.5) / 2
    z = 1 - 2*(i + 0.5)/M
    theta = 2*pi*i/phi
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)     # (M,3)
    weights = np.full(M, 4*pi/M)           # sum(weights)=4π
    return dirs, weights

# ------------------------ Grids & Spectral ops ------------------------
def make_kgrid(N, dx):
    L = N*dx
    k1 = 2*np.pi*fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    return kx, ky, kz, k2, L

def spectral_laplacian(f, k2):
    f_k = fft.fftn(f)
    out_k = -k2 * f_k
    return np.real(fft.ifftn(out_k))

def spectral_gradient(f, kx, ky, kz):
    f_k = fft.fftn(f)
    gx = np.real(fft.ifftn(1j*kx * f_k))
    gy = np.real(fft.ifftn(1j*ky * f_k))
    gz = np.real(fft.ifftn(1j*kz * f_k))
    return gx, gy, gz

def fd_laplacian(f, dx):
    """6-point 2nd-order periodic finite-difference Laplacian (optional)."""
    fxx = (np.roll(f, -1, 0) - 2*f + np.roll(f,  1, 0)) / dx**2
    fyy = (np.roll(f, -1, 1) - 2*f + np.roll(f,  1, 1)) / dx**2
    fzz = (np.roll(f, -1, 2) - 2*f + np.roll(f,  1, 2)) / dx**2
    return fxx + fyy + fzz

# ------------------------ Physics, params ------------------------
@dataclass
class PhysConst:
    c: float = 2.997925e8   # m/s
    G: float = 6.67430e-11  # SI

@dataclass
class SimParam:
    N: int = 128
    dx: float = 0.015625
    g: float = 0.5
    # closures
    use_flux_surface: bool = True
    use_volume_closure: bool = True
    # radii
    n_sphere_radii: int = 36
    r_min_frac: float = 0.06
    r_max_frac: float = 0.46
    # sphere quadrature
    n_sphere_pts: int = 8192
    # interpolation for flux: "cubic" (tricubic) or "linear"
    interp: str = "cubic"
    # laplacian type: "spectral" (default) or "fd" (finite difference)
    laplacian: str = "spectral"

# ------------------------ Sources & Solvers ------------------------
def gaussian_superposition(N, dx, A1, A2, r1, r2, s1, s2):
    xs = np.arange(N)*dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    def sqdist(x0, y0, z0):
        return (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2
    rho = A1*np.exp(-sqdist(*r1)/(2*s1*s1)) - A2*np.exp(-sqdist(*r2)/(2*s2*s2))
    rho -= rho.mean()  # DC correction
    return rho

def poisson_phi_from_rho(rho, const: PhysConst, par: SimParam):
    N, dx, g = par.N, par.dx, par.g
    kx, ky, kz, k2, L = make_kgrid(N, dx)
    rho_k = fft.fftn(rho)
    rhs_k = -(8*np.pi*const.G/(g*const.c**2)) * rho_k
    phi_k = np.zeros_like(rhs_k)
    mask = (k2 != 0.0)
    phi_k[mask] = rhs_k[mask]/(-k2[mask])
    return np.real(fft.ifftn(phi_k))

def correlation(a, b):
    a = a.ravel() - np.mean(a)
    b = b.ravel() - np.mean(b)
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a, b)/denom) if denom > 0 else np.nan

def tricubic_sampler(volume, dx):
    """Periodic tricubic sampler via map_coordinates(order=3)."""
    N = volume.shape[0]
    L = N*dx
    def eval_at(points):
        pts = np.mod(points, L)
        coords = (pts / dx).T  # (3, M)
        return map_coordinates(volume, coords, order=3, mode='wrap')
    return eval_at

def trilinear_sampler(volume, dx):
    """Periodic trilinear sampler via RegularGridInterpolator."""
    N = volume.shape[0]
    xs = np.arange(N)*dx
    interp = RegularGridInterpolator((xs, xs, xs), volume, bounds_error=False, fill_value=None)
    L = N*dx
    def eval_at(points):
        return interp(np.mod(points, L))
    return eval_at

def periodic_delta(arr, c_scalar, L):
    d = arr - c_scalar
    return (d + 0.5*L) % L - 0.5*L

# ------------------------ Closures ------------------------
def gauss_closure_flux(grad_ln_n, center, radii, par: SimParam):
    gx, gy, gz = grad_ln_n
    if par.interp == "cubic":
        eval_gx = tricubic_sampler(gx, par.dx)
        eval_gy = tricubic_sampler(gy, par.dx)
        eval_gz = tricubic_sampler(gz, par.dx)
    else:
        eval_gx = trilinear_sampler(gx, par.dx)
        eval_gy = trilinear_sampler(gy, par.dx)
        eval_gz = trilinear_sampler(gz, par.dx)

    normals, weights = sphere_quadrature(par.n_sphere_pts)  # (M,3), (M,)
    fluxes = []
    for R in radii:
        pts = center[None,:] + R*normals          # (M,3)
        gxv = eval_gx(pts)                        # (M,)
        gyv = eval_gy(pts)                        # (M,)
        gzv = eval_gz(pts)                        # (M,)
        g_on_surf = np.column_stack((gxv, gyv, gzv))  # (M,3)
        dotn = np.einsum('ij,ij->i', g_on_surf, normals)  # (M,)
        flux = (R**2) * np.sum(weights * dotn)   # sum(weights)=4π
        fluxes.append(flux)
    return np.array(fluxes)

def gauss_closure_volume(lap_ln_n, center, radii, par: SimParam):
    N, dx = par.N, par.dx
    xs = np.arange(N)*dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    L = N*dx
    RX = periodic_delta(X, center[0], L)
    RY = periodic_delta(Y, center[1], L)
    RZ = periodic_delta(Z, center[2], L)
    R2 = RX*RX + RY*RY + RZ*RZ
    vol = dx**3
    integrals = []
    for R in radii:
        mask = (R2 <= R*R)
        integrals.append(lap_ln_n[mask].sum()*vol)
    return np.array(integrals)

# ------------------------ Driver ------------------------
def run_gaussian_validation(const=PhysConst(), par=SimParam()):
    N, dx, g = par.N, par.dx, par.g
    kx, ky, kz, k2, L = make_kgrid(N, dx)
    center = np.array([0.5*L, 0.5*L, 0.5*L])

    # Two 3D Gaussians
    A1, A2 = 1.0, 0.8
    s1, s2 = 0.08*L, 0.07*L
    r1 = center + np.array([ 0.10*L, -0.05*L,  0.07*L])
    r2 = center + np.array([-0.12*L,  0.03*L, -0.06*L])
    rho  = gaussian_superposition(N, dx, A1, A2, r1, r2, s1, s2)

    # Poisson and fields
    phi  = poisson_phi_from_rho(rho, const, par)
    ln_n = g * phi

    # LHS vs RHS
    if par.laplacian == "fd":
        lap_ln_n = fd_laplacian(ln_n, dx)
    else:
        lap_ln_n = spectral_laplacian(ln_n, k2)

    rhs = -(8*np.pi*const.G/(const.c**2)) * rho

    diff = lap_ln_n - rhs
    relL2 = np.linalg.norm(diff.ravel())/max(1e-30, np.linalg.norm(rhs.ravel()))
    mae   = np.mean(np.abs(diff))
    rmse  = np.sqrt(np.mean(diff*diff))
    corr  = correlation(lap_ln_n, rhs)

    # Gradient for flux
    grad_ln_n = spectral_gradient(ln_n, kx, ky, kz)

    # Radii
    half = 0.5*L
    radii = np.linspace(par.r_min_frac*half, par.r_max_frac*half, par.n_sphere_radii)

    results = {}

    # Volume closure: ∫_V ∇² ln n dV + (8πG/c²) ∫_V ρ dV ≈ 0
    if par.use_volume_closure:
        I_lap = gauss_closure_volume(lap_ln_n, center, radii, par)
        xs = np.arange(N)*dx
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        Lbox = N*dx
        RX = periodic_delta(X, center[0], Lbox)
        RY = periodic_delta(Y, center[1], Lbox)
        RZ = periodic_delta(Z, center[2], Lbox)
        R2 = RX*RX + RY*RY + RZ*RZ
        vol = dx**3
        I_rho = np.array([rho[R2 <= (R*R)].sum()*vol for R in radii])
        closure_num = I_lap + (8*np.pi*const.G/(const.c**2))*I_rho
        results["relClose_vol"] = np.abs(closure_num)/np.maximum(1e-30, np.abs(I_lap))

    # Surface (flux) closure: ∮ ∇ln n · dS + (8πG/c²) ∫_V ρ dV ≈ 0
    if par.use_flux_surface:
        flux = gauss_closure_flux(grad_ln_n, center, radii, par)
        xs = np.arange(N)*dx
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        Lbox = N*dx
        RX = periodic_delta(X, center[0], Lbox)
        RY = periodic_delta(Y, center[1], Lbox)
        RZ = periodic_delta(Z, center[2], Lbox)
        R2 = RX*RX + RY*RY + RZ*RZ
        vol = dx**3
        I_rho = np.array([rho[R2 <= (R*R)].sum()*vol for R in radii])
        closure_num = flux + (8*np.pi*const.G/(const.c**2))*I_rho
        results["relClose_flux"] = np.abs(closure_num)/np.maximum(1e-30, np.abs(flux))
        results["_flux_raw"] = flux
        results["_I_rho"]    = I_rho

    metrics = {
        "RelL2": relL2,
        "MAE": mae,
        "RMSE": rmse,
        "Corr": corr,
        "N": N,
        "dx": dx,
        "g": g,
        "c_used": const.c,
        "G_true": const.G,
        "L": L,
        "interp": par.interp,
        "laplacian": par.laplacian,
    }
    fields = (rho, ln_n, lap_ln_n, rhs)
    aux    = {"center": center, "radii": radii}
    return metrics, results, fields, aux

# ------------------------ Run ------------------------
const = PhysConst()
par   = SimParam(
    N=128, dx=0.015625, g=0.5,
    use_flux_surface=True, use_volume_closure=True,
    n_sphere_radii=36, r_min_frac=0.06, r_max_frac=0.46,
    n_sphere_pts=8192,
    interp="cubic",        # "cubic" or "linear"
    laplacian="spectral"   # "spectral" or "fd"
)
metrics, results, fields, aux = run_gaussian_validation(const, par)
radii = aux["radii"]

# ------------------------ Plots ------------------------
if "relClose_vol" in results:
    plt.figure()
    plt.hist(results["relClose_vol"], bins=16)
    plt.title("Volume Closure Relative Error (v2.1)")
    plt.xlabel("relative error")
    plt.ylabel("count")
    plt.show()

if "relClose_flux" in results:
    plt.figure()
    plt.hist(results["relClose_flux"], bins=16)
    plt.title("Surface (Flux) Closure Relative Error (v2.1)")
    plt.xlabel("relative error")
    plt.ylabel("count")
    plt.show()

if "relClose_vol" in results:
    plt.figure()
    plt.plot(radii, results["relClose_vol"])
    plt.title("Volume Closure Relative Error vs Radius (v2.1)")
    plt.xlabel("radius")
    plt.ylabel("relative error")
    plt.show()

if "relClose_flux" in results:
    plt.figure()
    plt.plot(radii, results["relClose_flux"])
    plt.title("Surface (Flux) Closure Relative Error vs Radius (v2.1)")
    plt.xlabel("radius")
    plt.ylabel("relative error")
    plt.show()

# ------------------------ Optional: G_flux_est / c_flux_est ------------------------
G_flux_est = None
c_flux_est = None
if "_flux_raw" in results and "_I_rho" in results:
    F = results["_flux_raw"].astype(float)
    I = results["_I_rho"].astype(float)
    denom = float(np.dot(I, I))
    if denom > 1e-30:
        k_hat = - float(np.dot(F, I)) / denom
        G_flux_est = (const.c**2 / (8*np.pi)) * k_hat
        if k_hat > 1e-30:
            c_flux_est = float(np.sqrt(8*np.pi*const.G / k_hat))

# ------------------------ Final print summary ------------------------
print("=== Unified Field Equation Verification — Gaussian Mode (Final Summary, v2.1) ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

if "relClose_vol" in results:
    arr = np.array(results["relClose_vol"])
    print(f"[Volume closure] median={np.median(arr)*100:.3f}%  mean={np.mean(arr)*100:.3f}%  "
          f"[q1,q3]=[{np.quantile(arr,0.25)*100:.3f}%, {np.quantile(arr,0.75)*100:.3f}%]")

if "relClose_flux" in results:
    arr = np.array(results["relClose_flux"])
    print(f"[Flux closure]   median={np.median(arr)*100:.3f}%  mean={np.mean(arr)*100:.3f}%  "
          f"[q1,q3]=[{np.quantile(arr,0.25)*100:.3f}%, {np.quantile(arr,0.75)*100:.3f}%]")

if G_flux_est is not None:
    print(f"G_flux_est: {G_flux_est:.6e}")
if c_flux_est is not None:
    print(f"c_flux_est: {c_flux_est:.6e} m/s")

print("DONE v2.1.")
