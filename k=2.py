import numpy as np
from numpy import trapz as simps

# === 物理常数与单位 ===
G_CODATA = 6.67430e-11
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# === 网格定义 ===
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 200)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === collapse 场结构 ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0):
    def pulse(x0, t0_local, amp=1.0):
        if t < t0_local: return 0.0
        spatial = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        time = (t - t0_local)**k * np.exp(-((t - t0_local)**2) / (2 * sigma_t**2))
        return amp * spatial * time
    return A * (pulse(195, t0) + pulse(205, t0 + 0.1, -0.7))

# === K(t) 计算 ===
def compute_K(H_t, phi_t):
    log_H = np.log(np.abs(H_t) + 1e-30)
    log_phi = np.log(np.abs(phi_t) + 1e-30)
    dlogH_dt = np.gradient(log_H, dt)
    dlogphi_dt = np.gradient(log_phi, dt)
    return dlogH_dt / (dlogphi_dt + 1e-20)

# === G(t*) 计算核心（插入 K 分析） ===
def compute_G(phi2, A, sigma_t, t0, sigma_x, k, L_L, L_T, L_M):
    delta_xt = np.array([[delta_dag(xi, tj, A, sigma_x, sigma_t, k, t0) for xi in x] for tj in t])
    H_t = np.array([simps(delta_xt[i]**2, x) for i in range(len(t))])
    integral = np.array([simps(delta_xt[i], x) for i in range(len(t))])
    phi_c_t = np.gradient(integral, dt)
    K_t = compute_K(H_t, integral)
    valid = np.where((H_t > 1e-40) & (np.abs(phi_c_t) > 1e-20))[0]
    if len(valid) == 0:
        return np.nan, None
    G_struct = phi2 / (H_t[valid] * phi_c_t[valid])
    G_phys = G_struct * (L_L**3) / (L_M * L_T**2)
    idx = np.argmin(np.abs(G_phys - G_CODATA))
    return G_phys[idx], {
        "G": G_phys[idx],
        "t*": t[valid[idx]],
        "H": H_t[valid[idx]],
        "phi_c": phi_c_t[valid[idx]],
        "K": K_t[valid[idx]],
        "rel_error": abs(G_phys[idx] - G_CODATA) / G_CODATA * 100
    }

# === 默认参数 ===
base = dict(
    phi2=5e-30, A=0.1, sigma_t=0.3, t0=4.94,
    sigma_x=8, k=5, L_L=L_L, L_T=L_T, L_M=L_M
)

# === 输出结构预测值（含 K 分析） ===
print("=== G(t*) 精确结构预测结果（含 K 分析） ===")
G_best, res = compute_G(**base)
print(f"G(t*)        = {res['G']:.6e}")
print(f"参考值       = {G_CODATA:.6e}")
print(f"相对误差     = {res['rel_error']:.4f}%")
print(f"共振时间 t*  = {res['t*']:.3f} s")
print(f"φ_c(t*)      = {res['phi_c']:.6e}")
print(f"H(t*)        = {res['H']:.6e}")
print(f"K(t*)        = {res['K']:.4f}")
