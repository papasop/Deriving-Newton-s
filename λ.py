import numpy as np
from numpy import trapz as simps

# === Planck 单位测试用物理常数 ===
G_CODATA = 6.67430e-11
c = 299792458
hbar = 1.054571817e-34

# === 原始单位（以 L_P, T_P, M_P 构建） ===
L_L_base = 1e-15
L_T_base = 1e-20
L_M_base = 0.063826

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

# === 结构时间比率 K(t) 计算 ===
def compute_K(H_t, phi_t):
    log_H = np.log(np.abs(H_t) + 1e-30)
    log_phi = np.log(np.abs(phi_t) + 1e-30)
    dlogH_dt = np.gradient(log_H, dt)
    dlogphi_dt = np.gradient(log_phi, dt)
    return dlogH_dt / (dlogphi_dt + 1e-20)

# === G(t*) 计算核心 ===
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

# === 测试 λ 对称扰动（单位伸缩） ===
def lambda_symmetry_test(lambda_vals):
    print("=== λ 扰动对称性测试：G 是否不变 ===\n")
    for λ in lambda_vals:
        L_L = λ * L_L_base
        L_T = λ * L_T_base
        L_M = λ**3 * L_M_base  # 质量单位立方变换
        G_result, res = compute_G(
            phi2=5e-30, A=0.1, sigma_t=0.3, t0=4.94, sigma_x=8, k=5,
            L_L=L_L, L_T=L_T, L_M=L_M
        )
        print(f"λ = {λ:.3f}  →  G = {res['G']:.6e}, 误差 = {res['rel_error']:.4f}%,  K = {res['K']:.4f}")

# === 运行测试 ===
lambda_vals = [0.1, 0.5, 1.0, 2.0, 10.0]
lambda_symmetry_test(lambda_vals)
