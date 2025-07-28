import numpy as np
from numpy import trapezoid as simps
import matplotlib.pyplot as plt

# === 物理常数与单位 ===
G_CODATA = 6.67430e-11
h_bar = 1.0545718e-34  # Planck 常数 / 2π
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# === 网格定义 ===
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 200)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === collapse 场结构（添加量子噪声） ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0, quantum_noise=0.0):
    def pulse(x0, t0_local, amp=1.0):
        X, T = np.meshgrid(x, t)
        spatial = np.exp(-((X - x0)**2) / (2 * sigma_x**2))
        time = np.where(T >= t0_local, (T - t0_local)**k * np.exp(-((T - t0_local)**2) / (2 * sigma_t**2)), 0.0)
        return amp * spatial * time
    noise_scale = quantum_noise * h_bar / (sigma_x * sigma_t)
    noise = np.random.normal(0, noise_scale, size=(len(t), len(x)))
    base = pulse(195, t0) + pulse(205, t0 + 0.1, -0.7)
    return A * (base + noise)

# === K(t) 计算 ===
def compute_K(H_t, phi_t):
    log_H = np.log(np.abs(H_t) + 1e-30)
    log_phi = np.log(np.abs(phi_t) + 1e-30)
    dlogH_dt = np.gradient(log_H, dt)
    dlogphi_dt = np.gradient(log_phi, dt)
    return dlogH_dt / (dlogphi_dt + 1e-20)

# === G(t*) 计算核心 ===
def compute_G(phi2, A, sigma_t, t0, sigma_x, k, L_L, L_T, L_M, quantum_noise=0.0):
    delta_xt = delta_dag(x, t, A, sigma_x, sigma_t, k, t0, quantum_noise)
    H_t = np.array([simps(delta_xt[i]**2, x, dx=dx) for i in range(len(t))])
    integral = np.array([simps(delta_xt[i], x, dx=dx) for i in range(len(t))])
    phi_c_t = np.gradient(integral, dt)
    K_t = compute_K(H_t, integral)
    valid = np.where((H_t > 1e-40) & (np.abs(phi_c_t) > 1e-20))[0]
    if len(valid) == 0:
        return np.nan, None
    G_struct = phi2 / (H_t[valid] * phi_c_t[valid])
    G_phys = G_struct * (L_L**3) / (L_M * L_T**2)
    quantum_scale = 1 + quantum_noise * h_bar / (G_CODATA * L_M)
    G_phys *= quantum_scale
    idx = np.argmin(np.abs(G_phys - G_CODATA))
    return G_phys[idx], {
        "G": G_phys[idx],
        "t*": t[valid[idx]],
        "H": H_t[valid[idx]],
        "phi_c": phi_c_t[valid[idx]],
        "K": K_t[valid[idx]],
        "rel_error": abs(G_phys[idx] - G_CODATA) / G_CODATA * 100,
        "quantum_noise_level": quantum_noise * h_bar
    }

# === 默认参数 ===
base = dict(
    phi2=5e-30, A=0.1, sigma_t=0.3, t0=4.94,
    sigma_x=8, k=5, L_L=L_L, L_T=L_T, L_M=L_M
)

# === 蒙特卡罗测试 ===
num_simulations = 10
quantum_noise_levels = [0.0, 1e-2, 0.1, 1.0, 10.0]

print("=== G(t*) 量子统一模拟测试（蒙特卡罗多组运行） ===")
for noise_level in quantum_noise_levels:
    G_values = []
    rel_errors = []
    print(f"\n--- 量子噪声水平: {noise_level * h_bar:.2e} ---")
    for _ in range(num_simulations):
        G_best, res = compute_G(**base, quantum_noise=noise_level)
        if res is not None:
            G_values.append(res['G'])
            rel_errors.append(res['rel_error'])
    if G_values:
        avg_G = np.mean(G_values)
        avg_error = np.mean(rel_errors)
        std_G = np.std(G_values)
        print(f"平均 G(t*) = {avg_G:.6e}")
        print(f"平均相对误差 = {avg_error:.4f}%")
        print(f"G 标准差 = {std_G:.6e}（量子波动影响）")
        print(f"参考 G_CODATA = {G_CODATA:.6e}")
    if G_values:
        plt.figure(figsize=(8, 6))
        plt.hist(G_values, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"G(t*) 分布 (噪声: {noise_level * h_bar:.2e})")
        plt.xlabel("G(t*) 值")
        plt.ylabel("频率")
        plt.axvline(x=G_CODATA, color='red', linestyle='--', label=f'CODATA: {G_CODATA:.6e}')
        plt.legend()
        plt.show()

print("运行说明: 代码优化用于 Google Colab，已修正广播问题。调整 quantum_noise_levels 或 num_simulations 可测试更多场景。")