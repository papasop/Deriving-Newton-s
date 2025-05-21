import numpy as np
from scipy.ndimage import gaussian_filter1d

# === 模型结构参数 ===
A = 0.01
sigma_x = 9
sigma_t = 1.0
lambda_L = 1e-15   # 结构长度单位 [m]
lambda_T = 1e-20   # 结构时间单位 [s]
G_target = 6.67430e-11

# === 构建结构网格 ===
x = np.linspace(100, 300, 200)
t = np.linspace(0, 10, 100)
X, T = np.meshgrid(x, t)

# === 构造 δ(x,t)（高斯 × 时间斜率项）===
def delta_gaussian(X, T, A, x0, t0, sigma_x, sigma_t):
    return A * np.exp(-((X - x0)**2) / (2 * sigma_x**2)) * \
           (T - t0) * np.exp(-((T - t0)**2) / (2 * sigma_t**2))

x0 = 200
t0 = 5.0
delta = delta_gaussian(X, T, A, x0, t0, sigma_x, sigma_t)
print("δ(x,t) min/max:", np.min(delta), np.max(delta))

# === 结构张力计算 ===
dx = (x[1] - x[0]) * lambda_L
dt = (t[1] - t[0]) * lambda_T
H_t = np.sum(delta**2, axis=1) * dx
delta_integrated = np.sum(delta, axis=1) * dx
phi_c = np.gradient(delta_integrated, dt)
phi_c = gaussian_filter1d(phi_c, sigma=3)
phi_c = np.clip(phi_c, 1e-5, 1e3)
H_t = np.clip(H_t, 1e-25, 1e3)

# === 联合优化扫描区间 ===
phi_e_squared_range = np.linspace(1.68e-36, 1.73e-36, 50)
log_lambda_M_range = np.linspace(-1.19505, -1.19495, 500)

# === 联合最优值存储 ===
best_error = float('inf')
best_phi_e_squared = None
best_lambda_M = None
best_G_mean = None

print("\n===== 联合优化扫描中... =====")
for phi_e in phi_e_squared_range:
    for log_lambda_M in log_lambda_M_range:
        lambda_M = 10 ** log_lambda_M
        phi_e_scaled = phi_e * lambda_M**3
        G_t = phi_e_scaled / (H_t * phi_c)
        G_t = G_t[~np.isnan(G_t)]
        G_mean = np.mean(G_t)
        error = 100 * abs(G_mean - G_target) / G_target if G_mean != 0 else np.nan

        if error < best_error:
            best_error = error
            best_phi_e_squared = phi_e
            best_lambda_M = lambda_M
            best_G_mean = G_mean

# === 打印最终结果 ===
print("\n===== 最佳拟合结果 =====")
print(f"φ_e² = {best_phi_e_squared:.3e}")
print(f"λ_M = {best_lambda_M:.6f} kg")
print(f"mean(G(t)) = {best_G_mean:.5e}")
print(f"G_target   = {G_target:.5e}")
print(f"误差比例    = {best_error:.2f}%")
