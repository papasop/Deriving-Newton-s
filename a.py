import numpy as np

# === 模型参数 ===
A = 1.204e-6         # 振幅（已优化）
sigma_x = 55.51      # 空间高斯宽度（已优化）
sigma_t = 1.0        # 时间宽度
x0 = 200             # 空间中心
t0 = 5.0             # 时间中心

lambda_L = 1e-15     # 结构单位：米
lambda_T = 1e-20     # 结构单位：秒
phi_e_squared = 1.73e-36  # 模态能量密度

# === 网格设置 ===
x = np.linspace(100, 300, 200)
t = np.linspace(0, 10, 100)
X, T = np.meshgrid(x, t)

# === 构造 δ(x,t) 非-collapse 高斯结构（无时间斜率项）===
def delta_xt(X, T, A, x0, t0, sigma_x, sigma_t):
    return A * np.exp(-((X - x0)**2) / (2 * sigma_x**2)) * \
           np.exp(-((T - t0)**2) / (2 * sigma_t**2))

delta = delta_xt(X, T, A, x0, t0, sigma_x, sigma_t)

# === 结构熵 H(t) ===
dx = (x[1] - x[0]) * lambda_L
H_t = np.sum(delta**2, axis=1) * dx
H_t = np.clip(H_t, 1e-40, 1e-20)

# === 计算 α(t) = φ_e² / H(t) ===
alpha_t = phi_e_squared / H_t
target_alpha = 1 / 137.036

# === 找出最接近 α = 1/137 的时间点 ===
error = np.abs(alpha_t - target_alpha)
idx = np.argmin(error)
best_alpha = alpha_t[idx]
best_inv_alpha = 1 / best_alpha
best_H = H_t[idx]
best_time = t[idx]
rel_error = 100 * abs(best_alpha - target_alpha) / target_alpha

# === 输出结果 ===
print("===== 结构 α ≈ 1/137 拟合结果 =====")
print(f"最优时间点 t* = {best_time:.3f} s")
print(f"H(t*) = {best_H:.3e}")
print(f"φ_e² = {phi_e_squared:.3e}")
print(f"α(t*) = {best_alpha:.6e}")
print(f"1/α(t*) = {best_inv_alpha:.3f}")
print(f"目标 1/α = {1/target_alpha:.3f}")
print(f"相对误差 = {rel_error:.3f}%")
