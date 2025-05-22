import numpy as np
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

# 参数设置
A = 0.12
sigma_x = 4.0
sigma_t = 0.2
x0 = 200
t0 = 5.0
phi_e2 = 1.73e-36
noise_levels = [0.01, 0.05, 0.1]  # 噪声幅度：1%, 5%, 10%

# 定义空间与时间域
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 500)
dx = x[1] - x[0]
dt = t[1] - t[0]

# 计算 δ(x,t) 带噪声的函数
def delta_xt(x_val, t_val, noise_level):
    spatial = np.exp(-((x_val - x0) ** 2) / (2 * sigma_x ** 2))
    temporal = np.exp(-((t_val - t0) ** 2) / (2 * sigma_t ** 2))
    slope = (t_val - t0)
    base = A * spatial * slope * temporal
    noise = np.random.normal(0, noise_level * A, size=base.shape)
    return base + noise

# 测试不同噪声幅度的鲁棒性
for nl in noise_levels:
    delta = np.array([[delta_xt(xi, tj, nl) for xi in x] for tj in t])
    H_t = np.array([simpson(row**2, x, dx=dx) for row in delta])
    I_t = np.array([simpson(row, x, dx=dx) for row in delta])
    dI_dt = np.gradient(I_t, dt)
    phi_c_t = gaussian_filter1d(np.abs(dI_dt), sigma=1)
    alpha_t = phi_e2 / H_t
    phiG_t = phi_e2 / (H_t * phi_c_t)
    F_t = alpha_t - phiG_t

    idx_min = np.argmin(np.abs(F_t))

    print(f"--- Noise Level: {int(nl * 100)}% ---")
    print(f"t* (min F): {t[idx_min]:.3f} s")
    print(f"φc(t*): {phi_c_t[idx_min]:.6f}")
    print(f"F(t*): {F_t[idx_min]:.3e}")
    print(f"Relative Error: {np.abs(F_t[idx_min] / alpha_t[idx_min]) * 100:.4f}%")
    print()
