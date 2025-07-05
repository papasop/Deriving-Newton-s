import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# === 参数设定（归一化单位）===
params = dict(
    phi2=1.0,
    A=0.5,
    sigma_t=0.1,
    t0=0.48,
    sigma_x=0.05,
    k=5
)

# === delta_dag 函数定义 ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0):
    return A * (-(x**2) + 1) ** k * np.exp(-((t - t0) ** 2) / (2 * sigma_t ** 2))

# === 网格设置 ===
x = np.linspace(-1, 1, 500)
t = np.linspace(0, 1, 500)
dt = t[1] - t[0]

# === 计算 δ(x,t) ===
delta_xt = np.array([[delta_dag(xi, tj, params["A"], params["sigma_x"], params["sigma_t"], params["k"], params["t0"])
                      for xi in x] for tj in t])

# === 计算结构熵 H(t) ===
H_t = np.array([trapezoid(delta_xt[i]**2, x) for i in range(len(t))])

# === 计算 φ_c(t) 与 K(t) ===
integral = np.array([trapezoid(delta_xt[i], x) for i in range(len(t))])
phi_c_t = np.gradient(integral, dt)
K_t = phi_c_t**2 / H_t

# === 寻找 K=2 的共振点 t* ===
target_K = 2.0
idx_star = np.argmin(np.abs(K_t - target_K))
t_star = t[idx_star]
phi_c_star = phi_c_t[idx_star]
H_star = H_t[idx_star]
K_star = K_t[idx_star]

# === 计算 G_struct（归一化）===
if H_star != 0:
    G_struct = (phi_c_star ** 2) / H_star
    print("=== Normalized Structural G Prediction via K=2 ===")
    print(f"G_struct     = {G_struct:.4f}")
    print(f"t*           = {t_star:.3f}")
    print(f"H(t*)        = {H_star:.4f}")
    print(f"φ_c(t*)      = {phi_c_star:.4f}")
    print(f"K(t*)        = {K_star:.4f}")
else:
    print("❌ Failed to compute valid G_struct.")

# === 绘制 K(t) 曲线 ===
plt.figure(figsize=(10, 4))
plt.plot(t, K_t, label='K(t)', color='orange')
plt.axhline(2.0, color='red', linestyle='--', label='K = 2 Target')
plt.axvline(t_star, color='green', linestyle='--', label='t* (resonance)')
plt.title("K(t) Over Time")
plt.xlabel("t (normalized)")
plt.ylabel("K(t)")
plt.legend()
plt.grid(True)
plt.show()
