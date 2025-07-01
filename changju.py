import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# === Grid Setup ===
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 400)
dt = t[1] - t[0]

# === Structural Field Generator ===
def generate_delta_xt(x, t, sigma_x, N):
    x_centers = np.linspace(180, 220, N)
    psi_total = np.zeros_like(x)
    for x0 in x_centers:
        psi_total += np.exp(-((x - x0)**2) / (2 * sigma_x**2))
    psi_total /= np.sqrt(trapezoid(psi_total**2, x))
    envelope = 1 / np.cosh(3 * (t - 5.0))**2
    return envelope[:, None] * psi_total[None, :]

# === Effective Constant Evaluation (改进版) ===
def evaluate_constants(delta_xt):
    phi_eff = trapezoid(delta_xt, x, axis=1)
    dphi_eff = np.gradient(phi_eff, dt)
    H_t = trapezoid(delta_xt**2, x, axis=1)

    phi_eff_safe = np.maximum(phi_eff, 1e-12)
    H_t_safe = np.maximum(H_t, 1e-30)

    G_eff_t = 1 / (8 * np.pi * H_t_safe**2 * phi_eff_safe**2)
    alpha_eff_t = dphi_eff**2 / (phi_eff_safe**2)
    hbar_eff_t = np.sqrt(H_t_safe) * dt

    Q_eff_t = G_eff_t * alpha_eff_t * hbar_eff_t
    return G_eff_t, alpha_eff_t, hbar_eff_t, Q_eff_t

# === Run Test ===
sigma_x = 0.16
N = 400
delta_xt = generate_delta_xt(x, t, sigma_x, N)
G_eff_t, alpha_eff_t, hbar_eff_t, Q_eff_t = evaluate_constants(delta_xt)

# === 相关性矩阵计算 ===
data_matrix = np.vstack([G_eff_t, alpha_eff_t, hbar_eff_t])
corr_matrix = np.corrcoef(data_matrix)
print("三个结构常数的相关系数矩阵：")
print(corr_matrix)

# === 绘图对比三个常数的时间序列 ===
plt.figure(figsize=(14,6))
plt.plot(t, G_eff_t / np.max(G_eff_t), label="G_eff (归一化)")
plt.plot(t, alpha_eff_t / np.max(alpha_eff_t), label="alpha_eff (归一化)")
plt.plot(t, hbar_eff_t / np.max(hbar_eff_t), label="hbar_eff (归一化)")
plt.xlabel("时间 (s)")
plt.ylabel("归一化结构常数")
plt.title("结构常数时间序列对比（归一化处理）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
