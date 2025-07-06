import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors

# 基础常数
H_t_star = 1.612e-47
phi_c_t_star = 3.1e-6
G_CODATA = 6.67430e-11

# 模型参数（单位系统）
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# G结构模型
def compute_G(phi2):
    G_struct = phi2 / (H_t_star * phi_c_t_star)
    unit_factor = (L_L**3) / (L_M * L_T**2)
    return G_struct * unit_factor

# 基础 φ² 线性空间（log均匀分布）
phi2_base = np.logspace(-32, -27, 50)

# 扰动配置
perturbations = {
    "±10%": 0.10,
    "±20%": 0.20,
    "±50%": 0.50,
    "log ±0.1": "log",
    "noise": "noise"
}

# 初始化图像
plt.figure(figsize=(10, 6))

# 原始线
G_base = compute_G(phi2_base)
plt.plot(phi2_base, G_base, 'k-', label="Base G(φ²)", linewidth=2)

# 多种扰动类型绘图
colors = list(mcolors.TABLEAU_COLORS.values())
for i, (label, perturb) in enumerate(perturbations.items()):
    if isinstance(perturb, float):
        # ±比例扰动
        phi2_plus = phi2_base * (1 + perturb)
        phi2_minus = phi2_base * (1 - perturb)
        G_plus = compute_G(phi2_plus)
        G_minus = compute_G(phi2_minus)
        plt.plot(phi2_plus, G_plus, linestyle='--', color=colors[i], label=f"G(φ² × (1+{int(perturb*100)}%))")
        plt.plot(phi2_minus, G_minus, linestyle=':', color=colors[i], label=f"G(φ² × (1-{int(perturb*100)}%))")
    elif perturb == "log":
        # log ±0.1 扰动
        phi2_log_plus = 10**(np.log10(phi2_base) + 0.1)
        phi2_log_minus = 10**(np.log10(phi2_base) - 0.1)
        G_log_plus = compute_G(phi2_log_plus)
        G_log_minus = compute_G(phi2_log_minus)
        plt.plot(phi2_log_plus, G_log_plus, linestyle='--', color=colors[i], label="G(log₁₀(φ²)+0.1)")
        plt.plot(phi2_log_minus, G_log_minus, linestyle=':', color=colors[i], label="G(log₁₀(φ²)-0.1)")
    elif perturb == "noise":
        # 随机扰动 φ² × (1 + ε), ε ~ N(0, σ)
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, size=phi2_base.shape)  # σ = 10%
        phi2_noise = phi2_base * (1 + noise)
        G_noise = compute_G(phi2_noise)
        plt.plot(phi2_noise, G_noise, linestyle='-.', color=colors[i], label="G(φ² + noise)")

# 参考线
plt.axhline(G_CODATA, color='red', linestyle='--', label="G_CODATA")

# 图像设置
plt.xscale("log")
plt.yscale("log")
plt.xlabel("φ² (log scale)", fontsize=12)
plt.ylabel("G(φ²) (log scale)", fontsize=12)
plt.title("G(φ²) 扰动稳定性验证", fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.show()
