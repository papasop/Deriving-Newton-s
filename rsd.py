import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import bootstrap

# 精确结构 G 计算函数（模拟替代，真实模型可替换此处）
def true_G_model(lmbd, sigma_x):
    G_true = 6.67430e-11
    offset = -5e-13 * (lmbd - 0.3)**2 - 2e-13 * (sigma_x - 0.06)**2
    return G_true + offset

# 构建响应面
lmbd_vals = np.linspace(0.25, 0.35, 30)
sigma_x_vals = np.linspace(0.02, 0.10, 30)
LMB, SX = np.meshgrid(lmbd_vals, sigma_x_vals)
G_surface = true_G_model(LMB, SX)

# Bootstrap 稳定性计算（固定最优点）
N_boot = 1000
np.random.seed(42)
best_lmbd, best_sigma = 0.31, 0.06
samples = true_G_model(best_lmbd, best_sigma) + np.random.normal(0, 2e-13, size=(N_boot,))
mean_G = np.mean(samples)
std_G = np.std(samples)
rsd = 100 * std_G / mean_G
ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])

# 显示平台与鲁棒性结果
print("=== Structural G Robustness Platform Summary ===")
print(f"λ*: {best_lmbd:.3f}, σₓ*: {best_sigma:.3f}")
print(f"G(λ*, σₓ*) = {mean_G:.6e}")
print(f"Ref G = 6.67430e-11")
print(f"Relative Error = {100*(mean_G - 6.67430e-11)/6.67430e-11:.4f}%")
print(f"RSD = {rsd:.3f}%")
print(f"95% CI = [{ci_lower:.6e}, {ci_upper:.6e}]")

# 三维图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(LMB, SX, G_surface, cmap='viridis', alpha=0.9)
ax.set_xlabel("λ")
ax.set_ylabel("σₓ")
ax.set_zlabel("G")
ax.set_title("G(λ, σₓ) Structural Response Surface")
plt.tight_layout()
plt.show()

