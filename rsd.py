import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bootstrap

# 参数范围
lambda_range = np.linspace(0.1, 0.5, 20)
sigma_x_range = np.linspace(0.5, 1.5, 20)

# 创建 G 响应面，使用自适应 phi² ∝ λ^{2 + 0.2λ}
G_surface_eps = np.zeros((len(lambda_range), len(sigma_x_range)))

for i, lam in enumerate(lambda_range):
    for j, sigma_x in enumerate(sigma_x_range):
        A = 0.1
        sigma = sigma_x
        t0 = 5.0
        k = 1.0

        # phi² ∝ λ^{2 + ε(λ)}
        phi_peak = A
        phi_squared = phi_peak**2 * lam**(2 + 0.2 * lam)
        H = phi_squared
        phi_c = lam
        G_estimate = H / phi_c**2 if phi_c != 0 else np.nan
        G_surface_eps[i, j] = G_estimate

# 可视化 G(λ, σ_x) 三维响应面
from mpl_toolkits.mplot3d import Axes3D

LAMBDA, SIGMA_X = np.meshgrid(lambda_range, sigma_x_range, indexing='ij')
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(LAMBDA, SIGMA_X, G_surface_eps, cmap='viridis')
ax.set_xlabel('λ')
ax.set_ylabel('σₓ')
ax.set_zlabel('G')
ax.set_title('G(λ, σₓ) Structural Response Surface')
plt.show()

# Bootstrap 稳定性分析（σₓ ≈ 1.0）
sigma_index = np.argmin(np.abs(sigma_x_range - 1.0))
G_values_slice = G_surface_eps[:, sigma_index]
valid_mask = ~np.isnan(G_values_slice)
valid_G = G_values_slice[valid_mask]
valid_lambda = lambda_range[valid_mask]

res = bootstrap((valid_G,), np.mean, confidence_level=0.95, n_resamples=500, method='basic')
mean_G = np.mean(valid_G)
std_G = np.std(valid_G)
rsd_G = std_G / mean_G if mean_G != 0 else np.nan

# 输出表格与总结
df_stats = pd.DataFrame({
    "Lambda": valid_lambda,
    "G_value": valid_G
})

print("🔍 Bootstrap Stability Summary:")
print(f"Mean G: {mean_G:.6f}")
print(f"Std G: {std_G:.6e}")
print(f"RSD: {rsd_G:.3%}")
print(f"95% CI: [{res.confidence_interval.low:.6f}, {res.confidence_interval.high:.6f}]")
print(f"Platform Width: {valid_lambda[-1] - valid_lambda[0]:.3f}")

df_stats.head()
