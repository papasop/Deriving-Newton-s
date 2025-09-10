import numpy as np
import matplotlib.pyplot as plt

# ----- 参数（可调）-----
sigma = 0.10          # 径向速率 σ
omega = 0.03          # 旋转速率 ω（取小一些→近锁定，θ变化慢）
x0, y0 = 1.0, 0.6     # 初始点（避免 x 或 y ≈ 0）
T, dt = 200.0, 0.05
t = np.arange(0, T, dt)
g_target = 0.5        # 你论文中的目标 g，用来对照

# ----- 群表示动力学 -----
x = np.zeros_like(t); y = np.zeros_like(t)
x[0], y[0] = x0, y0
for i in range(1, len(t)):
    dx = sigma * x[i-1] - omega * y[i-1]
    dy = omega * x[i-1] + sigma * y[i-1]
    x[i] = x[i-1] + dx * dt
    y[i] = y[i-1] + dy * dt

# θ、扇区与 K 的稳健计算
theta = np.arctan2(y, x)
eps = 1e-6
den = (sigma - omega * np.tan(theta))
mask = (np.abs(np.sin(theta)*np.cos(theta)) > eps) & (np.abs(den) > eps)

K = np.empty_like(t); K[:] = np.nan
K[mask] = (sigma + omega/np.tan(theta[mask])) / den[mask]   # K = (σ+ω cotθ)/(σ-ω tanθ)
g_from_K = np.empty_like(t); g_from_K[:] = np.nan
g_from_K[mask] = 1.0 / K[mask]

# 去除极端离群：用百分位截尾
valid = np.isfinite(g_from_K)
g_vals = g_from_K[valid]
if g_vals.size == 0:
    print("No valid samples after masking.")
else:
    lo, hi = np.percentile(g_vals, [5, 95])
    robust = g_vals[(g_vals >= lo) & (g_vals <= hi)]

    # 统计
    frac_valid = np.sum(valid) / len(t)
    print("=== Structure–Phase Reciprocity (robust) ===")
    print(f"valid fraction: {frac_valid*100:.1f}%")
    print(f"g = 1/K   mean: {np.mean(robust):.4f}   median: {np.median(robust):.4f}   IQR: {np.percentile(robust,75)-np.percentile(robust,25):.4f}")
    # 与目标 g 的相关/偏差（把目标视为常数向量）
    if robust.size > 3:
        corr = np.corrcoef(robust, np.full_like(robust, g_target))[0,1]
        mae = np.mean(np.abs(robust - g_target))
        print(f"corr(g_target=0.5, 1/K [robust]) = {corr:.3f} (常数对照的相关性仅供参考)")
        print(f"MAE(1/K, 0.5)  [robust] = {mae:.4f}")

    # 简单“近锁定”时间窗（θ 变化小的子区间）来看中位数是否更靠近常数
    # 取前 1/4 段作为例子
    m = int(0.25 * len(t))
    sub = g_from_K[:m]
    sub = sub[np.isfinite(sub)]
    sub = sub[(sub >= lo) & (sub <= hi)]
    if sub.size > 3:
        print(f"[window head] median(1/K) = {np.median(sub):.4f}, mean = {np.mean(sub):.4f}")

# 画图
plt.figure(figsize=(10,5))
plt.plot(t[valid], g_from_K[valid], '.', ms=2, alpha=0.4, label='g = 1/K (masked)')
plt.hlines(g_target, t[0], t[-1], colors='r', linestyles='--', label='g target = 0.5')
plt.xlabel('t'); plt.ylabel('g(t)')
plt.title('Structure–Phase Reciprocity: g vs 1/K (masked & robust)')
plt.legend(); plt.grid(True)
plt.show()
