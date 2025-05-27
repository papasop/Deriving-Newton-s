import numpy as np
from numpy import trapz as simps

# === 基本物理常数 ===
G_CODATA = 6.67430e-11

# === 默认单位系统 ===
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# === 空间-时间网格 ===
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 200)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === collapse 场定义 ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0):
    def pulse(x0, t0_local, amp=1.0):
        if t < t0_local: return 0.0
        spatial = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        time_term = (t - t0_local)**k * np.exp(-((t - t0_local)**2) / (2 * sigma_t**2))
        return amp * spatial * time_term
    return A * (pulse(195, t0, 1.0) + pulse(205, t0 + 0.1, -0.7))

# === G 计算函数 ===
def compute_G(phi2, A, sigma_t, t0, sigma_x, k, L_L, L_T, L_M):
    delta_xt = np.array([[delta_dag(xi, tj, A, sigma_x, sigma_t, k, t0) for xi in x] for tj in t])
    H_t = np.array([simps(delta_xt[i]**2, x) for i in range(len(t))])
    integral = np.array([simps(delta_xt[i], x) for i in range(len(t))])
    phi_c_t = np.gradient(integral, dt)
    valid = np.where((H_t > 1e-40) & (np.abs(phi_c_t) > 1e-20))[0]
    if len(valid) == 0:
        return np.nan
    G_struct = phi2 / (H_t[valid] * phi_c_t[valid])
    G_phys = G_struct * (L_L**3) / (L_M * L_T**2)
    best = np.argmin(np.abs(G_phys - G_CODATA))
    return G_phys[best]

# === 基准参数 ===
base = {
    "phi2": 5e-30,
    "A": 0.1,
    "sigma_t": 0.3,
    "t0": 4.94,
    "sigma_x": 8,
    "k": 5,
    "L_L": 1e-15,
    "L_T": 1e-20,
    "L_M": 0.063826
}

# === 鲁棒测试函数 ===
def scan_param(param, values, base):
    errors = []
    for v in values:
        test = base.copy()
        test[param] = v
        G = compute_G(**test)
        if np.isnan(G): continue
        err = abs(G - G_CODATA) / G_CODATA * 100
        errors.append(err)
    return np.array(errors)

# === 参数扫描列表 ===
param_ranges = {
    "phi2":     np.linspace(4.5e-30, 5.5e-30, 5),
    "A":        np.linspace(0.09, 0.11, 5),
    "sigma_t":  np.linspace(0.25, 0.35, 5),
    "t0":       np.linspace(4.938, 4.942, 5),
    "sigma_x":  np.linspace(6, 10, 5),
    "k":        [3, 4, 5, 6, 7],
    "L_L":      np.logspace(-15.5, -14.5, 5),
    "L_T":      np.logspace(-20.5, -19.5, 5),
    "L_M":      np.linspace(0.05, 0.08, 5)
}

# === 执行所有参数扫描 ===
print("=== G(t*) 鲁棒性敏感度测试 ===")
scores = []
for param, vals in param_ranges.items():
    errs = scan_param(param, vals, base)
    if len(errs) > 0:
        score = np.std(errs)
        print(f"{param:<10s}  Std Dev (误差波动) = {score:.4f}%")
        scores.append((param, score))

# === 排名输出 ===
scores.sort(key=lambda x: -x[1])
print("\n=== 参数敏感度排名（越高越敏感） ===")
for i, (param, score) in enumerate(scores, 1):
    print(f"{i:>2}. {param:<10s}  → 敏感度评分: {score:.4f}%")
