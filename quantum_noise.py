import numpy as np
from scipy.integrate import simpson as simps  # 使用 SciPy 积分
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# === 物理常数与单位 ===
G_CODATA = 6.67430e-11
h_bar = 1.0545718e-34  # Planck 常数 / 2π
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# === 网格定义 ===
x = np.linspace(150, 250, 500)
t = np.linspace(4.7, 5.7, 100)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === collapse 场结构（添加量子噪声） ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0, quantum_noise=0.0):
    def pulse(x0, t0_local, amp=1.0):
        X, T = np.meshgrid(x, t)
        spatial = np.exp(-((X - x0)**2) / (2 * sigma_x**2))
        time = np.where(T >= t0_local, (T - t0_local)**k * np.exp(-((T - t0_local)**2) / (2 * sigma_t**2)), 0.0)
        return amp * spatial * time
    noise_scale = quantum_noise * h_bar / (sigma_x * sigma_t)
    noise = np.random.normal(0, noise_scale, size=(len(t), len(x)))
    kernel = np.outer(np.exp(-np.linspace(-2, 2, 5)**2), np.exp(-np.linspace(-2, 2, 5)**2))
    kernel /= np.sum(kernel)
    noise_smoothed = convolve2d(noise, kernel, mode='same', boundary='symm')
    base = pulse(195, t0) + pulse(205, t0 + 0.1, -0.7)
    return A * (base + noise_smoothed)

# === K(t) 计算 ===
def compute_K(H_t, phi_t):
    log_H = np.log(np.abs(H_t) + 1e-30)
    log_phi = np.log(np.abs(phi_t) + 1e-30)
    dlogH_dt = np.gradient(log_H, dt)
    dlogphi_dt = np.gradient(log_phi, dt)
    K_t = dlogH_dt / (dlogphi_dt + 1e-20)
    K_t = np.where(np.isfinite(K_t), K_t, 0)
    return K_t

# === G(t*) 计算核心 ===
def compute_G(phi2, A, sigma_t, t0, sigma_x, k, L_L, L_T, L_M, quantum_noise=0.0):
    delta_xt = delta_dag(x, t, A, sigma_x, sigma_t, k, t0, quantum_noise)
    H_t = np.array([simps(delta_xt[i]**2, x=x, dx=dx) for i in range(len(t))])
    integral = np.array([simps(delta_xt[i], x=x, dx=dx) for i in range(len(t))])
    phi_c_t = np.gradient(integral, dt)
    K_t = compute_K(H_t, integral)
    valid = np.where((H_t > 1e-40) & (np.abs(phi_c_t) > 1e-20))[0]
    if len(valid) == 0:
        return np.nan, None
    G_struct = phi2 / (H_t[valid] * phi_c_t[valid])
    G_phys = G_struct * (L_L**3) / (L_M * L_T**2)
    g_star = 0.5  # 渐近安全文献值
    quantum_scale = 1 + (quantum_noise * h_bar / (G_CODATA * L_M)) * (1 - np.exp(-quantum_noise / g_star))
    G_phys *= quantum_scale
    idx = np.argmin(np.abs(G_phys - G_CODATA))
    return G_phys[idx], {
        "G": G_phys[idx],
        "t*": t[valid[idx]],
        "H": H_t[valid[idx]],
        "phi_c": phi_c_t[valid[idx]],
        "K": K_t[valid[idx]],
        "K_t": K_t,
        "rel_error": abs(G_phys[idx] - G_CODATA) / G_CODATA * 100,
        "quantum_noise_level": quantum_noise * h_bar
    }

# === 默认参数 ===
base = dict(
    phi2=5e-30, A=0.1, sigma_t=0.4, t0=4.94,  # 折中 sigma_t
    sigma_x=8, k=5, L_L=L_L, L_T=L_T, L_M=L_M
)

# === 蒙特卡罗测试 ===
num_simulations = 50
quantum_noise_levels = [0.0, 1e-2, 0.1, 1.0, 10.0, 100.0]  # 增加高噪声

print("=== G(t*) 量子统一模拟测试（蒙特卡罗多组运行） ===")
for noise_level in quantum_noise_levels:
    G_values = []
    rel_errors = []
    K_t_values = []
    K_at_tstar = []
    print(f"\n--- 量子噪声水平: {noise_level * h_bar:.2e} ---")
    for _ in range(num_simulations):
        G_best, res = compute_G(**base, quantum_noise=noise_level)
        if res is not None:
            G_values.append(res['G'])
            rel_errors.append(res['rel_error'])
            K_t_values.append(res['K_t'])
            K_at_tstar.append(res['K'])
    if G_values:
        avg_G = np.mean(G_values)
        avg_error = np.mean(rel_errors)
        std_G = np.std(G_values)
        avg_K_at_tstar = np.mean(K_at_tstar)
        std_K_at_tstar = np.std(K_at_tstar)
        avg_K_t = np.mean(K_t_values, axis=0)
        std_K_t = np.std(K_t_values, axis=0)
        t0_idx = np.argmin(np.abs(t - base['t0']))
        print(f"平均 G(t*) = {avg_G:.6e}")
        print(f"平均相对误差 = {avg_error:.4f}%")
        print(f"G 标准差 = {std_G:.6e}（量子波动影响）")
        print(f"平均 K(t*) = {avg_K_at_tstar:.6e}")
        print(f"K(t*) 标准差 = {std_K_at_tstar:.6e}")
        print(f"平均 K(t) 样本（t0附近10个时间点, t[{t0_idx-5}:{t0_idx+5}]）: {avg_K_t[t0_idx-5:t0_idx+5]}")
        print(f"K(t) 标准差（t0附近10个时间点）: {std_K_t[t0_idx-5:t0_idx+5]}")
        print(f"参考 G_CODATA = {G_CODATA:.6e}")
        plt.figure(figsize=(8, 6))
        plt.hist(G_values, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"G(t*) 分布 (噪声: {noise_level * h_bar:.2e})")
        plt.xlabel("G(t*) 值")
        plt.ylabel("频率")
        plt.axvline(x=G_CODATA, color='red', linestyle='--', label=f'CODATA: {G_CODATA:.6e}')
        plt.legend()
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.plot(t, avg_K_t, label=f'平均 K(t) (噪声: {noise_level * h_bar:.2e})')
        plt.fill_between(t, avg_K_t - std_K_t, avg_K_t + std_K_t, alpha=0.2, color='skyblue', label='标准差范围')
        plt.axvline(x=base['t0'], color='red', linestyle='--', label=f't0 = {base["t0"]}')
        plt.xlabel('时间 t')
        plt.ylabel('K(t)')
        plt.title(f'K(t) 随时间变化 (噪声: {noise_level * h_bar:.2e})')
        plt.legend()
        plt.show()

print("运行说明: 使用 sigma_t=0.4，g_star=0.5，SciPy simpson 积分，测试更高噪声。")
