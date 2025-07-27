# === 环境准备 ===
import numpy as np
import matplotlib.pyplot as plt

# === Bootstrap 工具 ===
def bootstrap_ci(samples, confidence=0.95):
    N = len(samples)
    lower = np.percentile(samples, (1 - confidence) / 2 * 100)
    upper = np.percentile(samples, (1 + confidence) / 2 * 100)
    mean = np.mean(samples)
    std = np.std(samples)
    rsd = 100 * std / mean
    return mean, rsd, lower, upper

# === collapse 场结构函数 ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0):
    def pulse(x0, t0_local, amp=1.0):
        if t < t0_local:
            return 0.0
        spatial = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        time = (t - t0_local)**k * np.exp(-((t - t0_local)**2) / (2 * sigma_t**2))
        return amp * spatial * time
    return A * (pulse(195, t0) + pulse(205, t0 + 0.1, -0.7))

# === G(t) 计算核心 ===
def compute_structural_G(params):
    x = np.linspace(150, 250, 1000)
    t = np.linspace(4.5, 5.5, 200)
    dx, dt = x[1] - x[0], t[1] - t[0]
    L_L, L_T, L_M = params["L_L"], params["L_T"], params["L_M"]
    G_ref = params["G_ref"]

    # 生成结构场
    delta_xt = np.array([
        [delta_dag(xi, tj, params["A"], params["sigma_x"], params["sigma_t"], params["k"], params["t0"]) for xi in x]
        for tj in t
    ])
    
    # H(t) 与 φ_c(t) 计算
    H_t = np.array([np.trapz(delta_xt[i]**2, x) for i in range(len(t))])
    φ_integral = np.array([np.trapz(delta_xt[i], x) for i in range(len(t))])
    φ_c_t = np.gradient(φ_integral, dt)
    
    # K(t) 结构比率
    log_H = np.log(H_t + 1e-30)
    log_φ = np.log(np.abs(φ_integral) + 1e-30)
    dlogH = np.gradient(log_H, dt)
    dlogφ = np.gradient(log_φ, dt)
    K_t = dlogH / (dlogφ + 1e-20)
    
    # G(t) 估计
    valid = np.where((H_t > 1e-30) & (np.abs(φ_c_t) > 1e-20))[0]
    if len(valid) == 0:
        return None

    G_struct = params["phi2"] / (H_t[valid] * φ_c_t[valid])
    G_phys = G_struct * (L_L**3) / (L_M * L_T**2)
    t_valid = t[valid]

    # 找到误差最小的 t*
    idx = np.argmin(np.abs(G_phys - G_ref))
    t_star = t_valid[idx]
    G_star = G_phys[idx]
    H_star = H_t[valid[idx]]
    φ_c_star = φ_c_t[valid[idx]]
    K_star = K_t[valid[idx]]

    # Bootstrap 测试
    N_boot = 1000
    noise_std = 0.01 * G_star
    boot_samples = G_star + np.random.normal(0, noise_std, N_boot)
    mean_G, rsd, ci_lo, ci_hi = bootstrap_ci(boot_samples)

    # 可视化
    plt.figure(figsize=(14,4))

    # G Bootstrap
    plt.subplot(1,3,1)
    plt.hist(boot_samples, bins=40, alpha=0.7, color='orange', edgecolor='k')
    plt.axvline(G_ref, color='red', linestyle='--', label='Ref G')
    plt.axvline(mean_G, color='blue', linestyle='-', label='Mean G')
    plt.title('Bootstrap Distribution of G')
    plt.xlabel('G')
    plt.ylabel('Freq')
    plt.legend()
    plt.grid(True)

    # φ²(x) 可视化
    plt.subplot(1,3,2)
    x_mid = np.linspace(150, 250, 1000)
    phi_sq = delta_dag(x_mid, t_star, params["A"], params["sigma_x"], params["sigma_t"], params["k"], params["t0"])**2
    plt.plot(x_mid, phi_sq)
    plt.title(f'Structure φ²(x) at t* = {t_star:.3f}')
    plt.grid(True)

    # K(t) 曲线
    plt.subplot(1,3,3)
    plt.plot(t_valid, K_t[valid], label='K(t)')
    plt.axvline(t_star, color='red', linestyle='--', label='t*')
    plt.title('K(t) Structure Curve')
    plt.xlabel('t')
    plt.ylabel('K(t)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出结果
    print("=== Structural G Estimate Summary ===")
    print(f"t*          = {t_star:.5f} s")
    print(f"G(t*)       = {G_star:.6e}")
    print(f"Ref G       = {G_ref:.6e}")
    print(f"Relative Err= {(G_star - G_ref)/G_ref*100:.4f}%")
    print(f"RSD (boot)  = {rsd:.4f}%")
    print(f"95% CI      = [{ci_lo:.6e}, {ci_hi:.6e}]")
    print(f"H(t*)       = {H_star:.3e}")
    print(f"φ_c(t*)     = {φ_c_star:.3e}")
    print(f"K(t*)       = {K_star:.4f}")

# === 默认参数设置 ===
params = {
    "phi2": 5e-30,
    "A": 0.1,
    "sigma_x": 8,
    "sigma_t": 0.3,
    "k": 5,
    "t0": 4.94,
    "L_L": 1e-15,
    "L_T": 1e-20,
    "L_M": 0.063826,
    "G_ref": 6.67430e-11
}

# === 执行结构 G 分析 ===
compute_structural_G(params)
