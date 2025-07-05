import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# === Constants ===
G_true = 6.67430e-11
phi2 = 5e-30  # fixed phi^2

x = np.linspace(0, 1, 1000)
t = np.linspace(0, 1, 200)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === Collapse Field ===
def delta_dag(x, t, A, sigma_x, sigma_t, k, t0):
    def pulse(x0, t0_local, amp=1.0):
        if t < t0_local: return 0.0
        spatial = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        time = (t - t0_local)**k * np.exp(-((t - t0_local)**2) / (2 * sigma_t**2))
        return amp * spatial * time
    return A * (pulse(0.4, t0) + pulse(0.6, t0 + 0.05, -0.7))

# === Compute K(t) ===
def compute_K(H_t, phi_t):
    log_H = np.log(np.abs(H_t) + 1e-30)
    log_phi = np.log(np.abs(phi_t) + 1e-30)
    dlogH_dt = np.gradient(log_H, dt)
    dlogphi_dt = np.gradient(log_phi, dt)
    return dlogH_dt / (dlogphi_dt + 1e-20)

# === G_struct calculator ===
def compute_G_struct(params):
    A, sigma_x, t0, k = params
    delta_xt = np.array([[delta_dag(xi, tj, A, sigma_x, 0.1, k, t0) for xi in x] for tj in t])
    H_t = np.array([trapezoid(delta_xt[i]**2, x) for i in range(len(t))])
    integral = np.array([trapezoid(delta_xt[i], x) for i in range(len(t))])
    phi_c_t = np.gradient(integral, dt)
    K_t = compute_K(H_t, integral)

    valid = np.where((H_t > 1e-30) & (np.abs(phi_c_t) > 1e-30))[0]
    if len(valid) == 0: return None

    idx = np.argmin(np.abs(K_t[valid] - 2.0))
    H_star = H_t[valid][idx]
    phi_star = phi_c_t[valid][idx]
    K_star = K_t[valid][idx]
    t_star = t[valid][idx]
    G_struct = phi2 / (H_star * phi_star)

    return {
        "G": G_struct,
        "K": K_star,
        "t*": t_star,
        "A": A,
        "sigma_x": sigma_x,
        "t0": t0,
        "k": k
    }

# === Parameter Grid ===
A_vals = [0.05, 0.1]
sigma_x_vals = [0.04, 0.05]
t0_vals = [0.46, 0.48]
k_vals = [5, 6]

# === Run Tests ===
results = []
for A in A_vals:
    for sigma_x in sigma_x_vals:
        for t0 in t0_vals:
            for k in k_vals:
                res = compute_G_struct((A, sigma_x, t0, k))
                if res:
                    G_rel_error = abs(res["G"] - G_true) / G_true * 100
                    K_error = abs(res["K"] - 2.0)
                    pass_G = G_rel_error < 1
                    pass_K = K_error < 0.01
                    status = "✅ PASS" if (pass_G and pass_K) else "❌ FAIL"
                    print(f"{status} | G = {res['G']:.3e} (err={G_rel_error:.2f}%), K = {res['K']:.4f}, t* = {res['t*']:.3f}, A={res['A']}, σₓ={res['sigma_x']}, t₀={res['t0']}, k={res['k']}")
                    results.append((res["t*"], res["K"]))
                else:
                    print(f"⚠️ No valid result for A={A}, σₓ={sigma_x}, t₀={t0}, k={k}")

# === Optional: Visualize One Example ===
print("\nPlotting example K(t) curve:")
example = (0.1, 0.05, 0.48, 5)
delta_xt = np.array([[delta_dag(xi, tj, *example) for xi in x] for tj in t])
H_t = np.array([trapezoid(delta_xt[i]**2, x) for i in range(len(t))])
integral = np.array([trapezoid(delta_xt[i], x) for i in range(len(t))])
phi_c_t = np.gradient(integral, dt)
K_t = compute_K(H_t, integral)

plt.figure(figsize=(10, 4))
plt.plot(t, K_t, label="K(t)")
plt.axhline(2.0, color='r', linestyle='--', label='Target K=2')
plt.title("K(t) for A=0.1, σₓ=0.05, t₀=0.48, k=5")
plt.xlabel("Time (t)")
plt.ylabel("K(t)")
plt.grid(True)
plt.legend()
plt.show()
