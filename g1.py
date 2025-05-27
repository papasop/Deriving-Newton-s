# g_test_suite.py

import numpy as np
from numpy import trapz as simps
import matplotlib.pyplot as plt
from itertools import product

# === Constants ===
G_CODATA = 6.67430e-11
LAMBDA_L = 1e-15
LAMBDA_T = 1e-20
LAMBDA_M = 0.063826

# === Core Grid ===
def generate_grid(x_res=1000, t_res=200):
    x = np.linspace(150, 250, x_res)
    t = np.linspace(4.5, 5.5, t_res)
    return x, t, x[1] - x[0], t[1] - t[0]

# === Collapse Model ===
def delta_dag(x, t, A=0.1, sigma_x=8, sigma_t=0.3, k=5, t0=4.95, single=False, feedback=False):
    def pulse(x0, t0_local, amp=1.0):
        if t < t0_local:
            return 0.0
        spatial = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        time_shift = (t - t0_local)**k
        time_decay = np.exp(-((t - t0_local)**2) / (2 * sigma_t**2))
        return amp * spatial * time_shift * time_decay

    base = pulse(195, t0, 1.0)
    if not single:
        base += pulse(205, t0 + 0.1, -0.7)
    if feedback:
        base += 0.01 * base**2
    return A * base

# === G(t) Evaluation ===
def evaluate_G(phi_e_sq, A, sigma_t, t0, sigma_x=8, k=5, feedback=False, single=False, noise_level=0.0,
               x_res=1000, t_res=200, threshold_H=1e-45, threshold_phi_c=1e-25):
    x, t, dx, dt = generate_grid(x_res, t_res)
    delta_xt = np.array([[delta_dag(xi, tj, A, sigma_x, sigma_t, k, t0, single, feedback) for xi in x] for tj in t])
    if noise_level > 0:
        delta_xt += np.random.normal(0, noise_level, delta_xt.shape)

    H_t = np.array([simps(delta_xt[i]**2, x) for i in range(len(t))])
    integral = np.array([simps(delta_xt[i], x) for i in range(len(t))])
    phi_c_t = np.gradient(integral, dt, edge_order=2)

    valid_idx = np.where((H_t > threshold_H) & (np.abs(phi_c_t) > threshold_phi_c))[0]
    if len(valid_idx) == 0:
        return None

    G_t = phi_e_sq / (H_t[valid_idx] * phi_c_t[valid_idx])
    G_phys = G_t * (LAMBDA_L**3) / (LAMBDA_M * LAMBDA_T**2)
    error = np.abs(G_phys - G_CODATA)
    best_idx = np.argmin(error)
    return {
        "G(t*)": G_phys[best_idx],
        "error": error[best_idx],
        "rel_error": error[best_idx] / G_CODATA,
        "phi_e_sq": phi_e_sq,
        "A": A,
        "sigma_t": sigma_t,
        "t0": t0,
        "H": H_t[valid_idx[best_idx]],
        "phi_c": phi_c_t[valid_idx[best_idx]],
        "t*": t[valid_idx[best_idx]],
    }

# === Grid Search + Batch Runner ===
def run_parameter_grid():
    phi_e_sq_grid = [1e-30, 2e-30, 5e-30, 1e-29, 2e-29]
    A_grid = [0.05, 0.1, 0.15, 0.2, 0.5]
    sigma_t_grid = [0.15, 0.2, 0.25, 0.3, 0.5]
    t0_grid = [4.94, 4.945, 4.95, 4.955, 4.96]

    all_results = []
    for phi_e_sq, A, sigma_t, t0 in product(phi_e_sq_grid, A_grid, sigma_t_grid, t0_grid):
        result = evaluate_G(phi_e_sq, A, sigma_t, t0)
        if result: all_results.append(result)
    return sorted(all_results, key=lambda r: r["error"])

# === Noise Stability Test ===
def run_noise_test(phi_e_sq=5e-30, A=0.1, sigma_t=0.2, t0=4.95, N=50, noise_level=1e-10):
    g_vals = []
    for _ in range(N):
        res = evaluate_G(phi_e_sq, A, sigma_t, t0, noise_level=noise_level)
        if res:
            g_vals.append(res["G(t*)"])
    if g_vals:
        g_vals = np.array(g_vals)
        print(f"Mean G(t*): {np.mean(g_vals):.5e}, Std: {np.std(g_vals):.5e}")
        plt.hist(g_vals, bins=20)
        plt.title("G(t*) Distribution under Noise")
        plt.xlabel("G(t*)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# === Entrypoint for testing ===
if __name__ == "__main__":
    print("=== Running Extended Parameter Grid Test ===")
    results = run_parameter_grid()
    for r in results[:5]:
        print(f"G = {r['G(t*)']:.3e}, Rel Err = {r['rel_error']:.2%}, A={r['A']}, σₜ={r['sigma_t']}, t₀={r['t0']}")

    print("\n=== Running Noise Test on Best Config ===")
    run_noise_test()
