import numpy as np
from numpy import trapezoid as simps

# --- Physical Constants ---
G_codata = 6.67430e-11
lambda_L = 1e-15
lambda_T = 1e-20
lambda_M = 0.063826

# --- Grid Setup ---
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 200)
dx = x[1] - x[0]
dt = t[1] - t[0]

# --- Collapse Field Generator ---
def delta_dag(x, t, A=0.1, sigma_x=8, sigma_t=0.3, k=5, t0=4.95):
    def pulse(x0, t0_local, amp=1.0):
        if t < t0_local: return 0.0
        spatial = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        time_shift = (t - t0_local)**k
        time_decay = np.exp(-((t - t0_local)**2) / (2 * sigma_t**2))
        return amp * spatial * time_shift * time_decay
    return A * (pulse(195, t0, 1.0) + pulse(205, t0 + 0.1, -0.7))

# --- Refined Scan Grids ---
phi_e_sq_grid = [2e-30, 5e-30]
A_grid = [0.10, 0.15]
sigma_t_grid = [0.20, 0.25]
t0_grid = [4.945, 4.950, 4.955]

# --- Search ---
best_result = None
min_error = float("inf")

for phi_e_sq in phi_e_sq_grid:
    for A in A_grid:
        for sigma_t in sigma_t_grid:
            for t0 in t0_grid:
                delta_xt = np.array([[delta_dag(xi, tj, A=A, sigma_t=sigma_t, t0=t0) for xi in x] for tj in t])
                H_t = np.array([simps(delta_xt[i]**2, x) for i in range(len(t))])
                integral = np.array([simps(delta_xt[i], x) for i in range(len(t))])
                phi_c_t = np.gradient(integral, dt)

                valid_idx = np.where((H_t > 1e-40) & (np.abs(phi_c_t) > 1e-20))[0]
                if len(valid_idx) == 0: continue

                G_t = phi_e_sq / (H_t[valid_idx] * phi_c_t[valid_idx])
                G_physical = G_t * (lambda_L**3) / (lambda_M * lambda_T**2)
                error = np.abs(G_physical - G_codata)
                best_idx = np.argmin(error)

                if error[best_idx] < min_error:
                    min_error = error[best_idx]
                    best_result = {
                        "G(t*)": G_physical[best_idx],
                        "φ_c(t*)": phi_c_t[valid_idx[best_idx]],
                        "H(t*)": H_t[valid_idx[best_idx]],
                        "t*": t[valid_idx[best_idx]],
                        "φₑ²": phi_e_sq,
                        "A": A,
                        "σₜ": sigma_t,
                        "t₀": t0,
                        "abs_error": error[best_idx],
                        "rel_error": error[best_idx] / G_codata
                    }

# --- Output ---
print("\n=== Optimal Structural G(t) Result (Refined Scan) ===")
for k, v in best_result.items():
    if isinstance(v, float):
        print(f"{k:<12}: {v:.5e}")
    else:
        print(f"{k:<12}: {v}")
