# Collapse-only structural constants generation and Q_eff(t) evaluation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# === Grid Setup ===
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 400)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === Structural Field Generator ===
def generate_delta_xt(x, t, sigma_x, N):
    x_centers = np.linspace(180, 220, N)
    psi_total = np.zeros_like(x)
    for x0 in x_centers:
        psi_total += np.exp(-((x - x0)**2) / (2 * sigma_x**2))
    psi_total /= np.sqrt(trapezoid(psi_total**2, x))
    envelope = 1 / np.cosh(3 * (t - 5.0))**2
    return envelope[:, None] * psi_total[None, :]

# === Effective Constant Evaluation ===
def evaluate_constants(delta_xt):
    phi_eff = trapezoid(delta_xt, x, axis=1)
    dphi_eff = np.gradient(phi_eff, dt)
    H_t = trapezoid(delta_xt**2, x, axis=1)

    G_eff_t = 1 / (8 * np.pi * H_t**2 * np.maximum(phi_eff, 1e-12)**2)
    alpha_eff_t = dphi_eff**2 / np.maximum(phi_eff**2, 1e-12)
    hbar_eff_t = np.sqrt(H_t) * dt

    Q_eff_t = G_eff_t * alpha_eff_t * hbar_eff_t
    return Q_eff_t, phi_eff, H_t

# === Run Test ===
sigma_x = 0.16
N = 400
delta_xt = generate_delta_xt(x, t, sigma_x, N)
Q_eff_t, phi_eff_t, H_t = evaluate_constants(delta_xt)

# === Output Result ===
print("\n=== Q_eff(t) Stability Test ===")
print(f"  sigma_x = {sigma_x}, N = {N}")
print(f"  Mean Q_eff = {np.mean(Q_eff_t):.6e}")
print(f"  Std Q_eff  = {np.std(Q_eff_t):.3e}")

# === Optional: Plot Q_eff(t) ===
plt.plot(t, Q_eff_t)
plt.xlabel("t")
plt.ylabel("Q_eff(t)")
plt.title("Structural Closure Product Q_eff(t)")
plt.grid(True)
plt.tight_layout()
plt.show()