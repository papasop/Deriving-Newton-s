import numpy as np
import matplotlib.pyplot as plt

# === 常数 ===
G_CODATA = 6.67430e-11

# 单位系统（与你模型一致）
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

unit_factor = (L_L**3) / (L_M * L_T**2)

# 固定结构响应（来自你实测数据）
H_t_star_fixed = 1.612e-47         # 固定结构能
phi_c_t_star_fixed = 3.1e-6        # 固定结构导数

# 扫描 φ² 范围
phi2_vals = np.logspace(-32, -27, 100)
G_vals = [phi2 / (H_t_star_fixed * phi_c_t_star_fixed) * unit_factor for phi2 in phi2_vals]

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(phi2_vals, G_vals, 'o-', label='G(φ²)', linewidth=2)
plt.axhline(G_CODATA, color='red', linestyle='--', label='G_CODATA')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('φ² (log scale)', fontsize=12)
plt.ylabel('G(φ²) (log scale)', fontsize=12)
plt.title('固定结构响应下的 G(φ²) 线性验证', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
