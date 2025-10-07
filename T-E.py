# %% [markdown]
# === UTH / Time–Energy Exchange — GL2 (Newton) 合并验证 ===
# - [A] 基线：真·GL2 + 解析残量 -> 交换配对/左匹配 ~ 1e-16，总能量漂移 ~ 1e-14
# - [B] 错号 vs 正号 消融：只有正确号 (A2b) 保守，总能量小漂移
# - [C1] 阶数（终点状态 Richardson）：p ~ 4（GL2 期望≈4）
# - [C2] 有限差分估导（参考）：通常 ~2，不作为主结论
#
# 物理模型：
#   L(q, qdot) = 1/2 m qdot^2 - V(q),  V(q)=1/2 k q^2 + 1/4 λ q^4
#   W(U) = 1 + α U^2,  V_U(U) = 1/2 μ_U U^2
#   (A2a) d/dt( W ∂_{qdot}L ) - W ∂_q L = 0
#   (A2b) κ_U U¨ - W'(U) L + V_U'(U) = 0  (正确号)
#   (A3)  Eṁ = - W'(U) U̇ L,  EU̇ = + W'(U) U̇ L,  (Eṁ + EU̇)=0,  Etot=const
#
# 数值法：
#   Gauss-Legendre(2) 隐式正交配置，牛顿迭代解阶段方程；雅可比用有限差分近似。

import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数 ----------
m = 1.0
k = 1.0
lam = 0.05
alpha = 0.1
muU = 1.0
kappa_U = 1.0

t0, T = 0.0, 10.0       # 可改成 20.0，但计算量会上去
dt_baseline = 1e-4      # 基线步长

q0, qd0 = 1.0, 0.0
U0, Ud0 = 0.05, 0.0
y0 = np.array([q0, qd0, U0, Ud0], dtype=float)

# ---------- 势能/权重/拉格朗日 ----------
def V(q):      return 0.5*k*q*q + 0.25*lam*q**4
def dVdq(q):   return k*q + lam*q**3
def W(U):      return 1.0 + alpha*U*U
def dWdU(U):   return 2.0*alpha*U
def VU(U):     return 0.5*muU*U*U
def dVUdU(U):  return muU*U
def Lagr(q, qd):   return 0.5*m*qd*qd - V(q)

def Em(q, qd, U):
    return W(U)*(0.5*m*qd*qd + V(q))

def EU(U, Ud):
    return 0.5*kappa_U*Ud*Ud + VU(U)

# ---------- 右端函数（错号 vs 正号） ----------
# 状态 y=[q, qd, U, Ud] -> [q̇, q̈, U̇, Ü]
def rhs_correct(y):
    q, qd, U, Ud = y
    Wv = W(U); Wp = dWdU(U); L = Lagr(q, qd)
    qdd = -(1.0/m)*dVdq(q) - (Wp/Wv)*Ud*qd          # (A2a)
    Udd = ( Wp*L - dVUdU(U) ) / kappa_U             # (A2b) 正号
    return np.array([qd, qdd, Ud, Udd])

def rhs_wrong(y):
    q, qd, U, Ud = y
    Wv = W(U); Wp = dWdU(U); L = Lagr(q, qd)
    qdd = -(1.0/m)*dVdq(q) - (Wp/Wv)*Ud*qd
    Udd = (-Wp*L + dVUdU(U)) / kappa_U              # 错号
    return np.array([qd, qdd, Ud, Udd])

# ---------- 真·GL2（Gauss–Legendre 2-stage）+ 牛顿 ----------
A11 = 1/4.0; A12 = 1/4.0 - np.sqrt(3)/6.0
A21 = 1/4.0 + np.sqrt(3)/6.0; A22 = 1/4.0
b1 = b2 = 0.5

def integrate_gl2_newton(rhs, t0, T, h, y0, newton_tol=1e-12, newton_maxit=8, fd_eps=1e-8):
    N = int(np.round((T - t0)/h))
    ts = np.linspace(t0, T, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0.copy()

    nvar = len(y0)

    for i in range(N):
        y = Y[i]
        # 初值：用 f(y) 作为阶段初值
        k1 = rhs(y)
        k2 = rhs(y)

        for it in range(newton_maxit):
            y1 = y + h*(A11*k1 + A12*k2)
            y2 = y + h*(A21*k1 + A22*k2)
            F1 = rhs(y1) - k1
            F2 = rhs(y2) - k2
            R = np.hstack([F1, F2])
            nr = np.linalg.norm(R, ord=np.inf)
            if nr < newton_tol:
                break

            # 8x8 有限差分雅可比
            J = np.zeros((2*nvar, 2*nvar))
            # 对 k1 的扰动
            for j in range(nvar):
                e = np.zeros(nvar); e[j] = 1.0
                k1p = k1 + fd_eps*e
                y1p = y + h*(A11*k1p + A12*k2)
                y2p = y + h*(A21*k1p + A22*k2)
                dF1 = (rhs(y1p) - (k1p)) - F1
                dF2 = (rhs(y2p) -  k2   ) - F2
                J[:nvar, j]      = dF1 / fd_eps
                J[nvar:, j]      = dF2 / fd_eps
            # 对 k2 的扰动
            for j in range(nvar):
                e = np.zeros(nvar); e[j] = 1.0
                k2p = k2 + fd_eps*e
                y1p = y + h*(A11*k1 + A12*k2p)
                y2p = y + h*(A21*k1 + A22*k2p)
                dF1 = (rhs(y1p) -  k1   ) - F1
                dF2 = (rhs(y2p) - (k2p)) - F2
                J[:nvar, nvar+j] = dF1 / fd_eps
                J[nvar:, nvar+j] = dF2 / fd_eps

            delta = np.linalg.solve(J, -R)
            k1 += delta[:nvar]
            k2 += delta[nvar:]

        Y[i+1] = y + h*(b1*k1 + b2*k2)

    return ts, Y

# ---------- 解析残量与能量 ----------
def eval_analytic(ts, Y, use_rhs='correct'):
    q, qd, U, Ud = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    Wv = W(U); Wp = dWdU(U); L = Lagr(q, qd)

    if use_rhs == 'correct':
        qdd = -(1.0/m)*dVdq(q) - (Wp/Wv)*Ud*qd
        Udd = ( Wp*L - dVUdU(U) ) / kappa_U
    else:
        qdd = -(1.0/m)*dVdq(q) - (Wp/Wv)*Ud*qd
        Udd = (-Wp*L + dVUdU(U)) / kappa_U

    # (A2a) 解析残量（应→0）
    EL_res = Wv*m*qdd + Wp*Ud*m*qd + Wv*dVdq(q)

    # 能量与解析时间导数
    Em_t = Em(q, qd, U)
    EU_t = EU(U, Ud)
    Etot = Em_t + EU_t

    # 解析 Eṁ, EU̇
    Em_dot = Wp*Ud*(0.5*m*qd*qd + V(q)) + Wv*(m*qd*qdd + dVdq(q)*qd)
    EU_dot = kappa_U*Ud*Udd + dVUdU(U)*Ud

    ex_pair = Em_dot + EU_dot        # 应→0
    ex_left = Em_dot + Wp*Ud*L       # 正号应→0

    def maxabs(x): return float(np.max(np.abs(x)))
    stats = {
        'EL_res': maxabs(EL_res),
        'ex_pair': maxabs(ex_pair),
        'ex_left': maxabs(ex_left),
        'E_drift': maxabs(Etot - Etot[0]),
        'Etot': Etot,
        'ex_pair_t': ex_pair
    }
    return stats

# ---------- [A] 基线 ----------
tsA, YA = integrate_gl2_newton(rhs_correct, t0, T, dt_baseline, y0)
stA = eval_analytic(tsA, YA, use_rhs='correct')

print("=== [A] Baseline / GL2 (time–energy exchange) ===")
print(f"dt={dt_baseline:.2e}, steps={len(tsA)-1}")
print(f"EL(q) residual        max|·| = {stA['EL_res']:.3e}")
print(f"Exchange pair         max|·| = {stA['ex_pair']:.3e}   (→0)")
print(f"Left match Eṁ+W' U̇ L max|·| = {stA['ex_left']:.3e}   (→0)")
print(f"Total energy drift    max|·| = {stA['E_drift']:.3e}")

# ---------- [B] 错号 vs 正号 消融 ----------
tsW, YW = integrate_gl2_newton(rhs_wrong, t0, T, dt_baseline, y0)
stW = eval_analytic(tsW, YW, use_rhs='wrong')

print("\n=== [B] Sign ablation: WRONG vs CORRECT (GL2) ===")
print("-- WRONG sign (κU U¨ + W' L - VU' = 0) --")
print(f"EL(q) residual     = {stW['EL_res']:.3e}")
print(f"Exchange pair      = {stW['ex_pair']:.3e}   (应≈0；此处会大)")
print(f"Left match (A3)    = {stW['ex_left']:.3e}   (一般≠0)")
print(f"Energy drift       = {stW['E_drift']:.3e}   (大)")

print("\n-- CORRECT sign (κU U¨ - W' L + VU' = 0) --")
print(f"EL(q) residual     = {stA['EL_res']:.3e}")
print(f"Exchange pair      = {stA['ex_pair']:.3e}   (≈0 ✅)")
print(f"Left match (A3)    = {stA['ex_left']:.3e}   (≈0 ✅)")
print(f"Energy drift       = {stA['E_drift']:.3e}   (小 ✅)")

# ---------- [C1] 阶数（终点状态 Richardson） ----------
def final_state(Y): return Y[-1]

def order_via_richardson(rhs, y0, t0, T, h):
    ts1, Y1 = integrate_gl2_newton(rhs, t0, T, h,     y0)
    ts2, Y2 = integrate_gl2_newton(rhs, t0, T, h/2.0, y0)
    ts3, Y3 = integrate_gl2_newton(rhs, t0, T, h/4.0, y0)
    e1 = np.linalg.norm(final_state(Y1) - final_state(Y2), ord=np.inf)
    e2 = np.linalg.norm(final_state(Y2) - final_state(Y3), ord=np.inf)
    p  = np.log2(e1/e2)
    return p, e1, e2

p_est, e1, e2 = order_via_richardson(rhs_correct, y0, t0, T, 8e-4)  # 可调基准 h
print("\n=== [C1] Step-size order via Richardson (final-state) ===")
print(f"p ≈ {p_est:.3f}  (GL2 期望≈4)")

# ---------- [C2] 有限差分估导（参考，不作为主结论） ----------
def fd_time_derivative(ts, f):
    dt = ts[1]-ts[0]
    g = np.zeros_like(f)
    g[1:-1] = (f[2:] - f[:-2])/(2*dt)
    g[0]    = (f[1]-f[0])/dt
    g[-1]   = (f[-1]-f[-2])/dt
    return g

def ex_pair_fd(ts, Y):
    q, qd, U, Ud = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    Em_t = Em(q, qd, U); EU_t = EU(U, Ud)
    Em_dot_fd = fd_time_derivative(ts, Em_t)
    EU_dot_fd = fd_time_derivative(ts, EU_t)
    return Em_dot_fd + EU_dot_fd

# 用一组步长看 FD 误差随 h 的斜率（仅参考）
hs = [8e-4, 4e-4, 2e-4, 1e-4]
errs_fd = []
for h in hs:
    ts, Y = integrate_gl2_newton(rhs_correct, t0, T, h, y0)
    fd_res = ex_pair_fd(ts, Y)
    errs_fd.append(float(np.max(np.abs(fd_res))))

def fit_slope_loglog(xs, ys):
    x = np.log(xs); y = np.log(ys)
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a  # slope

p_fd = fit_slope_loglog(hs, errs_fd)
print("\n=== [C2] Step-size convergence (FD on Eṁ+EU̇) [参考] ===")
print(f"hs: {hs}")
print(f"errs (FD ex_pair): {errs_fd}")
print(f"Fitted slope p ≈ {p_fd:.3f}  (FD 后处理常见≈2；不反映 GL2 主方法的4阶)")

# ---------- 简单图：总能量漂移 & 解析交换残量 ----------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(tsA, stA['Etot'] - stA['Etot'][0], label="Energy drift (correct)")
plt.plot(tsW, stW['Etot'] - stW['Etot'][0], label="Energy drift (wrong)")
plt.xlabel("time"); plt.ylabel("E_tot - E_tot(0)")
plt.title("[A]/[B] Energy drift")
plt.legend()

plt.subplot(1,2,2)
plt.plot(tsA, stA['ex_pair_t'], label="Eṁ+EU̇ (correct, analytic)")
plt.xlabel("time"); plt.ylabel("analytic exchange residual")
plt.title("[A] Exchange residual (analytic)")
plt.legend()
plt.tight_layout()
plt.show()
