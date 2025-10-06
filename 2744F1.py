# ===============================================
# Colab Full Script — DAG·UTH 强场“钉子测试”一键版
# ===============================================

# ---- deps
import os, json, math, io, gzip, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from astropy.io import fits

# ---- utils: download/cache
def fetch(url, path):
    if os.path.exists(path):
        print(f"[cache] {os.path.basename(path)}")
        return path
    print(f"[download] {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    return path

# ---- data catalog (HFF CATS v4 kappa maps)
CLUSTERS = {
    "Abell2744_CATSv4": {
        "url": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits",
        "local": "hlsp_frontier_model_abell2744_cats_v4_kappa.fits",
        "z": 0.308, "M200": 18e14
    },
    "MACS0416_CATSv4": {
        "url": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/frontier/macs0416/models/cats/v4/hlsp_frontier_model_macs0416_cats_v4_kappa.fits",
        "local": "hlsp_frontier_model_macs0416_cats_v4_kappa.fits",
        "z": 0.396, "M200": 12e14
    },
    "Abell370_CATSv4": {
        "url": "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/frontier/abell370/models/cats/v4/hlsp_frontier_model_abell370_cats_v4_kappa.fits",
        "local": "hlsp_frontier_model_abell370_cats_v4_kappa.fits",
        "z": 0.375, "M200": 10e14
    }
}

# ---- basic ops
def grad2(img):
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:,1:-1] = 0.5*(img[:,2:]-img[:,:-2])
    gy[1:-1,:] = 0.5*(img[2:,:]-img[:-2,:])
    gx[:,0] = img[:,1]-img[:,0]; gx[:,-1] = img[:,-1]-img[:,-2]
    gy[0,:] = img[1,:]-img[0,:]; gy[-1,:] = img[-1,:]-img[-2,:]
    return gx, gy

def lap(img):
    L = np.zeros_like(img)
    L[1:-1,1:-1] = (img[1:-1,2:] + img[1:-1,:-2] + img[2:,1:-1] + img[:-2,1:-1] - 4*img[1:-1,1:-1])
    # 边缘简单镜像
    L[0,1:-1]   = (img[0,2:] + img[0,:-2] + img[1,1:-1] + img[0,1:-1] - 4*img[0,1:-1])
    L[-1,1:-1]  = (img[-1,2:] + img[-1,:-2] + img[-2,1:-1] + img[-1,1:-1] - 4*img[-1,1:-1])
    L[1:-1,0]   = (img[1:-1,1] + img[1:-1,0] + img[2:,0]    + img[:-2,0]   - 4*img[1:-1,0])
    L[1:-1,-1]  = (img[1:-1,-2]+ img[1:-1,-1]+ img[2:,-1]   + img[:-2,-1]  - 4*img[1:-1,-1])
    # 四角
    L[0,0]      = (img[0,1]+img[1,0]+img[0,0]+img[0,0]-4*img[0,0])
    L[0,-1]     = (img[0,-2]+img[1,-1]+img[0,-1]+img[0,-1]-4*img[0,-1])
    L[-1,0]     = (img[-1,1]+img[-2,0]+img[-1,0]+img[-1,0]-4*img[-1,0])
    L[-1,-1]    = (img[-1,-2]+img[-2,-1]+img[-1,-1]+img[-1,-1]-4*img[-1,-1])
    return L

# 非线性核 N(s)
def N_of_s(s, Lambda):
    L2 = Lambda*Lambda
    return (s/(1.0 + s/(L2)+1e-30))**2

# ===== 自适应门控（避免 cov=0） =====
def _gate_mask_from_s(s, min_cov=0.05, target_cov=0.20, min_pixels=500):
    s_f = s[np.isfinite(s)]
    if s_f.size < min_pixels:
        return np.zeros_like(s, dtype=bool), np.nan, 0.0

    # 1) Otsu 阈
    hist, bins = np.histogram(s_f, bins=256)
    hist = hist.astype(float) / max(1, hist.sum())
    omega = np.cumsum(hist)
    centers = 0.5*(bins[:-1]+bins[1:])
    mu = np.cumsum(hist*centers)
    mu_t = mu[-1]
    denom = (omega*(1-omega)+1e-12)
    sigma_b2 = (mu_t*omega - mu)**2 / denom
    k = np.nanargmax(sigma_b2)
    thr_otsu = centers[k]
    mask = (s > thr_otsu); cov = float(np.mean(mask))
    if cov >= min_cov and mask.sum() >= min_pixels:
        return mask, float(thr_otsu), cov

    # 2) 分位数 q=0.80
    q = np.quantile(s_f, 0.80)
    mask = (s > q); cov = float(np.mean(mask))
    if cov >= min_cov and mask.sum() >= min_pixels:
        return mask, float(q), cov

    # 3) 目标覆盖率 target_cov
    tq = np.quantile(s_f, 1.0 - target_cov)
    mask = (s > tq); cov = float(np.mean(mask))
    if mask.sum() < min_pixels:
        flat_idx = np.argsort(s, axis=None)
        top_idx = flat_idx[-min_pixels:]
        mask = np.zeros_like(s, dtype=bool).reshape(-1)
        mask[top_idx] = True
        mask = mask.reshape(s.shape)
        cov = float(np.mean(mask))
        tq = float(np.min(s.flatten()[top_idx]))
    return mask, float(tq), cov

# ===== (16′) 强场非线性 vs GR弱场 — 模型拟合 & 信息准则 =====
def fit_models(kappa, mask=None, l2_beta=1e-3):
    phi = gaussian_filter(kappa, sigma=1.0)
    gx, gy = grad2(phi); s = gx*gx + gy*gy
    L = lap(phi)

    if mask is None:
        mask, thr, cov = _gate_mask_from_s(s)
    else:
        thr = np.nan; cov = float(np.mean(mask))
    idx = np.where(mask)
    n = max(1, len(idx[0]))

    # 模型A：GR弱场（常数偏置）
    L_in = L[idx]
    cA = np.mean(L_in)
    resid_A = (L_in - cA)
    rss_A = float(np.sum(resid_A**2))
    k_A = 1
    aic_A  = n*np.log(rss_A/n + 1e-30) + 2*k_A
    aicc_A = aic_A + (2*k_A*(k_A+1))/max(n-k_A-1,1)
    bic_A  = n*np.log(rss_A/n + 1e-30) + k_A*np.log(n+1e-30)

    # 模型B：(16′) 非线性
    def obj_B(p):
        beta, Lambda, c0 = p
        Ns = N_of_s(s[idx], max(Lambda,1e-6))
        pred = L_in + beta*Ns - c0
        return np.mean(pred**2) + l2_beta*(beta**2)

    res = minimize(obj_B, x0=[0.1, 1.5, 0.0],
                   bounds=[(-1.0, 1.0), (1e-3, 50.0), (-np.inf, np.inf)],
                   method="L-BFGS-B")
    beta_hat, Lambda_hat, c0_hat = [float(v) for v in res.x]
    Ns = N_of_s(s[idx], max(Lambda_hat,1e-6))
    resid_B = (L_in + beta_hat*Ns - c0_hat)
    rss_B = float(np.sum(resid_B**2))
    k_B = 3
    aic_B  = n*np.log(rss_B/n + 1e-30) + 2*k_B
    aicc_B = aic_B + (2*k_B*(k_B+1))/max(n-k_B-1,1)
    bic_B  = n*np.log(rss_B/n + 1e-30) + k_B*np.log(n+1e-30)

    dAIC, dAICc, dBIC = aic_B - aic_A, aicc_B - aicc_A, bic_B - bic_A
    lnBF_BA = -0.5*dBIC  # Schwarz 近似

    return {
        "thr": float(thr), "cov": float(cov), "n": int(n),
        "beta": beta_hat, "Lambda": Lambda_hat, "c0": c0_hat,
        "rss_A": rss_A, "aic_A": aic_A, "aicc_A": aicc_A, "bic_A": bic_A,
        "rss_B": rss_B, "aic_B": aic_B, "aicc_B": aicc_B, "bic_B": bic_B,
        "dAIC": dAIC, "dAICc": dAICc, "dBIC": dBIC, "lnBF_BA": lnBF_BA
    }

# ===== 稳健 K-fold 交叉验证（避免空掩膜折） =====
def kfold_cv(kappa, K=5, seed=42):
    rng = np.random.default_rng(seed)
    phi = gaussian_filter(kappa, sigma=1.0)
    gx, gy = grad2(phi); s = gx*gx+gy*gy; L = lap(phi)
    mask, thr, cov = _gate_mask_from_s(s)
    idx = np.array(np.where(mask)).T
    if idx.shape[0] < 1000:
        K_eff = max(2, min(K, idx.shape[0]//200))
    else:
        K_eff = K
    rng.shuffle(idx)
    folds = np.array_split(idx, K_eff)
    scores = []
    for k in range(K_eff):
        test = folds[k]
        train = np.concatenate([folds[i] for i in range(K_eff) if i!=k], axis=0)

        def obj_B(p):
            beta, Lambda, c0 = p
            Ns = N_of_s(s[train[:,0], train[:,1]], max(Lambda,1e-6))
            pred = L[train[:,0], train[:,1]] + beta*Ns - c0
            return np.mean(pred**2) + 1e-3*(beta**2)

        res = minimize(obj_B, x0=[0.1, 1.5, 0.0],
                       bounds=[(-1.0,1.0),(1e-3,50.0),(-np.inf,np.inf)],
                       method="L-BFGS-B")
        beta_hat, Lambda_hat, c0_hat = res.x

        A = L[test[:,0], test[:,1]] - np.mean(L[train[:,0], train[:,1]])
        varA = np.var(A)
        B = L[test[:,0], test[:,1]] + beta_hat*N_of_s(s[test[:,0], test[:,1]], max(Lambda_hat,1e-6)) - c0_hat
        varB = np.var(B)
        scores.append({
            "fold":k+1, "varA":float(varA), "varB":float(varB),
            "beta":float(beta_hat), "Lambda":float(Lambda_hat), "c0":float(c0_hat),
            "cov":float(cov), "thr":float(thr)
        })
    return scores

# ===== 鲁棒性：PSF/抖动/LOS 扰动下的 lnBF 分布 =====
def robustness_lnBF(kappa, n_trials=20, seed=0):
    rng = np.random.default_rng(seed)
    lnBFs = []
    for _ in range(n_trials):
        # 轻度高斯 PSF + 抖动 + 背景 LOS 漂移
        k = kappa.copy()
        sigma = rng.uniform(0.6, 1.2)
        k = gaussian_filter(k, sigma=sigma)
        k += rng.normal(0, 0.01*np.std(k), size=k.shape)
        k += rng.normal(0, 0.02*np.std(k))  # LOS 背景

        out = fit_models(k)
        lnBFs.append(out["lnBF_BA"])
    return {
        "median_lnBF": float(np.median(lnBFs)),
        "p10": float(np.percentile(lnBFs,10)),
        "p90": float(np.percentile(lnBFs,90)),
        "all": [float(x) for x in lnBFs]
    }

# ===== 可视化：每团CV与鲁棒性面板 =====
def plot_cluster_panels(name, kappa, cv_scores, robust, outdir="figs"):
    os.makedirs(outdir, exist_ok=True)
    # CV 面板
    fig1 = plt.figure(figsize=(6,4))
    xs = [s["fold"] for s in cv_scores]
    r  = [s["varA"]/max(s["varB"],1e-12) for s in cv_scores]
    plt.plot(xs, r, marker="o")
    plt.axhline(1.0, ls="--")
    plt.title(f"[CV] varA/varB (>1 favors nonlinear)\n{name}")
    plt.xlabel("fold"); plt.ylabel("varA/varB")
    plt.tight_layout()
    cv_path = os.path.join(outdir, f"{name}_cv.png")
    plt.savefig(cv_path, dpi=150)

    # Robust 面板
    fig2 = plt.figure(figsize=(6,4))
    arr = np.array(robust["all"])
    plt.hist(arr, bins=12, alpha=0.8)
    med = np.median(arr)
    plt.axvline(0, color="k", ls="--")
    plt.axvline(med, color="r", ls="-")
    plt.title(f"[Robust] lnBF(B vs A)\n{name} (median={med:.2f})")
    plt.xlabel("lnBF"); plt.ylabel("count")
    plt.tight_layout()
    rb_path = os.path.join(outdir, f"{name}_robust.png")
    plt.savefig(rb_path, dpi=150)
    return cv_path, rb_path

# ===== 主流程：三团“钉子测试” =====
def run_nail_tests(cluster_dict=CLUSTERS):
    summary = {}
    print("=== DAG·UTH Nail Tests (A: GR weak vs B: Nonlinear 16′) ===")
    for name, meta in cluster_dict.items():
        print(f"\n=== {name} ===")
        path = fetch(meta["url"], meta["local"])
        with fits.open(path) as hdul:
            kappa = np.array(hdul[0].data, dtype=float)
        # 归一化到稳健尺度
        kappa = (kappa - np.nanmedian(kappa))/ (np.nanstd(kappa)+1e-12)

        # 模型选择
        out = fit_models(kappa)
        print(f"[ModelSel] cov={out['cov']:.3f}, thr={out['thr']:.3e}, "
              f"beta={out['beta']:.4g}, Lambda={out['Lambda']:.3g}, "
              f"ΔBIC={out['dBIC']:.2f}, lnBF(B vs A)={out['lnBF_BA']:.2f}")

        # K-fold CV
        cv_scores = kfold_cv(kappa, K=5)
        ratio = np.mean([s["varA"]/max(s["varB"],1e-12) for s in cv_scores])
        print(f"[CV] mean(varA/varB)={ratio:.2f}  (>1 表示非线性模型更好)")

        # 鲁棒性
        robust = robustness_lnBF(kappa, n_trials=30, seed=0)
        med_lnBF = robust["median_lnBF"]
        print(f"[Robust] median lnBF={med_lnBF:.2f}  "
              f"(+PSF/抖动/LOS 后仍>0 ⇒ 非线性偏好稳定)")

        # 绘图
        cv_png, rb_png = plot_cluster_panels(name, kappa, cv_scores, robust)

        summary[name] = {
            "z": meta["z"], "M200": meta["M200"],
            "cov": out["cov"], "thr": out["thr"], "n": out["n"],
            "beta": out["beta"], "Lambda": out["Lambda"], "c0": out["c0"],
            "dAIC": out["dAIC"], "dAICc": out["dAICc"], "dBIC": out["dBIC"], "lnBF": out["lnBF_BA"],
            "cv_mean_var_ratio": ratio,
            "robust_median_lnBF": med_lnBF,
            "cv_png": cv_png, "robust_png": rb_png
        }

    # 总结打印
    print("\n================= NAIL TESTS — SUMMARY =================")
    for name, v in summary.items():
        print(f"{name} | cov={v['cov']:.3f}, beta={v['beta']:.4g}, Lambda={v['Lambda']:.3g}, "
              f"ΔBIC={v['dBIC']:.2f}, lnBF={v['lnBF']:.2f}, n={int(v['n'])}")

    # 保存 JSON
    with open("dag_uth_nail_tests_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved JSON: dag_uth_nail_tests_summary.json")
    print("Saved figures: figs/<cluster>_cv.png, figs/<cluster>_robust.png")

    # 结束时把所有图 show 出来
    plt.show()
    return summary

# ====== RUN ======
summary = run_nail_tests()
