# -*- coding: utf-8 -*-
# ============================================================
# OGU 强场一键验证（整合版，拷贝即跑）
# - SCI (ζ–phase)：K 与 K~；(w, stride) 网格 + KS 对照 (GUE/Poisson)
# - g 扫描（naive）+ 注入–回收（现实噪声/抽窗；随机平移网格 + 抛物线细化）
# - EHT Sgr A*：各向异性散射环 + 两个 |u| 窗口（第一/第二瓣）
# - bin-wise 逐点预测；扇区面板；锁定率；联合 ΔAIC
# - 输出目录：/content/out
# ============================================================

# ========== 0) 安装与导入 ==========
import os, sys, json, math, time, random, warnings
from pathlib import Path
!pip -q install pyuvdata mpmath > /dev/null

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import ks_2samp
from scipy.special import j0
from astropy.io import fits
from pyuvdata import UVData
import mpmath as mp
from urllib.request import urlretrieve

warnings.filterwarnings("ignore", category=UserWarning)  # 避免 pyuvdata 的已知提醒
plt.rcParams["figure.dpi"] = 140
np.set_printoptions(suppress=True, linewidth=140)

# ========== 1) 超参（预注册） ==========
SEED            = 42
random.seed(SEED); np.random.seed(SEED)

# ζ 零点 & SCI
USE_TRUE_ZEROS  = True
N_ZERO          = 1200
BASE_M          = 600
W_LIST          = [40, 60, 80]
STRIDE_LIST     = [1, 2, 4]

# g 扫描
G_GRID          = np.linspace(-0.6, 0.6, 241)   # 补丁1：细网格 (~0.005)
W_FOR_GSCAN     = 80
N_BOOT_G        = 200
NOISE_TAU_SIG   = 0.03
DROP_RATE       = 0.20

# EHT uvfits
URL_UVFITS = "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/uvfits/ER6_SGRA_2017_096_hi_casa_netcal-LMTcal_StokesI.uvfits"
DST_UVFITS = "data/sgrA.uvfits"

# |u| 窗口（Gλ）
UV_WINDOWS_G = [(2.2, 4.2), (4.5, 7.5)]
NBINS        = 28
SECTORS_DEG  = list(range(0, 360, 45))  # 8×45°
D_GR_UAS     = 52.0

# 目录
OUT = Path("/content/out"); OUT.mkdir(parents=True, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ========== 2) ζ 零点 & SCI ==========
def load_zetazeros(N):
    mp.mp.dps = 50
    return np.array([float(mp.zetazero(k).imag) for k in range(1, N+1)], float)

def unfold_to_tau(gamma, base_M=600, start_idx=0):
    gaps = np.diff(gamma)
    lo, hi = start_idx, min(start_idx + base_M - 1, gaps.size-1)
    base = gaps[lo:hi+1]
    dbar = base.mean() if base.size>0 else gaps.mean()
    tau = gaps / (dbar if dbar>0 else 1.0)
    return tau, dbar, (lo,hi)

def window_spans(N, w, stride):
    spans=[]; j=0
    while j+w <= N:
        spans.append((j, j+w)); j += stride
    return spans

def sci_stats_windows(tau, w, stride):
    spans = window_spans(tau.size, w, stride)
    K, Kt = [], []
    for (a,b) in spans:
        x = tau[a:b]
        Phi = float(np.mean(x)); H = float(np.var(x, ddof=0))
        med = float(np.median(x)); MAD = float(np.median(np.abs(x-med))) + 1e-15
        K.append(Phi/(H if H>0 else 1e-15))
        Kt.append(med/MAD)
    return np.array(K), np.array(Kt), spans

def ks_against(series, ctrl="GUE", seed=SEED):
    m = series.size; rng = np.random.default_rng(seed)
    if ctrl=="GUE":  # Wigner (Rayleigh, mean=1 => σ=√(2/π))
        sigma = np.sqrt(2/np.pi)
        y = rng.rayleigh(scale=sigma, size=m)
    else:            # Poisson (Exp, mean=1)
        y = rng.exponential(scale=1.0, size=m)
    D,p = ks_2samp(series, y)
    return float(D), float(p)

# g 扫描（naive）
def gscan_medabs_dt(tau, w, g_grid):
    spans = window_spans(tau.size, w, 1)
    vals=[]
    for g in g_grid:
        ds=[]
        for (a,b) in spans:
            x = tau[a:b] - np.mean(tau[a:b])  # 每窗居中，⟨N⟩=1 规约
            dt = float(np.sum(np.exp(g*x)-1.0))
            ds.append(abs(dt))
        vals.append(np.median(ds) if ds else np.nan)
    return np.array(vals), spans

# 抛物线细化（补丁2：避免 g CI 退化）
def _quad_refine(x, y, i):
    if i <= 0 or i >= len(x)-1:
        return float(x[i])
    x0,x1,x2 = x[i-1], x[i], x[i+1]
    y0,y1,y2 = y[i-1], y[i], y[i+1]
    denom = (x0-x1)*(x0-x2)*(x1-x2)
    if denom == 0: return float(x1)
    a = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
    b = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom
    if a <= 0: return float(x1)
    gstar = -b/(2*a)
    lo, hi = min(x0,x2), max(x0,x2)
    return float(x1) if (gstar<lo or gstar>hi) else float(gstar)

# 注入–回收（补丁2：网格随机平移 + 抛物线细化）
def inject_recover_g(tau, w, g_true, n_boot, noise_tau_sig=0.03, drop_rate=0.2):
    rng = np.random.default_rng(SEED+7)
    base_spans = window_spans(tau.size, w, 1)
    g_hats=[]; samples=[]
    grid = G_GRID; step = float(grid[1]-grid[0])

    for _ in range(n_boot):
        # 每轮：抽窗
        spans = [s for s in base_spans if rng.random()>drop_rate] or base_spans[:]
        # 合成真 Δt
        dt_true=[]; x_cache=[]
        for (a,b) in spans:
            x = tau[a:b] - np.mean(tau[a:b])
            x += rng.normal(0, noise_tau_sig, size=b-a)
            x_cache.append(x)
            dt_true.append(float(np.sum(np.exp(g_true*x)-1.0)))
        # 网格随机平移
        shift = rng.uniform(-0.5*step, 0.5*step)
        ggrid = grid + shift
        # 打分
        scores=[]
        for g in ggrid:
            errs=[]
            for x,dt in zip(x_cache, dt_true):
                pred = float(np.sum(np.exp(g*x)-1.0))
                errs.append(abs(pred-dt))
            scores.append(np.median(errs) if errs else np.inf)
        scores = np.array(scores)
        i0 = int(np.argmin(scores))
        g_hat = _quad_refine(ggrid, scores, i0)
        g_hats.append(g_hat); samples.append(g_hat)

    g_hats = np.array(g_hats)
    ci_lo, ci_hi = np.percentile(g_hats, [16,84])

    # ΔAIC_Δt：以连续 g_best 近似
    def sse_for_g(g):
        spans = [s for s in base_spans if rng.random()>drop_rate] or base_spans[:]
        errs=[]
        for (a,b) in spans:
            x = tau[a:b] - np.mean(tau[a:b])
            x += rng.normal(0, noise_tau_sig, size=b-a)
            dt = float(np.sum(np.exp(g_true*x)-1.0))
            pred = float(np.sum(np.exp(g*x)-1.0))
            errs.append((pred-dt)**2)
        return float(np.sum(errs)), max(1,len(errs))
    g_best = float(np.median(g_hats))
    sse_best, n1 = sse_for_g(g_best)
    sse_null, n0 = sse_for_g(0.0)
    AIC_best = n1*np.log(max(sse_best,1e-12)/n1) + 2*1
    AIC_null = n0*np.log(max(sse_null,1e-12)/n0) + 2*0
    dAIC_dt = float(AIC_null - AIC_best)

    pd.DataFrame({"g_boot":g_hats}).to_csv(OUT/"g_bootstrap_samples.csv", index=False)
    return float(np.median(g_hats)), (float(ci_lo), float(ci_hi)), dAIC_dt

# ========== 3) 读 EHT uvfits ==========
if not Path(DST_UVFITS).exists():
    print("[download] fetching uvfits ...")
    urlretrieve(URL_UVFITS, DST_UVFITS)

sz = os.path.getsize(DST_UVFITS)
with fits.open(DST_UVFITS) as hdul:
    h = hdul[0].header
    print(f"[FITS] OBJECT:{h.get('OBJECT')} DATE-OBS:{h.get('DATE-OBS')} SIZE:{sz}")

uv = UVData()
uv.read_uvfits(DST_UVFITS)

data = uv.data_array
flag = uv.flag_array
nsmp = uv.nsample_array

# 兼容 3D/4D 形状
if data.ndim == 3:
    Nblts, Nfreq, Npol = data.shape
    data = data.reshape(Nblts, 1, Nfreq, Npol)
    flag = flag.reshape(Nblts, 1, Nfreq, Npol)
    nsmp = np.ones_like(flag, dtype=float) if nsmp is None else nsmp.reshape(Nblts,1,Nfreq,Npol)
else:
    Nblts, Nspw, Nfreq, Npol = data.shape
    nsmp = np.ones_like(flag, dtype=float) if nsmp is None else nsmp

freq_hz = float(uv.freq_array.flatten()[0])
c = 299792458.0
lam = c / freq_hz

uvw = uv.uvw_array
u_l = uvw[:,0] / lam
v_l = uvw[:,1] / lam
uvd = np.hypot(u_l, v_l)
theta = (np.degrees(np.arctan2(v_l,u_l)) % 360.0)

w = (~flag).astype(float) * nsmp
num = np.sum(data * w, axis=(1,2,3))
den = np.sum(w, axis=(1,2,3)) + 1e-12
vis = num / den
keep = (den > 0)
u_l = u_l[keep]; v_l=v_l[keep]; uvd=uvd[keep]; theta=theta[keep]
amp = np.abs(vis[keep])

print(f"[EHT] Nvis={amp.size}, freq={freq_hz/1e9:.3f} GHz, |u| range λ^-1: {uvd.min():.2e}–{uvd.max():.2e}")

# ========== 4) 振幅模型（各向异性散射 + bin-wise 逐点预测） ==========
def fwhm_to_sigma(fwhm_rad):
    return fwhm_rad / (2*np.sqrt(2*np.log(2)))

def model_amp_points(u, v, D_uas, A, Fmaj_uas, Fmin_uas, PA_deg):
    # ring: |J0(pi D q)|
    D_rad = D_uas * (np.pi/180/3600/1e6)
    q = np.hypot(u, v)
    ring = np.abs(j0(np.pi * D_rad * q))
    # anisotropic scatter (Gaussian in image -> Gaussian in uv)
    PA = np.deg2rad(PA_deg)
    up =  u*np.cos(PA) + v*np.sin(PA)
    vp = -u*np.sin(PA) + v*np.cos(PA)
    sigx = fwhm_to_sigma(Fmaj_uas * (np.pi/180/3600/1e6))
    sigy = fwhm_to_sigma(Fmin_uas * (np.pi/180/3600/1e6))
    scat = np.exp(-2*(np.pi**2)*(sigx**2 * up**2 + sigy**2 * vp**2))
    return A * ring * scat

def bin_by_uv_and_sector(u, v, amp, uvd, theta, lo_G, hi_G, nbins, sectors_deg=None):
    lo, hi = lo_G*1e9, hi_G*1e9
    m = (uvd>=lo) & (uvd<=hi)
    if not np.any(m): return None
    u, v, amp, uvd, theta = u[m], v[m], amp[m], uvd[m], theta[m]

    edges = np.linspace(lo, hi, nbins+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    med_amp=[]; std_amp=[]; counts=[]; masks=[]
    for a,b in zip(edges[:-1], edges[1:]):
        mm = (uvd>=a)&(uvd<b)
        masks.append(mm)
        if np.any(mm):
            vals = amp[mm]
            med_amp.append(np.median(vals))
            std_amp.append(np.std(vals))
            counts.append(int(vals.size))
        else:
            med_amp.append(np.nan); std_amp.append(np.nan); counts.append(0)
    res = {"edges":edges, "centers":centers,
           "med_amp":np.array(med_amp), "std_amp":np.array(std_amp),
           "counts":np.array(counts), "u":u, "v":v, "amp":amp, "uvd":uvd, "theta":theta,
           "masks":masks}

    # 扇区（简要：各扇区用等效各向同性估 D）
    if sectors_deg is not None:
        sec_rows=[]
        for s in sectors_deg:
            s_lo, s_hi = s, s+45
            ms = (theta>=s_lo) & (theta<s_hi)
            if not np.any(ms):
                sec_rows.append((f"{s}-{s+45}", np.nan, np.nan, 0, 0)); continue
            u_s, v_s, a_s = u[ms], v[ms], amp[ms]
            q_s = np.hypot(u_s, v_s)
            # bin in this sector (same edges)
            med=[]; std=[]; nbin=0
            for a,b in zip(edges[:-1], edges[1:]):
                mm = (q_s>=a)&(q_s<b)
                if np.any(mm):
                    med.append(np.median(a_s[mm]))
                    std.append(np.std(a_s[mm])+1e-6)
                    nbin += 1
                else:
                    med.append(np.nan); std.append(np.nan)
            med=np.array(med); std=np.array(std)
            ok = np.isfinite(med)
            if np.sum(ok) < 6:
                sec_rows.append((f"{s}-{s+45}", np.nan, np.nan, int(np.sum(ok)), int(np.sum(ms))))
                continue
            Uc = centers[ok]; Y = med[ok]; S = std[ok]
            def r_iso(p):
                D_uas, A = p
                D_rad = D_uas * (np.pi/180/3600/1e6)
                pred = A * np.abs(j0(np.pi * D_rad * Uc))
                return (pred - Y)/S
            sol = least_squares(r_iso, x0=[60.0, 1.0], bounds=([30.0, 1e-3],[120.0, 10.0]))
            D_hat = float(sol.x[0]); rmse = float(np.sqrt(np.mean(r_iso(sol.x)**2)))
            sec_rows.append((f"{s}-{s+45}", D_hat, rmse, int(np.sum(ok)), int(np.sum(ms))))
        sec_df = pd.DataFrame(sec_rows, columns=["sector","D_uas","RMSE","Nbins","Nvis"])
        res["sector_df"] = sec_df
    return res

def model_pred_binwise(R, params):
    # 用真实数据点逐点预测 → 取每个 bin 的预测中位数（与观测中位数配对）
    D_uas, A, Fmaj_uas, Fmin_uas, PA_deg = params
    pred_meds=[]
    for mm in R["masks"]:
        if not np.any(mm):
            pred_meds.append(np.nan); continue
        u, v = R["u"][mm], R["v"][mm]
        yhat = model_amp_points(u, v, D_uas, A, Fmaj_uas, Fmin_uas, PA_deg)
        pred_meds.append(float(np.median(yhat)))
    return np.array(pred_meds)

def fit_ring_aniso(R, fix_D=None):
    med = R["med_amp"].copy()
    std = R["std_amp"].copy()
    ok = np.isfinite(med) & (R["counts"]>0)
    if np.sum(ok) < 6:
        return {"theta":[np.nan]* (5 if fix_D is None else 4), "sse":np.nan, "n":int(np.sum(ok)),
                "aic":np.nan, "rmse":np.nan}
    Y = med[ok]; S = std[ok] + 1e-6

    if fix_D is None:
        # 初值/边界（更稳健）
        p0 = [60.0, 0.8, 25.0, 10.0, 90.0]  # D, A, Fmaj, Fmin, PA
        lb = [30.0, 1e-3,  0.0,  0.0,  0.0]
        ub = [120.0, 10.0, 80.0, 80.0, 180.0]
        def resid(p):
            pred = model_pred_binwise(R, p)
            return (pred[ok] - Y)/S
        # 防止初残差 NaN
        r0 = resid(p0); 
        if not np.all(np.isfinite(r0)):
            p0 = [60.0, 1.0, 10.0, 10.0, 0.0]
        sol = least_squares(resid, x0=p0, bounds=(lb,ub))
        theta = list(sol.x); k = 5
        res = resid(sol.x)
    else:
        p0 = [0.8, 25.0, 10.0, 90.0]  # A, Fmaj, Fmin, PA
        lb = [1e-3, 0.0, 0.0, 0.0]
        ub = [10.0, 80.0, 80.0, 180.0]
        def resid(p):
            pp = [fix_D] + list(p)
            pred = model_pred_binwise(R, pp)
            return (pred[ok] - Y)/S
        r0 = resid(p0)
        if not np.all(np.isfinite(r0)):
            p0 = [1.0, 10.0, 10.0, 0.0]
        sol = least_squares(resid, x0=p0, bounds=(lb,ub))
        theta = [fix_D] + list(sol.x); k = 4
        res = resid(sol.x)

    n = int(np.sum(ok))
    sse = float(np.sum((res*S)**2))
    aic = n*np.log(max(sse,1e-12)/n) + 2*k
    rmse = float(np.sqrt(np.mean(res**2)))
    return {"theta":theta, "sse":sse, "n":n, "aic":aic, "rmse":rmse}

def run_amp_pipeline(u_l, v_l, amp, uvd, theta, uv_windows, nbins, sectors_deg):
    rows=[]; agree_cells=[]
    for (lo,hi) in uv_windows:
        R = bin_by_uv_and_sector(u_l, v_l, amp, uvd, theta, lo, hi, nbins, sectors_deg)
        if R is None: continue
        # 存 profile
        pd.DataFrame({"uv":R["centers"], "med_amp":R["med_amp"], "std_amp":R["std_amp"], "counts":R["counts"]}).to_csv(
            OUT/f"sgrA_amp_profile_{lo}_{hi}.csv", index=False)

        # 拟合：自由D vs GR固定D
        fit_free = fit_ring_aniso(R, fix_D=None)
        fit_gr   = fit_ring_aniso(R, fix_D=D_GR_UAS)
        dAIC = fit_gr["aic"] - fit_free["aic"]
        D_free = fit_free["theta"][0]
        rows.append({"window":f"{lo}-{hi} Gλ",
                     "D_free":D_free,
                     "AIC_free":fit_free["aic"], "AIC_GR":fit_gr["aic"],
                     "ΔAIC_amp":dAIC,
                     "RMSE_free":fit_free["rmse"], "RMSE_GR":fit_gr["rmse"]})
        # bin-wise 逐点预测曲线图
        x = R["centers"]; y = R["med_amp"]; ok = np.isfinite(y)
        yhat_free = model_pred_binwise(R, fit_free["theta"])
        yhat_gr   = model_pred_binwise(R, [D_GR_UAS] + fit_gr["theta"][1:])
        fig,ax=plt.subplots(figsize=(5.6,3.6))
        ax.errorbar(x[ok], y[ok], yerr=R["std_amp"][ok], fmt="o", ms=3, lw=1, label="data")
        ax.plot(x, yhat_free, label=f"OGU freeD (D≈{D_free:.2f} μas)")
        ax.plot(x, yhat_gr,   label=f"GR fixed D=52 μas")
        ax.set_xlabel("|u| [λ^-1]"); ax.set_ylabel("median |V|")
        ax.set_title(f"Sgr A* amp fit ({lo}-{hi} Gλ)  ΔAIC≈{dAIC:.2f}")
        ax.legend(); plt.tight_layout(); plt.savefig(OUT/f"sgrA_ring_scat_fit_{lo}_{hi}.png"); plt.close()

        # 扇区结果
        if "sector_df" in R:
            sec = R["sector_df"]; sec.to_csv(OUT/f"sgrA_sector_{lo}_{hi}.csv", index=False)
            fig,ax=plt.subplots(figsize=(6.4,3.4))
            ax.bar(sec["sector"], sec["D_uas"])
            ax.axhline(D_GR_UAS, ls="--", lw=1, label="GR 52 μas")
            ax.set_ylabel("D [μas]"); ax.set_title(f"Sector D ({lo}-{hi} Gλ)")
            ax.legend(); plt.tight_layout(); plt.savefig(OUT/f"sgrA_sector_{lo}_{hi}.png"); plt.close()
            # 锁定格：以该窗 ΔAIC>10 视为支持 OGU
            agree_cells.append(int(dAIC>10))
        else:
            agree_cells.append(0)

        # 保存 bin-wise 逐点预测的原始点（用于复现）
        bw_rows=[]
        for idx,mm in enumerate(R["masks"]):
            if not np.any(mm): continue
            u,v = R["u"][mm], R["v"][mm]
            bw_rows.append([idx, float(R["centers"][idx]), int(mm.sum())])
        pd.DataFrame(bw_rows, columns=["bin_id","uv_center","Npts"]).to_csv(
            OUT/f"sgrA_binwise_pred_{lo}_{hi}.csv", index=False)

    lockrate = float(np.mean(agree_cells)) if agree_cells else np.nan
    with open(OUT/"lockrate_amp.json","w") as f:
        json.dump({"windows":[f"{a}-{b}" for (a,b) in uv_windows], "cells_supporting_OGU":agree_cells, "lock_rate":lockrate}, f)
    return pd.DataFrame(rows)

# ========== 5) 主流程 ==========
# 5.1 ζ–phase / SCI
gammas = load_zetazeros(N_ZERO) if USE_TRUE_ZEROS else None
tau, dbar, base_span = unfold_to_tau(gammas, BASE_M, 0)
print(f"[zeros] N={gammas.size}, τ_N={tau.size}, Δγ_base≈{dbar:.6f} (base gaps {base_span})")
np.savetxt(OUT/"tau_unfolded.txt", tau)

grid_rows=[]
for w in W_LIST:
    for s in STRIDE_LIST:
        K,Kt,spans = sci_stats_windows(tau, w, s)
        Dg,pg = ks_against(K,"GUE"); Dp,pp = ks_against(K,"Poisson")
        Dgr,pgr = ks_against(Kt,"GUE"); Dpr,ppr = ks_against(Kt,"Poisson")
        grid_rows.append({
            "w":w,"stride":s,"Nwin":len(spans),
            "KS_K_vs_GUE_D":Dg,"KS_K_vs_GUE_p":pg,
            "KS_K_vs_Poiss_D":Dp,"KS_K_vs_Poiss_p":pp,
            "KS_Kt_vs_GUE_D":Dgr,"KS_Kt_vs_GUE_p":pgr,
            "KS_Kt_vs_Poiss_D":Dpr,"KS_Kt_vs_Poiss_p":ppr,
            "K_med":np.median(K),"K_iqr":np.percentile(K,75)-np.percentile(K,25),
            "Kt_med":np.median(Kt),"Kt_iqr":np.percentile(Kt,75)-np.percentile(Kt,25)
        })
grid = pd.DataFrame(grid_rows).sort_values(["w","stride"])
grid.to_csv(OUT/"grid_scan_summary.csv", index=False)
print("\n[SCI] grid head:")
print(grid.head(9).to_string(index=False))

# 5.2 g 扫描（naive）
vals,_ = gscan_medabs_dt(tau, W_FOR_GSCAN, G_GRID)
g_star = float(G_GRID[np.nanargmin(vals)])
fig,ax=plt.subplots(figsize=(5.2,3.2))
ax.plot(G_GRID, vals, marker="o", ms=2, lw=1)
ax.axvline(g_star, ls="--", lw=1)
ax.set_xlabel("g"); ax.set_ylabel("median |Δt|"); ax.set_title(f"g-scan naive @ W={W_FOR_GSCAN} → g*≈{g_star:.3f}")
plt.tight_layout(); plt.savefig(OUT/f"gscan_medabsdt_W{W_FOR_GSCAN}.png"); plt.close()
pd.DataFrame({"g":G_GRID,"med_abs_dt":vals}).to_csv(OUT/f"gscan_W{W_FOR_GSCAN}.csv", index=False)
print(f"\n[g-scan naive] g*≈{g_star:.6f}")

# 5.3 注入–回收（现实噪声+抽窗 + 连续 g_hat）
g_true = 0.150
g_hat, (g_lo,g_hi), dAIC_dt = inject_recover_g(tau, W_FOR_GSCAN, g_true, N_BOOT_G, NOISE_TAU_SIG, DROP_RATE)
print(f"[inject–recover] g_true={g_true:.3f} → g^≈{g_hat:.3f} (68% CI [{g_lo:.3f},{g_hi:.3f}]),  ΔAIC_Δt={dAIC_dt:.2f}")
json.dump({"g_true":g_true,"g_hat":g_hat,"ci":[g_lo,g_hi],"ΔAIC_Δt":dAIC_dt},
          open(OUT/"gscan_inject_recover_summary.json","w"))
# 小图：自举分布
boot = pd.read_csv(OUT/"g_bootstrap_samples.csv")["g_boot"].values
fig,ax=plt.subplots(figsize=(5.0,3.0))
ax.hist(boot, bins=24, density=True)
ax.axvline(np.median(boot), color="k", ls="--", lw=1)
ax.set_xlabel("g^ (bootstrap)"); ax.set_ylabel("pdf"); ax.set_title("g^ bootstrap histogram")
plt.tight_layout(); plt.savefig(OUT/"gscan_inject_recover.png"); plt.close()
pd.DataFrame({"g":boot}).to_csv(OUT/"gscan_inject_recover_curve.csv", index=False)

# 5.4 振幅：各向异性散射 + bin-wise 逐点预测 + 扇区 ΔAIC
df_amp = run_amp_pipeline(u_l, v_l, amp, uvd, theta, UV_WINDOWS_G, NBINS, SECTORS_DEG)
df_amp.to_csv(OUT/"sgrA_amp_summary.csv", index=False)

# 5.5 联合 ΔAIC
dAIC_amp_total = float(np.nansum(df_amp["ΔAIC_amp"])) if not df_amp.empty else 0.0
dAIC_joint = dAIC_dt + dAIC_amp_total
json.dump({"ΔAIC_amp_total":dAIC_amp_total, "ΔAIC_Δt":dAIC_dt, "ΔAIC_joint":dAIC_joint},
          open(OUT/"joint_aic.json","w"))

# ========== 6) 汇总 ==========
print("\n=== SUMMARY (hard KPIs) ===")
g0 = pd.read_csv(OUT/"grid_scan_summary.csv").iloc[0]
print(f"SCI 分离: KS(K vs GUE) D≈{g0['KS_K_vs_GUE_D']:.3f}, p≈{g0['KS_K_vs_GUE_p']:.1e}")
print(f"g 扫描（naive）: g*≈{g_star:.3f}")
print(f"注入–回收: g_true={g_true:.3f}, g^≈{g_hat:.3f}, 68% CI=[{g_lo:.3f},{g_hi:.3f}],  ΔAIC_Δt≈{dAIC_dt:.2f}")
if not df_amp.empty:
    for _,row in df_amp.iterrows():
        print(f"振幅 ΔAIC_amp（{row['window']}）: OGU D≈{row['D_free']:.2f} μas  vs  GR {D_GR_UAS:.0f} μas,  ΔAIC_amp≈{row['ΔAIC_amp']:.2f}")
else:
    print("振幅 ΔAIC_amp：无有效窗口（数据/筛选为空）")
print(f"联合 ΔAIC: ΔAIC_joint≈{dAIC_joint:.2f}  （>10 强证据，>20 极强）")

print("\nOutputs in:", OUT)
for p in sorted(OUT.glob("*")):
    print(" -", p.name)
