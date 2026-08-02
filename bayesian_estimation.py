"""
贝叶斯参数估计 (Laplace 近似) — 量化外推不确定性
Deutsch 非线性参数: 后验 ≈ N(MLE, 2σ²H⁻¹), H 为 loss Hessian
Power 线性参数: 后验 = N(β, σ²(XᵀX)⁻¹) 精确
95% CI = est ± 1.96·sqrt(diag(cov))
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model
from modeling.deutsch import fit_deutsch_params, _deutsch_chain_vec


def numeric_hessian(f, x, rel_step = 1e-4) :
    n = len(x)
    H = np.zeros((n, n))
    h = np.maximum(rel_step * np.abs(x), rel_step)
    for i in range(n) :
        for j in range(i, n) :
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


def deutsch_posterior_ci(df, deutsch_model, bounds) :
    g = np.array(deutsch_model["g"])
    r_fixed = deutsch_model.get("r", 0.5)
    T_ref = np.array(deutsch_model["T_ref"])
    kA_0 = deutsch_model["kA_0"]
    alpha = np.array(deutsch_model["alpha"])
    popt = np.array([kA_0, alpha[0], alpha[1], alpha[2], alpha[3]])

    Tin = df["Temp_C"].values
    Cin = df["C_in_gNm3"].values
    Q = df["Q_Nm3h"].values
    U = [df[f"U{i}_kV"].values for i in range(1, 5)]
    T = [df[f"T{i}_s"].values for i in range(1, 5)]
    y = df["C_out_mgNm3"].values

    def expand(p) :
        return np.concatenate([p[0] * g, p[1:5]])

    def loss(p) :
        full = expand(p)
        pred = _deutsch_chain_vec(full, Tin, Cin, Q, U[0], U[1], U[2], U[3],
                                  T[0], T[1], T[2], T[3], T_ref, r = r_fixed)
        rr = np.log(np.maximum(pred, 0.01)) - np.log(np.maximum(y, 0.01))
        return np.sum(rr ** 2)

    H = numeric_hessian(loss, popt)
    n = len(y)
    p = len(popt)
    sigma2 = loss(popt) / (n - p)
    try :
        cov = 2 * sigma2 * np.linalg.inv(H)
    except np.linalg.LinAlgError :
        cov = 2 * sigma2 * np.linalg.pinv(H)

    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = 1.96
    names = ["kA_0", "alpha_1", "alpha_2", "alpha_3", "alpha_4"]
    ci = {}
    for i, name in enumerate(names) :
        ci[name] = {
            "est": float(popt[i]),
            "se": float(se[i]),
            "ci_lo": float(popt[i] - z * se[i]),
            "ci_hi": float(popt[i] + z * se[i]),
        }
        rel = abs(se[i] / popt[i]) * 100 if popt[i] != 0 else float("inf")
        print(f"  {name} : est = {popt[i]:.4f}, se = {se[i]:.4f}, 95%CI = [{popt[i]-z*se[i]:.4f}, {popt[i]+z*se[i]:.4f}], 相对不确定 = {rel:.1f}%")

    eigvals = np.linalg.eigvalsh(H)
    print(f"  Hessian 特征值 : min = {eigvals.min():.4e}, max = {eigvals.max():.4e}, 条件数 = {eigvals.max()/max(eigvals.min(),1e-30):.2e}")
    return ci, {"cov": cov.tolist(), "sigma2": float(sigma2), "hessian_eigvals": eigvals.tolist()}


def power_posterior_ci(df, power_model) :
    if not power_model.get("extended", False):
        print("  [SKIP] 电耗模型非扩展形式, 跳过贝叶斯区间")
        return None, None

    U = np.column_stack([df[f"U{i}_kV"].values ** 2 for i in range(1, 5)] + [np.ones(len(df))])
    P = df["P_total_kW"].values
    beta = np.array(power_model["k_ext"] + [power_model["c_ext"]])
    n, p = U.shape
    P_pred = U @ beta
    rss = np.sum((P - P_pred) ** 2)
    sigma2 = rss / (n - p)
    try :
        cov = sigma2 * np.linalg.inv(U.T @ U)
    except np.linalg.LinAlgError :
        cov = sigma2 * np.linalg.pinv(U.T @ U)

    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = 1.96
    names = ["k_1", "k_2", "k_3", "k_4", "c"]
    ci = {}
    for i, name in enumerate(names) :
        ci[name] = {
            "est": float(beta[i]),
            "se": float(se[i]),
            "ci_lo": float(beta[i] - z * se[i]),
            "ci_hi": float(beta[i] + z * se[i]),
        }
        rel = abs(se[i] / beta[i]) * 100 if beta[i] != 0 else float("inf")
        print(f"  {name} : est = {beta[i]:.4f}, se = {se[i]:.4f}, 95%CI = [{beta[i]-z*se[i]:.4f}, {beta[i]+z*se[i]:.4f}], 相对不确定 = {rel:.1f}%")
    return ci, {"cov": cov.tolist(), "sigma2": float(sigma2)}


def main() :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = os.path.join(base_dir, cfg["csv_path"]) if not os.path.isabs(cfg["csv_path"]) else cfg["csv_path"]
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    os.makedirs(out_dir, exist_ok = True)

    print("=" * 60)
    print("贝叶斯参数估计 (Laplace 近似)")
    print("=" * 60)

    df = load_raw(csv_path)
    df = clean_and_impute(df)
    bounds = df.attrs["bounds"]

    print("\n--- 电耗模型参数后验区间 ---")
    power_model = fit_power_model(df)
    power_ci, power_meta = power_posterior_ci(df, power_model)

    print("\n--- Deutsch 模型参数后验区间 ---")
    deutsch_model = fit_deutsch_params(df, bounds)
    deutsch_ci, deutsch_meta = deutsch_posterior_ci(df, deutsch_model, bounds)

    result = {
        "power_ci": power_ci,
        "deutsch_ci": deutsch_ci,
        "power_meta": power_meta,
        "deutsch_meta": deutsch_meta,
    }
    out_path = os.path.join(out_dir, "bayesian_ci.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent = 2, ensure_ascii = False)
    print(f"\n[INFO] 贝叶斯区间已保存至 {out_path}")


if __name__ == "__main__":
    main()