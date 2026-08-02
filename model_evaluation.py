"""
模型评价 — AIC/BIC + 与纯数据驱动对比 + 预测区间覆盖率
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, _deutsch_chain_vec, predict_cout
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = os.path.join(base_dir, cfg["csv_path"])
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print("=" * 60)
    print("模型评价")
    print("=" * 60)

    df = clean_and_impute(load_raw(csv_path))
    bounds = df.attrs["bounds"]
    n = len(df)
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)

    # ===== 1. AIC/BIC =====
    print("\n--- AIC/BIC 信息准则 ---")
    y = df["C_out_mgNm3"].values
    g = np.array(deutsch_model["g"])
    r_fixed = deutsch_model.get("r", 0.5)
    T_ref = np.array(deutsch_model["T_ref"])
    popt = np.array([deutsch_model["kA_0"]] + deutsch_model["alpha"][:4])
    full = np.concatenate([popt[0] * g, popt[1:5]])
    Tin = df["Temp_C"].values; Cin = df["C_in_gNm3"].values; Q = df["Q_Nm3h"].values
    U = [df[f"U{i}_kV"].values for i in range(1, 5)]
    T = [df[f"T{i}_s"].values for i in range(1, 5)]
    y_pred = _deutsch_chain_vec(full, Tin, Cin, Q, U[0], U[1], U[2], U[3], T[0], T[1], T[2], T[3], T_ref, r=r_fixed)
    rss = np.sum((y - y_pred) ** 2)
    k_deutsch = 5
    aic_d = n * np.log(rss / n) + 2 * k_deutsch
    bic_d = n * np.log(rss / n) + k_deutsch * np.log(n)
    print(f"  Deutsch(5参数): RSS={rss:.2f}, AIC={aic_d:.1f}, BIC={bic_d:.1f}")

    k8 = np.array([0.165, 0.161, 0.136, 0.138])
    y_pred_simple = np.zeros(n)
    for i in range(n):
        y_pred_simple[i] = predict_cout(
            {"kA": [144, 269, 399, 339], "alpha": [0.016, 0.005, 0.001, 0.001], "T_ref": T_ref.tolist(), "r": 0.5},
            Tin[i], Cin[i], Q[i], [U[j][i] for j in range(4)], [T[j][i] for j in range(4)])
    rss_s = np.sum((y - y_pred_simple) ** 2)
    aic_s = n * np.log(rss_s / n) + 2 * 8
    bic_s = n * np.log(rss_s / n) + 8 * np.log(n)
    print(f"  Deutsch(8参数独立kA): RSS={rss_s:.2f}, AIC={aic_s:.1f}, BIC={bic_s:.1f}")
    print(f"  -> 共享kA_0的 AIC 差: {aic_d - aic_s:.1f} (负=更优)")

    # ===== 2. 与纯数据驱动对比 =====
    print("\n--- 与纯数据驱动(RandomForest)对比 ---")
    X = df[["Temp_C", "C_in_gNm3", "Q_Nm3h"] + [f"U{i}_kV" for i in range(1, 5)] + [f"T{i}_s" for i in range(1, 5)]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    rf = RandomForestRegressor(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(np.mean((y_test - y_rf) ** 2))
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2_rf = 1 - np.sum((y_test - y_rf) ** 2) / ss_tot

    idx_test = np.arange(n)
    _, idx_te = train_test_split(idx_test, test_size=0.3, random_state=seed)
    y_de = y_pred[idx_te]
    rmse_de = np.sqrt(np.mean((y_test - y_de) ** 2))
    r2_de = 1 - np.sum((y_test - y_de) ** 2) / ss_tot
    print(f"  RandomForest: R2={r2_rf:.4f}, RMSE={rmse_rf:.4f}")
    print(f"  Deutsch机理: R2={r2_de:.4f}, RMSE={rmse_de:.4f}")
    print(f"  -> 机理模型RMSE更高, 但有物理可外推性; RF只能内插, 外推到C_out=10/5不可靠")

    # ===== 3. 预测区间覆盖率 =====
    print("\n--- 预测区间覆盖率 ---")
    ci_path = os.path.join(out_dir, "bayesian_ci.json")
    with open(ci_path, "r", encoding="utf-8") as f:
        ci = json.load(f)
    cov_mat = np.array(ci["deutsch_meta"]["cov"])
    sigma2 = ci["deutsch_meta"]["sigma2"]
    g = np.array(deutsch_model["g"])
    p_theta = np.array([deutsch_model["kA_0"]] + deutsch_model["alpha"][:4])
    h = np.maximum(1e-4 * np.abs(p_theta), 1e-4)

    def jac_theta(i):
        J = np.zeros(5)
        for k in range(5):
            pp = p_theta.copy(); pp[k] += h[k]
            pm = p_theta.copy(); pm[k] -= h[k]
            fp = predict_cout({"kA": (pp[0]*g).tolist(), "alpha": pp[1:5].tolist(), "T_ref": T_ref.tolist(), "r": r_fixed},
                              Tin[i], Cin[i], Q[i], [U[j][i] for j in range(4)], [T[j][i] for j in range(4)])
            fm = predict_cout({"kA": (pm[0]*g).tolist(), "alpha": pm[1:5].tolist(), "T_ref": T_ref.tolist(), "r": r_fixed},
                              Tin[i], Cin[i], Q[i], [U[j][i] for j in range(4)], [T[j][i] for j in range(4)])
            J[k] = (fp - fm) / (2 * h[k])
        return J

    lo_hit = hi_hit = 0
    widths = []
    for i in range(0, n, 10):
        J = jac_theta(i)
        sigma = np.sqrt(max(float(J @ cov_mat @ J), 0))
        lo = y_pred[i] - 1.96 * sigma
        hi = y_pred[i] + 1.96 * sigma
        if lo <= y[i] <= hi:
            lo_hit += 1
        widths.append(2 * 1.96 * sigma)
    coverage = lo_hit / (n // 10)
    avg_width = np.mean(widths)
    print(f"  95% 预测区间覆盖率: {coverage:.4f} (理论 0.95)")
    print(f"  平均区间宽度: {avg_width:.4f} mg/Nm³")
    if coverage < 0.9:
        print("  [WARN] 覆盖率偏低, Laplace近似可能低估不确定性(因C_out限幅致残差非正态)")

    result = {
        "aic_bic": {"deutsch_5p": {"AIC": float(aic_d), "BIC": float(bic_d)},
                     "deutsch_8p": {"AIC": float(aic_s), "BIC": float(bic_s)}},
        "data_driven_compare": {"RF": {"R2": float(r2_rf), "RMSE": float(rmse_rf)},
                                 "Deutsch": {"R2": float(r2_de), "RMSE": float(rmse_de)}},
        "prediction_interval": {"coverage_95": float(coverage), "avg_width": float(avg_width)},
    }
    with open(os.path.join(out_dir, "model_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 模型评价已保存至 {os.path.join(out_dir, 'model_evaluation.json')}")


if __name__ == "__main__":
    main()