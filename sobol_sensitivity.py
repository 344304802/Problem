"""
全局敏感性分析 (Sobol 指数) — 识别主效应/交互效应
对 8 个决策变量 (U1-4, T1-4) 算 C_out 和 P 的一阶/总效应 Sobol 指数
补充局部灵敏度(雅可比), 给全局视角
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from SALib.sample import saltelli
from SALib.analyze import sobol
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, predict_cout
from optim.regime import cluster_regimes


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = os.path.join(base_dir, cfg["csv_path"])
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print("=" * 60)
    print("全局敏感性分析 (Sobol 指数)")
    print("=" * 60)

    df = clean_and_impute(load_raw(csv_path))
    bounds = df.attrs["bounds"]
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)
    model = {"power": power_model, "deutsch": deutsch_model}

    regimes = cluster_regimes(df, k=cfg.get("regime_k", 5), seed=seed)
    high_regime = max(regimes["regimes"], key=lambda r: r["mean"]["C_in"])
    Tin = high_regime["mean"]["Temp"]
    Cin = high_regime["mean"]["C_in"]
    Q = high_regime["mean"]["Q"]
    params = deutsch_model
    print(f"[INFO] 工况{high_regime['id']} (C_in={Cin:.2f})")

    problem = {
        "num_vars": 8,
        "names": ["U1", "U2", "U3", "U4", "T1", "T2", "T3", "T4"],
        "bounds": [
            [bounds["U1"][0], bounds["U1"][1]],
            [bounds["U2"][0], bounds["U2"][1]],
            [bounds["U3"][0], bounds["U3"][1]],
            [bounds["U4"][0], bounds["U4"][1]],
            [bounds["T1"][0], min(bounds["T1"][1], bounds["T_crit1"])],
            [bounds["T2"][0], min(bounds["T2"][1], bounds["T_crit2"])],
            [bounds["T3"][0], min(bounds["T3"][1], bounds["T_crit3"])],
            [bounds["T4"][0], min(bounds["T4"][1], bounds["T_crit4"])],
        ],
    }

    N = 1024
    np.random.seed(seed)
    X = saltelli.sample(problem, N)
    print(f"[INFO] Saltelli 采样: N={N}, 总样本={len(X)}")

    Y_c = np.zeros(len(X))
    Y_p = np.zeros(len(X))
    for i, x in enumerate(X):
        U, T = x[:4], x[4:8]
        Y_c[i] = predict_cout(params, Tin, Cin, Q, U, T)
        Y_p[i] = predict_power(power_model, U, T=T)

    print("\n--- C_out Sobol 指数 ---")
    np.random.seed(seed)
    Si_c = sobol.analyze(problem, Y_c, print_to_console=False)
    print(f"{'变量':>6} {'一阶 S1':>10} {'总效应 ST':>10} {'交互 ST-S1':>10}")
    for i, name in enumerate(problem["names"]):
        inter = Si_c["ST"][i] - Si_c["S1"][i]
        print(f"{name:>6} {Si_c['S1'][i]:>10.4f} {Si_c['ST'][i]:>10.4f} {inter:>10.4f}")

    print("\n--- P Sobol 指数 ---")
    np.random.seed(seed)
    Si_p = sobol.analyze(problem, Y_p, print_to_console=False)
    print(f"{'变量':>6} {'一阶 S1':>10} {'总效应 ST':>10} {'交互 ST-S1':>10}")
    for i, name in enumerate(problem["names"]):
        inter = Si_p["ST"][i] - Si_p["S1"][i]
        print(f"{name:>6} {Si_p['S1'][i]:>10.4f} {Si_p['ST'][i]:>10.4f} {inter:>10.4f}")

    result = {
        "regime_id": high_regime["id"], "Cin": Cin, "N": N,
        "cout": {
            "S1": Si_c["S1"].tolist(), "ST": Si_c["ST"].tolist(),
            "S1_conf": Si_c["S1_conf"].tolist(), "ST_conf": Si_c["ST_conf"].tolist(),
        },
        "power": {
            "S1": Si_p["S1"].tolist(), "ST": Si_p["ST"].tolist(),
            "S1_conf": Si_p["S1_conf"].tolist(), "ST_conf": Si_p["ST_conf"].tolist(),
        },
        "names": problem["names"],
    }
    out_path = os.path.join(out_dir, "sobol_indices.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Sobol 指数已保存至 {out_path}")


if __name__ == "__main__":
    main()