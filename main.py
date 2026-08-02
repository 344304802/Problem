import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model
from modeling.deutsch import fit_deutsch_params, predict_cout
from modeling.feature import feature_importance
from optim.regime import cluster_regimes
from optim.solve import solve_all_regimes
from sensitivity.jacobian import numeric_jacobian
from sensitivity.priority import priority_rule
from sensitivity.compare import compare_two_regimes
from tighten.resolve import resolve_under_limit, delta_power, feasibility_check
from tighten.advice import high_conc_advice
from report.plots import (plot_relation_curves, plot_regime_scatter,
                          plot_param_compare, plot_sensitivity_heatmap, plot_delta_power,
                          plot_relation_3d)
from report.tables import to_markdown_tables


def load_config(path) :
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(cfg_path="config/config.yaml", skip_mysql=True):
    cfg = load_config(cfg_path)
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, cfg["csv_path"]) if not os.path.isabs(cfg["csv_path"]) else cfg["csv_path"]
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    os.makedirs(out_dir, exist_ok = True)

    print("=" * 60)
    print("步骤1: 数据加载与预处理")
    print("=" * 60)
    df = load_raw(csv_path)
    df = clean_and_impute(df)
    bounds = df.attrs["bounds"]

    print("\n" + "=" * 60)
    print("步骤2: 问题1 - 机理建模")
    print("=" * 60)
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)
    fi = feature_importance(df)

    model = {"power": power_model, "deutsch": deutsch_model}

    print("\n" + "=" * 60)
    print("步骤3: 问题2 - 工况划分与寻优")
    print("=" * 60)
    k = cfg.get("regime_k", 5)
    regimes = cluster_regimes(df, k = k, seed = seed)
    C_limit = cfg.get("c_limit", 10.0)
    sol_all = solve_all_regimes(regimes, model, bounds, C_limit = C_limit,
                               multi_start=cfg.get("multi_start", 10), seed=seed)

    print("\n" + "=" * 60)
    print("步骤4: 问题3 - 灵敏度分析")
    print("=" * 60)
    cmp = compare_two_regimes(sol_all, model, bounds)

    high_regime = cmp["regime_A"]
    high_sol = None
    for r in sol_all :
        if r["regime"]["id"] == high_regime["id"]:
            high_sol = r["sol"]
            break

    sens = None
    prio = None
    if high_sol and high_sol["U"]:
        x0 = high_sol["U"] + high_sol["T"]
        regime_obj = next(r["regime"] for r in sol_all if r["regime"]["id"] == high_regime["id"])
        sens = numeric_jacobian(model, x0, regime_obj, bounds, step_ratio=cfg.get("fd_step_ratio", 0.01))
        prio = priority_rule(sens)

    print("\n" + "=" * 60)
    print("步骤5: 问题4 - 排放收紧分析")
    print("=" * 60)
    C_limit_new = cfg.get("c_limit_new", 5.0)
    sol5_all = resolve_under_limit(regimes, model, bounds, C_limit_new = C_limit_new,
                                   multi_start=cfg.get("multi_start", 10), seed=seed)
    dp = delta_power(sol_all, sol5_all)
    feas = feasibility_check(sol5_all, bounds)

    advice = ""
    if high_sol and sol5_all :
        for r in sol5_all :
            if r["regime"]["id"] == high_regime["id"]:
                high_sol5 = r["sol"]
                regime_obj = next(rr["regime"] for rr in sol_all if rr["regime"]["id"] == high_regime["id"])
                advice = high_conc_advice(regime_obj, high_sol, high_sol5)
                break

    print("\n" + "=" * 60)
    print("步骤6: 输出与可视化")
    print("=" * 60)
    plot_relation_curves(model, df, out_dir)
    plot_relation_3d(model, df, out_dir)
    plot_regime_scatter(regimes, df, out_dir)
    plot_param_compare(cmp, out_dir)
    if sens :
        plot_sensitivity_heatmap(sens, out_dir)
    plot_delta_power(dp, out_dir)

    results = {
        "feature_importance": fi,
        "power_model": power_model,
        "deutsch_model": deutsch_model,
        "sol_all": sol_all,
        "compare": cmp,
        "priority": prio,
        "delta_power": dp,
        "feasibility": feas,
        "advice": advice,
    }

    md_str = to_markdown_tables(results)
    md_path = os.path.join(out_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_str)
    print(f"[INFO] 结果已保存至 {md_path}")

    meta = {
        "seed": seed,
        "n_rows": len(df),
        "k_regimes": regimes["k"],
        "C_limit": C_limit,
        "C_limit_new": C_limit_new,
        "power_r2": power_model["r2"],
        "deutsch_r2": deutsch_model["r2"] if deutsch_model else None,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent = 2, ensure_ascii = False)

    print("\n" + "=" * 60)
    print("全流程完成!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-mysql", action="store_true", default=True)
    args = parser.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(base, args.config)
    run_pipeline(cfg_path = cfg_path, skip_mysql = args.skip_mysql)