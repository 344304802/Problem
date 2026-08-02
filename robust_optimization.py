"""
鲁棒优化(机会约束) — P(C_out ≤ C_limit) ≥ 95%
利用贝叶斯 Laplace 协方差 Σ, 线性化: C_out ~ N(f_MLE, J_θ Σ J_θ^T)
约束变为: f_MLE(x) + z_0.95 · sqrt(J_θ Σ J_θ^T) ≤ C_limit
对比确定性优化(仅 f_MLE ≤ C_limit)的电耗代价
"""
import sys
import os
import json
import numpy as np
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, predict_cout
from optim.regime import cluster_regimes
from optim.solve import solve_all_regimes


def _cout_with_theta(params_base, p_theta, Tin, Cin, Q, U, T) :
    g = np.array(params_base["g"])
    kA = (p_theta[0] * g).tolist()
    alpha = p_theta[1:5].tolist()
    p = {"kA": kA, "alpha": alpha, "T_ref": params_base["T_ref"], "r": params_base.get("r", 0.5)}
    return predict_cout(p, Tin, Cin, Q, U, T)


def _jacobian_theta(params, Tin, Cin, Q, U, T) :
    g = np.array(params["g"])
    p_theta = np.array([params["kA_0"]] + params["alpha"][:4])
    n = len(p_theta)
    J = np.zeros(n)
    h = np.maximum(1e-4 * np.abs(p_theta), 1e-4)
    for k in range(n) :
        pp = p_theta.copy(); pp[k] += h[k]
        pm = p_theta.copy(); pm[k] -= h[k]
        fp = _cout_with_theta(params, pp, Tin, Cin, Q, U, T)
        fm = _cout_with_theta(params, pm, Tin, Cin, Q, U, T)
        J[k] = (fp - fm) / (2 * h[k])
    return J


def robust_solve_one(regime, model, bounds, cov, C_limit = 10.0, alpha = 0.95, multi_start = 10, seed = 42) :
    Tin = regime["mean"]["Temp"]
    Cin = regime["mean"]["C_in"]
    Q = regime["mean"]["Q"]
    params = model["deutsch"]
    power_model = model["power"]
    z = 1.6449 if abs(alpha - 0.95) < 1e-6 else 1.96

    lb = [bounds[f"U{i}"][0] for i in range(1, 5)] + [bounds[f"T{i}"][0] for i in range(1, 5)]
    ub = [bounds[f"U{i}"][1] for i in range(1, 5)] + [min(bounds[f"T{i}"][1], bounds[f"T_crit{i}"]) for i in range(1, 5)]
    bounds_arr = list(zip(lb, ub))
    np.random.seed(seed)

    def objective(x) :
        return predict_power(power_model, x[:4], T = x[4:8])

    def constraint_robust(x) :
        U, T = x[:4], x[4:8]
        f_mle = predict_cout(params, Tin, Cin, Q, U, T)
        J = _jacobian_theta(params, Tin, Cin, Q, U, T)
        var_f = float(J @ cov @ J)
        sigma_f = np.sqrt(max(var_f, 0))
        return C_limit - (f_mle + z * sigma_f)

    cons = [{"type": "ineq", "fun": constraint_robust}]
    best = None
    for start in range(multi_start) :
        if start == 0 :
            x0 = np.array([(lb[i] + ub[i]) / 2 for i in range(8)])
        else :
            x0 = np.array([np.random.uniform(lb[i], ub[i]) for i in range(8)])
        try :
            res = minimize(objective, x0, method="SLSQP", bounds=bounds_arr,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
            if res.success and constraint_robust(res.x) >= -1e-6 :
                P = predict_power(power_model, res.x[:4], T = res.x[4:8])
                if best is None or P < best["P"]:
                    f_mle = predict_cout(params, Tin, Cin, Q, res.x[:4], res.x[4:8])
                    J = _jacobian_theta(params, Tin, Cin, Q, res.x[:4], res.x[4:8])
                    sigma_f = np.sqrt(max(float(J @ cov @ J), 0))
                    best = {
                        "U": res.x[:4].tolist(), "T": res.x[4:8].tolist(),
                        "P": float(P), "Cout_mle": float(f_mle), "sigma": float(sigma_f),
                        "Cout_p95": float(f_mle + z * sigma_f), "success": True,
                    }
        except Exception :
            pass

    if best is None :
        print(f"[WARN] 工况{regime['id']} 鲁棒优化未收敛, 尝试DE")
        def de_obj(x) :
            if constraint_robust(x) < 0 :
                return 1e6 - constraint_robust(x) * 1e4
            return predict_power(power_model, x[:4], T = x[4:8])
        try :
            res = differential_evolution(de_obj, bounds = list(zip(lb, ub)), seed = seed, maxiter = 200, tol = 1e-8, polish = True, workers = 1)
            U, T = res.x[:4], res.x[4:8]
            f_mle = predict_cout(params, Tin, Cin, Q, U, T)
            J = _jacobian_theta(params, Tin, Cin, Q, U, T)
            sigma_f = np.sqrt(max(float(J @ cov @ J), 0))
            best = {
                "U": U.tolist(), "T": T.tolist(), "P": float(predict_power(power_model, U, T=T)),
                "Cout_mle": float(f_mle), "sigma": float(sigma_f),
                "Cout_p95": float(f_mle + z * sigma_f), "success": constraint_robust(res.x) >= -1e-6,
            }
        except Exception as e :
            print(f"[ERROR] DE也失败 : {e}")
            best = {"U": None, "T": None, "P": None, "success": False}

    print(f"[INFO] 工况{regime['id']}: P_robust={best.get('P')}, Cout_p95={best.get('Cout_p95')}, success={best['success']}")
    return best


def main() :
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = os.path.join(base_dir, cfg["csv_path"])
    out_dir = os.path.join(base_dir, cfg.get("out_dir", "outputs"))
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print("=" * 60)
    print("鲁棒优化(机会约束 P(C_out≤limit)≥95%)")
    print("=" * 60)

    df = clean_and_impute(load_raw(csv_path))
    bounds = df.attrs["bounds"]
    power_model = fit_power_model(df)
    deutsch_model = fit_deutsch_params(df, bounds)
    model = {"power": power_model, "deutsch": deutsch_model}

    ci_path = os.path.join(out_dir, "bayesian_ci.json")
    with open(ci_path, "r", encoding="utf-8") as f:
        ci = json.load(f)
    cov = np.array(ci["deutsch_meta"]["cov"])
    print(f"[INFO] 加载贝叶斯协方差矩阵 shape = {cov.shape}")

    k = cfg.get("regime_k", 5)
    regimes = cluster_regimes(df, k = k, seed = seed)
    C_limit = cfg.get("c_limit", 10.0)

    print("\n--- 确定性优化(基准) ---")
    sol_det = solve_all_regimes(regimes, model, bounds, C_limit=C_limit, multi_start=cfg.get("multi_start", 10), seed=seed)

    print("\n--- 鲁棒优化(机会约束) ---")
    sol_rob = []
    for r in regimes["regimes"]:
        best = robust_solve_one(r, model, bounds, cov, C_limit=C_limit, multi_start=cfg.get("multi_start", 10), seed=seed)
        sol_rob.append({"regime": r, "sol": best})

    print("\n--- 对比 ---")
    comparison = []
    for d, r in zip(sol_det, sol_rob) :
        rid = d["regime"]["id"]
        P_det = d["sol"]["P"]
        P_rob = r["sol"]["P"]
        dp = (P_rob - P_det) / P_det * 100 if P_det and P_rob else None
        comparison.append({"regime_id": rid, "P_det": P_det, "P_rob": P_rob, "delta_pct": dp,
                            "Cout_p95": r["sol"].get("Cout_p95"), "sigma": r["sol"].get("sigma")})
        print(f"工况{rid}: P_det={P_det:.1f}, P_robust={P_rob:.1f}, ΔP={dp:.2f}%, Cout_p95={r['sol'].get('Cout_p95'):.2f}, σ={r['sol'].get('sigma'):.2f}")

    avg_dp = np.mean([c["delta_pct"] for c in comparison if c["delta_pct"] is not None])
    print(f"\n鲁棒优化平均电耗代价 : +{avg_dp:.2f}%")

    result = {"comparison": comparison, "avg_delta_pct": float(avg_dp)}
    with open(os.path.join(out_dir, "robust_optimization.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent = 2, ensure_ascii = False)
    print(f"[INFO] 结果已保存至 {os.path.join(out_dir, 'robust_optimization.json')}")


if __name__ == "__main__":
    main()