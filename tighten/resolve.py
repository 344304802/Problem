import numpy as np
from optim.solve import solve_all_regimes


def resolve_under_limit(regimes, model, bounds, C_limit_new=5.0, multi_start=10, seed=42):
    results = solve_all_regimes(regimes, model, bounds, C_limit=C_limit_new, multi_start=multi_start, seed=seed)
    print(f"[INFO] 收紧至 C_limit={C_limit_new} 重求解完成")
    return results


def delta_power(sol10_list, sol5_list):
    deltas = []
    for r10, r5 in zip(sol10_list, sol5_list):
        rid = r10["regime"]["id"]
        P10 = r10["sol"]["P"]
        P5 = r5["sol"]["P"]
        if P10 is not None and P5 is not None and P10 > 0:
            dp = (P5 - P10) / P10 * 100.0
        else:
            dp = None
        deltas.append({"regime_id": rid, "P10": P10, "P5": P5, "delta_pct": dp})
        print(f"[INFO] 工况{rid}: P10={P10}, P5={P5}, ΔP={dp}")

    valid = [d for d in deltas if d["delta_pct"] is not None]
    overall = np.mean([d["delta_pct"] for d in valid]) if valid else None
    return {"deltas": deltas, "overall_delta_pct": float(overall) if overall is not None else None}


def feasibility_check(sol5_list, bounds):
    results = []
    for r in sol5_list:
        rid = r["regime"]["id"]
        sol = r["sol"]
        if sol["U"] is None:
            results.append({"regime_id": rid, "feasible": False, "reason": "无解"})
            continue
        feasible = True
        reasons = []
        for i in range(4):
            u_lo, u_hi = bounds[f"U{i+1}"]
            if sol["U"][i] > u_hi:
                feasible = False
                reasons.append(f"U{i+1}={sol['U'][i]:.2f}超过上限{u_hi:.2f}")
            t_lo, t_hi = bounds[f"T{i+1}"]
            if sol["T"][i] < t_lo:
                feasible = False
                reasons.append(f"T{i+1}={sol['T'][i]:.2f}低于下限{t_lo:.2f}")
        results.append({"regime_id": rid, "feasible": feasible, "reason": "; ".join(reasons) if reasons else "可行"})
        if not feasible:
            print(f"[WARN] 工况{rid}不可行: {reasons}")
    return results