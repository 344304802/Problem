import numpy as np


def priority_rule(sens):
    # 优先用无量纲弹性系数 E=∂ln y/∂ln x, 消除电压(kV)与振打(s)量纲差异
    # 性价比 = |E^C / E^P| = |(∂ln C)/(∂ln P)|, 单位统一为 kW/(mg/Nm³), 可直接比较
    # 电耗模型 P=ΣkU²+c 不含 T, ∂P/∂T≈0, 振打能降排放但不增电耗(近似免费), ratio=inf
    if "EC_U" in sens:
        EC_U = np.array(sens["EC_U"]); EC_T = np.array(sens["EC_T"])
        EP_U = np.array(sens["EP_U"]); EP_T = np.array(sens["EP_T"])
        kind = "弹性系数"
    else:
        EC_U = np.array(sens["SC_U"]); EC_T = np.array(sens["SC_T"])
        EP_U = np.array(sens["SP_U"]); EP_T = np.array(sens["SP_T"])
        kind = "原始灵敏度(量纲未统一)"

    # 阈值 1e-6: |EP| 小于此视为电耗对该变量不敏感
    eps_p = 1e-6
    ratio_U = np.where(np.abs(EP_U) > eps_p, np.abs(EC_U) / np.abs(EP_U), np.inf)
    ratio_T = np.where(np.abs(EP_T) > eps_p, np.abs(EC_T) / np.abs(EP_T), np.inf)
    free_T = np.abs(EP_T) <= eps_p
    free_U = np.abs(EP_U) <= eps_p

    avg_ratio_U = float(np.mean(ratio_U[np.isfinite(ratio_U)])) if np.any(np.isfinite(ratio_U)) else 0.0
    avg_ratio_T = float(np.mean(ratio_T[np.isfinite(ratio_T)])) if np.any(np.isfinite(ratio_T)) else 0.0
    n_free_T = int(np.sum(free_T))
    n_free_U = int(np.sum(free_U))

    if n_free_T > 0 and np.any(np.abs(EC_T[free_T]) > eps_p):
        priority = "优先调振打"
        reason = f"振打对电耗无影响(∂P/∂T≈0)且能降排放, 视为免费手段 ({kind})"
    elif n_free_U > 0 and np.any(np.abs(EC_U[free_U]) > eps_p):
        priority = "优先调电压"
        reason = f"电压对电耗无影响(∂P/∂U≈0)且能降排放, 视为免费手段 ({kind})"
    elif avg_ratio_U > avg_ratio_T:
        priority = "优先调电压"
        reason = f"电压性价比 {avg_ratio_U:.6f} > 振打性价比 {avg_ratio_T:.6f} ({kind})"
    else:
        priority = "优先调振打"
        reason = f"振打性价比 {avg_ratio_T:.6f} >= 电压性价比 {avg_ratio_U:.6f} ({kind})"

    print(f"[INFO] 优先级判定: {priority} ({reason})")
    return {
        "ratio_U": ratio_U.tolist(), "ratio_T": ratio_T.tolist(),
        "avg_ratio_U": avg_ratio_U, "avg_ratio_T": avg_ratio_T,
        "priority": priority, "reason": reason, "kind": kind,
        "n_free_T": n_free_T, "n_free_U": n_free_U,
    }