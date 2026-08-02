import numpy as np


def priority_rule(sens):
    # 优先用无量纲弹性系数 E=∂ln y/∂ln x, 消除电压(kV)与振打(s)量纲差异
    # 性价比 = |E^C / E^P| = |(∂ln C)/(∂ln P)|, 单位统一为 kW/(mg/Nm³), 可直接比较
    if "EC_U" in sens:
        EC_U = np.array(sens["EC_U"]); EC_T = np.array(sens["EC_T"])
        EP_U = np.array(sens["EP_U"]); EP_T = np.array(sens["EP_T"])
        kind = "弹性系数"
    else:
        EC_U = np.array(sens["SC_U"]); EC_T = np.array(sens["SC_T"])
        EP_U = np.array(sens["SP_U"]); EP_T = np.array(sens["SP_T"])
        kind = "原始灵敏度(量纲未统一)"

    ratio_U = np.abs(EC_U) / (np.abs(EP_U) + 1e-12)
    ratio_T = np.abs(EC_T) / (np.abs(EP_T) + 1e-12)

    avg_ratio_U = float(np.mean(ratio_U))
    avg_ratio_T = float(np.mean(ratio_T))

    if avg_ratio_U > avg_ratio_T:
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
    }