import numpy as np


def priority_rule(sens):
    SC_U = np.array(sens["SC_U"])
    SC_T = np.array(sens["SC_T"])
    SP_U = np.array(sens["SP_U"])
    SP_T = np.array(sens["SP_T"])

    ratio_U = np.abs(SC_U) / (np.abs(SP_U) + 1e-12)
    ratio_T = np.abs(SC_T) / (np.abs(SP_T) + 1e-12)

    avg_ratio_U = float(np.mean(ratio_U))
    avg_ratio_T = float(np.mean(ratio_T))

    if avg_ratio_U > avg_ratio_T:
        priority = "优先调电压"
        reason = f"电压性价比 {avg_ratio_U:.6f} > 振打性价比 {avg_ratio_T:.6f}"
    else:
        priority = "优先调振打"
        reason = f"振打性价比 {avg_ratio_T:.6f} >= 电压性价比 {avg_ratio_U:.6f}"

    print(f"[INFO] 优先级判定: {priority} ({reason})")
    return {
        "ratio_U": ratio_U.tolist(), "ratio_T": ratio_T.tolist(),
        "avg_ratio_U": avg_ratio_U, "avg_ratio_T": avg_ratio_T,
        "priority": priority, "reason": reason,
    }