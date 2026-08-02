import numpy as np
from modeling.deutsch import predict_cout
from modeling.power import predict_power


def numeric_jacobian(model, x0, regime, bounds, step_ratio=0.01):
    params = model["deutsch"]
    power_model = model["power"]
    Tin = regime["mean"]["Temp"]
    Cin = regime["mean"]["C_in"]
    Q = regime["mean"]["Q"]
    U = np.array(x0[:4])
    T = np.array(x0[4:8])

    ranges = []
    for i in range(1, 5):
        lo, hi = bounds[f"U{i}"]
        ranges.append(hi - lo)
    for i in range(1, 5):
        lo, hi = bounds[f"T{i}"]
        ranges.append(hi - lo)

    def cout_full(x):
        return predict_cout(params, Tin, Cin, Q, x[:4], x[4:8])

    def power_full(x):
        return predict_power(power_model, x[:4])

    x = np.concatenate([U, T])
    SC = np.zeros(8)
    SP = np.zeros(8)
    for i in range(8):
        h = ranges[i] * step_ratio
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        SC[i] = (cout_full(xp) - cout_full(xm)) / (2 * h)
        SP[i] = (power_full(xp) - power_full(xm)) / (2 * h)

    h_half = ranges[0] * step_ratio / 2
    xp = x.copy(); xp[0] += h_half
    xm = x.copy(); xm[0] -= h_half
    sc_half = (cout_full(xp) - cout_full(xm)) / (2 * h_half)
    consistency = abs(sc_half - SC[0]) / (abs(SC[0]) + 1e-12)
    if consistency > 0.1:
        print(f"[WARN] 步长减半不一致: {consistency:.4f}")

    # 无量纲弹性系数 E = ∂ln y / ∂ln x = (x/y)·∂y/∂x, 消除量纲差异 (电压kV vs 振打s)
    C0 = cout_full(x)
    P0 = power_full(x)
    EC = np.array([x[i] / (C0 + 1e-12) * SC[i] for i in range(8)])
    EP = np.array([x[i] / (P0 + 1e-12) * SP[i] for i in range(8)])

    return {
        "SC_U": SC[:4].tolist(), "SC_T": SC[4:].tolist(),
        "SP_U": SP[:4].tolist(), "SP_T": SP[4:].tolist(),
        "EC_U": EC[:4].tolist(), "EC_T": EC[4:].tolist(),
        "EP_U": EP[:4].tolist(), "EP_T": EP[4:].tolist(),
        "C0": float(C0), "P0": float(P0),
        "consistency": float(consistency),
    }