import numpy as np
from modeling.deutsch import predict_cout
from modeling.power import predict_power


def compare_two_regimes(regimes_results, model, bounds):
    sorted_by_conc = sorted(regimes_results, key=lambda r: r["regime"]["mean"]["C_in"])
    rA = sorted_by_conc[-1]
    rB = sorted_by_conc[0]

    params = model["deutsch"]
    power_model = model["power"]

    def detail(r):
        regime = r["regime"]
        sol = r["sol"]
        Tin, Cin, Q = regime["mean"]["Temp"], regime["mean"]["C_in"], regime["mean"]["Q"]
        U, T = sol["U"], sol["T"]
        cout = predict_cout(params, Tin, Cin, Q, U, T)
        P = predict_power(power_model, U, T=T)
        return {
            "id": regime["id"], "n": regime["n"],
            "Cin": Cin, "Temp": Tin, "Q": Q,
            "U": U, "T": T, "P": P, "Cout": cout,
        }

    dA = detail(rA)
    dB = detail(rB)

    reasons = []
    if dA["Cin"] > dB["Cin"]:
        reasons.append(f"高浓度工况(工况{dA['id']}, C_in={dA['Cin']:.2f})需更高电压承担除尘负荷")
    if sum(dA["U"]) > sum(dB["U"]):
        reasons.append(f"高浓度工况总电压 {sum(dA['U']):.1f}kV > 低浓度工况 {sum(dB['U']):.1f}kV")
    if sum(dA["T"]) < sum(dB["T"]):
        reasons.append(f"高浓度工况总振打周期 {sum(dA['T']):.0f}s < 低浓度工况 {sum(dB['T']):.0f}s, 需更频繁清灰")
    reasons.append("高浓度工况下粉尘负荷大, 极板积灰快, 需缩短振打周期并提高电压维持效率")
    reasons.append("低浓度工况下粉尘负荷小, 可适当降低电压省电, 延长振打周期减少机械磨损")

    print(f"[INFO] 对比工况{dA['id']}(高浓度) vs 工况{dB['id']}(低浓度)")
    return {"regime_A": dA, "regime_B": dB, "reasons": reasons}