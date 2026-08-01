import numpy as np


def high_conc_advice(regime, sol10, sol5):
    advice_lines = []
    rid = regime["id"]
    Cin = regime["mean"]["C_in"]
    advice_lines.append(f"## 高浓度工况(工况{rid}, C_in={Cin:.2f} g/Nm³)应对建议")
    advice_lines.append("")

    if sol10["U"] is not None and sol5["U"] is not None:
        dU = [sol5["U"][i] - sol10["U"][i] for i in range(4)]
        advice_lines.append("### 电压调整方向")
        for i in range(4):
            if dU[i] > 0.5:
                advice_lines.append(f"- 第{i+1}电场电压需提升 {dU[i]:.2f} kV (从 {sol10['U'][i]:.2f} → {sol5['U'][i]:.2f})")
            elif dU[i] < -0.5:
                advice_lines.append(f"- 第{i+1}电场电压可降低 {abs(dU[i]):.2f} kV")
            else:
                advice_lines.append(f"- 第{i+1}电场电压基本不变")

    if sol10["T"] is not None and sol5["T"] is not None:
        dT = [sol5["T"][i] - sol10["T"][i] for i in range(4)]
        advice_lines.append("")
        advice_lines.append("### 振打周期调整方向")
        for i in range(4):
            if dT[i] < -10:
                advice_lines.append(f"- 第{i+1}电场振打周期需缩短 {abs(dT[i]):.0f} s (更频繁清灰)")
            elif dT[i] > 10:
                advice_lines.append(f"- 第{i+1}电场振打周期可延长 {dT[i]:.0f} s")
            else:
                advice_lines.append(f"- 第{i+1}电场振打周期基本不变")

    advice_lines.append("")
    advice_lines.append("### 多电场协同策略")
    advice_lines.append("- 前电场(1-2)承担主要除尘负荷, 优先提升电压")
    advice_lines.append("- 后电场(3-4)精细控制, 维持低排放的同时尽量降低电压省电")
    advice_lines.append("- 高浓度工况下振打周期应整体缩短, 防止极板积灰导致效率骤降")
    advice_lines.append("- 若电压已达变压器上限仍不满足5 mg/Nm³, 需考虑硬件改造(增加电场数或提升变压器容量)")

    return "\n".join(advice_lines)