import numpy as np


def to_markdown_tables(results):
    lines = []
    lines.append("# 电除尘器协同优化控制 - 结果汇总\n")

    lines.append("## 问题1：关系分析与特征重要性\n")
    fi = results.get("feature_importance", {})
    if fi:
        lines.append("### 特征重要性 (随机森林)\n")
        lines.append("| 特征 | 重要性 |")
        lines.append("|------|--------|")
        for c, v in sorted(fi.get("rf", {}).items(), key=lambda x: -x[1]):
            lines.append(f"| {c} | {v:.4f} |")
        lines.append("")
        lines.append("### Pearson相关系数\n")
        lines.append("| 特征 | r | p值 |")
        lines.append("|------|---|-----|")
        for c, v in sorted(fi.get("pearson", {}).items(), key=lambda x: -abs(x[1]["r"])):
            lines.append(f"| {c} | {v['r']:.4f} | {v['p']:.4e} |")
        lines.append("")

    pm = results.get("power_model", {})
    if pm:
        lines.append(f"### 电耗模型\n")
        lines.append(f"$$P_{{total}} = \\sum_{{i=1}}^{{4}} k_i U_i^2$$\n")
        lines.append(f"$$k = {pm['k']}$$, $$R^2 = {pm['r2']:.4f}$$\n")

    dm = results.get("deutsch_model", {})
    if dm:
        lines.append(f"### Deutsch模型拟合\n")
        lines.append(f"$$R^2 = {dm['r2']:.4f}$$, $$RMSE = {dm['rmse']:.4f}$$\n")

    lines.append("## 问题2：工况划分与最优参数\n")
    sol2 = results.get("sol_all", [])
    if sol2:
        lines.append("| 工况 | 样本数 | C_in(g/Nm³) | Temp(℃) | U1 | U2 | U3 | U4 | T1 | T2 | T3 | T4 | P(kW) | C_out(mg/Nm³) |")
        lines.append("|------|--------|-------------|---------|----|----|----|----|----|----|----|----|-------|---------------|")
        for r in sol2:
            rg = r["regime"]; s = r["sol"]
            U = s["U"] if s["U"] else [0]*4
            T = s["T"] if s["T"] else [0]*4
            lines.append(f"| {rg['id']} | {rg['n']} | {rg['mean']['C_in']:.2f} | {rg['mean']['Temp']:.1f} | {U[0]:.1f} | {U[1]:.1f} | {U[2]:.1f} | {U[3]:.1f} | {T[0]:.0f} | {T[1]:.0f} | {T[2]:.0f} | {T[3]:.0f} | {s['P']:.1f} | {s['Cout']:.2f} |")
        lines.append("")

    lines.append("## 问题3：两工况对比与优先级\n")
    cmp = results.get("compare", {})
    if cmp:
        rA, rB = cmp["regime_A"], cmp["regime_B"]
        lines.append(f"### 高浓度工况(工况{rA['id']})\n")
        lines.append(f"- $$C_{{in}} = {rA['Cin']:.2f}$$ g/Nm³, $$T = {rA['Temp']:.1f}$$ ℃")
        lines.append(f"- $$U = {['%.1f' % u for u in rA['U']]}$$ kV, $$T = {['%.0f' % t for t in rA['T']]}$$ s")
        lines.append(f"- $$P = {rA['P']:.1f}$$ kW, $$C_{{out}} = {rA['Cout']:.2f}$$ mg/Nm³\n")
        lines.append(f"### 低浓度工况(工况{rB['id']})\n")
        lines.append(f"- $$C_{{in}} = {rB['Cin']:.2f}$$ g/Nm³, $$T = {rB['Temp']:.1f}$$ ℃")
        lines.append(f"- $$U = {['%.1f' % u for u in rB['U']]}$$ kV, $$T = {['%.0f' % t for t in rB['T']]}$$ s")
        lines.append(f"- $$P = {rB['P']:.1f}$$ kW, $$C_{{out}} = {rB['Cout']:.2f}$$ mg/Nm³\n")
        lines.append("### 差异原因\n")
        for r in cmp["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    prio = results.get("priority", {})
    if prio:
        lines.append(f"### 优先级判定\n")
        lines.append(f"$$\\text{{电压性价比}} = {prio['avg_ratio_U']:.6f}$$")
        lines.append(f"$$\\text{{振打性价比}} = {prio['avg_ratio_T']:.6f}$$")
        lines.append(f"**结论: {prio['priority']}**\n")

    lines.append("## 问题4：排放收紧影响分析\n")
    dp = results.get("delta_power", {})
    if dp:
        lines.append("### 各工况电耗增幅\n")
        lines.append("| 工况 | P*(10) (kW) | P*(5) (kW) | ΔP% |")
        lines.append("|------|-------------|------------|-----|")
        for d in dp["deltas"]:
            p10 = f"{d['P10']:.1f}" if d["P10"] else "N/A"
            p5 = f"{d['P5']:.1f}" if d["P5"] else "N/A"
            dpp = f"{d['delta_pct']:.2f}%" if d["delta_pct"] is not None else "N/A"
            lines.append(f"| {d['regime_id']} | {p10} | {p5} | {dpp} |")
        if dp.get("overall_delta_pct") is not None:
            lines.append(f"\n**整体平均电耗增幅: {dp['overall_delta_pct']:.2f}%**\n")

    feas = results.get("feasibility", [])
    if feas:
        lines.append("### 可行性校验\n")
        lines.append("| 工况 | 可行 | 说明 |")
        lines.append("|------|------|------|")
        for f in feas:
            lines.append(f"| {f['regime_id']} | {'✓' if f['feasible'] else '✗'} | {f['reason']} |")
        lines.append("")

    advice = results.get("advice", "")
    if advice:
        lines.append("### 高浓度工况应对建议\n")
        lines.append(advice)

    return "\n".join(lines)