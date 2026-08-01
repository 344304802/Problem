import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
COLOR = "#333333"


def _style(ax):
    ax.tick_params(colors=COLOR)
    for spine in ax.spines.values():
        spine.set_color(COLOR)
    ax.title.set_color(COLOR)
    ax.xaxis.label.set_color(COLOR)
    ax.yaxis.label.set_color(COLOR)


def plot_relation_curves(model, df, out_dir):
    params = model["deutsch"]
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    Tin_med = df["Temp_C"].median()
    Cin_med = df["C_in_gNm3"].median()
    Q_med = df["Q_Nm3h"].median()
    U_med = [df[f"U{i}_kV"].median() for i in range(1, 5)]
    T_med = [df[f"T{i}_s"].median() for i in range(1, 5)]

    from modeling.deutsch import predict_cout

    Ts = np.linspace(df["Temp_C"].min(), df["Temp_C"].max(), 50)
    axes[0, 0].plot(Ts, [predict_cout(params, t, Cin_med, Q_med, U_med, T_med) for t in Ts], color=COLOR)
    axes[0, 0].set_xlabel("入口温度 (℃)")
    axes[0, 0].set_ylabel("出口浓度 (mg/Nm³)")
    axes[0, 0].set_title("温度-出口浓度关系")
    _style(axes[0, 0])

    Cs = np.linspace(df["C_in_gNm3"].min(), df["C_in_gNm3"].max(), 50)
    axes[0, 1].plot(Cs, [predict_cout(params, Tin_med, c, Q_med, U_med, T_med) for c in Cs], color=COLOR)
    axes[0, 1].set_xlabel("入口浓度 (g/Nm³)")
    axes[0, 1].set_ylabel("出口浓度 (mg/Nm³)")
    axes[0, 1].set_title("入口浓度-出口浓度关系")
    _style(axes[0, 1])

    Us = np.linspace(df["U1_kV"].min(), df["U1_kV"].max(), 50)
    axes[1, 0].plot(Us, [predict_cout(params, Tin_med, Cin_med, Q_med, [u] + U_med[1:], T_med) for u in Us], color=COLOR)
    axes[1, 0].set_xlabel("U1 电压 (kV)")
    axes[1, 0].set_ylabel("出口浓度 (mg/Nm³)")
    axes[1, 0].set_title("电压-出口浓度关系")
    _style(axes[1, 0])

    from modeling.deutsch import predict_peak
    T1s = np.linspace(df["T1_s"].min(), df["T1_s"].max(), 50)
    axes[1, 1].plot(T1s, [predict_peak(params, [t] + T_med[1:], Cin_med) for t in T1s], color=COLOR)
    axes[1, 1].set_xlabel("T1 振打周期 (s)")
    axes[1, 1].set_ylabel("瞬时峰值 (mg/Nm³)")
    axes[1, 1].set_title("振打周期-瞬时峰值关系")
    _style(axes[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "relation_curves.png"), dpi=150)
    plt.close()


def plot_regime_scatter(regimes, df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = np.array(regimes["labels"])
    scatter = ax.scatter(df["C_in_gNm3"], df["Temp_C"], c=labels, cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax, label="工况编号")
    ax.set_xlabel("入口浓度 (g/Nm³)")
    ax.set_ylabel("入口温度 (℃)")
    ax.set_title("K-Means工况划分散点图")
    _style(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regime_scatter.png"), dpi=150)
    plt.close()


def plot_param_compare(cmp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rA, rB = cmp["regime_A"], cmp["regime_B"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(4)
    w = 0.35
    axes[0].bar(x - w/2, rA["U"], w, label=f"工况{rA['id']}(高浓度)", color="#4C72B0")
    axes[0].bar(x + w/2, rB["U"], w, label=f"工况{rB['id']}(低浓度)", color="#DD8452")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["U1", "U2", "U3", "U4"])
    axes[0].set_ylabel("电压 (kV)"); axes[0].set_title("电压对比"); axes[0].legend()
    _style(axes[0])

    axes[1].bar(x - w/2, rA["T"], w, label=f"工况{rA['id']}", color="#4C72B0")
    axes[1].bar(x + w/2, rB["T"], w, label=f"工况{rB['id']}", color="#DD8452")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["T1", "T2", "T3", "T4"])
    axes[1].set_ylabel("振打周期 (s)"); axes[1].set_title("振打周期对比"); axes[1].legend()
    _style(axes[1])

    axes[2].bar(["P_高浓度", "P_低浓度"], [rA["P"], rB["P"]], color=["#4C72B0", "#DD8452"])
    axes[2].set_ylabel("总电耗 (kW)"); axes[2].set_title("电耗对比")
    _style(axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "param_compare.png"), dpi=150)
    plt.close()


def plot_sensitivity_heatmap(sens, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    SC = np.array([sens["SC_U"], sens["SC_T"]])
    im0 = axes[0].imshow(SC, cmap="coolwarm", aspect="auto")
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["U", "T"])
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(["1", "2", "3", "4"])
    axes[0].set_title("浓度灵敏度 $S^C$"); plt.colorbar(im0, ax=axes[0])

    SP = np.array([sens["SP_U"], sens["SP_T"]])
    im1 = axes[1].imshow(SP, cmap="coolwarm", aspect="auto")
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["U", "T"])
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(["1", "2", "3", "4"])
    axes[1].set_title("电耗灵敏度 $S^P$"); plt.colorbar(im1, ax=axes[1])

    ratio = np.abs(SC) / (np.abs(SP) + 1e-12)
    im2 = axes[2].imshow(ratio, cmap="YlOrRd", aspect="auto")
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["U", "T"])
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels(["1", "2", "3", "4"])
    axes[2].set_title("性价比 $|S^C/S^P|$"); plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_heatmap.png"), dpi=150)
    plt.close()


def plot_delta_power(dp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    deltas = dp["deltas"]
    valid = [d for d in deltas if d["delta_pct"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ids = [d["regime_id"] for d in valid]
    p10 = [d["P10"] for d in valid]
    p5 = [d["P5"] for d in valid]
    x = np.arange(len(ids))
    w = 0.35
    axes[0].bar(x - w/2, p10, w, label="$P^*(10)$", color="#4C72B0")
    axes[0].bar(x + w/2, p5, w, label="$P^*(5)$", color="#C44E52")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"工况{i}" for i in ids])
    axes[0].set_ylabel("电耗 (kW)"); axes[0].set_title("收紧前后电耗对比"); axes[0].legend()
    _style(axes[0])

    dpct = [d["delta_pct"] for d in valid]
    axes[1].bar(x, dpct, color="#55A868")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"工况{i}" for i in ids])
    axes[1].set_ylabel("电耗增幅 (%)"); axes[1].set_title("$\\Delta P\\%$ 增幅")
    _style(axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delta_power.png"), dpi=150)
    plt.close()