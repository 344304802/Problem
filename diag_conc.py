import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, predict_cout, predict_eta
from optim.regime import cluster_regimes
from optim.solve import solve_all_regimes

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config/config.yaml"), encoding="utf-8"))
csv_path = os.path.join(os.path.dirname(__file__), cfg["csv_path"])
df = load_raw(csv_path)
df = clean_and_impute(df)
bounds = df.attrs["bounds"]

power_model = fit_power_model(df)
deutsch = fit_deutsch_params(df, bounds)
model = {"power": power_model, "deutsch": deutsch}

print("\n=== 拟合参数 ===")
print("kA =", deutsch["kA"])
print("alpha =", deutsch["alpha"])
print("T_ref =", deutsch["T_ref"])
print("power extended =", power_model.get("extended"))
if power_model.get("extended"):
    print("k_ext =", power_model["k_ext"])
    print("b_ext =", power_model["b_ext"])
    print("c_ext =", power_model["c_ext"])

regimes = cluster_regimes(df, k=cfg["regime_k"], seed=cfg["seed"])
sols = solve_all_regimes(regimes, model, bounds, C_limit=10.0, multi_start=cfg["multi_start"], seed=cfg["seed"])

print("\n=== 各工况求解结果 ===")
print(f"{'id':>2} {'C_in':>7} {'Temp':>7} {'Q':>9}  {'U1':>5}{'U2':>5}{'U3':>5}{'U4':>5}  {'T1':>5}{'T2':>5}{'T3':>5}{'T4':>5}  {'P':>8} {'Cout':>6}")
for r in sorted(sols, key=lambda x: x["regime"]["mean"]["C_in"]):
    rg, s = r["regime"], r["sol"]
    U, T = s["U"], s["T"]
    print(f"{rg['id']:>2} {rg['mean']['C_in']:>7.2f} {rg['mean']['Temp']:>7.1f} {rg['mean']['Q']:>9.0f}  "
          f"{U[0]:>5.1f}{U[1]:>5.1f}{U[2]:>5.1f}{U[3]:>5.1f}  {T[0]:>5.0f}{T[1]:>5.0f}{T[2]:>5.0f}{T[3]:>5.0f}  {s['P']:>8.1f} {s['Cout']:>6.2f}")

print("\n=== 浓度敏感性诊断 ===")
lo = sorted(sols, key=lambda x: x["regime"]["mean"]["C_in"])[0]
hi = sorted(sols, key=lambda x: x["regime"]["mean"]["C_in"])[-1]
print(f"低浓度工况 C_in={lo['regime']['mean']['C_in']:.2f}, P={lo['sol']['P']:.1f}, U={lo['sol']['U']}")
print(f"高浓度工况 C_in={hi['regime']['mean']['C_in']:.2f}, P={hi['sol']['P']:.1f}, U={hi['sol']['U']}")
print(f"浓度比 {hi['regime']['mean']['C_in']/lo['regime']['mean']['C_in']:.2f}, 电耗比 {hi['sol']['P']/lo['sol']['P']:.3f}")

print("\n=== 理论分析: 固定U=历史均值时 C_out 对 C_in 的依赖 ===")
U_hist = [58.4, 58.4, 48.2, 48.2]
T_hist = [230, 230, 441, 441]
Q_mean = df["Q_Nm3h"].mean()
for Cin in [20, 30, 40, 50, 60, 70]:
    cout = predict_cout(deutsch, 126, Cin, Q_mean, U_hist, T_hist)
    etas = predict_eta(deutsch, U_hist, T_hist, Q_mean)
    prod = np.prod([1-e for e in etas])
    print(f"C_in={Cin:>3}: C_out={cout:>7.2f}, Π(1-η)={prod:.6e}")

print("\n=== 要达标需要的 Π(1-η) ===")
for Cin in [20, 30, 40, 50, 60, 70]:
    needed = 10.0 / (Cin * 1000)
    print(f"C_in={Cin:>3}: 需 Π(1-η)={needed:.6e} (ln={np.log(needed):.3f})")

print("\n=== 各电场效率对U的敏感度 (在历史点) ===")
for i in range(4):
    U = list(U_hist)
    for dU in [-2, 0, 2]:
        Uu = U.copy(); Uu[i] += dU
        etas = predict_eta(deutsch, Uu, T_hist, Q_mean)
        print(f"  U{i+1}={Uu[i]:.1f}: η={etas}")