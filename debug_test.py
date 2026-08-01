import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model
from modeling.deutsch import fit_deutsch_params
from optim.regime import cluster_regimes
from optim.solve import solve_all_regimes
from sensitivity.jacobian import numeric_jacobian
from sensitivity.priority import priority_rule
from sensitivity.compare import compare_two_regimes
from tighten.resolve import resolve_under_limit, delta_power, feasibility_check

df = load_raw("../question/2026年XJTU校赛题目/2026年校赛题目/A/Cement_ESP_Data.csv")
df = clean_and_impute(df)
bounds = df.attrs["bounds"]
pm = fit_power_model(df)
dm = fit_deutsch_params(df, bounds)
model = {"power": pm, "deutsch": dm}
regimes = cluster_regimes(df, k=5, seed=42)
sol_all = solve_all_regimes(regimes, model, bounds, C_limit=10.0, multi_start=10, seed=42)
cmp = compare_two_regimes(sol_all, model, bounds)
print("cmp done")
high_regime = cmp["regime_A"]
print(f"high_regime id={high_regime['id']}")
high_sol = None
for r in sol_all:
    if r["regime"]["id"] == high_regime["id"]:
        high_sol = r["sol"]
        break
print(f"high_sol U={high_sol['U']}")
x0 = high_sol["U"] + high_sol["T"]
regime_obj = next(r["regime"] for r in sol_all if r["regime"]["id"] == high_regime["id"])
print("calling numeric_jacobian...")
try:
    sens = numeric_jacobian(model, x0, regime_obj, bounds, step_ratio=0.01)
    print(f"sens done: {sens}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")
prio = priority_rule(sens)
print(f"prio done: {prio}")
print("step5: tighten...")
sol5_all = resolve_under_limit(regimes, model, bounds, C_limit_new=5.0, multi_start=10, seed=42)
print("step5 done")
dp = delta_power(sol_all, sol5_all)
print(f"delta_power done: {dp}")
feas = feasibility_check(sol5_all, bounds)
print(f"feasibility done: {feas}")
print("ALL DONE")