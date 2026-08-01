import numpy as np
from scipy.optimize import minimize, differential_evolution
from modeling.deutsch import predict_cout
from modeling.power import predict_power


def solve_one_regime(regime, model, bounds, C_limit=10.0, algo="SLSQP", multi_start=10, seed=42):
    Tin = regime["mean"]["Temp"]
    Cin = regime["mean"]["C_in"]
    Q = regime["mean"]["Q"]
    params = model["deutsch"]
    power_model = model["power"]

    lb = [bounds[f"U{i}"][0] for i in range(1, 5)] + [bounds[f"T{i}"][0] for i in range(1, 5)]
    ub = [bounds[f"U{i}"][1] for i in range(1, 5)] + [min(bounds[f"T{i}"][1], bounds[f"T_crit{i}"]) for i in range(1, 5)]

    bounds_arr = list(zip(lb, ub))
    np.random.seed(seed)

    def objective(x):
        return predict_power(power_model, x[:4])

    def constraint_cout(x):
        cout = predict_cout(params, Tin, Cin, Q, x[:4], x[4:8])
        return C_limit - cout

    cons = [{"type": "ineq", "fun": constraint_cout}]

    best = None
    for start in range(multi_start):
        if start == 0:
            x0 = np.array([(lb[i] + ub[i]) / 2 for i in range(8)])
        else:
            x0 = np.array([np.random.uniform(lb[i], ub[i]) for i in range(8)])
        try:
            res = minimize(objective, x0, method="SLSQP",
                           bounds=bounds_arr, constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
            if res.success:
                cout = predict_cout(params, Tin, Cin, Q, res.x[:4], res.x[4:8])
                if cout <= C_limit + 1e-6:
                    P = predict_power(power_model, res.x[:4])
                    if best is None or P < best["P"]:
                        best = {
                            "U": res.x[:4].tolist(), "T": res.x[4:8].tolist(),
                            "P": float(P), "Cout": float(cout),
                            "success": True, "n_iter": int(res.nit),
                            "margin": float(C_limit - cout), "algo": "SLSQP",
                        }
        except Exception:
            pass

    if best is None:
        print(f"[WARN] 工况{regime['id']} SLSQP未收敛, 尝试差分进化")
        def de_obj(x):
            cout = predict_cout(params, Tin, Cin, Q, x[:4], x[4:8])
            if cout > C_limit:
                return 1e6 + (cout - C_limit) * 1e4
            return predict_power(power_model, x[:4])
        try:
            de_bounds = [(lb[i], ub[i]) for i in range(8)]
            res = differential_evolution(de_obj, bounds=de_bounds, seed=seed, maxiter=200, tol=1e-8, polish=True, workers=1)
            U, T = res.x[:4], res.x[4:8]
            cout = predict_cout(params, Tin, Cin, Q, U, T)
            best = {
                "U": U.tolist(), "T": T.tolist(), "P": float(predict_power(power_model, U)),
                "Cout": float(cout), "success": cout <= C_limit + 1e-6,
                "n_iter": int(res.nit), "margin": float(C_limit - cout), "algo": "DE",
            }
        except Exception as e:
            print(f"[ERROR] 差分进化也失败: {e}")
            best = {"U": None, "T": None, "P": None, "Cout": None, "success": False, "n_iter": 0, "margin": None, "algo": None}

    print(f"[INFO] 工况{regime['id']}: P={best.get('P')}, Cout={best.get('Cout')}, success={best['success']}")
    return best


def solve_all_regimes(regimes, model, bounds, C_limit=10.0, multi_start=10, seed=42):
    results = []
    for regime in regimes["regimes"]:
        sol = solve_one_regime(regime, model, bounds, C_limit=C_limit, multi_start=multi_start, seed=seed)
        results.append({"regime": regime, "sol": sol})
    return results
