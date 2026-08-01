import numpy as np
from scipy.optimize import minimize


def _deutsch_chain_vec(params, Tin, Cin, Q, U1, U2, U3, U4, T1, T2, T3, T4, T_ref):
    kA = np.array(params[:4])
    alpha = np.array(params[4:8])
    U = [U1, U2, U3, U4]
    T = [T1, T2, T3, T4]
    C = np.array(Cin) * 1000.0
    for i in range(4):
        eta = (1.0 - np.exp(-kA[i] * np.array(U[i]) ** 2 / np.array(Q))) * np.exp(-alpha[i] * np.maximum(0, np.array(T[i]) - T_ref[i]))
        eta = np.clip(eta, 0.0, 0.9999)
        C = C * (1.0 - eta)
    return C


def fit_deutsch_params(df, bounds):
    Tin = df["Temp_C"].values
    Cin = df["C_in_gNm3"].values
    Q = df["Q_Nm3h"].values
    U = [df[f"U{i}_kV"].values for i in range(1, 5)]
    T = [df[f"T{i}_s"].values for i in range(1, 5)]
    y = df["C_out_mgNm3"].values

    T_ref = np.array([np.median(T[i]) for i in range(4)])
    U_mean = [np.mean(U[i]) for i in range(4)]
    Q_mean = np.mean(Q)

    kA0 = []
    for i in range(4):
        kA0.append(1.6 * Q_mean / U_mean[i] ** 2)
    p0 = np.concatenate([np.array(kA0), np.array([0.001] * 4)])

    def loss(p):
        pred = _deutsch_chain_vec(p, Tin, Cin, Q, U[0], U[1], U[2], U[3], T[0], T[1], T[2], T[3], T_ref)
        r = np.log(np.maximum(pred, 0.01)) - np.log(np.maximum(y, 0.01))
        return np.sum(r ** 2)

    lower = np.concatenate([np.array([1.0] * 4), np.array([0.0] * 4)])
    upper = np.concatenate([np.array([5000.0] * 4), np.array([0.05] * 4)])

    best_p = None
    best_loss = np.inf
    for trial in range(50):
        if trial == 0:
            p_try = p0.copy()
        elif trial < 10:
            p_try = p0 * np.random.uniform(0.5, 2.0, size=8)
        else:
            p_try = np.array([np.random.uniform(lower[i], upper[i]) for i in range(8)])
        p_try = np.clip(p_try, lower, upper)
        try:
            res = minimize(loss, p_try, method="L-BFGS-B", bounds=list(zip(lower, upper)), options={"maxiter": 10000})
            if res.fun < best_loss:
                best_loss = res.fun
                best_p = res.x
        except Exception:
            pass

    if best_p is not None:
        popt = best_p
        y_pred = _deutsch_chain_vec(popt, Tin, Cin, Q, U[0], U[1], U[2], U[3], T[0], T[1], T[2], T[3], T_ref)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / len(y))
        print(f"[INFO] Deutsch拟合: R2={r2:.4f}, RMSE={rmse:.4f}")
        print(f"  kA={popt[:4]}, alpha={popt[4:8]}, T_ref={T_ref}")
        return {
            "kA": popt[:4].tolist(), "alpha": popt[4:8].tolist(),
            "T_ref": T_ref.tolist(), "r2": float(r2), "rmse": float(rmse),
        }
    else:
        print("[WARN] Deutsch拟合失败")
        return None


def predict_cout(params, Tin, Cin, Q, U, T):
    kA = np.array(params["kA"])
    alpha = np.array(params["alpha"])
    T_ref = np.array(params["T_ref"])
    C = Cin * 1000.0
    for i in range(4):
        eta = (1.0 - np.exp(-kA[i] * U[i] ** 2 / Q)) * np.exp(-alpha[i] * max(0, T[i] - T_ref[i]))
        eta = min(max(eta, 0.0), 0.9999)
        C = C * (1.0 - eta)
    return float(C)


def predict_eta(params, U, T, Q):
    kA = np.array(params["kA"])
    alpha = np.array(params["alpha"])
    T_ref = np.array(params["T_ref"])
    etas = []
    for i in range(4):
        eta = (1.0 - np.exp(-kA[i] * U[i] ** 2 / Q)) * np.exp(-alpha[i] * max(0, T[i] - T_ref[i]))
        etas.append(float(min(max(eta, 0.0), 0.9999)))
    return tuple(etas)


def predict_peak(params, T, Cin):
    alpha = np.array(params["alpha"])
    T_ref = np.array(params["T_ref"])
    peak = 0.0
    for i in range(4):
        thickness = max(0, T[i] - T_ref[i])
        peak += alpha[i] * thickness * Cin * 1000.0 * 0.1
    return float(peak)
