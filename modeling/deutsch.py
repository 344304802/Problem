import numpy as np
from scipy.optimize import minimize


def _deutsch_chain_vec(params, Tin, Cin, Q, U1, U2, U3, U4, T1, T2, T3, T4, T_ref, r = 0.5) :
    # 振打双向偏离惩罚: T>T_ref 周期过长(粉尘过厚), T<T_ref 周期过短(振打扬尘), 均降低效率
    # r∈[0.1,1] 为过频副作用占比, r=1 对称, r=0 退化为单向(旧模型)
    kA = np.array(params[:4])
    alpha = np.array(params[4:8])
    U = [U1, U2, U3, U4]
    T = [T1, T2, T3, T4]
    C = np.array(Cin) * 1000.0
    for i in range(4) :
        delta = np.array(T[i]) - T_ref[i]
        penalty = alpha[i] * np.where(delta >= 0, delta, r * (-delta))
        eta = (1.0 - np.exp(-kA[i] * np.array(U[i]) ** 2 / np.array(Q))) * np.exp(-penalty)
        eta = np.clip(eta, 0.0, 0.9999)
        C = C * (1.0 - eta)
    return C


def fit_deutsch_params(df, bounds) :
    Tin = df["Temp_C"].values
    Cin = df["C_in_gNm3"].values
    Q = df["Q_Nm3h"].values
    U = [df[f"U{i}_kV"].values for i in range(1, 5)]
    T = [df[f"T{i}_s"].values for i in range(1, 5)]
    y = df["C_out_mgNm3"].values

    T_ref = np.array([np.median(T[i]) for i in range(4)])
    U_mean = [np.mean(U[i]) for i in range(4)]
    Q_mean = np.mean(Q)

    # 物理先验: 同型号 ESP 四电场几何相似, kA 应相近。共享 kA_0, 参数从 8 降到 5, 缓解欠定。
    # 后电场粉尘更细/比电阻略高, 用 g_i 微调: g=[1.0,1.0,0.9,0.9]
    g = np.array([1.0, 1.0, 0.9, 0.9])
    U_mean = [np.mean(U[i]) for i in range(4)]
    Q_mean = np.mean(Q)
    # kA_0 初值: 令单级效率 η≈0.9, kA_0 = -Q/U²·ln(0.1) ≈ 2.3·Q/U² (用前电场电压)
    kA0_init = 2.3 * Q_mean / ((U_mean[0] + U_mean[1]) / 2) ** 2
    # 振打双向偏离比例 r 固定 0.5 (物理先验: 过频副作用约为过长的 50%)
    # 不拟合 r 因 C_out 限幅导致不可辨识(拟合会贴下界 0.1 退化为单向)
    r_fixed = 0.5
    p0 = np.array([kA0_init, 0.005, 0.005, 0.005, 0.005])

    def expand(p) :
        # p=[kA_0, α1..α4] -> full=[kA_0*g1..kA_0*g4, α1..α4]
        return np.concatenate([p[0] * g, p[1:5]])

    def loss(p) :
        full = expand(p)
        pred = _deutsch_chain_vec(full, Tin, Cin, Q, U[0], U[1], U[2], U[3], T[0], T[1], T[2], T[3], T_ref, r = r_fixed)
        r_res = np.log(np.maximum(pred, 0.01)) - np.log(np.maximum(y, 0.01))
        return np.sum(r_res ** 2)

    # kA_0 边界 [50, 2000], alpha 下界 0.001 防止拟合成 0
    lower = np.array([50.0, 0.001, 0.001, 0.001, 0.001])
    upper = np.array([2000.0, 0.05, 0.05, 0.05, 0.05])

    best_p = None
    best_loss = np.inf
    for trial in range(50) :
        if trial == 0 :
            p_try = p0.copy()
        elif trial < 10 :
            p_try = p0 * np.random.uniform(0.5, 2.0, size = 5)
        else :
            p_try = np.array([np.random.uniform(lower[i], upper[i]) for i in range(5)])
        p_try = np.clip(p_try, lower, upper)
        try :
            res = minimize(loss, p_try, method="L-BFGS-B", bounds=list(zip(lower, upper)), options={"maxiter": 10000})
            if res.fun < best_loss :
                best_loss = res.fun
                best_p = res.x
        except Exception :
            pass

    if best_p is not None :
        popt = best_p
        full = expand(popt)
        y_pred = _deutsch_chain_vec(full, Tin, Cin, Q, U[0], U[1], U[2], U[3], T[0], T[1], T[2], T[3], T_ref, r = r_fixed)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / len(y))
        print(f"[INFO] Deutsch拟合(共享kA_0,双向振打r = 0.5) : R2 = {r2:.4f}, RMSE = {rmse:.4f}")
        print(f"  kA_0 = {popt[0]:.2f}, kA = {full[:4].tolist()}, alpha = {full[4:8].tolist()}, r = {r_fixed}, T_ref = {T_ref.tolist()}")
        return {
            "kA": full[:4].tolist(), "alpha": full[4:8].tolist(),
            "T_ref": T_ref.tolist(), "r2": float(r2), "rmse": float(rmse),
            "kA_0": float(popt[0]), "g": g.tolist(), "r": float(r_fixed),
        }
    else :
        print("[WARN] Deutsch拟合失败")
        return None


def predict_cout(params, Tin, Cin, Q, U, T) :
    kA = np.array(params["kA"])
    alpha = np.array(params["alpha"])
    T_ref = np.array(params["T_ref"])
    r = params.get("r", 0.5)
    C = Cin * 1000.0
    for i in range(4) :
        delta = T[i] - T_ref[i]
        penalty = alpha[i] * (delta if delta >= 0 else r * (-delta))
        eta = (1.0 - np.exp(-kA[i] * U[i] ** 2 / Q)) * np.exp(-penalty)
        eta = min(max(eta, 0.0), 0.9999)
        C = C * (1.0 - eta)
    return float(C)


def predict_eta(params, U, T, Q) :
    kA = np.array(params["kA"])
    alpha = np.array(params["alpha"])
    T_ref = np.array(params["T_ref"])
    r = params.get("r", 0.5)
    etas = []
    for i in range(4) :
        delta = T[i] - T_ref[i]
        penalty = alpha[i] * (delta if delta >= 0 else r * (-delta))
        eta = (1.0 - np.exp(-kA[i] * U[i] ** 2 / Q)) * np.exp(-penalty)
        etas.append(float(min(max(eta, 0.0), 0.9999)))
    return tuple(etas)


def predict_peak(params, T, Cin) :
    alpha = np.array(params["alpha"])
    T_ref = np.array(params["T_ref"])
    r = params.get("r", 0.5)
    peak = 0.0
    for i in range(4) :
        delta = T[i] - T_ref[i]
        thickness = delta if delta >= 0 else r * (-delta)
        peak += alpha[i] * thickness * Cin * 1000.0 * 0.1
    return float(peak)
