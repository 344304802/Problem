import numpy as np
import pandas as pd


def fit_power_model(df):
    U = np.column_stack([df[f"U{i}_kV"].values ** 2 for i in range(1, 5)])
    P = df["P_total_kW"].values
    k, residuals, rank, sv = np.linalg.lstsq(U, P, rcond=None)
    P_pred = U @ k
    ss_res = np.sum((P - P_pred) ** 2)
    ss_tot = np.sum((P - P.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    cond = np.linalg.cond(U)
    k = np.maximum(k, 0)
    print(f"[INFO] 电耗模型 P=Σk_i*U_i^2: k={k}, R2={r2:.4f}, cond={cond:.2f}")

    if r2 < 0.9:
        print("[WARN] R2<0.9, 尝试扩展模型 P=Σ(k_i*U_i^2+b_i*U_i)+c")
        X_ext = np.column_stack(
            [df[f"U{i}_kV"].values ** 2 for i in range(1, 5)] +
            [df[f"U{i}_kV"].values for i in range(1, 5)] +
            [np.ones(len(df))]
        )
        coef, _, _, _ = np.linalg.lstsq(X_ext, P, rcond=None)
        P_pred2 = X_ext @ coef
        r2_ext = 1 - np.sum((P - P_pred2) ** 2) / ss_tot
        print(f"[INFO] 扩展模型 R2={r2_ext:.4f}")
        return {
            "k": k.tolist(), "r2": float(r2), "cond": float(cond),
            "extended": True, "coef": coef.tolist(),
            "k_ext": coef[:4].tolist(), "b_ext": coef[4:8].tolist(), "c_ext": float(coef[8]),
            "r2_ext": float(r2_ext),
        }
    return {"k": k.tolist(), "r2": float(r2), "cond": float(cond), "extended": False}


def predict_power(model, U):
    if isinstance(model, dict):
        if model.get("extended", False):
            k_ext = np.array(model["k_ext"])
            b_ext = np.array(model["b_ext"])
            c_ext = model["c_ext"]
            return float(sum(k_ext[i] * U[i] ** 2 + b_ext[i] * U[i] for i in range(4)) + c_ext)
        k = model["k"]
    else:
        k = model
    return sum(k[i] * U[i] ** 2 for i in range(4))
