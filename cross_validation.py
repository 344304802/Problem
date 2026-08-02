import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import yaml
from data_loader.loader import load_raw, clean_and_impute
from modeling.power import fit_power_model, predict_power
from modeling.deutsch import fit_deutsch_params, predict_cout


def metrics(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae = np.mean(np.abs(y - yhat))
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def run():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config/config.yaml"), encoding="utf-8"))
    csv_path = os.path.join(os.path.dirname(__file__), cfg["csv_path"])
    df = load_raw(csv_path)
    df = clean_and_impute(df)
    bounds = df.attrs["bounds"]

    n = len(df)
    n_train = 5 * 1440  # 前5天训练
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()
    print(f"\n=== 时序交叉验证: 训练 {n_train} 行(前5天), 测试 {n-n_train} 行(后2天) ===")

    # 在训练集拟合
    print("\n--- 训练集拟合 ---")
    power_m = fit_power_model(df_train)
    deutsch_m = fit_deutsch_params(df_train, bounds)

    # 在测试集预测
    print("\n--- 测试集评估 ---")
    y_c_test = df_test["C_out_mgNm3"].values
    y_p_test = df_test["P_total_kW"].values
    yhat_c = np.array([
        predict_cout(deutsch_m, r["Temp_C"], r["C_in_gNm3"], r["Q_Nm3h"],
                     [r["U1_kV"], r["U2_kV"], r["U3_kV"], r["U4_kV"]],
                     [r["T1_s"], r["T2_s"], r["T3_s"], r["T4_s"]])
        for _, r in df_test.iterrows()
    ])
    yhat_p = np.array([
        predict_power(power_m, [r["U1_kV"], r["U2_kV"], r["U3_kV"], r["U4_kV"]])
        for _, r in df_test.iterrows()
    ])

    rmse_c, mae_c, r2_c = metrics(y_c_test, yhat_c)
    rmse_p, mae_p, r2_p = metrics(y_p_test, yhat_p)
    print(f"\nC_out 预测: RMSE={rmse_c:.4f}, MAE={mae_c:.4f}, R²={r2_c:.4f}")
    print(f"P_total 预测: RMSE={rmse_p:.4f}, MAE={mae_p:.4f}, R²={r2_p:.4f}")

    # 训练集上的表现(对比过拟合)
    y_c_train = df_train["C_out_mgNm3"].values
    yhat_c_tr = np.array([
        predict_cout(deutsch_m, r["Temp_C"], r["C_in_gNm3"], r["Q_Nm3h"],
                     [r["U1_kV"], r["U2_kV"], r["U3_kV"], r["U4_kV"]],
                     [r["T1_s"], r["T2_s"], r["T3_s"], r["T4_s"]])
        for _, r in df_train.iterrows()
    ])
    rmse_c_tr, _, r2_c_tr = metrics(y_c_train, yhat_c_tr)
    print(f"\n(对比) 训练集 C_out: RMSE={rmse_c_tr:.4f}, R²={r2_c_tr:.4f}")
    print(f"泛化差距 ΔRMSE = {rmse_c - rmse_c_tr:.4f} (测试-训练)")

    return {"rmse_c_test": rmse_c, "r2_c_test": r2_c, "rmse_p_test": rmse_p, "r2_p_test": r2_p,
            "rmse_c_train": rmse_c_tr, "r2_c_train": r2_c_tr}


if __name__ == "__main__":
    run()