import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import pearsonr


FEATURE_COLS = [
    "Temp_C", "C_in_gNm3", "Q_Nm3h",
    "U1_kV", "U2_kV", "U3_kV", "U4_kV",
    "T1_s", "T2_s", "T3_s", "T4_s",
]


def feature_importance(df):
    X = df[FEATURE_COLS].values
    y = df["C_out_mgNm3"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X, y)

    rf_imp = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    gb_imp = dict(zip(FEATURE_COLS, gb.feature_importances_.tolist()))

    pearson = {}
    for c in FEATURE_COLS:
        r, p = pearsonr(df[c].values, y)
        pearson[c] = {"r": float(r), "p": float(p)}

    print("[INFO] 特征重要性(RF):")
    for c, v in sorted(rf_imp.items(), key=lambda x: -x[1]):
        print(f"  {c}: {v:.4f}")
    print("[INFO] Pearson相关:")
    for c, v in sorted(pearson.items(), key=lambda x: -abs(x[1]["r"])):
        print(f"  {c}: r={v['r']:.4f}, p={v['p']:.4e}")

    return {"rf": rf_imp, "gb": gb_imp, "pearson": pearson}