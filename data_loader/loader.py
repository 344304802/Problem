import pandas as pd
import numpy as np
from datetime import datetime


EXPECTED_COLS = [
    "timestamp", "Temp_C", "C_in_gNm3", "Q_Nm3h",
    "U1_kV", "U2_kV", "U3_kV", "U4_kV",
    "T1_s", "T2_s", "T3_s", "T4_s",
    "C_out_mgNm3", "P_total_kW",
]


def load_raw(path):
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLS:
        raise ValueError(f"列名不符: 期望{EXPECTED_COLS}, 实际{list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in EXPECTED_COLS[1:]:
        df[c] = df[c].astype("float64")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("时间戳非单调递增")
    diffs = df["timestamp"].diff().dropna().unique()
    if len(diffs) > 1:
        print(f"[WARN] 时间间隔不统一: {diffs}")
    print(f"[INFO] 加载 {len(df)} 行, 缺失统计:")
    miss = df.isna().sum()
    print(miss[miss > 0])
    return df


def clean_and_impute(df, method="time"):
    df = df.copy()
    df = df.set_index("timestamp")
    n_before = df["C_out_mgNm3"].isna().sum()
    if method == "time":
        df["C_out_mgNm3"] = df["C_out_mgNm3"].interpolate(method="time")
    else:
        df["C_out_mgNm3"] = df["C_out_mgNm3"].interpolate(method="linear")
    df = df.reset_index()
    print(f"[INFO] 插补C_out缺失 {n_before} 个")

    outliers = {}
    for c in EXPECTED_COLS[1:]:
        mu, sigma = df[c].mean(), df[c].std()
        mask = (df[c] < mu - 3 * sigma) | (df[c] > mu + 3 * sigma)
        if mask.sum() > 0:
            outliers[c] = int(mask.sum())
    df.attrs["outliers"] = outliers

    bounds = {}
    for i in range(1, 5):
        ucol = f"U{i}_kV"
        tcol = f"T{i}_s"
        u_min = df[ucol].min()
        u_max = df[ucol].max() * 1.1
        t_min = df[tcol].min()
        t_max = df[tcol].max()
        t_crit = df[tcol].quantile(0.95)
        bounds[f"U{i}"] = (float(u_min), float(u_max))
        bounds[f"T{i}"] = (float(t_min), float(t_max))
        bounds[f"T_crit{i}"] = float(t_crit)
    df.attrs["bounds"] = bounds
    print(f"[INFO] 边界统计: {bounds}")
    return df