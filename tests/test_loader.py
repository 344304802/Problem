import numpy as np
import pandas as pd
import pytest
from data_loader.loader import load_raw, clean_and_impute, EXPECTED_COLS


@pytest.fixture
def csv_path(tmp_path) :
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    rng = np.random.RandomState(0)
    data = {
        "timestamp": ts,
        "Temp_C": rng.uniform(100, 140, n),
        "C_in_gNm3": rng.uniform(20, 50, n),
        "Q_Nm3h": rng.uniform(40000, 60000, n),
        "U1_kV": rng.uniform(45, 85, n),
        "U2_kV": rng.uniform(45, 85, n),
        "U3_kV": rng.uniform(38, 70, n),
        "U4_kV": rng.uniform(37, 69, n),
        "T1_s": rng.uniform(157, 290, n),
        "T2_s": rng.uniform(154, 289, n),
        "T3_s": rng.uniform(359, 506, n),
        "T4_s": rng.uniform(358, 510, n),
        "C_out_mgNm3": rng.uniform(40, 50, n),
        "P_total_kW": rng.uniform(1500, 1900, n),
    }
    df = pd.DataFrame(data)
    df.loc[10, "C_out_mgNm3"] = np.nan
    df.loc[20, "C_out_mgNm3"] = np.nan
    p = tmp_path / "sample.csv"
    df.to_csv(p, index = False)
    return p


def test_load_raw_columns(csv_path) :
    df = load_raw(csv_path)
    assert list(df.columns) == EXPECTED_COLS


def test_load_raw_timestamp_monotonic(csv_path) :
    df = load_raw(csv_path)
    assert df["timestamp"].is_monotonic_increasing


def test_clean_impute_fills_nan(csv_path) :
    df = load_raw(csv_path)
    assert df["C_out_mgNm3"].isna().sum() == 2
    df2 = clean_and_impute(df)
    assert df2["C_out_mgNm3"].isna().sum() == 0


def test_clean_impute_bounds(csv_path) :
    df = load_raw(csv_path)
    df2 = clean_and_impute(df)
    bounds = df2.attrs["bounds"]
    for i in range(1, 5) :
        assert f"U{i}" in bounds and f"T{i}" in bounds and f"T_crit{i}" in bounds
        u_lo, u_hi = bounds[f"U{i}"]
        assert u_lo < u_hi
        t_lo, t_hi = bounds[f"T{i}"]
        assert t_lo < t_hi


def test_load_raw_bad_columns(tmp_path) :
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError) :
        load_raw(p)