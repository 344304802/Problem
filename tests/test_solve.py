import numpy as np
import pytest
from optim.solve import solve_one_regime


@pytest.fixture
def regime():
    return {"id": 0, "mean": {"Temp": 120.0, "C_in": 30.0, "Q": 50000.0}}


@pytest.fixture
def model():
    return {
        "deutsch": {
            "kA": [200.0, 200.0, 180.0, 180.0],
            "alpha": [0.01, 0.01, 0.01, 0.01],
            "T_ref": [200.0, 200.0, 400.0, 400.0],
            "r": 0.5,
        },
        "power": {"extended": False, "k": [0.1, 0.1, 0.05, 0.05]},
    }


@pytest.fixture
def bounds():
    b = {}
    for i in range(1, 5):
        b[f"U{i}"] = (40.0, 90.0)
        b[f"T{i}"] = (150.0, 300.0)
        b[f"T_crit{i}"] = 280.0
    b["U3"] = (30.0, 80.0)
    b["U4"] = (30.0, 80.0)
    b["T3"] = (350.0, 520.0)
    b["T4"] = (350.0, 520.0)
    b["T_crit3"] = 500.0
    b["T_crit4"] = 500.0
    return b


def test_solve_feasible(regime, model, bounds):
    sol = solve_one_regime(regime, model, bounds, C_limit=10.0, multi_start=5, seed=42)
    assert sol["success"], "应找到可行解"
    assert sol["P"] is not None and sol["P"] > 0
    assert sol["Cout"] <= 10.0 + 1e-3


def test_solve_respects_bounds(regime, model, bounds):
    sol = solve_one_regime(regime, model, bounds, C_limit=10.0, multi_start=5, seed=42)
    if sol["U"] is None:
        pytest.skip("无解")
    for i in range(4):
        u_lo, u_hi = bounds[f"U{i+1}"]
        assert u_lo - 1e-3 <= sol["U"][i] <= u_hi + 1e-3
    for i in range(4):
        t_lo, t_hi = bounds[f"T{i+1}"]
        t_crit = bounds[f"T_crit{i+1}"]
        assert t_lo - 1e-3 <= sol["T"][i] <= min(t_hi, t_crit) + 1e-3


def test_tighter_limit_higher_power(regime, model, bounds):
    sol10 = solve_one_regime(regime, model, bounds, C_limit=10.0, multi_start=5, seed=42)
    sol5 = solve_one_regime(regime, model, bounds, C_limit=5.0, multi_start=5, seed=42)
    if sol10["P"] and sol5["P"]:
        assert sol5["P"] >= sol10["P"] - 1.0, "更严排放约束应需更高(或相当)电耗"