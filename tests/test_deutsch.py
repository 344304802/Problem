import numpy as np
import pytest
from modeling.deutsch import _deutsch_chain_vec, predict_cout, predict_eta, predict_peak


@pytest.fixture
def params():
    return {
        "kA": [200.0, 200.0, 180.0, 180.0],
        "alpha": [0.01, 0.01, 0.01, 0.01],
        "T_ref": [200.0, 200.0, 400.0, 400.0],
        "r": 0.5,
    }


@pytest.fixture
def base_inputs():
    Tin = 120.0
    Cin = 30.0
    Q = 50000.0
    U = [60.0, 60.0, 50.0, 50.0]
    T = [200.0, 200.0, 400.0, 400.0]
    return Tin, Cin, Q, U, T


def test_eta_in_range(params, base_inputs):
    Tin, Cin, Q, U, T = base_inputs
    etas = predict_eta(params, U, T, Q)
    for e in etas:
        assert 0.0 <= e <= 0.9999


def test_higher_voltage_lower_cout(params, base_inputs):
    Tin, Cin, Q, U, T = base_inputs
    base = predict_cout(params, Tin, Cin, Q, U, T)
    U_high = [u + 10 for u in U]
    high = predict_cout(params, Tin, Cin, Q, U_high, T)
    assert high < base


def test_t_ref_is_optimum(params, base_inputs):
    Tin, Cin, Q, U, T = base_inputs
    base = predict_cout(params, Tin, Cin, Q, U, T)
    T_over = [t + 50 for t in T]
    over = predict_cout(params, Tin, Cin, Q, U, T_over)
    T_under = [t - 50 for t in T]
    under = predict_cout(params, Tin, Cin, Q, U, T_under)
    assert over > base, "T>T_ref 应使效率下降 C_out 上升"
    assert under > base, "T<T_ref 双向偏离应使效率下降 C_out 上升"


def test_bidirectional_r(params, base_inputs):
    Tin, Cin, Q, U, T = base_inputs
    T_over = [t + 50 for t in T]
    T_under = [t - 50 for t in T]
    over = predict_cout(params, Tin, Cin, Q, U, T_over)
    under = predict_cout(params, Tin, Cin, Q, U, T_under)
    assert under < over, "r=0.5 时过频惩罚应弱于过长惩罚"


def test_r_zero_degrades_to_unilateral(params, base_inputs):
    Tin, Cin, Q, U, T = base_inputs
    T_under = [t - 50 for t in T]
    params_uni = {**params, "r": 0.0}
    under_uni = predict_cout(params_uni, Tin, Cin, Q, U, T_under)
    base = predict_cout(params_uni, Tin, Cin, Q, U, T)
    assert abs(under_uni - base) < 1e-6, "r=0 时 T<T_ref 无惩罚(退化为单向)"


def test_vec_matches_scalar(params, base_inputs):
    Tin, Cin, Q, U, T = base_inputs
    scalar = predict_cout(params, Tin, Cin, Q, U, T)
    full = params["kA"] + params["alpha"]
    T_ref = np.array(params["T_ref"])
    vec = _deutsch_chain_vec(full, np.array([Tin]), np.array([Cin]), np.array([Q]),
                             np.array([U[0]]), np.array([U[1]]), np.array([U[2]]), np.array([U[3]]),
                             np.array([T[0]]), np.array([T[1]]), np.array([T[2]]), np.array([T[3]]),
                             T_ref, r=params["r"])
    assert abs(scalar - float(vec[0])) < 1e-4


def test_predict_peak_nonneg(params, base_inputs):
    _, Cin, _, _, T = base_inputs
    peak = predict_peak(params, T, Cin)
    assert peak >= 0.0