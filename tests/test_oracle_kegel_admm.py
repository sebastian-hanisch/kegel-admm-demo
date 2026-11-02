"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): ADMM gegen einen anderen Rechenweg.

- cvxpy mit Clarabel (Innere Punkte) löst dieselben vier Kugel-Programme auf Pfaden, die nichts mit unserem Kern zu tun haben (`cp.norm`, `cp.log_sum_exp`): der Optimalwert R* muss auf 10⁻⁶ stimmen
- cvxpy mit SCS (ADMM mit homogener Einbettung) löst die Lorentz-Kugel: gleicher Wert, damit ein Fehler in beiden ADMM-Fassungen gegen einen dritten Weg geprüft ist
- die Dualvariable y = ρu des Laufs ist zulässig für das Dualprogramm der Kugel: Σ_i y_i,0 = 1 (die Gleichung c + Aᵀy = 0 für R) und der Dualwert (Summe der gewichteten Punkte) ist gleich R*

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kad_admm as AD
import kad_forms as F
from kad_scenario import make_cloud

PTS = make_cloud(6, 3.0, 30, 7).points


def _cp():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp


def _ball(cp, norm):
    c, R = cp.Variable(2), cp.Variable()
    prob = cp.Problem(cp.Minimize(R), [norm(PTS[i] - c) <= R for i in range(len(PTS))])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    return float(R.value)


def _admm_value(prob, rho, tol=1e-9, cap=20000, adaptive=False):
    r = AD.admm(prob, rho=rho, tol=tol, max_iter=cap, adaptive=adaptive)
    assert r.converged
    return r


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_lorentz_and_psd_balls_equal_the_cvxpy_euclidean_ball():
    cp = _cp()
    ref = _ball(cp, cp.norm)
    assert _admm_value(F.build_soc_ball(PTS), 0.1).value == pytest.approx(ref, rel=1e-6)
    assert _admm_value(F.build_psd_ball(PTS), 0.1).value == pytest.approx(ref, rel=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_power_cone_ball_equals_the_cvxpy_p_norm_ball():
    cp = _cp()
    ref = _ball(cp, lambda v: cp.norm(v, 3))
    assert _admm_value(F.build_lp_ball(PTS, 3.0), 0.03).value == pytest.approx(ref, rel=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_exponential_cone_ball_equals_the_cvxpy_log_sum_exp_ball():
    cp = _cp()
    beta = 10.0
    c, R = cp.Variable(2), cp.Variable()
    cons = [cp.log_sum_exp(cp.hstack([beta * (PTS[i] - c), -beta * (PTS[i] - c)])) / beta <= R for i in range(len(PTS))]
    cp.Problem(cp.Minimize(R), cons).solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    assert _admm_value(F.build_lse_ball(PTS, beta), 0.01, tol=1e-8, adaptive=True).value == pytest.approx(float(R.value), rel=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_scs_the_admm_of_cvxpy_finds_the_same_value():
    cp = _cp()
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS fehlt")
    c, R = cp.Variable(2), cp.Variable()
    prob = cp.Problem(cp.Minimize(R), [cp.norm(PTS[i] - c) <= R for i in range(len(PTS))])
    prob.solve(solver=cp.SCS, eps=1e-9)
    assert _admm_value(F.build_soc_ball(PTS), 0.1).value == pytest.approx(float(R.value), rel=1e-5)


def test_the_dual_variable_is_feasible_for_the_dual_program_of_the_ball():
    """Dual der Kugel: max −Σ_i ⟨y_i, b_i⟩ unter Σ_i y_i,0 = 1 (Koeffizient von R), Σ_i y_i,(1:) = 0 (Koeffizient von c), y_i im Lorentz-Kegel; der Wert ist R*. Prüft y = ρu aus dem Lauf."""
    r = _admm_value(F.build_soc_ball(PTS), 0.1)
    prob = F.build_soc_ball(PTS)
    y = (r.rho * r.u).reshape(-1, 3)
    assert (y[:, 0] >= np.linalg.norm(y[:, 1:], axis=1) - 1e-7).all()
    assert np.abs(prob.c + prob.A.T @ y.reshape(-1)).max() < 1e-6
    assert -float(prob.b @ y.reshape(-1)) == pytest.approx(r.value, rel=1e-6)
