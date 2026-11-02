"""Randfälle, an denen ADMM erfahrungsgemäß bricht (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kad_admm as AD
import kad_constants as C
import kad_evaluation as ev
import kad_forms as F
from kad_scenario import make_cloud

PTS = make_cloud(8, 4.0, 45, 35).points


def test_extreme_rho_ends_at_the_cap_with_finite_values():
    prob = F.build_soc_ball(PTS)
    for rho in (1e-4, 1e3):
        r = AD.admm(prob, rho=rho, tol=1e-6, max_iter=1000)
        assert not r.converged and r.iterations == 1000 and np.isfinite(r.value) and np.isfinite(r.primal) and np.isfinite(r.dual)


def test_the_balancing_finds_its_way_on_data_of_very_different_scale():
    """Lorentz-Kugel der Wolke mal 10⁻⁴ und mal 10⁴, Start ρ = 1 mit Ausgleich: beide Läufe konvergieren (ρ endet bei 2 und bei etwa 0,002)."""
    for f, lo, hi in ((1e-4, 1.0, 4.0), (1e4, 1e-3, 1e-2)):
        prob = F.build_soc_ball(f * PTS)
        r = AD.admm(prob, rho=1.0, tol=1e-6, max_iter=20000, adaptive=True)
        assert r.converged and lo <= r.rho <= hi and r.rho_changes > 5, f


def test_the_tolerance_is_absolute_below_scale_one():
    """Die Toleranz ist relativ zu max(‖A x‖, ‖s‖, ‖b‖, 1): bei Daten der Größe 10⁻⁴ ist sie faktisch absolut. Das Verfahren hält an, der Wertfehler bleibt bei etwa 10⁻³ bis 10⁻⁴ (README)."""
    prob = F.build_soc_ball(1e-4 * PTS)
    ref = ev.solve_barrier(prob, 1e-12).value
    r = AD.admm(prob, rho=1.0, tol=1e-6, max_iter=20000, adaptive=True)
    assert r.converged and 1e-5 < ev.value_error(r.value, ref) < 1e-2
    tight = AD.admm(prob, rho=1.0, tol=1e-6, max_iter=20000, adaptive=True, ref_value=ref)
    assert tight.converged and ev.value_error(tight.value, ref) <= 1e-6


def test_a_huge_start_of_u_is_forgotten():
    prob = F.build_soc_ball(PTS)
    r = AD.admm(prob, rho=0.1, tol=1e-6, max_iter=3000, u0=1e3 * np.ones_like(prob.b))
    assert r.converged and r.iterations < 400


def test_tolerance_zero_runs_to_the_cap():
    prob = F.build_soc_ball(PTS)
    r = AD.admm(prob, rho=0.1, tol=0.0, max_iter=50)
    assert not r.converged and r.iterations == 50 and len(r.history["value"]) == 50


def test_clouds_of_two_and_three_points_solve_in_all_cones():
    for n in (2, 3):
        for case in ev.cases(ev.Settings(n=n)):
            r = AD.admm(case["prob"], rho=0.1, tol=1e-6, max_iter=3000)
            assert np.isfinite(r.value) and ev.value_error(r.value, case["rstar"]) < 1e-2, (n, case["kind"])
            if case["kind"] != "e" or n == 2:
                assert r.converged, (n, case["kind"])


def test_the_view_covers_every_slider_corner_for_every_cone():
    for kind in C.KINDS:
        for n, stretch, angle in ((C.N_MIN, C.STRETCH_MIN, C.ANGLE_MIN), (C.N_MIN, C.STRETCH_MAX, C.ANGLE_MAX)):
            s = ev.Settings(kind=kind, rho=C.RHO_CHOICES[0], tol=C.TOL_CHOICES[-1], n=n, stretch=stretch, angle=angle)
            view = ev.run_view(s)
            assert ev.verdict_run(s, view)[0] in {"success", "warning"} and len(view["iters"]) == len(view["primal"]) == len(view["dual"]) == len(view["error"]) <= 401
            assert view["iters"][-1] == view["summary"]["iterations"]


def test_the_history_is_thinned_to_at_most_400_points_and_ends_at_the_last_step():
    view = ev.run_view(ev.Settings(rho=100.0))
    assert view["summary"]["iterations"] == C.MAX_ITER and not view["summary"]["converged"] and len(view["iters"]) <= 401 and view["iters"][0] == 1 and view["iters"][-1] == C.MAX_ITER


def test_the_experiments_run_at_the_smallest_cloud():
    s = ev.Settings(n=C.N_MIN, stretch=C.STRETCH_MIN, angle=0, seed=1)
    assert len(ev.scan_rows(s, rhos=(0.1, 100.0), cap=200)) == 4
    assert len(ev.scale_rows(s, scales=(1.0, 100.0), cap=300)) == len(C.SCALE_KINDS)
    assert len(ev.relax_rows(s, cap=500)) == len(C.RELAX_KINDS)
    assert all(r["cold"] >= 1 for r in ev.warm_rows(s, cap=500))
    assert all(len(r["results"]) == 3 for r in ev.versus_rows(s, cap=2000))
