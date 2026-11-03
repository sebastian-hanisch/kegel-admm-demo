"""ADMM (kad_admm): Handbeispiele, Fixpunkt, Eigenschaften, Fehlerfälle (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kegel_kern as K
import kad_admm as AD
import kad_evaluation as ev
import kad_forms as F
import kad_projection as P
from kad_scenario import make_cloud


def test_the_lorentz_hand_program_step_by_step(tiny_cones):
    """min t mit (t; 3, 4) im Lorentz-Kegel, ρ = 1, Start x = 0, u = 0 (von Hand): s₀ = P(0; 3, 4) = (2,5; 1,5; 2).
    Schritt 1: x = −(1 + (−1)(2,5)) = 1,5; b − A x = (1,5; 3; 4), r = 5, P = ((5 + 1,5)/2)(1; 0,6; 0,8) = (3,25; 1,95; 2,6); u = (1,75; −1,05; −1,4); r = √6,125, d = |3,25 − 2,5| = 0,75."""
    prob, opt = tiny_cones["soc"]
    r1 = AD.admm(prob, rho=1.0, max_iter=1, tol=0.0)
    assert r1.x == pytest.approx([1.5]) and r1.s == pytest.approx([3.25, 1.95, 2.6]) and r1.u == pytest.approx([1.75, -1.05, -1.4])
    assert r1.primal == pytest.approx(np.sqrt(6.125)) and r1.dual == pytest.approx(0.75)
    r2 = AD.admm(prob, rho=1.0, max_iter=2, tol=0.0)
    assert r2.x == pytest.approx([4.0]) and r2.s == pytest.approx([4.5, 2.7, 3.6]) and r2.u == pytest.approx([2.25, -1.35, -1.8])
    assert r2.primal == pytest.approx(np.sqrt(0.5)) and r2.dual == pytest.approx(1.25)


def test_the_lorentz_hand_program_fixed_point(tiny_cones):
    """Fixpunkt: x = 5, s = (5; 3, 4), u = (1; −0,6; −0,8): c + ρ Aᵀu = 1 − 1 = 0, ⟨s, ρu⟩ = 5 − 1,8 − 3,2 = 0, y = ρu liegt auf dem Rand des Lorentz-Kegels."""
    prob, opt = tiny_cones["soc"]
    r = AD.admm(prob, rho=1.0, max_iter=400, tol=1e-12)
    assert r.converged and r.x == pytest.approx([5.0], abs=1e-9) and r.s == pytest.approx([5.0, 3.0, 4.0], abs=1e-9) and r.u == pytest.approx([1.0, -0.6, -0.8], abs=1e-9)
    assert r.value == pytest.approx(opt, abs=1e-9)
    y = r.rho * r.u
    assert np.linalg.norm(y[1:]) <= y[0] + 1e-9 and abs(r.s @ y) < 1e-9
    again = AD.admm(prob, rho=1.0, max_iter=3, tol=0.0, x0=r.x, u0=r.u)               # ein Start im Fixpunkt (mit u) bleibt dort
    assert again.x == pytest.approx(r.x, abs=1e-8) and again.u == pytest.approx(r.u, abs=1e-8)
    assert AD.admm(prob, rho=1.0, max_iter=1, tol=0.0, x0=np.array([5.0])).x == pytest.approx([4.0])      # ohne u bewegt sich x: der Zustand ist (s, u)


def test_the_orthant_hand_program(tiny_cones):
    """min x mit x ≥ 1 (A = −1, b = −1, ρ = 1): Schritt 1: x = 0, s = 0, u = 1; Schritt 2: x = 1, s = 0, u = 1, Residuum 0; Fixpunkt u = 1 = y."""
    prob, opt = tiny_cones["lp"]
    r1 = AD.admm(prob, rho=1.0, max_iter=1, tol=0.0)
    assert r1.x == pytest.approx([0.0]) and r1.s == pytest.approx([0.0]) and r1.u == pytest.approx([1.0]) and r1.primal == pytest.approx(1.0) and r1.dual == pytest.approx(0.0)
    r2 = AD.admm(prob, rho=1.0, max_iter=2, tol=0.0)
    assert r2.x == pytest.approx([1.0]) and r2.s == pytest.approx([0.0]) and r2.u == pytest.approx([1.0]) and r2.primal == pytest.approx(0.0, abs=1e-15)
    done = AD.admm(prob, rho=1.0, tol=1e-9)
    assert done.converged and done.iterations <= 4 and done.value == pytest.approx(opt)


def test_the_psd_hand_program_converges_to_the_optimum(tiny_cones):
    prob, opt = tiny_cones["sdp"]
    r = AD.admm(prob, rho=1.0, tol=1e-9, max_iter=3000)
    assert r.converged and r.value == pytest.approx(opt, abs=1e-6)


@pytest.fixture(scope="module")
def ball():
    pts = make_cloud(6, 3.0, 30, 7).points
    prob = F.build_soc_ball(pts)
    return prob, K.solve_barrier(prob, tol=1e-12)


def test_the_solution_satisfies_the_kkt_conditions(ball):
    """Am Ende gilt: s ∈ K, y = ρu ∈ K* (der Lorentz-Kegel ist selbstdual), ⟨s, y⟩ ≈ 0, A x + s ≈ b, c + Aᵀy ≈ 0."""
    prob, ref = ball
    r = AD.admm(prob, rho=0.1, tol=1e-9, max_iter=5000)
    assert r.converged and r.value == pytest.approx(ref.value, rel=1e-8)
    y = r.rho * r.u
    for blk in (r.s.reshape(-1, 3), y.reshape(-1, 3)):
        assert (blk[:, 0] >= np.linalg.norm(blk[:, 1:], axis=1) - 1e-7).all()
    assert abs(r.s @ y) < 1e-6 and np.abs(prob.A @ r.x + r.s - prob.b).max() < 1e-7 and np.abs(prob.c + prob.A.T @ y).max() < 1e-6


def test_the_result_does_not_depend_on_rho_or_on_the_residual_balancing(ball):
    prob, ref = ball
    values = [AD.admm(prob, rho=rho, tol=1e-9, max_iter=20000).value for rho in (0.03, 0.3, 3.0)]
    values += [AD.admm(prob, rho=100.0, tol=1e-9, max_iter=20000, adaptive=True).value, AD.admm(prob, rho=0.001, tol=1e-9, max_iter=20000, adaptive=True, every=10).value]
    assert max(abs(v - ref.value) for v in values) < 1e-7


def test_the_balancing_changes_rho_and_rescales_u(ball):
    prob, _ = ball
    plain = AD.admm(prob, rho=100.0, tol=1e-6, max_iter=300)
    adapt = AD.admm(prob, rho=100.0, tol=1e-6, max_iter=300, adaptive=True)
    assert plain.rho == 100.0 and plain.rho_changes == 0 and adapt.rho_changes > 0
    assert adapt.rho != 100.0 and set(adapt.history["rho"]) <= {100.0 * 2.0 ** k for k in range(-12, 12)} and len(adapt.history["value"]) == adapt.iterations


def test_the_scale_of_the_data_moves_the_good_rho():
    """Ein festes ρ = 1 löst die Wolke mit Faktor 1 und scheitert bei Faktor 100; das ρ des Bodens wandert mit (Kern von Experiment 3)."""
    pts = make_cloud(8, 4.0, 45, 35).points
    small, big = F.build_soc_ball(pts), F.build_soc_ball(100.0 * pts)
    assert AD.admm(small, rho=1.0, tol=1e-6, max_iter=3000).converged and not AD.admm(big, rho=1.0, tol=1e-6, max_iter=3000).converged
    assert AD.admm(big, rho=0.01, tol=1e-6, max_iter=3000).converged


def test_residual_stop_and_reference_stop(ball):
    prob, ref = ball
    a = AD.admm(prob, rho=0.1, tol=1e-6)
    b = AD.admm(prob, rho=0.1, tol=1e-6, ref_value=ref.value)
    assert a.converged and b.converged and abs(b.value - ref.value) <= 1e-6 * abs(ref.value) + 1e-12 and b.primal <= 1e-6 * (1 + np.linalg.norm(prob.b))


def test_the_history_has_one_entry_per_step(ball):
    prob, _ = ball
    r = AD.admm(prob, rho=0.1, tol=1e-6)
    assert all(len(v) == r.iterations for v in r.history.values()) and r.history["primal"][-1] == r.primal and r.history["value"][-1] == r.value


def test_rank_deficient_a_and_bad_rho_are_rejected(tiny_cones):
    prob, _ = tiny_cones["soc"]
    bad = K.ConeProblem(np.array([1.0, 1.0]), np.array([[-1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]), np.array([0.0, 3.0, 4.0]), (("q", 3),), np.array([6.0, 0.0]))
    with pytest.raises(ValueError, match="Spaltenrang"):
        AD.admm(bad)
    for rho in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            AD.admm(prob, rho=rho)


def test_an_unbounded_program_ends_at_the_step_cap_without_a_certificate():
    """min −t mit (t; 3, 4) im Lorentz-Kegel ist unbeschränkt. Diese Form hat kein Zertifikat (keine homogene Einbettung): der Lauf endet am Deckel, nicht gelöst."""
    unb = K.ConeProblem(np.array([-1.0]), np.array([[-1.0], [0.0], [0.0]]), np.array([0.0, 3.0, 4.0]), (("q", 3),), np.array([6.0]))
    r = AD.admm(unb, rho=1.0, tol=1e-6, max_iter=300)
    assert not r.converged and r.iterations == 300


def test_spec_of_groups_equal_cones_and_keeps_programs_groups():
    pts = make_cloud(6, 2.0, 10, 1).points
    assert AD.spec_of(F.build_soc_ball(pts)) == [("q", 3, 6)] and AD.spec_of(F.build_psd_ball(pts)) == [("s", 3, 6)]
    assert AD.spec_of(F.build_lp_ball(pts, 3.0)) == [("p", 1.0 / 3.0, 12), ("l", None, 6)] and AD.spec_of(F.build_lse_ball(pts, 2.0))[0] == ("e", None, 24)


def test_project_spec_projects_block_by_block():
    spec = [("l", None, 2), ("q", 3, 1), ("s", 2, 1), ("e", None, 1), ("p", 0.5, 1)]
    s = np.array([-1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 0.0, -1.0, 0.0, 1.0, 2.0, 2.0, 8.0, 4.1])
    out = AD.project_spec(spec, s)
    assert out[:2].tolist() == [0.0, 2.0] and out[2:5] == pytest.approx([3.0, 1.8, 2.4]) and out[5:8] == pytest.approx([2.0, 0.0, 0.0]) and out[8:11].tolist() == [0.0, 1.0, 2.0] and len(out) == len(s)
    assert P.pow_member(out[11:14][None, :], 0.5, 1e-9)[0] and not P.pow_member(s[11:14][None, :], 0.5)[0]


def test_hand_helpers_of_the_evaluation_match_the_tests():
    rows = ev.hand_rows()
    assert [r["step"] for r in rows][:4] == [1, 2, 3, 4] and rows[0]["x"] == pytest.approx(1.5) and rows[1]["x"] == pytest.approx(4.0) and rows[2]["x"] == pytest.approx(5.75) and rows[3]["x"] == pytest.approx(6.25)
    assert rows[-1]["x"] == pytest.approx(5.0, abs=1e-9) and rows[-1]["step"] == 80


def test_an_orthant_of_size_m_is_projected_on_every_row():
    """min x mit x ≥ 1 und x ≥ 2 (A = (−1; −1), b = (−1; −2), ein Orthant der Größe 2): der Wert ist 2 (von Hand). Die Zerlegung `spec_of` muss alle 2 Zeilen projizieren (vorher: ("l", 2, 1), eine Zeile,
    Wert 1); ein Orthant der Größe 2 neben einem Lorentz-Kegel behält Reihenfolge und Anzahl."""
    p = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [-1.0]]), np.array([-1.0, -2.0]), (("l", 2),), np.array([3.0]))
    assert AD.spec_of(p) == [("l", None, 2)]
    assert AD.project_spec(AD.spec_of(p), np.array([-1.0, 2.0])).tolist() == [0.0, 2.0]
    r = AD.admm(p, rho=1.0, tol=1e-8, max_iter=2000)
    assert r.converged and r.value == pytest.approx(2.0, abs=1e-6)
    mixed = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [-1.0], [0.0], [0.0]]), np.array([-1.0, -1.0, 3.0, 4.0]), (("l", 2), ("q", 2)), np.array([3.0]))
    assert [k for k, _, _ in AD.spec_of(mixed)] == ["l", "q"]
    two = K.ConeProblem(np.array([1.0]), np.array([[-1.0], [-1.0]]), np.array([-1.0, -2.0]), (("l", 1), ("l", 1)), np.array([3.0]))
    assert AD.admm(two, rho=1.0, tol=1e-8, max_iter=2000).value == pytest.approx(2.0, abs=1e-6)             # zwei Orthanten der Größe 1: derselbe Wert
