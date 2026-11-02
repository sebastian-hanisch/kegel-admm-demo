"""Die Projektionen (kad_projection): Handwerte, die Moreau-Probe als Zertifikat, Eigenschaften, Randfälle."""

import math

import numpy as np
import pytest

import kad_projection as P

CONES = [("l", None), ("q", None), ("s", 2), ("e", None), ("p", 0.3), ("p", 0.5), ("p", 0.75)]


def _points(rng, n=300, scale=2.0):
    return scale * rng.normal(size=(n, 3))


def test_orthant_by_hand():
    assert P.proj_orthant(np.array([[-1.0, 2.0, -3.0]]))[0].tolist() == [0.0, 2.0, 0.0]


def test_lorentz_by_hand():
    """(1; 3, 4): ‖v̄‖ = 5 > t, also ((5 + 1)/2) (1; 3/5, 4/5) = (3; 1,8; 2,4); (5; 3, 4) liegt im Kegel, (−5; 3, 4) im Polarkegel."""
    out = P.proj_lorentz(np.array([[1.0, 3.0, 4.0], [5.0, 3.0, 4.0], [-5.0, 3.0, 4.0]]))
    assert out[0] == pytest.approx([3.0, 1.8, 2.4]) and out[1].tolist() == [5.0, 3.0, 4.0] and out[2].tolist() == [0.0, 0.0, 0.0]


def test_psd_by_hand():
    """diag(2, −1) (svec: 2, 0, −1) gibt diag(2, 0); [[0, 1], [1, 0]] (svec: 0, √2, 0) gibt ½ [[1, 1], [1, 1]] (svec: ½, √2/2, ½)."""
    out = P.proj_psd(np.array([[2.0, 0.0, -1.0], [0.0, math.sqrt(2.0), 0.0]]), 2)
    assert out[0] == pytest.approx([2.0, 0.0, 0.0]) and out[1] == pytest.approx([0.5, math.sqrt(2.0) / 2.0, 0.5])


def test_exponential_cone_by_hand():
    """(0, 1, 2) liegt in K (1 · e⁰ = 1 ≤ 2), (1, −1, −1) im Polarkegel (−v = (−1, 1, 1) ∈ K*, e⁻¹ ≤ e)."""
    out = P.proj_exp(np.array([[0.0, 1.0, 2.0], [1.0, -1.0, -1.0]]))
    assert out[0].tolist() == [0.0, 1.0, 2.0] and out[1].tolist() == [0.0, 0.0, 0.0]
    assert P.exp_member(np.array([[0.0, 1.0, 2.0]]))[0] and P.exp_dual_member(np.array([[-1.0, 1.0, 1.0]]))[0] and not P.exp_member(np.array([[1.0, -1.0, -1.0]]))[0]


def test_exponential_face_by_hand():
    """v = (−1, −2, 3): die Projektion liegt auf der Fläche y = 0 ({x ≤ 0, z ≥ 0}): p = (−1, 0, 3); der Rest (0, −2, 0) liegt im Polarkegel (−r = (0, 2, 0) ∈ Abschluss von K*)."""
    V = np.array([[-1.0, -2.0, 3.0]])
    p = P.proj_exp(V)[0]
    assert p == pytest.approx([-1.0, 0.0, 3.0], abs=1e-9) and P.certificate("e", None, V, p[None, :])["ok"][0]


def test_power_cone_membership_by_hand():
    """a = ½: uv ≥ w² (u = 2, v = 8: |w| ≤ 4); der Dualkegel 4uv ≥ w²: (u, v, w) = (2, 8, 7) liegt in K*, nicht in K."""
    assert P.pow_member(np.array([[2.0, 8.0, 4.0]]), 0.5)[0] and not P.pow_member(np.array([[2.0, 8.0, 4.1]]), 0.5)[0]
    assert P.pow_dual_member(np.array([[2.0, 8.0, 7.0]]), 0.5)[0] and not P.pow_dual_member(np.array([[2.0, 8.0, 9.0]]), 0.5)[0]


@pytest.mark.parametrize("kind,par", CONES)
def test_the_projection_lies_in_the_cone_and_the_moreau_probe_passes(kind, par):
    rng = np.random.default_rng(1)
    V = _points(rng)
    Pv = P.project(kind, par, V)
    cert = P.certificate(kind, par, V, Pv)
    assert cert["in_cone"].all() and cert["orthogonal"].all()
    assert cert["ok"].mean() >= (1.0 if kind != "e" else 0.99)        # Exponentialkegel: Randfälle w = 0 der Dualkegel-Bedingung (Toleranz), siehe README


@pytest.mark.parametrize("kind,par", CONES)
def test_idempotent_nonexpansive_and_positively_homogeneous(kind, par):
    rng = np.random.default_rng(2)
    V = _points(rng, 400)
    Pv = P.project(kind, par, V)
    assert np.abs(P.project(kind, par, Pv) - Pv).max() < 1e-9                                        # Idempotenz
    A, B = V[:200], V[200:]
    assert np.all(np.linalg.norm(P.project(kind, par, A) - P.project(kind, par, B), axis=1) <= np.linalg.norm(A - B, axis=1) + 1e-9)       # nichtexpansiv
    lam = 3.7
    assert np.abs(P.project(kind, par, lam * V) - lam * Pv).max() < 1e-7 * (1 + np.abs(Pv).max())    # P(λ v) = λ P(v), λ > 0


@pytest.mark.parametrize("kind,par", CONES)
def test_the_projection_is_the_nearest_point(kind, par):
    """Kein Punkt einer Zufallsprobe des Kegels liegt näher an v als P(v) (Probe aus Projektionen anderer Punkte)."""
    rng = np.random.default_rng(3)
    V = _points(rng, 60)
    others = P.project(kind, par, _points(rng, 400))
    for v, p in zip(V, P.project(kind, par, V)):
        assert np.linalg.norm(v - p) <= np.linalg.norm(others - v, axis=1).min() + 1e-9


@pytest.mark.parametrize("kind,par", [("l", None), ("q", None), ("s", 2)])
def test_the_mirror_formula_holds_for_the_self_dual_cones(kind, par):
    V = _points(np.random.default_rng(4), 500)
    assert np.abs(V - (P.project(kind, par, V) - P.project(kind, par, -V))).max() < 1e-12


@pytest.mark.parametrize("kind,par", [("e", None), ("p", 0.3), ("p", 0.5)])
def test_the_mirror_formula_fails_for_the_cones_that_are_not_self_dual(kind, par):
    V = _points(np.random.default_rng(4), 500)
    assert np.abs(V - (P.project(kind, par, V) - P.project(kind, par, -V))).max() > 0.5
    assert not P.SELF_DUAL[kind]


def test_the_polar_part_is_the_projection_onto_the_polar_cone_for_self_dual_cones():
    V = _points(np.random.default_rng(5), 300)
    for kind, par in (("l", None), ("q", None), ("s", 2)):
        R = V - P.project(kind, par, V)
        assert np.abs(R - (-P.project(kind, par, -V))).max() < 1e-12


def test_v_inside_the_cone_or_inside_the_polar_cone():
    for kind, par, inside, polar in (("l", None, [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]), ("q", None, [5.0, 3.0, 4.0], [-5.0, 3.0, 4.0]), ("s", 2, [2.0, 0.0, 1.0], [-2.0, 0.0, -1.0]),
                                     ("e", None, [0.0, 1.0, 2.0], [1.0, -1.0, -1.0]), ("p", 0.5, [2.0, 8.0, 3.0], [-1.0, -1.0, 0.5])):
        assert np.allclose(P.project(kind, par, np.array([inside]))[0], inside) and np.allclose(P.project(kind, par, np.array([polar]))[0], 0.0), kind


def test_the_certificate_rejects_a_wrong_projection():
    """Ein beliebiger Punkt des Kegels ist keine Projektion: die Probe fällt durch."""
    V = np.array([[1.0, 0.5, -0.5]])
    p_true = P.proj_exp(V)
    assert P.certificate("e", None, V, p_true)["ok"][0]
    for wrong in (p_true + np.array([[0.05, 0.0, 0.05]]), np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 3.0]])):
        assert not P.certificate("e", None, V, wrong)["ok"][0]
    assert not P.certificate("q", None, np.array([[1.0, 3.0, 4.0]]), np.array([[3.0, 1.8, 2.5]]))["ok"][0]


def test_psd_projection_equals_the_eigenvalue_clip_for_larger_matrices():
    rng = np.random.default_rng(7)
    k = 5
    M = rng.normal(size=(20, k, k))
    M = (M + np.swapaxes(M, 1, 2)) / 2.0
    out = P.smat_stack(P.proj_psd(P.svec_stack(M), k), k)
    w, Q = np.linalg.eigh(M)
    ref = np.einsum("nij,nj,nkj->nik", Q, np.maximum(w, 0.0), Q)
    assert np.abs(out - ref).max() < 1e-12 and np.linalg.eigvalsh(out).min() > -1e-12


def test_svec_and_smat_are_inverse_and_isometric():
    rng = np.random.default_rng(8)
    M = rng.normal(size=(10, 4, 4))
    M = (M + np.swapaxes(M, 1, 2)) / 2.0
    S = P.svec_stack(M)
    assert np.allclose(P.smat_stack(S, 4), M) and np.allclose(np.linalg.norm(S, axis=1), np.linalg.norm(M.reshape(10, -1), axis=1))


def test_project_rejects_an_unknown_cone():
    with pytest.raises(ValueError, match="unbekannter Kegel"):
        P.project("x", None, np.zeros((1, 3)))
