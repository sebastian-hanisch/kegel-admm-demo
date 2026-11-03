"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Bänder statt Punktwerte, wo die Rundung oder der Rechner die Zahl bewegt; Zeiten werden nicht auf Millisekunden getestet, nur ihre Struktur (Verhältnisse, Reihenfolge der Größenordnungen).
Die Schrittzahlen von ADMM sind deterministisch bis auf die Rundung der BLAS-Routinen: die Bänder lassen etwa 10 % Luft."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import kegel_kern as K
import kad_admm as AD
import kad_constants as C
import kad_evaluation as ev
import kad_projection as P

S = ev.Settings()


@pytest.fixture(scope="module")
def run():
    return ev.run_view(S)


@pytest.fixture(scope="module")
def four():
    return {r["kind"]: r for r in ev.four_rows(S)}


@pytest.fixture(scope="module")
def scan():
    return {r["kind"]: r for r in ev.scan_rows(S)}


@pytest.fixture(scope="module")
def scale():
    rows, settings = ev.scale_rows(S), ev.setting_rows(S)
    return {r["kind"]: r for r in rows}, settings


@pytest.fixture(scope="module")
def relax():
    return {r["kind"]: r for r in ev.relax_rows(S)}, {r["kind"]: r for r in ev.warm_rows(S)}


@pytest.fixture(scope="module")
def versus():
    return {r["kind"]: r for r in ev.versus_rows(S)}


def _its(row):
    return {x["rho"]: (x["iterations"], x["converged"]) for x in row["results"]}


def test_the_default_run_and_the_four_optimum_values(run, four):
    """README: Standardlauf Lorentz mit ρ = 1: 457 Schritte, Wertfehler unter 10⁻¹², R* = 7,0359 (Lorentz, PSD), 6,2683 (ℓ₃), 5,0463 (log-sum-exp); Barriere 31, 30, 34, 43 Newton-Schritte."""
    sm = run["summary"]
    assert sm["converged"] and 400 <= sm["iterations"] <= 520 and sm["value_error"] < 1e-10 and sm["rho_changes"] == 0
    assert run["case"]["rstar"] == pytest.approx(7.0359, abs=5e-4) and four["s"]["rstar"] == pytest.approx(four["q"]["rstar"], abs=1e-7)
    assert four["p"]["rstar"] == pytest.approx(6.2683, abs=5e-4) and four["e"]["rstar"] == pytest.approx(5.0463, abs=5e-4)
    assert [four[k]["newton"] for k in ("q", "s", "p", "e")] == pytest.approx([31, 30, 34, 43], abs=6)
    assert 30 <= min(r["newton"] for r in four.values()) and max(r["newton"] for r in four.values()) <= 45


def test_the_four_cones_at_rho_one(four):
    """README (Experiment 1, ρ = 1, Toleranz 10⁻⁶): Lorentz 457, PSD 1066, ℓ₃ 620 Schritte; der Exponentialkegel ist nach 1500 Schritten nicht gelöst."""
    its = {k: r["summary"] for k, r in four.items()}
    assert 400 <= its["q"]["iterations"] <= 520 and 950 <= its["s"]["iterations"] <= 1200 and 550 <= its["p"]["iterations"] <= 700
    assert all(its[k]["converged"] for k in "qsp") and not its["e"]["converged"] and its["e"]["iterations"] == C.FOUR_CAP


def test_the_good_rho_is_a_flat_floor(scan):
    """README (Experiment 2): bestes ρ der Reihe: Lorentz 185 (ρ = 0,03 und 0,1), PSD 254 (0,03), ℓ₃ 207 (0,03), log-sum-exp 381 (0,01); die Schrittzahl wächst zu beiden Seiten."""
    best = {k: min((v[0], rho) for rho, v in _its(r).items() if v[1]) for k, r in scan.items()}
    assert 160 <= best["q"][0] <= 215 and 220 <= best["s"][0] <= 290 and 180 <= best["p"][0] <= 240 and 330 <= best["e"][0] <= 440
    assert {k: v[1] for k, v in best.items()} == {"q": 0.03, "s": 0.03, "p": 0.03, "e": 0.01} or best["q"][1] in (0.03, 0.1)
    assert all(0.01 <= v[1] <= 0.1 for v in best.values())
    for k, r in scan.items():
        its = _its(r)
        assert its[0.003][0] > best[k][0] * 1.3 and (not its[100.0][1])                     # links vom Boden mehr Schritte, rechts nichts gelöst


def test_the_floor_in_numbers(scan):
    """README: Lorentz bei ρ = 1 457 Schritte (2,5-mal der Boden), bei ρ = 3 nicht gelöst nach 1000; 16 von 36 Läufen der Reihe nicht gelöst (Deckel 1000)."""
    q = _its(scan["q"])
    assert 400 <= q[1.0][0] <= 520 and q[1.0][1] and not q[3.0][1] and q[3.0][0] == C.SCAN_CAP and 2.0 <= q[1.0][0] / q[0.1][0] <= 3.0
    failed = sum(1 for r in scan.values() for x in r["results"] if not x["converged"])
    assert sum(len(r["results"]) for r in scan.values()) == 36 and 12 <= failed <= 20
    assert not _its(scan["e"])[1.0][1] and _its(scan["e"])[0.01][1] and _its(scan["s"])[0.3][1] and not _its(scan["p"])[3.0][1]


def test_the_good_rho_moves_with_the_scale_and_the_balancing_recovers_most(scale):
    """README (Experiment 3, Start ρ = 1, Deckel 3000): Lorentz fest 185, 457, nicht gelöst, nicht gelöst; Ausgleich 185, 259, 460, 1486. PSD fest 268, 1066, –, –; Ausgleich 278, 333, 735, 1883."""
    rows, _ = scale
    for kind, fixed_ok, fixed, adapt in (("q", (True, True, False, False), (185, 457), ((185, 215), (230, 290), (400, 520), (1300, 1700))), ("s", (True, True, False, False), (268, 1066), ((250, 310), (290, 380), (650, 820), (1650, 2100)))):
        res = rows[kind]["results"]
        assert tuple(x["fixed"]["converged"] for x in res) == fixed_ok and all(x["adaptive"]["converged"] for x in res)
        assert abs(res[0]["fixed"]["iterations"] - fixed[0]) <= 25 and abs(res[1]["fixed"]["iterations"] - fixed[1]) <= 0.1 * fixed[1]
        for x, (lo, hi) in zip(res, adapt):
            assert lo <= x["adaptive"]["iterations"] <= hi, (kind, x["scale"])
        assert [x["scale"] for x in res] == list(C.SCALES)


def test_the_settings_of_the_balancing_scatter(scale):
    """README: Lorentz, Start ρ = 1: Skala 1: 213 bis 457 Schritte; Skala 100: 397 bis nicht gelöst (Deckel 3000); keine Einstellung ist auf beiden Skalen die beste."""
    _, settings = scale
    by = {f: [(row["every"], mu, row[mu]) for row in settings if row["scale"] == f for mu in C.MU_CHOICES] for f in C.SETTING_SCALES}
    a, b = [x[2]["iterations"] for x in by[1.0]], [x[2]["iterations"] for x in by[100.0]]
    assert len(a) == len(b) == 9 and 195 <= min(a) <= 235 and 420 <= max(a) <= 500 and 350 <= min(b) <= 450 and max(b) == C.SCALE_CAP and sum(1 for x in by[100.0] if not x[2]["converged"]) == 1
    assert max(b) / min(b) > 3
    best1 = min(by[1.0], key=lambda x: x[2]["iterations"])[:2]
    best100 = min(by[100.0], key=lambda x: x[2]["iterations"])[:2]
    assert best1 != best100


def test_over_relaxation_does_not_help(relax):
    """README (Experiment 4, bestes ρ aus 0,03/0,1/0,3): Lorentz α = 1; 1,4; 1,6; 1,8: 185, 206, 271, 496; PSD 254, 292, 390, 670: je größer α, desto mehr Schritte."""
    rel, _ = relax
    for kind, expected in (("q", (185, 206, 271, 496)), ("s", (254, 292, 390, 670))):
        its = [x["iterations"] for x in rel[kind]["results"]]
        assert all(x["converged"] for x in rel[kind]["results"]) and its == sorted(its) and len(set(its)) == 4
        assert all(abs(a - b) <= 0.12 * b for a, b in zip(its, expected))
        assert {x["alpha"] for x in rel[kind]["results"]} == set(C.ALPHAS)


def test_warm_start_continues_but_does_not_save(relax):
    """README: ρ = 0,1, Toleranz 10⁻⁶: kalt Lorentz 185, PSD 268, ℓ₃ 274, Exponential 1231; Start im groben x allein spart 4 bis 6 % (Exponential 50 %); grober Lauf (10⁻³) plus Warmstart mit x und u gleich dem kalten Lauf."""
    _, warm = relax
    cold = {k: r["cold"] for k, r in warm.items()}
    assert 165 <= cold["q"] <= 205 and 240 <= cold["s"] <= 300 and 245 <= cold["p"] <= 305 and 1100 <= cold["e"] <= 1360 and all(r["converged"] for r in warm.values())
    for k in "qsp":
        assert 0.02 <= 1 - warm[k]["x_only"] / cold[k] <= 0.09, k
    assert 0.40 <= 1 - warm["e"]["x_only"] / cold["e"] <= 0.60
    for r in warm.values():
        assert abs(r["rough"] + r["both"] - r["cold"]) <= 0.03 * r["cold"] + 3, r["name"]                    # das Weiterlaufen mit u ist genau die Fortsetzung
        assert r["both"] < r["x_only"] <= r["cold"]


def test_admm_against_the_barrier(versus):
    """README (Experiment 5, Ziele Wertfehler 10⁻³/10⁻⁶/10⁻⁹): Barriere Lorentz 18/23/28, PSD 17/22/26, ℓ₃ 21/29/34, Exponential 29/36/43 Newton-Schritte; ADMM Lorentz 52/165/252, PSD 77/206/324,
    ℓ₃ 60/191/344, Exponential 140/313/460 Schritte (ρ = 0,03 bei allen vier Kegeln); beide Verfahren treffen das Ziel."""
    bar = {"q": (18, 23, 28), "s": (17, 22, 26), "p": (21, 29, 34), "e": (29, 36, 43)}
    adm = {"q": (52, 165, 252), "s": (77, 206, 324), "p": (60, 191, 344), "e": (140, 313, 460)}
    for kind, row in versus.items():
        res = row["results"]
        assert [x["target"] for x in res] == list(C.TARGETS)
        assert all(abs(x["barrier_steps"] - b) <= 4 for x, b in zip(res, bar[kind])), kind
        assert all(abs(x["admm_iterations"] - a) <= 0.15 * a + 3 for x, a in zip(res, adm[kind])), kind
        assert all(x["admm_converged"] and x["barrier_error"] <= x["target"] and x["admm_error"] <= x["target"] * 1.0001 for x in res), kind
        assert row["variables"] in (3, 19, 35) and row["rows"] in (24, 48, 56, 104)


def test_the_time_ratios_have_the_claimed_structure(versus):
    """README: Lorentz und PSD: ADMM-Schritt billiger als ein Newton-Schritt, aber das Drei- bis Zwölffache an Schritten; Zeit gesamt 0,6- bis 3-mal die Barriere bei 10⁻³ (gleichauf), 2- bis 3-mal bei 10⁻⁹;
    Potenz- und Exponentialkegel: der ADMM-Schritt ist teurer als ein Newton-Schritt, ADMM braucht dort das Vier- bis Fünfundzwanzigfache der Zeit (Entwicklungsrechner 7 bis 25, Linux-Runner 4 bis 18). Zeiten schwanken mit dem Rechner: weite Bänder."""
    ratio_steps = [x["admm_iterations"] / x["barrier_steps"] for k in "qs" for x in versus[k]["results"]]
    assert 2.5 <= min(ratio_steps) <= 3.6 and 10.0 <= max(ratio_steps) <= 14.0
    for k in "qs":
        assert versus[k]["step_ms_admm"] < versus[k]["step_ms_barrier"]
    for k in "pe":
        assert versus[k]["step_ms_admm"] > versus[k]["step_ms_barrier"]
    coarse = [x["admm_seconds"] / x["barrier_seconds"] for k in "qs" for x in versus[k]["results"] if x["target"] == 1e-3]
    fine = [x["admm_seconds"] / x["barrier_seconds"] for k in "qs" for x in versus[k]["results"] if x["target"] == 1e-9]
    assert 0.4 <= min(coarse) and max(coarse) <= 3.0 and 1.5 <= min(fine) and max(fine) <= 8.0 and np.mean(fine) > np.mean(coarse)
    heavy = [x["admm_seconds"] / x["barrier_seconds"] for k in "pe" for x in versus[k]["results"]]
    assert min(heavy) >= 2.5 and max(heavy) <= 150.0


def test_the_projection_of_a_few_exponential_cones_costs_about_the_same_as_of_many():
    """README: ein Aufruf von proj_exp kostet bei 1 und bei 32 Kegeln etwa gleich viel (Python-Aufwand, unabhängig von der Zahl der Kegel bis etwa 30); bei 200 Kegeln wächst er. Nur das Verhältnis zählt."""
    import time

    rng = np.random.default_rng(0)
    cost = {}
    for n in (1, 32):
        V = rng.normal(size=(n, 3))
        P.proj_exp(V)
        best = min(((lambda t0: (P.proj_exp(V), time.perf_counter() - t0)[1])(time.perf_counter())) for _ in range(30))
        cost[n] = best
    assert cost[32] / cost[1] < 3.0


def test_accuracy_is_high_not_coarse():
    """README: Lorentz mit ρ = 1 hat nach 300 Schritten einen Wertfehler unter 10⁻⁸ und nach 1000 unter 10⁻¹³; bei ρ = 0,03 ist die Residuen-Toleranz 10⁻⁹ nach 270 Schritten erreicht."""
    case = ev._case(S, "q")
    for steps, bound in ((10, None), (300, 1e-8), (1000, 1e-13)):
        r = AD.admm(case["prob"], rho=1.0, tol=0.0, max_iter=steps)
        err = ev.value_error(r.value, case["rstar"])
        if bound:
            assert err < bound, steps
    r = AD.admm(case["prob"], rho=0.03, tol=1e-9, max_iter=5000)
    assert r.converged and 220 <= r.iterations <= 320 and ev.value_error(r.value, case["rstar"]) < 1e-8


def test_a_run_of_the_summary_matches_a_direct_call():
    case = ev._case(S, "q")
    direct = AD.admm(case["prob"], rho=0.1, tol=1e-6, max_iter=C.MAX_ITER)
    view = ev.run_view(ev.Settings(rho=0.1))
    assert view["summary"]["iterations"] == direct.iterations and view["res"].value == direct.value and view["iters"][-1] == direct.iterations


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 8: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-projektion-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
    assert K.SQ2 == pytest.approx(2.0 ** 0.5)
