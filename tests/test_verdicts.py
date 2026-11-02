"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import kad_constants as C
import kad_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich" and ev.sci(1e-6, 0) == "1·10⁻⁶"
    assert ev.rho_text(0.1) == "0,1" and ev.rho_text(1.0) == "1" and ev.rho_text(100.0) == "100" and ev.rho_text(0.003) == "0,003" and ev.rho_text(0.0125) == "0,013" and ev.rho_text(2.5) == "2,5"
    assert ev.steps_text(1) == "1 Schritt" and ev.steps_text(185) == "185 Schritte" and ev.ms_text(0.0043) == "4,3" and ev.ms_text(0.341) == "341" and ev.ms_text(0.0123) == "12"
    assert ev.count_text({"iterations": 457, "converged": True}) == "457" and ev.count_text({"iterations": 1000, "converged": False}) == "≥ 1000 (nicht gelöst)"
    assert ev.value_error(7.0, 7.0) == 0.0 and ev.value_error(7.7, 7.0) == __import__("pytest").approx(0.1)


def _view(converged=True, rho_changes=0, rho=0.1, primal=5e-8, dual=9e-7, iterations=185):
    return {"summary": {"iterations": iterations, "converged": converged, "value_error": 3e-12, "primal": primal, "dual": dual, "rho": rho, "rho_changes": rho_changes, "seconds": 0.01},
            "case": {"name": "Lorentz-Kegel", "rstar": 7.03591, "newton": 31}}


def test_run_verdict_names_steps_errors_residuals_and_the_barrier():
    kind, text = ev.verdict_run(ev.Settings(rho=0.1, n=8), _view())
    assert kind == "success"
    for part in ("ADMM mit ρ = 0,1 auf der Kugel der 8 Punkte als **Lorentz-Kegel**-Programm (R* = 7,03591, Barriere: 31 Newton-Schritte bis 10⁻⁹)", "**185 Schritte** bis zur Toleranz 1,0·10⁻⁶",
                 "Wertfehler 3,0·10⁻¹², primales Residuum 5,0·10⁻⁸, duales Residuum 9,0·10⁻⁷."):
        assert part in text, part
    kind, text = ev.verdict_run(ev.Settings(rho=100.0, adaptive=True), _view(rho_changes=3, rho=12.5))
    assert "ADMM mit Residuenausgleich mit ρ = 100" in text and "ρ wurde 3-mal geändert und steht am Ende bei 12,5" in text


def test_run_verdict_for_an_unsolved_run_names_the_larger_residual():
    kind, text = ev.verdict_run(ev.Settings(rho=100.0), _view(converged=False, iterations=5000, primal=3e-3, dual=1e-6))
    assert kind == "warning" and "**nicht gelöst** nach 5000 Schritten (Toleranz 1,0·10⁻⁶)" in text and "das primale Residuum ist das größere" in text
    assert "das duale Residuum ist das größere" in ev.verdict_run(ev.Settings(), _view(converged=False, iterations=5000, primal=1e-6, dual=3e-3))[1]


def _four(failed=False):
    def row(name, kind, its, ok, newton):
        return {"name": name, "kind": kind, "newton": newton, "summary": {"iterations": its, "converged": ok}}
    return [row("Lorentz-Kegel", "q", 457, True, 31), row("Exponentialkegel (log-sum-exp, β = 10)", "e", 1500, not failed, 43)]


def test_four_cone_verdict_counts_unsolved_runs_and_names_the_barrier():
    kind, text = ev.verdict_four(ev.Settings(rho=1.0), _four(failed=True))
    assert kind == "warning" and "Mit ρ = 1 und Toleranz 1,0·10⁻⁶" in text and "Lorentz-Kegel: 457 Schritte; Exponentialkegel: nicht gelöst nach 1500 Schritten" in text
    assert "31 bis 43 Newton-Schritte bis 10⁻⁹ (Lorentz-Kegel 31, Exponentialkegel 43)" in text
    kind, text = ev.verdict_four(ev.Settings(rho=0.1, adaptive=True), _four())
    assert kind == "info" and "Mit ρ = 0,1 und Residuenausgleich" in text and "Exponentialkegel: 1500 Schritte" in text


def _scan_rows():
    def res(rho, its, ok=True):
        return {"rho": rho, "iterations": its, "converged": ok}
    return [{"name": "Lorentz-Kegel", "kind": "q", "results": [res(0.03, 185), res(1.0, 457), res(100.0, 1000, False)]},
            {"name": "Exponentialkegel (log-sum-exp, β = 10)", "kind": "e", "results": [res(0.01, 381), res(1.0, 1000, False), res(100.0, 1000, False)]}]


def test_scan_verdict_names_the_floor_and_the_unsolved_runs():
    kind, text = ev.verdict_scan(_scan_rows(), 1.0, cap=1000)
    assert kind == "info"
    for part in ("Lorentz-Kegel: 185 Schritte bei ρ = 0,03, bei Ihrem ρ = 1: 457", "Exponentialkegel: 381 Schritte bei ρ = 0,01, bei Ihrem ρ = 1: ≥ 1000 (nicht gelöst)", "Von 6 Läufen sind 3 nach 1000 Schritten nicht gelöst"):
        assert part in text, part
    assert "bei Ihrem ρ" not in ev.verdict_scan(_scan_rows(), 3.0, cap=1000)[1]                      # ρ nicht in der Reihe: kein Vergleichswert


def _scale_rows():
    def res(f, fixed, fok, adapt, changes):
        return {"scale": f, "fixed": {"iterations": fixed, "converged": fok}, "adaptive": {"iterations": adapt, "converged": True, "rho_changes": changes, "rho": 0.03}}
    return [{"name": "Lorentz-Kegel", "kind": "q", "results": [res(0.1, 185, True, 185, 0), res(1.0, 457, True, 259, 2), res(10.0, 3000, False, 460, 4), res(100.0, 3000, False, 1486, 7)]}]


def _setting_rows():
    rows = []
    for f, base in ((1.0, 213), (100.0, 397)):
        for every in C.EVERY_CHOICES:
            row = {"scale": f, "every": every}
            for k, mu in enumerate(C.MU_CHOICES):
                row[mu] = {"iterations": base + 100 * k, "converged": not (f == 100.0 and every == 100 and mu == 100.0)}
            rows.append(row)
    return rows


def test_scale_verdict_lists_fixed_and_balanced_counts_and_the_spread():
    kind, text = ev.verdict_scale(_scale_rows(), _setting_rows(), ev.Settings(rho=1.0), cap=3000)
    assert kind == "info"
    for part in ("Skala der Wolke 0,1 / 1 / 10 / 100, Start ρ = 1", "Lorentz-Kegel: festes ρ 185, 457, ≥ 3000 (nicht gelöst), ≥ 3000 (nicht gelöst); Ausgleich 185, 259, 460, 1486", "(Deckel 3000 Schritte)",
                 "Skala 1: 213 bis 413; Skala 100: 397 bis ≥ 597"):
        assert part in text, part


def _relax_rows():
    def res(its):
        return [{"alpha": a, "iterations": n, "converged": True, "rho": 0.03} for a, n in zip(C.ALPHAS, its)]
    return [{"name": "Lorentz-Kegel", "kind": "q", "results": res((185, 206, 271, 496))}, {"name": "PSD-Kegel 3×3", "kind": "s", "results": res((254, 292, 390, 670))}]


def _warm_rows():
    return [{"name": "Lorentz-Kegel", "kind": "q", "rough": 88, "cold": 185, "x_only": 178, "both": 97, "converged": True},
            {"name": "Exponentialkegel (log-sum-exp, β = 10)", "kind": "e", "rough": 792, "cold": 1231, "x_only": 621, "both": 439, "converged": True}]


def test_relax_verdict_says_it_does_not_help_and_that_warm_start_only_continues():
    kind, text = ev.verdict_relax(_relax_rows(), _warm_rows())
    assert kind == "info"
    for part in ("Lorentz-Kegel: α = 1: 185, α = 1,4: 206, α = 1,6: 271, α = 1,8: 496", "Sie hilft hier nicht: je größer α, desto mehr Schritte", "Warmstart bis 10⁻⁶ (ρ = 0,1)",
                 "Lorentz-Kegel: kalt 185, mit x allein 178, mit x und u 97 (grober Lauf davor 88)", "Ein Start im groben x allein spart 4 % bis 50 % der kalten Schritte, mit x und u 48 % bis 64 %",
                 "88 + 97 = 185 gegen 185, 792 + 439 = 1231 gegen 1231"):
        assert part in text, part
    helps = [{**r, "results": [{**x, "iterations": x["iterations"] - (50 if x["alpha"] == 1.4 else 0)} for x in r["results"]]} for r in _relax_rows()]
    assert "Sie hilft bei einem Kegel." in ev.verdict_relax(helps, _warm_rows())[1]


def _versus_rows():
    def res(target, bs, bsec, ai, asec):
        return {"target": target, "barrier_steps": bs, "barrier_seconds": bsec, "barrier_error": target / 2, "admm_iterations": ai, "admm_converged": True, "admm_seconds": asec, "admm_rho": 0.03, "admm_error": target / 2}
    q = [res(1e-3, 18, 0.004, 52, 0.003), res(1e-6, 23, 0.005, 165, 0.010), res(1e-9, 28, 0.006, 252, 0.016)]
    e = [res(1e-3, 29, 0.013, 140, 0.15), res(1e-6, 36, 0.016, 313, 0.33), res(1e-9, 43, 0.02, 460, 0.47)]
    return [{"name": "Lorentz-Kegel", "kind": "q", "results": q, "step_ms_barrier": 0.2, "step_ms_admm": 0.06},
            {"name": "Exponentialkegel (log-sum-exp, β = 10)", "kind": "e", "results": e, "step_ms_barrier": 0.47, "step_ms_admm": 1.02}]


def test_versus_verdict_gives_steps_times_and_ratios():
    kind, text = ev.verdict_versus(_versus_rows())
    assert kind == "info"
    for part in ("Lorentz-Kegel: Barriere 23 Newton-Schritte / 5,0 ms, ADMM 165 Schritte / 10 ms", "Exponentialkegel: Barriere 36 Newton-Schritte / 16 ms, ADMM 313 Schritte / 330 ms",
                 "Ein ADMM-Schritt kostet in ms: Lorentz-Kegel 0,06, Exponentialkegel 1,02; ein Newton-Schritt der Barriere: Lorentz-Kegel 0,20, Exponentialkegel 0,47",
                 "ADMM braucht 0,8- bis 2,7-mal die Zeit der Barriere, bei Wertfehler 10⁻³ 0,8- bis 0,8-mal", "gewinnt die Barriere"):
        assert part in text, part


def test_scan_verdict_survives_a_cone_without_any_solved_run():
    rows = _scan_rows()
    rows[1] = {**rows[1], "results": [{"rho": x["rho"], "iterations": 1000, "converged": False} for x in rows[1]["results"]]}
    text = ev.verdict_scan(rows, 1.0, cap=1000)[1]
    assert "Exponentialkegel: kein ρ der Reihe löst es in 1000 Schritten, bei Ihrem ρ = 1: ≥ 1000 (nicht gelöst)" in text and "Lorentz-Kegel: 185 Schritte bei ρ = 0,03" in text
