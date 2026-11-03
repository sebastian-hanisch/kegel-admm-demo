"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import kegel_kern as K
import kad_admm as AD
import kad_barrier as B
import kad_constants as C
import kad_forms as F
from kad_scenario import make_cloud


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    rho: float = C.DEFAULT_RHO
    tol: float = C.DEFAULT_TOL
    adaptive: bool = False
    every: int = C.DEFAULT_EVERY
    mu: float = C.DEFAULT_MU
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def rho_text(rho: float) -> str:
    return de(rho, 3).rstrip("0").rstrip(",")


def _short(name: str) -> str:
    return name.split(" (")[0]


def steps_text(n: int) -> str:
    return "1 Schritt" if n == 1 else f"{n} Schritte"


def count_text(res: dict) -> str:
    """Schrittzahl eines Laufs, mit „≥“ und dem Deckel, wenn er nicht gelöst wurde."""
    return str(res["iterations"]) if res["converged"] else f"≥ {res['iterations']} (nicht gelöst)"


def ms_text(seconds: float) -> str:
    ms = 1e3 * seconds
    return de(ms, 1) if ms < 10 else de(ms, 0)


# ----------------------------------------------------------------------------- die vier Programme

def solve_barrier(prob, tol: float):
    return B.solve(prob, tol=tol) if hasattr(prob, "groups") else K.solve_barrier(prob, tol=tol)


def cases(s: Settings, scale: float = 1.0, kinds=C.KINDS) -> list:
    """Die Kugel der ersten n Punkte der Wolke (mit `scale` gestreckt) in den Kegeln `kinds`: Liste von Zeilen {name, kind, prob, rstar, newton, rows, variables}."""
    pts = scale * make_cloud(s.n, s.stretch, s.angle, s.seed).points
    builders = {"q": F.build_soc_ball, "s": F.build_psd_ball, "p": lambda p: F.build_lp_ball(p, 3.0), "e": lambda p: F.build_lse_ball(p, 10.0)}
    out = []
    for kind in kinds:
        prob = builders[kind](pts)
        out.append({"name": C.KIND_LABELS[kind], "kind": kind, "prob": prob, "rstar": solve_barrier(prob, 1e-12).value, "newton": solve_barrier(prob, 1e-9).newton_steps,
                    "rows": prob.A.shape[0], "variables": prob.A.shape[1]})
    return out


def _case(s: Settings, kind: str, scale: float = 1.0) -> dict:
    return cases(s, scale, (kind,))[0]


def value_error(value: float, rstar: float) -> float:
    return abs(value - rstar) / abs(rstar)



def _summary(res: AD.AdmmResult, rstar: float) -> dict:
    return {"iterations": res.iterations, "converged": res.converged, "value_error": value_error(res.value, rstar), "primal": res.primal, "dual": res.dual, "rho": res.rho,
            "rho_changes": res.rho_changes, "seconds": res.seconds}


# ----------------------------------------------------------------------------- Experiment 1: ein Lauf

def run_view(s: Settings) -> dict:
    """ADMM auf dem gewählten Kegel mit den Einstellungen der Seitenleiste; dazu der Verlauf (auf höchstens 400 Punkte ausgedünnt)."""
    case = _case(s, s.kind)
    res = AD.admm(case["prob"], rho=s.rho, tol=s.tol, max_iter=C.MAX_ITER, adaptive=s.adaptive, mu=s.mu, tau=C.TAU, every=s.every)
    h = res.history
    step = max(1, math.ceil(len(h["value"]) / 400))
    idx = list(range(0, len(h["value"]), step))
    if idx[-1] != len(h["value"]) - 1:
        idx.append(len(h["value"]) - 1)
    err = [value_error(h["value"][i], case["rstar"]) for i in idx]
    return {"case": case, "res": res, "summary": _summary(res, case["rstar"]), "iters": [i + 1 for i in idx], "primal": [h["primal"][i] for i in idx], "dual": [h["dual"][i] for i in idx],
            "error": err, "rho": [h["rho"][i] for i in idx], "dual_gap": float(abs(res.s @ (res.rho * res.u))) if res.converged else float("nan")}


def four_rows(s: Settings) -> list:
    """Dieselben Einstellungen (ρ, Toleranz, Ausgleich) auf allen vier Kegeln: Schritte, Wertfehler, Residuen, Zeit je Schritt."""
    rows = []
    for case in cases(s):
        res = AD.admm(case["prob"], rho=s.rho, tol=s.tol, max_iter=C.FOUR_CAP, adaptive=s.adaptive, mu=s.mu, tau=C.TAU, every=s.every)
        rows.append({"name": case["name"], "kind": case["kind"], "rstar": case["rstar"], "newton": case["newton"], "rows": case["rows"], "variables": case["variables"],
                     "summary": _summary(res, case["rstar"]), "step_us": 1e6 * res.seconds / res.iterations})
    return rows


def verdict_four(s: Settings, rows: list) -> tuple:
    parts = [f"{r['name'].split(' (')[0]}: " + (steps_text(r["summary"]["iterations"]) if r["summary"]["converged"] else f"nicht gelöst nach {r['summary']['iterations']} Schritten") for r in rows]
    newton = ", ".join(f"{r['name'].split(' (')[0]} {r['newton']}" for r in rows)
    failed = sum(1 for r in rows if not r["summary"]["converged"])
    return ("warning" if failed else "info"), (f"Mit ρ = {rho_text(s.rho)}{' und Residuenausgleich' if s.adaptive else ''} und Toleranz {sci(s.tol)} braucht ADMM auf denselben vier Programmen: " + "; ".join(parts)
                                               + f". Die Barriere braucht {min(r['newton'] for r in rows)} bis {max(r['newton'] for r in rows)} Newton-Schritte bis 10⁻⁹ ({newton}).")


def hand_program():
    """min t mit (t; 3, 4) im Lorentz-Kegel: A = (−1, 0, 0)ᵀ, b = (0, 3, 4), c = 1; Optimum t = 5."""
    return K.ConeProblem(np.array([1.0]), np.array([[-1.0], [0.0], [0.0]]), np.array([0.0, 3.0, 4.0]), (("q", 3),), np.array([6.0]))


def hand_rows(steps: int = 4) -> list:
    """Die ersten Schritte des Handbeispiels mit ρ = 1: x, s, u und die beiden Residuen, zuletzt der Fixpunkt."""
    prob = hand_program()
    rows = []
    for k in range(1, steps + 1):
        r = AD.admm(prob, rho=1.0, max_iter=k, tol=0.0)
        rows.append({"step": k, "x": float(r.x[0]), "s": r.s, "u": r.u, "primal": r.primal, "dual": r.dual})
    r = AD.admm(prob, rho=1.0, max_iter=400, tol=1e-12)
    rows.append({"step": r.iterations, "x": float(r.x[0]), "s": r.s, "u": r.u, "primal": r.primal, "dual": r.dual})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: wer wählt ρ?

def scan_rows(s: Settings, rhos=C.SCAN_RHOS, cap: int = C.SCAN_CAP) -> list:
    rows = []
    for case in cases(s):
        results = []
        for rho in rhos:
            r = AD.admm(case["prob"], rho=rho, tol=1e-6, max_iter=cap)
            results.append({"rho": rho, "iterations": r.iterations, "converged": r.converged})
        rows.append({"name": case["name"], "kind": case["kind"], "rstar": case["rstar"], "results": results})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: ρ wandert mit den Daten

def scale_rows(s: Settings, scales=C.SCALES, cap: int = C.SCALE_CAP) -> list:
    """Je Skala f der Wolke: festes ρ (aus der Seitenleiste) gegen Residuenausgleich ab demselben ρ, mit Intervall und Schwelle der Seitenleiste."""
    rows = []
    for kind in C.SCALE_KINDS:
        results = []
        for f in scales:
            case = _case(s, kind, f)
            fixed = AD.admm(case["prob"], rho=s.rho, tol=1e-6, max_iter=cap)
            adapt = AD.admm(case["prob"], rho=s.rho, tol=1e-6, max_iter=cap, adaptive=True, mu=s.mu, tau=C.TAU, every=s.every)
            results.append({"scale": f, "fixed": {"iterations": fixed.iterations, "converged": fixed.converged}, "adaptive": {"iterations": adapt.iterations, "converged": adapt.converged, "rho_changes": adapt.rho_changes,
                                                                                                                             "rho": adapt.rho}, "rstar": case["rstar"]})
        rows.append({"name": C.KIND_LABELS[kind], "kind": kind, "results": results})
    return rows


def setting_rows(s: Settings, cap: int = C.SCALE_CAP) -> list:
    """Intervall × Schwelle des Residuenausgleichs auf dem Lorentz-Programm, Start bei ρ aus der Seitenleiste, auf zwei Skalen."""
    rows = []
    for f in C.SETTING_SCALES:
        case = _case(s, "q", f)
        for every in C.EVERY_CHOICES:
            row = {"scale": f, "every": every}
            for mu in C.MU_CHOICES:
                r = AD.admm(case["prob"], rho=s.rho, tol=1e-6, max_iter=cap, adaptive=True, mu=mu, tau=C.TAU, every=every)
                row[mu] = {"iterations": r.iterations, "converged": r.converged}
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- Experiment 4: was nicht hilft

def relax_rows(s: Settings, cap: int = C.MAX_ITER) -> list:
    rows = []
    for kind in C.RELAX_KINDS:
        case = _case(s, kind)
        results = []
        for alpha in C.ALPHAS:
            best = None
            for rho in C.RELAX_RHOS:
                r = AD.admm(case["prob"], rho=rho, alpha=alpha, tol=1e-6, max_iter=cap)
                if best is None or (r.converged, -r.iterations) > (best[1], -best[0]):
                    best = (r.iterations, r.converged, rho)
            results.append({"alpha": alpha, "iterations": best[0], "converged": best[1], "rho": best[2]})
        rows.append({"name": case["name"], "kind": kind, "results": results})
    return rows


def warm_rows(s: Settings, cap: int = C.MAX_ITER) -> list:
    """Kalt (x = 0, u = 0) gegen Start im grob gelösten x (u = 0) gegen Start mit x und u aus dem groben Lauf (Toleranz 10⁻³), je bis 10⁻⁶; ρ = WARM_RHO."""
    rows = []
    for case in cases(s):
        rough = AD.admm(case["prob"], rho=C.WARM_RHO, tol=C.WARM_ROUGH_TOL, max_iter=cap)
        cold = AD.admm(case["prob"], rho=C.WARM_RHO, tol=1e-6, max_iter=cap)
        x_only = AD.admm(case["prob"], rho=C.WARM_RHO, tol=1e-6, max_iter=cap, x0=rough.x)
        both = AD.admm(case["prob"], rho=C.WARM_RHO, tol=1e-6, max_iter=cap, x0=rough.x, u0=rough.u)
        rows.append({"name": case["name"], "kind": case["kind"], "rough": rough.iterations, "cold": cold.iterations, "x_only": x_only.iterations, "both": both.iterations,
                     "converged": all(r.converged for r in (rough, cold, x_only, both))})
    return rows


# ----------------------------------------------------------------------------- Experiment 5: ADMM gegen die Barriere

def _best_time(fun, repeats: int = 3):
    out, best = None, math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fun()
        best = min(best, time.perf_counter() - t0)
    return out, best


def versus_rows(s: Settings, targets=C.TARGETS, cap: int = C.MAX_ITER) -> list:
    """Je Kegel und Wertfehler-Ziel: Newton-Schritte und Zeit der Barriere (kleinste Toleranz der Reihe, deren Wert das Ziel trifft) gegen Schritte und Zeit von ADMM (bestes ρ der kleinen Reihe,
    Abbruch, sobald Wertfehler und primales Residuum unter dem Ziel liegen; R* aus der Barriere ist dabei der Maßstab). Zeiten nach einem Warmlauf, beste von drei Läufen, ein BLAS-Thread."""
    rows = []
    with threadpool_limits(1):
        for case in cases(s):
            prob, rstar = case["prob"], case["rstar"]
            solve_barrier(prob, 1e-3)                                                                          # Warmlauf
            bar = []
            for tol in C.BARRIER_TOLS:
                res, sec = _best_time(lambda: solve_barrier(prob, tol))
                bar.append({"tol": tol, "steps": res.newton_steps, "seconds": sec, "error": value_error(res.value, rstar)})
            best = None                                                                                       # ρ einmal bei 10⁻⁶ aus der kleinen Reihe wählen
            for rho in C.RELAX_RHOS:
                r = AD.admm(prob, rho=rho, tol=1e-6, max_iter=cap, ref_value=rstar)
                if best is None or (r.converged, -r.iterations) > (best[0].converged, -best[0].iterations):
                    best = (r, rho)
            rho = best[1]
            results = []
            for target in targets:
                b = next((x for x in bar if x["error"] <= target), bar[-1])
                r = best[0] if target == 1e-6 else AD.admm(prob, rho=rho, tol=target, max_iter=cap, ref_value=rstar)
                _, sec = _best_time(lambda: AD.admm(prob, rho=rho, tol=target, max_iter=r.iterations, ref_value=rstar), 2)
                results.append({"target": target, "barrier_steps": b["steps"], "barrier_seconds": b["seconds"], "barrier_error": b["error"], "admm_iterations": r.iterations, "admm_converged": r.converged,
                                "admm_seconds": sec, "admm_rho": rho, "admm_error": value_error(r.value, rstar)})
            rows.append({"name": case["name"], "kind": case["kind"], "rstar": rstar, "variables": case["variables"], "rows": case["rows"], "results": results,
                         "step_ms_barrier": 1e3 * bar[-1]["seconds"] / bar[-1]["steps"], "step_ms_admm": 1e3 * results[-1]["admm_seconds"] / results[-1]["admm_iterations"]})
    return rows


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_run(s: Settings, view: dict) -> tuple:
    sm, case = view["summary"], view["case"]
    mode = " mit Residuenausgleich" if s.adaptive else ""
    head = f"ADMM{mode} mit ρ = {rho_text(s.rho)} auf der Kugel der {s.n} Punkte als **{case['name']}**-Programm (R* = {de(case['rstar'], 5)}, Barriere: {case['newton']} Newton-Schritte bis 10⁻⁹)"
    tail = f"Wertfehler {sci(sm['value_error'])}, primales Residuum {sci(sm['primal'])}, duales Residuum {sci(sm['dual'])}"
    if s.adaptive:
        tail += f"; ρ wurde {sm['rho_changes']}-mal geändert und steht am Ende bei {rho_text(sm['rho'])}"
    if sm["converged"]:
        return "success", f"{head}: **{steps_text(sm['iterations'])}** bis zur Toleranz {sci(s.tol)}. {tail}."
    bigger = "primale" if sm["primal"] > sm["dual"] else "duale"
    return "warning", f"{head}: **nicht gelöst** nach {C.MAX_ITER} Schritten (Toleranz {sci(s.tol)}). {tail}; das {bigger} Residuum ist das größere."


def verdict_scan(rows: list, rho: float, cap: int = C.SCAN_CAP) -> tuple:
    parts, failed, total = [], 0, 0
    for r in rows:
        ok = [x for x in r["results"] if x["converged"]]
        chosen = next((x for x in r["results"] if x["rho"] == rho), None)
        where = f"{steps_text(min(ok, key=lambda x: x['iterations'])['iterations'])} bei ρ = {rho_text(min(ok, key=lambda x: x['iterations'])['rho'])}" if ok else f"kein ρ der Reihe löst es in {cap} Schritten"
        parts.append(f"{r['name'].split(' (')[0]}: {where}" + (f", bei Ihrem ρ = {rho_text(rho)}: {count_text(chosen)}" if chosen else ""))
        failed += sum(1 for x in r["results"] if not x["converged"])
        total += len(r["results"])
    return "info", (f"Das beste ρ der Reihe je Kegel: " + "; ".join(parts) + f". Von {total} Läufen sind {failed} nach {cap} Schritten nicht gelöst (rechts vom flachen Boden). "
                    f"Die Schrittzahl wächst zu beiden Seiten des Bodens, soweit die Reihe dort Werte hat; wo der Boden liegt, steht oben je Kegel.")


def verdict_scale(rows: list, settings: list, s: Settings, cap: int = C.SCALE_CAP) -> tuple:
    parts = []
    for r in rows:
        fixed = ", ".join(count_text(x["fixed"]) for x in r["results"])
        adapt = ", ".join(count_text(x["adaptive"]) for x in r["results"])
        parts.append(f"{r['name'].split(' (')[0]}: festes ρ {fixed}; Ausgleich {adapt}")
    scales = " / ".join(rho_text(f) for f in C.SCALES)
    spread = []
    for f in C.SETTING_SCALES:
        cells = [row[mu] for row in settings if row["scale"] == f for mu in C.MU_CHOICES]
        its = [x["iterations"] for x in cells]
        spread.append(f"Skala {rho_text(f)}: {min(its)} bis {'≥ ' if not all(x['converged'] for x in cells) else ''}{max(its)}")
    return "info", (f"Skala der Wolke {scales}, Start ρ = {rho_text(s.rho)}: " + "; ".join(parts) + f" (Deckel {cap} Schritte). "
                    f"Der Ausgleich holt Läufe zurück, die mit festem ρ nicht gelöst werden, aber nicht gleichmäßig: Schritte der neun Einstellungen (Intervall × Schwelle) auf dem Lorentz-Programm bei Start ρ = {rho_text(s.rho)}, " + "; ".join(spread) + ".")


def verdict_relax(relax: list, warm: list) -> tuple:
    parts = []
    helps = False
    for r in relax:
        its = {x["alpha"]: x["iterations"] for x in r["results"]}
        helps = helps or min(its.values()) < its[1.0]
        parts.append(f"{r['name'].split(' (')[0]}: " + ", ".join(f"α = {rho_text(a)}: {n}" for a, n in its.items()))
    relax_text = "Überrelaxation (bestes ρ der kleinen Reihe je α): " + "; ".join(parts) + (". Sie hilft bei einem Kegel." if helps else ". Sie hilft hier nicht: je größer α, desto mehr Schritte.")
    save_x = [100.0 * (1.0 - r["x_only"] / r["cold"]) for r in warm]
    save_xu = [100.0 * (1.0 - r["both"] / r["cold"]) for r in warm]
    w = ", ".join(f"{_short(r['name'])}: kalt {r['cold']}, mit x allein {r['x_only']}, mit x und u {r['both']} (grober Lauf davor {r['rough']})" for r in warm)
    total = ", ".join(f"{r['rough']} + {r['both']} = {r['rough'] + r['both']} gegen {r['cold']}" for r in warm)
    return "info", (f"{relax_text} Warmstart bis 10⁻⁶ (ρ = {rho_text(C.WARM_RHO)}): {w}. Ein Start im groben x allein spart {de(min(save_x), 0)} % bis {de(max(save_x), 0)} % der kalten Schritte, mit x und u {de(min(save_xu), 0)} % bis {de(max(save_xu), 0)} %; "
                    f"der grobe Lauf und der Warmstart zusammen brauchen aber so viele Schritte wie der kalte Lauf ({total}): der Zustand des Verfahrens ist (s, u), und mit u setzt der Lauf nur fort, wo der grobe aufgehört hat.")


def verdict_versus(rows: list) -> tuple:
    lines = []
    for r in rows:
        mid = next(x for x in r["results"] if x["target"] == 1e-6)
        lines.append(f"{r['name'].split(' (')[0]}: Barriere {mid['barrier_steps']} Newton-Schritte / {ms_text(mid['barrier_seconds'])} ms, ADMM {mid['admm_iterations']} Schritte / {ms_text(mid['admm_seconds'])} ms")
    cheap = [r for r in rows if r["kind"] in ("q", "s")]
    ratios = [x["admm_seconds"] / x["barrier_seconds"] for r in cheap for x in r["results"]]
    coarse = [x["admm_seconds"] / x["barrier_seconds"] for r in cheap for x in r["results"] if x["target"] == C.TARGETS[0]]
    step = lambda key, digits: ", ".join(f"{_short(r['name'])} {de(r[key], digits)}" for r in rows)
    return "info", (f"Bei Wertfehler 10⁻⁶ (Zeiten nach Warmlauf): " + "; ".join(lines) + f". Ein ADMM-Schritt kostet in ms: {step('step_ms_admm', 2)}; ein Newton-Schritt der Barriere: {step('step_ms_barrier', 2)}. "
                    f"Für Lorentz- und PSD-Kegel ist der ADMM-Schritt billiger, aber es sind mehr Schritte: ADMM braucht {de(min(ratios), 1)}- bis {de(max(ratios), 1)}-mal die Zeit der Barriere, bei Wertfehler 10⁻³ {de(min(coarse), 1)}- bis {de(max(coarse), 1)}-mal. "
                    f"Beim Potenz- und Exponentialkegel ist schon der einzelne ADMM-Schritt teurer, weil ein Aufruf der Projektion (Wurzelsuche in numpy) bei diesen wenigen Kegeln ein bis zwei Millisekunden braucht. "
                    f"Auf Programmen mit 3 bis 35 Variablen liegt die Barriere bei hoher Genauigkeit vorn; bei grobem Ziel (10⁻³) liegen beide bei Lorentz und PSD nahe beieinander (Verhältnis oben); über große Programme sagt dieser Versuch nichts.")
