"""Barrierenmethode über Orthant, Exponentialkegel und Potenzkegel (gestapelt gerechnet), mit relativer Genauigkeit; aus kegel-exponential-demo (Stück 7) übernommen, nur für den Referenzwert R* der Kugeln.

Programm:  min c^T x   mit   s = b - A x in K,   K = Produkt von Gruppen
    ('l', None, m)   Orthant mit m Zeilen
    ('e', None, N)   N Exponentialkegel (je drei Zeilen, Kegel für Kegel)
    ('p', a,    N)   N Potenzkegel P_a (je drei Zeilen)
Der Kern der Reihe (kegel_kern.py) kennt diese beiden Kegel nicht und bleibt unverändert; das Verfahren ist dasselbe wie dort: Newton mit Rücksetzen
auf f_t = t c^T x + phi(b - A x), dann t <- mu t, bis nu / t unter der Toleranz liegt. Die Toleranz ist **relativ** zum Zielwert des Starts (der Maßstab der Daten
geht ein): bei 1000-fach größeren Daten erreicht eine absolute Schranke 1e-9 die Rundungsgrenze."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

import kad_cones as C


@dataclass(frozen=True, eq=False)
class Program:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    groups: tuple            # ((art, par, anzahl), ...) in der Zeilenreihenfolge von A
    x0: np.ndarray           # streng zulässiger Start

    @property
    def nu(self) -> int:
        return nu_of(self.groups)

    @property
    def rows(self) -> int:
        return rows_of(self.groups)


def nu_of(groups) -> int:
    return int(sum(n if k == "l" else C.NU * n for k, _, n in groups))


def rows_of(groups) -> int:
    return int(sum(n if k == "l" else 3 * n for k, _, n in groups))


def slack(prob: Program, x: np.ndarray) -> np.ndarray:
    return prob.b - prob.A @ x


def barrier(prob: Program, s: np.ndarray, hess: bool = True):
    """(Wert, Gradient nach s, Liste der Hesse-Blöcke je Gruppe); Wert inf, wenn s nicht streng im Inneren liegt."""
    val, grads, hs, off = 0.0, [], [], 0
    for kind, par, n in prob.groups:
        d = n if kind == "l" else 3 * n
        blk = s[off:off + d]
        if kind == "l":
            if np.any(blk <= 0):
                return math.inf, None, None
            val -= float(np.sum(np.log(blk)))
            grads.append(-1.0 / blk)
            hs.append(1.0 / blk ** 2)
        else:
            S = blk.reshape(n, 3)
            v, g, H = C.exp_grad_hess(S) if kind == "e" else C.pow_grad_hess(S, par)
            if g is None:
                return math.inf, None, None
            val += v
            grads.append(g.reshape(-1))
            hs.append(H)
        off += d
    return val, np.concatenate(grads), hs


def newton_matrix(prob: Program, hs) -> np.ndarray:
    """A^T H A mit blockdiagonaler Hesse-Matrix H, gruppenweise gestapelt."""
    nv = prob.A.shape[1]
    M = np.zeros((nv, nv))
    off = 0
    for (kind, par, n), H in zip(prob.groups, hs):
        if kind == "l":
            Ag = prob.A[off:off + n]
            M += Ag.T @ (H[:, None] * Ag)
            off += n
        else:
            Ak = prob.A[off:off + 3 * n].reshape(n, 3, nv)
            M += np.einsum("nia,nij,njb->ab", Ak, H, Ak, optimize=True)
            off += 3 * n
    return M


def _half_decrement(prob: Program, x: np.ndarray, t: float) -> float:
    s = slack(prob, x)
    _, g, hs = barrier(prob, s)
    if g is None:
        return math.inf
    grad = t * prob.c - prob.A.T @ g
    try:
        dx = -np.linalg.solve(newton_matrix(prob, hs), grad)
    except np.linalg.LinAlgError:
        return math.inf
    return -float(grad @ dx) / 2.0


def center(prob: Program, x: np.ndarray, t: float, tol: float = 1e-6, max_steps: int = 100):
    """Zentrieren für festes t: gedämpftes Newton-Verfahren. Rückgabe (x, Schritte, halbes Dekrement-Quadrat am Ende, Schrittlängen)."""
    steps, lengths, lam2 = 0, [], math.inf
    for _ in range(max_steps):
        s = slack(prob, x)
        val, g, hs = barrier(prob, s)
        if g is None:
            return x, steps, math.inf, lengths
        grad = t * prob.c - prob.A.T @ g
        try:
            dx = -np.linalg.solve(newton_matrix(prob, hs), grad)
        except np.linalg.LinAlgError:
            return x, steps, math.inf, lengths
        slope = -float(grad @ dx)                       # 2 * lambda^2
        lam2 = slope / 2.0
        if lam2 < tol:
            return x, steps, lam2, lengths
        f0 = t * float(prob.c @ x) + val
        a = 1.0
        while True:
            xn = x + a * dx
            vn, gn, _ = barrier(prob, slack(prob, xn), hess=False)
            if gn is not None and t * float(prob.c @ xn) + vn <= f0 - 0.25 * a * slope:
                break
            a *= 0.5
            if a < 1e-14:
                return x, steps, lam2, lengths
        decrease = f0 - (t * float(prob.c @ xn) + vn)
        x = xn
        steps += 1
        lengths.append(a)
        if decrease <= 1e-14 * max(1.0, abs(f0)) and lam2 < 1e-3:         # Rundungsstopp: ein Schritt senkt f_t nicht mehr messbar; das Dekrement am neuen Punkt entscheidet, ob er als zentriert gilt
            return x, steps, _half_decrement(prob, x, t), lengths
    return x, steps, lam2, lengths


@dataclass(frozen=True, eq=False)
class Result:
    x: np.ndarray
    value: float             # c^T x
    gap_bound: float         # nu / t am Ende
    newton_steps: int
    rounds: int
    t: float
    lam2: float              # halbes Dekrement-Quadrat der letzten Zentrierung
    centered: bool           # die letzte Zentrierung erreichte ihre Toleranz
    history: list = field(default_factory=list)   # je Runde (t, Newton-Schritte bisher, Lückenschranke, Zielwert)


def initial_t(prob: Program) -> float:
    """Startwert für t: minimiert die Norm des Gradienten von f_t im Start (so liegt x0 schon nahe am zentralen Punkt dieses t); sonst nu / |c^T x0|."""
    _, g, _ = barrier(prob, slack(prob, prob.x0), hess=False)
    cc = float(prob.c @ prob.c)
    if g is not None and cc > 0.0:
        t = float((prob.c @ (prob.A.T @ g)) / cc)          # t c = A^T grad phi  ->  Kleinste-Quadrate-Lösung
        if t > 0.0 and math.isfinite(t):
            return t
    return prob.nu / max(abs(float(prob.c @ prob.x0)), 1.0)


def solve(prob: Program, tol: float = 1e-9, mu: float = 8.0, t0: float | None = None, max_rounds: int = 80) -> Result:
    """Barrierenmethode bis nu / t <= tol * |c^T x0| (relative Genauigkeit zum Zielwert des Starts). BLAS läuft mit einem Thread: die Matrizen sind klein."""
    with threadpool_limits(1):
        return _solve(prob, tol, mu, t0, max_rounds)


def _solve(prob: Program, tol: float, mu: float, t0, max_rounds: int) -> Result:
    x = prob.x0.copy()
    if barrier(prob, slack(prob, x), hess=False)[1] is None:
        raise ValueError("der Startpunkt x0 liegt nicht streng im Inneren des Kegels")
    scale0 = abs(float(prob.c @ prob.x0))
    t = t0 or initial_t(prob)
    newton, rounds, hist, lam2 = 0, 0, [], math.inf
    while True:
        x, n, lam2, _ = center(prob, x, t)
        newton += n
        rounds += 1
        hist.append((t, newton, prob.nu / t, float(prob.c @ x)))
        if prob.nu / t <= tol * max(scale0, abs(float(prob.c @ x)), 1e-300) or rounds >= max_rounds:
            break
        t *= mu
    return Result(x, float(prob.c @ x), prob.nu / t, newton, rounds, t, lam2, lam2 < 1e-6, hist)
