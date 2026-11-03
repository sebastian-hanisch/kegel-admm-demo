"""ADMM für Kegelprogramme  min cᵀx  mit  s = b − A x ∈ K  (skalierte Form, Penalty ρ, Dualvariable u = y/ρ).

    x⁺ = argmin cᵀx + (ρ/2) ‖A x + s − b + u‖²     = −(AᵀA)⁻¹ (c/ρ + Aᵀ(s − b + u))        (AᵀA unabhängig von ρ: einmal invertiert)
    s⁺ = P_K(b − Â − u)                                mit Überrelaxation  Â = α A x + (1 − α)(b − s)      (P_K aus kad_projection)
    u⁺ = u + Â + s⁺ − b
Residuen:  primal  r = ‖A x + s⁺ − b‖,  dual  d = ρ ‖Aᵀ(s⁺ − s)‖.  Abbruch (relativ):  r ≤ tol · max(‖A x‖, ‖s‖, ‖b‖, 1)  und  d ≤ tol · max(ρ ‖Aᵀu‖, ‖c‖, 1).
Residuenausgleich (Boyd u. a. 2011, übliche Wahl μ = 10, τ = 2): alle `every` Schritte ρ ← τ ρ, wenn r > μ d, und ρ ← ρ/τ, wenn d > μ r; u wird mit umgerechnet, damit y = ρ u gleich bleibt.
Da AᵀA nicht von ρ abhängt, kostet ein Wechsel von ρ hier nichts (anders bei OSQP und SCS, wo die KKT-Matrix ρ enthält).
Das ist **kein** SCS: keine homogene selbstduale Einbettung, also keine Zertifikate für unzulässige oder unbeschränkte Programme (der Lauf endet dann am Schrittdeckel).
Der Kegel kommt als `spec` = [(Art, Parameter, Anzahl)] mit Art 'l' (Orthant, Anzahl Zeilen), 'q' (Lorentz, Parameter Größe), 's' (PSD, Parameter k), 'e' (Exponential), 'p' (Potenz, Parameter a)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import kad_projection as P


def spec_of(prob) -> list:
    """Die Kegelzerlegung eines Programms des Kerns (ConeProblem, `cones`) oder der Barriere (Program, `groups`)."""
    if hasattr(prob, "groups"):
        return list(prob.groups)
    spec = []
    for kind, m in prob.cones:
        if kind == "l":                                      # Orthant: m Zeilen, Parameter unbenutzt (project_spec liest die Anzahl als dritte Stelle)
            spec.append(("l", None, m))
        elif spec and spec[-1][0] == kind and spec[-1][1] == m:
            spec[-1] = (kind, m, spec[-1][2] + 1)
        else:
            spec.append((kind, m, 1))
    return spec


def project_spec(spec: list, s: np.ndarray) -> np.ndarray:
    out = np.empty_like(s)
    off = 0
    for kind, par, n in spec:
        if kind == "l":
            d = n
            out[off:off + d] = np.maximum(s[off:off + d], 0.0)
        elif kind == "q":
            d = par * n
            out[off:off + d] = P.proj_lorentz(s[off:off + d].reshape(n, par)).reshape(-1)
        elif kind == "s":
            d = par * (par + 1) // 2 * n
            out[off:off + d] = P.proj_psd(s[off:off + d].reshape(n, par * (par + 1) // 2), par).reshape(-1)
        elif kind == "e":
            d = 3 * n
            out[off:off + d] = P.proj_exp(s[off:off + d].reshape(n, 3)).reshape(-1)
        else:
            d = 3 * n
            out[off:off + d] = P.proj_pow(s[off:off + d].reshape(n, 3), par).reshape(-1)
        off += d
    return out


@dataclass(frozen=True, eq=False)
class AdmmResult:
    iterations: int
    converged: bool
    x: np.ndarray
    s: np.ndarray
    u: np.ndarray                # skalierte Dualvariable; die Dualvariable ist y = ρ u
    value: float                 # cᵀx am Ende
    primal: float                # r am Ende
    dual: float                  # d am Ende
    rho: float                   # ρ am Ende
    rho_changes: int
    seconds: float
    history: dict = field(default_factory=dict)   # je Schritt: "primal", "dual", "value", "rho" (Listen)


def admm(prob, rho: float = 1.0, alpha: float = 1.0, max_iter: int = 5000, tol: float = 1e-6, adaptive: bool = False, mu: float = 10.0, tau: float = 2.0, every: int = 25,
         x0=None, u0=None, ref_value: float | None = None) -> AdmmResult:
    """ADMM für `min cᵀx, b − A x ∈ K`. Start: x = x0 (Standard 0), s = P_K(b − A x), u = u0 (Standard 0).
    Mit `ref_value` (dem bekannten Optimalwert) endet der Lauf stattdessen, sobald |cᵀx − ref_value| ≤ tol |ref_value| und r ≤ tol (1 + ‖b‖): ein Maßstab für Vergleiche, kein Abbruchkriterium der Praxis."""
    A, b, c, spec = prob.A, prob.b, prob.c, spec_of(prob)
    if rho <= 0 or not np.isfinite(rho):
        raise ValueError("rho muss positiv sein")
    M = A.T @ A
    if np.linalg.matrix_rank(M) < M.shape[0]:
        raise ValueError("A hat keinen vollen Spaltenrang: AᵀA ist singulär")
    Minv = np.linalg.inv(M)
    nb, nc = float(np.linalg.norm(b)), float(np.linalg.norm(c))
    x = np.zeros(A.shape[1]) if x0 is None else np.array(x0, dtype=float)
    s = project_spec(spec, b - A @ x)
    u = np.zeros_like(b) if u0 is None else np.array(u0, dtype=float)
    hist = {"primal": [], "dual": [], "value": [], "rho": []}
    changes = 0
    t0 = time.perf_counter()
    r = d = float("inf")
    for it in range(1, max_iter + 1):
        x = -Minv @ (c / rho + A.T @ (s - b + u))
        Ax = A @ x
        Ax_rel = alpha * Ax + (1.0 - alpha) * (b - s)
        s_new = project_spec(spec, b - Ax_rel - u)
        u = u + Ax_rel + s_new - b
        r = float(np.linalg.norm(Ax + s_new - b))
        d = float(rho * np.linalg.norm(A.T @ (s_new - s)))
        s = s_new
        value = float(c @ x)
        hist["primal"].append(r)
        hist["dual"].append(d)
        hist["value"].append(value)
        hist["rho"].append(rho)
        if ref_value is not None:
            done = abs(value - ref_value) <= tol * abs(ref_value) and r <= tol * (1.0 + nb)
        else:
            done = r <= tol * max(float(np.linalg.norm(Ax)), float(np.linalg.norm(s)), nb, 1.0) and d <= tol * max(rho * float(np.linalg.norm(A.T @ u)), nc, 1.0)
        if done:
            return AdmmResult(it, True, x, s, u, value, r, d, rho, changes, time.perf_counter() - t0, hist)
        if adaptive and it % every == 0:
            if r > mu * d:
                rho *= tau
                u = u / tau
                changes += 1
            elif d > mu * r:
                rho /= tau
                u = u * tau
                changes += 1
    return AdmmResult(max_iter, False, x, s, u, float(c @ x), r, d, rho, changes, time.perf_counter() - t0, hist)
