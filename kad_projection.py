"""Projektionen auf die fünf Kegel der Reihe (gestapelt gerechnet) und ihr Zertifikat nach Moreau.

  Orthant        max(v, 0)
  Lorentz        (t, v̄): v ∈ K → v;  −v ∈ K → 0;  sonst ((‖v̄‖ + t)/2) (1, v̄/‖v̄‖)
  PSD            Eigenzerlegung, negative Eigenwerte auf 0 (svec-Koordinaten mit √2 auf der Außendiagonale wie im Kern)
  Exponential    drinnen → v; im Polarkegel → 0; sonst auf dem Rand: v = y u(ρ) + λ n(ρ) mit u = (ρ, 1, e^ρ), n = (e^ρ, e^ρ (1 − ρ), −1), u ⟂ n;
                 die Bedingung ⟨v, u × n⟩ = 0 ist eine Gleichung in ρ (Vorzeichenwechsel auf einem Gitter, Bisektion); dazu die Fläche y = 0
  Potenz P_a     dasselbe mit u = (ρ, 1, ρ^a), n = −∇(u^a v^(1−a) − w) und den Strahlen (u, 0, 0), (0, v, 0)
Moreau: v = P_K(v) + P_{K°}(v) mit dem Polarkegel K° = −K*, ⟨P_K(v), P_{K°}(v)⟩ = 0. **Umgekehrt ist jedes p mit p ∈ K, v − p ∈ K°, ⟨p, v − p⟩ = 0 die Projektion**: die
Probe `certificate` prüft eine Projektion ohne Referenzlöser. Die Formel P_K(v) − P_K(−v) = v gilt nur für selbstduale Kegel (K* = K im gegebenen Skalarprodukt)."""

from __future__ import annotations

import math

import numpy as np

KINDS = ("l", "q", "s", "e", "p")
KIND_NAMES = {"l": "Orthant", "q": "Lorentz-Kegel", "s": "PSD-Kegel (2×2)", "e": "Exponentialkegel", "p": "Potenzkegel"}
SELF_DUAL = {"l": True, "q": True, "s": True, "e": False, "p": False}


# ----------------------------------------------------------------------------- Orthant, Lorentz, PSD

def proj_orthant(V: np.ndarray) -> np.ndarray:
    return np.maximum(V, 0.0)


def proj_lorentz(V: np.ndarray) -> np.ndarray:
    """V (N, m) mit V[:, 0] = t."""
    t, vb = V[:, 0], V[:, 1:]
    r = np.linalg.norm(vb, axis=1)
    out = np.zeros_like(V)
    inside = r <= t
    out[inside] = V[inside]
    mid = (~inside) & (r > -t)
    a = (r[mid] + t[mid]) / 2.0
    out[mid, 0] = a
    out[mid, 1:] = a[:, None] * vb[mid] / r[mid][:, None]
    return out


_IDX: dict = {}


def _index(k: int):
    if k not in _IDX:
        iu = np.triu_indices(k)
        _IDX[k] = (iu, np.where(iu[0] != iu[1], math.sqrt(2.0), 1.0))
    return _IDX[k]


def smat_stack(S: np.ndarray, k: int) -> np.ndarray:
    iu, f = _index(k)
    M = np.zeros((len(S), k, k))
    M[:, iu[0], iu[1]] = S / f
    return M + np.swapaxes(M, 1, 2) - np.einsum("nij,ij->nij", M, np.eye(k))


def svec_stack(M: np.ndarray) -> np.ndarray:
    iu, f = _index(M.shape[1])
    return M[:, iu[0], iu[1]] * f


def proj_psd(S: np.ndarray, k: int) -> np.ndarray:
    """S (N, k(k+1)/2) in svec-Koordinaten."""
    w, Q = np.linalg.eigh(smat_stack(S, k))
    P = np.einsum("nij,nj,nkj->nik", Q, np.maximum(w, 0.0), Q)
    return svec_stack((P + np.swapaxes(P, 1, 2)) / 2.0)


# ----------------------------------------------------------------------------- Zugehörigkeit (mit Toleranz relativ zur Größe des Punktes)

def _tol(V: np.ndarray, tol: float) -> np.ndarray:
    return tol * np.maximum(1.0, np.linalg.norm(V, axis=1))


def exp_member(V: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """In K_exp = cl{y e^(x/y) <= z, y > 0}: y > 0 mit y log(z/y) >= x, oder die Fläche y = 0 mit x <= 0, z >= 0 (Toleranz tol · max(1, ‖v‖))."""
    t = _tol(V, tol)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    with np.errstate(all="ignore"):
        inside = (y > 0) & (z > 0) & (y * (np.log(np.maximum(z, 1e-300)) - np.log(np.maximum(y, 1e-300))) >= x - t)
    face = (np.abs(y) <= t) & (x <= t) & (z >= -t)
    return inside | face


def exp_dual_member(U: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """In K_exp* = {u < 0, −u e^(v/u) <= e w} (Abschluss: u = 0, v, w >= 0); die Bedingung wird mit e^(v/u) < 10^300 und der Toleranz tol · max(1, ‖z‖) auf w geprüft."""
    t = _tol(U, tol)
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    with np.errstate(all="ignore"):
        lhs = -u * np.exp(np.minimum(v / np.minimum(u, -1e-300), 690.0))
        inside = (u < 0) & (lhs <= math.e * w + t)
    closure = (np.abs(u) <= t) & (v >= -t) & (w >= -t)
    return inside | closure


def pow_member(V: np.ndarray, a: float, tol: float = 0.0) -> np.ndarray:
    t = _tol(V, tol)
    u, v, w = V[:, 0], V[:, 1], V[:, 2]
    with np.errstate(all="ignore"):
        return (u >= -t) & (v >= -t) & (np.maximum(u, 0) ** a * np.maximum(v, 0) ** (1 - a) >= np.abs(w) - t)


def pow_dual_member(U: np.ndarray, a: float, tol: float = 0.0) -> np.ndarray:
    t = _tol(U, tol)
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    with np.errstate(all="ignore"):
        return (u >= -t) & (v >= -t) & ((np.maximum(u, 0) / a) ** a * (np.maximum(v, 0) / (1 - a)) ** (1 - a) >= np.abs(w) - t)


def in_cone(kind: str, par, V: np.ndarray, tol: float = 0.0) -> np.ndarray:
    t = _tol(V, tol)
    if kind == "l":
        return np.all(V >= -t[:, None], axis=1)
    if kind == "q":
        return np.linalg.norm(V[:, 1:], axis=1) <= V[:, 0] + t
    if kind == "s":
        return np.linalg.eigvalsh(smat_stack(V, par)).min(axis=1) >= -t
    if kind == "e":
        return exp_member(V, tol)
    return pow_member(V, par, tol)


def in_polar(kind: str, par, R: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """R im Polarkegel K° = −K*, also −R in K*. Orthant, Lorentz und PSD sind selbstdual: −R in K."""
    if kind in ("l", "q", "s"):
        return in_cone(kind, par, -R, tol)
    return exp_dual_member(-R, tol) if kind == "e" else pow_dual_member(-R, par, tol)


# ----------------------------------------------------------------------------- Exponentialkegel

def _rho_grid() -> np.ndarray:
    pos = np.exp(np.linspace(math.log(1e-4), math.log(700.0), 240))
    return np.concatenate([-pos[::-1], [0.0], pos])


_RHO = _rho_grid()
_CACHE: dict = {}


def _exp_cross(rho: np.ndarray) -> np.ndarray:
    """u × n für u = (ρ, 1, e^ρ), n = (e^ρ, e^ρ (1 − ρ), −1), für ρ > 0 durch e^(2ρ) geteilt (sonst läuft e^(2ρ) über; nur das Vorzeichen der Wurzel zählt)."""
    r = np.asarray(rho, float)
    w = np.empty(r.shape + (3,))
    pos = r > 0
    e1 = np.exp(np.where(pos, -r, r))
    e2 = e1 * e1
    w[..., 0] = np.where(pos, -e2 - (1 - r), -1 - e2 * (1 - r))
    w[..., 1] = np.where(pos, 1 + r * e2, e2 + r)
    w[..., 2] = r * e1 * (1 - r) - e1
    return w


def _exp_ray(r: np.ndarray) -> np.ndarray:
    """Der Randstrahl u(ρ) = (ρ, 1, e^ρ); für ρ > 30 dieselbe Gerade als (ρ e^(−ρ), e^(−ρ), 1)."""
    big = r > 30.0
    u = np.stack([r, np.ones_like(r), np.exp(np.minimum(r, 30.0))], axis=1)
    eb = np.exp(-np.where(big, r, 0.0))
    ub = np.stack([r * eb, eb, np.ones_like(r)], axis=1)
    return np.where(big[:, None], ub, u)


def _pick_best(rows: np.ndarray, dist: np.ndarray, cand: np.ndarray, best: np.ndarray, bestd: np.ndarray) -> None:
    """Je Zeile den Kandidaten mit dem kleinsten Abstand übernehmen, wenn er besser ist als der bisherige."""
    order = np.lexsort((dist, rows))
    first = np.unique(rows[order], return_index=True)[1]
    sel = order[first]
    r_sel = rows[sel]
    better = dist[sel] < bestd[r_sel]
    best[r_sel[better]] = cand[sel[better]]
    bestd[r_sel[better]] = dist[sel[better]]


def _bisect(fun, a: np.ndarray, b: np.ndarray, iters: int) -> np.ndarray:
    """Nullstelle in [a, b] mit Vorzeichenwechsel: Illinois-Verfahren (regula falsi mit Halbierung des festgehaltenen Funktionswerts), gestapelt; am Ende der Punkt mit dem kleineren |f|.
    Fällt das Intervall nicht schnell genug, halbiert ein Bisektionsschritt je fünf Schritte."""
    fa, fb = fun(a), fun(b)
    side = np.zeros(len(a), dtype=int)
    for it in range(iters):
        if it % 5 == 4:
            c = (a + b) / 2.0
        else:
            denom = fb - fa
            c = np.where(denom != 0, (a * fb - b * fa) / np.where(denom != 0, denom, 1.0), (a + b) / 2.0)
            c = np.clip(c, np.minimum(a, b), np.maximum(a, b))
        fc = fun(c)
        same_a = np.sign(fc) == np.sign(fa)
        a_new = np.where(same_a, c, a)
        fa_new = np.where(same_a, fc, fa)
        b_new = np.where(same_a, b, c)
        fb_new = np.where(same_a, fb, fc)
        # Illinois: der festgehaltene Rand halbiert seinen Funktionswert, wenn derselbe Rand zweimal hintereinander bleibt
        fb_new = np.where(same_a & (side == 1), fb_new / 2.0, fb_new)
        fa_new = np.where((~same_a) & (side == -1), fa_new / 2.0, fa_new)
        side = np.where(same_a, 1, -1)
        a, fa, b, fb = a_new, fa_new, b_new, fb_new
    return np.where(np.abs(fa) <= np.abs(fb), a, b)


def proj_exp(V: np.ndarray, iters: int = 12) -> np.ndarray:
    """Projektion auf den Exponentialkegel über die Wurzel der Gleichung <v, u × n> = 0 (siehe Modul-Docstring)."""
    out = np.zeros_like(V)
    inK = exp_member(V)
    polar = exp_dual_member(-V) & ~inK
    out[inK] = V[inK]
    rest = np.flatnonzero(~(inK | polar))
    if len(rest) == 0:
        return out
    W = V[rest]
    best = np.stack([np.minimum(W[:, 0], 0.0), np.zeros(len(W)), np.maximum(W[:, 2], 0.0)], axis=1)          # Fläche y = 0
    bestd = np.linalg.norm(W - best, axis=1)
    if "exp" not in _CACHE:
        _CACHE["exp"] = _exp_cross(_RHO).T.copy()
    H = W @ _CACHE["exp"]
    chg = np.argwhere((np.sign(H[:, :-1]) * np.sign(H[:, 1:])) < 0)
    if len(chg):
        row, j = chg[:, 0], chg[:, 1]
        Vr = W[row]
        r = _bisect(lambda x: np.einsum("ni,ni->n", Vr, _exp_cross(x)), _RHO[j].copy(), _RHO[j + 1].copy(), iters)
        u = _exp_ray(r)
        y = np.einsum("ni,ni->n", Vr, u) / np.einsum("ni,ni->n", u, u)
        cand = np.maximum(y, 0.0)[:, None] * u
        dist = np.where(y > 0, np.linalg.norm(Vr - cand, axis=1), np.inf)
        _pick_best(row, dist, cand, best, bestd)
    out[rest] = best
    return out


# ----------------------------------------------------------------------------- Potenzkegel

TAU = np.linspace(-250.0, 250.0, 1001)


def _pow_cross(tau: np.ndarray, a: float) -> np.ndarray:
    """u × n für u = (ρ, 1, ρ^a), n = −∇(u^a v^(1−a) − w) = (−a ρ^(a−1), −(1−a) ρ^a, 1) (w >= 0), ρ = e^τ; auf Länge ~1 skaliert (nur das Vorzeichen der Wurzel zählt)."""
    rho = np.exp(tau)
    c = np.stack([1.0 + (1 - a) * rho ** (2 * a), -(a * rho ** (2 * a - 1) + rho), -(1 - a) * rho ** (a + 1) + a * rho ** (a - 1)], axis=-1)
    return c / np.max(np.abs(c), axis=-1, keepdims=True)


def proj_pow(V: np.ndarray, a: float, iters: int = 12) -> np.ndarray:
    out = np.zeros_like(V)
    inK = pow_member(V, a)
    polar = pow_dual_member(-V, a) & ~inK
    out[inK] = V[inK]
    rest = np.flatnonzero(~(inK | polar))
    if len(rest) == 0:
        return out
    W = V[rest].copy()
    sg = np.where(W[:, 2] >= 0, 1.0, -1.0)
    W[:, 2] = np.abs(W[:, 2])                       # Spiegelung w -> |w|: der Kegel ist symmetrisch in w
    cand0 = np.stack([np.maximum(W[:, 0], 0.0), np.zeros(len(W)), np.zeros(len(W))], axis=1)
    cand1 = np.stack([np.zeros(len(W)), np.maximum(W[:, 1], 0.0), np.zeros(len(W))], axis=1)
    d0, d1 = np.linalg.norm(W - cand0, axis=1), np.linalg.norm(W - cand1, axis=1)
    best = np.where((d0 <= d1)[:, None], cand0, cand1)
    bestd = np.minimum(d0, d1)
    if ("pow", a) not in _CACHE:
        _CACHE[("pow", a)] = _pow_cross(TAU, a).T.copy()
    H = W @ _CACHE[("pow", a)]
    chg = np.argwhere((np.sign(H[:, :-1]) * np.sign(H[:, 1:])) < 0)
    if len(chg):
        row, j = chg[:, 0], chg[:, 1]
        Wr = W[row]
        tau = _bisect(lambda x: np.einsum("ni,ni->n", Wr, _pow_cross(x, a)), TAU[j].copy(), TAU[j + 1].copy(), iters)
        rho = np.exp(tau)
        u = np.stack([rho, np.ones_like(rho), rho ** a], axis=1)
        y = np.einsum("ni,ni->n", Wr, u) / np.einsum("ni,ni->n", u, u)
        cand = np.maximum(y, 0.0)[:, None] * u
        dist = np.where(y > 0, np.linalg.norm(Wr - cand, axis=1), np.inf)
        _pick_best(row, dist, cand, best, bestd)
    best[:, 2] *= sg
    out[rest] = best
    return out


# ----------------------------------------------------------------------------- Auswahl und Zertifikat

def project(kind: str, par, V: np.ndarray) -> np.ndarray:
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if kind == "l":
        return proj_orthant(V)
    if kind == "q":
        return proj_lorentz(V)
    if kind == "s":
        return proj_psd(V, par)
    if kind == "e":
        return proj_exp(V)
    if kind == "p":
        return proj_pow(V, par)
    raise ValueError(f"unbekannter Kegel {kind!r}")


def certificate(kind: str, par, V: np.ndarray, P: np.ndarray, tol: float = 1e-7) -> dict:
    """Die Moreau-Probe einer Projektion P von V: P liegt in K, der Rest v − p liegt im Polarkegel, und ⟨p, v − p⟩ ≈ 0 (Toleranz relativ zu ‖v‖²). ok je Zeile."""
    R = V - P
    scale = np.maximum(1.0, np.linalg.norm(V, axis=1))
    inner = np.einsum("ni,ni->n", P, R)
    k_ok = in_cone(kind, par, P, tol)
    p_ok = in_polar(kind, par, R, tol)
    o_ok = np.abs(inner) <= tol * scale ** 2
    return {"ok": k_ok & p_ok & o_ok, "in_cone": k_ok, "in_polar": p_ok, "orthogonal": o_ok, "inner": inner}
