"""Barrieren des Exponential- und des Potenzkegels (je drei Koordinaten) und der Dualkegel; aus kegel-exponential-demo (Stück 7) übernommen, ohne die Automorphismen-Probe.

Exponentialkegel   K_exp = cl{ (x, y, z): y > 0, y exp(x / y) <= z }
                   Barriere  phi = -log(y log(z / y) - x) - log y - log z                         nu = 3
                   Dualkegel K_exp* = { (u, v, w): u < 0, -u exp(v / u) <= e w }  (Abschluss mit u = 0, v, w >= 0)
Potenzkegel P_a    P_a = { (u, v, w): u, v >= 0, u^a v^(1-a) >= |w| },  0 < a < 1
                   Barriere  phi = -log(u^(2a) v^(2-2a) - w^2) - (1 - a) log u - a log v           nu = 3
                   P_(1/2) ist ein gedrehter Lorentz-Kegel (u v >= w^2).

Alle Funktionen rechnen **gestapelt**: eine Matrix S der Form (N, 3) ist N Kegel. Gradient und Hesse sind in den Koordinaten (s0, s1, s2) des Kegels.
Liegt irgendein Punkt nicht im Inneren, ist der Wert `inf` und Gradient und Hesse sind `None` (die Linien-Suche der Barrierenmethode verlangt das)."""

from __future__ import annotations

import math

import numpy as np

NU = 3          # Barriere-Parameter beider Kegel


def exp_inside(S: np.ndarray) -> np.ndarray:
    """Streng im Inneren: y > 0, z > 0 und y log(z/y) - x > 0 (ohne exp(x/y), das überlaufen kann)."""
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    ok = (y > 0) & (z > 0)
    with np.errstate(all="ignore"):
        psi = y * (np.log(z) - np.log(y)) - x
    return ok & np.isfinite(psi) & (psi > 0)


def pow_inside(S: np.ndarray, a: float) -> np.ndarray:
    u, v, w = S[:, 0], S[:, 1], S[:, 2]
    ok = (u > 0) & (v > 0)
    with np.errstate(all="ignore"):
        phi = u ** (2 * a) * v ** (2 - 2 * a) - w * w
    return ok & np.isfinite(phi) & (phi > 0)


def exp_grad_hess(S: np.ndarray):
    """(Wert je Kegel, Gradient (N, 3), Hesse (N, 3, 3)) der Barriere des Exponentialkegels; None bei einem Punkt außerhalb des Inneren."""
    if not exp_inside(S).all():
        return math.inf, None, None
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    ly = np.log(y)
    lz = np.log(z)
    psi = y * (lz - ly) - x
    g_psi = np.stack([-np.ones_like(x), lz - ly - 1.0, y / z], axis=1)
    H_psi = np.zeros((len(x), 3, 3))
    H_psi[:, 1, 1] = -1.0 / y
    H_psi[:, 1, 2] = H_psi[:, 2, 1] = 1.0 / z
    H_psi[:, 2, 2] = -y / z ** 2
    val = float(np.sum(-np.log(psi) - ly - lz))
    g = -g_psi / psi[:, None] - np.stack([np.zeros_like(x), 1.0 / y, 1.0 / z], axis=1)
    H = np.einsum("ni,nj->nij", g_psi, g_psi) / (psi ** 2)[:, None, None] - H_psi / psi[:, None, None]
    H[:, 1, 1] += 1.0 / y ** 2
    H[:, 2, 2] += 1.0 / z ** 2
    return val, g, H


def pow_grad_hess(S: np.ndarray, a: float):
    """Dasselbe für den Potenzkegel P_a."""
    if not pow_inside(S, a).all():
        return math.inf, None, None
    u, v, w = S[:, 0], S[:, 1], S[:, 2]
    phi = u ** (2 * a) * v ** (2 - 2 * a) - w * w
    g_phi = np.stack([2 * a * u ** (2 * a - 1) * v ** (2 - 2 * a), 2 * (1 - a) * u ** (2 * a) * v ** (1 - 2 * a), -2 * w], axis=1)
    H_phi = np.zeros((len(u), 3, 3))
    H_phi[:, 0, 0] = 2 * a * (2 * a - 1) * u ** (2 * a - 2) * v ** (2 - 2 * a)
    H_phi[:, 0, 1] = H_phi[:, 1, 0] = 4 * a * (1 - a) * u ** (2 * a - 1) * v ** (1 - 2 * a)
    H_phi[:, 1, 1] = 2 * (1 - a) * (1 - 2 * a) * u ** (2 * a) * v ** (-2 * a)
    H_phi[:, 2, 2] = -2.0
    val = float(np.sum(-np.log(phi) - (1 - a) * np.log(u) - a * np.log(v)))
    g = -g_phi / phi[:, None] - np.stack([(1 - a) / u, a / v, np.zeros_like(u)], axis=1)
    H = np.einsum("ni,nj->nij", g_phi, g_phi) / (phi ** 2)[:, None, None] - H_phi / phi[:, None, None]
    H[:, 0, 0] += (1 - a) / u ** 2
    H[:, 1, 1] += a / v ** 2
    return val, g, H


# ----------------------------------------------------------------------------- Dualkegel

def dual_exp_inside(U: np.ndarray) -> np.ndarray:
    """Streng im Inneren des Dualkegels des Exponentialkegels: u < 0, w > 0 und log(-u) + v/u < 1 + log w."""
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    ok = (u < 0) & (w > 0)
    with np.errstate(all="ignore"):
        gap = 1.0 + np.log(w) - np.log(-u) - v / u
    return ok & np.isfinite(gap) & (gap > 0)


def exp_member(s) -> bool:
    """Liegt s im Exponentialkegel (y > 0, y e^(x/y) <= z, in der Form y log(z/y) >= x)?"""
    x, y, z = s
    return bool(y > 0 and z > 0 and y * (math.log(z) - math.log(y)) >= x)


def dual_exp_member(u_) -> bool:
    u, v, w = u_
    if u < 0:
        return bool(w > 0 and math.log(-u) + v / u <= 1.0 + math.log(w))
    return bool(u == 0 and v >= 0 and w >= 0)
