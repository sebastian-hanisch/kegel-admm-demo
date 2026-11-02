"""Die kleinste umschließende Kugel einer Punktwolke in vier Kegeln (Lorentz, PSD, Potenz, Exponential), aus kegel-exponential-demo (Stück 7) und kegel-primal-dual-demo (Stück 6) übernommen.

  lp     l_p-Kugel (p >= 1):   min R, ||x_i - c||_p <= R.  Potenzkegel (s_ij, R, x_ij - c_j) mit a = 1/p, also |x_ij - c_j| <= s_ij^(1/p) R^(1-1/p), und die Zeile R - sum_j s_ij >= 0.
                         p = 1: Raute, p = 2: Kugel, p -> unendlich: Würfel.
  lse    log-sum-exp-Kugel (beta > 0):  min R, (1/beta) log sum_{j, +-} exp(beta (+-(x_ij - c_j))) <= R.  Exponentialkegel (beta(+-(x_ij - c_j)) - beta R, 1, z_ijk),
                         die Zeile 1 - sum z_ijk >= 0. Für beta -> unendlich der Würfel.
  soc    die Kugel als Lorentz-Kegel (Kern der Reihe, ν = 2 je Punkt): Gegenprobe für p = 2.
Die Punkte sind um den Mittelwert verschoben (`kad_scenario.make_cloud`)."""

from __future__ import annotations

import math

import numpy as np

import kegel_kern as K
import kad_barrier as B


def _scale(points: np.ndarray, c0: np.ndarray) -> float:
    m = float(np.abs(points - c0).max())
    return m if m > 0 else 1.0


def build_lp_ball(points: np.ndarray, p: float) -> B.Program:
    """Variablen (c, R, s_ij); n d Potenzkegel (s_ij, R, x_ij - c_j) mit a = 1/p, dann n Zeilen R - sum_j s_ij >= 0."""
    n, d = points.shape
    a = 1.0 / p
    nv = d + 1 + n * d
    k = np.arange(n * d)
    i, j = k // d, k % d
    A = np.zeros((3 * n * d + n, nv))
    b = np.zeros(3 * n * d + n)
    A[3 * k, d + 1 + k] = -1.0                      # s0 = s_ij
    A[3 * k + 1, d] = -1.0                          # s1 = R
    A[3 * k + 2, j] = 1.0                           # s2 = x_ij - c_j
    b[3 * k + 2] = points[i, j]
    r = 3 * n * d + np.arange(n)                    # R - sum_j s_ij
    A[r, d] = -1.0
    for jj in range(d):
        A[r, d + 1 + np.arange(n) * d + jj] = 1.0
    c = np.zeros(nv)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    R0 = 4.0 * _scale(points, c0)
    x0 = np.concatenate([c0, [R0], np.full(n * d, R0 / (d + 1.0))])
    return B.Program(c, A, b, (("p", a, n * d), ("l", None, n)), x0)


def build_lse_ball(points: np.ndarray, beta: float) -> B.Program:
    """Variablen (c, R, z_ijk); 2 n d Exponentialkegel (beta (+-(x_ij - c_j)) - beta R, 1, z_ijk), dann n Zeilen 1 - sum_k z_ijk >= 0."""
    n, d = points.shape
    K2 = 2 * d
    nv = d + 1 + n * K2
    kk = np.arange(n * K2)
    i, jk = kk // K2, kk % K2
    j, sg = jk // 2, np.where(jk % 2 == 0, 1.0, -1.0)
    A = np.zeros((3 * n * K2 + n, nv))
    b = np.zeros(3 * n * K2 + n)
    A[3 * kk, j] = sg * beta                        # s0 = sg beta (x_ij - c_j) - beta R
    A[3 * kk, d] = beta
    b[3 * kk] = sg * beta * points[i, j]
    b[3 * kk + 1] = 1.0                             # s1 = 1
    A[3 * kk + 2, d + 1 + kk] = -1.0                # s2 = z
    r = 3 * n * K2 + np.arange(n)
    for q in range(K2):
        A[r, d + 1 + np.arange(n) * K2 + q] = 1.0
    b[r] = 1.0
    c = np.zeros(nv)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    R0 = 4.0 * _scale(points, c0) + 2.0 * math.log(2.0 * K2) / beta
    x0 = np.concatenate([c0, [R0], np.full(n * K2, 1.0 / (2.0 * K2))])
    return B.Program(c, A, b, (("e", None, n * K2), ("l", None, n)), x0)


def build_soc_ball(points: np.ndarray) -> K.ConeProblem:
    """Je Punkt s = (R; x_i - c) im Lorentz-Kegel der Größe d + 1 (wie Stück 2 und 5)."""
    n, d = points.shape
    A = np.zeros((n * (d + 1), d + 1))
    b = np.zeros(n * (d + 1))
    for i in range(n):
        r0 = i * (d + 1)
        A[r0, d] = -1.0
        A[r0 + 1:r0 + 1 + d, :d] = np.eye(d)
        b[r0 + 1:r0 + 1 + d] = points[i]
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [1.5 * float(np.linalg.norm(points - c0, axis=1).max()) + 1.0]])
    return K.ConeProblem(c, A, b, (("q", d + 1),) * n, x0)


def build_psd_ball(points: np.ndarray) -> K.ConeProblem:
    """Je Punkt die Matrix [[R I, v], [v^T, R]] mit v = x_i - c im PSD-Kegel der Größe d + 1 (Schur-Komplement: R >= ||v||); svec mit sqrt 2 auf der Außendiagonale."""
    n, d = points.shape
    k = d + 1
    iu = np.triu_indices(k)
    dim = len(iu[0])
    A = np.zeros((n * dim, d + 1))
    b = np.zeros(n * dim)
    for i in range(n):
        for r, (a, c_) in enumerate(zip(*iu)):
            f = 1.0 if a == c_ else K.SQ2
            row = i * dim + r
            if a == c_:
                A[row, d] = -1.0
            elif c_ == d:
                A[row, a] = f
                b[row] = f * points[i, a]
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [1.5 * float(np.linalg.norm(points - c0, axis=1).max()) + 1.0]])
    return K.ConeProblem(c, A, b, (("s", k),) * n, x0)
