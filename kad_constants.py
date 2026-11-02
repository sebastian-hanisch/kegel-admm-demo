"""Konstanten: Regler-Grenzen, feste Reihen der Experimente, Farben und Presets."""

import kad_projection as P

KINDS = ("q", "s", "p", "e")                           # die vier Kegel der Kugelprogramme
KIND_LABELS = {"q": "Lorentz-Kegel", "s": "PSD-Kegel 3×3", "p": "Potenzkegel (ℓ₃-Kugel)", "e": "Exponentialkegel (log-sum-exp, β = 10)"}
DEFAULT_KIND = "q"
RHO_CHOICES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_RHO = 1.0
TOL_CHOICES = (1e-3, 1e-4, 1e-6, 1e-8)
DEFAULT_TOL = 1e-6
EVERY_CHOICES = (10, 25, 100)                          # Intervall des Residuenausgleichs
DEFAULT_EVERY = 25
MU_CHOICES = (2.0, 10.0, 100.0)                        # Schwelle des Residuenausgleichs
DEFAULT_MU = 10.0
TAU = 2.0                                              # Faktor, um den ρ beim Ausgleich wechselt
N_MIN, N_MAX, DEFAULT_N = 4, 12, 8
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45
SEED_MAX, DEFAULT_SEED = 999999, 35

MAX_ITER = 5000                                        # Schrittdeckel des Hauptlaufs (Experiment 1 und 5)
SCAN_CAP = 1000                                        # Schrittdeckel der ρ-Reihe (Experiment 2)
SCAN_RHOS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
FOUR_CAP = 1500                                        # Schrittdeckel der vier Läufe in Experiment 1
SCALE_CAP = 3000                                       # Schrittdeckel der Skalenreihe (Experiment 3)
SCALES = (0.1, 1.0, 10.0, 100.0)                       # Streckung der Wolke in Experiment 3
SCALE_KINDS = ("q", "s")
SETTING_SCALES = (1.0, 100.0)                          # Skalen, auf denen die Einstellungen des Ausgleichs verglichen werden
ALPHAS = (1.0, 1.4, 1.6, 1.8)                          # Überrelaxation (Experiment 4)
RELAX_RHOS = (0.03, 0.1, 0.3)
RELAX_KINDS = ("q", "s")
WARM_RHO = 0.1
WARM_ROUGH_TOL = 1e-3
TARGETS = (1e-3, 1e-6, 1e-9)                           # Wertfehler, bei dem Barriere und ADMM verglichen werden (Experiment 5)
BARRIER_TOLS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)

COLORS = {"q": "#4c78a8", "s": "#8c6bb1", "e": "#2F6B65", "p": "#b08a2e", "primal": "#4c78a8", "dual": "#b08a2e", "value": "#2F6B65", "barrier": "#888888", "admm": "#2F6B65"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

_BASE = {"kind": DEFAULT_KIND, "rho": DEFAULT_RHO, "tol": DEFAULT_TOL, "adaptive": False, "every": DEFAULT_EVERY, "mu": DEFAULT_MU, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH,
         "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED}
PRESETS = {
    "Standardfall: Lorentz-Kugel, ρ = 1": dict(_BASE),
    "Gutes ρ: der flache Boden": dict(_BASE, rho=0.1),
    "ρ zu groß: nicht gelöst": dict(_BASE, rho=100.0),
    "Residuenausgleich rettet ρ = 100": dict(_BASE, rho=100.0, adaptive=True),
    "Potenzkegel (ℓ₃-Kugel), ρ = 0,03": dict(_BASE, kind="p", rho=0.03),
    "Exponentialkegel, ρ = 0,01": dict(_BASE, kind="e", rho=0.01),
}
PRESET_HELP = {
    "Standardfall: Lorentz-Kugel, ρ = 1": "Die kleinste umschließende Kugel der Wolke als Lorentz-Programm, ADMM mit ρ = 1 ohne Ausgleich: ein brauchbares, aber nicht das beste ρ.",
    "Gutes ρ: der flache Boden": "Dasselbe Programm mit ρ = 0,1, im flachen Boden der Reihe aus Experiment 2: deutlich weniger Schritte als bei ρ = 1.",
    "ρ zu groß: nicht gelöst": "ρ = 100 ohne Ausgleich: das duale Residuum fällt kaum, der Lauf endet am Schrittdeckel; nicht gelöst ist ein Ergebnis.",
    "Residuenausgleich rettet ρ = 100": "Derselbe Start ρ = 100, aber mit Residuenausgleich: ρ wird nach unten geführt, der Lauf konvergiert.",
    "Potenzkegel (ℓ₃-Kugel), ρ = 0,03": "Die ℓ₃-Kugel als Potenzkegel-Programm mit ρ = 0,03 (im Boden der Reihe): jede Projektion ist eine Wurzelsuche aus Stück 8.",
    "Exponentialkegel, ρ = 0,01": "Die log-sum-exp-Kugel (β = 10) als Exponentialkegel-Programm mit ρ = 0,01: mehr Schritte und teurere Schritte als die symmetrischen Kegel.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["kind"] in KINDS and p["rho"] in RHO_CHOICES and p["tol"] in TOL_CHOICES and p["every"] in EVERY_CHOICES and p["mu"] in MU_CHOICES for p in PRESETS.values())
assert set(KINDS) <= set(P.KINDS)
