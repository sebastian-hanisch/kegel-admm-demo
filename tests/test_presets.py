"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Aussagen der Hilfetexte."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kad_constants as C
import kad_evaluation as ev
import kad_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Lorentz-Kugel, ρ = 1"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds():
    for preset in C.PRESETS.values():
        assert preset["kind"] in C.KINDS and preset["rho"] in C.RHO_CHOICES and preset["tol"] in C.TOL_CHOICES and preset["every"] in C.EVERY_CHOICES and preset["mu"] in C.MU_CHOICES
        assert isinstance(preset["adaptive"], bool)
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.STRETCH_MIN <= preset["stretch"] <= C.STRETCH_MAX and C.ANGLE_MIN <= preset["angle"] <= C.ANGLE_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def _summary(name):
    return ev.run_view(ev.Settings(**C.PRESETS[name]))["summary"]


def test_the_help_texts_name_what_the_presets_produce():
    base = _summary(DEFAULT)
    good = _summary("Gutes ρ: der flache Boden")
    assert base["converged"] and good["converged"] and good["iterations"] < 0.5 * base["iterations"] and "ρ = 0,1" in C.PRESET_HELP["Gutes ρ: der flache Boden"]
    big = _summary("ρ zu groß: nicht gelöst")
    assert not big["converged"] and big["iterations"] == C.MAX_ITER and big["dual"] > 1e-6 and "ρ = 100" in C.PRESET_HELP["ρ zu groß: nicht gelöst"]
    saved = _summary("Residuenausgleich rettet ρ = 100")
    assert saved["converged"] and saved["rho_changes"] > 0 and saved["rho"] < 100.0 and saved["iterations"] < 1000
    p = _summary("Potenzkegel (ℓ₃-Kugel), ρ = 0,03")
    assert p["converged"] and p["iterations"] < 300 and "ρ = 0,03" in C.PRESET_HELP["Potenzkegel (ℓ₃-Kugel), ρ = 0,03"]
    e = _summary("Exponentialkegel, ρ = 0,01")
    assert e["converged"] and 300 <= e["iterations"] <= 500 and "ρ = 0,01" in C.PRESET_HELP["Exponentialkegel, ρ = 0,01"]


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["kind"] = "q"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["rho"] = "0.1"
    at.query_params["tol"] = "0.5"          # kein erlaubter Wert: bleibt Standard
    at.query_params["adaptive"] = "True"
    at.query_params["every"] = "abc"        # keine Zahl: bleibt Standard
    at.query_params["mu"] = "nan"           # keine Auswahl: bleibt Standard
    at.query_params["seed"] = "-5"          # unter der Grenze: wird begrenzt
    at.run()
    assert not at.exception
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX and at.sidebar.number_input(key="seed_input").value == 0
    assert at.session_state["rho_select"] == 0.1 and at.session_state["tol_select"] == C.DEFAULT_TOL and at.session_state["adaptive_check"] is True
    assert at.session_state["every_select"] == C.DEFAULT_EVERY and at.session_state["mu_select"] == C.DEFAULT_MU
    bad = AppTest.from_file(APP, default_timeout=600)
    bad.query_params["kind"] = "dreieck"
    bad.query_params["adaptive"] = "vielleicht"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="kind_select").value == C.DEFAULT_KIND and bad.session_state["adaptive_check"] is False
