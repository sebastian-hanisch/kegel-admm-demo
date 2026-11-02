"""Pfade der App, die der Rauchtest nicht erreicht: jeder Kegel, die fünf Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kad_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key == "kind_select":
        return at.sidebar.selectbox(key=key)
    if key == "seed_input":
        return at.sidebar.number_input(key=key)
    if key == "adaptive_check":
        return at.sidebar.checkbox(key=key)
    if key.endswith("_select"):
        return at.sidebar.select_slider(key=key)
    return at.sidebar.slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Schritte"] == "457" and values["ρ am Ende"] == "1" and values["Wertfehler"].endswith(("⁻¹²", "⁻¹³", "⁻¹⁴"))
    text = _texts(at)
    for part in ("ADMM mit ρ = 1 auf der Kugel der 8 Punkte als **Lorentz-Kegel**-Programm", "Mit ρ = 1 und Toleranz 1,0·10⁻⁶ braucht ADMM auf denselben vier Programmen", "Das beste ρ der Reihe je Kegel",
                 "Skala der Wolke 0,1 / 1 / 10 / 100", "Überrelaxation (bestes ρ der kleinen Reihe je α)", "Bei Wertfehler 10⁻⁶ (Zeiten nach Warmlauf)"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) >= 7 and not at.error


def test_experiment_one_shows_the_hand_example():
    at = _app()
    first = at.dataframe[0].value
    assert list(first["Schritt"])[:4] == [1, 2, 3, 4] and first["x"][0] == "1,5000" and first["s"][0] == "(3,250; 1,950; 2,600)" and first["u"][0] == "(1,750; −1,050; −1,400)"
    assert first["x"].iloc[-1] == "5,0000" and first["u"].iloc[-1] == "(1,000; −0,600; −0,800)"


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("kind", C.KINDS)
def test_every_cone_renders_without_error(kind):
    at = _app(kind_select=kind, rho_select=0.03 if kind != "e" else 0.01)
    assert not at.error
    assert any(f"als **{C.KIND_LABELS[kind]}**-Programm" in s.value for s in list(at.success) + list(at.warning))


@pytest.mark.parametrize("rho", C.RHO_CHOICES)
def test_every_rho_renders(rho):
    at = _app(rho_select=rho)
    assert not at.error and not at.exception


@pytest.mark.parametrize("tol", C.TOL_CHOICES)
def test_every_tolerance_renders(tol):
    at = _app(tol_select=tol)
    assert not at.error and not at.exception


def test_balancing_switch_and_its_settings_render():
    at = _app(adaptive_check=True, every_select=10, mu_select=2.0, rho_select=100.0)
    assert not at.error and any("mit Residuenausgleich" in s.value for s in list(at.success) + list(at.warning))
    assert {m.label: m.value for m in at.metric}["ρ am Ende"] != "100"


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_controls_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX), "rho_select": pick(C.RHO_CHOICES[0], C.RHO_CHOICES[-1]),
             "tol_select": pick(C.TOL_CHOICES[0], C.TOL_CHOICES[-1]), "every_select": pick(C.EVERY_CHOICES[0], C.EVERY_CHOICES[-1]), "mu_select": pick(C.MU_CHOICES[0], C.MU_CHOICES[-1])}
    at = _app(**state)
    assert not at.exception and not at.error
