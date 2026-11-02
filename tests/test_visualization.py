"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein, Diagramme bauen an den Rändern."""

import pytest

import kad_constants as C
import kad_evaluation as ev
import kad_visualization as V

S = ev.Settings(n=C.N_MIN)


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def view():
    return ev.run_view(S)


def test_run_chart_has_three_log_series_and_the_tolerance_line(view):
    fig = V.build_run(view, S.tol)
    assert _locked(fig) and [t.name for t in fig.data][0].startswith("primales") and len(fig.data) == 3 and fig.layout.yaxis.type == "log"
    assert all(y > 0 for t in fig.data for y in t.y) and list(fig.data[0].x) == view["iters"] and len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == S.tol


def test_scan_chart_marks_unsolved_runs_and_the_chosen_rho():
    rows = ev.scan_rows(S, rhos=(0.1, 100.0), cap=150)
    fig = V.build_scan(rows, 0.1, 150)
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    crosses = [t for t in fig.data if t.mode == "markers"]
    assert _locked(fig) and len(lines) == 4 and crosses and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log" and len(fig.layout.shapes) == 1
    assert sum(1 for t in fig.data if t.mode == "markers" and t.showlegend) == 1                       # „nicht gelöst“ steht nur einmal in der Legende
    assert all(all(not x["converged"] for x in [y for r in rows for y in r["results"] if y["rho"] == 100.0 and y["iterations"] == t.y[0]]) for t in crosses[:1])


def test_scale_chart_has_a_panel_per_cone_and_two_lines_each():
    scale = ev.scale_rows(S, scales=(1.0, 100.0), cap=300)
    fig = V.build_scale(scale, 300)
    names = [t.name for t in fig.data if t.mode == "lines+markers"]
    assert _locked(fig) and names.count("festes ρ") == len(scale) == names.count("Residuenausgleich") and len(names) == 2 * len(scale)
    assert sum(1 for t in fig.data if t.mode == "lines+markers" and t.showlegend) == 2                  # die Legende nennt jede Linie nur einmal
    assert len(fig.layout.annotations) == len(scale) and fig.layout.xaxis.type == "log"


def test_relax_and_warm_charts():
    relax = ev.relax_rows(S, cap=600)
    fig = V.build_relax(relax)
    assert _locked(fig) and len(fig.data) == len(C.RELAX_KINDS) and list(fig.data[0].x) == list(C.ALPHAS)
    warm = ev.warm_rows(S, cap=600)
    w = V.build_warm(warm)
    assert _locked(w) and [t.name for t in w.data] == ["kalt (x = 0, u = 0)", "Start im groben x, u = 0", "Start im groben x und u"] and list(w.data[0].y) == [r["cold"] for r in warm]


def test_versus_chart_has_steps_and_times_for_both_methods():
    rows = ev.versus_rows(S, targets=(1e-3, 1e-6), cap=2000)
    fig = V.build_versus(rows, 1e-6)
    assert _locked(fig) and len(fig.data) == 4 and fig.layout.barmode == "group" and fig.layout.annotations[0].text == "Schritte bis Wertfehler 1·10⁻⁶"
    assert list(fig.data[0].y) == [r["results"][1]["barrier_steps"] for r in rows] and list(fig.data[1].y) == [r["results"][1]["admm_iterations"] for r in rows]
    assert all(t > 0 for t in fig.data[2].y) and all(t > 0 for t in fig.data[3].y)
