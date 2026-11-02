"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kad_constants as C
from kad_evaluation import sci


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=False):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _positive(values, floor=1e-17):
    return [max(float(v), floor) for v in values]


def _short(name: str) -> str:
    return name.split(" (")[0]


# ----------------------------------------------------------------------------- Experiment 1

def build_run(view: dict, tol: float) -> go.Figure:
    fig = go.Figure()
    for key, name, color in (("primal", "primales Residuum ‖A x + s − b‖", C.COLORS["primal"]), ("dual", "duales Residuum ρ‖Aᵀ(s⁺ − s)‖", C.COLORS["dual"]), ("error", "Wertfehler |cᵀx − R*|/R*", C.COLORS["value"])):
        fig.add_trace(go.Scatter(x=view["iters"], y=_positive(view[key]), mode="lines", name=name, line=dict(color=color, width=2.5)))
    fig.add_hline(y=tol, line=dict(color="#888888", dash="dot"), annotation_text="Toleranz", annotation_position="top right")
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="Größe", type="log", exponentformat="power", dtick=2)
    return _base(fig, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 2

def build_scan(rows: list, rho: float, cap: int) -> go.Figure:
    fig = go.Figure()
    legend_done = False
    for r in rows:
        xs = [x["rho"] for x in r["results"]]
        ys = [x["iterations"] for x in r["results"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=_short(r["name"]), line=dict(color=C.COLORS[r["kind"]], width=3),
                                 hovertemplate="ρ = %{x}: %{y} Schritte<extra></extra>"))
        bad = [x for x in r["results"] if not x["converged"]]
        if bad:
            fig.add_trace(go.Scatter(x=[x["rho"] for x in bad], y=[x["iterations"] for x in bad], mode="markers", marker=dict(symbol="x", size=12, color=C.COLOR_BAD, line=dict(width=2)),
                                     name="nicht gelöst", showlegend=not legend_done, hovertemplate=f"nicht gelöst nach {cap} Schritten<extra></extra>"))
            legend_done = True
    fig.add_vline(x=rho, line=dict(color="#888888", dash="dot"), annotation_text="Ihr ρ", annotation_position="top")
    fig.update_xaxes(title_text="ρ", type="log")
    fig.update_yaxes(title_text=f"Schritte bis Toleranz 10⁻⁶ (Deckel {cap})", type="log", tickvals=[100, 200, 500, 1000, cap], ticktext=["100", "200", "500", "1000", str(cap)])
    return _base(fig, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 3

def build_scale(rows: list, cap: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(rows), subplot_titles=[_short(r["name"]) for r in rows], shared_yaxes=True)
    for col, r in enumerate(rows, start=1):
        xs = [x["scale"] for x in r["results"]]
        for key, name, color, dash in (("fixed", "festes ρ", C.COLOR_BAD, "dash"), ("adaptive", "Residuenausgleich", C.COLOR_OK, "solid")):
            ys = [x[key]["iterations"] for x in r["results"]]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name, line=dict(color=color, width=3, dash=dash), showlegend=col == 1,
                                     hovertemplate="Skala %{x}: %{y} Schritte<extra></extra>"), row=1, col=col)
            bad = [x for x in r["results"] if not x[key]["converged"]]
            if bad:
                fig.add_trace(go.Scatter(x=[x["scale"] for x in bad], y=[x[key]["iterations"] for x in bad], mode="markers", marker=dict(symbol="x", size=12, color=C.COLOR_BAD, line=dict(width=2)),
                                         showlegend=False, hovertemplate=f"nicht gelöst nach {cap} Schritten<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Streckung der Wolke", type="log", tickvals=list(C.SCALES), ticktext=[str(f).replace(".", ",") for f in C.SCALES], row=1, col=col)
    fig.update_yaxes(title_text="Schritte bis Toleranz 10⁻⁶", type="log", tickvals=[100, 300, 1000, cap], ticktext=["100", "300", "1000", str(cap)], row=1, col=1)
    return _base(fig, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 4

def build_relax(relax: list) -> go.Figure:
    fig = go.Figure()
    for r in relax:
        fig.add_trace(go.Scatter(x=[x["alpha"] for x in r["results"]], y=[x["iterations"] for x in r["results"]], mode="lines+markers", name=_short(r["name"]), line=dict(color=C.COLORS[r["kind"]], width=3),
                                 hovertemplate="α = %{x}: %{y} Schritte<extra></extra>"))
    fig.update_xaxes(title_text="Überrelaxation α", tickvals=list(C.ALPHAS))
    fig.update_yaxes(title_text="Schritte bis 10⁻⁶ (bestes ρ der kleinen Reihe)", rangemode="tozero")
    return _base(fig)


def build_warm(warm: list) -> go.Figure:
    names = [_short(r["name"]) for r in warm]
    fig = go.Figure()
    for key, label, color in (("cold", "kalt (x = 0, u = 0)", C.COLOR_BAD), ("x_only", "Start im groben x, u = 0", C.COLOR_WARN), ("both", "Start im groben x und u", C.COLOR_OK)):
        fig.add_trace(go.Bar(x=names, y=[r[key] for r in warm], name=label, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Schritte bis 10⁻⁶")
    return _base(fig, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 5

def build_versus(rows: list, target: float = 1e-6) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Schritte bis Wertfehler {sci(target, 0)}", "Zeit (ms, nach Warmlauf)"))
    names = [_short(r["name"]) for r in rows]
    pick = [next(x for x in r["results"] if x["target"] == target) for r in rows]
    fig.add_trace(go.Bar(x=names, y=[x["barrier_steps"] for x in pick], name="Barriere (Newton-Schritte)", marker_color=C.COLORS["barrier"]), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[x["admm_iterations"] for x in pick], name="ADMM (Schritte)", marker_color=C.COLORS["admm"]), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[1e3 * x["barrier_seconds"] for x in pick], name="Barriere", marker_color=C.COLORS["barrier"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=[1e3 * x["admm_seconds"] for x in pick], name="ADMM", marker_color=C.COLORS["admm"], showlegend=False), row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(tickangle=-20)
    return _base(fig, legend_below=True)
