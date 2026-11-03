"""ADMM für Kegelprogramme – ein fester Schritt, eine Projektion, und wer wählt ρ? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 9 der Reihe „Konvexe und konische Optimierung“: ADMM löst dasselbe Kegelprogramm wie die Barriere aus Stück 5 bis 7, aber mit einer festen linearen Lösung und der Projektion aus Stück 8 statt eines Newton-Systems.
Die Demo zeigt, wie viele Schritte das braucht, wie stark die Schrittzahl am Parameter ρ und an der Skala der Daten hängt, was der Residuenausgleich ausrichtet, was nicht hilft, und wie ADMM neben der Barriere dasteht.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kad_constants as C
from kad_evaluation import (
    Settings,
    count_text,
    de,
    four_rows,
    hand_rows,
    ms_text,
    relax_rows,
    rho_text,
    run_view,
    scale_rows,
    scan_rows,
    sci,
    setting_rows,
    steps_text,
    verdict_four,
    verdict_relax,
    verdict_run,
    verdict_scale,
    verdict_scan,
    verdict_versus,
    versus_rows,
    warm_rows,
)
from kad_layout import dunkles_design, kopf_css
from kad_pdf_export import build_pdf
from kad_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kad_visualization import build_relax, build_run, build_scale, build_scan, build_versus, build_warm

st.set_page_config(page_title="ADMM für Kegelprogramme – Sebastian Hanisch", layout="wide")

CLOUD_KEYS = ("n", "stretch", "angle", "seed")


@st.cache_data(show_spinner=False)
def _run(kind, rho, tol, adaptive, every, mu, n, stretch, angle, seed):
    return run_view(Settings(kind, rho, tol, adaptive, every, mu, n, stretch, angle, seed))


@st.cache_data(show_spinner=False)
def _four(kind, rho, tol, adaptive, every, mu, n, stretch, angle, seed):
    return four_rows(Settings(kind, rho, tol, adaptive, every, mu, n, stretch, angle, seed))


@st.cache_data(show_spinner=False)
def _scan(n, stretch, angle, seed):
    return scan_rows(Settings(n=n, stretch=stretch, angle=angle, seed=seed))


@st.cache_data(show_spinner=False)
def _scale(rho, every, mu, n, stretch, angle, seed):
    s = Settings(rho=rho, every=every, mu=mu, n=n, stretch=stretch, angle=angle, seed=seed)
    return scale_rows(s), setting_rows(s)


@st.cache_data(show_spinner=False)
def _relax(n, stretch, angle, seed):
    s = Settings(n=n, stretch=stretch, angle=angle, seed=seed)
    return relax_rows(s), warm_rows(s)


@st.cache_data(show_spinner=False)
def _versus(n, stretch, angle, seed):
    return versus_rows(Settings(n=n, stretch=stretch, angle=angle, seed=seed))


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kad_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔁 ADMM für Kegelprogramme – ein fester Schritt, eine Projektion, und wer wählt ρ?")
st.markdown(
    """
**ADMM tauscht den Newton-Schritt gegen zwei billige Dinge: eine feste lineare Lösung und die Projektion auf den Kegel aus Stück 8.** Für `min cᵀx, s = b − A x ∈ K` rechnet ein Schritt `x` aus einer einmal
invertierten Matrix, projiziert `s` auf `K` und schreibt den Rest in die Dualvariable `u` fort. Auf der Kugel der Standardwolke kommt das bei gutem Parameter ρ in einigen hundert Schritten auf `10⁻⁶`,
wo die Barriere bis zur Toleranz `10⁻⁹` 30 bis 43 Newton-Schritte braucht. **Der Befund dieser Demo ist ρ:** dieselbe Aufgabe braucht je nach ρ 185 Schritte oder gar keine Lösung, das beste ρ wandert mit der Skala der Daten,
und der Residuenausgleich holt die nicht gelösten Läufe zurück, aber nicht gleichmäßig.
Stück 9 der Reihe „Konvexe und konische Optimierung“; zweites Stück des Strangs Erste Ordnung.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Das Programm:** die kleinste umschließende Kugel einer Punktwolke, als Kegelprogramm `min R` mit `b − A x ∈ K` in vier Kegeln (Lorentz, PSD 3×3, Potenzkegel für die ℓ₃-Kugel, Exponentialkegel für die log-sum-exp-Kugel): dieselben Programme wie in Stück 5 bis 8.
2. **Ein Schritt:** `x ← −(AᵀA)⁻¹(c/ρ + Aᵀ(s − b + u))`, dann `s ← P_K(b − A x − u)` und `u ← u + A x + s − b`. `AᵀA` hängt nicht von ρ ab, die Projektion `P_K` ist die aus Stück 8.
3. **Abbruch:** primales Residuum `‖A x + s − b‖` und duales Residuum `ρ‖Aᵀ(s⁺ − s)‖` beide klein (relativ zur Größe der Daten). Der Wertfehler gegen `R*` aus der Barriere ist der Maßstab, kein Teil des Verfahrens.
4. **Der Parameter ρ:** eine Strafe auf die Verletzung von `A x + s = b`. Zu klein oder zu groß macht das Verfahren langsam (Experiment 2), und das beste ρ hängt an der Skala der Daten (Experiment 3).
5. **Residuenausgleich:** alle paar Schritte ρ verdoppeln, wenn das primale Residuum das duale um das μ-Fache übersteigt, und halbieren im umgekehrten Fall (übliche Wahl μ = 10).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Lauf (Ergebnis, Experiment 1 und 3)**")
    kind = st.selectbox("Kegel", C.KINDS, key="kind_select", format_func=C.KIND_LABELS.get, help="Auf welchem Kegel der Hauptlauf rechnet; Experiment 1 zeigt dieselben Einstellungen immer auf allen vieren.")
    rho = st.select_slider("Parameter ρ", options=C.RHO_CHOICES, key="rho_select", format_func=rho_text, help="Strafe auf die Verletzung von A x + s = b. Experiment 2 zeigt für jeden Kegel, welche Werte gut sind.")
    tol = st.select_slider("Toleranz", options=C.TOL_CHOICES, key="tol_select", format_func=sci, help="Relative Schranke für beide Residuen.")
    adaptive = st.checkbox("Residuenausgleich", key="adaptive_check", help="ρ alle paar Schritte verdoppeln oder halbieren, je nachdem, welches Residuum größer ist (ρ ist der Startwert).")
    every = st.select_slider("Ausgleich: Intervall (Schritte)", options=C.EVERY_CHOICES, key="every_select", help="Alle wie viele Schritte der Ausgleich prüft, ob er ρ ändert.")
    mu = st.select_slider("Ausgleich: Schwelle μ", options=C.MU_CHOICES, key="mu_select", format_func=rho_text, help="ρ wird erst geändert, wenn ein Residuum das andere um mehr als das μ-Fache übersteigt.")
    st.markdown("**Die Wolke (alle Experimente)**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat; je mehr, desto größer die Kegelprogramme und desto länger rechnen die Experimente.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_select": kind, "rho_select": float(rho), "tol_select": float(tol), "adaptive_check": bool(adaptive), "every_select": int(every), "mu_select": float(mu), "n_slider": int(n),
                   "stretch_slider": float(stretch), "angle_slider": int(angle), "seed_input": int(seed)})

settings = Settings(kind, float(rho), float(tol), bool(adaptive), int(every), float(mu), int(n), float(stretch), int(angle), int(seed))
args = (settings.kind, settings.rho, settings.tol, settings.adaptive, settings.every, settings.mu, settings.n, settings.stretch, settings.angle, settings.seed)
view = _run(*args)
sm = view["summary"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Schritte", str(sm["iterations"]) if sm["converged"] else f"≥ {sm['iterations']}", help=f"Schritte bis zur Toleranz {sci(settings.tol)} (Deckel {C.MAX_ITER}); ≥ heißt: nicht gelöst.")
m2.metric("Wertfehler", sci(sm["value_error"]), help="|cᵀx − R*|/R* mit R* aus der Barriere (Toleranz 10⁻¹²).")
m3.metric("Primales Residuum", sci(sm["primal"]), help="‖A x + s − b‖ am Ende des Laufs.")
m4.metric("Duales Residuum", sci(sm["dual"]), help="ρ‖Aᵀ(s⁺ − s)‖ am Ende des Laufs.")
m5.metric("ρ am Ende", rho_text(sm["rho"]), help="Ohne Residuenausgleich der Startwert; mit Ausgleich der Wert am Ende.")
st.plotly_chart(build_run(view, settings.tol), width="stretch", key="main_chart")
st.caption("Primales und duales Residuum und der Wertfehler gegen R*, je Schritt (logarithmisch); die gepunktete Linie ist die Toleranz.")

if st.button("📄 PDF des aktuellen Laufs erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, view))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-admm.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie viele Schritte braucht ADMM, und wer wählt ρ?")
verdict(verdict_run(settings, view))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein Schritt von Hand, derselbe Lauf auf vier Kegeln")
st.caption("Das Handbeispiel `min t` mit `(t; 3, 4)` im Lorentz-Kegel (Optimum `t = 5`), ρ = 1, Start `x = 0`, `u = 0`: die ersten vier Schritte und der Fixpunkt. Darunter dieselben Einstellungen der Seitenleiste auf den vier Kegeln der Wolke.")
hand = hand_rows()
st.dataframe(pd.DataFrame([{"Schritt": r["step"], "x": de(r["x"], 4), "s": "(" + "; ".join(de(float(v), 3) for v in r["s"]) + ")", "u": "(" + "; ".join(de(float(v), 3) for v in r["u"]) + ")",
                            "primales Residuum": de(r["primal"], 4) if r["primal"] > 1e-6 else sci(r["primal"]), "duales Residuum": de(r["dual"], 4) if r["dual"] > 1e-6 else sci(r["dual"])} for r in hand]), hide_index=True, width="stretch")
four = _four(*args)
st.dataframe(pd.DataFrame([{"Kegel": r["name"], "Zeilen × Variablen": f"{r['rows']} × {r['variables']}", "R*": de(r["rstar"], 5), "Barriere: Newton-Schritte (10⁻⁹)": r["newton"],
                            "ADMM: Schritte": count_text(r["summary"]), "Wertfehler": sci(r["summary"]["value_error"]), "µs je Schritt": de(r["step_us"], 0)} for r in four]), hide_index=True, width="stretch")
verdict(verdict_four(settings, four))

st.markdown("---")

st.subheader("🔬 Experiment 2: Wer wählt ρ?")
st.caption(f"Schritte bis zur Toleranz 10⁻⁶ für neun Werte von ρ auf allen vier Kegeln; Deckel {C.SCAN_CAP} Schritte, ein Kreuz heißt nicht gelöst.")
scan = _scan(settings.n, settings.stretch, settings.angle, settings.seed)
st.plotly_chart(build_scan(scan, settings.rho, C.SCAN_CAP), width="stretch", key="scan_chart")
st.dataframe(pd.DataFrame([{"Kegel": r["name"], **{f"ρ = {rho_text(x['rho'])}": count_text(x).replace(" (nicht gelöst)", "") for x in r["results"]}} for r in scan]), hide_index=True, width="stretch")
verdict(verdict_scan(scan, settings.rho))

st.markdown("---")

st.subheader("🔬 Experiment 3: ρ wandert mit den Daten")
st.caption(f"Die Wolke wird um den Faktor 0,1 bis 100 gestreckt; festes ρ (Start der Seitenleiste) gegen Residuenausgleich (Intervall und Schwelle der Seitenleiste), Deckel {C.SCALE_CAP} Schritte. Darunter die neun Einstellungen auf dem Lorentz-Programm.")
scale, settings_table = _scale(settings.rho, settings.every, settings.mu, settings.n, settings.stretch, settings.angle, settings.seed)
st.plotly_chart(build_scale(scale, C.SCALE_CAP), width="stretch", key="scale_chart")
st.dataframe(pd.DataFrame([{"Kegel": r["name"], "Ausgleich": "festes ρ", **{f"Skala {rho_text(x['scale'])}": count_text(x["fixed"]).replace(" (nicht gelöst)", "") for x in r["results"]}} for r in scale]
                           + [{"Kegel": r["name"], "Ausgleich": "mit Ausgleich", **{f"Skala {rho_text(x['scale'])}": f"{count_text(x['adaptive']).replace(' (nicht gelöst)', '')} ({x['adaptive']['rho_changes']} Wechsel)" for x in r["results"]}} for r in scale]),
             hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Skala": rho_text(r["scale"]), "Intervall": r["every"], **{f"Schwelle μ = {rho_text(mu_)}": count_text(r[mu_]).replace(" (nicht gelöst)", "") for mu_ in C.MU_CHOICES}} for r in settings_table]), hide_index=True, width="stretch")
verdict(verdict_scale(scale, settings_table, settings))

st.markdown("---")

st.subheader("🔬 Experiment 4: Was nicht hilft")
st.caption("Überrelaxation α und der Warmstart. Beide sind in Büchern empfohlen; hier die Zahlen auf denselben Programmen.")
relax, warm = _relax(settings.n, settings.stretch, settings.angle, settings.seed)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_relax(relax), width="stretch", key="relax_chart")
with c2:
    st.plotly_chart(build_warm(warm), width="stretch", key="warm_chart")
st.dataframe(pd.DataFrame([{"Kegel": r["name"], "Grober Lauf (10⁻³)": r["rough"], "kalt": r["cold"], "Start x": r["x_only"], "Start x und u": r["both"], "grob + Start x und u": r["rough"] + r["both"]} for r in warm]), hide_index=True, width="stretch")
verdict(verdict_relax(relax, warm))

st.markdown("---")

st.subheader("🔬 Experiment 5: ADMM neben der Barriere")
st.caption("Schritte und Zeit bis Wertfehler 10⁻³, 10⁻⁶ und 10⁻⁹ gegen R*, für ADMM (bestes ρ der kleinen Reihe) und die Barriere (kleinste Toleranz, deren Wert das Ziel trifft). Zeiten nach einem Warmlauf, beste von drei Läufen, ein BLAS-Thread; sie hängen vom Rechner ab.")
versus = _versus(settings.n, settings.stretch, settings.angle, settings.seed)
st.plotly_chart(build_versus(versus), width="stretch", key="versus_chart")
st.dataframe(pd.DataFrame([{"Kegel": r["name"], "Wertfehler": sci(x["target"], 0), "Barriere: Schritte": x["barrier_steps"], "Barriere: ms": ms_text(x["barrier_seconds"]), "ADMM: Schritte": count_text({"iterations": x["admm_iterations"], "converged": x["admm_converged"]}),
                            "ADMM: ms": ms_text(x["admm_seconds"]), "ADMM: ρ": rho_text(x["admm_rho"])} for r in versus for x in r["results"]]), hide_index=True, width="stretch")
verdict(verdict_versus(versus))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein festes ρ passt zu jeder Aufgabe** | Die Schrittzahl ist U-förmig in ρ (Experiment 2), und das beste ρ wandert mit der Skala der Daten (Experiment 3): ρ = 1 löst die Wolke mit Faktor 1 in einigen hundert Schritten und die mit Faktor 100 gar nicht. | Stück 10 |
| **Der Residuenausgleich macht ρ unwichtig** | Er holt die nicht gelösten Läufe zurück, aber die neun Einstellungen aus Intervall und Schwelle streuen bei Skala 100 um mehr als das Dreifache (Experiment 3); keine Einstellung ist auf beiden Skalen die beste. | – |
| **ADMM ist ein Verfahren für grobe Genauigkeit** | Auf diesen Programmen erreicht es ohne Mühe hohe Genauigkeit (Wertfehler unter 10⁻⁹ nach wenigen hundert Schritten); die Erwartung „grob“ stammt von großen Programmen, die hier nicht gerechnet werden. | Stück 10 |
| **Ein ADMM-Schritt ist billiger als ein Newton-Schritt, also ist ADMM schneller** | Bei 3 bis 35 Variablen ist der Lorentz- und PSD-Schritt billiger, aber es sind drei- bis zwölfmal so viele; der Potenz- und Exponentialkegel-Schritt ist schon einzeln teurer (Experiment 1 und 5). Der Vorteil „kein Newton-System“ braucht große Programme. | Stück 10 |
| **Die Barriere ist hier immer schneller** | Nur für diese kleinen Programme und diese Implementierungen (numpy, dichte Matrizen); bei grobem Ziel (10⁻³) liegen ADMM und Barriere für Lorentz- und PSD-Kegel gleichauf. Über große Programme sagt Experiment 5 nichts. | Stück 10 |
| **Ein Warmstart über `x` spart Schritte** | Der Zustand des Verfahrens ist `(s, u)`: ein Start im groben `x` allein spart bei Lorentz, PSD und Potenzkegel nur 4 bis 6 %, beim Exponentialkegel allerdings 50 %, mit `u` setzt der Lauf fort, wo der grobe aufgehört hat (Experiment 4). | – |
| **Überrelaxation beschleunigt** | Auf diesen Programmen nicht: α = 1 ist bestes oder gleich gut, je größer α, desto mehr Schritte (Experiment 4). | – |
| **ADMM erkennt unzulässige Programme** | Diese Form nicht: ohne homogene selbstduale Einbettung (wie in SCS) gibt es kein Zertifikat für unzulässige oder unbeschränkte Programme, der Lauf endet am Schrittdeckel. | – |
| **Die Zeiten gelten allgemein** | Sie hängen vom Rechner ab; die Projektionen des Exponential- und Potenzkegels haben je nach Rechner ein bis zwei Millisekunden festen Aufwand je Aufruf (numpy), bei 1 bis 32 Kegeln unabhängig von der Zahl der Kegel (bei Hunderten wächst er). Zeiten werden nur in ihrer Struktur getestet. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Programm.** $\min c^\top x$ unter $s = b - Ax \in K$ mit vollem Spaltenrang von $A$. Mit der Dualvariable $y$ der Gleichung $Ax + s = b$ und $u = y/\rho$ lautet ADMM (skalierte Form, Strafe $\rho > 0$):
$$x^{+} = \arg\min_x\ c^\top x + \tfrac{\rho}{2}\|Ax + s - b + u\|^2 = -(A^\top A)^{-1}\big(c/\rho + A^\top(s - b + u)\big),$$
$$s^{+} = P_K\big(b - Ax^{+} - u\big),\qquad u^{+} = u + Ax^{+} + s^{+} - b.$$
$A^\top A$ hängt nicht von $\rho$ ab und wird einmal invertiert. $P_K$ ist die Projektion aus Stück 8.

**Fixpunkt und Optimalität.** Ein Fixpunkt erfüllt $Ax + s = b$, $s \in K$, $c + \rho A^\top u = 0$ und $\rho u \in K^*$ mit $\langle s, \rho u\rangle = 0$; das sind genau die KKT-Bedingungen: $y = \rho u$ ist die Dualvariable. Im Handbeispiel
($A = (-1, 0, 0)^\top$, $b = (0, 3, 4)$, $c = 1$, $\rho = 1$) ist der Fixpunkt $x = 5$, $s = (5; 3, 4)$, $u = (1; -0{,}6; -0{,}8)$.

**Residuen und Abbruch.** Primal $r = \|Ax + s^{+} - b\|$, dual $d = \rho\|A^\top(s^{+} - s)\|$; Abbruch, wenn $r \le \varepsilon\max(\|Ax\|, \|s\|, \|b\|, 1)$ und $d \le \varepsilon\max(\rho\|A^\top u\|, \|c\|, 1)$.

**Überrelaxation.** $Ax$ wird im $s$- und $u$-Schritt durch $\alpha Ax + (1-\alpha)(b - s)$ ersetzt, $\alpha \in (0, 2)$.

**Residuenausgleich.** Alle $k$ Schritte: ist $r > \mu d$, so $\rho \leftarrow \tau\rho$ und $u \leftarrow u/\tau$; ist $d > \mu r$, so $\rho \leftarrow \rho/\tau$ und $u \leftarrow \tau u$ (übliche Wahl $\mu = 10$, $\tau = 2$). Da $A^\top A$ nicht von $\rho$ abhängt, kostet der Wechsel hier nichts; bei SCS und OSQP enthält die KKT-Matrix $\rho$.

**Abgrenzung.** SCS (O'Donoghue u. a.) wendet ADMM auf die homogene selbstduale Einbettung an und liefert damit Zertifikate für Unzulässigkeit und Unbeschränktheit; diese Demo rechnet die einfache Form ohne Einbettung.

**Literatur.** S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein: *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*, Foundations and Trends in Machine Learning 3 (1), 2011, S. 1–122 (ADMM, skalierte Form, Residuenausgleich).
B. O'Donoghue, E. Chu, N. Parikh, S. Boyd: *Conic Optimization via Operator Splitting and Homogeneous Self-Dual Embedding*, Journal of Optimization Theory and Applications 169 (3), 2016, S. 1042–1068 (SCS).
B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd: *OSQP: An Operator Splitting Solver for Quadratic Programs*, arXiv:1711.08013 (Preprint; die Zeitschriftenfassung ist hier nicht geprüft; adaptives ρ aus dem Verhältnis der normierten Residuen).
Die Regel μ = 10, τ = 2 wird in Folgearbeiten dem Kapitel 3.4.1 von Boyd u. a. zugeschrieben; hier als übliche Wahl gerechnet und gemessen, das Kapitel selbst ist nicht nachgelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
