"""PDF-Export des aktuellen ADMM-Laufs (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import kad_constants as C
from kad_evaluation import Settings, de, rho_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "α": "alpha", "ρ": "rho", "μ": "mu", "τ": "tau", "κ": "kappa", "€": "EUR",
            "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt", "‖": "|", "°": "o"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, view: dict) -> bytes:
    sm, case = view["summary"], view["case"]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("ADMM für ein Kegelprogramm"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Kleinste umschließende Kugel von {s.n} Punkten als {C.KIND_LABELS[s.kind]}-Programm. rho = {rho_text(s.rho)}, Toleranz {sci(s.tol)}, "
                            f"Residuenausgleich: {'ja (Intervall ' + str(s.every) + ', Schwelle ' + rho_text(s.mu) + ')' if s.adaptive else 'nein'}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rows = [("Schritte", str(sm["iterations"]) + ("" if sm["converged"] else " (nicht gelöst, Deckel)")), ("Optimalwert R* (Barriere)", de(case["rstar"], 6)), ("Wertfehler", sci(sm["value_error"])),
            ("primales Residuum", sci(sm["primal"])), ("duales Residuum", sci(sm["dual"])), ("rho am Ende", rho_text(sm["rho"])), ("Änderungen von rho", str(sm["rho_changes"])),
            ("Newton-Schritte der Barriere (1e-9)", str(case["newton"]))]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(80, 6, "Kennzahl", border=1)
    pdf.cell(100, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, v in rows:
        pdf.cell(80, 5.5, _s(k), border=1)
        pdf.cell(100, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-admm-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
