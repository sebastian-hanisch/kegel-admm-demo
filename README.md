# ADMM für Kegelprogramme – ein fester Schritt, eine Projektion, und wer wählt ρ? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-admm-demo.streamlit.app/)** -->

Stück 9 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Dies ist das zweite Stück des **zweiten Strangs (Erste Ordnung)**: dieselben Kegelprogramme wie in Stück 5 bis 8, gelöst mit der Projektion aus Stück 8 statt mit einem Newton-System.

## Kernfrage

**Wie viele Schritte braucht ADMM für ein Kegelprogramm, und wer wählt ρ?** Für `min cᵀx` mit `s = b − A x ∈ K` rechnet ein Schritt `x` aus einer **einmal invertierten Matrix** `AᵀA` (sie hängt nicht von ρ ab), projiziert `s` auf den Kegel (Stück 8) und schreibt den Rest in die Dualvariable `u = y/ρ` fort. Auf der Kugel der Standardwolke kommt das bei gutem ρ in 185 (Lorentz), 254 (PSD 3×3), 207 (ℓ₃-Kugel) und 381 Schritten (log-sum-exp-Kugel) auf eine Residuen-Toleranz von `10⁻⁶`, wo die Barriere bis zur Toleranz `10⁻⁹` 30 bis 43 Newton-Schritte braucht (beim gleichen Wertfehler `10⁻⁶` sind es 165 bis 313 ADMM-Schritte gegen 22 bis 36 Newton-Schritte, Experiment 5). **Der Befund ist ρ:** dieselbe Aufgabe braucht je nach ρ 185 Schritte oder gar keine Lösung innerhalb des Deckels, das beste ρ **wandert mit der Skala der Daten** (Experiment 3 zeigt das indirekt: festes ρ gegen Ausgleich), und der Residuenausgleich holt die nicht gelösten Läufe zurück, bleibt aber bei Skala 100 deutlich über dem Aufwand eines dort passend gewählten festen ρ (Faktor etwa 8 nach einer Nachprüfung außerhalb der Tests). Zwei Erwartungen bestätigen sich auf diesen Programmen nicht: ADMM ist hier **nicht** auf grobe Genauigkeit beschränkt (Wertfehler unter 10⁻¹³ nach 1 000 Schritten), und Überrelaxation **hilft nicht**; ein Warmstart über `x` allein spart nur 4 bis 6 % (beim Exponentialkegel 50 %).

Das Beispiel bleibt die kleinste umschließende Kugel einer Punktwolke, in vier Kegeln (Lorentz, PSD, Potenz als ℓ₃-Kugel, Exponential als log-sum-exp-Kugel).

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 8, unverändert): Lorentz- und PSD-Kegel-Programme der Kugel und ihre Barrierenmethode als Referenz für `R*`.
- **Verfahren** (`kad_admm.py`): ADMM in der skalierten Form `x⁺ = −(AᵀA)⁻¹(c/ρ + Aᵀ(s − b + u))`, `s⁺ = P_K(b − A x − u)`, `u⁺ = u + A x + s⁺ − b`, mit relativen Residuen als Abbruch (primal `‖A x + s − b‖`, dual `ρ‖Aᵀ(s⁺ − s)‖`), Überrelaxation `α`, Start `x₀`/`u₀` und Residuenausgleich (alle `k` Schritte ρ verdoppeln oder halbieren, `u` wird umgerechnet). Ein Wechsel von ρ kostet hier nichts, weil `AᵀA` nicht von ρ abhängt (anders bei OSQP und SCS, wo die KKT-Matrix ρ enthält). **Das ist kein SCS:** keine homogene selbstduale Einbettung, also kein Zertifikat für unzulässige oder unbeschränkte Programme (der Lauf endet am Schrittdeckel).
- **Projektionen** (`kad_projection.py`): die fünf Projektionen aus Stück 8 (gestapelt; Exponential- und Potenzkegel über die Wurzel von `⟨v, u × n⟩ = 0`), ohne die erste, falsche Fassung des Exponentialkegels.
- **Formen und Referenz** (`kad_forms.py`, `kad_barrier.py`, `kad_cones.py`): die vier Kugel-Programme und die Barrierenmethode für `R*` und den Vergleich, aus Stück 6 bis 8 übernommen.
- **Kein Eingriff in den Kern:** alles Neue steht in `kad_*.py`; BLAS-Threads laufen für die Zeitmessung und die Barriere mit einem Thread (`threadpoolctl`).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) das Handbeispiel Schritt für Schritt und dieselben Einstellungen auf den vier Kegeln; (2) Schritte gegen ρ für neun Werte (Deckel 1 000); (3) Skala der Wolke 0,1 bis 100: festes ρ gegen Residuenausgleich, dazu neun Einstellungen (Intervall × Schwelle); (4) Überrelaxation und Warmstart; (5) ADMM neben der Barriere: Schritte und Zeit bis Wertfehler `10⁻³`, `10⁻⁶`, `10⁻⁹`. `R*` kommt aus der Barrierenmethode mit Toleranz `10⁻¹²` und ist der Maßstab für den Wertfehler, nicht Teil des Verfahrens. Die Wolke (Punkte, Streckung, Drehung, Seed) und ρ, Toleranz und Ausgleich sind Regler; ein Schnellstart lädt sechs Szenarien, die Adresszeile spiegelt die Konfiguration.

## Befunde (gemessen, keine Behauptungen)

Die meisten Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung und Rechner, steht die gemessene Zahl trotzdem hier; Zeiten in Millisekunden, Linux-Werte und die ℓ₃-Vormessung stehen dort nicht oder nur als Band). Zeiten gelten für den Entwicklungsrechner und werden nur in ihrer Struktur getestet. Standardwolke: die ersten 8 Punkte (`n = 8`, Streckung 4, Drehung 45°, Seed 35).

| Frage | Befund |
|---|---|
| Die vier Optimalwerte | `R*` = **7,0359** (Lorentz und PSD), **6,2683** (ℓ₃-Kugel), **5,0463** (log-sum-exp β = 10); Barriere **31, 30, 34 und 43** Newton-Schritte bis `10⁻⁹` |
| Standardlauf (Lorentz, ρ = 1, Toleranz 10⁻⁶) | **457 Schritte**, Wertfehler 4,9·10⁻¹³, primales Residuum 1,3·10⁻⁸, duales 1,0·10⁻⁶. Mit ρ = 1 auf den vier Kegeln: Lorentz 457, PSD 1 066, ℓ₃ 620, **log-sum-exp nicht gelöst nach 1 500** |
| Genauigkeit | Lorentz mit ρ = 1: Wertfehler 2,4·10⁻¹⁰ nach 300 und 5,2·10⁻¹⁵ nach 1 000 Schritten; mit ρ = 0,03 ist die Residuen-Toleranz `10⁻⁹` nach 284 Schritten erreicht (Wertfehler 2,8·10⁻⁹). **Nicht „grob“** (die Erwartung stammt von großen Programmen) |
| Schritte gegen ρ (Toleranz 10⁻⁶, ρ = 0,003 / 0,01 / 0,03 / 0,1 / 0,3 / 1 / 3 / 10 / 100, Deckel 1 000) | Lorentz **431, 233, 185, 185, 208, 457**, dann nicht gelöst; PSD 349, 256, **254**, 268, 332, nicht gelöst (1 066 im größeren Deckel); ℓ₃ 614, 254, **207**, 274, 258, 620, dann nicht gelöst; log-sum-exp 659, **381**, 432, dann nicht gelöst. **16 von 36 Läufen** nicht gelöst. Ein flacher Boden von etwa einer Größenordnung von ρ (log-sum-exp: etwa einer halben), rechts ein steiler Anstieg |
| ρ wandert mit der Skala (Start ρ = 1, Deckel 3 000; Skala der Wolke 0,1 / 1 / 10 / 100) | Lorentz mit festem ρ: **185, 457, nicht gelöst, nicht gelöst**; mit Residuenausgleich (Intervall 25, Schwelle 10): **185, 259, 460, 1 486** (0, 2, 4, 7 Änderungen von ρ). PSD fest: 268, 1 066, –, –; Ausgleich 278, 333, 735, 1 883 |
| Einstellungen des Ausgleichs (Lorentz, Start ρ = 1; Intervall 10 / 25 / 100 × Schwelle 2 / 10 / 100) | Skala 1: **213 bis 457** Schritte (bei Schwelle 100 greift er nie ein: 457); Skala 100: **397 bis nicht gelöst** (Intervall 10, Schwelle 100: 397; Intervall 100, Schwelle 100: nicht gelöst nach 3 000; Intervall 25, Schwelle 10: 1 486). **Keine Einstellung ist auf beiden Skalen die beste** |
| Überrelaxation (bestes ρ aus 0,03 / 0,1 / 0,3) | α = 1 / 1,4 / 1,6 / 1,8: Lorentz **185, 206, 271, 496**; PSD **254, 292, 390, 670**. α = 1 ist bestes oder gleich gut; je größer α, desto mehr Schritte (auch mit feinerem ρ-Gitter, dort ab α ≈ 1,2) |
| Warmstart (ρ = 0,1, bis 10⁻⁶; grober Lauf bis 10⁻³) | Kalt: Lorentz 185, PSD 268, ℓ₃ 274, log-sum-exp 1 231. Start im groben `x` allein (mit `u = 0`) spart **4 bis 6 %** (Lorentz 178, PSD 252, ℓ₃ 259), beim log-sum-exp **50 %** (621). Mit `x` **und** `u`: Lorentz 97, PSD 161, ℓ₃ 165, log-sum-exp 439; zusammen mit dem groben Lauf davor (88, 107, 109, 792) genau so viele Schritte wie kalt (185, 268, 274, 1 231): **der Zustand ist `(s, u)`, der Warmstart setzt nur fort** |
| ADMM neben der Barriere (Ziele Wertfehler 10⁻³ / 10⁻⁶ / 10⁻⁹) | Newton-Schritte: Lorentz **18 / 23 / 28**, PSD 17 / 22 / 26, ℓ₃ 21 / 29 / 34, log-sum-exp 29 / 36 / 43. ADMM-Schritte (ρ = 0,03 bei allen vier Kegeln): Lorentz **52 / 165 / 252**, PSD 77 / 206 / 324, ℓ₃ 60 / 191 / 344, log-sum-exp 140 / 313 / 460: das **Drei- bis Zwölffache** der Schritte der Barriere (Lorentz und PSD) |
| Zeit (nach Warmlauf, ein BLAS-Thread, beste von 3 Läufen; Verhältnis ADMM zu Barriere) | Lorentz und PSD: **etwa gleichauf bei Ziel 10⁻³** (gemessen 1,0-mal; im Test Band 0,4 bis 3), **etwa 1,7- bis 3-mal** bei 10⁻⁶ und 10⁻⁹. ADMM-Schritt 0,06 bis 0,10 ms (Lorentz) gegen 0,2 ms je Newton-Schritt, PSD 0,10 bis 0,16 ms gegen 0,5 bis 0,9 ms: der Schritt ist billiger, die Schrittzahl größer. **ℓ₃ und log-sum-exp: 7- bis 25-mal** (Linux-Runner der CI: 4- bis 18-mal) so lang, weil ein ADMM-Schritt 1 bis 2 ms kostet (gegen etwa 0,5 bis 0,9 ms je Newton-Schritt): jeder Aufruf der Projektion ist eine Wurzelsuche in numpy mit **ein bis zwei Millisekunden festem Aufwand**, bei 1 Kegel wie bei 32 |
| Gegen cvxpy | Clarabel (Innere Punkte): `R*` aller vier Kugeln auf 10⁻⁶, über `cp.norm`, `cp.norm(·, 3)` und `cp.log_sum_exp`; SCS (ADMM mit Einbettung): dasselbe `R*`; die Dualvariable `y = ρu` ist zulässig für das Dualprogramm der Kugel und hat denselben Wert |

## Befunde und Korrekturen gegenüber dem Plan

- **Der ADMM-Schritt ist billiger als der Newton-Schritt (Lorentz, PSD), nicht „kaum“.** Die Vormessung nannte 0,13 gegen 0,19 ms je Schritt für Lorentz; sie lief ohne einen BLAS-Thread und ohne Warmlauf der Barriere. Im Bau (ein Thread, nach Warmlauf) sind es 0,06 bis 0,10 ms gegen 0,2 ms (Lorentz) und 0,10 bis 0,16 gegen 0,5 bis 0,9 ms (PSD). Die Gesamtzeit bleibt 2- bis 3-mal länger bei hohem Ziel, weil es das 3- bis 12-fache an Schritten sind.
- **Die Zeitverhältnisse der Vormessung (5,5-, 5-, 32-mal bei 10⁻⁹) waren falsch gewählt.** Dort stand ADMM bei Residuen-Toleranz `10⁻⁹` gegen die Barriere bei Toleranz `10⁻⁹`: verschiedene Größen, und die ℓ₃-Barriere lief im Warmlauf (150 statt 17 ms). Der Bau vergleicht bei **gleichem Wertfehler gegen `R*`** (ADMM bricht ab, sobald Wertfehler und primales Residuum unter dem Ziel liegen; die Barriere nimmt die kleinste Toleranz, deren Wert das Ziel trifft).
- **Der feste Aufwand der Projektion ist ein Befund des Baus.** Ein Aufruf von `proj_exp` kostet bei 1 und bei 32 Kegeln gleich viel, ein bis zwei Millisekunden je nach Rechner (Python-Schleife des Illinois-Verfahrens über kleine numpy-Felder); erst bei Hunderten von Kegeln wird er deutlich teurer. Die 4- bis 25-fache Zeit von ADMM beim Potenz- und Exponentialkegel ist also wahrscheinlich zum großen Teil Interpreter-Aufwand (aus der Unabhängigkeit von der Kegelzahl geschlossen, nicht profiliert); ob eine kompilierte Projektion den Schritt senkt, ist nicht gemessen. Die Aussage der Plan-Tabelle „die Wurzelsuche dominiert“ stimmt, aber nicht wegen der Rechenarbeit.
- **Experiment 3 und 4 rechnen mit Lorentz und PSD statt mit Lorentz und ℓ₃.** Die Vormessung nahm ℓ₃; die Läufe mit festem ρ auf Skala 10 und 100 (je 3 000 Schritte zu 1,6 ms) hätten die Seite um zehn Sekunden verlängert. Die Richtung ist dieselbe (Vormessung, hier nicht getestet: ℓ₃ bei Skala 0,01 bis 100 fest 232, 274, 620, 6 097, nicht gelöst; Ausgleich 250, 285, 298, 895, 2 059).
- **Die Deckel sind kleiner als in der Vormessung** (1 000 statt 20 000 in der ρ-Reihe, 3 000 in der Skalenreihe): ein Kreuz heißt „nicht gelöst nach dem Deckel“, nicht „nie“. PSD bei ρ = 1 braucht 1 066 Schritte und zeigt deshalb in der Reihe ein Kreuz, in der Skalenreihe (Deckel 3 000) 1 066.
- **Der Warmstart wurde gemessen** (die Vormessung nannte nur die Erklärung): Start im groben `x` allein spart 4 bis 6 % (Exponentialkegel: 50 %), mit `u` setzt der Lauf nur fort, wo der grobe aufgehört hat; eine Ersparnis gegenüber einem Lauf, der gleich weiterläuft, gibt es nicht.
- **Die Toleranz ist unter Skala 1 faktisch absolut.** Das Abbruchkriterium vergleicht mit `max(‖A x‖, ‖s‖, ‖b‖, 1)`: bei Daten der Größe 10⁻⁴ hält der Lauf an, obwohl der Wertfehler noch etwa 6·10⁻⁴ beträgt (Randfall-Test); mit dem Maßstab `R*` (Experiment 5) erreicht ADMM auch dort 10⁻⁶.
- **Quellen:** geprüft per Suche am 2026-11-02: Boyd, Parikh, Chu, Peleato, Eckstein (Foundations and Trends in Machine Learning 3 (1), 2011, S. 1–122, DOI 10.1561/2200000016; die Zitierzeile im PDF nennt 2010, die Verlagsangabe 26.7.2011), O'Donoghue, Chu, Parikh, Boyd (Journal of Optimization Theory and Applications 169 (3), 2016, S. 1042–1068, arXiv 1312.3039). **Stellato u. a. (OSQP)** ist nur als arXiv-Preprint (1711.08013) bestätigt; die Zeitschriftenfassung (Mathematical Programming Computation, 2020) ist nicht geprüft und steht deshalb nicht als Zitat. Die Regel μ = 10, τ = 2 des Residuenausgleichs wird in Folgearbeiten dem Kapitel 3.4.1 von Boyd u. a. zugeschrieben; das Kapitel ist nicht nachgelesen, die Demo nennt sie „übliche Wahl“ und misst sie. Inhalte der Arbeiten (Beweise, Raten) nicht nachgelesen.
- **Korrektur nach der unabhängigen Aussagenprüfung (Orthant und Potenzkegel-Tabelle).** `spec_of` legte einen Orthanten der Größe `m` aus dem Kern (`("l", m)`) als `("l", m, 1)` an, `project_spec` liest die Anzahl an dritter Stelle: es wurde nur **eine** Zeile projiziert. Jetzt `("l", None, m)`; Handbeispiel `min x` mit `x ≥ 1`, `x ≥ 2` (ein Orthant der Größe 2): Wert 2 (vorher 1). Die Zahlen dieser README ändert das nicht, weil keines der Programme dieser Demo einen Orthanten der Größe über 1 aus dem Kern enthält (die Barriere-Programme tragen `("l", None, n)`). Die Tabelle der Wurzelsuche des Potenzkegels (`_pow_cross`) lief für `a` ab etwa 1,4 (kein Kegel mehr) mit RuntimeWarnings über; sie rechnet jetzt alle Potenzen als `e^(Exponent − größter Exponent)`, ohne Warnhinweis und mit gleichen Ergebnissen (Test: Unterschied unter 10⁻⁹ zur einfachen Formel in `a ≤ 1`).
- **Korrektur: Exponentialkegel links vom Gitter.** Die Kopie von `proj_exp` hatte den Fehler aus Stück 8: für x/y unter etwa −700 liegt die Wurzel links vom Gitter der Suche, und die Projektion fiel auf die Fläche y = 0 (Abstandsfehler im Test: 2,06 statt 0,5 bei v = (−1500; 2; −0,5)). Der Randpunkt (x, y, y e^(x/y)) ist dort exakt und wird jetzt als Kandidat geprüft, `exp_member` rechnet z = 0 bei y > 0 als 10⁻³⁰⁰. Die Programme dieser Demo erreichen den Bereich nicht; drei Tests in `tests/test_proj_exp_tail.py`.
- **Port:** 9307 (`launch.json` und Netstat gelesen, frei).

## Ehrliche Grenzen

- **Eine Wolke, kleine Programme.** Alle Zahlen gelten für 8 Punkte der Standardwolke und Programme mit 3 bis 35 Variablen. Ob ADMM bei großen Programmen gegen die Barriere gewinnt (dort, wo das Newton-System teuer wird), ist **nicht gemessen**; das ist Stück 10 (Splitting gegen Innere Punkte).
- **Die U-Form, die Lage des Bodens und die Wanderung mit der Skala sind Messbefunde dieser Programme,** keine Aussage über alle Kegelprogramme.
- **Der Residuenausgleich ist eine Faustregel** mit eigenen Parametern: die neun Einstellungen streuen bei Skala 100 um mehr als das Dreifache.
- **Die Zeiten hängen vom Rechner ab** und werden nur in ihrer Struktur getestet (ADMM-Schritt billiger als Newton-Schritt bei Lorentz und PSD, teurer bei Potenz- und Exponentialkegel; Verhältnisse in weiten Bändern). Der feste Aufwand der Wurzelsuche ist wahrscheinlich Interpreter-Aufwand (nicht profiliert).
- **Kein SCS.** Die einfache Form ohne Einbettung erkennt unzulässige und unbeschränkte Programme nicht; der Lauf endet dort am Schrittdeckel (Randfall-Test). `A` braucht vollen Spaltenrang (sonst `ValueError`).
- **Residuen sind relativ zu `max(…, 1)`:** unter Skala 1 ist die Toleranz faktisch absolut (siehe oben).
- **Überrelaxation und Warmstart sind nur auf Lorentz und PSD (bzw. allen vier Kegeln beim Warmstart) bei kleinen ρ gemessen;** ob sie bei anderen Programmen helfen, ist offen.
- **Nur die vier Kugeln,** keine Anwendung.

## Verifikation

- **Verfahren** (`tests/test_admm.py`): das Lorentz-Handprogramm Schritt für Schritt (Schritt 1: `x = 1,5`, `s = (3,25; 1,95; 2,6)`, `u = (1,75; −1,05; −1,4)`, Residuen √6,125 und 0,75; Schritt 2: `x = 4`), sein Fixpunkt (`x = 5`, `s = (5; 3, 4)`, `u = (1; −0,6; −0,8)` nach 80 Schritten, KKT-Bedingungen, ein Start im Fixpunkt mit `u` bleibt dort, ohne `u` bewegt sich `x`), das Orthant-Handprogramm (zwei Schritte bis `x = 1, u = 1`; ein Orthant der Größe 2: `min x` mit `x ≥ 1`, `x ≥ 2` gibt 2), das PSD-Handprogramm, die KKT-Bedingungen auf einer Kugel (`s ∈ K`, `y = ρu ∈ K*`, Komplementarität), Unabhängigkeit des Ergebnisses von ρ und vom Ausgleich, Umrechnung von `u` beim Wechsel von ρ, Skala, Abbruch mit Referenzwert, Verlauf, Ablehnung von `A` ohne vollen Spaltenrang und von ρ ≤ 0, unbeschränktes Programm (Deckel), Zerlegung des Kegels und blockweise Projektion.
- **Projektionen und Formen** (`tests/test_projection.py`, `tests/test_forms.py`): aus Stück 8 übernommen (Handwerte aller fünf Kegel, Moreau-Probe, Idempotenz, Nichtexpansivität, Zertifikat; antipodale Punkte von Hand in allen vier Kugeln).
- **Orakel** (`tests/test_oracle_kegel_admm.py`): `R*` der vier Kugeln gegen cvxpy mit Clarabel (`cp.norm`, `cp.norm(·, 3)`, `cp.log_sum_exp`), die Lorentz-Kugel gegen SCS, die Dualvariable `y = ρu` gegen das Dualprogramm der Kugel.
- **Randfälle** (`tests/test_edge_cases.py`): extremes ρ (10⁻⁴, 10³), Daten der Skala 10⁻⁴ und 10⁴ mit Ausgleich, die absolute Toleranz unter Skala 1, großes `u₀`, Toleranz 0, Wolken aus 2 und 3 Punkten, alle Reglerecken für jeden Kegel, Ausdünnung des Verlaufs, Experimente an der kleinsten Wolke.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: die meisten Zahlen dieser README, Zeiten nur als Struktur), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich, jeder Kegel, jedes ρ, jede Toleranz, Ausgleich, PDF), Presets (die Aussagen der Hilfetexte), Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe, Kern.
- **Umfang:** 189 Tests (4 davon, cvxpy-Orakel, überspringen sich ohne cvxpy; mit cvxpy laufen alle 5 Orakel-Tests grün), rund 33 Minuten (gemessen 2026-11-03, Windows, bei gleichzeitig laufenden anderen Läufen; ohne Last etwa 19; die AppTest-Läufe tragen die Experimente 2 bis 5 bei jeder neuen Wolke); kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz; ohne es überspringen sich die Orakel-Tests. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 8 |
| `kad_admm.py` | ADMM für Kegelprogramme (Residuenausgleich, Überrelaxation, Warmstart), Zerlegung des Kegels |
| `kad_projection.py` | Projektionen auf fünf Kegel (gestapelt), Zugehörigkeit, Moreau-Zertifikat (aus Stück 8) |
| `kad_forms.py` | Die Kugel der Wolke als Lorentz-, PSD-, Potenz- und Exponentialkegel-Programm (aus Stück 6 bis 8) |
| `kad_barrier.py`, `kad_cones.py` | Barrierenmethode über Orthant, Exponential- und Potenzkegel für `R*` und den Vergleich (aus Stück 7) |
| `kad_constants.py` | Regler-Grenzen, feste Reihen der Experimente, Farben, Presets |
| `kad_evaluation.py` | Experimente, Verdikte, Formatierung |
| `kad_layout.py` | Fester Kopf (Stylesheet) |
| `kad_pdf_export.py` | PDF-Export (fpdf2) |
| `kad_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kad_scenario.py` | Punktwolken (gestreckt, gedreht) |
| `kad_visualization.py` | Plotly-Abbildungen (2D-Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Verfahren, Projektionen, Formen, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **Die homogene selbstduale Einbettung (SCS), Zertifikate für Unzulässigkeit und Unbeschränktheit:** würde das Stück sprengen; der Text nennt den Unterschied.
- **Große Programme, dünn besetztes `A`, Faktorisierungen, die sich lohnen:** Stück 10; hier nur der Vergleich im Kleinen.
- **Splitting-Varianten** (Douglas–Rachford, Chambolle–Pock, Beschleunigung, Anderson): nicht gebaut.
- **Ein Rate-Satz für ADMM:** nicht hergeleitet; die Demo misst.
- **Prox-Operatoren anderer Funktionen** (ℓ₁, Rauschfilter): die Inverse-Linie (ISTA/FISTA, Plug-and-Play, „kein ADMM“); nur Verweis.
- **Eine schnellere Projektion** (eingeklammertes Newton-Verfahren, kompiliert) für den Exponential- und Potenzkegel: nur genannt (Stück 8).
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html).
