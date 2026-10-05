# Eurojackpot-Experiment

Spielerisches Experiment: Eurojackpot-Ziehungen der letzten 18 Monate werden mit
Zusatzdaten zu einer Datenbank verknüpft und mit TensorFlow angelernt, um Zahlen
für die nächste Ziehung vorzuschlagen.

**Ehrlicher Hinweis:** Ziehungen sind zufällig und unabhängig. Das Modell kann
keine Vorhersage leisten, die besser als Zufall ist. Das Skript vergleicht
deshalb am Ende die Trefferquote auf ungesehenen Ziehungen mit dem Zufallswert.

## Merkmale pro Ziehung

| Gruppe | Inhalt | Quelle |
|---|---|---|
| Ziehungen | letzte 4 Ziehungen, Häufigkeit jeder Zahl (20 Ziehungen), Ziehungen seit dem letzten Auftreten | `data/draws.csv`, Stand 02.10.2026 (Archiv [dev-baris/lottery-archive](https://github.com/dev-baris/lottery-archive), `--refresh` lädt neu) |
| Weltraumwetter | Sonnenflecken (Tages- und 7-Tage-Wert), geomagnetischer Ap-Index, Radiofluss F10.7 | SILSO, GFZ Potsdam (jeweils Wert vom Vortag) |
| Astronomie | Abstand zu Sonne, Mond, Merkur–Saturn; Stellung am Himmel; Mondphase; Winkelabstand der Sonne zu 6 hellen Sternen | astropy |
| Weltereignisse | Anteil an der Berichterstattung und Stimmung zu Konflikt, Wirtschaft, Katastrophen, Politik, Sport (3-Tage-Mittel vor der Ziehung) | GDELT |
| Kalender | Dienstag/Freitag, Monat und Jahrestag als Sinus/Kosinus | berechnet |

Nicht verfügbare Quellen werden übersprungen, das Training läuft mit den
übrigen Merkmalen.

## Benutzung

```bash
pip install -r requirements.txt
python eurojackpot_experiment.py --refresh --build-db --train --predict 2026-10-06
```

Das Abrufen der GDELT-Daten dauert einige Minuten. Falls der Download der
Ziehungen nicht klappt, `data/draws.csv` selbst anlegen:

```
date,n1,n2,n3,n4,n5,e1,e2
2026-10-02,4,6,7,17,45,7,12
```

Datum als `2026-10-02` oder `02.10.2026`. Die Datei wird auf gültige Zahlen geprüft.

## Auswendig lernen (Experiment)

```bash
python eurojackpot_experiment.py --build-db --train --overfit --target-hits 6 --predict 2026-10-06
```

Trainiert ein großes Modell ohne Bremse, bis die bekannten Ziehungen im Schnitt
`--target-hits` von 7 richtig getippt werden. Die letzten 15 % der Ziehungen
bleiben ungesehen, damit der Testvergleich zeigt, ob das für neue Ziehungen etwas bringt.

## Tipps protokollieren und auswerten

Jedes `--predict` speichert die Tipps in `data/tips_log.csv`. Nach der Ziehung:

```bash
python eurojackpot_experiment.py --refresh --evaluate
```

Das vergleicht alle Tipps mit den echten Ziehungen und zeigt die Ø richtigen
Zahlen pro Tipp (getrennt nach normal/overfit) gegen den Zufallswert 0,83.

## Rückwärts-Test mit Signifikanzprüfung

```bash
python eurojackpot_experiment.py --refresh --backtest
```

Walk-Forward über alle 473 Ziehungen seit der Regeländerung im März 2022
(12 Eurozahlen, zusätzlich Dienstag): Das Modell wird alle 20 Ziehungen nur mit
früheren Ziehungen neu trainiert und auf den folgenden getestet (323 Testziehungen).
Verglichen wird mit „heißen“ und „überfälligen“ Zahlen sowie Zufallstipps, mit
Permutationstest (p-Wert). Die Weltereignisse (GDELT) fehlen hier, weil sie nicht
so weit zurückreichen.

## Abgelegte Modelle

`models/2026-10-05/` enthält die am 05.10.2026 trainierten Modelle (normal und
`--overfit`), ihre Merkmals-Metadaten, die Datenbank (ohne Sonnenflecken, Geomagnetik
und GDELT, die beim Training nicht erreichbar waren) und die Tipps für die Ziehung
am 06.10.2026 (`tips_log.csv`). Zum Weiterverwenden die Dateien nach `data/` kopieren.

## Datenquellen prüfen

```bash
python eurojackpot_experiment.py --check-sources
```

Testet Ziehungen, SILSO, GFZ (Ausweichquelle CelesTrak) und GDELT und zeigt je
einen Beispielwert. Vor dem ersten Training laufen lassen: Was nicht erreichbar
ist, wird beim Training übersprungen.

## Tipps für Dienstag, 06.10.2026

Aus den Modellen in `models/2026-10-05/` (ohne Sonnenflecken, Geomagnetik und GDELT trainiert):

| Modell | Zahlen | Eurozahlen |
|---|---|---|
| normal | 8, 21, 25, 40, 46 | 8, 9 |
| normal | 8, 21, 40, 41, 46 | 9, 11 |
| normal | 3, 8, 21, 40, 46 | 8, 11 |
| overfit | 9, 24, 25, 41, 50 | 8, 9 |
| overfit | 9, 24, 35, 41, 50 | 3, 9 |
| overfit | 9, 24, 25, 41, 50 | 8, 9 |

Rückwärts-Test auf 323 Ziehungen: Modell 0,805 richtige Zahlen pro Tipp, Zufall 0,831 –
kein Vorteil. Auswerten nach der Ziehung mit `--refresh --evaluate`.

## Overfit mit 7 von 7 (06.10.2026)

`models/2026-10-05-7von7/`: Modell, das alle bekannten Ziehungen auswendig kennt
(7,00 von 7 richtigen Zahlen nach 20 Durchläufen; auf 24 ungesehenen Ziehungen 1,08, Zufall 0,83).
Tipp für 06.10.2026: 22, 24, 25, 35, 50 / Eurozahlen 2, 9.

## Auswahl auf zurückgehaltenen Ziehungen (`--min-hits`)

```bash
python eurojackpot_experiment.py --min-hits 2 --holdout 4 --predict 2026-10-06
```

Trainiert Overfit-Modelle mit wechselndem Startwert ohne die letzten `--holdout`
Ziehungen, bis eines dort im Schnitt `--min-hits` von 7 trifft. Das ist eine
Auswahl auf genau diesen Ziehungen und sagt nichts über neue Ziehungen. Das Skript
zeigt die Zufallswahrscheinlichkeit pro Versuch (bei 2 von 7 auf 4 Ziehungen: 0,90 %).
