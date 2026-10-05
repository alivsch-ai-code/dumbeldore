# Eurojackpot-Experiment

Spielerisches Experiment: Eurojackpot-Ziehungen der letzten 2 Jahre werden mit
Zusatzdaten zu einer Datenbank verknüpft und mit TensorFlow angelernt, um Zahlen
für die nächste Ziehung vorzuschlagen.

**Ehrlicher Hinweis:** Ziehungen sind zufällig und unabhängig. Das Modell kann
keine Vorhersage leisten, die besser als Zufall ist. Das Skript vergleicht
deshalb am Ende die Trefferquote des Modells mit zufälligen Tipps.

## Datenquellen

| Merkmal | Quelle |
|---|---|
| Ziehungen (5 aus 50, 2 aus 12) | `data/draws.csv` (Download-Versuch oder manuell) |
| Sonnenaktivität | SILSO Sonnenfleckenzahl (täglich) |
| Planetenabstände (Sonne, Mond, Merkur–Saturn), Mondphase, Winkelabstand der Sonne zu 6 hellen Sternen | astropy |
| Weltereignisse | GDELT (Nachrichtenvolumen und Ton pro Tag, nur Näherung; die API deckt vermutlich keine vollen 2 Jahre ab, Lücken werden 0) |

## Benutzung

```bash
pip install -r requirements.txt
python eurojackpot_experiment.py --build-db
python eurojackpot_experiment.py --train --predict 2026-10-06
```

Falls der automatische Download der Ziehungen scheitert, lege `data/draws.csv` an:

```
date,n1,n2,n3,n4,n5,e1,e2
2026-10-02,3,17,22,41,48,5,9
```
