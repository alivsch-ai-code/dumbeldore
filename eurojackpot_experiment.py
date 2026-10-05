#!/usr/bin/env python3
"""
Eurojackpot-Experiment
======================
Baut eine Datenbank aus den Eurojackpot-Ziehungen der letzten 2 Jahre plus
Zusatzmerkmalen (Sonnenaktivität, Planetenabstände, Weltereignisse) und
trainiert ein kleines TensorFlow-Modell, das Zahlen für die nächste Ziehung
vorschlägt.

WICHTIG: Eurojackpot-Ziehungen sind zufällig und unabhängig. Kein Modell
kann daraus etwas lernen, das über Zufall hinausgeht. Das Skript ist ein
Experiment, keine Gewinnstrategie.

Benutzung:
    pip install -r requirements.txt
    python eurojackpot_experiment.py --build-db
    python eurojackpot_experiment.py --train --predict 2026-10-06

Wenn der automatische Download der Ziehungen nicht klappt, lege eine CSV
unter data/draws.csv ab mit den Spalten:
    date,n1,n2,n3,n4,n5,e1,e2
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
DRAWS_CSV = DATA_DIR / "draws.csv"
DB_CSV = DATA_DIR / "database.csv"
MODEL_PATH = DATA_DIR / "model.keras"

MAIN_MAX, MAIN_PICK = 50, 5
EURO_MAX, EURO_PICK = 12, 2
YEARS_BACK = 2
WINDOW = 8  # so viele vorherige Ziehungen sieht das Modell als Kontext

# Quellen -------------------------------------------------------------------
SILSO_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# Mögliche CSV-Quellen für Ziehungen (Format kann sich ändern, deshalb auch
# manueller Fallback über data/draws.csv)
DRAW_SOURCES = [
    "https://www.lotto.de/api/stats/entities.eurojackpot/draws.csv",
]


# ---------------------------------------------------------------------------
# 1. Ziehungen
# ---------------------------------------------------------------------------
def load_draws() -> pd.DataFrame:
    """Lädt Ziehungen aus data/draws.csv oder versucht einen Download."""
    DATA_DIR.mkdir(exist_ok=True)
    if DRAWS_CSV.exists():
        df = pd.read_csv(DRAWS_CSV, parse_dates=["date"])
    else:
        df = _download_draws()
        if df is None:
            sys.exit(
                "Konnte Ziehungen nicht laden. Bitte data/draws.csv anlegen "
                "(Spalten: date,n1..n5,e1,e2)."
            )
        df.to_csv(DRAWS_CSV, index=False)
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=YEARS_BACK)
    df = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)
    print(f"{len(df)} Ziehungen seit {cutoff.date()} geladen.")
    return df


def _download_draws() -> pd.DataFrame | None:
    cols = ["date", "n1", "n2", "n3", "n4", "n5", "e1", "e2"]
    for url in DRAW_SOURCES:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.text), sep=None, engine="python")
            # Heuristik: erste Spalte Datum, danach 7 Zahlenspalten
            raw = raw.iloc[:, :8]
            raw.columns = cols
            raw["date"] = pd.to_datetime(raw["date"], dayfirst=True)
            return raw
        except Exception as exc:  # noqa: BLE001
            print(f"Quelle {url} fehlgeschlagen: {exc}")
    return None


# ---------------------------------------------------------------------------
# 2. Sonnenaktivität (SILSO Sonnenfleckenzahl)
# ---------------------------------------------------------------------------
def load_sunspots() -> pd.Series:
    try:
        r = requests.get(SILSO_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(
            io.StringIO(r.text), sep=";", header=None,
            names=["y", "m", "d", "frac", "ssn", "std", "obs", "prov"],
        )
        df["date"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
        s = df.set_index("date")["ssn"].replace(-1, np.nan)
        return s.interpolate()
    except Exception as exc:  # noqa: BLE001
        print(f"Sonnenfleckendaten nicht verfügbar ({exc}) -> 0")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 3. Planetenabstände (Erde -> Planeten, Sonne, Mond) via astropy
# ---------------------------------------------------------------------------
def planet_distances(dates: pd.Series) -> pd.DataFrame:
    from astropy.coordinates import get_body
    from astropy.time import Time
    import astropy.units as u

    bodies = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]
    times = Time([d.to_pydatetime().replace(hour=19, minute=0) for d in dates])
    out = {}
    for b in bodies:
        coord = get_body(b, times)
        out[f"dist_{b}_au"] = coord.distance.to(u.au).value
    return pd.DataFrame(out, index=dates.index)


# ---------------------------------------------------------------------------
# 4. Weltereignisse (GDELT: Nachrichtenvolumen + durchschnittlicher Ton)
# ---------------------------------------------------------------------------
def gdelt_series(mode: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """mode: 'timelinevol' (Volumen) oder 'timelinetone' (Stimmung)."""
    params = {
        "query": "sourcelang:english",
        "mode": mode,
        "format": "json",
        "timelinesmooth": 0,
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    try:
        r = requests.get(GDELT_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["timeline"][0]["data"]
        s = pd.Series(
            {pd.to_datetime(p["date"]).normalize(): p["value"] for p in data}
        )
        return s.groupby(level=0).mean()
    except Exception as exc:  # noqa: BLE001
        print(f"GDELT {mode} nicht verfügbar ({exc}) -> 0")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 5. Datenbank bauen
# ---------------------------------------------------------------------------
def build_database() -> pd.DataFrame:
    draws = load_draws()
    start, end = draws["date"].min(), draws["date"].max()

    ssn = load_sunspots()
    draws["sunspots"] = draws["date"].map(ssn).fillna(0.0)

    print("Berechne Planetenabstände ...")
    draws = draws.join(planet_distances(draws["date"]))

    print("Lade Weltereignis-Daten (GDELT) ...")
    vol = gdelt_series("timelinevol", start, end)
    tone = gdelt_series("timelinetone", start, end)
    draws["news_volume"] = draws["date"].map(vol).fillna(0.0)
    draws["news_tone"] = draws["date"].map(tone).fillna(0.0)

    draws.to_csv(DB_CSV, index=False)
    print(f"Datenbank gespeichert: {DB_CSV} ({len(draws)} Zeilen)")
    return draws


# ---------------------------------------------------------------------------
# 6. Modell
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "sunspots", "dist_sun_au", "dist_moon_au", "dist_mercury_au",
    "dist_venus_au", "dist_mars_au", "dist_jupiter_au", "dist_saturn_au",
    "news_volume", "news_tone",
]


def multi_hot(row: pd.Series) -> np.ndarray:
    v = np.zeros(MAIN_MAX + EURO_MAX, dtype="float32")
    for c in ["n1", "n2", "n3", "n4", "n5"]:
        v[int(row[c]) - 1] = 1
    for c in ["e1", "e2"]:
        v[MAIN_MAX + int(row[c]) - 1] = 1
    return v


def make_dataset(db: pd.DataFrame):
    feats = db[FEATURE_COLS].astype("float32")
    feats = (feats - feats.mean()) / (feats.std().replace(0, 1))
    labels = np.stack([multi_hot(r) for _, r in db.iterrows()])
    X, y = [], []
    for i in range(WINDOW, len(db)):
        past = labels[i - WINDOW:i].reshape(-1)
        X.append(np.concatenate([past, feats.iloc[i].values]))
        y.append(labels[i])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), feats, labels


def build_model(input_dim: int):
    import tensorflow as tf

    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(MAIN_MAX + EURO_MAX, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


def train(db: pd.DataFrame):
    X, y, *_ = make_dataset(db)
    split = int(len(X) * 0.85)
    model = build_model(X.shape[1])
    model.fit(
        X[:split], y[:split], validation_data=(X[split:], y[split:]),
        epochs=60, batch_size=16, verbose=2,
    )
    model.save(MODEL_PATH)
    print(f"Modell gespeichert: {MODEL_PATH}")

    # Ehrlicher Vergleich: Trefferquote Modell vs. Zufall auf Testdaten
    import tensorflow as tf  # noqa: F401

    pred = model.predict(X[split:], verbose=0)
    hits_model, hits_random = [], []
    rng = np.random.default_rng(0)
    for p, truth in zip(pred, y[split:]):
        top = _pick(p)
        rnd = _pick(rng.random(MAIN_MAX + EURO_MAX))
        hits_model.append(_hits(top, truth))
        hits_random.append(_hits(rnd, truth))
    print(
        f"Ø richtige Zahlen im Test – Modell: {np.mean(hits_model):.2f} | "
        f"Zufall: {np.mean(hits_random):.2f}  (erwartet ~0.7 bei Zufall)"
    )
    return model


def _pick(probs: np.ndarray, temperature: float = 0.0, rng=None):
    main, euro = probs[:MAIN_MAX].copy(), probs[MAIN_MAX:].copy()
    if temperature > 0:
        rng = rng or np.random.default_rng()
        main = main + rng.gumbel(size=main.shape) * temperature
        euro = euro + rng.gumbel(size=euro.shape) * temperature
    m = sorted((np.argsort(main)[-MAIN_PICK:] + 1).tolist())
    e = sorted((np.argsort(euro)[-EURO_PICK:] + 1).tolist())
    return m, e


def _hits(picked, truth: np.ndarray) -> int:
    m, e = picked
    return int(sum(truth[n - 1] for n in m) + sum(truth[MAIN_MAX + n - 1] for n in e))


def predict(db: pd.DataFrame, target_date: str, n_tickets: int = 3):
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, feats, labels = make_dataset(db)

    # Features für das Zieldatum berechnen
    target = pd.Timestamp(target_date)
    row = pd.DataFrame({"date": [target]})
    row = row.join(planet_distances(row["date"]))
    ssn = load_sunspots()
    row["sunspots"] = float(ssn.iloc[-1]) if len(ssn) else 0.0
    row["news_volume"] = db["news_volume"].iloc[-7:].mean()
    row["news_tone"] = db["news_tone"].iloc[-7:].mean()

    mean = db[FEATURE_COLS].mean()
    std = db[FEATURE_COLS].std().replace(0, 1)
    f = ((row[FEATURE_COLS] - mean) / std).astype("float32").values[0]

    past = labels[-WINDOW:].reshape(-1)
    x = np.concatenate([past, f]).astype("float32")[None, :]
    probs = model.predict(x, verbose=0)[0]

    print(f"\nVorschläge für die Ziehung am {target.date()}:")
    rng = np.random.default_rng(int(target.strftime("%Y%m%d")))
    for i in range(n_tickets):
        m, e = _pick(probs, temperature=0.0 if i == 0 else 0.02, rng=rng)
        print(f"  Tipp {i + 1}: {m}  Eurozahlen: {e}")
    print("\nHinweis: Gewinnchance je Tipp bleibt 1 : 140.000.000 (ca.).")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-db", action="store_true", help="Datenbank bauen")
    ap.add_argument("--train", action="store_true", help="Modell trainieren")
    ap.add_argument("--predict", metavar="YYYY-MM-DD", help="Zahlen für Datum")
    ap.add_argument("--tickets", type=int, default=3)
    args = ap.parse_args()

    if not (args.build_db or args.train or args.predict):
        ap.print_help()
        return

    db = build_database() if args.build_db or not DB_CSV.exists() else \
        pd.read_csv(DB_CSV, parse_dates=["date"])
    if args.train:
        train(db)
    if args.predict:
        predict(db, args.predict, args.tickets)


if __name__ == "__main__":
    main()
