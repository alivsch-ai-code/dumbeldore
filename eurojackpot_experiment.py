#!/usr/bin/env python3
"""
Eurojackpot-Experiment
======================
Baut eine Datenbank aus den Eurojackpot-Ziehungen der letzten 18 Monate plus
Zusatzmerkmalen (Weltraumwetter, Planeten- und Sternstellungen, Kalender,
Weltereignisse via GDELT) und trainiert ein kleines TensorFlow-Modell, das
Zahlen für die nächste Ziehung vorschlägt.

WICHTIG: Eurojackpot-Ziehungen sind zufällig und unabhängig. Kein Modell
kann daraus etwas lernen, das über Zufall hinausgeht. Das Skript ist ein
Experiment, keine Gewinnstrategie. Am Ende des Trainings wird die
Trefferquote des Modells mit dem Zufallswert verglichen.

Benutzung:
    pip install -r requirements.txt
    python eurojackpot_experiment.py --refresh --build-db --train --predict 2026-10-06
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
DRAWS_CSV = DATA_DIR / "draws.csv"
DB_CSV = DATA_DIR / "database.csv"
MODEL_PATH = DATA_DIR / "model.keras"
META_PATH = DATA_DIR / "model_meta.json"

MAIN_MAX, MAIN_PICK = 50, 5
EURO_MAX, EURO_PICK = 12, 2
N_OUT = MAIN_MAX + EURO_MAX
ID_COLS = ["date", "n1", "n2", "n3", "n4", "n5", "e1", "e2"]

MONTHS_BACK = 18   # Trainingszeitraum
WINDOW = 4         # so viele vorherige Ziehungen sieht das Modell direkt
FREQ_WIN = 20      # Häufigkeit jeder Zahl in den letzten N Ziehungen
GAP_CAP = 40       # "Ziehungen seit letztem Auftreten", gedeckelt

# Quellen -------------------------------------------------------------------
DRAW_SOURCES = [
    # Öffentliches Archiv (täglich aktualisiert), Spalten: date,n1..n5,e1,e2
    "https://raw.githubusercontent.com/dev-baris/lottery-archive/main/eu/eurojackpot/results.csv",
]
SILSO_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
GFZ_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT-Themen: Name -> (Suchanfrage, auch Stimmung abfragen?)
NEWS_TOPICS = {
    "all": ("sourcelang:english", True),
    "conflict": ("(war OR attack OR military OR missile) sourcelang:english", True),
    "economy": ("(economy OR inflation OR stocks OR recession) sourcelang:english", True),
    "disaster": ("(earthquake OR flood OR hurricane OR wildfire) sourcelang:english", False),
    "politics": ("(election OR president OR parliament) sourcelang:english", False),
    "sport": ("(football OR olympics OR championship) sourcelang:english", False),
}

# Helle Sterne (RA, Dec in Grad, ICRS)
STARS = {
    "sirius": (101.287, -16.716),
    "betelgeuse": (88.793, 7.407),
    "aldebaran": (68.980, 16.509),
    "regulus": (152.093, 11.967),
    "spica": (201.298, -11.161),
    "antares": (247.352, -26.432),
}


# ---------------------------------------------------------------------------
# 1. Ziehungen
# ---------------------------------------------------------------------------
def _valid_draws(df: pd.DataFrame) -> bool:
    """Prüft: 5 verschiedene Zahlen 1-50 und 2 verschiedene Eurozahlen 1-12."""
    try:
        main = df[["n1", "n2", "n3", "n4", "n5"]].astype(int).values
        euro = df[["e1", "e2"]].astype(int).values
    except (KeyError, ValueError):
        return False
    return bool(
        len(df) > 0
        and main.min() >= 1 and main.max() <= MAIN_MAX
        and euro.min() >= 1 and euro.max() <= EURO_MAX
        and all(len(set(r)) == MAIN_PICK for r in main)
        and all(len(set(r)) == EURO_PICK for r in euro)
    )


def _download_draws() -> pd.DataFrame | None:
    for url in DRAW_SOURCES:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.text), sep=None, engine="python")
            raw = raw.iloc[:, :8]
            raw.columns = ID_COLS
            raw["date"] = pd.to_datetime(raw["date"], dayfirst=True)
            if not _valid_draws(raw):
                print(f"Quelle {url}: Format passt nicht, verworfen.")
                continue
            return raw
        except Exception as exc:  # noqa: BLE001
            print(f"Quelle {url} fehlgeschlagen: {exc}")
    return None


def load_all_draws() -> pd.DataFrame:
    """Alle Ziehungen (volle Historie) aus data/draws.csv, sonst Download."""
    DATA_DIR.mkdir(exist_ok=True)
    if DRAWS_CSV.exists():
        df = pd.read_csv(DRAWS_CSV)
        # Datum robust lesen: 02.10.2026 (deutsch) oder 2026-10-02 (ISO)
        first = str(df["date"].iloc[0])
        df["date"] = pd.to_datetime(df["date"], dayfirst="." in first)
        if not _valid_draws(df):
            sys.exit(
                "data/draws.csv ist ungültig: erwartet date,n1..n5 (1-50, "
                "verschieden),e1,e2 (1-12, verschieden)."
            )
    else:
        df = _download_draws()
        if df is None:
            sys.exit(
                "Konnte Ziehungen nicht laden. Bitte data/draws.csv anlegen "
                "(Spalten: date,n1..n5,e1,e2)."
            )
        df.to_csv(DRAWS_CSV, index=False)
    return df.sort_values("date").reset_index(drop=True)


def load_draws() -> pd.DataFrame:
    """Ziehungen der letzten MONTHS_BACK Monate (Trainingszeitraum)."""
    df = load_all_draws()
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=MONTHS_BACK)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    print(f"{len(df)} Ziehungen seit {cutoff.date()} geladen.")
    return df


# ---------------------------------------------------------------------------
# 2. Weltraumwetter: Sonnenflecken (SILSO), geomagnetischer Ap-Index und
#    Radiofluss F10.7 (GFZ Potsdam)
# ---------------------------------------------------------------------------
def load_space_weather() -> pd.DataFrame:
    """Tagesreihen: sunspots, ap, f107 (Index = Datum). Leer, wenn nicht ladbar."""
    series = {}
    try:
        r = requests.get(SILSO_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(
            io.StringIO(r.text), sep=";", header=None,
            names=["y", "m", "d", "frac", "ssn", "std", "obs", "prov"],
        )
        df["date"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d))
        series["sunspots"] = df.set_index("date")["ssn"].replace(-1, np.nan)
    except Exception as exc:  # noqa: BLE001
        print(f"Sonnenflecken (SILSO) nicht verfügbar: {exc}")

    try:
        cols = (["yyyy", "mm", "dd", "days", "days_m", "bsr", "dbn"]
                + [f"kp{i}" for i in range(1, 9)]
                + [f"ap{i}" for i in range(1, 9)]
                + ["ap_daily", "sn", "f107obs", "f107adj", "d"])
        r = requests.get(GFZ_URL, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=r"\s+", comment="#",
                         header=None, names=cols)
        df["date"] = pd.to_datetime(
            dict(year=df.yyyy, month=df.mm, day=df.dd))
        df = df[df["date"] >= pd.Timestamp.today() - pd.DateOffset(years=4)]
        df = df.set_index("date")
        series["ap"] = df["ap_daily"].where(df["ap_daily"] >= 0)
        series["f107"] = df["f107obs"].where(df["f107obs"] > 0)
    except Exception as exc:  # noqa: BLE001
        print(f"Geomagnetik/F10.7 (GFZ) nicht verfügbar: {exc}")

    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index().interpolate(limit_area="inside")


def space_features(dates: pd.Series, space: pd.DataFrame) -> pd.DataFrame:
    """Werte vom Vortag (am Ziehungstag selbst ist der Tag noch nicht
    abgeschlossen); liegt kein Wert vor, gilt der zuletzt bekannte."""
    out = pd.DataFrame(index=dates.index)
    if space.empty:
        return out
    ext = space.copy()
    if "sunspots" in ext:
        ext["sunspots_7d"] = ext["sunspots"].rolling(7, min_periods=1).mean()
    if "ap" in ext:
        ext["ap_3d"] = ext["ap"].rolling(3, min_periods=1).mean()
    prev = pd.DatetimeIndex(dates) - pd.Timedelta(days=1)
    full_idx = ext.index.union(prev)
    ext = ext.reindex(full_idx).ffill()
    vals = ext.loc[prev]
    vals.index = dates.index
    return vals


# ---------------------------------------------------------------------------
# 3. Astronomie (astropy): Abstände, Stellungen, Mondphase, Sternabstände
# ---------------------------------------------------------------------------
def astro_features(dates: pd.Series) -> pd.DataFrame:
    from astropy.coordinates import (GeocentricMeanEcliptic, SkyCoord,
                                     get_body)
    from astropy.time import Time
    import astropy.units as u

    bodies = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]
    times = Time([d.to_pydatetime().replace(hour=18, minute=0) for d in dates])
    out = {}
    coords = {}
    for b in bodies:
        coords[b] = get_body(b, times)
        out[f"dist_{b}_au"] = coords[b].distance.to(u.au).value
        # Stellung am Himmel (ekliptikale Länge) als Sinus/Kosinus
        lon = coords[b].transform_to(GeocentricMeanEcliptic(equinox=times)).lon.rad
        out[f"lon_{b}_sin"] = np.sin(lon)
        out[f"lon_{b}_cos"] = np.cos(lon)

    # Mondphase: Winkel Sonne-Mond (0 = Neumond, 180 = Vollmond)
    out["moon_phase_deg"] = coords["sun"].separation(coords["moon"]).deg

    # Sterne: echte Entfernungen sind praktisch konstant, veränderlich ist
    # ihre Stellung relativ zur Sonne -> Winkelabstand als Merkmal.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (ra, dec) in STARS.items():
            star = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            out[f"sep_sun_{name}_deg"] = coords["sun"].separation(star).deg
    return pd.DataFrame(out, index=dates.index)


# ---------------------------------------------------------------------------
# 4. Kalender
# ---------------------------------------------------------------------------
def calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    doy = d.dt.dayofyear.values
    month = d.dt.month.values
    return pd.DataFrame({
        "is_tuesday": (d.dt.weekday == 1).astype(float).values,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "doy_sin": np.sin(2 * np.pi * doy / 366),
        "doy_cos": np.cos(2 * np.pi * doy / 366),
    }, index=dates.index)


# ---------------------------------------------------------------------------
# 5. Weltereignisse (GDELT: Anteil an der Berichterstattung + Stimmung)
# ---------------------------------------------------------------------------
def _gdelt_chunk(query: str, mode: str, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.Series | None:
    params = {
        "query": query, "mode": mode, "format": "json", "timelinesmooth": 0,
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    for attempt in range(2):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()["timeline"][0]["data"]
            idx = pd.to_datetime([p["date"] for p in data], utc=True)
            s = pd.Series([p["value"] for p in data],
                          index=idx.tz_localize(None).normalize())
            return s.groupby(level=0).mean()
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return None


def news_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Tagesreihen je Thema: news_<thema>_vol (Anteil %) und _tone."""
    cols: dict[str, pd.Series] = {}
    failures = 0
    for topic, (query, with_tone) in NEWS_TOPICS.items():
        modes = [("timelinevol", "vol")] + ([("timelinetone", "tone")] if with_tone else [])
        for mode, short in modes:
            parts = []
            cur = start
            while cur <= end:
                chunk_end = min(cur + pd.Timedelta(days=89), end)
                s = _gdelt_chunk(query, mode, cur, chunk_end)
                if s is not None:
                    parts.append(s)
                    failures = 0
                else:
                    failures += 1
                    if failures >= 3 and not cols:
                        print("GDELT nicht erreichbar -> Weltereignis-Merkmale entfallen.")
                        return pd.DataFrame()
                cur = chunk_end + pd.Timedelta(days=1)
                time.sleep(1.0)  # GDELT bittet um Zurückhaltung
            if parts:
                cols[f"news_{topic}_{short}"] = pd.concat(parts).groupby(level=0).mean()
    if not cols:
        print("Keine GDELT-Daten verfügbar -> Weltereignis-Merkmale entfallen.")
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def news_features(dates: pd.Series, daily: pd.DataFrame) -> pd.DataFrame:
    """Mittel der 3 Tage vor dem Ziehungstag."""
    out = pd.DataFrame(index=dates.index)
    if daily.empty:
        return out
    daily = daily.reindex(
        pd.date_range(daily.index.min(), max(daily.index.max(),
                                             dates.max()), freq="D"))
    roll = daily.rolling(3, min_periods=1).mean()
    prev = pd.DatetimeIndex(dates) - pd.Timedelta(days=1)
    vals = roll.reindex(roll.index.union(prev)).ffill().loc[prev]
    vals.index = dates.index
    vals.columns = [f"{c}_3d" for c in vals.columns]
    return vals


# ---------------------------------------------------------------------------
# 6. Datenbank bauen
# ---------------------------------------------------------------------------
def build_database() -> pd.DataFrame:
    draws = load_draws()
    dates = draws["date"]
    start, end = dates.min(), dates.max()

    print("Lade Weltraumwetter (Sonne, Geomagnetik) ...")
    space = load_space_weather()
    print("Berechne Planeten-, Mond- und Sternstellungen ...")
    astro = astro_features(dates)
    cal = calendar_features(dates)
    print("Lade Weltereignis-Daten (GDELT, kann einige Minuten dauern) ...")
    news = news_daily(start - pd.Timedelta(days=4), end)

    db = pd.concat([draws, cal, astro, space_features(dates, space),
                    news_features(dates, news)], axis=1)
    db.to_csv(DB_CSV, index=False)
    feats = [c for c in db.columns if c not in ID_COLS]
    print(f"Datenbank gespeichert: {DB_CSV} ({len(db)} Zeilen, "
          f"{len(feats)} Merkmale)")
    return db


# ---------------------------------------------------------------------------
# 7. Datensatz und Modell
# ---------------------------------------------------------------------------
def multi_hot_matrix(df: pd.DataFrame) -> np.ndarray:
    L = np.zeros((len(df), N_OUT), dtype="float32")
    rows = np.arange(len(df))
    for c in ["n1", "n2", "n3", "n4", "n5"]:
        L[rows, df[c].astype(int).values - 1] = 1
    for c in ["e1", "e2"]:
        L[rows, MAIN_MAX + df[c].astype(int).values - 1] = 1
    return L


def history_features(L: np.ndarray, i: int) -> np.ndarray:
    """Aus den Ziehungen VOR Index i: letzte Ziehungen, Häufigkeit, Abstand."""
    past = L[max(0, i - WINDOW):i]
    if len(past) < WINDOW:
        past = np.vstack([np.zeros((WINDOW - len(past), N_OUT), "float32"), past])
    freq = L[max(0, i - FREQ_WIN):i].mean(axis=0) if i > 0 else np.zeros(N_OUT)
    gap = np.full(N_OUT, GAP_CAP, dtype="float32")
    for k in range(1, min(GAP_CAP, i) + 1):
        hit = (L[i - k] == 1) & (gap == GAP_CAP)
        gap[hit] = k - 1
    return np.concatenate([past.reshape(-1), freq, gap / GAP_CAP]).astype("float32")


def feature_columns(db: pd.DataFrame) -> list[str]:
    """Nur Merkmale mit genug Werten und echter Streuung."""
    cols = []
    for c in db.columns:
        if c in ID_COLS:
            continue
        if db[c].notna().mean() >= 0.8 and db[c].std() > 0:
            cols.append(c)
    return cols


def make_dataset(db: pd.DataFrame, meta: dict, full: pd.DataFrame, L: np.ndarray):
    idx_by_date = {d: i for i, d in enumerate(full["date"])}
    cols, mean, std = meta["cols"], pd.Series(meta["mean"]), pd.Series(meta["std"])
    feats = ((db[cols].fillna(mean) - mean) / std).astype("float32").values
    X, y = [], []
    for r, d in enumerate(db["date"]):
        i = idx_by_date[d]
        X.append(np.concatenate([history_features(L, i), feats[r]]))
        y.append(L[i])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def build_model(input_dim: int, overfit: bool = False):
    import tensorflow as tf

    if overfit:  # groß, ohne Dropout/Regularisierung -> darf auswendig lernen
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(N_OUT, activation="sigmoid"),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy")
        return m
    reg = tf.keras.regularizers.l2(1e-3)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(N_OUT, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m


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


def train(db: pd.DataFrame, overfit: bool = False, target_hits: float = 5.0,
          max_epochs: int = 3000):
    import tensorflow as tf

    full = load_all_draws()
    L = multi_hot_matrix(full)

    cols = feature_columns(db)
    dropped = [c for c in db.columns if c not in ID_COLS and c not in cols]
    if dropped:
        print(f"Merkmale ohne brauchbare Werte entfallen: {', '.join(dropped)}")
    meta = {
        "cols": cols,
        "mean": db[cols].mean().to_dict(),
        "std": db[cols].std().replace(0, 1).to_dict(),
    }
    X, y = make_dataset(db, meta, full, L)

    n = len(X)
    a, b = int(n * 0.70), int(n * 0.85)   # chronologisch: Train / Val / Test
    model = build_model(X.shape[1], overfit)
    if overfit:
        # Trainieren, bis die Zahlen der bekannten Ziehungen (Train+Val)
        # im Schnitt >= target_hits richtig getippt werden. Die letzten 15 %
        # bleiben ungesehen, damit man sieht, ob das etwas bringt.
        class StopAtHits(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10:
                    return
                p = self.model.predict(X[:b], verbose=0)
                h = np.mean([_hits(_pick(q), t) for q, t in zip(p, y[:b])])
                print(f"Epoche {epoch + 1}: Ø richtige Zahlen auf bekannten "
                      f"Ziehungen {h:.2f} von 7")
                if h >= target_hits:
                    self.model.stop_training = True

        model.fit(X[:b], y[:b], epochs=max_epochs, batch_size=16, verbose=0,
                  callbacks=[StopAtHits()])
    else:
        model.fit(
            X[:a], y[:a], validation_data=(X[a:b], y[a:b]),
            epochs=80, batch_size=16, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True)],
        )
    model.save(MODEL_PATH)
    META_PATH.write_text(json.dumps(meta))
    print(f"Modell gespeichert: {MODEL_PATH} ({len(cols)} Merkmale)")

    # Ehrlicher Vergleich auf ungesehenen Testziehungen
    pred = model.predict(X[b:], verbose=0)
    rng = np.random.default_rng(0)
    hits_model = [_hits(_pick(p), t) for p, t in zip(pred, y[b:])]
    hits_random = [np.mean([_hits(_pick(rng.random(N_OUT)), t) for _ in range(50)])
                   for t in y[b:]]
    theory = MAIN_PICK * MAIN_PICK / MAIN_MAX + EURO_PICK * EURO_PICK / EURO_MAX
    print(
        f"Ø richtige Zahlen auf {n - b} Testziehungen – Modell: "
        f"{np.mean(hits_model):.2f} | Zufall: {np.mean(hits_random):.2f} "
        f"(theoretisch {theory:.2f})"
    )
    return model


def predict(target_date: str, n_tickets: int = 3):
    import tensorflow as tf

    if not (MODEL_PATH.exists() and META_PATH.exists()):
        sys.exit("Kein Modell gefunden. Erst mit --train trainieren.")
    model = tf.keras.models.load_model(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    full = load_all_draws()
    L = multi_hot_matrix(full)

    target = pd.Timestamp(target_date)
    if target.weekday() not in (1, 4):  # Dienstag, Freitag
        print("Achtung: Eurojackpot wird dienstags und freitags gezogen.")
    last = full["date"].max()
    if (target - last).days > 4:
        print(f"Achtung: Letzte Ziehung in den Daten ist vom {last.date()}; "
              "mit --refresh aktualisieren.")

    # Merkmale für das Zieldatum
    d = pd.Series([target])
    parts = [calendar_features(d), astro_features(d)]
    parts.append(space_features(d, load_space_weather()))
    today = pd.Timestamp.today().normalize()
    parts.append(news_features(d, news_daily(today - pd.Timedelta(days=10), today)))
    row = pd.concat(parts, axis=1)

    cols, mean, std = meta["cols"], pd.Series(meta["mean"]), pd.Series(meta["std"])
    row = row.reindex(columns=cols)
    missing = [c for c in cols if row[c].isna().any()]
    if missing:
        print(f"Hinweis: {len(missing)} Merkmale nicht verfügbar, "
              "Trainingsmittel wird verwendet.")
    f = ((row.fillna(mean) - mean) / std).astype("float32").values[0]

    i = int((full["date"] < target).sum())
    x = np.concatenate([history_features(L, i), f])[None, :].astype("float32")
    probs = model.predict(x, verbose=0)[0]

    print(f"\nVorschläge für die Ziehung am {target.date()}:")
    rng = np.random.default_rng(int(target.strftime("%Y%m%d")))
    for k in range(n_tickets):
        m, e = _pick(probs, temperature=0.0 if k == 0 else 0.02, rng=rng)
        print(f"  Tipp {k + 1}: {m}  Eurozahlen: {e}")
    print("\nHinweis: Gewinnchance je Tipp bleibt 1 : 140.000.000 (ca.).")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true",
                    help="Ziehungen neu herunterladen (data/draws.csv ersetzen)")
    ap.add_argument("--build-db", action="store_true", help="Datenbank bauen")
    ap.add_argument("--train", action="store_true", help="Modell trainieren")
    ap.add_argument("--predict", metavar="YYYY-MM-DD", help="Zahlen für Datum")
    ap.add_argument("--tickets", type=int, default=3)
    ap.add_argument("--overfit", action="store_true",
                    help="so lange trainieren, bis die bekannten Ziehungen "
                         "passen (lernt auswendig, siehe Testvergleich)")
    ap.add_argument("--target-hits", type=float, default=5.0,
                    help="Ziel: Ø richtige Zahlen (von 7) auf bekannten Ziehungen")
    args = ap.parse_args()

    if not (args.refresh or args.build_db or args.train or args.predict):
        ap.print_help()
        return

    if args.refresh:
        df = _download_draws()
        if df is None:
            sys.exit("Download fehlgeschlagen, vorhandene data/draws.csv bleibt.")
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(DRAWS_CSV, index=False)
        print(f"data/draws.csv aktualisiert, letzte Ziehung: {df['date'].max().date()}")

    db = None
    if args.build_db:
        db = build_database()
    if args.train:
        if db is None:
            if not DB_CSV.exists():
                db = build_database()
            else:
                db = pd.read_csv(DB_CSV, parse_dates=["date"])
        train(db, args.overfit, args.target_hits)
    if args.predict:
        predict(args.predict, args.tickets)


if __name__ == "__main__":
    main()
