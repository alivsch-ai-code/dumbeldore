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
LOG_CSV = DATA_DIR / "tips_log.csv"


def model_paths(kind=False) -> tuple[Path, Path]:
    """Modelle liegen getrennt: normal, overfit (True) oder eigener Name (str)."""
    s = "" if not kind else ("_overfit" if kind is True else f"_{kind}")
    return DATA_DIR / f"model{s}.keras", DATA_DIR / f"model_meta{s}.json"

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
CELESTRAK_URL = "https://celestrak.org/SpaceData/SW-All.csv"  # Ausweichquelle
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
            raw["date"] = pd.to_datetime(
                raw["date"], dayfirst="." in str(raw["date"].iloc[0]))
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

    # Ausweichquelle CelesTrak (Spalten DATE, AP_AVG, ISN, F10.7_OBS)
    if not {"ap", "f107", "sunspots"} <= set(series):
        try:
            r = requests.get(CELESTRAK_URL, timeout=90)
            r.raise_for_status()
            cs = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"])
            cs = cs.set_index("DATE").sort_index()
            cs = cs[(cs.index >= pd.Timestamp.today() - pd.DateOffset(years=4))
                    & (cs.index <= pd.Timestamp.today())]
            series.setdefault("ap", cs["AP_AVG"].where(cs["AP_AVG"] >= 0))
            series.setdefault("f107", cs["F10.7_OBS"].where(cs["F10.7_OBS"] > 0))
            series.setdefault("sunspots", cs["ISN"].where(cs["ISN"] >= 0))
        except Exception as exc:  # noqa: BLE001
            print(f"Ausweichquelle CelesTrak nicht verfügbar: {exc}")

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
    model_path, meta_path = model_paths(overfit)
    model.save(model_path)
    meta_path.write_text(json.dumps(meta))
    print(f"Modell gespeichert: {model_path} ({len(cols)} Merkmale)")

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


def train_holdout(db: pd.DataFrame, holdout: int = 4, min_hits: float = 2.0,
                  max_tries: int = 300):
    """Overfit-Modelle mit wechselndem Startwert trainieren, bis eines auf den
    letzten `holdout` Ziehungen (nicht im Training) im Schnitt >= min_hits
    richtige Zahlen von 7 trifft. ACHTUNG: Das ist eine Auswahl auf genau
    diesen Ziehungen (Datenschnüffelei) und sagt nichts über neue Ziehungen."""
    import tensorflow as tf

    full = load_all_draws()
    L = multi_hot_matrix(full)
    cols = feature_columns(db)
    meta = {"cols": cols, "mean": db[cols].mean().to_dict(),
            "std": db[cols].std().replace(0, 1).to_dict()}
    X, y = make_dataset(db, meta, full, L)
    Xtr, ytr, Xho, yho = X[:-holdout], y[:-holdout], X[-holdout:], y[-holdout:]
    ho_dates = [d.date() for d in db["date"].iloc[-holdout:]]

    rng = np.random.default_rng(0)
    sims = np.array([np.mean([_hits(_pick(rng.random(N_OUT)), t) for t in yho])
                     for _ in range(5000)])
    p_rand = float((sims >= min_hits).mean())
    print(f"Zurückgehaltene Ziehungen: {ho_dates[0]} bis {ho_dates[-1]} "
          f"({holdout}). Ein Zufallstipp erreicht Ø >= {min_hits} dort mit "
          f"Wahrscheinlichkeit {p_rand:.2%} pro Versuch.")

    for seed in range(1, max_tries + 1):
        tf.keras.utils.set_random_seed(seed)
        model = build_model(X.shape[1], overfit=True)
        model.fit(Xtr, ytr, epochs=20, batch_size=16, verbose=0)
        p = model.predict(Xho, verbose=0)
        hits = [_hits(_pick(q), t) for q, t in zip(p, yho)]
        avg = float(np.mean(hits))
        if avg >= min_hits:
            model_path, meta_path = model_paths("holdout")
            model.save(model_path)
            meta_path.write_text(json.dumps(meta))
            print(f"Versuch {seed}: Ø {avg:.2f} von 7 auf den zurückgehaltenen "
                  f"Ziehungen (je Ziehung {hits}) -> gespeichert: {model_path}")
            return model
        if seed % 25 == 0:
            print(f"  {seed} Versuche, bester Wert bisher knapp darunter "
                  f"(zuletzt Ø {avg:.2f})")
    print(f"Kein Versuch erreichte Ø {min_hits} in {max_tries} Versuchen.")
    return None


def predict(target_date: str, n_tickets: int = 3, overfit: bool = False,
            tag: str | None = None):
    import tensorflow as tf

    model_path, meta_path = model_paths(tag or overfit)
    if not (model_path.exists() and meta_path.exists()):
        sys.exit("Kein Modell gefunden. Erst mit --train "
                 f"{'--overfit ' if overfit else ''}trainieren.")
    model = tf.keras.models.load_model(model_path)
    meta = json.loads(meta_path.read_text())
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
    rows = []
    for k in range(n_tickets):
        m, e = _pick(probs, temperature=0.0 if k == 0 else 0.02, rng=rng)
        print(f"  Tipp {k + 1}: {m}  Eurozahlen: {e}")
        rows.append({
            "made_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "target_date": target.date().isoformat(),
            "mode": tag or ("overfit" if overfit else "normal"),
            "ticket": k + 1,
            **{f"n{j + 1}": v for j, v in enumerate(m)},
            **{f"e{j + 1}": v for j, v in enumerate(e)},
        })
    DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(LOG_CSV, mode="a", header=not LOG_CSV.exists(),
                              index=False)
    print(f"\nTipps gespeichert in {LOG_CSV} (Auswertung: --evaluate).")
    print("Hinweis: Gewinnchance je Tipp bleibt 1 : 140.000.000 (ca.).")


def evaluate_tips():
    """Vergleicht gespeicherte Tipps mit den echten Ziehungen."""
    if not LOG_CSV.exists():
        sys.exit("Noch keine Tipps gespeichert (erst --predict ausführen).")
    log = pd.read_csv(LOG_CSV, parse_dates=["target_date"])
    full = load_all_draws().set_index("date")
    mc, ec = ["n1", "n2", "n3", "n4", "n5"], ["e1", "e2"]
    res = []
    for _, r in log.iterrows():
        if r["target_date"] not in full.index:
            continue
        d = full.loc[r["target_date"]]
        hm = len(set(r[mc].astype(int)) & set(d[mc].astype(int)))
        he = len(set(r[ec].astype(int)) & set(d[ec].astype(int)))
        res.append({"target_date": r["target_date"].date(), "mode": r["mode"],
                    "ticket": r["ticket"], "main": hm, "euro": he, "total": hm + he})
    open_dates = sorted(set(log["target_date"].dt.date)
                        - {x["target_date"] for x in res})
    if not res:
        print("Noch keine Ziehung zu den Tipps in den Daten "
              "(erst --refresh nach der Ziehung).")
    else:
        df = pd.DataFrame(res)
        print(df.to_string(index=False))
        theory = MAIN_PICK * MAIN_PICK / MAIN_MAX + EURO_PICK * EURO_PICK / EURO_MAX
        print("\nØ richtige Zahlen pro Tipp (Zufall theoretisch "
              f"{theory:.2f}):")
        for mode, g in df.groupby("mode"):
            print(f"  {mode}: {g['total'].mean():.2f} aus {len(g)} Tipps "
                  f"({g['target_date'].nunique()} Ziehungen), "
                  f"beste Ziehung {g['total'].max()} von 7")
    if open_dates:
        print("\nNoch offen (Ziehung nicht in den Daten): "
              + ", ".join(str(x) for x in open_dates))


# ---------------------------------------------------------------------------
def check_sources():
    """Testet alle Datenquellen und zeigt je einen Beispielwert."""
    def run(name, fn):
        try:
            print(f"OK     {name}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            print(f"FEHLER {name}: {exc}")

    def draws():
        df = _download_draws()
        if df is None:
            raise RuntimeError("Download fehlgeschlagen")
        return f"{len(df)} Ziehungen, letzte vom {df['date'].max().date()}"

    def silso():
        r = requests.get(SILSO_URL, timeout=60)
        r.raise_for_status()
        last = r.text.strip().splitlines()[-1]
        return f"{len(r.text.splitlines())} Zeilen, letzte: {last}"

    def space():
        sw = load_space_weather()
        if sw.empty:
            raise RuntimeError("keine Daten")
        last = sw.dropna(how="all").iloc[-1]
        return (f"Spalten {list(sw.columns)}, letzter Tag "
                f"{sw.dropna(how='all').index[-1].date()}: "
                + ", ".join(f"{k}={v:.1f}" for k, v in last.items()))

    def gdelt():
        today = pd.Timestamp.today().normalize()
        s = _gdelt_chunk("sourcelang:english", "timelinevol",
                         today - pd.Timedelta(days=6), today)
        if s is None or s.empty:
            raise RuntimeError("keine Antwort")
        return f"{len(s)} Tage, Ø Anteil {s.mean():.3f}"

    run("Ziehungen (GitHub-Archiv)", draws)
    run("Sonnenflecken (SILSO)", silso)
    run("Weltraumwetter (GFZ, sonst CelesTrak)", space)
    run("Weltereignisse (GDELT)", gdelt)


RULES_START = pd.Timestamp("2022-03-25")  # 12 Eurozahlen, Dienstag + Freitag


def _picks_matrix(scores: np.ndarray) -> np.ndarray:
    """Je Zeile die Top-Tipps (5 aus 50, 2 aus 12) als 0/1-Matrix."""
    P = np.zeros_like(scores, dtype="float32")
    for r, s in enumerate(scores):
        m, e = _pick(s)
        P[r, [x - 1 for x in m]] = 1
        P[r, [MAIN_MAX + x - 1 for x in e]] = 1
    return P


def _perm_test(P: np.ndarray, T: np.ndarray, n_perm: int, rng):
    """Permutationstest: Passen die Tipps zu IHRER Ziehung besser als zu
    vertauschten Ziehungen? (fängt auch ab, dass ein Modell immer dieselben
    beliebten Zahlen tippt)"""
    n = len(P)
    H = P @ T.T                      # H[i, j] = Treffer von Tipp i gegen Ziehung j
    obs = float(np.trace(H) / n)
    idx = np.arange(n)
    null = np.array([H[idx, rng.permutation(n)].mean() for _ in range(n_perm)])
    p = (1 + int((null >= obs).sum())) / (n_perm + 1)
    return obs, float(null.mean()), p


def backtest(initial: int = 150, step: int = 20, n_perm: int = 5000):
    """Rückwärts-Test (Walk-Forward): Das Modell wird nur mit Ziehungen VOR
    dem getesteten Block trainiert, alle `step` Ziehungen neu, und auf den
    folgenden Ziehungen geprüft. Merkmale: Ziehungshistorie, Kalender,
    Astronomie, Weltraumwetter (GDELT reicht nicht so weit zurück)."""
    import tensorflow as tf

    full = load_all_draws()
    full = full[full["date"] >= RULES_START].reset_index(drop=True)
    n = len(full)
    if n < initial + step:
        sys.exit("Zu wenige Ziehungen für den Rückwärts-Test.")
    print(f"Rückwärts-Test auf {n - initial} Ziehungen "
          f"(ab {full['date'][initial].date()}), Training wächst mit; "
          f"nur Ziehungen seit {RULES_START.date()} (gleiche Regeln).")
    L = multi_hot_matrix(full)

    dates = full["date"]
    ctx = pd.concat([calendar_features(dates), astro_features(dates),
                     space_features(dates, load_space_weather())], axis=1)
    ctx = ctx.loc[:, ctx.notna().mean() >= 0.8].fillna(ctx.mean())
    hist = np.stack([history_features(L, i) for i in range(n)])
    freq_sl = slice(WINDOW * N_OUT, WINDOW * N_OUT + N_OUT)
    gap_sl = slice(WINDOW * N_OUT + N_OUT, WINDOW * N_OUT + 2 * N_OUT)

    preds = np.zeros((n, N_OUT), dtype="float32")
    first_row = FREQ_WIN  # frühe Zeilen haben zu wenig Historie
    for start in range(initial, n, step):
        end = min(start + step, n)
        c = ctx.iloc[first_row:start]
        mean, std = c.mean(), c.std().replace(0, 1)
        feats = ((ctx - mean) / std).astype("float32").values
        X = np.hstack([hist, feats]).astype("float32")
        tr = np.arange(first_row, start)
        cut = int(len(tr) * 0.85)
        tf.keras.utils.set_random_seed(start)
        model = build_model(X.shape[1])
        model.fit(
            X[tr[:cut]], L[tr[:cut]], validation_data=(X[tr[cut:]], L[tr[cut:]]),
            epochs=60, batch_size=16, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=6, restore_best_weights=True)],
        )
        preds[start:end] = model.predict(X[start:end], verbose=0)
        print(f"  Ziehungen {start}–{end - 1} getestet "
              f"(Training mit {len(tr)} Ziehungen)")

    test = slice(initial, n)
    T = L[test]
    rng = np.random.default_rng(0)
    methods = {
        "Modell": preds[test],
        "Heiße Zahlen (Häufigkeit)": hist[test, freq_sl],
        "Überfällige Zahlen": hist[test, gap_sl],
    }
    theory = MAIN_PICK * MAIN_PICK / MAIN_MAX + EURO_PICK * EURO_PICK / EURO_MAX
    print(f"\nØ richtige Zahlen pro Tipp (Zufall theoretisch {theory:.2f}) "
          f"auf {len(T)} Ziehungen:")
    print(f"{'Methode':28s} {'Ø Treffer':>9s} {'Permut.-Ø':>10s} {'p-Wert':>8s}")
    for name, scores in methods.items():
        obs, null, p = _perm_test(_picks_matrix(scores), T, n_perm, rng)
        print(f"{name:28s} {obs:9.3f} {null:10.3f} {p:8.3f}")
    sim = np.array([
        np.mean([_hits(_pick(rng.random(N_OUT)), t) for t in T])
        for _ in range(300)
    ])
    print(f"{'Zufallstipps (Simulation)':28s} {sim.mean():9.3f}   "
          f"Streuung ±{sim.std():.3f}")
    print("\nLesehilfe: p-Wert = Wahrscheinlichkeit, dass ein so guter Wert "
          "auch bei reinem Zufall vorkommt. Bei 3 Methoden gilt erst "
          "p < 0,017 als auffällig (Mehrfachtest).")


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
    ap.add_argument("--evaluate", action="store_true",
                    help="gespeicherte Tipps mit den echten Ziehungen vergleichen")
    ap.add_argument("--backtest", action="store_true",
                    help="Rückwärts-Test über alle Ziehungen seit 03/2022 "
                         "mit Permutationstest")
    ap.add_argument("--check-sources", action="store_true",
                    help="alle Datenquellen testen und Beispielwerte zeigen")
    ap.add_argument("--min-hits", type=float, default=None,
                    help="Overfit-Modelle mit wechselndem Startwert trainieren, "
                         "bis eines auf den letzten Ziehungen (nicht im Training) "
                         "Ø >= dieser Wert von 7 trifft (Auswahl, keine Prognose)")
    ap.add_argument("--holdout", type=int, default=4,
                    help="Anzahl zurückgehaltener letzter Ziehungen für --min-hits")
    args = ap.parse_args()

    if args.check_sources:
        check_sources()
        return

    if not (args.refresh or args.build_db or args.train or args.predict
            or args.evaluate or args.backtest or args.min_hits is not None):
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
    if args.train or args.min_hits is not None:
        if db is None:
            if not DB_CSV.exists():
                db = build_database()
            else:
                db = pd.read_csv(DB_CSV, parse_dates=["date"])
    if args.train:
        train(db, args.overfit, args.target_hits)
    found = False
    if args.min_hits is not None:
        found = train_holdout(db, args.holdout, args.min_hits) is not None
    if args.predict and (args.min_hits is None or found):
        predict(args.predict, args.tickets, args.overfit,
                tag="holdout" if args.min_hits is not None else None)
    if args.evaluate:
        evaluate_tips()
    if args.backtest:
        backtest()


if __name__ == "__main__":
    main()
