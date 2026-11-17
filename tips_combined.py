"""20 Tipps: Statistik-Gewichte (heiß/überfällig) x Modellwahrscheinlichkeit (Mond, Sonne, Planeten, Weltraumwetter)."""
import sys, os, random, collections
repo = sys.argv[1]; target = sys.argv[2]
sys.path.insert(0, repo); os.chdir(repo)
import numpy as np, pandas as pd, json
import eurojackpot_experiment as ej
import tensorflow as tf

mp, meta_p = ej.model_paths(False)
model = tf.keras.models.load_model(mp); meta = json.loads(meta_p.read_text())
full = ej.load_all_draws(); L = ej.multi_hot_matrix(full)
t = pd.Timestamp(target); d = pd.Series([t])
astro = ej.astro_features(d)
row = pd.concat([ej.calendar_features(d), astro, ej.space_features(d, ej.load_space_weather())], axis=1)
cols, mean, std = meta["cols"], pd.Series(meta["mean"]), pd.Series(meta["std"])
row = row.reindex(columns=cols)
f = ((row.fillna(mean) - mean) / std).astype("float32").values[0]
i = int((full["date"] < t).sum())
x = np.concatenate([ej.history_features(L, i), f])[None, :].astype("float32")
probs = model.predict(x, verbose=0)[0]
pm, pe = probs[:50], probs[50:]
print("Mond/Sonne-Merkmale:", {c: round(float(astro[c].iloc[0]), 3) for c in astro.columns if "moon" in c or "sun" in c})

# Statistik-Gewichte wie tips_stat.py
seq_m = [list(map(int, r)) for r in full[["n1","n2","n3","n4","n5"]].values]
seq_e = [list(map(int, r)) for r in full[["e1","e2"]].values]
def stat_w(seq, N, k, recent=20):
    hot = collections.Counter(v for s in seq[-recent:] for v in s)
    w = {}
    for n in range(1, N+1):
        gap = next((j for j, s in enumerate(reversed(seq)) if n in s), len(seq))
        w[n] = 1 + hot[n] / (recent*k/N) + max(0, gap/(N/k) - 1)
    return w
sm, se = stat_w(seq_m, 50, 5), stat_w(seq_e, 12, 2)
# Kombination: Statistik x (Modell relativ zum Mittel)
wm = {n: sm[n] * pm[n-1] / pm.mean() for n in range(1, 51)}
we = {n: se[n] * pe[n-1] / pe.mean() for n in range(1, 13)}
print("Modell Top-10 Hauptzahlen:", [int(n) for n in np.argsort(pm)[::-1][:10] + 1])
print("Modell Top-4 Eurozahlen:", [int(n) for n in np.argsort(pe)[::-1][:4] + 1])
print("Kombiniert Top-10:", sorted(wm, key=lambda n: -wm[n])[:10])
print("Kombiniert Euro Top-4:", sorted(we, key=lambda n: -we[n])[:4])

rng = random.SystemRandom()
past = {tuple(sorted(s)) for s in seq_m}
def pick(w, k):
    pool, out = dict(w), []
    for _ in range(k):
        v = rng.choices(list(pool), weights=list(pool.values()))[0]; out.append(v); del pool[v]
    return sorted(out)
def ok(tk):
    if any(tk[j+2]-tk[j] == 2 for j in range(3)): return False
    if len({tk[j+1]-tk[j] for j in range(4)}) == 1: return False
    if len({n//10 for n in tk}) < 3 or sum(n > 31 for n in tk) < 2: return False
    return tuple(tk) not in past and 100 <= sum(tk) <= 190
tickets, euros = [], []
while len(tickets) < 20:
    tk = pick(wm, 5)
    if ok(tk) and all(len(set(tk) & set(u)) <= 2 for u in tickets): tickets.append(tk)
while len(euros) < 20:
    e = tuple(pick(we, 2))
    if e not in euros and e[1]-e[0] > 1: euros.append(e)
for j, (tk, e) in enumerate(zip(tickets, euros), 1):
    print(f"| {j} | {', '.join(map(str, tk))} | {e[0]}, {e[1]} |")
