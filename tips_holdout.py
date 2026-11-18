import sys, os, json, random
os.chdir(sys.argv[1]); sys.path.insert(0, '.')
import numpy as np, pandas as pd, tensorflow as tf
import eurojackpot_experiment as ej
mp, metap = ej.model_paths("holdout")
model = tf.keras.models.load_model(mp); meta = json.loads(metap.read_text())
full = ej.load_all_draws(); L = ej.multi_hot_matrix(full)
t = pd.Timestamp(sys.argv[2]); d = pd.Series([t])
row = pd.concat([ej.calendar_features(d), ej.astro_features(d), ej.space_features(d, ej.load_space_weather())], axis=1)
cols, mean, std = meta["cols"], pd.Series(meta["mean"]), pd.Series(meta["std"])
f = ((row.reindex(columns=cols).fillna(mean) - mean) / std).astype("float32").values[0]
i = int((full["date"] < t).sum())
p = model.predict(np.concatenate([ej.history_features(L, i), f])[None, :].astype("float32"), verbose=0)[0]
pm, pe = p[:50], p[50:]
print("Top-12 Hauptzahlen:", [int(x) for x in np.argsort(pm)[::-1][:12] + 1])
print("Top-5 Eurozahlen:", [int(x) for x in np.argsort(pe)[::-1][:5] + 1])
top_m = sorted(int(x) for x in np.argsort(pm)[::-1][:5] + 1); top_e = sorted(int(x) for x in np.argsort(pe)[::-1][:2] + 1)
rng = random.SystemRandom()
past = {tuple(sorted(map(int, r))) for r in full[["n1","n2","n3","n4","n5"]].values}
def pick(w, k):
    pool, out = {n+1: float(v) for n, v in enumerate(w)}, []
    for _ in range(k):
        x = rng.choices(list(pool), weights=list(pool.values()))[0]; out.append(x); del pool[x]
    return sorted(out)
tickets, euros = [top_m], [tuple(top_e)]
while len(tickets) < 10:
    tk = pick(pm, 5)
    if tuple(tk) not in past and all(len(set(tk) & set(u)) <= 2 for u in tickets): tickets.append(tk)
while len(euros) < 10:
    e = tuple(pick(pe, 2))
    if e not in euros: euros.append(e)
for j, (tk, e) in enumerate(zip(tickets, euros), 1):
    print(f"| {j} | {', '.join(map(str, tk))} | {e[0]}, {e[1]} |")
