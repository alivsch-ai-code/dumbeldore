import csv, collections, random, itertools, sys
rng = random.SystemRandom()
rows = sorted([r for r in csv.reader(open(sys.argv[1])) if r[0][:1].isdigit()], key=lambda r: r[0])
past = {tuple(sorted(map(int, r[1:6]))) for r in rows}
def weights(cols, N, recent):
    seq = [[int(r[c]) for c in cols] for r in rows]
    hot = collections.Counter(x for s in seq[-recent:] for x in s)
    gap = {}
    for k in range(1, N+1):
        gap[k] = next((i for i, s in enumerate(reversed(seq)) if k in s), len(seq))
    exp_gap = N / len(cols)
    # Gewicht: 1 + Häufigkeit (heiß) + Überfälligkeit relativ zum erwarteten Abstand
    return {k: 1 + hot[k] / (recent*len(cols)/N) + max(0, gap[k]/exp_gap - 1) for k in range(1, N+1)}, hot, gap
wm, hot, gap = weights(range(1, 6), 50, 20)
we, hote, gape = weights(range(6, 8), 12, 20)
def pick(w, k):
    pool, out = dict(w), []
    for _ in range(k):
        x = rng.choices(list(pool), weights=list(pool.values()))[0]
        out.append(x); del pool[x]
    return sorted(out)
def ok(t):
    if any(t[i+2]-t[i] == 2 for i in range(3)): return False
    if len({t[i+1]-t[i] for i in range(4)}) == 1: return False
    if len({n//10 for n in t}) < 3: return False
    if sum(n > 31 for n in t) < 2: return False
    return tuple(t) not in past and 100 <= sum(t) <= 190
tickets, euros = [], []
while len(tickets) < 20:
    t = pick(wm, 5)
    if ok(t) and all(len(set(t) & set(u)) <= 2 for u in tickets):
        tickets.append(t)
while len(euros) < 20:
    e = tuple(pick(we, 2))
    if e not in euros and e[1]-e[0] > 1: euros.append(e)
top = sorted(wm, key=lambda k: -wm[k])[:12]
print("Top-Gewichte Hauptzahlen:", [(k, round(wm[k],2), hot[k], gap[k]) for k in top])
print("Eurozahlen-Gewichte:", [(k, round(we[k],2), hote[k], gape[k]) for k in sorted(we, key=lambda k:-we[k])])
for i, (t, e) in enumerate(zip(tickets, euros), 1):
    print(f"| {i} | {', '.join(map(str,t))} | {e[0]}, {e[1]} |")
