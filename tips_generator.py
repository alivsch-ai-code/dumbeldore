import random, itertools, csv, sys
rng = random.SystemRandom()
draws = [tuple(map(int, r[1:6])) for r in csv.reader(open(sys.argv[1])) if r[0][0].isdigit()]
past = set(tuple(sorted(d)) for d in draws)
def ok(t):
    t = sorted(t)
    if sum(n > 31 for n in t) < 3: return False          # Geburtstagsbereich meiden
    if sum(n <= 12 for n in t) > 1: return False         # Monatszahlen meiden
    if any(t[i+2]-t[i] == 2 for i in range(3)): return False  # keine 3er-Folge
    d = [t[i+1]-t[i] for i in range(4)]
    if len(set(d)) == 1: return False                     # keine arithm. Reihe
    if len({n // 10 for n in t}) < 3: return False        # über Zehner streuen
    if len({n % 10 for n in t}) < 4: return False         # keine Endziffern-Muster
    if all(n % 7 == 0 for n in t): return False
    if tuple(t) in past: return False
    return 115 <= sum(t) <= 200
tickets = []
while len(tickets) < 20:
    t = sorted(rng.sample(range(1, 51), 5))
    if ok(t) and all(len(set(t) & set(u)) <= 1 for u in tickets):
        tickets.append(t)
# Eurozahlen: 20 verschiedene Paare, jede Zahl 3-4 mal, Paar (1,2) etc. vermeiden
pairs = [p for p in itertools.combinations(range(1, 13), 2) if p[1]-p[0] > 1]
while True:
    sel = rng.sample(pairs, 20)
    cnt = [sum(n in p for p in sel) for n in range(1, 13)]
    if max(cnt) - min(cnt) <= 1: break
sel.sort(key=lambda p: rng.random())
cover = len(set(itertools.chain(*tickets)))
for i, (t, e) in enumerate(zip(tickets, sel), 1):
    print(f"{i:2d} | {', '.join(f'{n:2d}' for n in t)} | {e[0]:2d}, {e[1]:2d}")
print("abgedeckte Hauptzahlen:", cover, "von 50")
