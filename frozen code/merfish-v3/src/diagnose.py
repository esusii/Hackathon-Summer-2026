import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
F = np.load(os.path.join(CACHE, "features.npz"), allow_pickle=True)
R = np.load(os.path.join(CACHE, "routes.npz"), allow_pickle=True)
names = list(F["names"])
CLASSES = F["classes"]
NC = len(CLASSES)
ytr, yre = F["ytr"], F["yre"]
rtr, rte, rre = R["rtr"], R["rte"], R["rre"]

col = {n: i for i, n in enumerate(names)}
eh = [col[f"eh_{c}"] for c in CLASSES]
sh15 = [col[f"sh15_{c}"] for c in CLASSES]
sh60 = [col[f"sh60_{c}"] for c in CLASSES]
sp = [col[f"sp_{c}"] for c in CLASSES]
Xtr, Xte = F["Xtr"], F["Xte"]

print("=== class distribution ===")
ptr = np.bincount(ytr, minlength=NC) / len(ytr)
pre = np.bincount(yre, minlength=NC) / len(yre)
print(f"total variation distance train vs reference: {0.5*np.abs(ptr-pre).sum():.4f}")
print(f"reference class count: min {np.bincount(yre, minlength=NC).min()}, "
      f"max {np.bincount(yre, minlength=NC).max()}")
print(f"train class count: min {np.bincount(ytr, minlength=NC).min()}")

print("\n=== route structure (candidate sets learned from the 136,574 reference cells) ===")
route_labels = {}
for r, y in zip(rre, yre):
    route_labels.setdefault(r, set()).add(int(y))
sizes = np.array([len(v) for v in route_labels.values()])
print(f"routes in reference: {len(route_labels)}; single-label routes: {(sizes==1).sum()}")
print(f"candidate-set sizes: {sorted(sizes)}")

cover = np.array([r in route_labels for r in rtr])
print(f"\ntrain cells whose route exists in reference: {cover.mean():.4f}")
ok = np.array([(r in route_labels) and (int(y) in route_labels[r]) for r, y in zip(rtr, ytr)])
print(f"ROUTE CEILING on train (true label inside its reference candidate set): {ok.mean():.4f}")
te_cover = np.array([r in route_labels for r in rte])
print(f"test cells whose route exists in reference: {te_cover.mean():.4f}")

sz_tr = np.array([len(route_labels.get(r, set(range(NC)))) for r in rtr])
print(f"\ntrain cells in a 1-label route: {(sz_tr==1).mean():.4f}  (these are decided by metadata alone)")
for lo, hi in [(1,1),(2,3),(4,8),(9,20),(21,60)]:
    m = (sz_tr>=lo)&(sz_tr<=hi)
    if m.sum():
        print(f"  candidates {lo}-{hi}: {m.sum():5d} cells ({m.mean():.3f})")

# route-prior baseline: most common reference label within the route
prior_pred = np.zeros(len(rtr), dtype=int)
route_counts = {}
for r, y in zip(rre, yre):
    d = route_counts.setdefault(r, np.zeros(NC))
    d[y] += 1
for i, r in enumerate(rtr):
    prior_pred[i] = route_counts[r].argmax() if r in route_counts else np.bincount(yre).argmax()
print(f"\nbaseline: predict the route's most common reference label -> {(prior_pred==ytr).mean():.4f}")

print("\n=== how strong is each neighbour feature on its own (argmax, train cells) ===")
for nm, idx in [("expression kNN histogram (k=50)", eh),
                ("spatial kNN histogram (k=15)", sh15),
                ("spatial kNN histogram (k=60)", sh60),
                ("section class prior", sp)]:
    p = Xtr[:, idx]
    print(f"  {nm:38s} {(p.argmax(1)==ytr).mean():.4f}")

# combined, with and without the route mask
mask = np.zeros((len(rtr), NC), dtype=bool)
for i, r in enumerate(rtr):
    s = route_labels.get(r)
    if s:
        mask[i, list(s)] = True
    else:
        mask[i, :] = True

for nm, idx in [("expression kNN histogram", eh), ("spatial k=15", sh15), ("spatial k=60", sh60)]:
    p = Xtr[:, idx].copy()
    p[~mask] = -1
    print(f"  {nm:38s} + route mask -> {(p.argmax(1)==ytr).mean():.4f}")

comb = Xtr[:, eh] + Xtr[:, sh15] + Xtr[:, sh60]
print(f"  sum of the three histograms              {(comb.argmax(1)==ytr).mean():.4f}")
comb2 = comb.copy(); comb2[~mask] = -1
print(f"  sum of the three + route mask            {(comb2.argmax(1)==ytr).mean():.4f}")
