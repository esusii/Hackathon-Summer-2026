"""Greedy ensemble selection (Caruana). Members are chosen on the reference
holdout only; competition train is reported as an uncontaminated check."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, build_route_labels, load, log, route_mask
from members import MEMBERS

D = load(); NC = len(D["classes"])
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(D["rre"], yre)
Mtr = route_mask(D["rtr"], rl, NC)
rng = np.random.default_rng(0); perm = rng.permutation(len(yre))
va = perm[:10000]
Mva = route_mask(D["rre"][va], rl, NC)
yva = yre[va]

Z = np.load(os.path.join(CACHE, "screen_probs.npz"))
T = [Z[f"t{i}"] for i in range(len(MEMBERS))]
V = [Z[f"v{i}"] for i in range(len(MEMBERS))]

def sc(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())

log("solo members (ref_val / comp_train):")
for i, m in enumerate(MEMBERS):
    log(f"  m{i:2d} {str(m):46s} {sc(V[i], Mva, yva):.4f}  {sc(T[i], Mtr, ytr):.4f}")

log("greedy forward selection with replacement, scored on ref_val:")
pick, accV, accT = [], np.zeros_like(V[0]), np.zeros_like(T[0])
best_hist = []
for step in range(20):
    best, bi = -1, None
    for i in range(len(MEMBERS)):
        s = sc((accV + V[i]) / (len(pick) + 1), Mva, yva)
        if s > best:
            best, bi = s, i
    pick.append(bi); accV += V[bi]; accT += T[bi]
    t = sc(accT / len(pick), Mtr, ytr)
    best_hist.append((best, t, list(pick)))
    log(f"  step {step+1:2d}: +m{bi:<2d}  ref_val {best:.4f}   comp_train {t:.4f}")

bi = max(range(len(best_hist)), key=lambda i: best_hist[i][0])
log(f"-> best by ref_val at step {bi+1}: ref_val {best_hist[bi][0]:.4f}  comp_train {best_hist[bi][1]:.4f}")
from collections import Counter
cnt = Counter(best_hist[bi][2])
log("   chosen weights:")
for i, c in cnt.most_common():
    log(f"     {c}x  m{i}  {MEMBERS[i]}")
np.save(os.path.join(CACHE, "chosen.npy"), np.array(best_hist[bi][2]))

log("reference points:")
log(f"  equal weight all 12                 ref_val {sc(sum(V)/12, Mva, yva):.4f}  comp_train {sc(sum(T)/12, Mtr, ytr):.4f}")
small = [i for i, m in enumerate(MEMBERS) if m[2] == (512, 256)]
log(f"  equal weight small nets {small}  ref_val {sc(sum(V[i] for i in small)/len(small), Mva, yva):.4f}  "
    f"comp_train {sc(sum(T[i] for i in small)/len(small), Mtr, ytr):.4f}")
