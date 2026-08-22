"""Does ensembling across diverse configurations beat the best single one?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_route_labels, load, log, route_mask
from members import MEMBERS, Data
from mlp import Blocks, train_mlp

D = load(); B = Blocks(D); NC = B.NC
D_ = Data(D, B)
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(D["rre"], yre)
Mtr = route_mask(D["rtr"], rl, NC)
rng = np.random.default_rng(0); perm = rng.permutation(len(yre))
va, fit = perm[:10000], perm[10000:]
Mva = route_mask(D["rre"][va], rl, NC)

def sc(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())

ARCH = dict(dropout=0.30, lr=3e-3, wd=1e-4, label_smooth=0.02, epochs=60)
Pens_t = np.zeros((5000, NC)); Pens_v = np.zeros((len(va), NC))
store = {}
for i, (nk, spk, hid) in enumerate(MEMBERS):
    Xre = D_.matrix("re", nk, spk); Xtr_ = D_.matrix("tr", nk, spk)
    pt = np.zeros((5000, NC)); pv = np.zeros((len(va), NC))
    for s in range(2):
        a, b = train_mlp(Xre[fit], yre[fit], [Xre[va], Xtr_], NC, seed=10 * i + s,
                         hidden=hid, verbose=False, **ARCH)
        pv += a; pt += b
    pt /= 2; pv /= 2
    store[i] = (pv, pt)
    Pens_t += pt; Pens_v += pv
    log(f"  m{i:2d} {nk:12s} {spk:5s} {str(hid):18s} solo {sc(pt, Mtr, ytr):.4f} "
        f"| cumulative ensemble  comp_train {sc(Pens_t/(i+1), Mtr, ytr):.4f}  "
        f"ref_val {sc(Pens_v/(i+1), Mva, yre[va]):.4f}")

np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "screen_probs.npz"),
                    **{f"t{i}": store[i][1] for i in store}, **{f"v{i}": store[i][0] for i in store})
log(f"FINAL ensemble comp_train {sc(Pens_t/len(MEMBERS), Mtr, ytr):.4f}")
