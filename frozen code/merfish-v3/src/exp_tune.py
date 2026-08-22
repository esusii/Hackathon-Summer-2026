"""Hyperparameter sweep. Selection is made on a held-out reference split
(n=10,000); competition train is reported alongside as an honest check."""
import itertools
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_route_labels, load, log, route_mask
from mlp import Blocks, train_mlp

D = load()
B = Blocks(D)
NC = B.NC
ytr, yre = D["ytr"], D["yre"]
route_labels = build_route_labels(D["rre"], yre)
Mtr = route_mask(D["rtr"], route_labels, NC)

BS = {"G", "Q", "M", "S", "SP"}
Xre_all = B.get("re", BS)
Xtr_all = B.get("tr", BS)
log(f"feature dim {Xre_all.shape[1]}")

rng = np.random.default_rng(0)
perm = rng.permutation(len(yre))
va, fit = perm[:10000], perm[10000:]
Mva = route_mask(D["rre"][va], route_labels, NC)


def score(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())


def run(tag, **kw):
    Pva, Ptr = train_mlp(Xre_all[fit], yre[fit], [Xre_all[va], Xtr_all], NC,
                         verbose=False, **kw)
    a_va = score(Pva, Mva, yre[va])
    a_tr = score(Ptr, Mtr, ytr)
    log(f"{tag:52s} ref_val {a_va:.4f}   comp_train {a_tr:.4f}")
    return a_va, a_tr, Ptr


results = {}
log("--- depth / width ---")
for hidden in [(512, 256), (1024, 512, 256), (2048, 1024, 512), (2048, 1024, 512, 256), (4096, 2048, 1024)]:
    results[f"h{hidden}"] = run(f"hidden={hidden}", hidden=hidden, epochs=60, seed=0)

log("--- epochs ---")
for ep in (60, 120, 200, 300):
    results[f"e{ep}"] = run(f"epochs={ep}", hidden=(2048, 1024, 512), epochs=ep, seed=0)

log("--- dropout ---")
for dr in (0.15, 0.25, 0.35, 0.45):
    results[f"d{dr}"] = run(f"dropout={dr}", hidden=(2048, 1024, 512), epochs=120,
                            dropout=dr, seed=0)

log("--- lr / weight decay / label smoothing ---")
for lr, wd, ls in itertools.product((1.5e-3, 3e-3), (1e-4, 1e-3), (0.0, 0.05)):
    results[f"l{lr}_{wd}_{ls}"] = run(f"lr={lr} wd={wd} ls={ls}", hidden=(2048, 1024, 512),
                                      epochs=120, dropout=0.25, lr=lr, wd=wd,
                                      label_smooth=ls, seed=0)

log("")
for k, v in sorted(results.items(), key=lambda kv: -kv[1][0])[:8]:
    log(f"  {k:28s} ref_val {v[0]:.4f}  comp_train {v[1]:.4f}")
