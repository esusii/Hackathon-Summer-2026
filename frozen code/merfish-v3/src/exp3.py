"""Normalisation and spatial-block screening, 3 seeds averaged per config.

Training uses reference cells only, so the 5,000 competition-train cells stay a
clean, unbiased evaluation set. A 10,000-cell reference holdout is reported too.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, build_route_labels, load, log, route_mask
from mlp import Blocks, train_mlp

ARCH = dict(hidden=(512, 256), epochs=60, dropout=0.30, lr=3e-3, wd=1e-4, label_smooth=0.02)
NSEED = 3

D = load(); B = Blocks(D); NC = B.NC
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(D["rre"], yre)
Mtr = route_mask(D["rtr"], rl, NC)

C = np.load(os.path.join(CACHE, "counts.npz"), allow_pickle=True)
S2 = np.load(os.path.join(CACHE, "spatial2.npz"), allow_pickle=True)
Ctr, Cte, Cre = (C[k].astype(np.float32) for k in ("Ctr", "Cte", "Cre"))
TARGET = float(np.median(Ctr.sum(1)))
p_g = Cre.sum(0) / Cre.sum()

rng = np.random.default_rng(0); perm = rng.permutation(len(yre))
va, fit = perm[:10000], perm[10000:]
Mva = route_mask(D["rre"][va], rl, NC)

def norm(Cm, kind):
    tot = np.maximum(Cm.sum(1, keepdims=True), 1e-9)
    if kind == "log":     return np.log1p(Cm / tot * TARGET)
    if kind == "sqrt":    return np.sqrt(Cm / tot * TARGET)
    if kind == "raw_log": return np.log1p(Cm)
    if kind == "pearson":
        mu = tot * p_g[None, :]
        return np.clip((Cm - mu) / np.sqrt(mu + mu * mu / 100.0 + 1e-9), -10, 10)
    if kind == "log+pearson":
        return np.hstack([norm(Cm, "log"), norm(Cm, "pearson")])
    if kind == "log+sqrt":
        return np.hstack([norm(Cm, "log"), norm(Cm, "sqrt")])
    raise ValueError(kind)

def sc(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())

def evaluate(tag, Xre, Xtr_):
    Pv = np.zeros((len(va), NC)); Pt = np.zeros((len(Xtr_), NC))
    for s in range(NSEED):
        a, b = train_mlp(Xre[fit], yre[fit], [Xre[va], Xtr_], NC, seed=s, verbose=False, **ARCH)
        Pv += a; Pt += b
    v, t = sc(Pv, Mva, yre[va]), sc(Pt, Mtr, ytr)
    log(f"  {tag:34s} dim={Xre.shape[1]:4d}  ref_val {v:.4f}   comp_train {t:.4f}")
    return v, t

REST = {w: B.get(w, {"Q", "M", "SP"}) for w in ("tr", "re")}
SP_OLD = {w: B.get(w, {"S"}) for w in ("tr", "re")}
SP_NEW = {"tr": S2["Str"], "re": S2["Sre"]}

log("=== normalisation (spatial = original S) ===")
res = {}
for kind in ("log", "sqrt", "raw_log", "pearson", "log+pearson", "log+sqrt"):
    Xre = np.hstack([norm(Cre, kind), REST["re"], SP_OLD["re"]]).astype(np.float32)
    Xtr_ = np.hstack([norm(Ctr, kind), REST["tr"], SP_OLD["tr"]]).astype(np.float32)
    res[kind] = evaluate(kind, Xre, Xtr_)
best_norm = max(res, key=lambda k: res[k][0])
log(f"  -> best by ref_val: {best_norm}")

log("=== spatial block (normalisation = %s) ===" % best_norm)
Gre, Gtr_ = norm(Cre, best_norm), norm(Ctr, best_norm)
for tag, sre, str_ in [("S  (k=15,60)", SP_OLD["re"], SP_OLD["tr"]),
                       ("S2 (k=3..250 + gauss)", SP_NEW["re"], SP_NEW["tr"]),
                       ("S + S2", np.hstack([SP_OLD["re"], SP_NEW["re"]]),
                                   np.hstack([SP_OLD["tr"], SP_NEW["tr"]]))]:
    res["sp:" + tag] = evaluate(tag, np.hstack([Gre, REST["re"], sre]).astype(np.float32),
                                     np.hstack([Gtr_, REST["tr"], str_]).astype(np.float32))
