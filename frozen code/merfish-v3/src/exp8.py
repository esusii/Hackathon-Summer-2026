"""A stronger 500-gene teacher, then re-measure the student's gain."""
import os, sys
import anndata as ad
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, build_route_labels, load, log, route_mask
from members import Data
from mlp import Blocks, train_mlp
from distill import train_distill

D = load(); B = Blocks(D); NC = B.NC
D_ = Data(D, B)
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(D["rre"], yre)
Mtr = route_mask(D["rtr"], rl, NC)
rng = np.random.default_rng(0); perm = rng.permutation(len(yre))
va, fitr = perm[:10000], perm[10000:]
Mva = route_mask(D["rre"][va], rl, NC)

def sc(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())

TP2 = os.path.join(CACHE, "teacher_oof_v2.npy")
if not os.path.exists(TP2):
    ROOT = DATA_ROOT
    A = ad.read_h5ad(os.path.join(ROOT, "external", "MERFISH_spinal_cord_resolved_0718.h5ad"))
    A.obs.index = [str(x) for x in A.obs.index]
    mt = pd.read_csv(os.path.join(ROOT, "data", "meta_train.csv"), index_col=0)
    ms = pd.read_csv(os.path.join(ROOT, "data", "meta_test.csv"), index_col=0)
    comp = set(map(str, mt.index)) | set(map(str, ms.index))
    keep = np.array([c not in comp for c in A.obs.index])
    ct = pd.read_csv(os.path.join(ROOT, "data", "counts_train.csv"), index_col=0)
    cs = pd.read_csv(os.path.join(ROOT, "data", "counts_test.csv"), index_col=0)
    gi = [list(A.var_names).index(g) for g in ct.columns]
    X5 = A.X[np.flatnonzero(keep)]
    X5 = np.asarray(X5.todense() if hasattr(X5, "todense") else X5).astype(np.float32)
    fp = {r.tobytes() for r in np.vstack([ct.values, cs.values]).astype(np.int32)}
    dup = np.array([r.tobytes() in fp for r in np.rint(X5[:, gi]).astype(np.int32)])
    X5 = X5[~dup]
    tot = np.maximum(X5.sum(1, keepdims=True), 1e-9)
    XT = np.hstack([np.log1p(X5), np.log1p(X5 / tot * 57.0),
                    D_.rest["re"], D_.sp_old["re"], D_.sp_new["re"]]).astype(np.float32)
    del X5
    log(f"teacher feature dim {XT.shape[1]}")
    Q = np.zeros((len(yre), NC), dtype=np.float32)
    folds = np.arange(len(yre)) % 5
    for f in range(5):
        tr_i, te_i = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        acc = np.zeros((len(te_i), NC))
        for s in range(3):
            (p,) = train_mlp(XT[tr_i], yre[tr_i], [XT[te_i]], NC, seed=10*f+s,
                             hidden=(1024, 512, 256), epochs=100, dropout=0.25,
                             lr=3e-3, wd=1e-4, label_smooth=0.02, verbose=False)
            acc += p
        Q[te_i] = acc / 3
        log(f"  fold {f}: {float((Q[te_i].argmax(1)==yre[te_i]).mean()):.4f}")
    np.save(TP2, Q); del XT
Q2 = np.load(TP2)
Q1 = np.load(os.path.join(CACHE, "teacher_oof.npy"))
log(f"teacher v1 OOF {float((Q1.argmax(1)==yre).mean()):.4f}   v2 OOF {float((Q2.argmax(1)==yre).mean()):.4f}")

Xre = D_.matrix("re", "log", "S"); Xtr_ = D_.matrix("tr", "log", "S")
for name, Q in (("teacher v1", Q1), ("teacher v2", Q2)):
    for alpha, T in [(0.4, 2.0), (0.3, 2.0), (0.3, 3.0)]:
        pv = np.zeros((len(va), NC)); pt = np.zeros((5000, NC))
        for s in range(3):
            a, b = train_distill(Xre[fitr], yre[fitr], Q[fitr], [Xre[va], Xtr_], NC,
                                 seed=s, alpha=alpha, T=T)
            pv += a; pt += b
        log(f"  {name} alpha={alpha} T={T}   ref_val {sc(pv, Mva, yre[va]):.4f}   "
            f"comp_train {sc(pt, Mtr, ytr):.4f}")
