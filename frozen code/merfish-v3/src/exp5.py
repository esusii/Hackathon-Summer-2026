"""(a) how much do the 300 unseen genes matter, (b) a specialist for the big
non-neuronal route, (c) the marginal value of the competition-train rows."""
import os, sys
import anndata as ad
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, build_route_labels, load, log, route_mask
from members import Data
from mlp import Blocks, train_mlp

ARCH = dict(hidden=(512, 256), epochs=60, dropout=0.30, lr=3e-3, wd=1e-4, label_smooth=0.02)
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

def ens(Xre, Xtr_, fitidx, yfit, evals, n=3, **kw):
    outs = None
    for s in range(n):
        o = train_mlp(Xre[fitidx] if fitidx is not None else Xre, yfit, evals, NC,
                      seed=s, verbose=False, **{**ARCH, **kw})
        outs = [a.copy() for a in o] if outs is None else [x + y for x, y in zip(outs, o)]
    return [o / n for o in outs]

# ---------------------------------------------------------------- (a) 500-gene ceiling
log("=== (a) ceiling: what the 300 unseen genes are worth ===")
ROOT = DATA_ROOT
A = ad.read_h5ad(os.path.join(ROOT, "external", "MERFISH_spinal_cord_resolved_0718.h5ad"))
A.obs.index = [str(x) for x in A.obs.index]
mt = pd.read_csv(os.path.join(ROOT, "data", "meta_train.csv"), index_col=0)
ms = pd.read_csv(os.path.join(ROOT, "data", "meta_test.csv"), index_col=0)
comp = set(map(str, mt.index)) | set(map(str, ms.index))
keep = np.array([c not in comp for c in A.obs.index])
ct = pd.read_csv(os.path.join(ROOT, "data", "counts_train.csv"), index_col=0)
GENES = list(ct.columns)
gi = [list(A.var_names).index(g) for g in GENES]
X500 = A.X[np.flatnonzero(keep)]
X500 = np.asarray(X500.todense() if hasattr(X500, "todense") else X500).astype(np.float32)
X200 = X500[:, gi]
fp = {r.tobytes() for r in np.vstack([ct.values,
      pd.read_csv(os.path.join(ROOT, "data", "counts_test.csv"), index_col=0).values]).astype(np.int32)}
dup = np.array([r.tobytes() in fp for r in np.rint(X200).astype(np.int32)])
X500, X200 = X500[~dup], X200[~dup]
assert len(X500) == len(yre)
rest_re = D_.rest["re"]; sp_re = D_.sp_old["re"]
for tag, Xg in [("200 genes (what we get)", X200), ("500 genes (full reference)", X500)]:
    Xf = np.hstack([np.log1p(Xg), rest_re, sp_re]).astype(np.float32)
    (pv,) = ens(Xf, None, fit, yre[fit], [Xf[va]], n=2)
    log(f"  {tag:32s} ref_val {sc(pv, Mva, yre[va]):.4f}")
del X500, X200

# ------------------------------------------------------- (b) specialist for big route
log("=== (b) specialist for the 16-class non-neuronal route ===")
Xre = D_.matrix("re", "log", "S"); Xtr_ = D_.matrix("tr", "log", "S")
pv_g, pt_g = ens(Xre, None, fit, yre[fit], [Xre[va], Xtr_], n=3)
log(f"  global model                     comp_train {sc(pt_g, Mtr, ytr):.4f}  ref_val {sc(pv_g, Mva, yre[va]):.4f}")

sizes = {r: len(s) for r, s in rl.items()}
big = max(sizes, key=lambda r: sizes[r])
log(f"  biggest route '{big}' has {sizes[big]} candidates")
cls = sorted(rl[big]); remap = {c: i for i, c in enumerate(cls)}
re_in = np.array([r == big for r in D["rre"]])
tr_in = np.array([r == big for r in D["rtr"]])
va_in = re_in[va]; fit_in = re_in[fit]
log(f"  reference cells in route: {re_in.sum()}, comp-train cells: {tr_in.sum()}")

ysp = np.array([remap[c] for c in yre[fit][fit_in]], dtype=np.int32)
Xsp = Xre[fit][fit_in]
outs = None
for s in range(3):
    o = train_mlp(Xsp, ysp, [Xre[va][va_in], Xtr_[tr_in]], len(cls), seed=s, verbose=False, **ARCH)
    outs = [a.copy() for a in o] if outs is None else [x + y for x, y in zip(outs, o)]
sv, st = [o / 3 for o in outs]

for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    pt = pt_g.copy(); pv = pv_g.copy()
    sub_t = np.zeros((tr_in.sum(), NC)); sub_t[:, cls] = st
    sub_v = np.zeros((va_in.sum(), NC)); sub_v[:, cls] = sv
    pt[tr_in] = (1 - w) * pt[tr_in] + w * sub_t
    pv[va_in] = (1 - w) * pv[va_in] + w * sub_v
    log(f"    specialist weight {w:.1f}  comp_train {sc(pt, Mtr, ytr):.4f}  ref_val {sc(pv, Mva, yre[va]):.4f}")

# --------------------------------------------- (c) marginal value of comp-train rows
log("=== (c) marginal value of the 5,000 competition-train rows ===")
r2 = np.random.default_rng(7); pp = r2.permutation(5000)
h1, h2 = pp[:2500], pp[2500:]
(p_no,) = ens(Xre, None, fit, yre[fit], [Xtr_[h2]], n=3)
log(f"  reference only                     -> {sc(p_no, Mtr[h2], ytr[h2]):.4f}")
Xb = np.vstack([Xre[fit], Xtr_[h1]]); yb = np.concatenate([yre[fit], ytr[h1]])
(p_yes,) = ens(Xb, None, None, yb, [Xtr_[h2]], n=3)
log(f"  reference + 2,500 competition rows -> {sc(p_yes, Mtr[h2], ytr[h2]):.4f}")
