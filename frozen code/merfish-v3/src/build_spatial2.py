"""Richer spatial context: more kNN scales plus fixed-radius Gaussian kernels.

Reference cells always exclude themselves. Query cells (competition train/test)
never contribute labels to anything.
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.spatial as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, load, log

ROOT = DATA_ROOT
DATA = os.path.join(ROOT, "data")

D = load()
CLASSES = D["classes"]
NC = len(CLASSES)
yre = D["yre"]

meta_train = pd.read_csv(os.path.join(DATA, "meta_train.csv"), index_col=0)
meta_test = pd.read_csv(os.path.join(DATA, "meta_test.csv"), index_col=0)
A = ad.read_h5ad(os.path.join(ROOT, "external", "MERFISH_spinal_cord_resolved_0718.h5ad"))
A.obs.index = [str(x) for x in A.obs.index]
comp_ids = set(map(str, meta_train.index)) | set(map(str, meta_test.index))

# reference rows in exactly the order used by the cache
re_sec = D["re_sec"].astype(str)
keep = np.array([c not in comp_ids for c in A.obs.index])
O = A.obs[keep]
# rebuild the same duplicate filter used in build_features
GENES = list(pd.read_csv(os.path.join(DATA, "counts_train.csv"), index_col=0, nrows=1).columns)
gi = [list(A.var_names).index(g) for g in GENES]
X = A.X[np.ix_(np.flatnonzero(keep), gi)]
X = np.rint(np.asarray(X.todense() if hasattr(X, "todense") else X)).astype(np.int32)
ct = pd.read_csv(os.path.join(DATA, "counts_train.csv"), index_col=0).values.astype(np.int32)
cs = pd.read_csv(os.path.join(DATA, "counts_test.csv"), index_col=0).values.astype(np.int32)
fp = {r.tobytes() for r in np.vstack([ct, cs])}
dup = np.array([r.tobytes() in fp for r in X])
O = O[~dup]
assert len(O) == len(yre), (len(O), len(yre))
ref_xy = np.column_stack([O["center_x"].values, O["center_y"].values]).astype(np.float64)
assert (np.array([str(s) for s in O["Section ID"].values]) == re_sec).all()
log(f"reference geometry aligned: {ref_xy.shape}")

K_LIST = (3, 5, 10, 30, 120, 250)
SIGMAS = (25.0, 60.0, 150.0)

sec_index = {s: np.flatnonzero(re_sec == s) for s in np.unique(re_sec)}
trees = {s: sp.cKDTree(ref_xy[i]) for s, i in sec_index.items()}
nn_dist = {}

# typical nearest-neighbour spacing, to sanity-check the sigma choices
d1 = []
for s, idx in list(sec_index.items())[:5]:
    d, _ = trees[s].query(ref_xy[idx], k=2, workers=-1)
    d1.append(d[:, 1])
log(f"median nearest reference-cell spacing: {np.median(np.concatenate(d1)):.1f} units")


def build(meta_xy, secs, is_ref):
    n = len(secs)
    K = {k: np.zeros((n, NC + 1), dtype=np.float32) for k in K_LIST}
    Gk = {s_: np.zeros((n, NC + 1), dtype=np.float32) for s_ in SIGMAS}
    for s in np.unique(secs):
        rows = np.flatnonzero(secs == s)
        ridx = sec_index[s]
        rlab = yre[ridx]
        tree = trees[s]
        q = meta_xy[rows]
        for k in K_LIST:
            kk = min(k + (1 if is_ref else 0), len(ridx))
            d, j = tree.query(q, k=kk, workers=-1)
            d, j = np.atleast_2d(d), np.atleast_2d(j)
            if is_ref and d.shape[1] > 1:
                d, j = d[:, 1:], j[:, 1:]
            w = 1.0 / (1.0 + d)
            lab = rlab[j]
            h = np.zeros((len(rows), NC), dtype=np.float32)
            np.add.at(h, (np.repeat(np.arange(len(rows)), lab.shape[1]), lab.ravel()), w.ravel())
            h /= np.maximum(h.sum(1, keepdims=True), 1e-9)
            K[k][rows, :NC] = h
            K[k][rows, NC] = d.mean(1)
        # Gaussian kernels, evaluated over one wide kNN query (bounded work)
        kk = min(400 + (1 if is_ref else 0), len(ridx))
        d, j = tree.query(q, k=kk, workers=-1)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        if is_ref and d.shape[1] > 1:
            d, j = d[:, 1:], j[:, 1:]
        lab = rlab[j]
        rowid = np.repeat(np.arange(len(rows)), lab.shape[1])
        for sg in SIGMAS:
            w = np.exp(-0.5 * (d / sg) ** 2)
            h = np.zeros((len(rows), NC), dtype=np.float32)
            np.add.at(h, (rowid, lab.ravel()), w.ravel())
            eff = w.sum(1)
            h /= np.maximum(h.sum(1, keepdims=True), 1e-9)
            Gk[sg][rows, :NC] = h
            Gk[sg][rows, NC] = np.log1p(eff)
    return K, Gk


tr_xy = np.column_stack([meta_train["center_x"].values, meta_train["center_y"].values])
te_xy = np.column_stack([meta_test["center_x"].values, meta_test["center_y"].values])
Ktr, Gtr = build(tr_xy, D["tr_sec"].astype(str), False); log("train done")
Kte, Gte = build(te_xy, D["te_sec"].astype(str), False); log("test done")
Kre, Gre = build(ref_xy, re_sec, True); log("ref done")


def pack(K, G):
    return np.hstack([K[k] for k in K_LIST] + [G[s] for s in SIGMAS]).astype(np.float32)


names = []
for k in K_LIST:
    names += [f"k{k}_{c}" for c in CLASSES] + [f"k{k}_meand"]
for s in SIGMAS:
    names += [f"g{int(s)}_{c}" for c in CLASSES] + [f"g{int(s)}_n"]

np.savez_compressed(os.path.join(CACHE, "spatial2.npz"),
                    Str=pack(Ktr, Gtr), Ste=pack(Kte, Gte), Sre=pack(Kre, Gre),
                    names=np.array(names))
log(f"saved spatial2: {pack(Ktr, Gtr).shape}")
