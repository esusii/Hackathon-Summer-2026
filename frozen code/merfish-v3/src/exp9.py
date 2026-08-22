"""Test the four substantive ideas in the proposed v3 against the current model."""
import os, sys
import numpy as np
import pandas as pd
import scipy.spatial as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, build_route_labels, load, log, route_mask
from sklearn.neighbors import KNeighborsClassifier

D = load(); NC = len(D["classes"]); EPS = 1e-12
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(np.concatenate([D["rre"], D["rtr"]]), np.concatenate([yre, ytr]))
Mtr = route_mask(D["rtr"], rl, NC)
P = np.load(os.path.join(CACHE, "screen_final_comp.npy"))   # current model, comp_train

def sc(Q):
    q = Q.copy(); q[~Mtr] = -np.inf
    return float((q.argmax(1) == ytr).mean())
log(f"current model on comp_train: {sc(P):.4f}")

# ---- (1) does the proposed self-exclusion actually work in spatial_hist? -----
ROOT = DATA_ROOT
mt = pd.read_csv(os.path.join(ROOT, "data", "meta_train.csv"), index_col=0)
mt.index = mt.index.astype(str)
src_xy = np.vstack([np.column_stack([mt["center_x"].values, mt["center_y"].values])])
src_sec = D["tr_sec"].astype(str); src_lab = ytr           # train cells as their own source
H = np.zeros((len(ytr), NC), np.float32)
for s in np.unique(src_sec):
    idx = np.flatnonzero(src_sec == s)
    tree = sp.cKDTree(src_xy[idx])
    d, j = tree.query(src_xy[idx], k=min(16, len(idx)))     # k+1, self INCLUDED
    d, j = np.atleast_2d(d), np.atleast_2d(j)
    w = 1.0 / (1.0 + d)                                     # self gets weight 1.0
    lab = src_lab[idx][j]
    h = np.zeros((len(idx), NC), np.float32)
    np.add.at(h, (np.repeat(np.arange(len(idx)), lab.shape[1]), lab.ravel()), w.ravel())
    H[idx] = h / h.sum(1, keepdims=True)
log(f"(1) spatial histogram WITH the cell itself in the source: argmax acc {float((H.argmax(1)==ytr).mean()):.4f}")
log("    -> the feature would encode the label directly; the proposed _selftrim compares")
log("       section-LOCAL neighbour indices against GLOBAL source indices, so it never fires")

# ---- (2) prior-exponent correction ------------------------------------------
prior = np.bincount(ytr, minlength=NC) / len(ytr)
log("(2) prior-exponent alpha (P / prior**alpha):")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    Q = P / (prior[None, :] ** a + EPS)
    log(f"      alpha={a:<5} {sc(Q / Q.sum(1, keepdims=True)):.4f}")

# ---- (3) transductive spatial smoothing -------------------------------------
def smooth(Pm, xy, sec, k=8):
    S = Pm.copy()
    for s in np.unique(sec):
        idx = np.flatnonzero(sec == s)
        if len(idx) < 3: continue
        tree = sp.cKDTree(xy[idx])
        d, j = tree.query(xy[idx], k=min(k + 1, len(idx)))
        d, j = np.atleast_2d(d)[:, 1:], np.atleast_2d(j)[:, 1:]
        w = 1.0 / (1.0 + d)
        S[idx] = (w[:, :, None] * Pm[idx][j]).sum(1) / w.sum(1, keepdims=True)
    return S
Sm = smooth(P, src_xy, src_sec)
log(f"(3) transductive smoothing over the held-out set's own spatial graph "
    f"(~{len(ytr)/len(np.unique(src_sec)):.0f} cells/section):")
for b in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8):
    C = np.exp((1 - b) * np.log(P + EPS) + b * np.log(Sm + EPS))
    log(f"      beta={b:<5} {sc(C):.4f}")

# ---- (4) kNN classifier as an extra blend member ----------------------------
names = D["names"]; col = {n: i for i, n in enumerate(names)}
pc = [col[n] for n in names if n.startswith("pca_")]
knn = KNeighborsClassifier(n_neighbors=30, weights="distance", n_jobs=-1).fit(D["Xre"][:, pc], yre)
Pk = knn.predict_proba(D["Xtr"][:, pc])
log(f"(4) kNN(PCA-64, k=30) solo: {sc(Pk):.4f}")
for w in (0.0, 0.1, 0.2, 0.3, 0.5):
    log(f"      blended at weight {w:<4} {sc((1 - w) * P + w * Pk):.4f}")
