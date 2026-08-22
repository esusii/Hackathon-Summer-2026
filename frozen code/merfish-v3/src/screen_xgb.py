"""Screening run: train on reference cells only, score on all 5,000 competition
train cells. Their labels touch nothing upstream, so this is unbiased at n=5000."""
import os, sys

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, build_route_labels, load, log, report

D = load()
CLASSES, names = D["classes"], D["names"]
NC = len(CLASSES)
Xre, yre, Xtr, ytr = D["Xre"], D["yre"], D["Xtr"], D["ytr"]
route_labels = build_route_labels(D["rre"], yre)
from common import route_mask
Mtr = route_mask(D["rtr"], route_labels, NC)
log(f"reference {Xre.shape}, competition train {Xtr.shape}")

# hold out a slice of the reference purely for early stopping
rng = np.random.default_rng(0)
perm = rng.permutation(len(Xre))
n_es = 12000
es_idx, fit_idx = perm[:n_es], perm[n_es:]

params = dict(
    n_estimators=1200, learning_rate=0.10, max_depth=8,
    subsample=0.85, colsample_bytree=0.4, min_child_weight=4,
    reg_lambda=1.5, objective="multi:softprob", num_class=NC,
    tree_method="hist", device="cuda", eval_metric="mlogloss",
    early_stopping_rounds=40, n_jobs=32, random_state=42,
)
log("fitting XGBoost on GPU ...")
clf = XGBClassifier(**params)
clf.fit(Xre[fit_idx], yre[fit_idx], eval_set=[(Xre[es_idx], yre[es_idx])], verbose=50)
log(f"best iteration {clf.best_iteration}")

P = clf.predict_proba(Xtr)
report("XGB(all features, reference-trained)", P, Mtr, ytr)
np.save(os.path.join(CACHE, "P_xgb_screen.npy"), P)

# where does the remaining error sit?
sz = np.array([len(route_labels[r]) for r in D["rtr"]])
pm = P.copy(); pm[~Mtr] = -np.inf
pred = pm.argmax(1)
for lo, hi in [(1, 1), (2, 3), (4, 8), (9, 20)]:
    m = (sz >= lo) & (sz <= hi)
    if m.sum():
        log(f"  routes with {lo}-{hi} candidates: n={m.sum():5d} acc={float((pred[m]==ytr[m]).mean()):.4f}")

log("worst classes (n>=40 in competition train):")
for c in range(NC):
    m = ytr == c
    if m.sum() >= 40:
        a = float((pred[m] == ytr[m]).mean())
        if a < 0.75:
            log(f"    {CLASSES[c]:34s} n={m.sum():4d} acc={a:.3f}")
