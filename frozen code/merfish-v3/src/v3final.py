"""v3 final recipe, one code path for both modes.

  screen : train on reference cells only -> score the 5,000 competition-train
           cells (unbiased: their labels touch nothing upstream) + a reference holdout
  final  : train on reference + all competition train -> write prediction.csv

Recipe = greedy-selected ensemble of small MLPs across normalisations, plus a
specialist for the 16-class non-neuronal route, plus the anatomy route mask.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, build_route_labels, load, log, route_mask
from members import Data
from mlp import Blocks, train_mlp
from distill import train_distill

MODE = sys.argv[1] if len(sys.argv) > 1 else "screen"
SEEDS_PER_SLOT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SPEC_W = 0.0   # specialist proved neutral once distillation was added
ARCH = dict(epochs=60, dropout=0.30, lr=3e-3, wd=1e-4, label_smooth=0.02)  # verbose not used by train_distill
# (normalisation, spatial block, hidden, slots) from greedy selection on ref_val
CHOSEN = [("log", "S", (512, 256), 5), ("raw_log", "S", (512, 256), 2),
          ("pearson", "S", (512, 256), 2), ("log", "S2", (512, 256), 1),
          ("log", "both", (1024, 512, 256), 1), ("sqrt", "S2", (512, 256), 1)]

D = load(); B = Blocks(D); NC = B.NC
D_ = Data(D, B)
CLASSES = D["classes"]
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(np.concatenate([D["rre"], D["rtr"]]), np.concatenate([yre, ytr]))
big = max(rl, key=lambda r: len(rl[r]))
cls = sorted(rl[big]); remap = {c: i for i, c in enumerate(cls)}

Q_ref = np.load(os.path.join(CACHE, "teacher_oof_v2.npy"))
Q_tr = np.eye(NC, dtype=np.float32)[ytr] * 0.98 + 0.02 / NC
ALPHA, TEMP = 0.4, 2.0

rng = np.random.default_rng(0); perm = rng.permutation(len(yre))
va, fitr = perm[:10000], perm[10000:]

if MODE == "screen":
    ridx, use_train = fitr, False
    Mva = route_mask(D["rre"][va], rl, NC)
    Mtr = route_mask(D["rtr"], rl, NC)
else:
    ridx, use_train = np.arange(len(yre)), True
    Mte = route_mask(D["rte"], rl, NC)

def build(nk, spk):
    Xre = D_.matrix("re", nk, spk)
    Xfit = Xre[ridx]; yfit = yre[ridx]; rfit = D["rre"][ridx]; qfit = Q_ref[ridx]
    if use_train:
        Xfit = np.vstack([Xfit, D_.matrix("tr", nk, spk)])
        yfit = np.concatenate([yfit, ytr])
        rfit = np.concatenate([rfit, D["rtr"]])
        qfit = np.vstack([qfit, Q_tr])
    evals = ([Xre[va], D_.matrix("tr", nk, spk)] if MODE == "screen"
             else [D_.matrix("te", nk, spk)])
    return Xfit, yfit, rfit, qfit, evals

def sc(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())

n_eval = 2 if MODE == "screen" else 1
acc = None; tot = 0
spec_acc = None; spec_tot = 0
for nk, spk, hid, slots in CHOSEN:
    Xfit, yfit, rfit, qfit, evals = build(nk, spk)
    reps = slots * SEEDS_PER_SLOT
    for s in range(reps):
        o = train_distill(Xfit, yfit, qfit, evals, NC, seed=hash((nk, spk, s)) % 10000,
                          hidden=hid, alpha=ALPHA, T=TEMP, **ARCH)
        acc = [a.copy() for a in o] if acc is None else [x + y for x, y in zip(acc, o)]
        tot += 1
    if SPEC_W == 0:
        log(f'  {nk:12s} {spk:5s} {str(hid):18s} x{reps} done'); continue
    # specialist on the same feature view
    in_route = rfit == big
    Xs, ys = Xfit[in_route], np.array([remap[c] for c in yfit[in_route]], dtype=np.int32)
    Qs_ = qfit[in_route][:, cls]
    Qs_ = Qs_ / np.maximum(Qs_.sum(1, keepdims=True), 1e-9)
    ev_masks = ([D["rre"][va] == big, D["rtr"] == big] if MODE == "screen"
                else [D["rte"] == big])
    ev_sub = [e[m] for e, m in zip(evals, ev_masks)]
    for s in range(SEEDS_PER_SLOT):
        o = train_distill(Xs, ys, Qs_, ev_sub, len(cls), seed=hash((nk, spk, "sp", s)) % 10000,
                          hidden=hid, alpha=ALPHA, T=TEMP, **ARCH)
        spec_acc = [a.copy() for a in o] if spec_acc is None else [x + y for x, y in zip(spec_acc, o)]
        spec_tot += 1
    log(f"  {nk:12s} {spk:5s} {str(hid):18s} x{reps} done")

P = [a / tot for a in acc]
S = [a / spec_tot for a in spec_acc] if spec_acc is not None else None
ev_masks = ([D["rre"][va] == big, D["rtr"] == big] if MODE == "screen"
            else [D["rte"] == big])
Pb = [p.copy() for p in P]
for k in range(n_eval if SPEC_W > 0 else 0):
    m = ev_masks[k]
    sub = np.zeros((m.sum(), NC)); sub[:, cls] = S[k]
    Pb[k][m] = (1 - SPEC_W) * Pb[k][m] + SPEC_W * sub

if MODE == "screen":
    log(f"ensemble            ref_val {sc(P[0], Mva, yre[va]):.4f}   comp_train {sc(P[1], Mtr, ytr):.4f}")
    log(f"ensemble+specialist ref_val {sc(Pb[0], Mva, yre[va]):.4f}   comp_train {sc(Pb[1], Mtr, ytr):.4f}")
    np.save(os.path.join(CACHE, "screen_final_comp.npy"), Pb[1])
else:
    pm = Pb[0].copy(); pm[~Mte] = -np.inf
    pred = CLASSES[pm.argmax(1)]
    sub = pd.DataFrame({"Cell_ID": D["test_ids"].astype(str),
                        "MERFISH_cell_type_annotation.y": pred})
    assert len(sub) == 5000 and sub["Cell_ID"].is_unique
    assert sub["MERFISH_cell_type_annotation.y"].isin(set(CLASSES)).all()
    out = r"C:\Users\noron\Downloads\v3_submission"
    os.makedirs(out, exist_ok=True)
    sub.to_csv(os.path.join(out, "prediction.csv"), index=False)
    sub.to_csv(r"C:\Users\noron\Downloads\prediction_v3.csv", index=False)
    np.save(os.path.join(CACHE, "final_test_probs.npy"), Pb[0])
    log(f"wrote prediction.csv ({len(sub)} rows, {sub['MERFISH_cell_type_annotation.y'].nunique()} distinct labels)")
