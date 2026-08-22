"""Export probability matrices for the team blend."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, build_route_labels, load, log, route_mask

ROOT = DATA_ROOT
D = load()
CLASSES = D["classes"].astype(str)
NC = len(CLASSES)
ytr, yre = D["ytr"], D["yre"]

Ptr = np.load(os.path.join(CACHE, "screen_final_comp.npy"))   # comp-train, reference-only model
Pte = np.load(os.path.join(CACHE, "final_test_probs.npy"))    # test, final model
assert Ptr.shape == (5000, NC) and Pte.shape == (5000, NC)

mt = pd.read_csv(os.path.join(ROOT, "data", "meta_train.csv"), index_col=0)
ms = pd.read_csv(os.path.join(ROOT, "data", "meta_test.csv"), index_col=0)
mt.index, ms.index = mt.index.astype(str), ms.index.astype(str)

rl = build_route_labels(np.concatenate([D["rre"], D["rtr"]]), np.concatenate([yre, ytr]))
Mtr = route_mask(D["rtr"], rl, NC)
Mte = route_mask(D["rte"], rl, NC)

# row order must match the official files exactly
assert (D["train_ids"].astype(str) == mt.index.values).all()
assert (D["test_ids"].astype(str) == ms.index.values).all()
assert np.allclose(Ptr.sum(1), 1, atol=1e-4) and np.allclose(Pte.sum(1), 1, atol=1e-4)

def acc(P, M, y):
    a = float((P.argmax(1) == y).mean())
    p = P.copy(); p[~M] = -np.inf
    return a, float((p.argmax(1) == y).mean())
raw, msk = acc(Ptr, Mtr, ytr)
log(f"train-cell probabilities: raw {raw:.4f}  route-masked {msk:.4f}")
log(f"route mask keeps a median of {int(np.median(Mte.sum(1)))} of {NC} classes per test cell")

out = os.path.join(os.path.dirname(CACHE), "oof_probabilities.npz")
np.savez_compressed(
    out,
    train_probs=Ptr.astype(np.float32), train_ids=mt.index.values.astype(str),
    train_labels=CLASSES[ytr],
    test_probs=Pte.astype(np.float32), test_ids=ms.index.values.astype(str),
    classes=CLASSES, train_route_mask=Mtr, test_route_mask=Mte,
)
log(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")

z = np.load(out, allow_pickle=True)
log("contents: " + ", ".join(f"{k}{z[k].shape}" for k in z.files))
log(f"reload check - train argmax acc {float((z['train_probs'].argmax(1)==ytr).mean()):.4f}")
