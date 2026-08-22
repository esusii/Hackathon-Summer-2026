"""
Final step — produces prediction/prediction.csv.

"""
import os

import numpy as np
import pandas as pd

W = 0.40
EPS = 1e-9
NPZ = "oof_probabilities.npz"

meta_test = pd.read_csv("data/meta_test.csv", index_col=0)
meta_train = pd.read_csv("data/meta_train.csv", index_col=0)
labels = sorted(meta_train["MERFISH_cell_type_annotation"].unique())

P = np.load("v2_test_probs.npy")
assert P.shape == (len(meta_test), len(labels))

if os.path.exists(NPZ):
    z = np.load(NPZ, allow_pickle=True)
    assert [str(c) for c in z["classes"]] == labels
    assert all(str(a) == str(b) for a, b in zip(z["test_ids"], meta_test.index))
    Q = z["test_probs"].astype(np.float64)
    assert Q.shape == P.shape
    B = np.power(np.clip(P, EPS, 1), 1 - W) * np.power(np.clip(Q, EPS, 1), W)
    B /= B.sum(axis=1, keepdims=True)
    print(f"blended: P ({1 - W:.2f}) x Q ({W:.2f})")
    if "test_route_mask" in z:
        M = z["test_route_mask"].astype(bool)
        Bm = B * M
        s = Bm.sum(axis=1, keepdims=True)
        keep = (s > 0).ravel()               # a fully-masked row keeps the unmasked blend
        B[keep] = Bm[keep] / s[keep]
        print(f"route mask applied ({int(keep.sum())} cells masked)")
else:
    B = P
    print(f"{NPZ} not found -> declared fallback: P alone")

pred = np.array(labels)[B.argmax(axis=1)]
sub = pd.DataFrame({"Cell_ID": meta_test.index,
                    "MERFISH_cell_type_annotation.y": pred})
assert len(sub) == len(meta_test)
assert (sub["Cell_ID"].values == meta_test.index.values).all()
assert sub["MERFISH_cell_type_annotation.y"].isin(labels).all()
sub.to_csv("prediction.csv", index=False)
print(f"wrote prediction.csv ({len(sub)} rows)")
