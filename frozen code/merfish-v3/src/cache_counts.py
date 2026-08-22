"""Cache raw integer counts so normalisation variants can be tested."""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, load, log

ROOT = DATA_ROOT
DATA = os.path.join(ROOT, "data")
D = load()

ct = pd.read_csv(os.path.join(DATA, "counts_train.csv"), index_col=0)
cs = pd.read_csv(os.path.join(DATA, "counts_test.csv"), index_col=0)
GENES = list(ct.columns)
mt = pd.read_csv(os.path.join(DATA, "meta_train.csv"), index_col=0)
ms = pd.read_csv(os.path.join(DATA, "meta_test.csv"), index_col=0)
comp_ids = set(map(str, mt.index)) | set(map(str, ms.index))

A = ad.read_h5ad(os.path.join(ROOT, "external", "MERFISH_spinal_cord_resolved_0718.h5ad"))
A.obs.index = [str(x) for x in A.obs.index]
keep = np.array([c not in comp_ids for c in A.obs.index])
gi = [list(A.var_names).index(g) for g in GENES]
X = A.X[np.ix_(np.flatnonzero(keep), gi)]
X = np.rint(np.asarray(X.todense() if hasattr(X, "todense") else X)).astype(np.int32)
fp = {r.tobytes() for r in np.vstack([ct.values, cs.values]).astype(np.int32)}
dup = np.array([r.tobytes() in fp for r in X])
X = X[~dup]
assert len(X) == len(D["yre"])

np.savez_compressed(os.path.join(CACHE, "counts.npz"),
                    Ctr=ct.values.astype(np.int16), Cte=cs.values.astype(np.int16),
                    Cre=X.astype(np.int16), genes=np.array(GENES))
log(f"cached raw counts: ref {X.shape}, max {X.max()}")
