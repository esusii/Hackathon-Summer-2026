"""Which feature blocks actually help the MLP? Reference-trained, screened on the
5,000 competition train cells."""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_route_labels, load, log, route_mask
from mlp import Blocks, train_mlp

D = load()
B = Blocks(D)
NC = B.NC
ytr, yre = D["ytr"], D["yre"]
route_labels = build_route_labels(D["rre"], yre)
M = route_mask(D["rtr"], route_labels, NC)

rng = np.random.default_rng(0)
perm = rng.permutation(len(yre))
va, fit = perm[:10000], perm[10000:]

CONFIGS = [
    ("G", "genes only"),
    ("GQ", "+ QC"),
    ("GQM", "+ metadata"),
    ("GQMS", "+ spatial histograms"),
    ("GQMSP", "+ section prior"),
    ("GQMSPN", "+ neighbourhood expression"),
    ("GQMSPNE", "+ PCA-kNN histogram"),
    ("GQMSPNEP", "+ PCA block"),
]
results = {}
for blocks, desc in CONFIGS:
    bl = set(blocks.replace("SP", "\x00")) if False else None
    # explicit block sets to avoid substring ambiguity between S / SP
    setmap = {
        "G": {"G"}, "GQ": {"G", "Q"}, "GQM": {"G", "Q", "M"},
        "GQMS": {"G", "Q", "M", "S"}, "GQMSP": {"G", "Q", "M", "S", "SP"},
        "GQMSPN": {"G", "Q", "M", "S", "SP", "N"},
        "GQMSPNE": {"G", "Q", "M", "S", "SP", "N", "E"},
        "GQMSPNEP": {"G", "Q", "M", "S", "SP", "N", "E", "P"},
    }
    bs = setmap[blocks]
    Xre = B.get("re", bs)
    Xtr = B.get("tr", bs)
    (Ptr,) = train_mlp(Xre[fit], yre[fit], [Xtr], NC, seed=0, epochs=60,
                       val=(Xre[va], yre[va]), verbose=False)
    raw = float((Ptr.argmax(1) == ytr).mean())
    pm = Ptr.copy(); pm[~M] = -np.inf
    msk = float((pm.argmax(1) == ytr).mean())
    results[blocks] = msk
    log(f"{blocks:10s} {desc:28s} dim={Xre.shape[1]:4d}  raw {raw:.4f}  masked {msk:.4f}")

log("")
best = max(results, key=results.get)
log(f"best block set: {best} at {results[best]:.4f}")
