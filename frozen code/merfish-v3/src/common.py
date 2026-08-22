"""Shared loading, route mask, and scoring for the v3 experiments."""
import os
import time

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

# Competition data root. Override with the MERFISH_ROOT environment variable;
# otherwise look for a `data/` directory beside the project, then fall back to
# the development layout.
def _resolve_root():
    env = os.environ.get("MERFISH_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [here,
             os.path.join(here, "Hackathon-Summer-2026-main"),
             os.path.join(os.path.expanduser("~"), "Downloads",
                          "Hackathon-Summer-2026-main", "Hackathon-Summer-2026-main")]
    for c in cands:
        if os.path.isdir(os.path.join(c, "data")):
            return c
    return here


DATA_ROOT = _resolve_root()

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def load():
    F = np.load(os.path.join(CACHE, "features.npz"), allow_pickle=True)
    R = np.load(os.path.join(CACHE, "routes.npz"), allow_pickle=True)
    d = {k: F[k] for k in F.files}
    d.update({k: R[k] for k in R.files})
    d["names"] = list(d["names"])
    return d


def route_mask(routes, route_labels, NC):
    """Boolean (n, NC): which labels the cell's anatomy route permits."""
    M = np.zeros((len(routes), NC), dtype=bool)
    for i, r in enumerate(routes):
        s = route_labels.get(r)
        if s:
            M[i, list(s)] = True
        else:
            M[i, :] = True
    return M


def build_route_labels(rre, yre):
    d = {}
    for r, y in zip(rre, yre):
        d.setdefault(r, set()).add(int(y))
    return d


def masked_acc(proba, mask, y):
    p = proba.copy()
    p[~mask] = -np.inf
    return float((p.argmax(1) == y).mean())


def report(name, proba, mask, y):
    raw = float((proba.argmax(1) == y).mean())
    msk = masked_acc(proba, mask, y)
    log(f"{name:44s} raw {raw:.4f}   masked {msk:.4f}")
    return msk
