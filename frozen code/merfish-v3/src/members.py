"""The diverse member list shared by screening and the final fit."""
import os
import numpy as np
from common import CACHE
from mlp import Blocks

MEMBERS = [
    ("log",         "S",     (512, 256)),
    ("log",         "S2",    (512, 256)),
    ("pearson",     "S",     (512, 256)),
    ("pearson",     "S2",    (512, 256)),
    ("raw_log",     "S",     (512, 256)),
    ("raw_log",     "S2",    (1024, 512, 256)),
    ("log",         "both",  (1024, 512, 256)),
    ("pearson",     "both",  (2048, 1024, 512)),
    ("log+pearson", "both",  (1024, 512, 256)),
    ("sqrt",        "S2",    (512, 256)),
    ("log+sqrt",    "S",     (512, 256)),
    ("log",         "S2",    (4096, 2048, 1024)),
]

class Data:
    def __init__(self, D, B):
        self.D, self.B = D, B
        C = np.load(os.path.join(CACHE, "counts.npz"), allow_pickle=True)
        S2 = np.load(os.path.join(CACHE, "spatial2.npz"), allow_pickle=True)
        self.Cnt = {"tr": C["Ctr"].astype(np.float32), "te": C["Cte"].astype(np.float32),
                    "re": C["Cre"].astype(np.float32)}
        self.TARGET = float(np.median(self.Cnt["tr"].sum(1)))
        self.p_g = self.Cnt["re"].sum(0) / self.Cnt["re"].sum()
        self.rest = {w: B.get(w, {"Q", "M", "SP"}) for w in ("tr", "te", "re")}
        self.sp_old = {w: B.get(w, {"S"}) for w in ("tr", "te", "re")}
        self.sp_new = {"tr": S2["Str"], "te": S2["Ste"], "re": S2["Sre"]}

    def norm(self, w, kind):
        Cm = self.Cnt[w]
        tot = np.maximum(Cm.sum(1, keepdims=True), 1e-9)
        if kind == "log":     return np.log1p(Cm / tot * self.TARGET)
        if kind == "sqrt":    return np.sqrt(Cm / tot * self.TARGET)
        if kind == "raw_log": return np.log1p(Cm)
        if kind == "pearson":
            mu = tot * self.p_g[None, :]
            return np.clip((Cm - mu) / np.sqrt(mu + mu * mu / 100.0 + 1e-9), -10, 10)
        if kind == "log+pearson":
            return np.hstack([self.norm(w, "log"), self.norm(w, "pearson")])
        if kind == "log+sqrt":
            return np.hstack([self.norm(w, "log"), self.norm(w, "sqrt")])
        raise ValueError(kind)

    def matrix(self, w, norm_kind, spatial):
        sp = {"S": self.sp_old[w], "S2": self.sp_new[w],
              "both": np.hstack([self.sp_old[w], self.sp_new[w]])}[spatial]
        return np.hstack([self.norm(w, norm_kind), self.rest[w], sp]).astype(np.float32)
