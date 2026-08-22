"""Knowledge distillation: a 500-gene teacher (0.95 accurate) teaches the
200-gene student which classes are genuinely confusable.

Teacher probabilities are produced OUT-OF-FOLD so the student never sees a
teacher that memorised the cell it is scoring. The teacher uses only reference
data (500 genes exist only there); no competition label is involved.
"""
import os, sys
import anndata as ad
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, CACHE, build_route_labels, load, log, route_mask
from members import Data
from mlp import Blocks, train_mlp

D = load(); B = Blocks(D); NC = B.NC
D_ = Data(D, B)
ytr, yre = D["ytr"], D["yre"]
rl = build_route_labels(D["rre"], yre)
Mtr = route_mask(D["rtr"], rl, NC)
rng = np.random.default_rng(0); perm = rng.permutation(len(yre))
va, fitr = perm[:10000], perm[10000:]
Mva = route_mask(D["rre"][va], rl, NC)
ARCH = dict(hidden=(512, 256), epochs=60, dropout=0.30, lr=3e-3, wd=1e-4, label_smooth=0.02)

def sc(P, M, y):
    p = P.copy(); p[~M] = -np.inf
    return float((p.argmax(1) == y).mean())

# ------------------------------------------------------------------ 500-gene teacher
TPATH = os.path.join(CACHE, "teacher_oof.npy")
if os.path.exists(TPATH):
    Q = np.load(TPATH); log("loaded cached teacher OOF probabilities")
else:
    ROOT = DATA_ROOT
    A = ad.read_h5ad(os.path.join(ROOT, "external", "MERFISH_spinal_cord_resolved_0718.h5ad"))
    A.obs.index = [str(x) for x in A.obs.index]
    mt = pd.read_csv(os.path.join(ROOT, "data", "meta_train.csv"), index_col=0)
    ms = pd.read_csv(os.path.join(ROOT, "data", "meta_test.csv"), index_col=0)
    comp = set(map(str, mt.index)) | set(map(str, ms.index))
    keep = np.array([c not in comp for c in A.obs.index])
    ct = pd.read_csv(os.path.join(ROOT, "data", "counts_train.csv"), index_col=0)
    cs = pd.read_csv(os.path.join(ROOT, "data", "counts_test.csv"), index_col=0)
    gi = [list(A.var_names).index(g) for g in ct.columns]
    X5 = A.X[np.flatnonzero(keep)]
    X5 = np.asarray(X5.todense() if hasattr(X5, "todense") else X5).astype(np.float32)
    fp = {r.tobytes() for r in np.vstack([ct.values, cs.values]).astype(np.int32)}
    dup = np.array([r.tobytes() in fp for r in np.rint(X5[:, gi]).astype(np.int32)])
    X5 = X5[~dup]
    assert len(X5) == len(yre)
    XT = np.hstack([np.log1p(X5), D_.rest["re"], D_.sp_old["re"]]).astype(np.float32)
    del X5
    Q = np.zeros((len(yre), NC), dtype=np.float32)
    folds = np.arange(len(yre)) % 5
    for f in range(5):
        tr_i, te_i = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        (p,) = train_mlp(XT[tr_i], yre[tr_i], [XT[te_i]], NC, seed=f, verbose=False, **ARCH)
        Q[te_i] = p
        log(f"  teacher fold {f}: held-out acc {float((p.argmax(1)==yre[te_i]).mean()):.4f}")
    np.save(TPATH, Q)
    del XT
log(f"teacher OOF accuracy over the whole reference: {float((Q.argmax(1)==yre).mean()):.4f}")

# ------------------------------------------------------------------------- student
def train_distill(Xfit, yfit, Qfit, evals, seed, alpha, T, hidden=(512, 256), epochs=60):
    dev = "cuda"; torch.manual_seed(seed)
    Xg = torch.tensor(Xfit, device=dev)
    mu, sd = Xg.mean(0, keepdim=True), Xg.std(0, keepdim=True).clamp_min(1e-6)
    Xg = (Xg - mu) / sd
    yg = torch.tensor(yfit.astype(np.int64), device=dev)
    Qg = torch.tensor(Qfit, device=dev).clamp_min(1e-8)
    Qs = (Qg.log() / T).softmax(1)            # temperature-softened teacher
    layers, d = [], Xfit.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.GELU(),
                   torch.nn.Dropout(0.30)]; d = h
    layers.append(torch.nn.Linear(d, NC))
    net = torch.nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    bs = 4096
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3,
                                                total_steps=epochs * (len(Xg)//bs + 1))
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
    for ep in range(epochs):
        net.train()
        pm = torch.randperm(len(Xg), device=dev)
        for s in range(0, len(Xg), bs):
            i = pm[s:s+bs]
            if len(i) < 2: continue
            opt.zero_grad()
            z = net(Xg[i])
            loss = alpha * ce(z, yg[i]) + (1 - alpha) * (T * T) * torch.nn.functional.kl_div(
                torch.log_softmax(z / T, 1), Qs[i], reduction="batchmean")
            loss.backward(); opt.step(); sched.step()
    net.eval(); out = []
    with torch.no_grad():
        for Xe in evals:
            Xe_ = (torch.tensor(Xe, device=dev) - mu) / sd
            out.append(torch.cat([torch.softmax(net(Xe_[s:s+20000]), 1)
                                  for s in range(0, len(Xe_), 20000)]).cpu().numpy())
    del Xg, Qg, Qs, net; torch.cuda.empty_cache()
    return out

Xre = D_.matrix("re", "log", "S"); Xtr_ = D_.matrix("tr", "log", "S")
log("student on the 200-gene view (3 seeds each):")
for alpha, T in [(1.0, 1.0), (0.7, 1.0), (0.5, 1.0), (0.5, 2.0), (0.3, 2.0), (0.0, 1.0)]:
    pv = np.zeros((len(va), NC)); pt = np.zeros((5000, NC))
    for s in range(3):
        a, b = train_distill(Xre[fitr], yre[fitr], Q[fitr], [Xre[va], Xtr_], s, alpha, T)
        pv += a; pt += b
    tag = "hard labels only" if alpha == 1.0 else ("teacher only" if alpha == 0.0 else f"alpha={alpha} T={T}")
    log(f"  {tag:26s} ref_val {sc(pv, Mva, yre[va]):.4f}   comp_train {sc(pt, Mtr, ytr):.4f}")
