"""Is a supervised embedding a better space for label transfer than PCA-64?

Everything is fit on reference cells only and scored on the 5,000 competition
train cells, whose labels are never used upstream.
"""
import os, sys

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_route_labels, load, log, route_mask

D = load()
CLASSES, names = D["classes"], D["names"]
NC = len(CLASSES)
ytr, yre = D["ytr"], D["yre"]
col = {n: i for i, n in enumerate(names)}
gcols = [col[n] for n in names if n.startswith("g_")]
pcols = [col[n] for n in names if n.startswith("pca_")]
Gre, Gtr = D["Xre"][:, gcols], D["Xtr"][:, gcols]
Pre, Ptr = D["Xre"][:, pcols], D["Xtr"][:, pcols]
route_labels = build_route_labels(D["rre"], yre)
M = route_mask(D["rtr"], route_labels, NC)
log(f"gene block {Gre.shape}, pca block {Pre.shape}")


def knn_hist(Ere, Etr, k, name):
    """Distance-weighted reference-label histogram in a given embedding."""
    Ere = np.ascontiguousarray(Ere, dtype=np.float32)
    Etr = np.ascontiguousarray(Etr, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=-1).fit(Ere)
    d, j = nn.kneighbors(Etr)
    w = 1.0 / (1.0 + d)
    lab = yre[j]
    h = np.zeros((len(Etr), NC), dtype=np.float32)
    np.add.at(h, (np.repeat(np.arange(len(Etr)), k), lab.ravel()), w.ravel())
    h /= np.maximum(h.sum(1, keepdims=True), 1e-9)
    raw = float((h.argmax(1) == ytr).mean())
    hm = h.copy(); hm[~M] = -1
    log(f"  {name:46s} k={k:4d}  raw {raw:.4f}  masked {float((hm.argmax(1)==ytr).mean()):.4f}")
    return h


log("baseline: unsupervised PCA-64 space")
for k in (25, 50, 100, 200):
    knn_hist(Pre, Ptr, k, "PCA-64")

log("supervised: LDA fit on reference labels")
lda = LinearDiscriminantAnalysis(solver="svd", n_components=NC - 1).fit(Gre, yre)
Lre, Ltr = lda.transform(Gre).astype(np.float32), lda.transform(Gtr).astype(np.float32)
log(f"  LDA space {Lre.shape}")
for k in (25, 50, 100, 200):
    knn_hist(Lre, Ltr, k, "LDA-59")
pl = lda.predict_proba(Gtr)
plm = pl.copy(); plm[~M] = -1
log(f"  LDA direct predict_proba                     raw {float((pl.argmax(1)==ytr).mean()):.4f}  "
    f"masked {float((plm.argmax(1)==ytr).mean()):.4f}")

# supervised neural embedding: train a small MLP on the reference, use its
# penultimate layer as the metric space
log("supervised: MLP embedding trained on reference")
dev = "cuda"
Xg = torch.tensor(Gre, device=dev)
yg = torch.tensor(yre.astype(np.int64), device=dev)
mu, sd = Xg.mean(0, keepdim=True), Xg.std(0, keepdim=True).clamp_min(1e-6)
Xg = (Xg - mu) / sd
torch.manual_seed(0)
EMB = 64
net = torch.nn.Sequential(
    torch.nn.Linear(Gre.shape[1], 512), torch.nn.BatchNorm1d(512), torch.nn.GELU(), torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), torch.nn.GELU(), torch.nn.Dropout(0.2),
    torch.nn.Linear(256, EMB), torch.nn.BatchNorm1d(EMB), torch.nn.GELU(),
    torch.nn.Linear(EMB, NC),
).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=40 * (len(Xg) // 4096 + 1))
lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
for ep in range(40):
    net.train()
    perm = torch.randperm(len(Xg), device=dev)
    tot = 0.0
    for s in range(0, len(Xg), 4096):
        idx = perm[s:s + 4096]
        if len(idx) < 2:
            continue
        opt.zero_grad()
        loss = lossf(net(Xg[idx]), yg[idx])
        loss.backward(); opt.step(); sched.step()
        tot += float(loss) * len(idx)
    if ep % 10 == 9:
        log(f"    epoch {ep+1} train loss {tot/len(Xg):.4f}")

net.eval()
emb_net = torch.nn.Sequential(*list(net.children())[:-1])
with torch.no_grad():
    Ere_ = emb_net(Xg).cpu().numpy()
    Xt = (torch.tensor(Gtr, device=dev) - mu) / sd
    Etr_ = emb_net(Xt).cpu().numpy()
    pm_ = torch.softmax(net(Xt), 1).cpu().numpy()
pmm = pm_.copy(); pmm[~M] = -1
log(f"  MLP direct softmax                           raw {float((pm_.argmax(1)==ytr).mean()):.4f}  "
    f"masked {float((pmm.argmax(1)==ytr).mean()):.4f}")
for k in (25, 50, 100, 200):
    knn_hist(Ere_, Etr_, k, "MLP-64 embedding")
