"""MLP harness: configurable feature blocks, reference-trained, screened on the
5,000 competition train cells."""
import numpy as np
import torch

from common import CACHE, build_route_labels, load, log, route_mask


def onehot(codes, n):
    codes = np.asarray(codes).astype(int)
    out = np.zeros((len(codes), n), dtype=np.float32)
    valid = codes >= 0
    out[np.flatnonzero(valid), codes[valid]] = 1.0
    return out


class Blocks:
    """Assembles feature matrices for train/test/reference from named blocks."""

    def __init__(self, D):
        self.D = D
        self.names = D["names"]
        self.col = {n: i for i, n in enumerate(self.names)}
        self.CLASSES = D["classes"]
        self.NC = len(self.CLASSES)

    def _idx(self, pred):
        return [i for i, n in enumerate(self.names) if pred(n)]

    def get(self, which, blocks):
        X = self.D[{"tr": "Xtr", "te": "Xte", "re": "Xre"}[which]]
        parts = []
        c = self.col
        CL = self.CLASSES
        if "G" in blocks:
            parts.append(X[:, self._idx(lambda n: n.startswith("g_"))])
        if "P" in blocks:
            parts.append(X[:, self._idx(lambda n: n.startswith("pca_"))])
        if "Q" in blocks:
            parts.append(X[:, [c["qc_logtotal"], c["qc_ndet"]]])
        if "E" in blocks:
            parts.append(X[:, [c[f"eh_{k}"] for k in CL] + [c["eh_meand"], c["eh_d1"], c["eh_max"]]])
        if "S" in blocks:
            for k in (15, 60):
                parts.append(X[:, [c[f"sh{k}_{q}"] for q in CL] + [c[f"sh{k}_meand"]]])
        if "N" in blocks:
            parts.append(X[:, self._idx(lambda n: n.startswith("nb_"))])
        if "SP" in blocks:
            parts.append(X[:, [c[f"sp_{k}"] for k in CL]])
        if "M" in blocks:
            num = X[:, [c["volume"], c["log_volume"], c["AP_position"],
                        c["center_x_rel"], c["center_y_rel"], c["radius_rel"]]]
            parts.append(num)
            # categorical one-hots (codes are -1 for missing, so shift by 1)
            for name, n in [("Region", 7), ("Segment", 24), ("EI", 3),
                            ("Mouse_ID", 11), ("Datasets", 7), ("Section_code", 109)]:
                parts.append(onehot(X[:, c[name]] + 1, n))
            parts.append(X[:, [c["Gender"]]])
        return np.hstack(parts).astype(np.float32)


def train_mlp(Xfit, yfit, Xevals, NC, seed=0, epochs=60, hidden=(1024, 512, 256),
              dropout=0.30, lr=3e-3, wd=1e-4, bs=4096, label_smooth=0.02,
              val=None, verbose=True):
    """Returns softmax probabilities for each matrix in Xevals."""
    dev = "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xg = torch.tensor(Xfit, device=dev)
    yg = torch.tensor(yfit.astype(np.int64), device=dev)
    mu = Xg.mean(0, keepdim=True)
    sd = Xg.std(0, keepdim=True).clamp_min(1e-6)
    Xg = (Xg - mu) / sd

    layers, d = [], Xfit.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.GELU(),
                   torch.nn.Dropout(dropout)]
        d = h
    layers.append(torch.nn.Linear(d, NC))
    net = torch.nn.Sequential(*layers).to(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(Xg) // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)

    Vx = Vy = None
    if val is not None:
        Vx = (torch.tensor(val[0], device=dev) - mu) / sd
        Vy = torch.tensor(val[1].astype(np.int64), device=dev)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xg), device=dev)
        tot = 0.0
        for s in range(0, len(Xg), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(net(Xg[idx]), yg[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        if verbose and (ep % 20 == 19 or ep == epochs - 1):
            msg = f"    epoch {ep+1:3d} loss {tot/len(Xg):.4f}"
            if Vx is not None:
                net.eval()
                with torch.no_grad():
                    msg += f"  val_acc {float((net(Vx).argmax(1)==Vy).float().mean()):.4f}"
            log(msg)

    net.eval()
    out = []
    with torch.no_grad():
        for Xe in Xevals:
            Xe_ = (torch.tensor(Xe, device=dev) - mu) / sd
            ps = []
            for s in range(0, len(Xe_), 20000):
                ps.append(torch.softmax(net(Xe_[s:s + 20000]), 1).cpu().numpy())
            out.append(np.vstack(ps))
    del Xg, yg, net
    torch.cuda.empty_cache()
    return out
