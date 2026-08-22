"""Distilled MLP. Rows whose teacher distribution is a one-hot (competition
cells, which have no 500-gene measurement) fall back to plain hard-label
training; reference rows get the teacher's soft targets."""
import numpy as np
import torch


def train_distill(Xfit, yfit, Qfit, evals, NC, seed=0, alpha=0.4, T=2.0,
                  hidden=(512, 256), epochs=60, dropout=0.30, lr=3e-3,
                  wd=1e-4, label_smooth=0.02, bs=4096):
    dev = "cuda"
    torch.manual_seed(seed)
    Xg = torch.tensor(Xfit, device=dev)
    mu, sd = Xg.mean(0, keepdim=True), Xg.std(0, keepdim=True).clamp_min(1e-6)
    Xg = (Xg - mu) / sd
    yg = torch.tensor(yfit.astype(np.int64), device=dev)
    Qs = (torch.tensor(Qfit, device=dev).clamp_min(1e-8).log() / T).softmax(1)

    layers, d = [], Xfit.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.GELU(),
                   torch.nn.Dropout(dropout)]
        d = h
    layers.append(torch.nn.Linear(d, NC))
    net = torch.nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * (len(Xg) // bs + 1))
    ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)
    for _ in range(epochs):
        net.train()
        pm = torch.randperm(len(Xg), device=dev)
        for s in range(0, len(Xg), bs):
            i = pm[s:s + bs]
            if len(i) < 2:
                continue
            opt.zero_grad()
            z = net(Xg[i])
            loss = alpha * ce(z, yg[i]) + (1 - alpha) * (T * T) * torch.nn.functional.kl_div(
                torch.log_softmax(z / T, 1), Qs[i], reduction="batchmean")
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        for Xe in evals:
            Xe_ = (torch.tensor(Xe, device=dev) - mu) / sd
            out.append(torch.cat([torch.softmax(net(Xe_[s:s + 20000]), 1)
                                  for s in range(0, len(Xe_), 20000)]).cpu().numpy())
    del Xg, yg, Qs, net
    torch.cuda.empty_cache()
    return out
