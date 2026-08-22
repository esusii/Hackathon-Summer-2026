"""Stage 1 - build and cache the full feature set for v3.

Integrity: every one of the 10,000 competition cell IDs is removed from the
reference before anything else, then exact 200-gene count-vector duplicates are
removed. No competition label is ever read out of the reference.
"""
import hashlib
import os
import time

import anndata as ad
import numpy as np
from common import DATA_ROOT
import pandas as pd
import scipy.spatial as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ROOT = DATA_ROOT
DATA = os.path.join(ROOT, "data")
H5AD = os.path.join(ROOT, "external", "MERFISH_spinal_cord_resolved_0718.h5ad")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE, exist_ok=True)

LABEL = "MERFISH_cell_type_annotation"
N_PCA = 64
K_EXPR = 50          # expression-space neighbour histogram
K_SPATIAL = (15, 60) # spatial neighbour histograms, two scales
K_NBR_EXPR = 20      # spatial neighbourhood mean expression
N_NBR_PCA = 32

# --------------------------------------------------------------- competition data
counts_train = pd.read_csv(os.path.join(DATA, "counts_train.csv"), index_col=0)
meta_train = pd.read_csv(os.path.join(DATA, "meta_train.csv"), index_col=0)
counts_test = pd.read_csv(os.path.join(DATA, "counts_test.csv"), index_col=0)
meta_test = pd.read_csv(os.path.join(DATA, "meta_test.csv"), index_col=0)
for d in (counts_train, meta_train, counts_test, meta_test):
    d.index = d.index.astype(str)
GENES = list(counts_train.columns)
assert list(counts_test.columns) == GENES
log(f"competition: {len(meta_train)} train, {len(meta_test)} test, {len(GENES)} genes")

# ------------------------------------------------------------------- reference
assert hashlib.md5(open(H5AD, "rb").read()).hexdigest() == "ce06f62c0ec4973581dae17bb76f0cd9"
A = ad.read_h5ad(H5AD)
A.obs.index = [str(x) for x in A.obs.index]

comp_ids = set(meta_train.index) | set(meta_test.index)
assert len(comp_ids) == 10000
in_comp = np.array([c in comp_ids for c in A.obs.index])
assert int(in_comp.sum()) == 10000, f"only {in_comp.sum()} competition IDs found in reference"

gi = [list(A.var_names).index(g) for g in GENES]
def dense(rows):
    X = A.X[np.ix_(rows, gi)]
    X = X.todense() if hasattr(X, "todense") else X
    return np.rint(np.asarray(X)).astype(np.int32)

ref_rows = np.flatnonzero(~in_comp)
RX = dense(ref_rows)
comp_fp = {r.tobytes() for r in
           np.vstack([counts_train.values, counts_test.values]).astype(np.int32)}
dup = np.array([r.tobytes() in comp_fp for r in RX])
ref_rows, RX = ref_rows[~dup], RX[~dup]
log(f"reference: {len(ref_rows)} cells (dropped 10000 competition IDs, {int(dup.sum())} duplicates)")

# ------------------------------------- translate reference metadata to our encoding
pos = {c: i for i, c in enumerate(A.obs.index)}
O_tr = A.obs.iloc[[pos[c] for c in meta_train.index]]
O_ref = A.obs.iloc[ref_rows]

def sarr(s):
    """Plain Python strings, with any missing value rendered as 'nan'.

    pandas 3.0's astype(str) keeps NA as missing rather than the literal
    string, which silently empties groups downstream.
    """
    out = []
    for x in np.asarray(s, dtype=object):
        if x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x)):
            out.append("nan")
        else:
            out.append(str(x))
    return np.array(out, dtype=object)


def learn_map(their_col, our_col):
    """Learn their-value -> our-value using competition TRAIN rows only."""
    df = pd.DataFrame({"a": sarr(O_tr[their_col]), "b": sarr(meta_train[our_col])})
    g = df.groupby("a", observed=True)["b"]
    purity = g.agg(lambda s: s.value_counts().iloc[0] / len(s))
    assert purity.min() == 1.0, f"{their_col}->{our_col} not a bijection (min purity {purity.min()})"
    m = g.agg(lambda s: s.value_counts().idxmax())
    print(f"    {their_col} -> {our_col}: {len(m)} levels, purity 1.000")
    return m

norm_label = lambda x: str(x).replace("-", "_").replace(" ", "_")

ref = pd.DataFrame(index=[f"REF_{i}" for i in range(len(ref_rows))])
for c in ("volume", "center_x", "center_y"):
    ref[c] = O_ref[c].values.astype(np.float64)
for our, their in [("Segment", "Laminae"), ("Region", "Region"),
                   ("AP_position", "Axial level"),
                   ("Excitatory_vs_Inhibitory", "Excitatory_vs_Inhibitory")]:
    mapped = pd.Series(sarr(O_ref[their])).map(learn_map(their, our))
    n_unmapped = int(mapped.isna().sum())
    assert n_unmapped == 0, f"{their}: {n_unmapped} reference values absent from competition train"
    ref[our] = mapped.values
ref["Mouse_ID"] = sarr(O_ref["Mouse ID"])
ref["Gender"] = sarr(O_ref["Gender"])
ref["Datasets"] = sarr(O_ref["Datasets"])
ref["Section_ID"] = sarr(O_ref["Section ID"])
ref[LABEL] = np.array([norm_label(x) for x in sarr(O_ref["MERFISH cell type annotation"])], dtype=object)

assert not (set(ref.index) & comp_ids)
CLASSES = np.array(sorted(set(sarr(meta_train[LABEL]))))
assert len(CLASSES) == 60
assert set(ref[LABEL]) <= set(CLASSES), set(ref[LABEL]) - set(CLASSES)
assert set(ref["Section_ID"]) == set(sarr(meta_train["Section_ID"]))
log("metadata translation verified (all bijections, labels inside taxonomy)")

# ----------------------------------------------------------------- normalisation
def norm_counts(C, target):
    tot = C.sum(axis=1).astype(np.float64)
    tot[tot == 0] = np.nan
    return np.nan_to_num(np.log1p(C / tot[:, None] * target)).astype(np.float32)

Ctr = counts_train.values.astype(np.float64)
Cte = counts_test.values.astype(np.float64)
Cre = RX.astype(np.float64)
TARGET = float(np.median(Ctr.sum(axis=1)))
Gtr, Gte, Gre = norm_counts(Ctr, TARGET), norm_counts(Cte, TARGET), norm_counts(Cre, TARGET)
log(f"normalised (target sum {TARGET:.0f})")

# QC features
def qc(C):
    tot = C.sum(axis=1)
    return np.column_stack([np.log1p(tot), (C > 0).sum(axis=1)]).astype(np.float32)
Qtr, Qte, Qre = qc(Ctr), qc(Cte), qc(Cre)

# ----------------------------------------------- reference-fit PCA (shared 200 genes)
mu, sd = Gre.mean(0), Gre.std(0)
sd[sd == 0] = 1.0
pca = PCA(n_components=N_PCA, random_state=42, svd_solver="randomized").fit((Gre - mu) / sd)
Ptr = pca.transform((Gtr - mu) / sd).astype(np.float32)
Pte = pca.transform((Gte - mu) / sd).astype(np.float32)
Pre = pca.transform((Gre - mu) / sd).astype(np.float32)
log(f"reference PCA-{N_PCA} fit (explains {pca.explained_variance_ratio_.sum():.3f})")

# --------------------------------------------------------------- label bookkeeping
cix = {c: i for i, c in enumerate(CLASSES)}
ref_code = np.array([cix[v] for v in ref[LABEL]], dtype=np.int32)
NC = len(CLASSES)

# --------------------------------------------- expression neighbour histograms
nn = NearestNeighbors(n_neighbors=K_EXPR + 1, n_jobs=-1, algorithm="brute").fit(Pre)

def expr_hist(P, is_ref):
    out = np.zeros((len(P), NC + 3), dtype=np.float32)
    step = 20000
    for s in range(0, len(P), step):
        d, j = nn.kneighbors(P[s:s + step])
        if is_ref:
            d, j = d[:, 1:], j[:, 1:]
        else:
            d, j = d[:, :K_EXPR], j[:, :K_EXPR]
        w = 1.0 / (1.0 + d)
        lab = ref_code[j]
        h = np.zeros((len(d), NC), dtype=np.float32)
        np.add.at(h, (np.repeat(np.arange(len(d)), lab.shape[1]), lab.ravel()), w.ravel())
        h /= np.maximum(h.sum(1, keepdims=True), 1e-9)
        out[s:s + step, :NC] = h
        out[s:s + step, NC] = d.mean(1)
        out[s:s + step, NC + 1] = d[:, 0]
        out[s:s + step, NC + 2] = h.max(1)
        log(f"  expr_hist {'ref' if is_ref else 'comp'} {s + len(d)}/{len(P)}")
    return out

Etr, Ete = expr_hist(Ptr, False), expr_hist(Pte, False)
Ere = expr_hist(Pre, True)
log("expression neighbour histograms done")

# ------------------------------------------------- spatial histograms + neighbourhood
ref_sec = ref["Section_ID"].values
ref_xy = ref[["center_x", "center_y"]].values
sec_index = {s: np.flatnonzero(ref_sec == s) for s in np.unique(ref_sec)}
trees = {s: sp.cKDTree(ref_xy[i]) for s, i in sec_index.items()}

def spatial_feats(meta, is_ref):
    n = len(meta)
    hists = {k: np.zeros((n, NC + 1), dtype=np.float32) for k in K_SPATIAL}
    nbr_pca = np.zeros((n, N_NBR_PCA), dtype=np.float32)
    sec_prior = np.zeros((n, NC), dtype=np.float32)
    secs = sarr(meta["Section_ID"])
    xy = meta[["center_x", "center_y"]].values
    row_of = {c: i for i, c in enumerate(meta.index)}
    for s in np.unique(secs):
        rows = np.flatnonzero(secs == s)
        ridx = sec_index[s]
        rlab = ref_code[ridx]
        tree = trees[s]
        # section-level class prior from reference cells (self removed for ref rows)
        cnt = np.bincount(rlab, minlength=NC).astype(np.float32)
        for k in K_SPATIAL:
            kk = min(k + (1 if is_ref else 0), len(ridx))
            d, j = tree.query(xy[rows], k=kk, workers=-1)
            d, j = np.atleast_2d(d), np.atleast_2d(j)
            if is_ref and d.shape[1] > 1:
                d, j = d[:, 1:], j[:, 1:]
            w = 1.0 / (1.0 + d)
            lab = rlab[j]
            h = np.zeros((len(rows), NC), dtype=np.float32)
            np.add.at(h, (np.repeat(np.arange(len(rows)), lab.shape[1]), lab.ravel()), w.ravel())
            h /= np.maximum(h.sum(1, keepdims=True), 1e-9)
            hists[k][rows, :NC] = h
            hists[k][rows, NC] = d.mean(1)
        kk = min(K_NBR_EXPR + (1 if is_ref else 0), len(ridx))
        d, j = tree.query(xy[rows], k=kk, workers=-1)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        if is_ref and d.shape[1] > 1:
            j = j[:, 1:]
        nbr_pca[rows] = Pre[ridx[j]][:, :, :N_NBR_PCA].mean(axis=1)
        sec_prior[rows] = cnt
        if is_ref:
            # a reference cell must not count itself in its own section prior
            sec_prior[rows, rlab] -= 1.0
        sec_prior[rows] /= np.maximum(sec_prior[rows].sum(1, keepdims=True), 1e-9)
    return hists, nbr_pca, sec_prior

Htr, Ntr, Str = spatial_feats(meta_train, False)
log("spatial train done")
Hte, Nte, Ste = spatial_feats(meta_test, False)
log("spatial test done")
Hre, Nre, Sre = spatial_feats(ref, True)
log("spatial ref done")

# ------------------------------------------------------------------ metadata block
def meta_block(m):
    out = pd.DataFrame(index=m.index)
    out["volume"] = pd.to_numeric(m["volume"], errors="coerce")
    out["log_volume"] = np.log1p(out["volume"])
    out["center_x"] = pd.to_numeric(m["center_x"], errors="coerce")
    out["center_y"] = pd.to_numeric(m["center_y"], errors="coerce")
    out["AP_position"] = pd.to_numeric(m["AP_position"], errors="coerce")
    reg = pd.to_numeric(m["Region"], errors="coerce")
    seg = pd.to_numeric(m["Segment"], errors="coerce")
    out["Region"] = reg.fillna(-1)
    out["Segment"] = seg.fillna(-1)
    ei = pd.Series(sarr(m["Excitatory_vs_Inhibitory"]), index=m.index)
    out["EI"] = ei.map({"excitatory": 1, "inhibitory": 0}).fillna(-1)
    out["Gender"] = (sarr(m["Gender"]) == "male").astype(int)
    out["Mouse_ID"] = pd.Categorical(sarr(m["Mouse_ID"]),
                                     categories=sorted(set(sarr(meta_train["Mouse_ID"])))).codes
    out["Datasets"] = pd.Categorical(sarr(m["Datasets"]),
                                     categories=sorted(set(sarr(meta_train["Datasets"])))).codes
    out["Section_code"] = pd.Categorical(sarr(m["Section_ID"]),
                                         categories=sorted(set(sarr(meta_train["Section_ID"])))).codes
    return out

Mtr, Mte, Mre = meta_block(meta_train), meta_block(meta_test), meta_block(ref)

# section-normalised coordinates (tissue is positioned differently per section)
allxy = pd.concat([Mtr[["center_x", "center_y"]].assign(s=sarr(meta_train["Section_ID"])),
                   Mte[["center_x", "center_y"]].assign(s=sarr(meta_test["Section_ID"])),
                   Mre[["center_x", "center_y"]].assign(s=sarr(ref["Section_ID"]))])
gstat = allxy.groupby("s")[["center_x", "center_y"]].agg(["mean", "std"])
def add_rel(M, secs):
    s = pd.Series(secs, index=M.index)
    for c in ("center_x", "center_y"):
        mu_ = s.map(gstat[(c, "mean")]).astype(float)
        sd_ = s.map(gstat[(c, "std")]).astype(float).replace(0, 1)
        M[c + "_rel"] = ((M[c] - mu_) / sd_).values
    M["radius_rel"] = np.sqrt(M["center_x_rel"] ** 2 + M["center_y_rel"] ** 2)
    return M
Mtr = add_rel(Mtr, sarr(meta_train["Section_ID"]))
Mte = add_rel(Mte, sarr(meta_test["Section_ID"]))
Mre = add_rel(Mre, sarr(ref["Section_ID"]))

META_COLS = list(Mtr.columns)
log(f"metadata block: {META_COLS}")

# ------------------------------------------------------------------- assemble
def assemble(G, P, Q, E, H, N, S, M):
    return np.hstack([G, P, Q, E] + [H[k] for k in K_SPATIAL] + [N, S, M.values.astype(np.float32)]
                     ).astype(np.float32)

Xtr = assemble(Gtr, Ptr, Qtr, Etr, Htr, Ntr, Str, Mtr)
Xte = assemble(Gte, Pte, Qte, Ete, Hte, Nte, Ste, Mte)
Xre = assemble(Gre, Pre, Qre, Ere, Hre, Nre, Sre, Mre)

names = ([f"g_{g}" for g in GENES] + [f"pca_{i}" for i in range(N_PCA)]
         + ["qc_logtotal", "qc_ndet"]
         + [f"eh_{c}" for c in CLASSES] + ["eh_meand", "eh_d1", "eh_max"])
for k in K_SPATIAL:
    names += [f"sh{k}_{c}" for c in CLASSES] + [f"sh{k}_meand"]
names += [f"nb_{i}" for i in range(N_NBR_PCA)] + [f"sp_{c}" for c in CLASSES] + META_COLS
assert len(names) == Xtr.shape[1], (len(names), Xtr.shape[1])

ytr = np.array([cix[v] for v in sarr(meta_train[LABEL])], dtype=np.int32)
yre = ref_code

np.savez_compressed(
    os.path.join(CACHE, "features.npz"),
    Xtr=Xtr, Xte=Xte, Xre=Xre, ytr=ytr, yre=yre,
    classes=CLASSES, names=np.array(names),
    train_ids=np.array(meta_train.index), test_ids=np.array(meta_test.index),
    tr_sec=sarr(meta_train["Section_ID"]).astype(str),
    te_sec=sarr(meta_test["Section_ID"]).astype(str),
    re_sec=sarr(ref["Section_ID"]).astype(str),
)
# route keys for the anatomy mask
def route_key(m):
    return np.array([f"{a}|{b}|{c}" for a, b, c in
                     zip(sarr(m["Region"]), sarr(m["Excitatory_vs_Inhibitory"]), sarr(m["Segment"]))],
                    dtype=object).astype(str)
np.savez_compressed(os.path.join(CACHE, "routes.npz"),
                    rtr=route_key(meta_train), rte=route_key(meta_test), rre=route_key(ref))
log(f"cached: Xtr {Xtr.shape} Xte {Xte.shape} Xre {Xre.shape}")
