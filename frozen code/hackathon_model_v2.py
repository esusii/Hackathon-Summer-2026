"""
MERFISH cell type prediction v2 — University of Rochester Biomedical DS Hackathon 2026

v1 (LDA+RF+ET blend on the four provided CSVs): leaderboard 0.7638.
v2 adds EXTERNAL PUBLIC DATA, permitted by the organizers (confirmed 2026-08-19):
the public Zenodo deposit behind the source atlas paper (Wang ... Meltzer,
bioRxiv 2026.01.10.698734; Zenodo record 18039571). The competition data is a
10,000-cell subset of that deposit. We use the OTHER 136,574 cells as extra
labelled training examples.

INTEGRITY: all 10,000 competition cell IDs are excluded from the reference BEFORE
anything else, plus 47 cells whose 200-gene count vectors exactly duplicate
competition cells. No competition test cell's label is ever read. The withheld-cell
exclusion is verifiable: `assert not (set(ref.index) & set(comp_ids))`.

Pipeline:
  1. Download MERFISH_spinal_cord_resolved_0718.h5ad (99 MB; md5
     ce06f62c0ec4973581dae17bb76f0cd9) into ./external/
  2. Build the reference: exclude competition IDs, dedupe fingerprints, translate
     metadata into the competition encoding (mappings learned on TRAIN cells only;
     all are exact bijections, purity 1.000: Laminae->Segment, Region->Region,
     'Axial level'->AP_position, E/I->E/I; Section_ID/Mouse_ID/Gender/Datasets
     match verbatim).
  3. Build reference-derived neighbour-label histogram features for every cell:
     a 1/(1+d)-weighted 60-class histogram over (a) the 15 nearest reference cells
     in the same tissue section and (b) the 25 nearest reference cells in a PCA-50
     expression space fit on the reference. Reference labels are external data and
     always known, so these features are computed identically for train and test —
     no fold logic, no leakage. (Reference rows exclude themselves as neighbours.)
  4. Train the same LDA(0.10)/RF(0.65)/ET(0.25) blend on competition-train +
     reference rows, all with histogram features; predict the 5,000 test cells.

Validation (stratified 50/50 ShuffleSplit, the scheme that tracked the leaderboard
to within 0.4 pts): base 0.7679 -> +reference rows 0.7762 -> +histograms 0.8038,
confirmed on a held-out seed with 5/5 wins at each step.

Requires: anndata, h5py (pip install anndata h5py), plus the v1 stack.
"""

import hashlib
import os
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

LABEL = "MERFISH_cell_type_annotation"
NUMERIC_COLS = ["volume", "center_x", "center_y", "AP_position"]
CATEGORICAL_COLS = ["Mouse_ID", "Gender", "Excitatory_vs_Inhibitory",
                    "Region", "Segment", "Datasets"]
W_LDA, W_RF, W_ET = 0.10, 0.65, 0.25

ZENODO_URL = ("https://zenodo.org/api/records/18039571/files/"
              "MERFISH_spinal_cord_resolved_0718.h5ad/content")
H5AD = "external/MERFISH_spinal_cord_resolved_0718.h5ad"
H5AD_MD5 = "ce06f62c0ec4973581dae17bb76f0cd9"


# ------------------------------------------------------------ load competition data

counts_train = pd.read_csv("data/counts_train.csv", index_col=0)
meta_train = pd.read_csv("data/meta_train.csv", index_col=0)
counts_test = pd.read_csv("data/counts_test.csv", index_col=0)
meta_test = pd.read_csv("data/meta_test.csv", index_col=0)
for d in (counts_train, meta_train, counts_test, meta_test):
    d.index = d.index.astype(str)
GENES = list(counts_train.columns)


# ------------------------------------------------------- download + build reference

os.makedirs("external", exist_ok=True)
if not os.path.exists(H5AD):
    print("downloading reference from Zenodo (99 MB)...")
    urllib.request.urlretrieve(ZENODO_URL, H5AD)
assert hashlib.md5(open(H5AD, "rb").read()).hexdigest() == H5AD_MD5, "bad download"

A = ad.read_h5ad(H5AD)
A.obs.index = [str(x) for x in A.obs.index]

# STEP 1 — exclude every competition cell BY ID, before anything else
comp_ids = set(meta_train.index) | set(meta_test.index)
keep = np.array([c not in comp_ids for c in A.obs.index])

gi = [list(A.var_names).index(g) for g in GENES]
def dense(rows):
    X = A.X[np.ix_(rows, gi)]
    return np.rint(np.asarray(X.todense() if hasattr(X, "todense") else X)).astype(np.int32)

# STEP 2 — drop cells whose 200-gene count vector exactly matches a competition cell
comp_fp = {r.tobytes() for r in
           np.vstack([counts_train.values, counts_test.values]).astype(np.int32)}
ref_rows = np.flatnonzero(keep)
RX = dense(ref_rows)
dup = np.array([r.tobytes() in comp_fp for r in RX])
ref_rows, RX = ref_rows[~dup], RX[~dup]
print(f"reference: {len(ref_rows)} cells "
      f"(excluded {int((~keep).sum())} competition IDs, {int(dup.sum())} duplicates)")

# STEP 3 — translate reference metadata into the competition's encoding.
# Mappings are learned from competition TRAIN cells only; each is a 1:1 bijection.
pos = {c: i for i, c in enumerate(A.obs.index)}
tr_rows = [pos[c] for c in meta_train.index]
O_tr, O_ref = A.obs.iloc[tr_rows], A.obs.iloc[ref_rows]

def learn_map(their_col, our_col):
    df = pd.DataFrame({"a": O_tr[their_col].astype(str).values,
                       "b": meta_train[our_col].astype(str).values})
    g = df.groupby("a", observed=True)["b"]
    purity = g.agg(lambda s: s.value_counts().iloc[0] / len(s))
    assert purity.min() == 1.0, f"{their_col} -> {our_col} is not a clean bijection"
    return g.agg(lambda s: s.value_counts().idxmax())

norm_label = lambda x: str(x).replace("-", "_").replace(" ", "_")
ref = pd.DataFrame(index=[f"REF_{i}" for i in range(len(ref_rows))])
for c in ("volume", "center_x", "center_y"):
    ref[c] = O_ref[c].values
for our, their in [("Segment", "Laminae"), ("Region", "Region"),
                   ("AP_position", "Axial level"),
                   ("Excitatory_vs_Inhibitory", "Excitatory_vs_Inhibitory")]:
    ref[our] = pd.Series(O_ref[their].astype(str).values).map(
        learn_map(their, our)).fillna("Missing").values
ref["Mouse_ID"] = O_ref["Mouse ID"].astype(str).values
ref["Gender"] = O_ref["Gender"].astype(str).values
ref["Datasets"] = O_ref["Datasets"].astype(str).values
ref["Section_ID"] = O_ref["Section ID"].astype(str).values
ref[LABEL] = O_ref["MERFISH cell type annotation"].astype(str).map(norm_label).values
ref_counts = pd.DataFrame(RX, index=ref.index, columns=GENES)

assert not (set(ref.index) & comp_ids)
assert set(ref[LABEL]) <= set(meta_train[LABEL]), "label outside competition taxonomy"


# ------------------------------------------- reference-derived histogram features

import scipy.spatial as _sp
from sklearn.decomposition import PCA

K_SP, K_EX = 15, 25
CLASSES = np.array(sorted(meta_train[LABEL].unique()))
NCLS = len(CLASSES)
_cix = {c: i for i, c in enumerate(CLASSES)}
ref_code = np.array([_cix[v] for v in ref[LABEL]])


def spatial_hist(meta, is_ref):
    """1/(1+d)-weighted class histogram over the K_SP nearest reference cells in
    the same section (+ mean neighbour distance). Reference rows drop themselves."""
    F = np.zeros((len(meta), NCLS + 1), dtype=np.float32)
    ix = {c: i for i, c in enumerate(meta.index)}
    for s_, g in meta.groupby("Section_ID"):
        rmask = (ref["Section_ID"] == s_).values
        rXY = ref.loc[rmask, ["center_x", "center_y"]].values
        rlab = ref_code[rmask]
        if len(rXY) == 0:
            continue
        tree = _sp.cKDTree(rXY)
        k = min(K_SP + (1 if is_ref else 0), len(rXY))
        d, j = tree.query(g[["center_x", "center_y"]].values, k=k)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        if is_ref and d.shape[1] > 1:
            d, j = d[:, 1:], j[:, 1:]
        for r, cell in enumerate(g.index):
            w = 1.0 / (1.0 + d[r])
            h = np.zeros(NCLS, dtype=np.float32)
            np.add.at(h, rlab[j[r]], w)
            tot = h.sum()
            F[ix[cell], :NCLS] = h / tot if tot > 0 else 0
            F[ix[cell], NCLS] = d[r].mean()
    return F


# expression histograms: neighbours in a PCA-50 space fit on the reference
_gm, _gs = None, None  # set after normalization below

# ------------------------------------------------------------------------- features

def normalize_counts(counts, target_sum):
    total = counts.sum(axis=1).replace(0, np.nan)
    return np.log1p(counts.div(total, axis=0) * target_sum).fillna(0.0)

median_total = counts_train.sum(axis=1).median()
g_train = normalize_counts(counts_train, median_total)
g_test = normalize_counts(counts_test, median_total)
g_ref = normalize_counts(ref_counts, median_total)

_gm, _gs = g_ref.mean(), g_ref.std().replace(0, 1)
_pca = PCA(n_components=50, random_state=42).fit(((g_ref - _gm) / _gs).values)
_nn = NearestNeighbors(n_neighbors=K_EX + 1, n_jobs=-1).fit(
    _pca.transform(((g_ref - _gm) / _gs).values))


def expr_hist(G, is_ref):
    d, j = _nn.kneighbors(_pca.transform(((G - _gm) / _gs).values))
    d, j = (d[:, 1:], j[:, 1:]) if is_ref else (d[:, :K_EX], j[:, :K_EX])
    F = np.zeros((len(G), NCLS + 1), dtype=np.float32)
    for r in range(len(G)):
        w = 1.0 / (1.0 + d[r])
        h = np.zeros(NCLS, dtype=np.float32)
        np.add.at(h, ref_code[j[r]], w)
        F[r, :NCLS] = h / h.sum()
        F[r, NCLS] = d[r].mean()
    return F


H_train = np.hstack([spatial_hist(meta_train, False), expr_hist(g_train, False)])
H_test = np.hstack([spatial_hist(meta_test, False), expr_hist(g_test, False)])
H_ref = np.hstack([spatial_hist(ref, True), expr_hist(g_ref, True)])


def local_density(meta_all, meta_target, k=10):
    out = pd.Series(np.nan, index=meta_target.index)
    for _, idx in meta_all.groupby("Section_ID").groups.items():
        idx = list(idx)
        targets = [i for i in idx if i in meta_target.index]
        if not targets or len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx)))
        nn.fit(meta_all.loc[idx, ["center_x", "center_y"]].values)
        dist, _ = nn.kneighbors(meta_all.loc[targets, ["center_x", "center_y"]].values)
        out.loc[targets] = dist[:, 1:].mean(axis=1)
    return out.fillna(out.median())

all_meta = pd.concat([meta_train, meta_test])
dens_train = local_density(all_meta, meta_train).rename("local_density")
dens_test = local_density(all_meta, meta_test).rename("local_density")
dens_ref = local_density(ref, ref).rename("local_density")  # reference sections stand alone


def build_metadata_features(meta):
    num = meta[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    cat = meta[CATEGORICAL_COLS].astype(str).fillna("Missing").replace("nan", "Missing")
    return pd.concat([num, pd.get_dummies(cat, prefix=CATEGORICAL_COLS)], axis=1)

mf_train = build_metadata_features(meta_train)
mf_test = build_metadata_features(meta_test)
mf_ref = build_metadata_features(ref)
for col in NUMERIC_COLS:
    fill = mf_train[col].median()
    for mf in (mf_train, mf_test, mf_ref):
        mf[col] = mf[col].fillna(fill)
mf_test = mf_test.reindex(columns=mf_train.columns, fill_value=0)
mf_ref = mf_ref.reindex(columns=mf_train.columns, fill_value=0)

X_train = pd.concat([g_train, mf_train, dens_train], axis=1)
X_test = pd.concat([g_test, mf_test, dens_test], axis=1)[X_train.columns]
X_ref = pd.concat([g_ref, mf_ref, dens_ref], axis=1)[X_train.columns]

X_fit = np.vstack([np.hstack([X_train.values, H_train]),
                   np.hstack([X_ref.values, H_ref])]).astype(np.float32)
y_fit = np.concatenate([meta_train[LABEL].values, ref[LABEL].values])
X_test_final = np.hstack([X_test.values, H_test]).astype(np.float32)
print(f"training on {X_fit.shape[0]:,} cells ({len(X_train):,} competition + {len(X_ref):,} reference)")


# ---------------------------------------------------------------------------- model

def fit_predict(Xf, yf, Xp):
    scaler = StandardScaler().fit(Xf)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(scaler.transform(Xf), yf)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1).fit(Xf, yf)
    et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                              class_weight="balanced_subsample",
                              random_state=42, n_jobs=-1).fit(Xf, yf)
    assert list(lda.classes_) == list(rf.classes_) == list(et.classes_)
    proba = (W_LDA * lda.predict_proba(scaler.transform(Xp))
             + W_RF * rf.predict_proba(Xp)
             + W_ET * et.predict_proba(Xp))
    np.save("v2_test_probs.npy", proba)   # consumed by make_final_blend.py
    return lda.classes_[proba.argmax(axis=1)]

test_pred = fit_predict(X_fit, y_fit, X_test_final)

submission = pd.DataFrame({
    "Cell_ID": meta_test.index,
    "MERFISH_cell_type_annotation.y": test_pred,
})
assert len(submission) == len(meta_test)
assert (submission["Cell_ID"].values == meta_test.index.values).all()
assert submission["MERFISH_cell_type_annotation.y"].isin(meta_train[LABEL].unique()).all()
submission.to_csv("prediction.csv", index=False)
print(f"wrote prediction.csv ({len(submission)} rows)")
print(submission["MERFISH_cell_type_annotation.y"].value_counts().head())
