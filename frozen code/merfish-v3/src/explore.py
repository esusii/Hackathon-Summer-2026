import anndata as ad
import numpy as np
import pandas as pd

DATA = r"C:\Users\noron\Downloads\Hackathon-Summer-2026-main\Hackathon-Summer-2026-main\data"
H5AD = r"C:\Users\noron\Downloads\Hackathon-Summer-2026-main\Hackathon-Summer-2026-main\external\MERFISH_spinal_cord_resolved_0718.h5ad"

ct = pd.read_csv(DATA + r"\counts_train.csv", index_col=0)
mt = pd.read_csv(DATA + r"\meta_train.csv", index_col=0)
cs = pd.read_csv(DATA + r"\counts_test.csv", index_col=0)
ms = pd.read_csv(DATA + r"\meta_test.csv", index_col=0)
for d in (ct, mt, cs, ms):
    d.index = d.index.astype(str)

print("counts_train", ct.shape, "meta_train", mt.shape)
print("counts_test", cs.shape, "meta_test", ms.shape)
print("\nmeta_train columns:", list(mt.columns))
print("\nmeta_train dtypes:\n", mt.dtypes)
print("\nmeta_train head:\n", mt.head(3).to_string())
print("\nn classes:", mt["MERFISH_cell_type_annotation"].nunique())
print("\nclass counts (tail 10):\n", mt["MERFISH_cell_type_annotation"].value_counts().tail(10))
print("\nn sections train:", mt["Section_ID"].nunique(), "test:", ms["Section_ID"].nunique())
for c in ["Region", "Segment", "Excitatory_vs_Inhibitory", "Datasets", "Gender", "Mouse_ID"]:
    print(f"\n{c} ({mt[c].nunique()}):", sorted(map(str, mt[c].unique()))[:25])

print("\n=== reference ===")
A = ad.read_h5ad(H5AD)
print(A)
print("\nobs columns:", list(A.obs.columns))
print("\nobs head:\n", A.obs.head(3).to_string())
print("\nvar head:\n", A.var.head(3).to_string())
print("\nn var:", A.n_vars, "n obs:", A.n_obs)
print("\nX dtype/type:", type(A.X), getattr(A.X, "dtype", None))
sub = A.X[:5, :10]
print("X sample:\n", np.asarray(sub.todense() if hasattr(sub, "todense") else sub))
print("\ngenes match:", set(ct.columns) <= set(map(str, A.var_names)), len(ct.columns))
print("\nlayers:", list(A.layers.keys()) if A.layers else None)
print("obsm:", list(A.obsm.keys()) if A.obsm is not None else None)
for c in A.obs.columns:
    u = A.obs[c].astype(str).unique()
    print(f"  {c}: {len(u)} uniq, sample {list(u[:8])}")
