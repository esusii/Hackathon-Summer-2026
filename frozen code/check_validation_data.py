"""
Diagnostic only — run the moment the validation data lands (no model changes).

  - Do the new CSVs have the expected shape/columns?
  - Are the validation cells present in the public deposit (by ID / fingerprint)?
    (Determines whether the reference exclusion matters and how much context exists.)
  - Do their Section_IDs overlap the reference sections?
    (Determines whether the spatial histograms will be populated or empty.)
  - Any new metadata categories the one-hot alignment will drop?
"""
import numpy as np
import pandas as pd

mtr = pd.read_csv("data/meta_train.csv", index_col=0)
mte = pd.read_csv("data/meta_test.csv", index_col=0)
ctr = pd.read_csv("data/counts_train.csv", index_col=0)
cte = pd.read_csv("data/counts_test.csv", index_col=0)
for d in (mtr, mte, ctr, cte):
    d.index = d.index.astype(str)

print(f"test cells: {len(mte)}   counts shape: {cte.shape}")
print(f"columns match train: {list(cte.columns) == list(ctr.columns)}")
print(f"meta columns: {list(mte.columns)}")
print(f"counts/meta rows aligned: {cte.index.equals(mte.index)}")

try:
    import anndata as ad
    A = ad.read_h5ad("external/MERFISH_spinal_cord_resolved_0718.h5ad", backed="r")
    dep_ids = set(str(x) for x in A.obs_names)
    in_dep = mte.index.isin(dep_ids).sum()
    print(f"\nvalidation cells present in deposit BY ID: {in_dep}/{len(mte)}")
    dep_sections = set(str(x) for x in A.obs["Section ID"].unique())
    sec_overlap = mte["Section_ID"].astype(str).isin(dep_sections).sum()
    print(f"validation cells whose Section_ID exists in reference: {sec_overlap}/{len(mte)}")
    print("  -> spatial histograms will be POPULATED for those cells, EMPTY for the rest")
except Exception as e:
    print(f"\n(deposit check skipped: {e})")

for c in ["Mouse_ID", "Gender", "Excitatory_vs_Inhibitory", "Region", "Segment", "Datasets"]:
    if c in mte.columns:
        new_vals = set(mte[c].astype(str)) - set(mtr[c].astype(str))
        if new_vals:
            print(f"NEW categories in {c}: {sorted(new_vals)[:6]} "
                  f"({mte[c].astype(str).isin(new_vals).sum()} cells) — one-hot will drop these (handled)")

tot = cte.sum(axis=1)
print(f"\ntranscripts/cell: median {tot.median():.0f} (train was 21) — "
      f"large shifts here change expected accuracy")
print("\nIf all checks look sane: run hackathon_model_v2.py, then make_final_blend.py.")
