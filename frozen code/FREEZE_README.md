# Team N=4! — frozen model (code freeze 2026-08-22, 3pm)

## Model

Geometric blend of two independent member models, then an anatomy route mask:

1. **`hackathon_model_v2.py`** (weight 0.60) — LDA(0.10)/RandomForest(0.65)/
   ExtraTrees(0.25) trained on competition training cells plus **136,574 externally
   labelled reference cells** (public Zenodo record 18039571,
   `MERFISH_spinal_cord_resolved_0718.h5ad`, MD5-verified in-script). **All cell IDs
   present in the competition data are excluded from the reference before any label
   is read** (asserted), plus 47 exact count-vector duplicates. Features: 200
   log-normalised genes, one-hot metadata, local density, reference-derived
   neighbour-label histograms (15 spatial + 25 expression neighbours,
   1/(1+d)-weighted). Exports `v2_test_probs.npy`.
2. **`merfish-v3/`** (Reuben's pipeline, weight 0.40) — 48 distilled MLPs over the
   same external reference: a 500-gene teacher (reference genes) distilled into the
   200-gene competition panel. Same ID-exclusion discipline. Exports
   `oof_probabilities.npz` (test_probs + route masks).
3. **Route mask** — a label is allowed for a cell iff it ever co-occurred with the
   cell's (Region | E/I | Segment) triple in training; provably contains the true
   label for 100% of labelled cells. Applied after blending.

Blend weight w = 0.40 and the mask were selected on `StratifiedShuffleSplit`
seed 0 and confirmed untouched on seed 777 (CV 0.8086 -> 0.8227, 5/5 splits).

## Run order (validation runbook)

```
pip install numpy pandas scikit-learn anndata h5py
python check_validation_data.py        # 1 min, diagnostics only
python hackathon_model_v2.py           # ~30 min -> v2_test_probs.npy
# merfish-v3 pipeline (see its README) -> oof_probabilities.npz
python make_final_blend.py             # -> prediction.csv
```

Both member pipelines re-derive the reference exclusion from whatever
`data/meta_test.csv` currently contains — replacing the test CSVs with validation
data requires no code changes.

**Declared fallbacks** (frozen as part of the model): if `oof_probabilities.npz`
cannot be produced, `make_final_blend.py` submits P alone. The merfish-v3 pipeline
carries its own internal fallback ladder (see its README).

## Validation protocol

Every modelling decision this week was selected on seed 0 and confirmed with
parameters fixed on seed 777 before adoption (~60 configurations tested; the
survivors are what is described above).
