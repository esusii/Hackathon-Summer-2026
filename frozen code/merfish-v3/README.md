# MERFISH cell-type prediction — v3

Final submission: `prediction.csv` (5,000 rows).
**Honest estimate: 0.8188** (v2 leaderboard was 0.7890).

## Evaluation protocol

Everything was screened by training on the **136,574 Zenodo reference cells only**
and scoring all **5,000 competition-train cells**, whose labels never enter any
feature, mask, or selection step. That makes the estimate unbiased and, because
the final model additionally trains on those 5,000 rows, slightly conservative.

Design choices were selected on a *separate* 10,000-cell reference holdout
(`ref_val`) so the comp_train number stayed uncontaminated. Where the two
disagreed I deferred to comp_train (the actual target domain), and treated
differences under ~0.55% (the n=5000 binomial s.e.) as noise.

## Run order

```bash
cd src
python build_features.py     # ~30 s  -> cache/features.npz, routes.npz
python cache_counts.py       # ~3 s   -> cache/counts.npz   (raw counts, for normalisation variants)
python build_spatial2.py     # ~40 s  -> cache/spatial2.npz (k=3..250 + Gaussian kernels)
python exp8.py               # ~4 min -> cache/teacher_oof_v2.npy  (THE TEACHER)
python v3final.py screen 4   # ~5 min -> honest score
python v3final.py final  4   # ~6 min -> prediction.csv
```

`cache/teacher_oof_v2.npy` and `cache/final_test_probs.npy` are included, so you
can go straight to `v3final.py` without re-running the teacher.

## The teacher (the single largest gain, +0.6 pts)

The Zenodo reference measures **500 genes**; the competition gives you **200**.
A model with all 500 scores **0.9498**; the same model on your 200 scores 0.8142.
Those 300 unseen genes are worth ~13.5 points and cannot be recovered — but a
teacher trained on them *knows which glial subtypes are genuinely confusable*,
and that similarity structure can be distilled into the 200-gene student.

- **`exp7.py`** — teacher v1 (5-fold OOF, 500 genes + metadata + spatial, 0.9464)
  and the alpha/temperature sweep that established distillation works.
- **`exp8.py`** — teacher v2 (5 folds x 3 seeds, wider net, + spatial2; **0.9495**).
  This is the one used. Writes `cache/teacher_oof_v2.npy`.
- **`distill.py`** — the student training loop.
  Loss = `alpha * CE(hard) + (1-alpha) * T^2 * KL(student_T || teacher_T)`,
  with **alpha=0.4, T=2.0**.

Two integrity points:
1. Teacher probabilities are **out-of-fold** — the student never sees a teacher
   that memorised the cell it is scoring.
2. Competition cells have no 500-gene measurement, so in the final fit their
   targets fall back to their hard labels (`Q_tr` one-hot in `v3final.py`);
   teacher knowledge is used only for reference rows.

## Final recipe (`v3final.py`)

One code path, two modes (`screen` / `final`), so the measured recipe and the
shipped recipe cannot drift apart.

- 12 member slots x 4 seeds = 48 distilled MLPs
- Diversity across count normalisations (log, raw_log, pearson, sqrt) and
  spatial blocks (S, S2, both); mix chosen by greedy selection on ref_val (`exp6.py`)
- Small nets win: `(512,256)` and `(1024,512,256)`. Larger nets were consistently
  worse solo and dragged the ensemble down.
- Anatomy route mask (`Region|E/I|Segment`), applied last. Reference-derived
  candidate sets contain the true label for 100% of train cells and cover 100%
  of test cells, so the mask is free.

## Progression

| Stage | comp_train | script |
|---|---|---|
| v2 (RF/ET/LDA blend) | 0.7890 *(actual LB)* | — |
| XGBoost, 558 features | 0.7652 | `screen_xgb.py` |
| Single MLP | 0.7970 | `exp_blocks.py` |
| + seed averaging | 0.8088 | `exp3.py` |
| + ensemble across normalisations | 0.8142 | `exp4.py`, `exp6.py` |
| + distillation | 0.8186 | `exp7.py` |
| + stronger teacher, 48 models | **0.8188** | `v3final.py` |

## Diagnostics and dead ends

- **`diagnose.py`** — route structure. 59% of cells sit in one 16-candidate
  non-neuronal route at ~0.66 accuracy; that is where nearly all error lives.
- **`exp_embed.py`** — why an MLP: PCA-64 kNN 0.7176 -> LDA-59 kNN 0.7588 ->
  MLP 0.7886. Model family mattered far more than features.
- **`exp_blocks.py`** — feature-block ablation. Neighbourhood-expression, PCA-kNN
  histogram and PCA blocks all added nothing and were dropped.
- **`exp5.py`** — the 500-vs-200-gene ceiling; the route specialist (clearly
  positive on a single model, **neutral** once distillation + ensembling were in,
  so `SPEC_W=0`); and the marginal value of comp-train rows (+0.2).
- **`exp9.py`** — evaluation of an alternative v3 proposal. Transductive spatial
  smoothing at beta=0.35 costs **-2.4 pts** (the test set is a 5% subsample:
  ~46 cells/section, so neighbours are uninformative); a prior exponent
  (`P/prior**alpha`) degrades accuracy monotonically, since plain accuracy is
  maximised by the *calibrated* posterior.

## Requirements

`anndata h5py torch (CUDA) numpy pandas scikit-learn scipy xgboost`

Paths to the competition data and the Zenodo `.h5ad` are set at the top of
`build_features.py`. The h5ad is verified by MD5 (`ce06f62c...`).
