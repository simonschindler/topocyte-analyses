# Liver pathology across density slices

How do the pathologies of the GTEx liver cohort sit in the fPCA space of
persistent homology, and how does that change as the cells are thresholded by
density? All 594 GTEx liver slides with cells (126,219,124 cells) are each
assigned one pathology group by severity from the cohort table; the cells are
thresholded at the ten pooled deciles of the 10-nearest-neighbour density, as
in [delaunay-density-slices](../delaunay-density-slices/README.md), and at each
level the alpha persistent homology of every slide's kept cells gives an H0 and
an H1 Betti curve. Four fPCA models per level (H0 and H1, raw and cosine), nine
components each, are plotted as fractions of variance explained and as every
pair of the first four components and a UMAP of all nine, coloured by
pathology. The scores of all slices are then combined into one space by
multiple factor analysis, for all 40 slice-by-model blocks and for each model
family across its ten slices, and the separation of cirrhosis from healthy
along the combined PC2 is checked against the densest-decile share of cells.

The study is descriptive: it fits no classifier and draws no verdict.
`progress.md` records every ruling, including the group rule Simon chose;
`report.html` holds the figures.

## Run

Needs the cohort in `~/data/gtex_liver_cells/gtex_liver_cells`
(`gtex_liver_cells.h5ad` and `liver_cohort.csv`, opened read only; the paths are
`COHORT` in `run_pathology_slices.py`) and a Python environment with topocyte's
dependencies; the run used the topocyte checkout's `.venv`. Set `PYTHONPATH` to
a topocyte checkout at the commit below; `run_pathology_slices.py` refuses a
topocyte that comes from anywhere else.

| scripts | topocyte commit | run |
|---|---|---|
| `run_pathology_slices.py groups`, `density`, `slice 0`...`9`; `umap_scores.py` (umap-learn 0.5.12), `combined_spaces.py`, `pc2_density_check.py` | `bd1d5c0` | 2026-11-18 |

```bash
export PYTHONPATH=/path/to/topocyte-bd1d5c0
PY=/path/to/topocyte/.venv/bin/python
$PY run_pathology_slices.py groups
$PY run_pathology_slices.py density
for k in 0 1 2 3 4 5 6 7 8 9; do $PY run_pathology_slices.py slice $k; done
$PY umap_scores.py
$PY combined_spaces.py
$PY pc2_density_check.py
$PY verify_results.py
$PY build_report.py
```

Each slice runs in its own process; slice 0, all cells, is the largest. The
script calls topocyte's core directly (`tc.ph.compute_persistent_homology` per
slide, `tc.ph.FPCA` on the slides' results), which is what
`tc.tl.persistent_homology` and `tc.tl.fpca` run, because an AnnData of 126
million cells would build a string index of 126 million names. Each run's
runtime is in its `results/*.json`.

## Outputs

Not in git: `results/*.npz` (`results/density.npz`, the per-cell density, is
1.3 GB), `logs/` and `report.html`. `results/pathology.csv` (the group of every
slide) and `results/*.json` are small and tracked.
