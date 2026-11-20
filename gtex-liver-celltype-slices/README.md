# Liver pathology across density slices, per cell group

The [Betti-curve study](../gtex-liver-pathology-slices/README.md) repeated with
the cells split by HistoPLUS cell type into four groups (epithelial, stromal,
immune, other).
- **Slices and topology.** The ten density slices are the sibling's, cut on the all-cell 10-nearest-neighbour density. Each group gets its own alpha persistent homology and Betti-curve fPCA (H0 and H1, raw and cosine, nine components) at each slice.
- **Spaces.** Combined by multiple factor analysis for every choice of point sets and models (465 spaces).
- **Readouts.** Two readouts measure how well each space separates the seven pathology groups:
  - cross-validated logistic regression (macro one-vs-rest AUC);
  - the purity of each slide's 15 nearest neighbours relative to chance.
- **Comparisons fixed in advance** (`design.md`, committed before any readout):
  - *Primary:* does the space of all cells plus the four groups separate better than all-cell topology, at four components?
  - *Composition guard:* does per-group topology add to density summaries and cell counts per group?
- **Improvement phase.** An exploratory phase tries modelling ideas on four fifths of the slides and scores the chosen ones once on a holdout fixed before any result (`results/holdout.csv`).

`design.md` holds the question, Simon's rulings, the decision rules and the
rulings Claude made while he was away; `plan.md` the implementation plan;
`progress.md` the ledger of what ran; `report.html` the figures;
`improvements.md` the improvement phase.

## Run

Needs:
- the sibling study's results in `../gtex-liver-pathology-slices/results` (`pathology.csv`, `density.npz`, `slice_<k>.npz`, `combined.npz`);
- the cohort in `~/data/gtex_liver_cells/gtex_liver_cells`, opened read only;
- the topocyte checkout's `.venv` with umap-learn.

`run_celltype_slices.py` refuses a topocyte other than a clean checkout at
the commit below.

| scripts | topocyte commit | run |
|---|---|---|
| `make_holdout.py`, `run_celltype_slices.py groups`, `gate 9`, `gate 8`, `slice 9`...`0`, `spaces.py` (umap-learn 0.5.12), `readouts.py` (scikit-learn 1.9.0), `verify_results.py`, `build_report.py`, `percell_fpca.py slice 9`...`0` and `collect` (topocyte's private `_betti_operator` and `_fit_matrix_free` at this commit), `improve.py`, `improve_holdout.py`, `build_improvements.py`, `pls_study.py` | `bd1d5c0` | 2026-11-19/20 |

```bash
export PYTHONPATH=/path/to/topocyte-bd1d5c0
PY=/path/to/topocyte/.venv/bin/python
$PY -m pytest -q test_celltype.py
$PY run_celltype_slices.py groups
$PY run_celltype_slices.py gate 9; $PY run_celltype_slices.py gate 8
for k in 9 8 7 6 5 4 3 2 1 0; do $PY run_celltype_slices.py slice $k; done
$PY spaces.py
$PY readouts.py
$PY verify_results.py
$PY build_report.py
for k in 9 8 7 6 5 4 3 2 1 0; do $PY percell_fpca.py slice $k; done
$PY percell_fpca.py collect
$PY -m pytest -q test_improve.py
$PY improve.py
$PY improve_holdout.py      # once only: it refuses to score the holdout twice
$PY build_improvements.py
$PY -m pytest -q test_pls.py
$PY pls_study.py            # development slides only; never the holdout
$PY build_report.py         # again, for the improvement, PLS and discussion sections
```

The per-group slices take about 45 minutes in all (slice 0, all 126 million
cells, about nine minutes), the per-cell fits about 75 minutes; the readouts
a few minutes.

## Outputs

Not in git: `results/*.npz`, `logs/` and `report.html`. `results/*.json`,
`results/holdout.csv` and `improvements.md` are small and tracked.
