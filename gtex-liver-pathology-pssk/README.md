# Liver pathology across density slices, with the persistence scale-space kernel

The [Betti-curve study](../gtex-liver-pathology-slices/README.md) repeated with
gudhi's `PersistenceScaleSpaceKernel` (Reininghaus et al., CVPR 2015) in place
of the Betti-curve inner product. The slides, pathology groups, density slice
levels and persistent homology are that study's; at each level, four kernel
PCA models (H0 and H1, raw and cosine), nine components each, are plotted as
fractions of variance explained and as every pair of the first four components
and a UMAP of all nine, coloured by pathology. The scores of all slices are
combined into one space by multiple factor analysis, the separation of
cirrhosis from healthy along the combined PC2 is checked against the
densest-decile share of cells by the same pre-registered rule, and a table sets
the key numbers beside the Betti study's.

An exact kernel entry needs one Gaussian per pair of points of two diagrams
(about 10^11 at slice 0), so the kernel goes through gudhi's own approximation,
random Fourier features (scikit-learn's `RBFSampler`, 1000 features per
dimension), computed once per slide instead of once per pair. Measured against
gudhi's exact kernel at slice 9, the H0 kernel comes out within about 1%, the
H1 kernel about 13% off; the H1 cosine model's PC3 and PC4 move with the draw
of the features, and the report marks them.

The study is descriptive: it fits no classifier and draws no verdict.
`progress.md` records every ruling and the validation; `report.html` holds the
figures.

## Run

Needs the sibling study's results in `../gtex-liver-pathology-slices/results`
(`pathology.csv`, `density.npz`, and `slice_<k>.npz` for the check on the bar
counts; `combined.npz` and `pc2_check.npz` for the comparison), the cohort in
`~/data/gtex_liver_cells/gtex_liver_cells` (opened read only), and a Python
environment with topocyte's dependencies, gudhi and umap-learn; the run used
the topocyte checkout's `.venv`. Set `PYTHONPATH` to a topocyte checkout at the
commit below; `run_pssk_slices.py` refuses a topocyte that comes from anywhere
else.

| scripts | topocyte commit | run |
|---|---|---|
| `run_pssk_slices.py slice 9`...`0`, `validate_kernel.py` (gudhi 3.13.0, scikit-learn 1.9.0); `umap_scores.py` (umap-learn 0.5.12), `combined_spaces.py`, `pc2_density_check.py` | `bd1d5c0` | 2026-11-18 |

```bash
export PYTHONPATH=/path/to/topocyte-bd1d5c0
PY=/path/to/topocyte/.venv/bin/python
for k in 9 8 7 6 5 4 3 2 1 0; do $PY run_pssk_slices.py slice $k; done
$PY validate_kernel.py
$PY umap_scores.py
$PY combined_spaces.py
$PY pc2_density_check.py
$PY verify_results.py
$PY build_report.py
```

Each slice runs in its own process and recomputes the persistent homology as
the sibling study did (its diagrams were not saved); it stops unless every
slide's numbers of H0 and H1 bars equal that study's. Slice 0, all cells, is
the largest. `umap_scores.py`, `combined_spaces.py` and `pc2_density_check.py`
are the sibling study's, changed only to read the groups and densities from
there. Each run's runtime is in its `results/*.json`.

## Outputs

Not in git: `results/*.npz` (each slice's file holds every slide's random
features, about 10 MB), `logs/` and `report.html`. `results/*.json` are small
and tracked.
