# Density slices of the GTEx liver cohort

Is the function-Delaunay bifiltration (Alonso et al., SODA 2024), with γ the
10-nearest-neighbour cell density, worth adopting in topocyte? Its horizontal
slices are the alpha filtrations of the cells with γ above a threshold, so ten
slices at the pooled cohort deciles were computed with topocyte's existing alpha
pipeline for the 123 GTEx liver slides (66 healthy, 57 cirrhosis), classified
one by one and combined, each against a density-only control. Later sections ask
whether persistent homology adds anything beyond density (section 7), test the
H1 death/birth ratio, built to escape density, against a rule written before the
run (section 8), and refit every concluding arm with standardized features
(post-hoc, sections 5-8).

Verdict (report section 9): no slice, and no combination of slices, beats both
slice 0 and its density-only control by more than the noise. Persistent homology
is not a linear proxy for density but adds nothing usable beyond it, and the
death/birth ratio adds nothing beyond density either. `progress.md` records
every ruling and result; `report.html` is the full write-up.

## Run

Needs the GTEx liver cohort zarr store, read only (its path is `STORE` in
`run_slices.py`), and a Python environment with topocyte's dependencies; the
runs used the topocyte checkout's `.venv` with scikit-learn 1.9.0. Set
`PYTHONPATH` to a topocyte checkout at the commit below; `run_slices.py`, which
the later scripts import, refuses a topocyte that comes from anywhere else.

| scripts | topocyte commit | run |
|---|---|---|
| `run_slices.py density`, `run_slices.py slice 0`...`9`, `combined_classify.py` | `7db4d26` | 2026-11-16 |
| `density_adds.py`, `ratio_slices.py`, `scaled_*.py` | `bd1d5c0` | 2026-11-17 |

At `bd1d5c0`, `ratio_slices.py` recomputed the H1 diagrams of slices 0 and 6-9
and reproduced the saved fPCA scores bit for bit, and `scaled_ph_only.py`
reproduced every per-seed accuracy of the 2026-11-16 classifications. Every
`results/*.json` that records a topocyte version gives `0.1.0.post651.dev0+7db4d26`,
because the version comes from the interpreter's installed metadata, not from the
checkout.
Both commits are on topocyte's local `main`; on 2026-11-18 neither had been
pushed (`origin/main` was `e10778b`).

```bash
export PYTHONPATH=/path/to/topocyte-bd1d5c0
PY=/path/to/topocyte/.venv/bin/python
$PY run_slices.py density
for k in 0 1 2 3 4 5 6 7 8 9; do $PY run_slices.py slice $k; done
$PY combined_classify.py
$PY density_adds.py
$PY ratio_slices.py
$PY scaled_nested.py
$PY scaled_section7.py
$PY scaled_ratio_alone.py
$PY scaled_single_block.py
$PY scaled_ph_only.py
$PY verify_results.py
$PY build_report.py
```

`verify_results.py` exits 0 only if every check passes; `build_report.py`
writes `report.html`. Each script's runtime is in its `results/*.json`, and the
report states the total.

## Outputs

Not in git: `results/*.npz` (345 MB; `results/density.npz` alone, the per-cell
density, is 324 MB), `logs/`, `figures/` (the per-slice scatter plots),
`superseded/` (the first, 2026-11-16 run of section 7, which the verifier
compares against when present) and `report.html`. The copy the report was built
from sits in this folder, ignored by git; at publication it goes to Zenodo.
