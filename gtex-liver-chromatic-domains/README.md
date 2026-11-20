# Microanatomical domains from local chromatic persistent homology (planned)

**Status: planned.** The design and the implementation plan are fixed, and the
30 pilot slides are drawn. Nothing else has run.

Around each cell of a GTEx liver slide, does the arrangement of cell types
relative to each other define recurring microanatomical domains that density,
composition and pairwise contacts do not? The arrangement is measured by the
chromatic persistent homology of the cell's 150 µm neighbourhood. The pilot
asks this of 30 slides, five per condition, with 1,500 sampled centre cells
each.
- **Level 1, compartments:** from the density and composition of each centre's neighbourhood, with no topology.
- **Level 2, subdomains:** inside each compartment, from the domain and image Betti curves of each class's chromatic six-pack (class → all cells), measured against 19 random labellings of the same cells.
- **Guard, fixed in advance:** the subdomains must not be predictable from Delaunay contacts per class pair, density and composition.

`design.md` holds the question, Simon's rulings and the decision rules;
`plan.md` the implementation plan; `progress.md` the ledger of what ran.

## Run

Needs:
- the cohort in `~/data/gtex_liver_cells/gtex_liver_cells`, opened read only;
- the sibling studies' results: `density.npz` and `pathology.csv` in `../gtex-liver-pathology-slices/results`, and `holdout.csv` in `../gtex-liver-celltype-slices/results`;
- the study venv `.venv/` on Python 3.13, from `requirements.txt` (`results/venv_freeze.txt` lists every package the run used);
- a topocyte checkout at the commit below, named first on `PYTHONPATH`.

The scripts refuse a topocyte from anywhere else, or a checkout whose package
differs from the commit.

```bash
git -C /path/to/topocyte worktree add --detach ../topocyte-18e3aa5 18e3aa5
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

| scripts | topocyte commit | run |
|---|---|---|
| `run_pilot.py pilot` | `18e3aa5` | 2026-11-20 |

The run order so far, from this folder:

```bash
export PYTHONPATH=/path/to/topocyte-18e3aa5
PY=.venv/bin/python
$PY -m pytest -q test_common.py
$PY run_pilot.py pilot
```

## Outputs

Tracked: `results/pilot.csv` (the 30 slides with their condition, half and
slide index) and `results/venv_freeze.txt`. The `.venv/` stays out of git.
