# Microanatomical domains from local chromatic persistent homology (pilot)

**Status: the pilot has run; under the rule fixed in advance, topology does not add.**
Level 1 gives one composition stratum, the control passes, and level 2 gives no
subdomains, so the verdict is "does not add" ("no compartment with subdomains")
and the pairwise guard had nothing to test. `report.html` has the curves and the
reasons. The design's fallback does not hold either: if topology did not add,
the level-1 strata were to give the planned second study its regions, and there
is one stratum. The pilot gives that study no regions at either level; where
they come from instead is Simon's decision. The H&E atlas was planned for
compartments and subdomains; with one stratum and no subdomains there are no
groups to compare, and whether to build it is Simon's decision.

`verify_results.py` recomputes, for every centre or fit, the level-1 and
pairwise features from the saved class and pair counts, every centre's spreads,
every fit's fPCA and MFA and every level-2 row, and the results of
`domains.py`. On samples it recomputes the bars of 230 balls (200 sampled,
against chalc; one per slide, bit for bit; the two samples disjoint) with their
class counts, the pair counts of 50 of them (Qhull on 20, bit for bit on 30),
and the other centres' block scores and the control's level-2 rows on the 200
sampled centres. Most checks take another route than the code they check; the
bit-for-bit bars, the Gram matrices (`curves.inner_matrix`, as `features.py`
used it) and the k-means fits (the same scikit-learn calls and seeds) reproduce
rather than verify. It does not redo the pre-flight, the pilot draw (it checks
its counts), the features run's gate diagnostics or the runs' timings, sizes
and versions.

Around each cell of a GTEx liver slide, does the arrangement of cell types
relative to each other define recurring microanatomical domains that density,
composition and pairwise contacts do not? The arrangement is measured by the
chromatic persistent homology of the cell's 150 µm neighbourhood. The pilot
asks this of 30 slides, five per condition, with 1,500 sampled centre cells
each.
- **Level 1, compartments:** from the density and composition of each centre's neighbourhood, with no topology; reported as composition strata, never as modes.
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

| scripts | code | topocyte commit | run |
|---|---|---|---|
| `run_pilot.py pilot` | `602ace5` | `18e3aa5` | 2026-11-20 |
| `run_pilot.py preflight` | `0b031a3`, re-run at `381d0ce` | `18e3aa5` | 2026-11-21, 2026-11-23 |
| `run_pilot.py bars` | `e8db8a4` | `18e3aa5` | 2026-11-22 |
| `features.py` | blob `f7ae634` at `da7e578` | `18e3aa5` | 2026-11-22 |
| `domains.py` | `381d0ce` | `18e3aa5` | 2026-11-23 |
| `verify_results.py`, `build_report.py` | `f62ff28` and the task review's fixes after it | `18e3aa5` | 2026-11-23 |

The run order, from this folder. The long runs went on AC power inside
`caffeinate -is`, with `/usr/bin/time -l` and their logs in `logs/` (gitignored,
so a fresh checkout creates it); `build_report.py` reads the features run's time
and peak RSS from `logs/features.log`, whose first line names the md5 of
`features.py`. Run `verify_results.py` with nothing else heavy open: on the
16 GB Mac its peak RSS reached 10.46 GB and its memory footprint 13.75 GB,
climbing across the blocks (allocator cache, it seems).

```bash
export PYTHONPATH=/path/to/topocyte-18e3aa5
PY=.venv/bin/python
mkdir -p logs
$PY -m pytest -q -W error                       # the study's tests, about 2 min
$PY run_pilot.py pilot
$PY run_pilot.py preflight                      # about 12 min
$PY run_pilot.py bars --workers 8               # about 3.3 h, 9.33 GB of bars
{ echo "started $(date '+%F %T'), features.py md5 $(md5 -q features.py), curves.py md5 $(md5 -q curves.py)"
  /usr/bin/time -l $PY features.py; } > logs/features.log 2>&1   # about 68 min, peak RSS about 10 GB
$PY domains.py                                  # about 5 min
$PY verify_results.py                           # about 69 min, peak RSS about 10.5 GB
$PY build_report.py                             # about 90 s, peak RSS about 7 GB
```

## Outputs

Tracked: `results/pilot.csv` (the 30 slides with their condition, half and
slide index), `results/venv_freeze.txt`, `results/preflight.json`,
`results/bars/*.json` (each slide's timings, sizes and versions),
`results/domains.json` (the rule's curves, choices and verdict) and
`results/verify.json` (every check with its sample, tolerance and measured
value, the md5 of `verify_results.py`, and the sha256 of every gitignored file
it read).

Not in git: `results/bars/*.npz` (9.33 GB), `results/features.npz` (206 MB),
`results/domains.npz`, `logs/`, `report.html` and the `.venv/`. The scripts
regenerate them; the copy the report was built from is in this folder on
Simon's machine, and `build_report.py` checks `features.npz` and the bars files
against the sums in `results/verify.json`.
