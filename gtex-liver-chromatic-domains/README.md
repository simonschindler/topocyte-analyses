# Microanatomical domains from local chromatic persistent homology (pilot)

**Status: the pilot has run; under the rule fixed in advance, topology does not add.**
Level 1 gives one composition stratum, the control passes, and level 2 gives no
subdomains, so the verdict is "does not add" ("no compartment with subdomains")
and the pairwise guard had nothing to test. `report.html` has the curves and the
reasons, and `verify_results.py` checks every saved result by an independent
route. The H&E atlas was planned for compartments and subdomains; with one
stratum and no subdomains there are no groups to compare, and whether to build
it is Simon's decision.

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
| `verify_results.py`, `build_report.py` | the commit that adds them | `18e3aa5` | 2026-11-23 |

The run order, from this folder. The long runs went on AC power inside
`caffeinate -is`, with `/usr/bin/time -l` and their logs in `logs/`;
`build_report.py` reads the features run's time and peak RSS from
`logs/features.log`, whose first line names the md5 of `features.py`.

```bash
export PYTHONPATH=/path/to/topocyte-18e3aa5
PY=.venv/bin/python
$PY -m pytest -q -W error                       # the study's tests, about 2 min
$PY run_pilot.py pilot
$PY run_pilot.py preflight                      # about 12 min
$PY run_pilot.py bars --workers 8               # about 3.3 h, 9.33 GB of bars
{ echo "started $(date '+%F %T'), features.py md5 $(md5 -q features.py), curves.py md5 $(md5 -q curves.py)"
  /usr/bin/time -l $PY features.py; } > logs/features.log 2>&1   # about 68 min, peak RSS about 10 GB
$PY domains.py                                  # about 5 min
$PY verify_results.py                           # about 31 min, peak RSS about 9 GB
$PY build_report.py                             # about 90 s, peak RSS about 7 GB
```

## Outputs

Tracked: `results/pilot.csv` (the 30 slides with their condition, half and
slide index), `results/venv_freeze.txt`, `results/preflight.json`,
`results/bars/*.json` (each slide's timings, sizes and versions),
`results/domains.json` (the rule's curves, choices and verdict) and
`results/verify.json` (every check with its sample, tolerance and measured
value).

Not in git: `results/bars/*.npz` (9.33 GB), `results/features.npz` (206 MB),
`results/domains.npz`, `logs/`, `report.html` and the `.venv/`. The scripts
regenerate them; the copy the report was built from is in this folder on
Simon's machine.
