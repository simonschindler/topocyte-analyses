# Ledger: class-density curves, their fPCA against D+C

## Request (Simon, 2026-11-23)

"lets design a experiment that performs functional pca on these curves. first computing the mean curve per cohort,
filtering cohort outliers, fpca on inner product spaces as with betti curves normalized and unnormalized,
classification." Then: "cohort means pathology group. i want to know if it beats D+C."

## Rulings

### Simon (2026-11-23)

`design.md` ("Rulings") has them in full. The design was committed in e07c262, before any result.

- **Cohort** means pathology group: the eight groups of the pathology study's `pathology.csv` (healthy, congestion,
  steatosis, inflammation/hepatitis, fibrosis, cirrhosis, necrosis, unclassified).
- **The question:** does the curves' fPCA beat `D+C`?
- **Outlier filtering** happens only on the training slides, inside each fold. The test slides are always scored,
  and `D+C` is trained on the same cleaned training slides.
- **Components:** best against best decides. The number of MFA components of the curves (4 to 51) is chosen by
  inner cross-validation. Four components are reported beside it.
- **Computation:** each class's exact Gram matrix over all 594 slides is computed once. Every fold's fPCA is then
  linear algebra on its submatrices.

### Controller (2026-11-23)

- **Environment:** no study venv and no download; the uv cache was empty.
  - The study runs in topocyte's main-checkout venv, `/Users/simon/rep/lbi/reindero/topocyte/.venv/bin/python`,
    as the cell-type study's readouts did. Its versions: Python 3.14.0, numpy 2.5.1, scipy 1.18.0,
    scikit-learn 1.9.0, h5py 3.16.0, pandas 3.0.5, matplotlib 3.11.1 and pytest 9.1.1.
  - `requirements.txt` records these versions and installs nothing.
  - Every run puts the pinned checkout first on `PYTHONPATH`, so that `common.check_topocyte()` finds topocyte
    imported from there and not from the venv's editable install of the main checkout. The checkout is
    `/Users/simon/rep/lbi/reindero/topocyte-18e3aa5`, a clean detached worktree of `18e3aa5`
    (`git -C /Users/simon/rep/lbi/reindero/topocyte worktree add --detach ../topocyte-18e3aa5 18e3aa5`).
  - The tests, from this folder:
    `PYTHONPATH=/Users/simon/rep/lbi/reindero/topocyte-18e3aa5 /Users/simon/rep/lbi/reindero/topocyte/.venv/bin/python -m pytest -q -W error test_study.py`.
- **topocyte check:** `check_topocyte` is copied from the chromatic-domains study, not imported from it, so that
  each study stays self-contained and pinned to its own topocyte.
- **Commits:** local, on topocyte-analyses `main`, one per task (Simon's practice for this repository). Simon pushes.

## Runs
