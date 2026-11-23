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
- **D+C gates:** `np.allclose` with `rtol=0` and `atol=1e-12`, an absolute tolerance alone. NumPy's default
  `rtol=1e-5` would let a difference of 5e-6 pass at a probability of 0.5.
- **Outlier rule:** unchanged, as pre-registered, although the first timed outer fold dropped 40 of its 380 training
  slides. Every run records its drops as they fall.

## Runs

- 2026-11-23: `curves.py spike`, the Gram matrix of the first 20 slides per class (210 pairs, 10 threads): epithelial
  3,030,629 bars in 0.58 s, stromal 0.12 s, immune 0.07 s, other 0.07 s. Extrapolated by pair count to the 594
  slides (176,715 pairs): **0.20 h in all** (epithelial 0.14 h), far below the plan's 6-hour limit. The estimate
  leans high: 20 rows over 10 threads leave the busiest thread 30 of the 210 pairs. The first 20 slides are typical:
  their mean bars per slide are within 1% (epithelial) to 25% (immune) of all 594 slides'. 3.0 s wall and peak RSS
  3.0 GB (`/usr/bin/time -l`); `results/spike.json` names curves.py md5 e014819e….
  - Checked ad hoc on the real data before the `gram` stage (a scratch script, not kept): the decile gate passes
    (`decile_counts` in 3.6 s); `rho`'s minimum equals decile 0 exactly, so `rho >= d_0` keeps every cell, as the
    cell-type study's slice 0 does; every slide has at least 668 cells of every class (no zero curve); and on four
    slides per class every Gram entry equals `sum_of_minima` within a relative 5.1e-15.
- 2026-11-23: `curves.py gram`, run by the controller in the background (`caffeinate -is /usr/bin/time -l`,
  `logs/gram.log`): 617.6 s wall (10.3 min, against the spike's 0.20 h), 4,645 s user CPU, peak RSS 3.25 GB, peak
  memory footprint 3.83 GB. The decile gate passed: the 10 × 594 × 4 counts equal `kept_by_group` (126,219,124 cells
  at decile 0; 3.7 s). Gram seconds per class (10 threads): epithelial 421.4 (89,633,804 bars), stromal 86.8
  (18,853,368), immune 47.6 (8,202,328), other 51.9 (9,529,624); building each class's curves took 0.4–3.1 s.
  `results/gram.json` names curves.py md5 e014819e…, the committed file.
  - Checked after the run (a scratch script, not kept): `results/gram.npz` (11 MB, gitignored) holds four float64
    594 × 594 matrices, finite and exactly symmetric, with positive diagonals (the smallest, other's, 5.6e8), and
    `slide_ids` in the spaces.npz order. Each matrix's smallest eigenvalue is positive (1.2e-12 to 6.5e-12 of its
    largest). Five random entries per class equal `sum_of_minima` within a relative 3.4e-15.
- 2026-11-23: `run_study.py development`, in the foreground (`caffeinate -is /usr/bin/time -l`,
  `logs/development.log`): 39.6 s wall (38.2 s in the json), 37.4 s user CPU, peak RSS 0.32 GB, peak memory footprint
  0.25 GB. No warning, and no logistic fit failed to converge.
  - **The D+C gate passed.** The unfiltered D+C arm, averaged over the 20 repeats, equals the improvement phase's
    `improve_dev.npz` `baseline:D+C-features@51` exactly (largest absolute difference 0; gate `rtol=0`,
    `atol=1e-12`).
  - **`k` chosen in the 100 outer folds:** 4 (25 folds), 6 (37), 10 (7), 12 (6), 16 (17), 20 (3), 24 (4), 32 (1).
    Never 8, 40 or 51.
  - **Drops,** among the training slides only:
    - per outer fold, 36–56 (median 44) of 380–381 training slides, 9.4–14.7%;
    - overall, 4,460 of the 38,080 training-slide places, 11.7%;
    - by group: healthy 13.4%, congestion 12.1%, steatosis 11.7%, necrosis 17.6%, inflammation/hepatitis 15.1%,
      fibrosis 4.5%, cirrhosis 13.0%, unclassified 11.2%;
    - by slide: of the 476 development slides, 382 were never dropped, 94 at least once, and 37 in all 80 outer
      folds in which they trained.
    - The kept training slides number 325–345 per fold. The smallest group among them, inflammation/hepatitis, keeps
      at least 6, enough for the inner 5-fold.
  - Checked after the run (a scratch script, not kept):
    - `results/development.npz` (gitignored) holds the four arms, each 20 × 476 × 8; every slide's probabilities sum
      to 1 in every repeat.
    - It also holds `k` (20 × 5, all in `K_GRID`), `dropped` (20 × 5 × 476, never a test slide), `folds` (equal to
      `make_folds(..., seed=1)`) and `slide_ids` (the development slides in order).
    - The json's `chosen_k` and `dropped_per_fold` match the npz.
    - `results/development.json` names run_study.py md5 319f1f10…, folds.py c6b1d654… and common.py db4b4915…, the
      committed files.
  - Checked before the run (scratch, not kept): the unfiltered D+C arm over the 100 folds equalled the reference
    bitwise, both under default threading and with one BLAS thread. One outer fold took 0.39 s.
  - No AUC was computed and no arm was compared with another: the readouts are Task 5's.
