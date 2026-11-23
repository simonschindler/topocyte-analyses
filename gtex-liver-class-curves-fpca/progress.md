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
  leans high: 20 rows over 10 threads leave the busiest thread 30 of the 210 pairs. The first 20 slides are larger
  than average, which leans it high too: their mean bars per slide exceed all 594 slides' by 0.4% (epithelial),
  8.4% (stromal), 5.9% (other) and 24.6% (immune). (Corrected after the holdout; the entry first called them
  typical.) 3.0 s wall and peak RSS 3.0 GB (`/usr/bin/time -l`); `results/spike.json` names curves.py md5 e014819e….
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
- 2026-11-23: the `readouts` and `holdout` stages, their tests and the verdict function were committed as a58c812
  before either stage ran, so both result files name committed code (run_study.py md5 034e0ad5…; folds.py and
  common.py unchanged since the development run). The holdout fit gets the labels with the holdout slides' set to
  −1, so that no holdout label can reach it.
- 2026-11-23: `run_study.py readouts`, in the foreground (`caffeinate -is /usr/bin/time -l`, `logs/readouts.log`):
  6.9 s wall (5.5 s in the json), peak RSS 0.31 GB. It added `readouts` to `results/development.json` and kept the
  development run's metadata unchanged.
  - **The D+C gate passed.** The unfiltered D+C, averaged over the repeats, has development macro AUC 0.7645, the
    improvement phase's 0.764.
  - **Macro AUC over the seven conditions, from the probabilities averaged over the 20 repeats.** The per-repeat
    macro AUCs follow in parentheses (mean ± SD over the repeats).
    - Curves at the chosen `k`: 0.750 (0.722 ± 0.013).
    - Curves at `k = 4`: 0.740 (0.729 ± 0.013).
    - D+C on the cleaned training slides: 0.756 (0.749 ± 0.011).
    - D+C on all the training slides: 0.764 (0.759 ± 0.009).
  - **Paired differences against D+C on the cleaned training slides,** with 95% intervals over 4000 bootstrap draws
    stratified by group (`default_rng(12345)`):
    - curves at the chosen `k`: −0.0064 [−0.0241, +0.0116]. The interval contains 0.
    - curves at `k = 4`: −0.0164 [−0.0398, +0.0078] (reported beside, with no verdict).
  - Each arm's per-condition AUCs and per-repeat macro AUCs are in the json.
  - Checked after the run (a scratch script, not kept): scikit-learn's `roc_auc_score` reproduces the four macro
    AUCs within 1e-12 from `development.npz`, with the labels read directly from `spaces.npz` and `holdout.csv`. A
    rewrite of the stratified bootstrap gives, on its first 400 draws, intervals within 0.001 of the json's.
- 2026-11-23 20:07: **`run_study.py holdout`, run once,** in the foreground (`logs/holdout.log`): 5.5 s wall (4.0 s in
  the json), peak RSS 0.33 GB. No warning, and no logistic fit failed to converge. Nothing failed, so there was no
  rerun.
  - **The fit on the 476 development slides:**
    - the filter dropped 55 (11.6%): healthy 7 of 53, congestion 15 of 124, steatosis 19 of 162, necrosis 3 of 14,
      inflammation/hepatitis 1 of 10, fibrosis 1 of 40, cirrhosis 6 of 46, unclassified 3 of 27;
    - `choose_k` (inner seed 999) chose `k = 6`.
  - **The D+C gate passed.**
    - The unfiltered D+C holdout probabilities equal the cell-type study's `improve_holdout.npz` "D+C features"
      (`e2__proba`) exactly (largest absolute difference 0; gate `rtol=0`, `atol=1e-12`).
    - That file lists the same holdout slides with the same groups, in the same order.
    - Their macro AUC is 0.7739, the improvement phase's 0.774.
  - **Holdout macro AUCs:**
    - curves at the chosen `k`: 0.769;
    - curves at `k = 4`: 0.765;
    - D+C on the cleaned development slides: 0.779;
    - D+C on all of them: 0.774.
  - **Paired differences against D+C on the cleaned development slides,** with 95% intervals over 4000 bootstrap
    draws of the holdout slides, stratified by group (`default_rng(12345)`):
    - curves at the chosen `k`: −0.0101 [−0.0421, +0.0169];
    - curves at `k = 4`: −0.0144 [−0.0537, +0.0183].
    - The holdout's groups of 2 (inflammation/hepatitis) and 4 (necrosis) slides make these intervals too narrow;
      `holdout.json` says so.
  - **Verdict, by the rule fixed in advance: level with D+C.**
    - The development interval of curves minus D+C, [−0.0241, +0.0116], contains 0.
    - In that case the holdout difference does not enter the rule. It was negative too (−0.010).
  - Beside the verdict: the filter cost D+C 0.008 of development macro AUC (0.756 against 0.764) and gained it 0.005
    on the holdout (0.779 against 0.774).
  - Checked after the run (a scratch script, not kept):
    - `results/holdout.npz` (gitignored) holds the four arms (118 × 8; every row sums to 1), `k`, `dropped` (476),
      and the holdout and development slide IDs and holdout labels, in the order of `holdout.csv`.
    - Its unfiltered D+C equals `e2__proba` bitwise.
    - scikit-learn's `roc_auc_score` reproduces the four holdout macro AUCs within 1e-12.
- 2026-11-23: `verify_results.py`, in the background (`caffeinate -is /usr/bin/time -l`, `logs/verify.log`): **17 of
  17 checks pass**, 451.9 s wall, peak RSS 5.19 GB. It runs no stage and fits no classifier: every AUC it computes
  comes from the saved probabilities, so the holdout is not scored again.
  - `sum_of_minima`, the Gram check's oracle, equals `np.minimum.outer(a, b).sum()` exactly on ties and empty arrays
    (within 2.2e-16 on random values). 30 Gram entries per class equal `sum_of_minima` within a relative 1.4e-14.
  - The decile counts, by one flat bincount per decile and by `decile_counts`, equal `composition.npz`.
  - Repeat 0, fold 0 (340 kept, 96 test slides): topocyte's `FPCA(eigen_solver="dense")`, fitted on the kept
    slides' curves, gives every block's fitted scores exactly (it computes its own Gram matrix of those curves), and
    its `transform` gives the test slides' projections within 3.3e-9 of the block's largest score. `curve_space`'s
    MFA equals an eigendecomposition of the summed block Gram matrices within 4.3e-11.
  - Every saved AUC equals its recount by `celltype.auc` from the saved probabilities (121 development and 33
    holdout values, difference 0), and `celltype.auc` equals a count of pairs. The intervals on the redrawn bootstrap
    and the verdict are reproduced exactly, and the D+C gates are exact.
  - The holdout was scored once: `logs/holdout.log` (20:07:17), `holdout.npz` and `holdout.json` (20:07:23) were
    created in that order within one run, each written once, before 7b1fa2e (20:08:49) committed `holdout.json`,
    unchanged since. `development.json` was last written at 20:06:42.
  - Every md5 the results name equals the script at its commit: `run_study.py` 319f1f10… at 727667e for the
    development run, 034e0ad5… at a58c812 for the readouts and the holdout, the current file. The cell-type study's
    `common.py` (54b1dff9…) was last committed in 0207c6d, before this study's design, and is unchanged.
    `verify.json` records the sha256 of the 11 data files read, the cohort's .h5ad included.
  - An earlier draft of the script also passed 17 of 17; it changed only in its check names and docstrings, and
    this run's `verify.json` replaced that draft's.
- 2026-11-23: `build_report.py` wrote `report.html` (0.2 MB, about 6 s); every number in its prose is formatted from
  the results and asserted. It computes two descriptive results that no stage saved and no verdict uses:
  - each group's mean class curves on the development slides, counted exactly at thresholds 25 cells per mm² apart
    up to 15,000 (the plots' grid), with the counts at the deciles asserted equal to `composition.npz`;
  - each block's fractions of variance, from `BlockFPCA` on all 476 development slides: the first component holds
    86.1–95.5% in the raw blocks and 79.9–82.6% in the cosine blocks.
  - Looked at in the browser pane: the summary, eight sections, three figures and the tables render.
- 2026-11-23: `README.md` written: status, run recipe, the holdout's single scoring and the stages' missing guard
  against a rerun, and the outputs. `requirements.txt` now also records pillow and topocyte's own dependencies. The
  repository's `README.md` and `EXPERIMENTS.md` (section 7) list the study. A new test runs the real first development
  fold with its test slides' labels set to −1 and gets the same filter, k and arms; 41 tests pass with `-W error`.
- 2026-11-23, after the review: `README.md` no longer says that a fresh copy regenerates the outputs. This folder
  holds the study's one run: a clone carries `results/holdout.json`, so its holdout stage refuses; the verifier's
  holdout and provenance checks fit this run only, and the report needs them to pass. So `results/*.npz` (their
  sha256 in `verify.json`), `report.html` and `logs/` here are the kept copy, and a rerun is a new run in a new
  folder with its own provenance rows. The recipe now runs the tests after `curves.py gram`, and the real-fold test
  skips without `results/gram.npz` (checked with `common.RESULTS` patched to an empty folder: 1 skipped); 41 tests
  pass. `verify_results.py` and `build_report.py` are unchanged.
