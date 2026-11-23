# Class-curve fPCA against D+C: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a verdict, fixed in advance, on whether the fPCA of each slide's class curves over the 10-NN density
threshold beats the cell-type study's `D+C` features at separating the pathology groups.

**Architecture:**
1. One script computes each class's exact Gram matrix over all 594 slides once, with topocyte.
2. Every fold's work is linear algebra on those matrices. That work is the outlier filter, the per-block fPCA via
   topocyte's `fpca_from_gram` with exact projection of the other slides, the MFA, and the logistic regression.
3. The development run, the single holdout run, a verifier and a report follow the sibling studies' pattern.

**Tech stack:** Python 3.13; topocyte at `18e3aa5` from a pinned checkout on `PYTHONPATH`; numpy, scipy,
scikit-learn, h5py, pandas, matplotlib; pytest.

**Spec:** `gtex-liver-class-curves-fpca/design.md` (committed `e07c262`). Read it first. This plan argues from it.

## Global Constraints

- **Location:** everything lives in `topocyte-analyses/gtex-liver-class-curves-fpca/`. Commit locally on
  `topocyte-analyses` main, one commit per task. Never push. Every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Read only:**
  - `~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad` (h5py mode "r");
  - `../gtex-liver-pathology-slices/results/density.npz`;
  - `../gtex-liver-celltype-slices/results/{spaces.npz, holdout.csv, improve_dev.npz, improve_holdout.npz, composition.npz}`.
- **Never write:** under `~/data`, or in another study's folder.
- **topocyte:** `TOPOCYTE_COMMIT = "18e3aa5"`, unchanged, imported from a clean checkout named first on
  `PYTHONPATH`. Refuse anything else; copy `check_topocyte` from `../gtex-liver-chromatic-domains/common.py`.
- **Exact bar-endpoint arithmetic only:** no curve is evaluated on a grid for any computation. Grids appear only in
  the report's plots of the descriptive mean curves.
- **Fixed values:**
  - classes `("epithelial", "stromal", "immune", "other")` (the cell-type study's `CELL_GROUPS`);
  - 9 fPCA components per block;
  - blocks = 4 classes × `("none", "cosine")`;
  - `K_GRID = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 51)`;
  - outlier factor `3 × 1.4826`;
  - development folds `make_folds(dev_labels, 20, 5, seed=1)`;
  - inner folds `StratifiedKFold(5, shuffle=True, random_state=1000 + 5 * repeat + fold)`;
  - bootstrap: 4000 draws, `stratified_bootstrap(labels, 4000, np.random.default_rng(12345))`;
  - logistic regression as the cell-type study's `cv_probabilities` (multinomial L2, `C=1`,
    `class_weight="balanced"`, lbfgs, `max_iter=5000`, features standardized on the training slides), all eight
    groups;
  - macro AUC over the seven conditions (unclassified unscored).
- **The cell-type study's helpers** are loaded from `../gtex-liver-celltype-slices/common.py` under the module name
  `celltype` (with `importlib.util.spec_from_file_location`), because both folders have a `common.py`. Reuse its
  `CELL_GROUPS`, `GROUP_ORDER`, `CONDITIONS`, `read_cell_groups`, `macro_auc`, `make_folds`, `stratified_bootstrap`
  and `cv_probabilities`. Do not copy them.
- **Slide order:** the cell-type study's `spaces.npz` `slide_ids`, which `holdout.csv` repeats. Labels are indices
  into `GROUP_ORDER`. Development is `holdout == False` (476 slides), holdout is `holdout == True` (118).
- **Environment:** a study `.venv` from `requirements.txt`, pinned to the chromatic study's versions:
  - numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, h5py 3.16.0, pandas 3.1.1, matplotlib 3.11.2;
  - anndata 0.13.4, ripser 0.6.15, psutil 7.2.2, numba 0.68.1, pytest 9.2.0;
  - installed from the uv cache with `uv pip install`, no other downloads.
- **Pinned topocyte checkout:** `git -C /Users/simon/rep/lbi/reindero/topocyte worktree add --detach ../topocyte-18e3aa5 18e3aa5`.
- **Long runs** (Gram matrices, development run) go in the background with `caffeinate -is /usr/bin/time -l`, logs
  in `logs/` (gitignored).

## Review Focus

1. **Label leakage:** permuting the test slides' labels must leave every test probability unchanged, in both arms,
   filter included. Test: `test_test_labels_never_change_test_predictions` (Task 4).
2. **Small groups:**
   - With `MAD == 0` the rule drops nobody in that group.
   - A group with fewer than 3 training slides is not filtered.
   - The filter never empties a group.
   - Test: `test_outlier_filter_keeps_small_and_flat_groups` (Task 3).
3. **`k` above the components available:** a training set whose MFA has fewer than 51 columns caps the grid at the
   columns available, never errors. Test: `test_choose_k_caps_the_grid_at_the_columns_available` (Task 4).
4. **The holdout scored twice:** the holdout stage refuses when `results/holdout.json` exists. Test:
   `test_holdout_stage_refuses_a_second_run` (Task 5).
5. **A zero curve under cosine** (a slide without a class's cells): raise a `ValueError` naming the slide and class.
   Test: `test_cosine_refuses_a_zero_curve_with_its_slide` (Task 3).

---

### Task 1: Scaffold, environment and inputs

**Files:**
- Create: `requirements.txt`, `common.py`, `test_study.py`, `progress.md` (the ledger: request, rulings, then runs).

**Interfaces:**
- Produces, in `common.py`:
  - constants `HERE`, `RESULTS`, `LOGS`, `SIB_DENSITY = HERE.parent / "gtex-liver-pathology-slices/results/density.npz"`,
    `CELLTYPE = HERE.parent / "gtex-liver-celltype-slices"`, `TOPOCYTE_COMMIT`, `CLASSES`, `NORMALIZATIONS = ("none", "cosine")`,
    `N_COMPONENTS = 9`, `K_GRID`, `MAD_FACTOR = 3 * 1.4826`, `DEV_FOLD_SEED = 1`, `BOOT_SEED = 12345`, `N_BOOT = 4000`;
  - `celltype` (the loaded cell-type `common` module);
  - `check_topocyte() -> None`;
  - `load_inputs() -> dict`, with keys:
    - `slide_ids: list[str]`, `labels: np.ndarray[int]` (594), `dev: np.ndarray[bool]`, `holdout: np.ndarray[bool]`;
    - `DC: np.ndarray (594, 51)` = `hstack(D_features, C_features)` from `spaces.npz`;
    - `bounds: np.ndarray (595,)` and `deciles: np.ndarray (10,)` from `density.npz`, with that file's `slide_ids`
      equal to `slide_ids` in order, else `RuntimeError`.

- [ ] **Step 1: Write the failing test**

```python
def test_load_inputs_matches_the_celltype_split():
    d = c.load_inputs()
    assert len(d["slide_ids"]) == 594 and d["DC"].shape == (594, 51)
    assert d["dev"].sum() == 476 and d["holdout"].sum() == 118 and not (d["dev"] & d["holdout"]).any()
    assert np.bincount(d["labels"][d["dev"]], minlength=8).tolist() == [53, 124, 162, 14, 10, 40, 46, 27]
    assert d["bounds"][0] == 0 and d["bounds"][-1] == 126_219_124
```

The counts follow `GROUP_ORDER`: healthy, congestion, steatosis, necrosis, inflammation/hepatitis, fibrosis,
cirrhosis, unclassified.

- [ ] **Step 2: Run it.** `cd gtex-liver-class-curves-fpca && PYTHONPATH=../../topocyte-18e3aa5 .venv/bin/python -m pytest -q -W error test_study.py` → FAIL (no `common`).
- [ ] **Step 3: Implement** `requirements.txt`, the venv, the topocyte worktree, and `common.py` as in the
  Interfaces. Then `progress.md`: the request quote, Simon's rulings (the design's) and "Runs" (empty).
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** (`requirements.txt common.py test_study.py progress.md`): "Add the class-curve fPCA study's inputs and environment".

### Task 2: Class curves and exact Gram matrices

**Files:**
- Create: `curves.py`.
- Modify: `test_study.py`, `progress.md`.

**Interfaces:**
- Consumes: `common.load_inputs`, `celltype.read_cell_groups`.
- Produces, in `curves.py`:
  - `sum_of_minima(a: np.ndarray, b: np.ndarray) -> float`: `Σ_i Σ_j min(a_i, b_j)`, from sorted `b` and its prefix
    sums, O(n log n). Used by the tests and the verifier, never by the Gram.
  - `class_curve(rho: np.ndarray) -> BettiCurve`: `BettiCurve(np.column_stack([np.zeros(len(rho)), rho]), dim=0)`.
  - `decile_counts(rho: np.ndarray, classes: np.ndarray, bounds, deciles) -> np.ndarray (10, 594, 4)`: cells with
    `rho >= d_j` per slide and class.
  - `main(argv)`, run as `python curves.py spike` or `python curves.py gram`:
    - **`spike`:** the Gram of the first 20 slides per class. It times them and extrapolates to 594 slides by pair
      count, then writes `results/spike.json`.
    - **`gram`:**
      1. Read `rho` (float64) and the cells' classes.
      2. Gate: `decile_counts` equals the cell-type study's `composition.npz` `kept_by_group` exactly, else
         `RuntimeError`.
      3. For each class, build the 594 `class_curve`s and `betti_inner_product_matrix(curves, dim=0, n_jobs=-1)`.
         Build them one class at a time, freeing the curves between classes.
      4. Write `results/gram.npz` (keys `gram_<class>`, float64 594 × 594, and `slide_ids`) and `results/gram.json`
         (per-class seconds, the decile gate, versions, `TOPOCYTE_COMMIT`, md5 of `curves.py`).

- [ ] **Step 1: Write the failing tests**

```python
def test_betti_inner_product_of_class_curves_is_the_sum_of_minima():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(1, 5000, 300), rng.uniform(1, 5000, 170)
    G = betti_inner_product_matrix([cv.class_curve(a), cv.class_curve(b)], dim=0)
    assert np.isclose(G[0, 1], cv.sum_of_minima(a, b), rtol=1e-12)
    assert np.isclose(G[0, 0], cv.sum_of_minima(a, a), rtol=1e-12)

def test_decile_counts_keep_a_cell_at_its_own_threshold():
    rho = np.array([1.0, 2.0, 2.0, 3.0]); classes = np.array([0, 1, 1, 0]); bounds = np.array([0, 4])
    deciles = np.array([1.0, 2.0, 3.0] + [9.0] * 7)
    counts = cv.decile_counts(rho, classes, bounds, deciles)
    assert counts[:3, 0, :2].tolist() == [[2, 2], [1, 2], [1, 0]]
```

- [ ] **Step 2: Run them.** Expected: FAIL (no `curves`).
- [ ] **Step 3: Implement `curves.py`** as in the Interfaces.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the spike** (foreground): `PYTHONPATH=../../topocyte-18e3aa5 .venv/bin/python curves.py spike`.
  Record the extrapolated hours in `progress.md`. If the estimate exceeds 6 hours, stop and ask Simon.
- [ ] **Step 6: Run the Gram** (background, `caffeinate -is /usr/bin/time -l … > logs/gram.log`). Expected: the
  decile gate passes; `results/gram.npz` has four symmetric matrices with positive diagonals. Record the times and
  peak RSS in `progress.md`.
- [ ] **Step 7: Commit** (`curves.py test_study.py progress.md results/gram.json results/spike.json`; `gram.npz` is
  gitignored as `*.npz`): "Compute each class's exact Gram matrix of the class curves".

### Task 3: Fold linear algebra: filter, fPCA, MFA

**Files:**
- Create: `folds.py`.
- Modify: `test_study.py`.

**Interfaces:**
- Consumes: `results/gram.npz` (or any Gram dict), `topocyte.ph.decomposition.fpca_from_gram`.
- Produces, in `folds.py`. Indices are positions in the 594-slide order, and `grams` is a dict
  `class -> (594, 594)` array:
  - `analysis_kernel(G: np.ndarray, normalization: str) -> np.ndarray`: `G`, or the cosine similarities
    `D^-1/2 G D^-1/2`. Raises `ValueError` naming the slide index when a diagonal entry is 0 under cosine.
  - `block_distances(K: np.ndarray, train: np.ndarray, labels: np.ndarray) -> np.ndarray (len(train),)`: each
    training slide's squared distance to its group's mean curve over `train`, from the analysis kernel `K`. The mean
    includes the slide.
  - `outlier_keep(grams, train, labels) -> np.ndarray[bool] (len(train),)`:
    - eight blocks; each block's distances divided by their median over `train`, then summed;
    - a group drops slides above `median + MAD_FACTOR * MAD` of its summed distances;
    - nobody is dropped when `MAD == 0` or the group has fewer than 3 training slides.
  - `class BlockFPCA`:
    - `fit(K_tt: np.ndarray) -> self`, wrapping `fpca_from_gram` on `K_tt` with `normalization="none"`, because `K`
      is already the analysis kernel; keeps `N_COMPONENTS` or fewer;
    - `transform(K_xt: np.ndarray) -> np.ndarray`: rows centred with the training kernel's means,
      `score_k = k_c · v_k / sqrt(rho_k)`.
  - `class MFA`:
    - `fit(blocks: list[np.ndarray]) -> self`: each block divided by the SD of its first column; concatenated,
      centred and decomposed by SVD; sign of each left singular vector fixed by its largest-magnitude entry, as
      `celltype.pca`; up to 51 components;
    - `transform(blocks) -> np.ndarray`.
  - `curve_space(grams, fit_idx, other_idx) -> tuple[np.ndarray, np.ndarray]`: MFA scores of `fit_idx` and of
    `other_idx`, with the eight `BlockFPCA`s and the `MFA` fitted on `fit_idx`.

- [ ] **Step 1: Write the failing tests**
  - `test_block_fpca_matches_topocyte_fpca_transform`: build 12 random `class_curve`s and fit topocyte's
    `FPCA(dim=0, n_components=9, normalization=n, eigen_solver="dense")` on the first 9. For both normalizations,
    `BlockFPCA` on `analysis_kernel(G[:9, :9])` reproduces the fitted scores (`atol=1e-8` relative to the largest),
    and `transform` of the other 3 equals `FPCA.transform` to the same tolerance.
  - `test_mfa_transform_of_the_fit_rows_reproduces_the_fit`: `MFA().fit(blocks).transform(blocks)` equals the
    fitted scores, and the first column equals `celltype.mfa(blocks, 51)[0][:, 0]`.
  - `test_outlier_filter_drops_a_far_slide_and_keeps_small_and_flat_groups`:
    - a group of 10 with one slide far away drops exactly that slide;
    - a group of 2 drops nobody;
    - a group with identical curves drops nobody.
  - `test_cosine_refuses_a_zero_curve_with_its_slide`: `analysis_kernel` on a Gram with a zero diagonal at index 3
    raises `ValueError` matching `"slide 3"`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement `folds.py`.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit**: "Add the class-curve study's fold algebra: outlier filter, exact fPCA projection, MFA".

### Task 4: Arms, inner cross-validation and the development run

**Files:**
- Create: `run_study.py` (stages `development` and, in Task 5, `holdout`).
- Modify: `test_study.py`, `progress.md`.

**Interfaces:**
- Consumes: `folds.outlier_keep`, `folds.curve_space`, `celltype.macro_auc`, `celltype.make_folds`.
  - The per-fold logistic fit mirrors the cell-type improvement phase's `fit_logistic` and its prediction
    (`../gtex-liver-celltype-slices/improve.py`, from line 176) exactly: the same standardization, the same
    `LogisticRegression` arguments, and columns mapped through `model.classes_`.
  - The `D+C` gate compares against that phase's saved probabilities, so any difference shows.
- Produces, in `run_study.py`:
  - `fit_predict(Z_train, y_train, Z_test) -> np.ndarray (n_test, 8)`: the cell-type logistic regression; columns
    in `GROUP_ORDER`.
  - `choose_k(grams, train, labels, seed) -> int`:
    - inner `StratifiedKFold(5, shuffle=True, random_state=seed)` on `train`;
    - per inner fold, `curve_space(grams, inner_train, inner_test)` and `fit_predict` for every `k` in `K_GRID`
      capped at the columns available;
    - the macro AUC of the pooled inner out-of-fold probabilities; the highest wins, ties to the smaller `k`.
  - `outer_fold(grams, DC, labels, train, test, seed) -> dict`, with keys:
    - `keep` (mask over `train`);
    - `k` (the chosen `k`);
    - `curves` (n_test × 8 at `k`), `curves4` (n_test × 8 at `k = 4`);
    - `dc` (n_test × 8, `D+C` on the cleaned training slides), `dc_unfiltered` (n_test × 8, `D+C` on all of `train`).
  - `run_development(inputs, grams) -> dict`:
    - every repeat and fold of `make_folds(labels[dev], 20, 5, seed=DEV_FOLD_SEED)`;
    - inner seed `1000 + 5 * repeat + fold`;
    - writes `results/development.npz`: per-arm out-of-fold probabilities (20 × 476 × 8), the chosen `k` and
      dropped masks per fold.

- [ ] **Step 1: Write the failing tests**
  - `test_test_labels_never_change_test_predictions`: synthetic Grams for 60 slides in 3 groups. Permuting
    `labels[test]` leaves every array `outer_fold` returns for the test slides unchanged.
  - `test_choose_k_caps_the_grid_at_the_columns_available`: with 3 blocks of 2 components (6 columns), `choose_k`
    returns a value in {4, 6}.
  - `test_choose_k_prefers_the_smaller_k_on_ties`: with a monkeypatched constant macro AUC, it returns 4.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** the functions and the `development` stage.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the development stage** (background, logged).
  - Gate inside the run: `dc_unfiltered` averaged over repeats equals the cell-type study's `improve_dev.npz`
    `baseline:D+C-features@51` (`np.allclose`, `atol=1e-12`), else `RuntimeError`.
  - Record the time, the dropped counts and the chosen `k`s in `progress.md`.
- [ ] **Step 6: Commit** (`run_study.py test_study.py progress.md results/development.json`): "Run the class-curve study's development comparison".

### Task 5: Readouts, verdict and the single holdout run

**Files:**
- Modify: `run_study.py`, `test_study.py`, `progress.md`.

**Interfaces:**
- Produces:
  - `paired_interval(p_a, p_b, labels, draws) -> (diff, lo, hi)`: the macro AUC difference of two
    repeat-averaged probability arrays, with the 2.5 and 97.5 percentiles over `draws`.
  - `verdict(dev_lo: float, dev_hi: float, holdout_diff: float) -> str`:
    - `"beats D+C"` if `dev_lo > 0` and `holdout_diff > 0`;
    - `"not confirmed"` if `dev_lo > 0` and `holdout_diff <= 0`;
    - `"level with D+C"` if `dev_lo <= 0 <= dev_hi`;
    - `"below D+C"` if `dev_hi < 0`.
  - The development readouts in `results/development.json`:
    - each arm's macro AUC, the per-repeat spread and the per-condition AUCs;
    - the curves-minus-`D+C` difference with its interval at the chosen `k` and at `k = 4`;
    - the gate (`D+C` unfiltered: 0.764).
  - The `holdout` stage: `python run_study.py holdout`.
    1. Refuse if `results/holdout.json` exists.
    2. Fit on all 476 development slides: the filter, `choose_k` with seed 999, the curve space, the classifiers.
    3. Predict the 118 holdout slides.
    4. Gate: the unfiltered `D+C` holdout probabilities equal the cell-type study's `improve_holdout.npz` entry
       whose `names` value is the `D+C` features (`np.allclose`, `atol=1e-12`); the macro AUC is then 0.774.
    5. Write `results/holdout.json` and `.npz`: AUCs, the paired difference with its interval
       (`stratified_bootstrap` over the holdout labels, seed 12345), the verdict, and the small-group note.

- [ ] **Step 1: Write the failing tests**
  - `test_verdict_follows_the_rule_fixed_in_advance`: all four outcomes, plus the boundary `dev_lo == 0` →
    `"level with D+C"`.
  - `test_holdout_stage_refuses_a_second_run`: with a `results/holdout.json` present (`tmp_path` and a patched
    `RESULTS`), `main(["holdout"])` raises `RuntimeError` matching `"scored once"`.
  - `test_paired_interval_is_zero_for_identical_arms`: the difference and both ends are 0.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** them.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run `development` readouts** (if Task 4 saved only probabilities), then **the holdout stage once**.
  Record both in `progress.md` with the verdict.
- [ ] **Step 6: Commit** (`run_study.py test_study.py progress.md results/development.json results/holdout.json`): "Score the class-curve study's holdout once: the verdict".

### Task 6: Verification, report and README

**Files:**
- Create: `verify_results.py`, `build_report.py`, `README.md`.
- Modify: `progress.md`, `../README.md` (the Studies table), `../EXPERIMENTS.md` (a section for this study).

**Interfaces:**
- `verify_results.py` writes `results/verify.json` and exits 1 on any failure. Checks, each by another route:
  1. 30 random Gram entries per class by `sum_of_minima`.
  2. `decile_counts` against `composition.npz`.
  3. On one outer fold, the `BlockFPCA` projection against topocyte's `FPCA(eigen_solver="dense")` fit on the `T'`
     curves.
  4. The MFA by eigendecomposition of the sum of block Gram matrices.
  5. Every saved macro AUC by `celltype.auc` from the saved probabilities.
  6. The development folds against `make_folds(..., seed=1)`.
  7. The holdout counted once.
  8. The script md5s named in the results.
- `build_report.py` writes `report.html` (dataviz palette, class colours as in the viewer), with every number in the
  prose computed and asserted. Sections:
  1. the question and rulings;
  2. each group's mean class curves over density (counts and divided by cells), on the development slides;
  3. how often each slide was dropped, with its group and notes;
  4. fractions of variance per block;
  5. the chosen `k`s;
  6. development and holdout AUCs per arm, with the paired differences;
  7. the verdict;
  8. methods and run.

- [ ] **Step 1:** Write `verify_results.py`; run it. Expected: every check `ok`.
- [ ] **Step 2:** Write `build_report.py`; run it; look at `report.html` in the browser pane.
- [ ] **Step 3:** Write the `README.md` (status line with the verdict, run recipe, outputs). Add the Studies row and
  an `EXPERIMENTS.md` section.
- [ ] **Step 4:** Run the full tests: `PYTHONPATH=../../topocyte-18e3aa5 .venv/bin/python -m pytest -q -W error`.
- [ ] **Step 5: Commit**: "Verify and report the class-curve fPCA study".
