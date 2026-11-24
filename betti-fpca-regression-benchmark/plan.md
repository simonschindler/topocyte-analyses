# Betti fPCA regression benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, by the rule fixed in `design.md`, whether Betti-curve fPCA regression (arm F) beats the
persistence-diagram kernel of Aggarwal, Nichols and Turner (arm K) on the paper's three tasks, with an exploratory
Nadaraya–Watson arm on Betti L² distances (arm B), and report it in this folder.

**Architecture:** Stages, each a script that writes `results/<stage>.json` (tracked) and, where large,
`results/<stage>.npz` (ignored):
- `fetch_upstream.py`: the pinned clone;
- `inputs.py`: loaders, gate G5;
- `matrices.py`: every pairwise matrix computed once (W2, Betti L², Betti Gram), gates G2 and G4;
- `run_study.py arms`: K, then gate G1, then gate G3, then F and B;
- `run_study.py readouts`: paired intervals and verdicts;
- `verify_results.py`: other routes;
- `build_report.py`: the report.

Library modules: `common.py`, `safe_pickle.py`, `wasserstein.py`, `betti.py`, `estimators.py`, `readouts.py`. Every
module and test file lives in `betti-fpca-regression-benchmark/`.

**Tech Stack:** Python 3.14 in topocyte's main venv: numpy 2.5.1, scipy 1.18.0, pandas 3.0.5, ripser 0.6.15,
matplotlib 3.11.1, pytest. topocyte is the pinned checkout `/Users/simon/rep/lbi/reindero/topocyte-18e3aa5` (clean,
at `18e3aa5`), first on `PYTHONPATH`.

**Spec:** `design.md` in this folder, committed in 7075684 before any result. Executors read both.

## Global Constraints

- **Environment.** Every command runs from `/Users/simon/rep/lbi/reindero/topocyte-analyses/betti-fpca-regression-benchmark`
  after
  ```bash
  export PYTHONPATH=/Users/simon/rep/lbi/reindero/topocyte-18e3aa5
  PY=/Users/simon/rep/lbi/reindero/topocyte/.venv/bin/python
  ```
  Every script calls `common.check_topocyte()` first.
- **Tests:** `$PY -m pytest -q -W error test_study.py`. `-W error` survives topocyte's Rips import (checked: persim is
  not loaded).
- **No installs.** No new dependency. POT is absent, so W2 comes from scipy.
- **Upstream is data only.** `upstream/` at `fe7d652eff478ffe6081c7e6765458651ad3cd74` is never imported, never
  executed and never on `sys.path`. Its pickles are read only through `safe_pickle.load_pickle`.
- **No grids.** Exact bar-endpoint arithmetic only; no curve is evaluated on a grid.
- **No test target in any choice:** the bandwidth, (K0, K1) and the shift use training data only.
- **Order.** F and B are not computed until K passes G1, inside `run_study.py arms`. Any deviation from `design.md`
  found while building goes to Simon before `run_study.py arms` runs, and into `progress.md`.
- **Numbers fixed by the design:**

  | name | value |
  |---|---|
  | `K_MAX` | 20 |
  | `LEVERAGE_CAP` | 1 − 1e-10 |
  | bandwidths | 50, in `[0.1, 2] ×` the median positive training distance |
  | `N_BOOT` | 10,000 |
  | bootstrap generator | `np.random.default_rng(0)`, a fresh one for every interval |
  | `FLUID_SHIFT` | 26.0 |
  | fluid splits | 0–99; the paper's are 0–4 |
  | G1 tolerance | 1e-9 absolute |
  | G2 tolerance | 1e-12 relative |
  | G3 tolerance | 1e-9 relative to the largest score |
  | G4 tolerance | 1e-12 relative |
- **Commits.** Stage only this study's files, plus the shared files a task names, with `git add <paths>`, never
  `-A`: other sessions commit to this repo's `main` at the same time. Messages follow the repo's style: one
  imperative sentence, a short body, and a last line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Commit with `git commit -F <file>`, and never push.
- **Long stages** run in the background as `caffeinate -is /usr/bin/time -l $PY <script> [stage] > logs/<stage>.log 2>&1`.
  Each one appends a dated line to `progress.md`: the stage, its gates, its wall time and its peak RSS from the
  log.

## Review Focus

1. **Fluid diagrams with only essential bars** (6–14 of 400 at each time) have empty Betti curves: zero Gram rows,
   L2 = 0 between two of them, and a zero curve to project. Every arm must still give finite predictions. Test:
   Task 5 `test_arms_give_finite_predictions_with_empty_curves`.
2. **A training set whose curves in one dimension are all empty** gives a rank-0 fPCA. Then only K_d = 0 is
   eligible, with no division by √0 and no warning. Tests: Task 4
   `test_fpca_scores_of_an_all_empty_dimension_has_no_columns`, Task 5
   `test_loo_select_uses_only_dimensions_with_components`.
3. **Identical diagrams in a training set** (distance 0) are left out of the bandwidth median, and their weight is
   the kernel's maximum. Test: Task 5 `test_nw_median_ignores_zero_distances`.
4. **Large diagrams in W2.** The augmented cost matrix holds (n + m)² floats, so G2 caps its pairs at n + m ≤ 4,000
   finite bars. Test: Task 6 `test_g2_sample_caps_size_and_has_ten_large_checks`.
5. **Bars below 0.** topocyte's inner product drops the part below 0, so the fluid shift must make every value
   ≥ 0 without changing any inner product. Tests: Task 4 `test_shift_makes_the_gram_full_line` and
   `test_unshifted_negative_bars_lose_their_part_below_zero`.

---

### Task 1: Study folder, guard, whitelist loader, pinned upstream

**Files:**
- Create: `common.py`, `safe_pickle.py`, `fetch_upstream.py`, `test_study.py`, `progress.md`, `requirements.txt`
- Modify: `../.gitignore` (add the line `upstream/`)

**Interfaces:**
- Produces, in `common.py`:
  - paths `HERE`, `RESULTS`, `LOGS`, `UPSTREAM` (`HERE / "upstream"`);
  - `UPSTREAM_URL` = `https://github.com/saksham-aggarwal-2002/kernel-regression-persistence-diagrams.git`,
    `UPSTREAM_SHA`, `TOPOCYTE_COMMIT = "18e3aa5"`;
  - `INTERP_SIZES = tuple(range(10, 101, 10))`, `INTERP_RUNS = 10`, `INTERP_N_TEST = 50`;
  - `FLUID_TIMES = (1.0, 2.5, 5.0, 7.5, 9.0)`, `FLUID_SIZES = (25, 50, 100, 200, 300)`, `FLUID_N_TEST = 100`,
    `FLUID_SPLITS = 100`, `PAPER_SPLITS = 5`, `FLUID_SHIFT = 26.0`;
  - `K_MAX`, `LEVERAGE_CAP`, `N_BANDWIDTHS = 50`, `N_BOOT`, `BOOT_SEED = 0`, `G1_TOL`, `G2_TOL`, `G3_TOL`,
    `G4_TOL`;
  - `check_topocyte() -> None`, copied from `../gtex-liver-class-curves-fpca/common.py` with a comment saying so;
  - `r2(y: np.ndarray, p: np.ndarray) -> float`, which is `1 − SSE/SST` with SST about `y.mean()`;
  - `provenance(*scripts: Path) -> dict`, giving the md5 of each script and the versions of numpy, scipy, pandas
    and ripser, `TOPOCYTE_COMMIT` and `UPSTREAM_SHA`. Every stage's json embeds it.
- Produces, in `safe_pickle.py`:
  - `ALLOWED: dict[tuple[str, str], object]`, the 11 globals of `design.md` (Safety) mapped to the real objects;
  - `load_pickle(path: Path) -> object`, which raises `pickle.UnpicklingError` naming any other global.
- Produces, in `fetch_upstream.py`:
  - `ensure_upstream(source: str = UPSTREAM_URL) -> str`. If `UPSTREAM` is absent, it clones `source` and checks out
    `UPSTREAM_SHA` detached. If `UPSTREAM` is present, it requires `HEAD == UPSTREAM_SHA` and a clean tree, or
    raises `RuntimeError`. It returns the SHA.
  - The CLI `fetch_upstream.py [--source PATH_OR_URL]` writes `results/upstream.json` with the SHA, the source and
    the date.

- [ ] **Step 1: Write the failing tests** in `test_study.py`. At module level: `_Touch` (a class whose
  `__reduce__` returns `(pathlib.Path.touch, (marker,))` with the marker path given to its constructor) and
  `needs_upstream = pytest.mark.skipif(not (c.UPSTREAM / ".git").exists(), reason="upstream not fetched")`.

```python
def test_safe_pickle_refuses_a_global_outside_the_whitelist(tmp_path):
    marker = tmp_path / "ran"
    path = tmp_path / "evil.pkl"; path.write_bytes(pickle.dumps(_Touch(marker)))
    with pytest.raises(pickle.UnpicklingError, match="pathlib"):
        safe_pickle.load_pickle(path)
    assert not marker.exists()

def test_safe_pickle_reads_numpy_arrays(tmp_path):
    a = np.arange(6.0).reshape(3, 2); path = tmp_path / "a.pkl"; path.write_bytes(pickle.dumps(a))
    assert np.array_equal(safe_pickle.load_pickle(path), a)

def test_check_topocyte_needs_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        c.check_topocyte()

def test_r2_is_one_minus_sse_over_sst():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])
    assert c.r2(y, p) == pytest.approx(1 - 1.25 / np.sum((y - y.mean()) ** 2))

def test_ensure_upstream_refuses_a_clone_at_another_commit(tmp_path, monkeypatch):
    # a fresh `git init` repository with one commit stands in for upstream
    ...
    monkeypatch.setattr(fetch_upstream, "UPSTREAM", repo)
    with pytest.raises(RuntimeError, match="fe7d652"):
        fetch_upstream.ensure_upstream()
```

- [ ] **Step 2: Run** `$PY -m pytest -q -W error test_study.py`. Expected: errors on the missing modules.
- [ ] **Step 3: Implement** `common.py`, `safe_pickle.py` and `fetch_upstream.py` as specified.
- [ ] **Step 4: Run the tests.** Expected: 5 passed.
- [ ] **Step 5: Ignore `upstream/`.** Check `git -C .. diff --quiet -- .gitignore` first; if another session has
  uncommitted changes there, stop and ask. Then add the line `upstream/` under the existing patterns, with a
  comment that a study's pinned third-party clone is data.
- [ ] **Step 6: Fetch.** Run `$PY fetch_upstream.py --source /private/tmp/claude-502/-Users-simon-rep-lbi-reindero-topocyte--claude-worktrees-fpca-betti-curves-benchmark-9a48f4/4b9274ef-48aa-4f9f-977f-e27f8a354f3a/scratchpad/upstream/krpd`.
  If that clone is gone, run it with no `--source`, which uses the URL. Expected: `git -C upstream rev-parse HEAD`
  prints `fe7d652eff478ffe6081c7e6765458651ad3cd74`, and `results/upstream.json` exists.
- [ ] **Step 7: Write `progress.md` and `requirements.txt`.**
  - `progress.md` takes the sibling ledger's shape: Request (Simon's words, from `design.md`), Rulings (Simon's,
    from `design.md`; the design committed in 7075684 before any result), and a Results section with this stage's
    line.
  - `requirements.txt` is a record, as the sibling's is, not an install list. It gives the venv's versions from
    `importlib.metadata`, and topocyte's pinned checkout.
- [ ] **Step 8: Commit** these paths: `../.gitignore` and this folder's `common.py`, `safe_pickle.py`,
  `fetch_upstream.py`, `test_study.py`, `progress.md`, `requirements.txt`, `design.md` (unchanged),
  `plan.md` and `results/upstream.json`. Message: "Add the Betti FPCR benchmark's guard, whitelist loader and
  pinned upstream".

### Task 2: Inputs and gate G5

**Files:** Create `inputs.py`; Modify `test_study.py`, `progress.md`.

**Interfaces:**
- Consumes: `safe_pickle.load_pickle`, `common`.
- Produces, in `inputs.py`:
  - `Diagram = tuple[np.ndarray, np.ndarray]`: the (H0, H1) arrays, each `(n, 2)` float, with death `inf` for an
    essential bar.
  - `interpolation_datasets() -> list[dict]`: the 100 datasets of upstream's `interpolation_dataset.pkl`, in its
    order (training size ascending, runs 0–9). Keys: `n_train`, `run`, `clouds` (the training clouds, then the 50
    test clouds) and `y` (length `n_train + 50`).
  - `regenerate_interpolation() -> list[dict]`: the same, made by upstream's recipe
    (`interpolation_dataset_creation.py`, read, not run): `np.random.seed(0)`; for each size and run,
    `T = np.random.uniform(0, 1, n_train + 50)`; then for each t,
    `noise = np.random.uniform(-0.2, 0.2, 40)` and θ = `np.linspace(0, 2π, 20)`.
  - `curvature_data() -> dict`: `clouds` (100 arrays), `y` (100,), `train` and `test`, read from upstream's index
    pickles and asserted to be `arange(50)` and `arange(50, 100)`.
  - `rips_diagram(X: np.ndarray) -> Diagram`: `tc.ph.compute_persistent_homology(X, filtration="rips", max_dim=1)`.
  - `same_pairs(a: np.ndarray, b: np.ndarray) -> bool`: equal as rows sorted by birth, then death
    (`np.lexsort((d, b))`).
  - `fluid_sim_ids() -> list[int]`, by upstream's formula in `fluid/distances.py`.
  - `fluid_ln_re(ids: np.ndarray) -> np.ndarray`, which is `(4/3)(1 + i/1000) ln 2`.
  - `fluid_split(s: int) -> tuple[np.ndarray, np.ndarray]`, giving `(test, pool)` from
    `np.random.default_rng(s).permutation(400)`.
  - `fluid_diagrams(t: float) -> list[Diagram]`, in `fluid_sim_ids()` order, read from the DataFrames' Birth,
    Death and HomologyG columns.
  - `fluid_cached_w2(t: float, h: int) -> np.ndarray`: 400 × 400, symmetric, with a zero diagonal, from
    `pairs_h{h}.pkl` and its shards. It raises `KeyError` on a missing pair.
  - `main()`, the stage, writes `results/inputs.json`:
    - G5: the interpolation datasets equal `regenerate_interpolation()`; all 10,600 clouds give the same pairs by
      `same_pairs` as `ripser.ripser(X)["dgms"]`; all 2,000 fluid diagrams are present;
    - the counts: bars per fluid diagram (min, median, max), and empty finite H0 and H1 per time;
    - the lowest fluid birth, with `ceil(-lowest) == FLUID_SHIFT` asserted;
    - the essential pattern, asserted `{(1, 2)}`;
    - `provenance(...)` and the runtime.
    The stage exits with status 1 if a check fails.

- [ ] **Step 1: Write the failing tests:**

```python
def test_fluid_sim_ids_follow_upstreams_formula():
    ids = inp.fluid_sim_ids()
    assert len(ids) == len(set(ids)) == 400 and ids[:5] == [0, 22, 45, 67, 89] and ids[-3:] == [7954, 7977, 7999]

def test_fluid_ln_re_is_upstreams_reynolds_number():
    assert inp.fluid_ln_re(np.array([0, 7999])) == pytest.approx(np.array([4 / 3, 4 / 3 * 8.999]) * np.log(2))

def test_fluid_split_zero_is_upstreams_first_split():
    test, pool = inp.fluid_split(0); perm = np.random.default_rng(0).permutation(400)
    assert np.array_equal(test, perm[:100]) and np.array_equal(pool, perm[100:])

def test_rips_diagram_equals_ripser_as_pairs():
    X = np.random.default_rng(3).normal(size=(20, 2)); ref = ripser.ripser(X)["dgms"]
    assert all(inp.same_pairs(mine, theirs) for mine, theirs in zip(inp.rips_diagram(X), ref))

def test_same_pairs_sees_a_repairing():
    a = np.array([[0.0, 2.0], [1.0, 3.0]]); b = np.array([[0.0, 3.0], [1.0, 2.0]])
    assert not inp.same_pairs(a, b) and inp.same_pairs(a, a[::-1])
```

  Add four `@needs_upstream` tests:
  - `test_interpolation_regenerates_from_seed_zero`: every `clouds` and `y` array equal;
  - `test_curvature_trains_on_the_first_fifty_clouds`: the 50 training K are `linspace(-2, 2, 50)`;
  - `test_fluid_diagrams_have_one_h0_and_two_h1_essential_bars`: t = 5, the first 20 simulations;
  - `test_fluid_cached_w2_is_symmetric_and_complete`: t = 1, h = 0, finite.
- [ ] **Step 2: Run them.** Expected: failures on the missing `inputs`.
- [ ] **Step 3: Implement `inputs.py`.**
- [ ] **Step 4: Run the tests.** Expected: all pass.
- [ ] **Step 5: Run the stage** with `$PY inputs.py` (about a minute; log to `logs/inputs.log`). Expected:
  `results/inputs.json` with G5 passed, the lowest birth −25.09…, and shift 26.
- [ ] **Step 6:** Add the stage's line to `progress.md`, then commit `inputs.py`, `test_study.py`, `progress.md` and
  `results/inputs.json`: "Add the Betti FPCR benchmark's inputs and their gate".

### Task 3: Exact 2-Wasserstein distance

**Files:** Create `wasserstein.py`; Modify `test_study.py`.

**Interfaces:**
- Produces, in `wasserstein.py`:
  - `w2(a: np.ndarray, b: np.ndarray) -> float`, between two diagrams of one dimension, `(n, 2)`, whose deaths
    may be `inf`;
  - `w2_matrix(diagrams: Sequence[np.ndarray], n_jobs: int = 1) -> np.ndarray`, symmetric with a zero diagonal.

The algorithm is gudhi's `wasserstein_distance(order=2, internal_p=2)` with the essential parts kept.
- **Finite parts** A (n × 2) and B (m × 2) give an (n + m) × (n + m) cost matrix C:
  - `C[:n, :m] = ‖A_i − B_j‖²`;
  - `C[:n, m:]` = `(d_i − b_i)² / 2` for row i, the same in every column;
  - `C[n:, :m]` = `(d'_j − b'_j)² / 2` for column j, the same in every row;
  - `C[n:, m:] = 0`.

  The matching is `scipy.optimize.linear_sum_assignment(C)`, and the finite cost is the sum of the matched
  entries.
- **Essential parts:** both diagrams' essential births are sorted and paired, at cost Σ Δ². Unequal counts give
  `inf`.
- **Result:** `√(finite + essential)`. A non-finite birth raises `ValueError`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_w2_of_a_point_against_nothing_is_its_distance_to_the_diagonal():
    assert ws.w2(np.array([[1.0, 3.0]]), np.empty((0, 2))) == pytest.approx(2 / np.sqrt(2))

def test_w2_prefers_the_diagonal_when_it_is_cheaper():
    a, b = np.array([[0.0, 0.2]]), np.array([[5.0, 5.4]])
    assert ws.w2(a, b) == pytest.approx(np.sqrt(0.2**2 / 2 + 0.4**2 / 2))

def test_w2_matches_essential_births_in_sorted_order():
    a = np.array([[0.0, np.inf], [0.5, np.inf]]); b = np.array([[0.4, np.inf], [0.1, np.inf]])
    assert ws.w2(a, b) == pytest.approx(np.sqrt(0.1**2 + 0.1**2))

def test_w2_is_infinite_for_unequal_essential_counts():
    assert ws.w2(np.array([[0.0, np.inf]]), np.empty((0, 2))) == np.inf
```

  Also:
  - `test_w2_matches_brute_force_on_small_diagrams`: 200 random pairs with n, m ≤ 3; the oracle is the minimum
    over all permutations of the augmented matrix; relative 1e-12;
  - `test_w2_matrix_is_symmetric_with_zero_diagonal`;
  - `@needs_upstream test_w2_reproduces_upstreams_cached_distance_of_small_fluid_diagrams`: the two t = 1
    diagrams with the fewest bars, both dimensions, relative 1e-12.
- [ ] **Step 2: Run the tests.** Expected: they fail.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run the tests.** Expected: they pass.
- [ ] **Step 5: Commit:** "Add the Betti FPCR benchmark's exact 2-Wasserstein distance".

### Task 4: Betti curves, exact matrices, fPCA projection

**Files:** Create `betti.py`; Modify `test_study.py`.

**Interfaces:**
- Produces, in `betti.py`:
  - `curves(diagrams: Sequence[np.ndarray], dim: int, shift: float = 0.0) -> list[tc.ph.BettiCurve]`: the finite
    bars of each diagram, every endpoint plus `shift`;
  - `gram(curves, dim: int) -> np.ndarray`: `tc.ph.betti_inner_product_matrix(curves, dim=dim, n_jobs=-1)`;
  - `l2_matrix(curves, dim: int) -> np.ndarray`: `tc.ph.betti_l2_distance` for every pair (no Gram cancellation),
    symmetric, with a zero diagonal;
  - `fpca_scores(G: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, with
    shapes `(len(train), rank)` and `(len(test), rank)`. It runs
    `r = tc.ph.fpca_from_gram(G[np.ix_(train, train)], normalization="none")`; the training scores are
    `r.scores`. topocyte's convention centres as `K_c = H K H`, with eigenpairs `(ρ, V)` and training scores
    `V √ρ`, so the test scores are this exact projection:

```python
K, g = G[np.ix_(train, train)], G[np.ix_(test, train)]
S_test = (g - g.mean(axis=1, keepdims=True) - K.mean(axis=0) + K.mean()) @ (r.eigenvectors / np.sqrt(r.gram_eigenvalues))
```

- [ ] **Step 1: Write the failing tests.** The oracle is `overlap(A, B)` =
  Σᵢⱼ max(0, min(dᵢ, d'ⱼ) − max(bᵢ, b'ⱼ)), the full-line inner product of two Betti curves, summed with
  `math.fsum`.
  - `test_gram_is_the_sum_of_bar_overlaps`: random bars ≥ 0, relative 1e-12.
  - `test_shift_makes_the_gram_full_line`: bars with negative births; `gram` at `shift=s` equals `gram` at
    `shift=s + 10` to 1e-12, and both equal `overlap` on the unshifted bars.
  - `test_unshifted_negative_bars_lose_their_part_below_zero`: this pins topocyte's rule.
    ```python
    assert tc.ph.betti_norm(tc.ph.BettiCurve(np.array([[-2.0, 1.0]]), dim=1), dim=1) == pytest.approx(1.0)
    ```
    The full-line norm would be √3.
  - `test_l2_matrix_is_the_overlap_distance`: ‖f − g‖² = ⟨f, f⟩ + ⟨g, g⟩ − 2⟨f, g⟩ by `overlap`, relative 1e-9.
  - `test_fpca_scores_reproduce_topocyte_fit_and_transform` (G3's oracle): 30 random curves in each dimension,
    20 training, 10 test. Training scores equal `tc.ph.FPCA(dim=h, eigen_solver="dense").fit(tr).result_.scores`,
    and test scores equal its `.transform(te)`, to 1e-9 × the largest score.
  - `test_fpca_scores_of_an_all_empty_dimension_has_no_columns`: the shapes are `(n, 0)` and `(m, 0)`, with no
    warning under `-W error`.
- [ ] **Step 2: Run the tests.** Expected: they fail.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run the tests.** Expected: they pass.
- [ ] **Step 5: Commit:** "Add the Betti FPCR benchmark's curves, exact matrices and fPCA projection".

### Task 5: Estimators, Nadaraya–Watson and FPCR

**Files:** Create `estimators.py`; Modify `test_study.py`.

**Interfaces:**
- Consumes: `betti.fpca_scores`.
- Produces, in `estimators.py`:
  - `nw_fit_predict(D_train: np.ndarray, y: np.ndarray, D_test: np.ndarray) -> tuple[np.ndarray, float]`, for
    arms K and B. `D_test` is test by train; it returns the predictions and h.
    - The grid is `np.linspace(0.1 * m, 2.0 * m, 50)`, with m = `np.median(D_train[D_train > 0])`.
    - The weights are `exp(−(D/h)²/2) / √(2π)`.
    - The LOO leaves out the sample itself (zero diagonal weight). A zero weight sum gives `(y.sum() − yᵢ)/(n − 1)`.
    - h is `hs[np.argmin(mse)]`.
    - In prediction, a zero weight sum gives `y.mean()`.
    - This is upstream's `fluid/nw_regression.py`, which upstream states equals `kernel_regression.kernel_estimator`.
  - `loo_select(S0: np.ndarray, S1: np.ndarray, y: np.ndarray, k_max: int = K_MAX) -> tuple[int, int, float]`.
    - The candidates are (K0, K1) with 0 ≤ K_d ≤ min(k_max, S_d.shape[1]) and K0 + K1 ≤ n − 2, ordered by
      (K0 + K1, K0).
    - Z = [1, S0[:, :K0], S1[:, :K1]], with thin SVD Z = U Σ Wᵀ.
    - A candidate is ineligible if σ_min ≤ 1e-12 σ_max, or if max hᵢ ≥ `LEVERAGE_CAP`, where hᵢ = Σⱼ Uᵢⱼ².
    - e = y − U Uᵀ y, and the LOO MSE is mean((e / (1 − h))²).
    - The first minimum wins. (0, 0) is always eligible.
  - `fpcr_fit_predict(G0: np.ndarray, G1: np.ndarray, y_train: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, int, int]`.
    It runs `fpca_scores` per dimension, then `loo_select` on the training scores, then OLS with
    `np.linalg.lstsq` on the chosen columns, and predicts the test rows.

- [ ] **Step 1: Write the failing tests.**
  - `test_nw_matches_a_loop_transcription_of_upstreams_estimator`: the test carries a loop transcription of
    upstream's `kernel_estimator`: the bandwidth loop, `np.delete` and the fallbacks. On a random symmetric D and
    y, the predictions are equal to 1e-12 and h is the same.
  - `test_nw_falls_back_to_means_when_every_weight_underflows`: test distances of 1e6 give predictions
    `== y.mean()`.
  - `test_nw_median_ignores_zero_distances`: duplicate training rows, and the grid is built from the median of the
    positive entries.
  - `test_loo_select_press_equals_refitting_without_each_sample`: random S0 (15 × 4), S1 (15 × 3) and y. For the
    chosen pair and five other eligible pairs, the closed form equals the mean squared error of 15 `lstsq`
    refits, relative 1e-10.
  - `test_loo_select_picks_the_mean_for_constant_targets`: the result is (0, 0).
  - `test_loo_select_respects_n_minus_two_and_k_max`: n = 6 gives K0 + K1 ≤ 4, and `k_max=2` gives no K_d > 2.
  - `test_loo_select_uses_only_dimensions_with_components`: S1 with 0 columns gives K1 == 0.
  - `test_fpcr_recovers_a_linear_functional_of_the_scores`: y = 2 + 3 ξ₁ − ξ₂ of the training fit's H0 scores.
    The chosen K0 is at least 2, and the test predictions equal 2 + 3 ξ₁ − ξ₂ of the projected test scores
    (1e-8).
  - `test_arms_give_finite_predictions_with_empty_curves`: 40 synthetic diagrams, 5 of them with only essential
    bars. `nw_fit_predict` and `fpcr_fit_predict` give finite predictions.
- [ ] **Step 2: Run the tests.** Expected: they fail.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run the tests.** Expected: they pass.
- [ ] **Step 5: Commit:** "Add the Betti FPCR benchmark's Nadaraya-Watson and FPCR estimators".

### Task 6: Matrices stage with gates G2 and G4

**Files:** Create `matrices.py`; Modify `test_study.py`, `progress.md`.

**Interfaces:**
- Consumes: `inputs`, `wasserstein.w2`, `wasserstein.w2_matrix`, `betti.curves`, `betti.gram`, `betti.l2_matrix`.
- Produces, in `matrices.py`:
  - `mkey(task: str, kind: str, h: int, idx: int | None = None) -> str`. Its forms are `"interp_{idx:02d}_{kind}_{h}"`,
    `"curv_{kind}_{h}"` and `"fluid_{idx}_{kind}_{h}"`, with `kind ∈ ("w2", "l2", "gram")`. idx is the
    interpolation dataset (0–99) or the fluid time's index in `FLUID_TIMES`.
  - `results/matrices.npz` holds every matrix:
    - interpolation: `(n_train + 50)²`, in `interpolation_datasets()` order, training clouds first;
    - curvature: 100²;
    - fluid: 400², in `fluid_sim_ids()` order;
    - fluid `w2` is `fluid_cached_w2`; the fluid curves are shifted by `FLUID_SHIFT`; the Rips curves are not
      shifted;
    - the targets are `interp_{idx:02d}_y`, `curv_y` and `fluid_y` (ln Re).
  - `load_matrices() -> dict[str, np.ndarray]`, which loads with `allow_pickle=False`.
  - `g2_sample(bars: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int, int]]`.
    - `bars[t, h, i]` is the number of finite bars, and the result lists (time index, h, p, q).
    - It draws 100 uniform same-time pairs and both dimensions, rejecting checks whose n + m > 4,000.
    - It then adds pairs until at least 10 checks have more than 1,000 finite bars on both sides.
    - The stage passes `np.random.default_rng(2)`.
  - The stage `main()` computes:
    - G2: `w2` against `fluid_cached_w2` on `g2_sample`'s checks;
    - G4: every fluid Gram matrix at `FLUID_SHIFT` against `FLUID_SHIFT + 10`.

    It writes the npz only if both pass. If G2 fails, it exits with status 1 and the failure goes to Simon
    (`design.md`). It writes `results/matrices.json` with the gates, the shapes, `provenance(...)` and the
    runtime.

- [ ] **Step 1: Write the failing tests:**

```python
def test_mkey_names_every_matrix():
    assert mat.mkey("interp", "w2", 1, 7) == "interp_07_w2_1"
    assert mat.mkey("fluid", "gram", 0, 4) == "fluid_4_gram_0" and mat.mkey("curv", "l2", 0) == "curv_l2_0"
```

  Also:
  - `test_g2_sample_caps_size_and_has_ten_large_checks`: fake counts `bars[5, 2, 400] ~ U(0, 3000)`. Every
    check has n + m ≤ 4,000, there are at least 10 with both sides above 1,000, and at least 100 checks;
  - `test_dataset_matrices_of_a_tiny_task`: 6 synthetic clouds through the stage's per-dataset function. The six
    matrices are symmetric; w2 and l2 have a zero diagonal; gram's smallest eigenvalue is ≥ −1e-9 × its trace.
- [ ] **Step 2: Run the tests.** Expected: they fail.
- [ ] **Step 3: Implement** the module. Parallelism is the builder's call; the whole stage should take under an
  hour.
- [ ] **Step 4: Run the tests.** Expected: they pass.
- [ ] **Step 5: Run the stage in the background:**
  `caffeinate -is /usr/bin/time -l $PY matrices.py > logs/matrices.log 2>&1`. Expected: `matrices.json` with G2
  and G4 passed, and `matrices.npz` written.
- [ ] **Step 6:** Add the stage's line to `progress.md`, then commit `matrices.py`, `test_study.py`, `progress.md`
  and `results/matrices.json`: "Compute the Betti FPCR benchmark's exact matrices and their gates".

### Task 7: Arms stage: K, gate G1, gate G3, F, B

**Files:** Create `run_study.py`; Modify `test_study.py`, `progress.md`.

**Interfaces:**
- Consumes: `matrices.load_matrices`, `matrices.mkey`, `estimators`, `inputs`, `betti.curves`,
  `betti.fpca_scores`.
- Produces, in `run_study.py`:
  - `fluid_cells() -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]`, yielding
    (time index, split, size index, train, test), where `test, pool = fluid_split(s)` and `train = pool[:n]`;
  - `run_arm(arm: str, task: str, M: dict) -> dict[str, np.ndarray]` for arms `"K"` (D = w2_0 + w2_1),
    `"B"` (D = l2_0 + l2_1) and `"F"` (gram_0, gram_1). Interpolation trains on `arange(n)` and tests on
    `arange(n, n + 50)`; curvature on 0–49 and 50–99;
  - `g1(k: dict) -> dict`, against upstream's CSVs:
    - interpolation: per size, the mean and SD (ddof = 1) of `interp_r2_K` against `PD_kernel_R2_mean` and
      `PD_kernel_R2_std`;
    - curvature: `curv_mse_K` against the `PD_kernel` row;
    - fluid: R² at splits 0–4 against `results_fluid_splits.csv`, keyed by `(t, split, n_train)`.
    It passes if and only if every |diff| ≤ `G1_TOL`.
  - `g3(M: dict) -> dict`, on these cells:
    - interpolation datasets 0, 45 and 99;
    - curvature;
    - fluid (time 0, split 0, n 25), (2, 7, 100) and (4, 99, 300);

    in both dimensions. It compares `betti.fpca_scores`' test rows with
    `tc.ph.FPCA(dim=h, eigen_solver="dense").fit(train curves).transform(test curves)`, the curves rebuilt from
    `inputs`.
  - `results/arms.npz`, for arm ∈ {K, F, B}:
    - `interp_pred_{arm}` (100, 50), `interp_r2_{arm}` (100,);
    - `curv_pred_{arm}` (50,), `curv_mse_{arm}` ();
    - `fluid_pred_{arm}` (5, 100, 5, 100), indexed [time, split, size, test], and `fluid_r2_{arm}` (5, 100, 5);
    - for K and B, h as `{task}_h_{arm}`; for F, `interp_k_F` (100, 2), `curv_k_F` (2,) and `fluid_k_F`
      (5, 100, 5, 2);
    - `interp_y_test` (100, 50), `curv_y_test` (50,), `fluid_y` (400,) and `fluid_test` (100, 100).
  - `results/arms.json`: G1 (the worst |diff| per task and the cells compared), G3 (the worst relative difference
    per task), the runtime per arm, and `provenance(...)`.
  - The CLI `run_study.py arms` runs K on every task, then G1. If G1 fails, it writes `arms.json` with
    `"g1": {"passed": false, ...}` and exits with status 1, and F and B never run. Then G3, which exits with
    status 1 on failure. Then F, then B, then it saves.

- [ ] **Step 1: Write the failing tests.**
  - `test_fluid_training_sets_are_nested_prefixes_of_the_pool`: for split 3, the training sets of the five sizes
    are prefixes of one another and never meet the test set.
  - `test_g1_passes_within_tolerance_and_fails_beyond`: K results 5e-10 away from a monkeypatched CSV pass;
    2e-9 away, they fail.
  - `test_arms_stage_stops_before_f_and_b_when_g1_fails`: with `g1` monkeypatched to fail, `run_arm` is called
    with `"K"` only, the exit code is 1, and `arms.json` says G1 failed.
  - `test_run_arm_on_a_tiny_fluid_task`: sizes, splits and simulations monkeypatched small; the output shapes and
    finite values.
- [ ] **Step 2: Run the tests.** Expected: they fail.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run the tests.** Expected: they pass.
- [ ] **Step 5: Run the stage in the background:**
  `caffeinate -is /usr/bin/time -l $PY run_study.py arms > logs/arms.log 2>&1`. Expected: G1 and G3 passed and
  `arms.npz` written. If G1 fails: stop and report to Simon, and change nothing.
- [ ] **Step 6:** Add the stage's line to `progress.md`, then commit `run_study.py`, `test_study.py`,
  `progress.md` and `results/arms.json`: "Run the Betti FPCR benchmark's arms after their gates".

### Task 8: Readouts and verdicts

**Files:** Create `readouts.py`; Modify `run_study.py` (add the `readouts` stage), `test_study.py`, `progress.md`.

**Interfaces:**
- Consumes: `results/arms.npz`.
- Produces, in `readouts.py`:
  - `interval_interp(delta: np.ndarray) -> tuple[float, float, float]`. delta is (10 sizes, 10 runs):

```python
rng = np.random.default_rng(BOOT_SEED)
idx = rng.integers(0, 10, size=(N_BOOT, 10, 10))
draws = np.take_along_axis(np.broadcast_to(delta, (N_BOOT, 10, 10)), idx, axis=2).mean(axis=(1, 2))
return float(delta.mean()), *map(float, np.percentile(draws, [2.5, 97.5]))
```

  - `interval_units(values: np.ndarray) -> tuple[float, float, float]`, with one value per unit:
    - curvature: a test cloud's `se_Y − se_X`;
    - fluid: a split's mean of Δ over the times and sizes;
    - descriptive: one size's 10 runs.

    `idx = default_rng(BOOT_SEED).integers(0, len(values), size=(N_BOOT, len(values)))`; the draws are
    `values[idx].mean(1)`; the percentiles are as above.
  - `verdict(lo: float, hi: float) -> str`: `"beats"` if lo > 0, `"below"` if hi < 0, `"level"` otherwise.
  - `readouts(arms: dict) -> dict`. Δ(X, Y) is R²(X) − R²(Y) for interpolation and fluid, and per test cloud
    `se_Y − se_X` for curvature, so positive favours X. It gives:
    - the primary (F, K) per task, with its verdict;
    - (B, K) and (F, B), exploratory, without a verdict;
    - Δ per size (interpolation) and per time (fluid), with intervals;
    - the fluid readout on splits 0–4 alone;
    - the counts of the chosen (K0, K1);
    - the paper's reported tables (upstream's CSVs) beside K, F and B.
  - `run_study.py readouts` writes `results/readouts.json`.

- [ ] **Step 1: Write the failing tests:**

```python
@pytest.mark.parametrize("lo,hi,expected", [(0.01, 0.2, "beats"), (-0.2, -0.01, "below"), (-0.1, 0.1, "level"),
                                            (0.0, 0.1, "level"), (-0.1, 0.0, "level")])
def test_verdict_follows_the_rule_fixed_in_advance(lo, hi, expected):
    assert ro.verdict(lo, hi) == expected

def test_intervals_of_identical_arms_are_zero():
    assert ro.interval_interp(np.zeros((10, 10))) == (0.0, 0.0, 0.0)

def test_interval_interp_resamples_runs_within_sizes():
    delta = np.repeat(np.linspace(-1, 1, 10)[:, None], 10, axis=1)   # no variation within a size
    mean, lo, hi = ro.interval_interp(delta)
    assert lo == pytest.approx(mean) and hi == pytest.approx(mean)
```

  Also:
  - `test_curvature_delta_is_signed_so_positive_favours_f`: se_K > se_F everywhere gives `"beats"`;
  - `test_interval_units_is_reproducible`: two calls are equal.
- [ ] **Step 2: Run the tests.** Expected: they fail.
- [ ] **Step 3: Implement** the module and the stage.
- [ ] **Step 4: Run the tests.** Expected: they pass.
- [ ] **Step 5: Run** `$PY run_study.py readouts > logs/readouts.log 2>&1`. Then add to `progress.md` the three
  verdicts, each with its mean Δ and interval, copied from `readouts.json`.
- [ ] **Step 6: Commit** `readouts.py`, `run_study.py`, `test_study.py`, `progress.md` and
  `results/readouts.json`: "Score the Betti FPCR benchmark's readouts: the verdicts".

### Task 9: Verification by other routes

**Files:** Create `verify_results.py`; Modify `progress.md`.

`verify_results.py` writes `results/verify.json`, with every check's measured value and tolerance, and the sha256
of the npz files it read. It exits with status 1 if a check fails. Its checks:

- [ ] **Step 1: Implement these checks:**
  - **W2:** 30 random interpolation and curvature entries (`default_rng(5)`) recomputed by
    `scipy.optimize.linprog` (HiGHS) on the transport LP of the augmented problem, an algorithm other than
    `linear_sum_assignment`; relative 1e-9.
  - **Gram:** 30 entries per task by the bar-overlap sum (`np.minimum.outer` / `np.maximum.outer`, `math.fsum`),
    relative 1e-12. L2 entries from the same sums, relative 1e-9.
  - **G1,** recomputed from `arms.npz` against upstream's CSVs.
  - **NW:** 5 cells per task and arm (K, B), recomputed by a loop transcription of upstream's `kernel_estimator`
    on the saved matrices: the same h and the predictions to 1e-12.
  - **FPCR:** 5 cells per task, recomputed with `tc.ph.FPCA(dim, eigen_solver="dense").fit/transform` on the
    curves (not the Gram) and scikit-learn's `LinearRegression` at the saved (K0, K1): predictions to 1e-8. For
    one cell with n_train = 10, the LOO choice is derived again by refitting without each sample.
  - **Readouts:** Δ, the intervals and the verdicts, recomputed from `arms.npz` by a bootstrap written separately
    with the same recipe and seed, equal to `readouts.json`.
  - **Provenance:** each md5 that a stage's json records equals that of the script committed with that json.
- [ ] **Step 2: Run** `caffeinate -is /usr/bin/time -l $PY verify_results.py > logs/verify.log 2>&1`. Expected:
  every check passes.
- [ ] **Step 3:** Add the stage's line to `progress.md`, then commit `verify_results.py`, `progress.md` and
  `results/verify.json`: "Verify the Betti FPCR benchmark's results by other routes".

### Task 10: Report, README, the repository's indexes

**Files:**
- Create: `build_report.py`, `README.md`
- Modify: `progress.md`, `../README.md` (the Studies table), `../EXPERIMENTS.md` (its recap of every study)

- [ ] **Step 1: Implement `build_report.py`.** It reads `results/*.json`, `arms.npz`, upstream's CSVs and the
  logs' `/usr/bin/time -l` lines, and refuses to run unless `verify.json` passed every check and its sha256 sums
  match. It writes `report.html` with its figures embedded as PNGs, in the sibling reports' look. No number is
  typed by hand. It holds:
  - the verdict per task;
  - the paper's three tables: reported `PD_kernel`, and `PI_kernel` or the Turkeš methods; K rerun; F; B;
  - R² against training size per task, as the paper's Figures 6 and 9: each arm's mean ± SD, and one panel per
    time for fluid;
  - the primary and exploratory Δ, with intervals;
  - the chosen (K0, K1);
  - the runtimes;
  - the caveats:
    - the curvature inputs;
    - the essential bars that F and B drop;
    - the single curvature training set;
    - the 100 fluid splits of the same 400 simulations;
    - the paper's swap of line and circle.
- [ ] **Step 2: Write `README.md`** in the sibling's shape:
  - a status line with the three verdicts and their intervals;
  - Simon's request;
  - the design in brief;
  - Results;
  - Run, with the commands as they ran;
  - Outputs;
  - the topocyte and upstream commits.

  `build_report.py` asserts that the README states the verdicts and intervals of `readouts.json`.
- [ ] **Step 3: Index the study.** Add the study's row (study, question, verdict) to the Studies table of
  `../README.md`, and its recap to `../EXPERIMENTS.md` in that file's style. Check `git -C .. diff --quiet --
  README.md EXPERIMENTS.md` first; if another session has uncommitted changes there, stop and ask.
- [ ] **Step 4: Run** `$PY build_report.py`. Expected: `report.html` is written.
- [ ] **Step 5: Commit** `build_report.py`, `README.md`, `progress.md`, `../README.md` and `../EXPERIMENTS.md`:
  "Report the Betti FPCR benchmark".
