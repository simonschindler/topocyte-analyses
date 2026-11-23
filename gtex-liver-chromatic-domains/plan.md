# Chromatic-domains pilot: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a pilot of 30 GTEx liver slides, find two levels of domains and test the second against a guard fixed in advance. Level-1 compartments come from the density and composition of 150 µm balls. Level-2 subdomains come from the local chromatic persistent homology of each ball, measured against random labellings. Then build an asserted report, and after Simon downloads the slides, an H&E atlas.

**Architecture:** One folder in topocyte-analyses, built like the sibling studies. A run script with subcommands writes `results/`; downstream scripts read only `results/`. A verifier recomputes by independent routes, and `build_report.py` asserts every claim it prints. The numerics live in three focused modules, each unit-tested on synthetic inputs:
- the port of chalc's chromatic filtration (`chromatic.py`);
- exact weighted step-function algebra (`curves.py`);
- the rule for k and the guard (`domains.py`).

**Tech stack:**
- A study venv on Python 3.13 with chalc 15.0.1 (test oracle), phimaker 0.5.0, gudhi 3.13.0, numba 0.68.1, numpy, scipy, scikit-learn, h5py, pandas, matplotlib and pytest, plus topocyte's own dependencies (anndata, ripser).
- topocyte at `18e3aa5` through `PYTHONPATH`.

**Spec:** `design.md` (this folder), approved by Simon on 2026-11-20 ("go", relayed by the parent session).

## Global Constraints

- **Cohort, read only:** opened with h5py in mode "r"; nothing is written under `~/data`. Slide rows come from `../gtex-liver-pathology-slices/results/density.npz` (`slide_ids`, `bounds`).
- **topocyte pin:** imported from the clean checkout `/Users/simon/rep/lbi/reindero/topocyte-18e3aa5` at `18e3aa5`, named first on `PYTHONPATH`. Every script refuses another topocyte, or a checkout whose `topocyte/` differs from the commit (the sibling's check).
- **Venv:** the study venv `.venv/` in this folder, Python 3.13, with direct dependencies pinned in `requirements.txt`. `.venv/` is gitignored.
- **Units:**
  - level-0 pixels at 0.4942 µm/px;
  - `DELTA_PX = 150 / 0.4942` (303.52084176446783), closed ball;
  - filtration values are squared smallest-enclosing-ball radii in px² (topocyte's α² convention).
- **Exact bar-endpoint arithmetic only:** no grids, quadrature or changes of measure; no curve is ever evaluated on a grid.
- **Classes:** the four groups of `../gtex-liver-celltype-slices/common.py` (`GROUP_MEMBERS`), codes epithelial 0, stromal 1, immune 2, other 3.
- **Sizes:**
  - 30 slides, 5 per condition: healthy, congestion, steatosis, inflammation/hepatitis, fibrosis, cirrhosis;
  - halves of 15;
  - 1,500 centres per slide;
  - R = 19 random labellings;
  - fit samples of 400 centres per slide;
  - 9 fPCA components per block, 20 MFA components.
- **Seeds:**
  - `SEED = 20261120`;
  - every per-slide, per-centre and per-labelling generator comes from `np.random.SeedSequence([SEED, slide_index, centre_index, stream])`;
  - k-means `random_state=0`, `n_init=20`.
- **Chromatic engine:**
  - faces up to dimension 2;
  - phimaker `num_threads=1`;
  - numba kernels serial;
  - the run uses 8 processes (spawn), never threads around the port.
- **Rule for k** (design step 10):
  - gap statistic against a matched Gaussian, B = 20;
  - Tibshirani's stopping rule from k = 1: k̂ is the smallest k ≥ 1 with Gap(k) ≥ Gap(k+1) − s_{k+1}, k_max if none (corrected 2026-11-21, design step 10);
  - if k̂ ≥ 2, its gain Gap(k̂) − Gap(1) must be ≥ 0.03 (Simon, 2026-11-21; first 0.02), else 1;
  - k = the smaller of the two halves' choices;
  - prediction strength ≥ 0.8 in both directions, else k = 1;
  - k_max is 10 for compartments and 6 for subdomains;
  - no subdomains for a compartment with fewer than 500 centres in either half.
- **Guard** (design "Checks fixed in advance", 2):
  - `HistGradientBoostingClassifier(random_state=0)` on the 15 pairwise, density and composition features;
  - balanced accuracy;
  - 4000 bootstrap draws of the test half's slides, `default_rng(12345)`;
  - a compartment passes if BA_stab − BA_pw has its 95% interval above 0 in both directions;
  - the study's verdict is "adds" if a compartment with at least 10% of all centres passes.
- **Control:** labelling 1 against labellings 2..19, projected into the observed bases. It must give k = 1 in every compartment, or level 2 stops and the report says so.
- **Level-1 compartments are composition strata** (Simon, 2026-11-21; design step 8). The rule is unchanged. The report never calls them modes. The pre-flight gains a cluster-free composition null (Dirichlet(6, 2, 1, 1) fractions, Poisson(120) counts, 22,500 × 5) that is reported, not gated; Task 7 adds it.
- **Commits:**
  - local only, in topocyte-analyses, in the repository's message style with the Co-Authored-By line; never push;
  - per-task commits only if Simon agrees at hand-off;
  - `design.md` and `plan.md` first, before any result.

## Review Focus

1. **A ball with one class, or a class with one cell.** The lift has s = 1, image = domain = codomain, and the shuffles all agree (s = 0). Expected: empty or zero curves, no crash. A zero-spread centre with a non-zero residual keeps a finite value through s0. Tests: `test_single_class_ball` (Task 2), `test_zero_spread_nonzero_residual_is_finite` (Task 6).
2. **Coincident cell coordinates of different classes** (duplicate HistoPLUS centroids). Expected: the same six-packs as chalc, and valid pair counts. Test: `test_coincident_points_match_chalc` (Task 2).
3. **Tiny or degenerate balls** at tissue edges: 1 or 2 cells, collinear cells. Expected: valid diagrams and pair counts from GUDHI's exact Delaunay, with no Qhull error. Test: `test_tiny_and_collinear_balls` (Task 2).
4. **Dense balls** of 500+ cells under 8 processes. Expected: completes, with the per-slide maximum ball size, time and peak RSS logged. Tests: `test_dense_ball` (Task 2); the logs come from Task 5.
5. **An interrupted run.** Expected: a rerun of `bars` skips slides whose npz is complete and redoes a partial one. Test: `test_resume_skips_complete_and_redoes_partial` (Task 5).

---

### Task 1: Scaffold, venv, pilot draw

**Files:**
- Create: `README.md`, `progress.md`, `requirements.txt`, `common.py`, `run_pilot.py` (subcommand `pilot`), `test_common.py`.
- Modify: `../.gitignore` (add `.venv/`), `../README.md` (Studies row: "Planned: design and plan").

**Interfaces (produces), in `common.py`:**
- **Constants:**
  - `CLASSES`, `GROUP_MEMBERS`, `CONDITIONS` (in the order above), `PX_UM`, `DELTA_UM`, `DELTA_PX`;
  - `R = 19`, `N_CENTRES = 1500`, `PER_CONDITION = 5`, `FIT_PER_SLIDE = 400`, `SEED = 20261120`, `TOPOCYTE_COMMIT = "18e3aa5"`;
  - `MODULES = ("domain", "image")`;
  - `BLOCKS = tuple((t, m, p) for t in range(4) for m in MODULES for p in (0, 1))` (16, indexed by b);
  - `PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))` (10);
  - paths `HERE`, `RESULTS`, `LOGS`, `H5AD`, `SIB`, `CT`.
- **Readers and checks:**
  - `check_topocyte() -> None`;
  - `class_codes(categories: Sequence[str]) -> np.ndarray` (int8; `ValueError` names any unmapped category);
  - `read_slide(slide_id: str) -> tuple[np.ndarray, np.ndarray]`: float64 xy (n, 2) and int8 classes (n,), read only.
- **Pilot draw:**
  - `eligible_slides(holdout_csv: Path, pathology_csv: Path) -> pd.DataFrame` with `slide_id, condition, n_cells`. It keeps development slides (`holdout` False), no `quality_flag`, and a condition in `CONDITIONS`.
  - `draw_pilot(eligible: pd.DataFrame, seed: int = SEED) -> pd.DataFrame` with `slide_id, condition, half, slide_index`. Per condition, in `CONDITIONS` order: the candidates sorted by `slide_id`, 5 drawn without replacement, then a permutation whose first 3 (even condition index) or 2 (odd) go to half A. `slide_index` is 0..29 in table order.
- **Centres and labellings:**
  - `draw_centres(n_cells: int, slide_index: int) -> np.ndarray`: 1,500 distinct rows, in draw order (the order is the centre index).
  - `labelling(cls: np.ndarray, slide_index: int, centre_index: int, r: int) -> np.ndarray`: `cls` for r = 0, else a seeded permutation of it.
- `results/pilot.csv`.

- [ ] **Step 1: Clean topocyte checkout and venv**

```bash
git -C /Users/simon/rep/lbi/reindero/topocyte worktree add --detach /Users/simon/rep/lbi/reindero/topocyte-18e3aa5 18e3aa5
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip freeze --python .venv/bin/python > results/venv_freeze.txt
```

`requirements.txt` pins:
- the engines: chalc==15.0.1, phimaker==0.5.0, gudhi==3.13.0, numba==0.68.1;
- the stack: numpy==2.5.5, scipy==1.18.1, scikit-learn==1.9.1, h5py==3.16.0, pandas==3.1.1, matplotlib==3.11.2;
- topocyte's own dependencies and tools: anndata==0.13.4, ripser==0.6.15, psutil==7.2.2, and pytest.

Expected: `PYTHONPATH=../../topocyte-18e3aa5 .venv/bin/python -c "import topocyte, chalc, phimaker"` succeeds.

- [ ] **Step 2: Write the failing tests** in `test_common.py`

```python
def test_class_codes_maps_all_14(): ...  # the cell-type study's 14-category table, same groups
def test_class_codes_unknown_raises():
    with pytest.raises(ValueError, match="Hepatocyte"):
        common.class_codes(["Hepatocyte"])
def test_delta_px():
    assert common.DELTA_PX == pytest.approx(303.52084176446783, abs=1e-12)
def test_blocks_and_pairs():
    assert len(common.BLOCKS) == 16 and common.BLOCKS[:2] == ((0, "domain", 0), (0, "domain", 1))
    assert len(common.PAIRS) == 10 and common.PAIRS[:2] == ((0, 0), (0, 1))
def test_eligible_excludes_holdout_and_preservation(tmp_path): ...  # fake csvs: holdout, "preservation", necrosis rows dropped
def test_draw_pilot_stratified_and_reproducible():
    p = common.draw_pilot(fake_eligible(per_condition=8))
    assert p.equals(common.draw_pilot(fake_eligible(per_condition=8)))
    assert p.groupby("condition").size().to_dict() == {c: 5 for c in common.CONDITIONS}
    assert [((p.condition == c) & (p.half == "A")).sum() for c in common.CONDITIONS] == [3, 2, 3, 2, 3, 2]
    assert list(p.slide_index) == list(range(30))
def test_draw_centres():
    a = common.draw_centres(10_000, 3)
    assert a.shape == (1500,) and np.unique(a).size == 1500 and a.max() < 10_000
    assert np.array_equal(a, common.draw_centres(10_000, 3))
def test_labelling_keeps_counts_and_is_reproducible():
    cls = np.repeat(np.arange(4), [40, 30, 20, 10]).astype(np.int8)
    assert np.array_equal(common.labelling(cls, 4, 17, 0), cls)
    a = common.labelling(cls, 4, 17, 5)
    assert np.array_equal(a, common.labelling(cls, 4, 17, 5))
    assert np.array_equal(np.bincount(a, minlength=4), [40, 30, 20, 10])
    assert not np.array_equal(a, common.labelling(cls, 4, 17, 6))
```

- [ ] **Step 3: Run them:** `.venv/bin/python -m pytest -q test_common.py`. Expected: FAIL (no module `common`).
- [ ] **Step 4: Implement `common.py` and `run_pilot.py pilot`.** `pilot` calls `check_topocyte()`, writes `results/pilot.csv`, and prints the counts per condition and half.
- [ ] **Step 5: Run the tests and the draw.** pytest passes, and `PYTHONPATH=../../topocyte-18e3aa5 .venv/bin/python run_pilot.py pilot` writes 30 rows, 15 per half. Record the slide list in `progress.md`.
- [ ] **Step 6: Commit** `design.md`, `plan.md` and the scaffold, before any result (if agreed).

### Task 2: Chromatic six-packs and pair counts (`chromatic.py`)

**Files:** Create `chromatic.py`, `test_chromatic.py`.

`chromatic.py` is the port from chalc-scaling `8f370c6:report/port/chromatic_port.py`. Its header credits chalc and di Montesano et al. (arXiv:2212.03128) and states the changes:
- values squared;
- faces only up to dimension 2 by default; a private `_max_face_dim=3` route is kept for the test;
- one boundary matrix per complex, reused for every class present;
- classes keep their codes 0..3, and the lift uses only the classes present.

**Interfaces:**
- **Consumes:** `common.PAIRS`.
- **Produces:**
  - `ball_sixpacks(xy: np.ndarray, cls: np.ndarray, *, full: bool = False) -> dict[tuple[str, int, int], np.ndarray]`.
    - Keys are `(module, t, p)` for module in `("domain", "image")`, t in 0..3 and p in (0, 1), plus `("codomain", -1, p)`. With `full`, there are also `"kernel"`, `"cokernel"` and `"relative"`.
    - The values are `(n, 2)` float64 bars in px², with `death > birth` and essential bars as `inf`. An absent class gives `(0, 2)`.
  - `pair_counts(xy: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, int]`: Delaunay edges of the ball's cells per `PAIRS`, from GUDHI `DelaunayComplex` (exact predicates), and `n_edges`. With n < 2 there are no edges.
  - `pair_expectation(class_counts: np.ndarray, n_edges: int) -> np.ndarray`: `n_edges·n_a·n_b/C(n,2)` for a ≠ b and `n_edges·C(n_a,2)/C(n,2)` for a = b.

- [ ] **Step 1: Write the failing tests**

```python
CLOUDS = [(n, s, seed) for seed, (n, s) in enumerate([(30, 1), (60, 2), (120, 3), (200, 4), (150, 4)])]
@pytest.mark.parametrize("n,s,seed", CLOUDS)
def test_matches_chalc(n, s, seed): ...
    # xy ~ uniform(0, 300)^2, cls ~ integers(0, s); chalc SubChromaticInclusion(f, [code(t)]).sixpack(1),
    # chalc radii squared; all six modules, every class present, degrees 0-1:
    # same bar count and bottleneck <= 1e-9 * scale
@pytest.mark.parametrize("n,s,seed", CLOUDS)
def test_betti_identities(n, s, seed): ...
    # at every bar endpoint: dom = ker + im, cod = im + cok, rel_p = cok_p + ker_{p-1}; max violation == 0
def test_dim2_equals_dim3(): ...          # full six-packs identical with _max_face_dim=3
def test_codomain_is_alpha_ph(): ...      # vs tc.ph.compute_persistent_homology(xy, filtration="alpha", max_dim=1)
def test_absent_class_is_empty(): ...     # classes {0, 2}: keys for 1 and 3 exist with shape (0, 2)
def test_single_class_ball(): ...         # domain == image == codomain; kernel, cokernel empty
def test_coincident_points_match_chalc(): ...  # 150 points + 50 copies of their coordinates with other classes
def test_tiny_and_collinear_balls(): ...  # n = 1: no finite bars, no edges; n = 2 of two classes: codomain H0 = one finite bar
                                          # [0, (dist/2)^2) + one essential, pair_counts one edge; 5 collinear: no error, valid bars
def test_dense_ball(): ...                # 700 points, 4 classes: identities hold, finishes in < 5 s
def test_pair_expectation_exhaustive():   # cls = [0,0,1,1,2,3] on a fixed 6-point cloud
    # the mean of pair_counts over all 180 distinct labellings == pair_expectation, as fractions
def test_pair_counts_sum(): ...           # counts.sum() == n_edges on a random cloud
```

- [ ] **Step 2: Run:** `.venv/bin/python -m pytest -q test_chromatic.py`. Expected: FAIL (no module).
- [ ] **Step 3: Implement `chromatic.py`.** Port the code. Square the values after the facet-max pass. Build the matrix once, and set `in_dom = (mask & ~(1 << code_t)) == 0` per class.
- [ ] **Step 4: Run:** all pass. Note the chalc and test timings in `progress.md`.
- [ ] **Step 5: Commit** (if agreed).

### Task 3: Exact weighted step functions (`curves.py`)

**Files:** Create `curves.py`, `test_curves.py`.

**Interfaces (produces):**
- **The data type:** `@dataclass CurveSet(b, d, w, offsets)`. Curve i is `Σ w·1[b, d)` over its rows `offsets[i]:offsets[i+1]`. Finite bars only, float64.
- **Building curve sets:**
  - `curve_set(diagrams: Sequence[np.ndarray]) -> CurveSet`: one curve per diagram, unit weights, essential bars dropped (topocyte's reduced curve).
  - `combine(sets: Sequence[CurveSet], coeffs: Sequence[float]) -> CurveSet`: curve i is `Σ_j coeffs[j]·sets[j][i]`.
  - `scale(cs: CurveSet, factors: np.ndarray) -> CurveSet`.
- **Inner products:**
  - `inner_matrix(a: CurveSet, b: CurveSet | None = None, n_jobs: int = -1) -> np.ndarray`: exact L² inner products by a merged sweep per pair, numba `prange`; the Gram matrix when `b` is None.
  - `spread2(refs: Sequence[CurveSet]) -> np.ndarray`: per curve index, `(1/R)·Σ_r ‖ref_r‖² − ‖mean_r ref_r‖²` from the R × R inner products.
- **Components and scores:**
  - `@dataclass Components(x, phi, mean_term)`: breakpoints `x` (M,), values `phi` (M, K) on `[x_m, x_{m+1})`, and `mean_term` (K,) = `⟨f̄, φ_k⟩`.
  - `components(fit: CurveSet, result: FPCAResult, n_components: int) -> Components`: `φ_k = Σ_i v_ik/√ρ_k · f_i`, with event sums by Neumaier-compensated summation.
  - `score(cs: CurveSet, comp: Components) -> np.ndarray`: `⟨g, φ_k⟩ − mean_term_k`, from the antiderivative of φ at each bar endpoint.

- [ ] **Step 1: Write the failing tests**

```python
def test_inner_matches_topocyte():          # 50 random diagrams, unit weights
    G = curves.inner_matrix(curves.curve_set(diags))
    T = tc.ph.betti_inner_product_matrix([tc.ph.BettiCurve(d, 0) for d in diags], dim=0)
    assert np.allclose(G, T, rtol=1e-12, atol=0)
def test_signed_weights_bilinear(): ...      # <a - b/19, c> == <a, c> - <b, c>/19, rel 1e-12
def test_spread_matches_bruteforce(): ...   # spread2 == mean_r ||ref_r - mean||^2 (combine + inner), rel 1e-10
def test_scores_reproduce_fpca_from_gram(): # 300 random signed curves
    res = tc.ph.fpca_from_gram(curves.inner_matrix(cs))
    S = curves.score(cs, curves.components(cs, res, 9))
    assert np.max(np.abs(S - res.scores[:, :9])) <= 1e-9 * np.max(np.abs(res.scores[:, :9]))
def test_score_is_linear(): ...             # score(obs - mean refs) == score(obs) - mean(score(refs)), rel 1e-10
def test_component_sum_compensated(): ...   # 2e6 events of +-1e8 and +-1e-8: phi equals math.fsum per segment, rel 1e-12
def test_empty_curves(): ...                # curves without bars: zero rows/columns of the Gram, zero scores minus mean_term
```

- [ ] **Step 2: Run:** `.venv/bin/python -m pytest -q test_curves.py`. Expected: FAIL.
- [ ] **Step 3: Implement `curves.py`.**
- [ ] **Step 4: Run:** all pass, with `PYTHONPATH` on the topocyte checkout.
- [ ] **Step 5: Commit** (if agreed).

### Task 4: Rule for k and pre-flight

**Files:**
- Create: `domains.py` (functions only, for now), `test_domains.py`.
- Modify: `run_pilot.py` (subcommand `preflight`).

**Interfaces (produces), in `domains.py`:**
- `kmeans_fit(X: np.ndarray, k: int, seed: int = 0) -> KMeans`: `n_init=20`. For k = 1 the single centroid is the mean.
- `gap_choice(X: np.ndarray, k_max: int, *, B: int = 20, min_gain: float = 0.03, seed: int = 0) -> dict` with keys `k`, `gap` (k_max,) and `s` (k_max,), per design step 10.
- `ps_from_labels(test_labels: np.ndarray, train_assign: np.ndarray) -> float`: the minimum over test clusters of the share of within-cluster pairs that `train_assign` puts together. A test cluster of one centre counts as 0.
- `prediction_strength(train: np.ndarray, test: np.ndarray, k: int, *, seed: int = 0) -> float`: Tibshirani–Walther. `kmeans_fit` on train and on test, then `ps_from_labels` of the test labels against the training centroids' assignment of the test rows.
- `choose_k(gap_A: np.ndarray, gap_B: np.ndarray, directions: Sequence[tuple[np.ndarray, np.ndarray]], k_max: int, *, seed: int = 0) -> dict` with keys `k`, `k_A`, `k_B`, `ps` (one value per direction at the chosen k), `gap_A` and `gap_B` (the curves).
- `results/preflight.json`; `preflight` exits 1 if any case fails.

- [ ] **Step 1: Write the failing tests**

```python
def test_gap_cluster_free_elongated():
    X = np.random.default_rng(1).normal(size=(4000, 20)) * np.sqrt(1 / np.arange(1, 21))
    assert domains.gap_choice(X, 6)["k"] == 1
def test_gap_skewed_unimodal(): ...        # lognormal first axis: k == 1
def test_gap_two_and_three_clusters(): ... # separations as in design step 10: k == 2 and k == 3
def test_prediction_strength_separated(): ...  # 3 far-apart blobs, train/test split: PS(3) == 1.0
def test_ps_from_labels_hand_example():
    assert domains.ps_from_labels(np.array([0, 0, 0, 1, 1]), np.array([5, 5, 6, 7, 7])) == pytest.approx(1 / 3)
def test_choose_k_min_of_halves_and_ps(monkeypatch): ...  # halves 3 and 2 -> 2; PS 0.79 in one direction -> 1
```

- [ ] **Step 2: Run:** `.venv/bin/python -m pytest -q test_domains.py`. Expected: FAIL.
- [ ] **Step 3: Implement the four functions.** `gap_choice` draws its B Gaussians from `default_rng(seed)` with X's mean and `np.cov(X, rowvar=False)`.
- [ ] **Step 4: Run** the tests (pass), then `run_pilot.py preflight`. The pre-flight cases use matched Gaussians with PCA-like variances at the pilot's sizes (22,500 × 5; 5,000 and 22,500 × 20) and skewed cluster-free clouds at the same sizes (first axis lognormal with σ = 0.6), where k must be 1, and two- and three-cluster mixtures, where k must be 2 and 3. Expected: `results/preflight.json` has `"passed": true`. Record the gap gains in `progress.md`.
- [ ] **Step 5: Commit** (if agreed).

### Task 5: Bars of the pilot (the 2-hour run)

**Files:**
- Modify: `run_pilot.py` (subcommand `bars [--slides ...] [--workers 8]`), `test_common.py`.

**Interfaces:**
- **Consumes:** `common.read_slide`, `draw_centres`, `labelling`, `DELTA_PX`; `chromatic.ball_sixpacks`, `pair_counts`.
- **Produces `results/bars/{slide_id}.npz`, per centre:**
  - `centre_rows` (1500,), `centre_xy` (1500, 2);
  - `n_cells`, `class_counts` (1500, 4), `pair_counts` (1500, 10), `n_edges`, `seconds`.
- **Produces the bars in the same file:**
  - `bars` (N, 2): finite bars only;
  - `owner` (N,) int32 = `(centre·20 + labelling)·16 + block`, sorted, with `offsets` (1500·20·16 + 1,);
  - `codomain`, `codomain_owner` = `centre·2 + p`, for labelling 0 only.
- **Produces `results/bars/{slide_id}.json`:** wall time, workers, maximum ball size, maximum seconds per centre, peak RSS of the workers, package versions.

- [ ] **Step 1: Write the failing tests**
  - `test_owner_roundtrip`: decode(encode(c, l, b)) is the same triple for the extremes.
  - `test_resume_skips_complete_and_redoes_partial`: a fake two-slide run with `read_slide` patched to 300-cell synthetic slides and `N_CENTRES` patched to 20. A truncated `.npz` is redone; a complete one is left untouched, by its mtime.
- [ ] **Step 2: Run:** FAIL.
- [ ] **Step 3: Implement `bars`.**
  - Warm the numba cache in the parent first.
  - `ProcessPoolExecutor(8)` over chunks of 100 centres.
  - Write each slide atomically (`.tmp` then rename), and skip a slide whose npz has every key and whose `offsets[-1] == len(bars)`.
- [ ] **Step 4: Run one slide in the foreground.**
  - Expected: the median seconds per centre is within 2× of the spike's (0.77 s healthy, 1.38 s cirrhosis, serial).
  - Spot-check 3 centres: the codomain equals topocyte's alpha PH, and the class counts sum to `n_cells`.
- [ ] **Step 5: Run all 30 slides in the background,** logs in `logs/`. Expected: about 2 h. Record the per-slide maxima (Review Focus 4) in `progress.md`.
- [ ] **Step 6: Commit** the scripts and `results/bars/*.json` (the npz files stay ignored), if agreed.

### Task 6: Features (`features.py`)

**Files:** Create `features.py`, `test_features.py`.

**Interfaces:**
- **Consumes:** `results/bars/*.npz`, `results/pilot.csv`; `curves.*`; `tc.ph.fpca_from_gram`; the MFA and sign rule of the sibling's `common.pca` / `common.mfa`, copied.
- **Produces `results/features.npz`** (N = 45,000 rows, in pilot and centre order).
  - **Per centre:** `slide_index`, `half`, `centre_row`, `xy`;
  - `L1` (N, 5): log10 cells, then four class fractions;
  - `P` (N, 10): pair counts minus `pair_expectation`.
- **Per fitting set f ∈ {`A`, `B`, `all`}:**
  - `L2_obs_{f}`, `L2_ctl_{f}` (N, 20);
  - `blk_obs_{f}` (N, 16, 9);
  - `s0_{f}` (16,), `evr_{f}` (16, 9), `mfa_evr_{f}` (20,), `dropped_{f}`.
- **Diagnostics:** `zero_spread_nonzero` (16,).
- **Rules** (design steps 5–7):
  - **Observed curves:** `combine([lab0, lab1..lab19], [1, -1/19, ...])`, divided by `sqrt(spread2(lab1..19) + s0²)`.
  - **Control:** `combine([lab1, lab2..lab19], [1, -1/18, ...])` with its own spread and the observed s0.
  - **s0²** is the median of the positive s² over the fitting set's fit sample.
  - **fPCA and MFA:** fitted on the fit sample, 400 centres per slide drawn from `SeedSequence([SEED, slide_index, 0xF17])`; every centre is scored in each basis.

- [ ] **Step 1: Write the failing tests**
  - `test_level1_features`: counts [[50, 30, 15, 5]] give `[2.0, .5, .3, .15, .05]`.
  - `test_control_uses_labelling_1_against_2_to_19`: synthetic bars per labelling; the control curve equals lab1 − mean(lab2..19) by inner products.
  - `test_s0_median_positive_only`.
  - `test_zero_spread_nonzero_residual_is_finite`: Review Focus 1.
  - `test_mfa_projection_matches_fit`: the projection of the fit sample equals its PCA scores to 1e-12.
- [ ] **Step 2: Run:** FAIL.
- [ ] **Step 3: Implement `features.py`,** one block at a time to bound memory. Two gates in the run, which stops if either fails:
  - the unit-weight Gram of 200 observed lab0 curves equals topocyte's `betti_inner_product_matrix` (rel 1e-10);
  - the scores of each fit sample from `components` equal `fpca_from_gram`'s to 1e-9 (relative to the largest score).
- [ ] **Step 4: Run** the tests (pass) and `features.py`. Record in `progress.md` the dropped blocks, the s0 values, `zero_spread_nonzero`, the FVE and the gate errors.
- [ ] **Step 5: Commit** (if agreed).

### Task 7: Compartments, subdomains, control, guard (`domains.py` main)

**Files:** Modify `domains.py`, `test_domains.py`.

**Interfaces:**
- **Consumes:** `results/features.npz`; Task 4's functions.
- **Produces `results/domains.npz`:**
  - `compartment` (N,);
  - `subdomain` (N,): an index within the compartment, −1 where none;
  - `control_k` (n_compartments,);
  - the guard's bootstrap draws.
- **Produces `results/domains.json`:**
  - level 1: k, k_A, k_B, the gap curves and PS;
  - per compartment: its size per half, the level-2 choice and the control's choice;
  - `control_passed`;
  - the guard per compartment and direction (BA_stab, BA_pw, difference, interval, pass) and the verdict;
  - the descriptives: BA_dom (from `blk_obs` domain blocks), cross-half ARI and the contiguity ratios.
- **Functions:**
  - `match_labels(own: np.ndarray, final: np.ndarray) -> dict[int, int]`: Hungarian assignment on the contingency table, via `scipy.optimize.linear_sum_assignment`;
  - `guard(...) -> dict`;
  - `contiguity(xy, slide, compartment, subdomain, k_nn=5) -> dict`: the observed share against the exact hypergeometric expectation among same-compartment neighbours.

- [ ] **Step 1: Write the failing tests**
  - `test_match_labels_hungarian`: permuted labels map back.
  - `test_guard_detects_pairwise_determined`: subdomains as a function of P, so no pass.
  - `test_guard_passes_when_pairwise_uninformative`: P is pure noise and the subdomains are clear in L2, so a pass.
  - `test_bootstrap_resamples_slides`: each draw holds whole slides.
  - `test_contiguity_expectation_exhaustive`: on a 6-centre toy, the expectation equals the mean over all reassignments.
  - `test_small_compartment_gets_no_subdomains`: 499 centres in one half give k = 1 and `subdomain == -1`.
- [ ] **Step 2: Run:** FAIL.
- [ ] **Step 3: Implement the main.**
  - **Level 1:** z-score L1 per fitting set, then `choose_k` with k_max 10, then the final `kmeans_fit` on all centres. These are composition strata (Global Constraints).
  - **Level 2:** for each compartment, `choose_k` on `L2_obs_A[A]` and `L2_obs_B[B]`. The directions are (`L2_obs_A[A]`, `L2_obs_A[B]`) and (`L2_obs_B[B]`, `L2_obs_B[A]`), with k_max 6. Final fit on `L2_obs_all`.
  - **Control:** the same on `L2_ctl_*`.
  - **Guard:** only if `control_passed`.
- [ ] **Step 4: Run:** the tests pass; then `domains.py` prints the level-1 k, the k per compartment, the control gate and the verdict. Record them in `progress.md`.
- [ ] **Step 5: Commit** (if agreed).

### Task 8: Verifier and report

**Files:**
- Create: `verify_results.py`, `build_report.py`.
- Modify: `README.md` (run table, topocyte commit, venv recipe, outputs), `progress.md`, `../README.md` (Studies row).

- [ ] **Step 1: Write `verify_results.py`** with the design's independent checks. It prints `ALL CHECKS PASSED`.
  - **Six-packs:** the port against chalc 15.0.1 on 200 sampled real balls (faces to dimension 3 and 2), and the three Betti identities on them.
  - **Pair counts:** the expectation against 10,000 shuffles on 20 balls.
  - **Inner products:** against topocyte for unit weights.
  - **Spread:** s² by brute force on 20 balls, and s0 per block.
  - **Decompositions:** fPCA against `fpca_from_gram`; MFA by eigendecomposition.
  - **Clustering and verdict:** k-means, gap (from the saved seeds), PS and balanced accuracies recomputed; the verdict recomputed.
  - **Arrays:** every saved array finite.
- [ ] **Step 2: Run it:** `ALL CHECKS PASSED`.
- [ ] **Step 3: Write `build_report.py`.** Sections 1–7 and 9 of the design; section 8 reads "pending Simon's review of the H&E atlas". Use the sibling dataviz palette and PNGs quantized to 128 colours, and assert every number in the prose. The per-subdomain mean residual curves are exact means of the scaled residual curves (`curves.combine` with weights 1/n), drawn with `drawstyle="steps-post"`.
- [ ] **Step 4: Build and look:** render `report.html`, open it in the browser pane, and check overflow and legibility.
- [ ] **Step 5: Commit** (if agreed; the report stays out of git).

### Task 9: Whole-study review and hand-over

- [ ] **Step 1: Review.** Dispatch one fresh reviewer (model opus) on the folder: code, claims, and the rule for k, the control and the guard applied as written.
- [ ] **Step 2: Fix and rebuild.** Fix the confirmed findings, then rerun `verify_results.py` and `build_report.py`.
- [ ] **Step 3: Hand over.**
  - Update the memory note `topocyte-chromatic-domains-study.md`.
  - Send Simon the report.
  - Ask him to download the 30 SVS files listed in `results/pilot.csv` from the cluster, and to say where they are.

### Task 10: H&E atlas and viewer (after Simon's download)

**Files:**
- Create: `atlas.py`, `viewer/backend.py`, `viewer/static/index.html`, `viewer/static/app.js`, `viewer/static/overlay.js`, `test_atlas.py`.
- Modify: `requirements.txt` (openslide-python, openslide-bin, fastapi, uvicorn, pillow), `build_report.py` (section 8), `progress.md`.

**Interfaces:**
- **`atlas.py` writes two pages:**
  - `atlas_blind.html`: groups with anonymous IDs in random order (`default_rng(SEED)`), no compartment or subdomain names, no summaries;
  - `atlas_reveal.html`: the IDs with their compartment and subdomain, composition profiles and mean residual curves.
- **Atlas contents:**
  - 12 centres per group, at most 3 per slide, seeded;
  - crops of 810 × 810 px (400 µm) at level 0, centred on the centre;
  - the ball's outline at radius `DELTA_PX`;
  - cells as dots coloured by class, with a toggle.
- **`viewer/`:** adapted from `tissuegeometry/web/cosine_viewer`, keeping the DeepZoom tile serving and the WebGL overlay, with no tissuegraphs or Rust imports. Endpoints:
  - `/tiles/{level}/{x}/{y}` and `/slide_info`;
  - `/cells`: a binary blob `[N u32][N*2 f32][N i32 class]`;
  - `/centres`: the same blob with compartment and subdomain;
  - `/ball/{i}`: the members of centre i's ball.

- [ ] **Step 1: Write the failing tests** in `test_atlas.py`:
  - `test_crop_geometry`: the box is centred, 810 px, and the outline radius is 303.52 px;
  - `test_blind_page_hides_labels`: no compartment or subdomain name or ID mapping appears in `atlas_blind.html`.
- [ ] **Step 2: Implement** `atlas.py` and the viewer. Run the tests (pass); open both pages and one slide in the viewer in the browser pane and look.
- [ ] **Step 3: Blind pass.** Simon describes each blind group in his words; record them verbatim in `progress.md`. Then the reveal.
- [ ] **Step 4: Report section 8.** Write it from his words, rebuild, and rerun `verify_results.py`.
- [ ] **Step 5: Commit** (if agreed).
