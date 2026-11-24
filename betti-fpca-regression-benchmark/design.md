# Betti-curve fPCA regression against the persistence-diagram kernel of Aggarwal, Nichols and Turner (design)

## Intent

Simon, 2026-11-24, with the paper's PDF: "Benchmark fPCA regression using inner product space of betti curves against
their reported performances". Then: "whether Betti fPCA regression beats their kernel; download the repo. report in
topocyte analyses." Then he chose approach B below.

**The paper.** S. Aggarwal, J. Nichols, K. Turner, *Kernel regression using persistence diagrams*,
arXiv:2611.25295v1 (18 Nov 2026). Their estimator, `PD_kernel`, is Nadaraya–Watson regression with a Gaussian kernel
on the distance W2(H0) + W2(H1) between persistence diagrams, with the bandwidth chosen by leave-one-out (LOO)
cross-validation. They argue that working in diagram space, without a Hilbert embedding, predicts better, and they
test it only against persistence images (`PI_kernel`). Their code and data are public:
github.com/saksham-aggarwal-2002/kernel-regression-persistence-diagrams, commit
`fe7d652eff478ffe6081c7e6765458651ad3cd74` (2026-11-15), called *upstream* below.

**Why.** topocyte's exact Betti-curve fPCA is a Hilbert embedding too. The question is whether functional principal
component regression on it (FPCR: a linear regression of the target on the fPCA scores) beats `PD_kernel` on the
paper's three tasks, with both methods seeing the same diagrams, training sets and test sets.

## Rulings (Simon, 2026-11-24)

- **The question:** does Betti fPCA regression beat their kernel?
- **Data:** download upstream (done: a scratch clone at `fe7d652`, 203 MB with its history).
- **Report:** in topocyte-analyses, as a study folder.
- **Approach B** (below): the FPCR head-to-head, as laid out on 2026-11-24 (including the 100 fluid splits), plus an
  exploratory arm that runs `PD_kernel`'s estimator on Betti-curve distances.

## Goals, non-goals, anti-goals

- **Goals:**
  - a verdict per task, fixed before any result of the two Betti arms, on whether FPCR beats `PD_kernel`;
  - the paper's three tables, with `PD_kernel` as reported, `PD_kernel` rerun, FPCR and the exploratory arm side by
    side;
  - from the exploratory arm, whether a gap between FPCR and `PD_kernel` comes from the representation (Betti
    curves against diagrams under W2) or from the estimator (linear on scores against local averaging).
- **Non-goals:**
  - any change to topocyte: no periodic cubical complex, no regression API;
  - the geodesic-distance variant of the curvature task;
  - rerunning `PI_kernel` or the ten Turkeš et al. baselines (their reported numbers only);
  - the raw fluid snapshots (about 4 GB from ETH; upstream's diagrams suffice);
  - other regressors on the scores (ridge, nonlinear), cosine normalization, tuning beyond the grids below.
- **Anti-goals:**
  - arms that see different diagrams, training sets or test sets;
  - a test target in any choice: the bandwidth, the numbers of components, the fluid shift;
  - a "beats" read off table cells without the paired interval;
  - running any upstream code, or unpickling upstream files without the whitelist (see Safety);
  - evaluating a curve on a grid (exact bar-endpoint arithmetic only, Simon's standing rule);
  - changing the comparison after seeing a result of the Betti arms.

## Constraints

- **Read only:** upstream, cloned at the pinned commit into the study's `upstream/` (ignored by git).
- **topocyte:** the local `main` checkout at a pinned commit (`18e3aa5` at the time of writing), unchanged. Used:
  `compute_persistent_homology` (Rips), `BettiCurve`, `FPCA`, `betti_inner_product_matrix`, `fpca_from_gram`,
  `betti_l2_distance`.
- **Environment:** topocyte's main venv (Python 3.14, numpy 2.5.1, scipy 1.18.0, pandas 3.0.5, ripser 0.6.15,
  scikit-learn 1.9.0). No new dependency. POT, upstream's Wasserstein backend through gudhi, is not installed, so W2
  is computed with scipy (below).
- **Machine:** M4, 10 cores, 16 GB.

## The three tasks

All three as in the paper and upstream; every number below was checked on the clone on 2026-11-24.

1. **Interpolation (paper Table 1).** Clouds of 20 points
   `(cos θ_i + U, (1 − t) cos θ_i + t sin θ_i + U)`, with θ = `linspace(0, 2π, 20)`, U ~ U(−0.2, 0.2) independent
   per coordinate and t ~ U(0, 1); the target is t. For each training size n ∈ {10, 20, …, 100} there are 10 runs,
   each with its own training set and 50 test clouds: 100 independent datasets, upstream's
   `interpolation/data/interpolation_dataset.pkl`. Regenerating them with `np.random.seed(0)` and upstream's recipe
   gives identical arrays. Metric: R² on the 50 test clouds (`1 − SSE/SST`, SST about the test mean); the paper
   reports its mean and SD over the 10 runs of each size. The paper's text swaps line and circle: in the code, t = 0
   is a line and t = 1 a circle.
2. **Curvature (paper Table 2).** Samples of 50 points from the unit disk on a surface of constant curvature K
   (Turkeš, Montúfar, Otter 2022). Training: 50 clouds, K = `linspace(−2, 2, 50)`; test: 50 clouds, K ~ U(−2, 2).
   One dataset, generated without a seed, upstream's `curvature/data/` (clouds 0–49 train, 50–99 test). Metric: MSE
   on the 50 test clouds. The input of `PD_kernel`, and so of every arm here, is the Euclidean Rips filtration of the
   stored coordinates: 3D Cartesian coordinates on the sphere for K > 0 (53 clouds), 2D Poincaré-disk coordinates
   scaled by 1/√−K for K < 0 (47 clouds). The Turkeš baselines in Table 2 used geodesic distance matrices instead;
   the report says so.
3. **Fluid (paper Table 3).** 400 simulations of upstream's `SIM_IDS` (from its formula in `fluid/distances.py`) of
   the 2D unsteady flow ensemble of Jakob, Gross and Günther (2021); simulation i has
   Re_i = 2^((4/3)(1 + i/1000)), and the target is ln Re. A sample's diagram is the sublevel filtration of the
   vorticity on the periodic cubical complex (GUDHI), H0 and H1, with the essential bars kept: one in H0 and two in
   H1 in every diagram. Snapshot times t ∈ {1, 2.5, 5, 7.5, 9}. Split s: `perm = np.random.default_rng(s)
   .permutation(400)`, test = `perm[:100]`, training = the first n of `perm[100:]`, n ∈ {25, 50, 100, 200, 300}. The
   paper runs s = 0, …, 4. Metric: R². Upstream holds all 2,000 diagrams (3 to 6,493 bars each, median 1,535) and
   the W2 distance of every pair of simulations at the same time, H0 and H1 separately (399,000 pairs each).

## Approaches considered

- **A:** FPCR only, against `PD_kernel` rerun on the same data, with the decision rule below.
- **B (chosen):** A, plus an exploratory arm: `PD_kernel`'s own estimator on exact Betti-curve L² distances. If FPCR
  loses, it shows whether the Betti curves or the linear model is to blame. It stays outside the verdict.
- **Dropped:** kernel ridge regression on the Betti Gram matrix. It shrinks all components instead of truncating
  them, so it is not fPCA regression, and it leaves topocyte's `FPCA` unused.

## Pipeline

### Inputs

- **Clone** upstream at the pinned commit into `upstream/`, and refuse to use a clone at any other commit.
- **Load** its pickles only through the whitelist loader (Safety).
- **Diagrams:**
  - interpolation and curvature: `tc.ph.compute_persistent_homology(X, filtration="rips", max_dim=1)`, which runs
    ripser with its defaults. It gives the same birth–death pairs as upstream's `ripser.ripser(X)["dgms"]` on all
    10,600 clouds (10,500 interpolation, 100 curvature), compared as rows sorted by birth then death, on 2026-11-24;
    gate G5 repeats the check in the run. Computed once and used by all arms;
  - fluid: upstream's diagrams, columns Birth, Death, HomologyG.

### Arm K: `PD_kernel`, rerun

Reimplemented from upstream's `kernel_regression.py` and `fluid/nw_regression.py`, which are read, never run.

- **Distance** between two samples: W2(H0) + W2(H1). Each is the 2-Wasserstein distance with the Euclidean ground
  metric (gudhi's `order=2, internal_p=2`):
  - finite bars: an optimal matching of the two diagrams' finite points, each point matched to a point of the other
    diagram (cost: squared Euclidean distance) or to the diagonal (cost (d − b)²/2);
  - essential bars: matched only among themselves, by sorted birth (cost: squared difference of births). Unequal
    counts give infinity, which never happens here;
  - W2 = √(finite cost + essential cost).
- **Computing W2** exactly: `scipy.optimize.linear_sum_assignment` on the (n + m) × (n + m) augmented cost matrix.
  It agreed with upstream's cached fluid distances (POT) to 1.1e-15 relative on 40 random pairs at t = 5
  (2026-11-24). Fluid uses upstream's cached distances for all pairs; this W2 is only checked against them (G2).
- **Estimator,** on each training set:
  - bandwidth grid: `linspace(0.1 m, 2 m, 50)`, with m the median of the positive entries of the full symmetric
    training distance matrix;
  - LOO mean squared error for each h. The weights are `exp(−(d/h)²/2) / √(2π)`, without the sample itself; an
    all-zero weight sum predicts the mean of the other training targets;
  - h is the first minimizer;
  - prediction: the weighted mean of the training targets; an all-zero weight sum predicts the training mean.

### Arm F: Betti FPCR (primary)

- **Curves:** the reduced Betti curves, `tc.ph.BettiCurve`, of H0 and H1: finite bars only. topocyte's
  representation drops essential bars, so F does not see the fluid diagrams' three essential births, which W2 uses;
  the report says so.
- **Fluid shift:** topocyte's Betti inner product integrates over [0, ∞) only (`metrics._levels`: a bar born below 0
  counts nothing below 0). Vorticity values are negative (lowest birth −25.09), so every fluid bar is shifted by
  s = 26, the ceiling of minus the lowest birth over all 2,000 diagrams (no targets involved). With nothing below 0
  the inner product is the full-line one, which a common shift leaves unchanged: gate G4. Rips values are never
  negative, so no shift there.
- **fPCA:** per dimension, topocyte's fPCA fitted on the training curves only, normalization `"none"` (raw curves:
  their mass carries information), other parameters at their defaults. Test curves get their exact scores on the
  training components. Two routes, either allowed anywhere:
  - direct: `tc.ph.FPCA(dim=h).fit(train).transform(test)`;
  - Gram (needed for speed on the 100 fluid splits): per task, the exact Gram matrix of all curves of a dataset
    (interpolation), the dataset (curvature) or a time (fluid), computed once with `betti_inner_product_matrix`
    (about 11 s for H0 and 6.5 s for H1 per fluid time with `n_jobs=-1`); `fpca_from_gram` on the training block;
    the test scores by exact projection of the test rows. Gate G3.
- **Regression:** OLS with an intercept of the target on the first K0 H0 scores and the first K1 H1 scores.
  - (K0, K1) is chosen on the training set by LOO mean squared error, in closed form: e_i / (1 − h_ii), with e the
    residuals and h_ii the leverages of the design matrix [1, scores].
  - Eligible: 0 ≤ K_d ≤ min(20, components fitted in dimension d); K0 + K1 ≤ n − 2; every h_ii < 1 − 1e-10.
  - (0, 0) is the training mean.
  - Ties go to the smallest K0 + K1, then the smallest K0.
  - The fPCA sees every training curve before the LOO, so the LOO refits only the regression, as `PD_kernel`'s LOO
    refits only its weights.

### Arm B: Betti Nadaraya–Watson (exploratory)

- Distance: L2(H0) + L2(H1) between the reduced Betti curves, each exact (`tc.ph.betti_l2_distance`, on the shifted
  curves for fluid), summed as arm K sums its two W2s.
- Everything else as arm K.

### Runs

- **Interpolation and curvature:** every arm on every dataset of the paper.
- **Fluid:** every arm on splits s = 0, …, 99 by the paper's recipe, for every time and training size. Splits 0–4
  are the paper's; all 100 feed the fluid readout, because a paired interval over five splits is weak, and
  upstream's distances cover every pair.

### Gates, before any readout

| gate | check | tolerance |
|---|---|---|
| G1 | arm K reproduces upstream's results: interpolation's mean and SD of R² per size (`results_interpolation.csv`), curvature's MSE (`results_curvature.csv`), fluid's R² of every split 0–4 (`results_fluid_splits.csv`) | 1e-9 absolute |
| G2 | the scipy W2 against upstream's cached fluid distances, on at least 100 random same-time pairs, at least 10 of them with more than 1,000 finite bars in that dimension | 1e-12 relative |
| G3 | if the Gram route is used: the projected test scores against `FPCA.fit(...).transform(...)`, on at least 3 cells per task | 1e-9 relative to the largest score |
| G4 | the fluid Gram matrices at shift s and at s + 10 | 1e-12 relative |
| G5 | inputs: interpolation equals its seed-0 regeneration; every cloud's Rips diagrams equal `ripser.ripser`'s as birth–death pairs (rows sorted by birth then death, never births and deaths sorted separately); every one of the 2,000 fluid diagrams present | exact |

If G1 or G2 fails, the run stops before any readout, and the failure goes to Simon. G2 can fail because of upstream
rather than this study: gudhi calls POT's `emd2` with an iteration cap, so a cached distance from an early stop would
lie above the exact W2. If G3, G4 or G5 fails, that is a bug to fix before going on.

### Decision rule, fixed in advance

**Primary: arm F against arm K, per task.** Δ is the paired difference, signed so that Δ > 0 favours F.

- **Interpolation:** Δ = R²(F) − R²(K) per dataset (100). The readout is mean Δ, with a 95% percentile bootstrap
  interval that resamples the 10 runs within each training size (10,000 resamples, `default_rng(0)`).
- **Curvature:** Δ = MSE(K) − MSE(F) on the 50 test clouds. Its 95% interval comes from a bootstrap over the test
  clouds, resampling their paired squared errors (10,000 resamples, `default_rng(0)`). It covers test sampling only,
  since there is one training set; the report says so.
- **Fluid:** Δ = R²(F) − R²(K) per time, split and size. The readout is the mean over 5 times × 100 splits × 5
  sizes, with a 95% interval from a bootstrap over the 100 splits (10,000 resamples, `default_rng(0)`). The splits
  all draw from the same 400 simulations, so the interval bounds the mean Δ over random splits of these simulations,
  not over new simulations; the report says so.

**Verdict per task:**
- **beats `PD_kernel`** if the interval lies above 0;
- **below `PD_kernel`** if it lies below 0;
- **level** otherwise: the readout cannot tell the two apart, which does not mean they are equal.

The headline gives the three verdicts, with no pooled verdict.

**Descriptive** (no verdict):
- the paper's tables with the reported values, arm K rerun, F and B;
- Δ by size and time, with the same interval recipes;
- the fluid readout on splits 0–4 alone;
- B against K (same estimator, other representation) and F against B (same representation, other estimator), with
  the same interval recipes;
- the chosen (K0, K1);
- runtimes;
- F against the reported `PI_kernel` and Turkeš baselines.

## Safety of upstream data

Upstream is data only:
- cloned at the pinned commit into `upstream/`;
- never imported, never executed, never on `sys.path`; scripts run from the study folder.

Its pickles are read by an unpickler whose `find_class` resolves exactly these globals and raises for any other:
`numpy._core.multiarray._reconstruct`, `numpy._core.multiarray.scalar`, `numpy.ndarray`, `numpy.dtype`,
`builtins.slice`, `pandas._libs.internals._unpickle_block`, `pandas.core.frame.DataFrame`,
`pandas.core.indexes.base.Index`, `pandas.core.indexes.base._new_Index`, `pandas.core.indexes.range.RangeIndex`,
`pandas.core.internals.managers.BlockManager`.

A static scan of all 2,068 upstream pickles on 2026-11-24 found exactly these. It used inert stand-ins for every
global, so nothing ran. With them, pandas 3.0.5 reads upstream's pandas 2.3.3 frames.

## Layout and report

The layout follows topocyte-analyses' conventions:
- `README.md`, `design.md` (this), `plan.md`, `progress.md`;
- the scripts, `verify_results.py`, `build_report.py`, `test_study.py`;
- `results/*.json` tracked, `*.npz` ignored;
- `upstream/` ignored by a line in the top-level `.gitignore`;
- an import guard: topocyte must be a clean checkout at the pinned commit, named first on `PYTHONPATH`.

`report.html` holds:
- the verdict per task;
- the three tables in the paper's format;
- R² against training size per task (as the paper's Figures 6 and 9), every arm drawn;
- Δ with intervals;
- the exploratory comparisons;
- the chosen (K0, K1);
- the caveats: the curvature inputs, the dropped essential bars, one training set for curvature, 100 fluid splits.

## Calls made after the builder check (Claude, 2026-11-24)

The check found nothing important left. It raised three points, settled here:

- **What the fluid interval covers:** random splits of the same 400 simulations, not new simulations. The recipe is
  kept, and the caveat goes in the decision rule and the report.
- **G2 failing because of upstream:** stop and report it as for G1, rather than recompute upstream's 798,000
  distances.
- **G5 compares whole pairs.** The feasibility check of 2026-11-24 had sorted births and deaths separately, which
  cannot see a re-pairing. Rerun on rows sorted as pairs, it found all 10,600 clouds identical, and G5 compares rows.

## Left to the builder

- File split and names; parallelism;
- the report's styling (as the other studies' `build_report.py`);
- how G2 samples its pairs, within the stated minimums;
- whether the clone in the session scratchpad is copied or cloned afresh.
