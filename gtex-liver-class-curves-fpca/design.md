# Class-density curves: does their fPCA beat density and composition? (design)

## Intent

Simon, 2026-11-23: "lets design a experiment that performs functional pca on these curves. first computing the mean
curve per cohort, filtering cohort outliers, fpca on inner product spaces as with betti curves normalized and
unnormalized, classification." Then: "cohort means pathology group. i want to know if it beats D+C."

**The curves.** For one slide and one of the four cell classes (epithelial, stromal, immune, other), `n(t)` is the
number of the class's cells whose 10-nearest-neighbour density is at or above `t`, for every density `t` in cells
per mm². The liver cell viewer draws them (its stacked view divides them by the slide's cell count).

**Why.** The cell-type study ([gtex-liver-celltype-slices](../gtex-liver-celltype-slices/README.md)) found that
`D+C` is the representation to beat. `D+C` is 51 features: density summaries, and the cells of each class at each
density decile. On the study's 118-slide holdout it reached macro AUC 0.774, and no topology beat it. The class
curves carry the same information continuously:
- at the ten deciles, they are `C`'s counts;
- their sum over the classes gives `D`'s kept shares.

The question is whether a functional PCA of the whole curves separates the pathology groups better than `D+C`'s 51
features.

## Rulings (Simon, 2026-11-23)

- **Cohort** means pathology group: the eight groups of the pathology study's `pathology.csv` (healthy, congestion,
  steatosis, inflammation/hepatitis, fibrosis, cirrhosis, necrosis, unclassified).
- **The question:** does the curves' fPCA beat `D+C`?
- **Outlier filtering** happens only on the training slides, inside each fold. The test slides are always scored,
  and `D+C` is trained on the same cleaned training slides.
- **Components:** best against best decides. The number of MFA components of the curves (4 to 51) is chosen by
  inner cross-validation. Four components are reported beside it.
- **Computation:** each class's exact Gram matrix over all 594 slides is computed once. Every fold's fPCA is then
  linear algebra on its submatrices.

## Goals, non-goals, anti-goals

- **Goals:**
  - a verdict, fixed before any result, on whether the curves' fPCA beats `D+C` on the development slides and on
    the holdout;
  - each group's mean curves and the slides flagged as outliers, as a description.
- **Non-goals:**
  - topology of cell positions (no persistent homology here);
  - improving `D+C`;
  - other curve views (shares among the cells kept, per-class kept shares) as inputs;
  - tuning the classifier;
  - neighbour purity.
- **Anti-goals:**
  - anything fitted with a test slide's label: the filter, the group means, the fPCA, the MFA and the choice of `k`
    all see training slides only;
  - a win that comes from dropping test slides;
  - evaluating curves on a grid (exact bar-endpoint arithmetic only, Simon's standing rule);
  - changing the comparison after seeing results.

## Constraints

- **Read only:**
  - the cohort (`~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad`);
  - the pathology study's `density.npz` (each cell's density `rho`, the deciles, the slides' rows) and
    `pathology.csv`;
  - the cell-type study's `results/composition.npz` (cells per class and slice), `results/holdout.csv`, and its
    code.
- **topocyte:** the local `main` checkout at a pinned commit (`18e3aa5` at the time of writing), unchanged. Its
  functions are used:
  - `BettiCurve` for the curves;
  - `betti_inner_product_matrix` for the exact Gram matrices;
  - `fpca_from_gram` for every fit.
- **Readout code:** the cell-type study's (`common.baseline_features`, `macro_auc`, `make_folds`,
  `stratified_bootstrap` and its logistic regression), reused rather than rewritten, so that `D+C` and the readout
  are identical to that study's.
- **Machine:** M4, 10 cores, 16 GB; about 14 GB of free disk.

## Approaches considered

1. **Where the outliers are filtered.**
   - **A (chosen):** on the training slides inside each fold.
   - B: once on all slides before everything. Rejected: it removes the hard slides by their labels, so both AUCs
     rise and the answer covers only typical slides.
   - C: descriptive only. Kept as the descriptive part.
2. **How many components.**
   - A: chosen by inner cross-validation (fair, best against best).
   - B: four (compact, the cell-type study's headline rule).
   - **C (chosen):** A decides, and B is reported beside it.
3. **How the fold-wise fPCA is computed.**
   - **A (chosen):** each class's exact Gram matrix once, then `fpca_from_gram` on the training submatrix and an exact
     projection of the other slides.
   - B: refit topocyte's `FPCA` estimator in every fold. Rejected: the same numbers, at days of compute.
   - C: fit the fPCA once on all slides. Rejected: the fPCA would see the outliers, against "filter, then fPCA".

## Pipeline

1. **Curves.**
   - For each of the 594 slides and each class, take the class's cells' densities `rho` (from `density.npz`, float64)
     and the class map of the cell-type study.
   - The curve is the reduced Betti curve of the bars `(0, rho_i)`: topocyte's `BettiCurve`, dimension 0.
     - A bar is alive on `[0, rho_i)`, so the curve counts `rho > t`, where the viewer counts `rho >= t`. The two
       differ only at the points `rho_i`, which no integral sees.
   - **Gate:** the class counts at the ten deciles, computed directly with `rho >= d_j`, equal the cell-type study's
     `kept_by_group`.
2. **Gram matrices.**
   - For each class, `G = betti_inner_product_matrix(curves, n_jobs=-1)` over all 594 slides: exactly,
     `G[f, g] = sum_i sum_j min(rho_i, rho_j)`. Saved (four 594 × 594 matrices).
   - The cosine model uses the same `G` (`fpca_from_gram(..., normalization="cosine")`).
   - **Gate:** a sample of entries recomputed by a direct sum of minima over sorted densities.
   - A timing spike on a few slide pairs comes first.
3. **Development slides and folds.**
   - The cell-type study's `holdout.csv`: 476 development slides, 118 holdout slides.
   - On the development slides, 20 repeats of stratified 5-fold over the eight groups: `make_folds(..., seed=1)`,
     the improvement phase's development folds.
4. **In every outer fold** (training slides `T`, test slides `S`):
   1. **Outlier filter on `T`.**
      - There are eight blocks: four classes × {raw, cosine}.
      - In each block, each training slide gets its squared distance to its group's mean curve over `T`, from the
        Gram matrix:
        - raw: `G_ff - 2 mean_{j in g} G_fj + mean_{j, k in g} G_jk`;
        - cosine: the same on `C = D^(-1/2) G D^(-1/2)`.
      - Each block's distances are divided by their median over `T`, and the eight are summed: one distance per
        slide.
      - A slide is an outlier when its distance exceeds its group's median plus 3 × 1.4826 × MAD. This applies to
        all eight groups, unclassified included.
      - `T'` is `T` without the outliers.
   2. **fPCA of each block on `T'`.**
      - `fpca_from_gram` on `G[T', T']`, normalization none or cosine; the first nine components (fewer if the rank
        is lower).
      - Every other slide (`S`, and the dropped ones) is projected exactly: its centred inner products with the `T'`
        curves times the fitted coefficients, the formula of `FPCA.transform`.
      - **Gate:** on one fold, the projection equals topocyte's `FPCA` fitted on the `T'` curves and its
        `transform`.
   3. **MFA on `T'`.**
      - Each block's scores are divided by the standard deviation of its first score over `T'`.
      - The eight blocks are concatenated (72 columns), centred on `T'`, and decomposed by SVD; up to 51 components
        are kept.
      - Signs are fixed so that each component's largest-magnitude loading is positive.
      - `S` is projected with `T'`'s means and scales.
   4. **Classifiers**, the cell-type study's logistic regression: multinomial L2, `C = 1`,
      `class_weight="balanced"`, lbfgs, `max_iter` 5000, features standardized on the training slides, all eight
      groups.
      - **Curves, primary:** `k` is chosen by inner stratified 5-fold cross-validation on `T'`.
        - Within each inner fold, steps 2–3 are refitted on the inner training slides; the outer filter is not
          redone.
        - `k` is taken from {4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 51}, capped at the components available.
        - The highest inner macro AUC wins; ties go to the smaller `k`.
        - Then the model is fitted on `T'` with that `k`, and `S` is predicted.
      - **Curves, secondary:** `k = 4`.
      - **`D+C`:** the 51 features of `baseline_features`, fitted on `T'`; `S` is predicted.
   5. Every development slide gets out-of-fold probabilities in every repeat, averaged over the 20 repeats.
5. **Readouts on the development slides.**
   - For each arm, the macro AUC over the seven conditions (unclassified unscored), from the repeat-averaged
     probabilities.
   - The paired difference, curves minus `D+C`, with its 95% interval: 4000 bootstrap draws of slides, stratified by
     group (`default_rng(12345)`).
   - The per-repeat spread and each condition's AUC are saved.
   - **Gate:** `D+C` without filtering reproduces the improvement phase's development macro AUC, 0.764.
6. **Holdout, scored once.**
   - The whole pipeline is fitted on the 476 development slides:
     - the filter;
     - the fPCA and MFA;
     - `k` by inner 5-fold cross-validation on the cleaned development slides.
   - It is applied to the 118 holdout slides. `D+C` is fitted on the same cleaned development slides.
   - Reported: each arm's macro AUC, and the paired difference with its interval (bootstrap of holdout slides,
     stratified). The groups of two and four slides make that interval too narrow, and the report says so.
   - The script refuses to score the holdout twice.
   - **Gate:** `D+C` without filtering reproduces the improvement phase's holdout macro AUC, 0.774.
7. **Decision rule, fixed before any result.**
   - **Beats `D+C`:** the development difference (curves at the chosen `k` minus `D+C`) has its 95% interval above
     0, and the holdout difference is positive.
   - **Not confirmed:** the development interval is above 0 but the holdout difference is not positive.
   - **Level with `D+C`:** the development interval contains 0.
   - **Below `D+C`:** the development interval lies below 0.
   - `k = 4`: reported beside, with no verdict.
8. **Descriptive.**
   - Each group's mean curve per class over density, as counts and divided by the slide's cell count.
   - How often each slide was dropped across the 100 outer folds, with its group, pathology notes and slide ID.
   - The fractions of variance per block on all development slides.
   - The distribution of the chosen `k`.
   - Each condition's AUC per arm.
9. **Report and verification**, as in the sibling studies.
   - `report.html`, with every number in the prose computed and asserted.
   - `verify_results.py`, independent of the main path:
     - the Gram entries by direct sums;
     - the decile counts against `composition.npz`;
     - one fold's projection against topocyte's `FPCA.transform`;
     - the MFA by eigendecomposition;
     - the AUCs by the Mann–Whitney formula from the saved probabilities;
     - the folds against the cell-type study's development folds;
     - the holdout scored once.
   - `progress.md` is the ledger.

## Calls made after the builder check (Claude, 2026-11-23)

The builder check found nothing that needs Simon. Its small points are settled here:

- **Inner cross-validation:** one stratified 5-fold split of `T'`, seeded per outer fold
  (`random_state = 1000 + 5 × repeat + fold`). The inner macro AUC comes from the out-of-fold probabilities pooled
  over the inner folds.
- **Group mean:** includes the slide itself, as the distance formula has it. Leaving it out only rescales each
  group's distances by a constant, which the median-and-MAD rule ignores.
- **Holdout interval:** 4000 bootstrap draws of the holdout slides, stratified by group, `default_rng(12345)`.
- **topocyte:** pinned at `18e3aa5`.
- **Descriptive group means:** use the development slides until the holdout is scored.
- **One-fold projection gate:** topocyte's `FPCA` with `eigen_solver="dense"`, so that the gate compares exact fits.
  Its `"auto"` would pick the matrix-free solver on about 380 curves.
- **Unfiltered `D+C` gate:** compared on the out-of-fold probabilities saved in the cell-type study's
  `improve_dev.npz`, not only on the AUC.

## Left to the builder

- **Engineering:** code layout, file names, caching, parallelism and the timing spike; the projection's
  implementation (exact, from the Gram matrices).
- **Gates and checks:** the order of the gates; the tolerances of the checks.
- **Report:** the figures (dataviz palette, class colours as in the viewer).
