# Liver pathology across density slices, per cell group: design

2026-11-19. Simon asked for the Betti-curve liver pathology study
([`gtex-liver-pathology-slices`](../gtex-liver-pathology-slices/README.md))
to be repeated with the cells split by HistoPLUS cell type, worked out in a
brainstorm and settled by his answers to four batched questions at 21:30. He
is away overnight. After the report, the study goes on to an improvement
phase that Claude runs on its own; a report is due in the morning.

## Intent

Does splitting the cells into cell groups make the liver conditions easier to
tell apart, both visually (the score scatters and UMAPs) and measured? The
Betti-curve study found that the leading combined component is tissue density
and that the cirrhosis gap along PC2 is the densest-decile share of cells.
This study asks whether each group's own topology, taken at the same tissue
density levels, adds separation that all-cell topology does not have.

## Rulings (Simon)

| Question | Ruling |
|---|---|
| Density | Density among all cells: the sibling study's 10-NN density `rho` (cells/mm², 0.4942 µm/px assumed) and its ten pooled-decile thresholds. At slice `s`, a group's point cloud is the slide's cells of that group with `rho >= threshold_s`. |
| Cell groups | epithelial = Epithelial + Cancer cell; stromal = Fibroblasts + Muscle Cell + Endothelial Cell + Minor Stromal Cell; immune = Lymphocytes + Neutrophils + Macrophages + Plasmocytes + Eosinophils; other = Apoptotic Body + Red blood cell + Mitotic Figures. |
| Readouts | Both: cross-validated logistic regression and neighbour purity. No MLP. |
| Headline (fixed in advance) | Everything (all cells + the four groups) against all-cell topology, at the same number of components. Secondary, also fixed in advance: per-group topology on top of density and composition summaries. |
| Honesty of the improvement phase | A seeded, stratified fifth of the slides (`results/holdout.csv`, 118 slides, committed before any result) is held out for the improvement phase only. The main experiment uses all 594 slides. |
| Commits | Local commits in topocyte-analyses as each piece is verified; Simon pushes. |

## Goals, non-goals, anti-goals

- **Goals.**
  - Per-group Betti-curve fPCA (H0/H1, raw/cosine, nine components) at the ten slices.
  - Combined spaces for every choice of point sets and models.
  - Both readouts for every space, a baseline ladder with density and composition, and the two comparisons fixed in advance.
  - An HTML report whose numbers are asserted, plus a verifier.
  - Then the improvement phase.
- **Non-goals.**
  - A classifier as a product.
  - Neural models.
  - Cross-type constructions that need new dependencies (chromatic alpha complexes).
  - Any change to topocyte itself.
- **Anti-goals.**
  - A gain that is only cell counts dressed up as topology. A group's raw H0 Betti curve starts at that group's cell count, so composition must be a baseline.
  - Readouts tuned to the data.
  - Grids, quadrature or changes of measure: exact bar-endpoint arithmetic only.

## Constraints

- **Machine.** 16 GB RAM, 10 cores, about 32 GB of free disk. No persistence diagrams are saved.
- **Data.** The cohort `~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad` is opened read only (h5py mode "r"):
  - `obs/slide_id` and `obs/cell_type` as categorical codes;
  - `obsm/spatial` coordinates.
- **topocyte.** Commit `bd1d5c0` through `PYTHONPATH`, as in the sibling studies.
- **Units.** Coordinates are pixels and alpha filtration values are squared pixels, as in the sibling study.

## Approaches considered

1. **Per-group PH at the all-cell density slices, fPCA per block, MFA combinations (chosen).** This is Simon's pipeline, and it reuses the sibling study's code path and all-cell scores.
2. **Class-resolved Betti curves of the all-cell filtration** (each bar attributed to the cell types of its simplices). This was an earlier open question. It needs attribution code that topocyte has only for local log-odds, and Simon asked for per-group PH.
3. **Chromatic alpha complexes** (inter-group topology). This needs chalc, a new dependency whose integration rulings are pending. It is out of scope.

## Pipeline

1. **Cell groups and composition.**
   - Read the 14 `cell_type` categories and map them to the four groups. A category absent from the map stops the run.
   - Count the cells per slide, group and slice (`kept_by_group[s, slide, g]`). Each slice's group counts must sum to the sibling study's `kept_per_slide`.
2. **Per-group persistent homology and fPCA.**
   - For each slice `s` (9 down to 0) and group `g`: alpha persistent homology of each slide's group cloud (`tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)`, float64, in row order), in threads. Use five workers when the slice's group holds more than 70 million cells, otherwise all cores, as the sibling study did.
   - Then fit the four `tc.ph.FPCA` models (`dim` 0/1, `normalization` none/cosine, `n_components=9`, `n_jobs=-1`, solver auto, which is arpack here), exactly as the sibling `run_pathology_slices.py` does.
   - One group's diagrams are held at a time.
   - Saved per slice: scores, fractions of variance, bar counts, kept counts per group, runtimes.
   - A slide whose group cloud makes a model undefined (fewer than 3 cells, or a zero curve under cosine) stops the run. The cell-type recon found at least 24 cells per group and slide even at slice 9.
3. **All-cell blocks.**
   - Reused from the sibling study's `slice_<k>.npz`.
   - Gate: this study's code path recomputes the all-cell models at slices 9 and 8. Every component must match the sibling scores (absolute correlation at least 0.999999), and the fractions of variance must agree to 1e-6.
4. **Blocks.** 5 point sets × 4 models × 10 slices = 200 blocks of nine scores each.
5. **Spaces.**
   - **Single blocks:** 200.
   - **MFA spaces:** every non-empty subset of the point sets {all, epithelial, stromal, immune, other} (31) × every non-empty subset of the models {h0_none, h1_none, h0_cosine, h1_cosine} (15), each over all ten slices: 465 spaces.
     - MFA as in the sibling study: each block is divided by the square root of the variance of its first score, then PCA (SVD) of the centred concatenation.
     - Signs are fixed so that the largest-magnitude entry of each left singular vector is positive.
     - Each space keeps 20 components.
     - Named spaces:
       - `all-cell` = {all} × all models (the sibling's `all` space);
       - `groups` = the four groups × all models;
       - `everything` = all five point sets × all models.
   - **Baselines** (standardized features, then PCA):
     - `D`, the density summaries: log10 median density, log10 cell count, and the decile shares `kept[s]/kept[0]` for s = 1..9 (11 features).
     - `C`, the composition: `log10(1 + kept_by_group[s, g])` (40 features).
     - `D+C` (51 features).
6. **Readouts** for every space on its leading `k` components: `k` = 4 and 20 (single blocks: 4 and 9; baselines: 4 and all columns).
   - **Logistic regression.**
     - Features standardized within each training fold.
     - Multinomial L2 logistic regression (`C = 1`, `class_weight="balanced"`, lbfgs, `max_iter` 5000) on all eight groups; unclassified is its own class.
     - 20 repeats of stratified 5-fold (seeded, the same folds for every space) give out-of-fold probabilities.
     - **Macro AUC:** the one-vs-rest AUC of each of the seven conditions among the condition slides (unclassified is not scored), computed from each slide's out-of-fold probabilities averaged over the 20 repeats, then averaged over the seven. The mean and spread of the per-repeat macro AUCs are also saved.
     - **Each condition against healthy:** the AUC of `log p_c - log p_healthy` between the slides of `c` and of healthy.
   - **Neighbour purity.**
     - In the space's leading `k` components (scores as fitted, not standardized), each condition slide's 15 nearest neighbours, by Euclidean distance among all 594 slides, itself excluded.
     - The share of neighbours in its own group, divided by the chance share `(n_g - 1)/(N - 1)`.
     - Averaged per condition, then over the seven conditions (macro).
   - **Intervals.** 4000 bootstrap draws of slides, stratified by group (`default_rng(12345)`), paired whenever two spaces are compared:
     - logistic regression: on the repeat-averaged out-of-fold probabilities;
     - purity: over the query slides, with the neighbour graph fixed.
7. **Comparisons fixed in advance** (this rule is written before any per-group result exists).
   - **Primary.** `everything` against `all-cell` at `k = 4`: the difference in macro AUC and in macro purity, each with its 95% paired interval.
     - **Enhanced:** both intervals lie above 0.
     - **Mixed:** exactly one does.
     - **Not enhanced:** neither does. The report says so if either lies below 0.
   - **Secondary (composition guard).** Nested logistic regression on the 51 standardized `D+C` features against the same plus the first 20 components of `groups`: the difference in macro AUC with its 95% paired interval. Per-group topology adds to composition if the interval lies above 0.
   - **Descriptive only:** everything else, i.e. `k = 20`, the heatmaps, the ladder, and each condition against healthy.
   - **Ladder** (logistic regression, plus purity where the rung is a space): `D`, `C`, `D+C`, `all-cell`, `groups`, `everything`, then `D+C` plus the first 20 components of `all-cell`, of `groups` and of `everything`.
8. **Report (`report.html`).**
   - Built like the sibling study's: dataviz palette and group colours, PNGs quantized to 128 colours.
   - Sections:
     1. question and rulings;
     2. cell groups and composition per slice and condition;
     3. per-group fractions of variance;
     4. key spaces (`all-cell`, each group, `everything`): every pair of the first four components and a UMAP (sibling `umap_scores.embed` parameters, on the weighted concatenation);
     5. readout heatmaps (465 MFA spaces, 200 single blocks by slice);
     6. ladder;
     7. the two comparisons fixed in advance, with their verdicts;
     8. each condition against healthy for the key spaces;
     9. methods.
   - Every number in the prose is computed and asserted in `build_report.py`.
9. **Verification (`verify_results.py`), independent of the main code path.**
   - Group map and counts recomputed from the h5ad.
   - Slice 9: per-group fPCA recomputed through the dense route (`betti_inner_product_matrix` + `fpca_from_gram`) and compared with the saved arpack scores.
   - MFA recomputed by eigendecomposing the sum of block Gram matrices.
   - Purity recomputed by brute-force distances.
   - AUCs recomputed by the Mann–Whitney formula from saved probabilities.
   - Folds identical across spaces.
   - Pre-registered numbers recomputed.
10. **Improvement phase** (after the report; exploratory).
    - Ideas are compared on the 476 development slides with the same readouts (20 × stratified 5-fold, drawn on the development slides).
    - Every idea is logged in `progress.md` with its score, failures included.
    - At the end, the main pipeline's `all-cell` and `everything` spaces and the best one to three ideas are fitted on the development slides and scored once on the 118 holdout slides: macro AUC with its interval, and purity of the holdout slides against neighbours drawn from the development slides.
    - Limits: topocyte at `bd1d5c0`, numpy, scipy, scikit-learn, gudhi and umap as installed. No new installs, no changes to topocyte, exact bar-endpoint arithmetic only.
    - Candidate ideas (not commitments): per-cell scaled group curves (Gram scaled by cell counts), density-residualised blocks, block or slice selection by inner cross-validation, supervised projections fitted inside the folds, `C` chosen by inner cross-validation, more components per block, rank functions for H1.

## Left to the builder

- File layout, code structure, figure details, colour choices within the dataviz rules.
- Exact wording of the report.
- Which ideas the improvement phase tries, within the limits above.
