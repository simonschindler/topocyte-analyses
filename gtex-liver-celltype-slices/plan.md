# Cell-group density-slice study: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task (native execution: Simon is away overnight and asked for the study and an improvement phase to run on their own; one fresh reviewer on the most capable model checks the whole study at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** per-group Betti-curve fPCA of the 594 GTEx liver slides at the ten all-cell density slices, combined spaces, two separability readouts, the two comparisons fixed in advance, an asserted HTML report, then an exploratory improvement phase checked once on a holdout.

**Architecture:**
- One folder in topocyte-analyses, built like the sibling studies:
  - a run script with subcommands writes `results/*.npz` + `.json`;
  - downstream scripts read only `results/`;
  - a verifier recomputes everything by independent routes;
  - `build_report.py` asserts every claim it prints.
- Pure helpers live in `common.py` and are unit-tested in `test_celltype.py`.

**Tech stack:** Python from the topocyte checkout's `.venv` (numpy, scipy, pandas, h5py, scikit-learn 1.9.0, umap-learn 0.5.12, matplotlib), topocyte `bd1d5c0` via `PYTHONPATH`, pytest.

**Spec:** `design.md` (this folder).

## Global Constraints

- Cohort opened read only: h5py mode "r"; nothing is written under `~/data`.
- topocyte comes from the checkout `PYTHONPATH` names first, at `bd1d5c0`; the run script refuses any other.
- Exact bar-endpoint arithmetic only: no grids, quadrature or changes of measure.
- No persistence diagrams on disk (about 32 GB free).
- At most five PH workers when a slice's point set holds more than 70 million cells.
- 16 GB RAM: one point set's diagrams in memory at a time.
- Logistic regression:
  - `C = 1`, `class_weight="balanced"`, lbfgs, `max_iter=5000`, features standardized within each training fold;
  - 20 repeats of stratified 5-fold, seed 0, the same folds for every space;
  - unclassified fitted as its own class and never scored.
- Purity: 15 neighbours, Euclidean, itself excluded, all 594 slides as candidates, chance `(n_g - 1)/(N - 1)`.
- Bootstrap: 4000 draws of slides stratified by group, `default_rng(12345)`, paired between spaces.
- Decision rule (fixed in advance):
  - primary: `everything` vs `all-cell` at `k = 4`; enhanced iff both the macro-AUC and the macro-purity 95% intervals of the difference lie above 0; mixed iff exactly one does;
  - secondary: `D+C` vs `D+C` + first 20 components of `groups`, macro AUC interval above 0.
- Holdout (`results/holdout.csv`, 118 slides) is touched only by the final step of the improvement phase.
- Commits: local only, in topocyte-analyses, message style of the repository, with the Co-Authored-By line; never push.

## Review Focus

1. **A group cloud gives a zero curve** (no H1 bar among a few dozen stromal cells at slice 9): the cosine model is undefined for that slide. Expected: that block (group × model × slice) is dropped from every space, and the report names it. No crash later, no silent zeros. Covered in Task 2 (the `slice` command records `dropped_blocks`) and in Task 3 (`build_blocks` skips them).
2. **Slice 0 epithelial** (about 90 million cells) running out of memory. Expected: five workers, peak memory logged in the slice json. Covered in Task 2.
3. **Small groups in a resample or a fold** (inflammation/hepatitis 12, necrosis 18). Expected: stratified folds and bootstrap keep every group present, and AUCs are never computed on one class. Covered in Task 4 (`test_bootstrap_keeps_every_group`).
4. **Single blocks have only nine components.** Expected: `k = 20` becomes `k = 9` for them, labelled as such. Covered in Task 4 (`leading(scores, k)`).
5. **lbfgs convergence warnings** on a few spaces. Expected: counted per space and reported, not hidden. Covered in Task 4 (`n_convergence_warnings`).

---

### Task 1: common helpers and composition

**Files:** create `common.py`, `test_celltype.py`, `run_celltype_slices.py` (subcommand `groups`).

**Interfaces (produces):**
- `GROUPS: dict[str, tuple[str, ...]]`, as in the design.
- `POINT_SETS = ("all", "epithelial", "stromal", "immune", "other")`.
- `MODELS = (("h0_none", 0, "none"), ("h1_none", 1, "none"), ("h0_cosine", 0, "cosine"), ("h1_cosine", 1, "cosine"))`.
- `CONDITIONS` (the 7 groups, without unclassified) and `GROUP_ORDER` (as in the sibling study).
- `group_codes(categories: list[str]) -> np.ndarray`: maps a category index to a group index 0..3; raises `ValueError` on an unknown category.
- `read_slides()`, `read_coords()`, `read_cell_types() -> (codes int8, categories)`: from h5py, read only.
- `mfa(blocks: list[np.ndarray], n_keep: int) -> (scores, evr, weights)`: weights are `1/sqrt(var(block[:,0]))`; centred SVD; sign rule as in the sibling study.
- `results/composition.npz` with:
  - `kept_by_group` (10, 594, 4) int64;
  - `group_names`, `slide_ids`, `category_map`.

- [ ] Tests:
  - `test_group_codes_maps_all_14` (the 14 cohort categories map, and every group gets at least one);
  - `test_group_codes_unknown_raises`;
  - `test_mfa_equals_gram_eigh` (synthetic blocks: the MFA scores' Gram equals the centred, weighted sum of block Grams; absolute scores match the eigh route up to sign);
  - `test_mfa_signs` (the largest-magnitude entry of each left singular vector is positive).
- [ ] Run pytest: the tests fail (no module).
- [ ] Implement `common.py` and the `groups` subcommand. It also checks that `kept_by_group.sum(-1) == sibling kept_per_slide` and stops otherwise.
- [ ] pytest passes; `run_celltype_slices.py groups` prints counts that match the recon (slice 0 shares 71.0 / 14.9 / 6.5 / 7.55%).
- [ ] Commit.

### Task 2: per-group PH and fPCA, all-cell gate

**Files:** `run_celltype_slices.py` (subcommands `slice K`, `gate K`).

**Interfaces (produces):**
- `results/slice_K.npz` with:
  - per point set `g` in the four groups and per model: `{g}_{model}_scores` (594, 9), `{g}_{model}_evr` (9,);
  - `{g}_n_bars` (594, 2), `{g}_kept` (594,);
  - `dropped_blocks` (str array).
- `results/slice_K.json`: runtimes, workers, peak RSS, topocyte version.
- `results/gate_K.json`: per model, the minimum absolute correlation of each component and the maximum FVE difference against the sibling; `passed` bool.

- [ ] Write `slice K`:
  - Per group: PH in threads (workers rule), then the four `tc.ph.FPCA(dim, n_components=9, normalization, n_jobs=-1).fit(results)`, H1 first, then dropping H1 diagrams, as in the sibling study.
  - Catch `ValueError` from a cosine fit with zero-norm curves: record the block in `dropped_blocks` and go on.
- [ ] Write `gate K`: the same code path on all cells, compared with the sibling's `slice_K.npz`. Threshold: absolute correlation of every component ≥ 0.999999 and FVE difference ≤ 1e-6.
- [ ] Run `gate 9`, `gate 8`: both `passed: true`. If one fails, stop and debug before any per-group run.
- [ ] Run `slice 9` in the foreground; look at its counts and timings.
- [ ] Launch slices 8..0 in one background job, in order, with logs in `logs/`. About 55 min.
- [ ] Commit the scripts (results stay ignored; the json files are tracked).

### Task 3: blocks, spaces, baselines, UMAPs

**Files:** create `spaces.py`.

**Interfaces:**
- Consumes: `slice_K.npz` here, the sibling's `slice_K.npz` (all-cell), `composition.npz`, the sibling's `density.npz` (`slide_medians`, `kept_per_slide`).
- Produces `results/spaces.npz`:
  - `names` (array of space names);
  - per space `{name}__scores` (594, ≤20) and `{name}__evr`;
  - per MFA space `{name}__blocks` (block keys) and `{name}__weights`.
- Produces `results/umap.npz`: 2-D UMAPs of the key spaces (weighted concatenation, sibling `umap_scores.PARAMS`).
- Space names:
  - single blocks `block:{pointset}:{model}:{slice}`;
  - MFA spaces `mfa:{pointsets joined by +}:{models joined by +}`;
  - aliases `all-cell`, `groups`, `everything`;
  - baselines `D`, `C`, `D+C` (PCA of z-scored features; all components kept).

- [ ] Test `test_baseline_features_shapes` (D has 11 columns, C has 40, finite).
- [ ] Implement `spaces.py`, including `build_blocks()`, which skips `dropped_blocks`.
- [ ] Run it, and check that `mfa:all:h0_none+h1_none+h0_cosine+h1_cosine` equals the sibling's `combined.npz` `all` scores (absolute correlation of the first 20 ≥ 0.999999).
- [ ] Commit.

### Task 4: readouts and comparisons

**Files:** create `readouts.py`; add helper tests.

**Interfaces (produces):**
- `results/folds.npz`: (20, 594) fold ids.
- `results/readouts.npz`, per space and k:
  - out-of-fold probabilities averaged over repeats (594, 8);
  - per-repeat macro AUC (20,);
  - macro AUC and per-condition AUC;
  - condition-vs-healthy AUCs;
  - per-slide purity (594,) and macro purity;
  - `n_convergence_warnings`.
- `results/readouts.json`: the table of point estimates.
- `results/comparisons.json`:
  - primary and secondary differences with 95% intervals and verdicts;
  - ladder rows with intervals;
  - per-condition AUCs vs healthy for the key spaces.
- Helpers in `common.py`:
  - `leading(scores, k)`;
  - `macro_auc(proba, groups) -> (macro, per_condition)`;
  - `purity(scores, groups, n_neighbors=15) -> per_slide`;
  - `stratified_bootstrap(groups, n_draws, rng) -> index arrays`;
  - `verdict(auc_interval, purity_interval) -> str`.

- [ ] Tests:
  - `test_macro_auc_matches_sklearn` (`roc_auc_score` one-vs-rest on synthetic data);
  - `test_purity_toy` (two tight clusters: purity = 1/chance for each);
  - `test_bootstrap_keeps_every_group`;
  - `test_verdict_rule` (all four cases);
  - `test_leading_caps_at_available`.
- [ ] Implement. Logistic regression runs over spaces in a process pool of 10. Warnings are captured per fit.
- [ ] Run (about 15 min); print both comparisons fixed in advance.
- [ ] Commit.

### Task 5: verifier

**Files:** create `verify_results.py`.

- [ ] Checks:
  - group map and counts against the h5ad;
  - `gate_*.json` passed;
  - slice 9 per-group fPCA through `betti_inner_product_matrix` + `fpca_from_gram` against the saved arpack scores (|corr| ≥ 0.99999 for each component, FVE to 1e-6);
  - each MFA space's scores against eigh of the summed weighted block Grams for 10 random spaces and the three named ones;
  - purity by brute force for the named spaces;
  - AUCs by the Mann–Whitney formula from the saved probabilities;
  - folds identical across spaces;
  - comparisons recomputed from the saved arrays;
  - every saved array finite.
- [ ] Run: prints `ALL CHECKS PASSED`.
- [ ] Commit.

### Task 6: report

**Files:** create `build_report.py`, `README.md`, `progress.md`; modify the top-level `README.md` (Studies row).

- [ ] Figures:
  - composition by slice and condition (small multiples);
  - per-group FVE;
  - scatters and UMAPs of `all-cell`, the four groups and `everything`;
  - readout heatmaps;
  - ladder;
  - per-condition AUCs.
- [ ] Claims asserted. Sibling dataviz palette.
- [ ] Render it, then open it in the browser pane and look at it (overflow, legibility).
- [ ] Commit (the report stays out of git).

### Task 7: improvement phase

**Files:** `improve/` scripts, `improvements.md`; results in `results/improve_*.json`.

- [ ] Draw development folds (20 × stratified 5-fold, seed 1, development slides only). Reproduce the baseline readouts of `all-cell` and `everything` on the development slides.
- [ ] Try ideas one at a time. Log each in `progress.md`: what, why, development macro AUC and purity at k = 4 and 20, with paired intervals against `everything`.
- [ ] Choose at most three. Fit them on the development slides; score `all-cell`, `everything` and the chosen ideas once on the holdout. Log the result whatever it is.
- [ ] Write `improvements.md` (and a report section). Commit.

### Task 8: whole-study review and morning summary

- [ ] Dispatch one fresh reviewer (model opus) on the folder: code, claims, and the decision rule applied as written.
- [ ] Fix the confirmed findings and rerun `verify_results.py` and `build_report.py`.
- [ ] Update the memory files. Send Simon the report and summary.
