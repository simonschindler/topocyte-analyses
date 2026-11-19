# Ledger: liver pathology per cell group

## Task 1: rulings (Simon, 2026-11-19, about 21:30)

- **Density:** density among all cells: the sibling's ρ and its ten decile thresholds. A group's cloud at slice s is the slide's cells of that group with ρ ≥ the threshold.
- **Cell groups:** epithelial (Epithelial, Cancer cell) / stromal (Fibroblasts, Muscle Cell, Endothelial Cell, Minor Stromal Cell) / immune (Lymphocytes, Neutrophils, Macrophages, Plasmocytes, Eosinophils) / other (Apoptotic Body, Red blood cell, Mitotic Figures).
- **Readouts:** both, cross-validated logistic regression and neighbour purity. No MLP: with 594 slides and groups of 12 and 18 it would add seed noise to the paired comparison and little else.
- **Headline, fixed in advance:** everything vs all-cell at the same number of components. Secondary: per-group topology on top of density and composition.
- **Holdout:** a seeded, stratified fifth for the improvement phase only (`make_holdout.py`, 118 slides, committed in 620b43c before any result).
- **Commits:** local, in topocyte-analyses; Simon pushes.

## Task 2: Claude's calls

- **Run.**
  - The fPCA runs exactly the sibling's way (`tc.ph.FPCA`, matrix-free ARPACK), not the dense route through the Gram matrix. That route is the verifier's independent check.
  - The all-cell blocks are the sibling's saved scores. A gate reran this study's code path on all cells at slices 9 and 8, and it must match them.
  - No diagrams are saved (32 GB free disk).
- **Spaces.**
  - All 31 × 15 combinations of point sets and models, plus the 200 single blocks and three baselines (`D`, `C`, `D+C`, PCA of z-scored features).
  - The secondary comparison and the ladder use the raw `D` and `C` features, standardized inside each training fold.
- **Readouts:**
  - logistic regression with `C = 1` fixed (a readout, not a tuned model);
  - k = 4 (the scatter view) and 20;
  - macro AUC from the repeat-averaged out-of-fold probabilities;
  - percentile intervals from 4000 stratified bootstrap draws.
  - For purity, the neighbour graph stays fixed and only the query slides are resampled.
- **Improvement phase:** the three questions of the builder check, settled in `design.md` ("Rulings made after the builder check") and committed in b5865d0 before any readout.

## Task 3: run

- 21:42: composition counts match the sibling's kept cells at every slice. At slice 9 the fewest cells of any group on a slide are 24 (stromal).
- 21:46:
  - gate 9 passed: |r| = 1.000000000 and FVE identical for all four models;
  - slice 9 per group in 45 s, with no slide lacking H1 bars and no block dropped.
- 22:27: slices 8 to 0 done; 45.2 min of per-group PH and fPCA in all (slice 0 in 9 min, peak RSS 6.3 GB, five PH workers for epithelial at slices 0 to 2). No block dropped; every slide keeps H1 bars in every group at every slice. Gate 8 also passed exactly.
- 22:29: `spaces.py`: 668 spaces (200 blocks, 465 MFA, 3 baselines). The all-cell space reproduces the sibling's combined space (min |r| 1.000000000).
- 22:30: `readouts.py`: 1342 jobs × 100 fits in 74 s, no convergence warnings.
  - **Primary (fixed in advance): enhanced.** At k = 4, `everything` vs `all-cell`: macro AUC 0.705 vs 0.599, +0.106 [0.073, 0.138]; purity 2.12 vs 1.60, +0.52 [0.27, 0.79]. The same holds at k = 20 (descriptive): +0.027 [0.001, 0.053] and +0.32 [0.03, 0.60].
  - **Secondary (fixed in advance): does not add.** `D+C` + 20 components of `groups` 0.773 vs `D+C` 0.773, 0.000 [−0.023, 0.024].
  - **Descriptive.**
    - The composition features alone reach 0.775; composition PCA on 4 components gives AUC 0.747 and purity 2.20, above `everything`@4 on both.
    - The best combined spaces are the H0 cosine models of immune/other/stromal (post hoc).
    - Reading: the gain from splitting by cell type is the cell counts per group and density level, not their topology.

Ruling: the summary leads with a "Reading" paragraph and states the composition PCA at four components beside the primary result. Why: the primary verdict alone would read as "topology per group helps", which the secondary and the ladder contradict. Cost if wrong: none to the numbers; it frames them, and every sentence in it is asserted against them.

## Task 4: improvement phase (exploratory; development slides only until the holdout step)

- 22:35: verifier: 61 checks pass in 290 s. Report built (1.6 MB) and looked at in the browser: composition, scatters and heatmaps render; the reading leads.
- 22:36, round 1 (`improve.py`, 14 entries, 52 s). Development macro AUC at k = 4, against `everything` (0.696):
  - zscore-columns −0.024;
  - residual-density −0.066;
  - residual-density-composition −0.150 (0.546, near chance);
  - cosine-everything +0.015;
  - select20/40-kruskal +0.017;
  - lda4-everything20 +0.035 (interval crosses 0);
  - **pls4-everything50 +0.075 [0.031, 0.117]**;
  - lda4-allcell20 +0.025;
  - lda4-DC +0.054;
  - lda4-DC+everything20 +0.078.
- 22:40–22:49: per-cell normalization.
  - The dense Gram route costs about 60 s per matrix at slice 9 (an estimated 4 h for everything).
  - Ruling: fit the per-cell curves (β / n) matrix-free with topocyte's private `_betti_operator` and `_fit_matrix_free` at the pinned commit, scaling the product (`D⁻¹ G D⁻¹`). Why: exact and as cheap as the main fits. Cost if wrong: private API; the identity gate (D = I reproduces the raw scores, |r| = 1 − 2e−16 at slice 9) guards the code path.
  - A first, mistaken run of the old dense script at slice 9 (7 min) was moved to the Trash.
- 22:50, round 2 (post hoc, after round 1 and the main readouts): h0cosine-everything +0.023; **pls4-allcolumns +0.072 [0.044, 0.100]**; pls4-DC +0.040; pls4-DC+everything50 +0.043.
- 22:50, diagnostic: **pls4-residual-density-composition 0.539 [0.488, 0.590]**. With density and composition regressed out (linearly), even a supervised projection finds no separation: the topology per group carries nothing linear beyond composition.
- About 23:00: a whole-study review (fresh reviewer, model opus), before the holdout ran. Verdict: with fixes. The main study's numbers, verdicts and verification hold. The fix pass, before the single holdout run:
  - **Ruling: how ideas reach the holdout** (my reading of design.md's ruling 2, fixed before the run).
    - Only entries whose role is `idea` compete. Baselines, controls (the same method without topology, or on all-cell topology) and diagnostics never go.
    - Mechanisms are canonical, and ideas sharing one compete for a single place:
      - block weighting;
      - shape only;
      - block selection;
      - supervised projection (topology);
      - supervised projection (composition + topology);
      - per-cell curves.
    - An idea goes only if its paired development AUC interval against `everything` lies above 0; up to three, best development AUC first. If none does, the best candidate goes as a check.
    - Purity guard: every four-dimensional view, supervised ones included, must not lose out-of-fold purity against `everything`, i.e. the upper end of the paired interval must not be below 0. Out-of-fold purity is each test slide against its 15 nearest training-fold slides in that fold's view.
    - Why: the reviewer showed that the first implementation would have sent three LDA/PLS variants, one of them a no-topology yardstick, and never read the interval.
    - Cost if wrong: another reading would choose other ideas. This one is recorded before the run.
  - **Ruling: what the holdout compares** (`improve_holdout.py`).
    - The holdout also scores the best supervised control without topology.
    - It compares every entry against both `everything@4` and the `D+C` features. The latter is the anti-goal check: a gain carried by cell counts.
    - It saves its probabilities, purities and draws in `improve_holdout.npz`.
    - Why: without these, the single run could not answer whether a gain is topology.
  - Also fixed:
    - the holdout-honesty wording (the main readouts read every label);
    - idea descriptions (z-scoring, input sizes);
    - the report's reading softened to "nothing detectable", with the composition PCA stated by its paired intervals;
    - README run steps for `percell_fpca.py`;
    - the small-group caveat;
    - the report's improvement section.
  - Round 1 and 2 rerun under the fixed code: the same AUCs, now with out-of-fold purity.
- 23:14, round 2: `umap4-everything` (the report's UMAP in four dimensions as the view). AUC −0.042 [−0.077, −0.009] against `everything`, but purity +0.46 [0.19, 0.71]: UMAP keeps neighbourhoods, which a linear readout cannot use. Not eligible.
- 00:03: per-cell fits done (75.1 min in all).
  - `percell-everything` +0.022 [−0.014, 0.056];
  - `percell-only` +0.025 [−0.011, 0.058].
  - Both intervals cross 0, so neither is eligible.
- 00:04: selection dry run (reads only `improve_dev.json`): `lda4-DC+everything20` and `pls4-everything50`; control `lda4-DC`. Pre-holdout state committed (a395cff).
- 00:04: **holdout scored once** (118 slides; holdout macro AUC):
  - `everything@4` 0.715 [0.635, 0.797];
  - `all-cell@4` 0.627;
  - `D+C` features 0.774 [0.719, 0.828];
  - `D+C` + groups20 0.747;
  - control `lda4-DC` 0.768;
  - `lda4-DC+everything20` 0.784 [0.749, 0.821]: +0.069 [0.002, 0.136] vs everything, +0.010 [−0.027, 0.051] vs D+C;
  - `pls4-everything50` 0.781: +0.066 [−0.007, 0.141] vs everything, +0.007 [−0.036, 0.053] vs D+C.
  - Both comparisons fixed in advance point the same way on the holdout: primary +0.088 [0.036, 0.143], secondary −0.026 [−0.075, 0.022].
  - Reading: supervision and cell counts improve the four-component view; topology does not.
