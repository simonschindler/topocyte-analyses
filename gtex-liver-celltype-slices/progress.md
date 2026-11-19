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
- 22:33: `readouts.py`: 1342 jobs × 100 fits in 74 s, no convergence warnings.
  - **Primary (fixed in advance): enhanced.** At k = 4, `everything` vs `all-cell`: macro AUC 0.705 vs 0.599, +0.106 [0.073, 0.138]; purity 2.12 vs 1.60, +0.52 [0.27, 0.79]. The same holds at k = 20 (descriptive): +0.027 [0.001, 0.053] and +0.32 [0.03, 0.60].
  - **Secondary (fixed in advance): does not add.** `D+C` + 20 components of `groups` 0.773 vs `D+C` 0.773, 0.000 [−0.023, 0.024].
  - **Descriptive.**
    - The composition features alone reach 0.775; composition PCA on 4 components gives AUC 0.747 and purity 2.20, above `everything`@4 on both.
    - The best combined spaces are the H0 cosine models of immune/other/stromal (post hoc).
    - Reading: the gain from splitting by cell type is the cell counts per group and density level, not their topology.

Ruling: the summary leads with a "Reading" paragraph and states the composition PCA at four components beside the primary result. Why: the primary verdict alone would read as "topology per group helps", which the secondary and the ladder contradict. Cost if wrong: none to the numbers; it frames them, and every sentence in it is asserted against them.
