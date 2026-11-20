# Improvement phase: what was tried, and what held on the holdout

Exploratory. Claude ran it overnight on 2026-11-19 while Simon was away, under the rules in `design.md`
The names (`everything@4`, `D+C`, `lda4-X`, ...) are explained in the glossary of `report.html`.
("Rulings made after the builder check") and the rulings in `progress.md`.

- **Development slides.** Every idea is compared on the 476 development slides:
  - 20 × stratified 5-fold (seed 1);
  - the main experiment's logistic regression on the idea's four-dimensional view;
  - 4000 bootstrap draws (seed 54321) for the intervals.
  - Purity is out of fold: each test slide against its 15 nearest training-fold slides in that fold's view.
- **Fitting.** Unsupervised steps saw all slides, without labels. Supervised projections and block selections were fitted inside each training fold.
- **Roles.**
  - `idea`: a candidate for the holdout.
  - `baseline`: the main experiment's spaces.
  - `control`: the same method without topology, never chosen.
  - `diagnostic`: built to remove signal.
- **Rounds.**
  - Round 1 was conceived before any readout existed.
  - Round 2 came after the main readouts and round 1's results. It is post hoc, and the holdout is a weaker check for it.
- **The holdout is not pristine.**
  - The main experiment's readouts used all 594 slides' labels, the holdout's included, as Simon ruled.
  - `improve.py` never reads the holdout labels.
  - `improve_holdout.py` scored the holdout once.

## Every entry, in the order tried

| # | entry | role | round | mechanism | inputs; what | dev macro AUC [95%] | vs `everything`@4 | out-of-fold purity ×chance (vs `everything`) | dev AUC k=20 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `baseline:everything` | baseline | 1 | baseline | MFA of all 200 blocks; the main experiment's everything space, for reference | k=4: 0.696 [0.656, 0.736] | 0.000 [0.000, 0.000] | 1.92 (0.00 [0.00, 0.00]) | 0.749 |
| 2 | `baseline:all-cell` | baseline | 1 | baseline | MFA of the 40 all-cell blocks; the main experiment's all-cell space | k=4: 0.616 [0.575, 0.659] | −0.081 [−0.118, −0.043] | 1.61 (−0.31 [−0.52, −0.10]) | 0.724 |
| 3 | `baseline:D+C-features` | baseline | 1 | baseline | 51 features; density summaries and cells per group and slice, no topology | k=51: 0.764 [0.727, 0.800] | — | 1.96 | — |
| 4 | `zscore-columns` | idea | 1 | block weighting | PCA of 1800 z-scored block columns; every fPCA component weighs the same, instead of each block's first | k=4: 0.673 [0.630, 0.714] | −0.024 [−0.049, 0.002] | 1.90 (−0.02 [−0.15, 0.12]) | 0.747 |
| 5 | `residual-density` | diagnostic | 1 | density removed | MFA of 200 residual blocks; every score regressed on the 11 density summaries first | k=4: 0.630 [0.586, 0.675] | −0.066 [−0.121, −0.012] | 1.59 (−0.33 [−0.59, −0.06]) | 0.661 |
| 6 | `residual-density-composition` | diagnostic | 1 | density and composition removed | MFA of 200 residual blocks; every score regressed on the 51 density and composition features first | k=4: 0.546 [0.499, 0.594] | −0.150 [−0.199, −0.099] | 0.94 (−0.98 [−1.23, −0.74]) | 0.551 |
| 7 | `cosine-everything` | idea | 1 | shape only | MFA of the 100 cosine blocks; curve shape only, no magnitude, so no raw cell counts | k=4: 0.711 [0.670, 0.751] | +0.015 [−0.020, 0.049] | 2.01 (+0.09 [−0.11, 0.29]) | 0.753 |
| 8 | `select20-kruskal` | idea | 1 | block selection | MFA of 20 of 200 blocks; each training fold keeps the 20 blocks whose first four scores differ most across groups (Kruskal-Wallis) | k=4: 0.713 [0.673, 0.753] | +0.017 [−0.020, 0.049] | 2.08 (+0.16 [−0.03, 0.37]) | — |
| 9 | `select40-kruskal` | idea | 1 | block selection | MFA of 40 of 200 blocks; the same with 40 blocks | k=4: 0.713 [0.673, 0.753] | +0.017 [−0.019, 0.048] | 2.08 (+0.16 [−0.06, 0.39]) | — |
| 10 | `lda4-everything20` | idea | 1 | supervised projection (topology) | 20 MFA components, z-scored; shrinkage LDA to four discriminants, fitted in each training fold | k=4: 0.732 [0.688, 0.775] | +0.035 [−0.015, 0.079] | 2.01 (+0.09 [−0.13, 0.31]) | — |
| 11 | `pls4-everything50` | idea | 1 | supervised projection (topology) | 50 MFA components of all 200 blocks, z-scored (whitened); class-balanced PLS-DA to four components, fitted in each training fold | k=4: 0.771 [0.729, 0.808] | +0.075 [0.031, 0.117] | 2.15 (+0.23 [−0.07, 0.53]) | — |
| 12 | `lda4-allcell20` | control | 1 | supervised projection (all-cell topology) | 20 all-cell MFA components, z-scored; the same LDA on all-cell topology: what supervision alone buys | k=4: 0.721 [0.683, 0.760] | +0.025 [−0.012, 0.063] | 1.82 (−0.10 [−0.28, 0.09]) | — |
| 13 | `lda4-DC` | control | 1 | supervised projection (composition) | 51 density and composition features, z-scored; the same LDA without topology: the yardstick for the supervised ideas | k=4: 0.750 [0.704, 0.792] | +0.054 [0.012, 0.094] | 2.21 (+0.29 [0.05, 0.56]) | — |
| 14 | `lda4-DC+everything20` | idea | 1 | supervised projection (composition + topology) | 51 density and composition features + 20 MFA components, z-scored; shrinkage LDA to four discriminants, fitted in each training fold | k=4: 0.774 [0.736, 0.812] | +0.078 [0.044, 0.113] | 2.42 (+0.50 [0.24, 0.77]) | — |
| 15 | `h0cosine-everything` | idea | 2 | shape only | MFA of the 50 H0 cosine blocks; the H0 cosine blocks alone (the main heatmaps' strongest model) | k=4: 0.720 [0.680, 0.759] | +0.023 [−0.007, 0.052] | 2.01 (+0.09 [−0.12, 0.32]) | 0.761 |
| 16 | `pls4-allcolumns` | idea | 2 | supervised projection (topology) | 1800 block columns, z-scored (MFA weights do not survive); class-balanced PLS-DA to four components, fitted in each training fold | k=4: 0.769 [0.736, 0.801] | +0.072 [0.044, 0.100] | 2.23 (+0.31 [0.10, 0.51]) | — |
| 17 | `pls4-DC` | control | 2 | supervised projection (composition) | 51 density and composition features, z-scored; the same PLS-DA without topology | k=4: 0.736 [0.698, 0.773] | +0.040 [0.009, 0.070] | 2.00 (+0.08 [−0.12, 0.30]) | — |
| 18 | `pls4-DC+everything50` | idea | 2 | supervised projection (composition + topology) | 51 features + 50 MFA components, z-scored; class-balanced PLS-DA to four components, fitted in each training fold | k=4: 0.739 [0.700, 0.777] | +0.043 [0.013, 0.072] | 2.14 (+0.22 [0.02, 0.43]) | — |
| 19 | `pls4-residual-density-composition` | diagnostic | 2 | supervised projection (topology beyond composition) | 50 MFA components of 200 residual blocks, z-scored; PLS-DA after regressing every score on density and composition: is any signal left? | k=4: 0.539 [0.488, 0.590] | −0.158 [−0.222, −0.091] | 1.03 (−0.89 [−1.13, −0.64]) | — |
| 20 | `umap4-everything` | idea | 2 | nonlinear embedding | UMAP to four dimensions of the MFA-weighted 1800 block columns; the UMAP the report draws, in four dimensions (n_neighbors 15, min_dist 0.1, seed 0), as the view | k=4: 0.655 [0.612, 0.696] | −0.042 [−0.077, −0.009] | 2.38 (+0.46 [0.19, 0.71]) | 0.655 |
| 21 | `percell-everything` | idea | 1 | per-cell curves | MFA of 100 per-cell and 100 cosine blocks; raw blocks replaced by per-cell ones (Betti curve / number of points) | k=4: 0.719 [0.679, 0.758] | +0.022 [−0.014, 0.056] | 1.99 (+0.07 [−0.14, 0.28]) | 0.765 |
| 22 | `percell-only` | idea | 1 | per-cell curves | MFA of the 100 per-cell blocks; the per-cell H0 and H1 blocks alone | k=4: 0.721 [0.684, 0.759] | +0.025 [−0.011, 0.058] | 1.89 (−0.03 [−0.25, 0.20]) | 0.762 |

On the 476 development slides, the main experiment's `everything` space reaches a macro AUC of 0.696 [0.656, 0.736] on four components and 0.749 on twenty. `D+C` alone reaches 0.764 [0.727, 0.800] (51 features).

- **Best idea:** `lda4-DC+everything20` (supervised projection (composition + topology), 51 density and composition features + 20 MFA components, z-scored) at 0.774, +0.078 [0.044, 0.113] against `everything`.
- **Beyond `everything`:** 4 ideas beat `everything` with an interval above 0: `pls4-everything50`, `lda4-DC+everything20`, `pls4-allcolumns`, `pls4-DC+everything50`.
- **Supervised four-component views:**
  - They use more input than `everything`'s four components: 20 to 1800 dimensions, listed under "what" in the table.
  - On equal inputs, `everything` on 20 components reaches 0.749.
- **Diagnostics,** built to remove signal: `residual-density` 0.630 [0.586, 0.675]; `residual-density-composition` 0.546 [0.499, 0.594]; `pls4-residual-density-composition` 0.539 [0.488, 0.590].

## Holdout, scored once

- **Chosen by the fixed rule:** `lda4-DC+everything20`, `pls4-everything50`.
- **Control:** the best supervised control without topology, `lda4-DC`.
- **When and what:** scored 2026-11-20 00:04 on 118 slides (healthy 13, congestion 31, steatosis 40, necrosis 4, inflammation/hepatitis 2, fibrosis 10, cirrhosis 11, unclassified 7).
- **Uncertainty:** with groups of two and four slides, bootstrap draws within a group understate it, so every holdout interval is too narrow for the small groups.

| entry | holdout macro AUC [95%] | vs `everything@4` | vs `D+C` | purity ×chance (vs `everything@4`) |
|---|---|---|---|---|
| everything@4 | 0.715 [0.635, 0.797] | 0.000 [0.000, 0.000] | −0.059 [−0.110, −0.008] | 2.19 (0.00 [0.00, 0.00]) |
| all-cell@4 | 0.627 [0.559, 0.695] | −0.088 [−0.143, −0.036] | −0.147 [−0.204, −0.093] | 1.50 (−0.69 [−1.10, −0.34]) |
| D+C | 0.774 [0.719, 0.828] | +0.059 [0.008, 0.110] | 0.000 [0.000, 0.000] | — |
| D+C + groups@20 | 0.747 [0.697, 0.799] | +0.032 [−0.040, 0.106] | −0.026 [−0.075, 0.022] | — |
| lda4-DC@4 (control) | 0.768 [0.719, 0.816] | +0.053 [0.002, 0.105] | −0.005 [−0.030, 0.018] | 3.25 (+1.05 [0.52, 1.56]) |
| lda4-DC+everything20@4 | 0.784 [0.749, 0.821] | +0.069 [0.002, 0.136] | +0.010 [−0.027, 0.051] | 2.99 (+0.80 [0.27, 1.31]) |
| pls4-everything50@4 | 0.781 [0.737, 0.823] | +0.066 [−0.007, 0.141] | +0.007 [−0.036, 0.053] | 3.01 (+0.82 [−0.53, 2.17]) |

- `lda4-DC+everything20`: above `everything` (+0.069 [0.002, 0.136]), level with `D+C` (+0.010 [−0.027, 0.051]).
- `pls4-everything50`: level with `everything` (+0.066 [−0.007, 0.141]), level with `D+C` (+0.007 [−0.036, 0.053]).
- Primary comparison on the holdout (`everything` against `all-cell`, four components): +0.088 [0.036, 0.143].
- Secondary comparison on the holdout (`D+C` + 20 components of `groups` against `D+C`): −0.026 [−0.075, 0.022].

**Reading.**

- **No chosen idea beats composition.** On unseen slides no chosen idea's difference to `D+C` (0.774) has its interval above 0: `lda4-DC+everything20` +0.010 [−0.027, 0.051]; `pls4-everything50` +0.007 [−0.036, 0.053].
- **The topology-free control matches them.** `lda4-DC`, the supervised control without topology, reaches 0.768, above the unsupervised `everything` space.
- **`lda4-DC+everything20` beats `everything`.** What improves separability on a four-component view is supervision and the cell counts, not more topology.
