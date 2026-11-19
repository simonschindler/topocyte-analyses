Ledger: the liver pathology density-slice study repeated with the
persistence scale-space kernel (PSSK) in place of the Betti-curve inner
product. Started 2026-11-18 (session in worktree nervous-shamir-3e400e).
Sibling of ../gtex-liver-pathology-slices, whose groups, densities and slice
levels it reuses.

Task 1 (Simon, 2026-11-18, "repeat this experiment with the
gudhi.representations.kernel_methods.PersistenceScaleSpaceKernel instead of
the betti_curve inner product"). Facts found first: gudhi 3.13.0 defines the
kernel as 0.5 * sum over p in D1 u mirror(D1), q in D2 u mirror(D2) of
w(p) w(q) exp(-|p - q|^2 / (2 h^2)) / (sqrt(2 pi) h), with w = +1 on or above
the diagonal and -1 below it (the mirrored points); it evaluates the kernel
pair by pair, with a Python loop over the points. Slides average about
212,000 H0 and 237,000 H1 finite bars at slice 0, so one exact entry is about
10^11 Gaussian terms.
Ruling (Simon, batched multiple-choice, all recommended options):
- approximation: random Fourier features through gudhi's own kernel_approx
  route, scikit-learn's RBFSampler with gamma = 1 / (2 h^2), m = 1000
  features, computed once per slide instead of once per pair; checked
  against gudhi's function on small diagrams and against the exact kernel on
  a few small slide pairs. Error about 1/sqrt(m) per entry.
- bandwidth: for each slice and dimension, h = the median persistence of all
  finite bars pooled over the 594 slides.
- scope: everything the Betti study did - per-slice FVE, PC1-4 pairs and
  UMAP, the MFA-combined spaces, the PC2 density check under the same rule -
  and a table comparing the key numbers with the Betti study.
Ruling (Claude, defaults): the same slices, densities, groups and PH as the
Betti study, which is recomputed (its diagrams were not saved). Gate: every
slide's number of H0 and H1 bars equals the Betti study's saved count. Finite
bars only (gudhi's kernel takes no essential points), in the alpha
filtration's own coordinates (squared radii, px^2, as the Betti curves
used). One sampler per slice and dimension, random_state 0. Two
normalizations, as before: raw (the kernel as it is, K = c Phi Phi^T,
c = 0.5 / (sqrt(2 pi) h)) and cosine (K_ij / sqrt(K_ii K_jj), the
row-normalized features). Kernel PCA by SVD of the centred features, nine
components saved, the FVE over the full spectrum of the approximate kernel.
Cost if wrong: an approximate kernel, measured against the exact one where
that is computable; the bandwidth is not tuned against the labels.
Smoke run, slice 9 (71.7 s; PH 18.0 s, gate passed: bar counts equal the
Betti study's; features 24-25 s per dimension; h = 95.82 px^2 for H0, 16.4
px^2 for H1).
Validation on the six smallest slides at slice 9 (495-1,509 H0 bars):
- our once-per-slide features reproduce gudhi's own kernel_approx output to
  2e-16 (relative), for H0 and H1.
- against gudhi's exact kernel: H0 median entry error 0.3% (max 1.0%), cosine
  error at most 0.0004; H1 median 12.9% (max 15.8%), cosine error up to 0.027
  against off-diagonal cosines of 0.81-0.94. More features help slowly: mean
  H1 error about 4% at m = 4000, 2.7% at 16000, 1.5% at 64000 (five seeds,
  four slides). The H1 points mostly lie farther apart than h, and the
  random-feature noise on the many near-zero pairs swamps the few close
  ones; a sine feature map that builds the mirror cancellation in
  analytically (same kernel) did no better (H1 about 10% at 2000 features).
- stability, slice 9 H1 kernel PCA, other draws (m = 1000 seed 1, m = 4000
  seeds 0 and 1) against the saved run: raw PC1-4 |corr| >= 0.988; cosine
  PC1 0.999, PC2 0.988-0.993, PC3 0.892-0.942, PC4 0.80-0.89; H1 raw PC1's
  FVE moves 0.922-0.941.
Ruling (Simon, asked with these numbers): keep m = 1000 for both dimensions;
the report states the measured error and marks H1 cosine PC3-4 as limited by
the approximation. Cost if wrong: H1 entries about 13% off; the top
components and the combined spaces rest on the stable parts.

Task 2 (Claude, 2026-11-18, defaults for the rest of the scope, set while
slices 8-0 ran and before any of them was looked at beyond the gate line).
- downstream: umap_scores.py, combined_spaces.py and pc2_density_check.py are
  the Betti study's, changed only in wording and in reading pathology.csv and
  density.npz from ../gtex-liver-pathology-slices/results. The PC2 rule is
  that study's Task 4 rule, unchanged.
- validate_kernel.py redoes the Task 1 validation as a script (the six slides
  with the fewest kept cells at slice 9, a stable sort; exact kernel; the H1
  error at 1000-64000 features, five draws; three other draws of the whole
  slice-9 kernel PCA, now for H0 too) and saves the numbers to
  results/validation.json, so the report types none of them. It adds one
  number: how full the exact entries are (median of an entry over its largest
  possible value, every Gaussian at its peak), the stated reason H1 is the
  harder kernel.
- marking: the H1 cosine model's PC3 and PC4 carry a dagger on every axis,
  table column and caption that shows them; the report also gives their share
  of the combined spaces' PC1 and PC2 loadings.
- comparison: one table of key numbers, Betti vs PSSK (PC1 FVE ranges of the
  raw and cosine models, lowest PC1-4 FVE, the all-blocks combined PC1 FVE,
  the combined PC1-density and PC2-share correlations, the PC2 cirrhosis gap,
  and the PC2 check's AUCs and surviving spaces), and the absolute correlation
  of the two studies' PC1 and PC2 scores per model and slice and per combined
  space.
- verify_results.py: the gate re-checked from the saved counts; finite bars =
  kept cells - 1 for H0 (one essential class) and every H1 bar; the saved
  random features are the stated sampler's; every slice's kernel PCA
  recomputed through an eigendecomposition of the centred Gram matrix; slice 9
  recomputed from the cohort with the feature map written out by hand; and the
  Betti study's UMAP, MFA and PC2 checks.
Cost if wrong: presentation only; no number changes.

Task 3 (results, 2026-11-18). Slices 9-0 ran one process each (slice 0: 24
min, peak 9.8 GB); every slice passed the gate. Total compute with validation,
UMAP, MFA and the PC2 check: 137.6 min. verify_results.py: all checks passed.
- validation (validate_kernel.py; the saved seed-0 features): equivalence with
  gudhi's kernel_approx 2e-16; H0 entries off by a median 0.3% (max 1.0%), H1
  by 12.9% (max 15.8%); H1 error 11.5%, 3.6%, 2.8%, 1.4% at 1000, 4000, 16000,
  64000 features (mean of five draws each); an H1 entry is a median 3.0e-3 of
  its ceiling (every Gaussian at its peak), an H0 entry 0.11. Stability at
  slice 9: H0 raw, H1 raw and H0 cosine PC1-4 |r| >= 0.983; H1 cosine PC1
  0.999, PC2 0.988-0.993, PC3 0.892-0.942, PC4 0.800-0.886.
- FVE of PC1: raw 0.810-0.986, H0 cosine 0.885-0.945, H1 cosine 0.461-0.793;
  above the Betti curves' at every slice for H0 raw and H0 cosine, at 9 of 10
  for H1 raw (not slice 3: 0.886 vs 0.887) and H1 cosine (not slice 0).
- against the Betti study: raw PC1 scores |r| >= 0.92 at every slice; cosine
  PC1 agrees at slice 0 (H0 1.00, H1 0.87) and parts at slices 5-9 (at most
  0.56 and 0.58). Combined PC1 agrees at 0.85-1.00 in every space but H1
  cosine (0.11).
- combined (MFA): PC1 is log median density in 4 of 5 spaces (|r| 0.79-0.85);
  in the H1 cosine space PC1 follows the densest-decile share (0.80) and PC2
  the median density (0.81). PC2 cirrhosis-vs-healthy gap 1.15-1.64 SD where
  PC1 is density, 0.45 in the H1 cosine space. PC2 follows the share in the
  all, H0 raw and H1 raw spaces (|r| 0.42-0.58), the median density more in
  H0 cosine (0.41 vs 0.25). Arch R^2 0.01-0.16 (Betti 0.14-0.57). The dagger
  columns carry at most 0.3% of PC1's and PC2's squared loadings.
- PC2 check (the Betti study's Task 4 rule): share alone AUC 0.933 [0.883,
  0.973], above PC2 in every space (0.647-0.888). Share removed: survives in
  3 of 5 (all 0.732 [0.636, 0.819]; H1 raw 0.659 [0.555, 0.753]; H0 cosine
  0.757 [0.666, 0.838]); vanishes in H0 raw (0.514 [0.409, 0.617]); reverses
  in H1 cosine (0.291 [0.203, 0.390]). Betti: 2 of 5. Share, arch, density
  and cells removed: none above 0.5 (0.363-0.525); reversed in all (0.363
  [0.262, 0.468]) and H0 cosine (0.365 [0.267, 0.467]).
- report.html (6.5 MB) from build_report.py, every claim asserted. The prose
  of sections 6-7 was written after these results (the Betti study's claims
  did not all carry over: the H1 cosine space swaps PC1 and PC2, the arch is
  small, two strict residuals reverse).
