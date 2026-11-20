# Microanatomical domains from local chromatic persistent homology: design

2026-11-20. Simon asked for a brainstorm on topologically defined
microanatomical domains in the GTEx liver whole-slide images, built from
local chromatic persistent homology. The idea joins two future directions of
the cell-type study
([`gtex-liver-celltype-slices`](../gtex-liver-celltype-slices/README.md),
report section 13): topology between cell types, and local, regional
analysis. It was settled by Simon's answers to batched questions on
2026-11-20 and by a timing spike he asked for (appendix).

## Intent

The cell-type study found that the per-type topology of whole slides
re-describes density and composition. Its discussion gave three reasons:

- each cell type was a cloud of its own, so relations between types never
  entered;
- whole-slide curves pool portal, lobular and septal regions;
- Betti curves forget which birth belongs to which death.

This study asks the next question at the scale where the liver is organized.
Around each cell, does the arrangement of cell types relative to each other,
measured by the chromatic persistent homology of the cell's neighbourhood,
define recurring microanatomical domains that density, composition and
pairwise contacts do not?

It is a pilot and a precondition. If topology adds domains, a second study
combines them with topocyte's local log-odds and attribution maps to find
where the pathology conditions differ, region by region. If it does not, the
composition compartments of level 1 still give that study its regional
labels.

## Rulings (Simon, 2026-11-20)

| Question | Ruling |
|---|---|
| What a domain is | Two levels. **Level 1, compartments:** from the density and composition of each centre's neighbourhood (the cellular-neighbourhood recipe), with no topology. **Level 2, subdomains:** inside each compartment, from chromatic topology features measured against a reference. |
| Pairwise guard | Fixed in advance: the subdomains must not be predictable from pairwise co-occurrence features, i.e. Delaunay edge counts per class pair against the same reference. |
| Scope | A pilot of about 30 slides stratified by condition, about 1,500 sampled centre cells each. Pathology localisation is study 2. |
| Validation | Two checks. **H&E atlas:** random centres per domain, labels hidden first. **Recurrence** across two halves of the slides. Simon downloads the pilot's SVS files from the cluster when the study reaches the atlas. |
| Reference | **Random labelling.** The observed chromatic PH always uses the true classes. The reference keeps the same cells, positions and count per class, and shuffles the classes within the neighbourhood. Each shuffle goes through the same chromatic PH, and a feature is the observed curve minus the mean reference curve. Simon asked why classes are permuted at all; the answer: the shuffles are only the reference, never the feature. |
| Neighbourhood | The closed δ-ball around each centre cell, δ = 150 µm. |
| Summaries | **Per class, as Betti curves:** the six-pack of the inclusion (class t → all cells) for each class. Only its domain and image curves are kept; see "Why domain and image suffice". |
| Clustering | k-means at both levels, with k chosen by prediction strength across the two slide halves. Refined after the builder check (next row). |
| Choosing k (after the builder check) | Prediction strength alone finds clusters in cluster-free elongated clouds, so k comes from the **gap statistic against a Gaussian with the same mean and covariance**: k is where the gap stops rising, counting only gains of at least 0.02. **Prediction strength** of at least 0.8 in both directions then confirms recurrence; otherwise k = 1. Both levels (step 10). |
| Scaling of residuals (after the builder check) | **Moderated spread:** each residual curve is divided by sqrt(s² + s0²), the centre's own reference spread plus the block's median reference spread. A structure never seen in the shuffles keeps a finite, large value (step 5). |
| Cost | Measured by a timing spike before this design (appendix). |

Claude's calls during the brainstorm, stated to Simon without objection:

- **Classes:** the cell-type study's four groups (epithelial, stromal, immune, other).
- **Scale:** one, δ = 150 µm; cirrhotic nodules (millimetre scale) are out of scope.
- **Location:** the study lives in `topocyte-analyses/gtex-liver-chromatic-domains/`, and nothing in topocyte changes.
- **Construction:** each ball's chromatic complex is built from its own cells.
- **Engine:** the numba port of chalc's chromatic Delaunay–Čech filtration plus phimaker, with chalc as test oracle.

Claude's calls after the builder check, flagged in the hand-over:

- **Control coordinates:** the control is projected into the observed bases rather than refitting its own (step "Checks fixed in advance", 1).
- **Combining the halves:** k is the smaller of the two halves' gap choices (step 10).
- **Pre-flight:** the rule for k is checked on synthetic clouds before the run.
- **Reference draws:** B = 20 for the gap statistic.

Point clouds shown to Simon in a browser go through an adaptation of
`tissuegeometry/web/cosine_viewer`, never the viewer as is (Simon,
2026-11-20).

## Goals, non-goals, anti-goals

- **Goals.**
  - **Pilot slides and centres:** about 30 slides, two halves, about 1,500 centres per slide.
  - **Per centre:**
    - the local chromatic six-packs of the observed classes and of R = 19 random labellings;
    - residual Betti curves;
    - exact fPCA;
    - the pairwise rung.
  - **Domains:** level-1 compartments and level-2 subdomains, with k chosen by the gap statistic against a matched Gaussian and confirmed by prediction strength.
  - **Checks:**
    - the pairwise guard, fixed in advance;
    - a control that must find no subdomains;
    - recurrence and spatial contiguity.
  - **Atlas:** the H&E atlas and an adapted viewer.
  - **Outputs:** an HTML report whose numbers are asserted, a verifier, and a ledger.
- **Non-goals.**
  - Pathology localisation, local log-odds, slide-level classification.
  - Any change to topocyte, including the chalc integration: its nine rulings stay open.
  - Rank functions and class-pair inclusions (class t → t ∪ u).
  - The monochromatic inclusion, several scales, cirrhotic nodules (δ ≳ 1 mm).
  - Summaries for every cell: only sampled centres get them, and the maps show centres.
  - The full cohort.
- **Anti-goals.**
  - Subdomains that are density, composition or pairwise contacts in disguise.
  - Choices tuned on results: δ, R, k ranges, thresholds and features are fixed here, before any domain exists.
  - Grids, quadrature or changes of measure: exact bar-endpoint arithmetic only.

## Constraints

- **Machine:** M4, 10 cores, 16 GB RAM, about 31 GB of free disk.
- **Data, read only:**
  - the cohort `~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad`, opened with h5py in mode "r": `obsm/spatial`, `obs/cell_type`, slide rows from the sibling study's `density.npz` (`slide_ids`, `bounds`);
  - `pathology.csv` (sibling study);
  - `holdout.csv` (cell-type study).
- **Units.**
  - Coordinates are level-0 pixels at an assumed 0.4942 µm/px (the cohort has no mpp; every local SVS has 0.4942), so δ = 150 µm = 303.52 px.
  - Filtration values are squared radii in px², topocyte's alpha convention: the port's smallest-enclosing-ball radius, squared.
- **Software.**
  - A study venv on Python 3.13, as in the spike: numba, gudhi ≥ 3.11 (`DelaunayComplex`), phimaker 0.5.0, chalc 15.0.1 (test oracle only), numpy, scipy, scikit-learn, h5py, pandas, plus topocyte's own dependencies.
  - topocyte itself is imported from a clean checkout of a pinned commit through `PYTHONPATH`, as in the sibling studies; the plan fixes the commit, today's local main `18e3aa5`.
  - The port comes from chalc-scaling commit `8f370c6` (`report/port/chromatic_port.py`) and is copied into the study with credit to chalc and to di Montesano et al. (arXiv:2212.03128).
- **Commits:** local, in topocyte-analyses, only when Simon asks; Simon pushes. This design should be committed before any domain result exists, if Simon agrees.

## Approaches considered

| Decision | Chosen | Alternatives and why not |
|---|---|---|
| Construction of each ball's complex | Its own cells: the chromatic Delaunay–Čech complex of exactly the ball's cells, for observed and shuffled classes alike | **Restricting one global chromatic complex to the ball**, as `compute_local_persistent_homology` does: a shuffle inside one ball changes the global complex, so every shuffle would need a whole-slide rebuild. **Ball plus margin, restricted to the ball:** closer to topocyte's definition, about 2× the cost, and still not the global complex. The ball's edge closes up the same way in observed and reference, so the residual cancels it. |
| Reference | Random labelling within the ball | **Rotating classes against each other about the centre** (the random-shift test) keeps each class's own pattern and isolates the between-class part, but rotated cells overlap. **No reference** (post-hoc guard only) is cheapest, but not by construction, and the last study suggests the ladder would win. |
| Summaries | Per class (t → all), Betti curves, domain and image | **Rank functions** keep the pairing, so all five label-dependent modules carry distinct information, but the fPCA is heavier and the cohort study needed 8–32 rank scores. **The monochromatic inclusion** needs 4× fewer reductions but pools the classes. |
| Neighbourhood | δ-ball, 150 µm | **k nearest neighbours:** constant cost, but a radius of 90–220 µm across slides, so the same anatomy is seen at different scales. **Delaunay n-hop:** the same issue. The δ-ball shares topocyte's δ convention, so study 2 can reuse it. |
| Clustering | k-means, k by prediction strength across halves | **Leiden on a kNN graph** (the microdomain spike): a resolution instead of k, recurrence only through label transfer, new installs, and tiny outlier clusters. |
| Engine | The numba port + phimaker (`num_threads=1`), chalc as oracle | **chalc itself:** a filtration 2.6–54× slower than the port's in the integration report (the gap narrows with more classes), and an API that broke a month ago. **A numba six-pack** (strategy D of the report): correctness-critical, not for a pilot. |

## Why domain and image suffice

For the inclusion f: L → K, at every filtration value r, rank–nullity and the
long exact sequence of the pair give, for the pointwise dimensions (hence for
the Betti curves):

- dim H_p(L_r) = dim ker f_r + dim im f_r, that is **dom = ker + im**;
- dim H_p(K_r) = dim im f_r + dim coker f_r, that is **cod = im + cok**;
- dim H_p(K_r, L_r) = dim coker f_r (degree p) + dim ker f_r (degree p − 1), that is **rel_p = cok_p + ker_{p−1}**.

The spike verified all three exactly, at every bar endpoint, on 12 real
liver balls. The codomain is the alpha PH of all the ball's cells. It does
not depend on the labels, so it is the same in every random labelling and its
residual is zero. For each class and degree, the domain and image residual
curves therefore determine every Betti-curve residual of the six-pack.

- The **domain** of class t is the alpha PH of the ball's t cells: the class's own arrangement.
- The **image** is the between-class part. In H1 it holds the loops of t cells that stay holes among all cells, such as a lumen ringed by t. The kernel (dom − im) holds the t loops that other cells fill. In H0 the image tracks when t components join through any cells.

## Pipeline

1. **Pilot slides.**
   - Eligible slides: the cell-type study's 476 development slides (its holdout excluded), minus the 20 with `quality_flag` "preservation". Eligible counts: healthy 53, congestion 118, steatosis 158, inflammation/hepatitis 9, fibrosis 40, cirrhosis 44.
   - Five slides each are drawn from those six conditions, uniformly within the condition with a fixed seed (`default_rng(20261120)`): 30 slides. Necrosis and unclassified are left out.
   - **Halves:** within each condition, a seeded split into 3 + 2 slides, alternating which half gets 3, so halves A and B hold 15 slides each and every condition appears in both.
   - Written to `results/pilot.csv` (slide, condition, half) before anything else runs.
2. **Centres and balls.**
   - Per slide, 1,500 centre cells are drawn uniformly without replacement from all its cells, seeded per slide.
   - A centre's ball holds every cell of the slide within 303.52 px of it, boundary included, the centre itself too.
   - Each cell's class is one of the four groups (the cell-type study's `common.GROUP_MEMBERS`).
3. **Observed and reference labellings.**
   - Labelling 0 is the true classes; labellings 1..19 are seeded random permutations of the ball's class vector (positions fixed, counts per class kept).
   - The seeds derive from (slide, centre) through `np.random.SeedSequence`, so any centre can be recomputed alone.
4. **Chromatic six-packs, per labelling.**
   - The chromatic Delaunay–Čech filtration of the ball's cells with the classes present: the lift, GUDHI `DelaunayComplex`, numba faces and miniball.
   - Faces up to dimension 2: the spike found identical six-packs in degrees 0–1 with and without dimension 3, and a gate re-checks this (Verification).
   - Values are squared radii in px².
   - For each class t present, phimaker's six-pack of (L_t → K) on one thread, where L_t holds the simplices whose vertices are all of class t.
   - Kept per labelling: the domain and image diagrams in degrees 0 and 1 for each class present, and the codomain once, for the checks.
   - An absent class has empty diagrams in every labelling.
5. **Residual curves.**
   - For each centre, class t, module m ∈ {domain, image} and degree p (16 blocks): the reduced Betti curve (finite bars only, topocyte's `BettiCurve` convention) of labelling 0 minus the mean of labellings 1..19.
   - This is an exact signed step function: the observed bars with weight +1, the reference bars with weight −1/19.
   - **Scaling (Simon, moderated spread).** The residual is divided by sqrt(s² + s0²).
     - s² = mean_r ‖β_r − β̄‖² is the centre's own reference spread in that block, with β̄ the mean reference curve and the norm the exact L² norm. It is computed exactly as (1/R) Σ_r ‖β_r‖² − ‖β̄‖², from the 19 × 19 inner products of the references.
     - s0² is the median of s² over the fitting set's centres with s > 0 in that block.
     - Why: chance fluctuations grow with the number of cells, so raw residuals would let density back in through their variance. Dividing by the centre's own spread alone would zero, or blow up, a structure the shuffles never produce. In the builder check's 80 real balls, all 19 references agreed (s = 0) while the observed curve differed in up to 2 of 40 balls per block, all in image H1, such as a lumen ringed by one class. With s0 those keep a finite, large value.
     - A block with no centre of positive s in the fitting set carries no variation and is dropped, which is reported.
6. **Exact fPCA per block.**
   - Three fits (fitting sets): half A and half B, for the halves' gap statistics and prediction strength, and all 30 slides, for the final features. Each uses a fit sample of 400 centres per slide of its slides.
   - Gram matrix of the scaled residual curves from exact L² inner products of weighted bars (study code, exact event sums).
   - Components from `topocyte.ph.fpca_from_gram` (raw normalization, divisor n), 9 per block.
   - Scores of every centre by exact evaluation against the component step functions; no curve is evaluated on a grid.
   - Gate: with unit weights, the study's inner products match `topocyte.ph.betti_inner_product_matrix` to 1e-10 relative.
7. **MFA.** As in the sibling studies:
   - each block divided by the square root of the variance of its first score (on the fit sample);
   - the 16 × 9 = 144 columns concatenated;
   - PCA of the centred concatenation, signs fixed by the sibling rule;
   - 20 components kept as the level-2 feature space.
8. **Level 1, compartments.**
   - Features per centre: log10 of the ball's cell count, and the ball's four class fractions. They are z-scored with a fitting set's mean and SD: each half's own for its gap statistic, the training half's in each prediction-strength direction, and all 30 slides' for the final fit.
   - k-means (scikit-learn, `n_init=20`, fixed seed) for k = 1..10; k is chosen by the rule of step 10.
   - Final compartments: k-means with the chosen k on all 30 slides' centres. With k = 1 there is one compartment, and level 2 runs on all centres.
9. **Level 2, subdomains.**
   - Inside each compartment, k-means on the level-2 features for k = 1..6, with k chosen by the rule of step 10.
   - k = 1 (no subdomains) is a valid outcome.
   - A compartment with fewer than 500 centres in either half gets no subdomains.
   - Final subdomains: k-means with the chosen k on all its centres, in the basis fitted on all 30 slides.
10. **Choosing k (Simon, after the builder check): the gap statistic against a matched Gaussian, confirmed by prediction strength.**
    - **Why not prediction strength alone.** It measures how stable a split is, not whether clusters exist. The reviewer showed, and Claude confirmed, that a single cluster-free Gaussian with variances falling off like PCA scores gives PS(2) ≈ 0.98 at 4,000 points per half. The drafted rule would therefore have split noise, and the control would have stopped every run.
    - **Gap statistic** (Tibshirani, Walther and Hastie, 2001), with a matched reference, on each half separately:
      - for a feature matrix X, W_k is the k-means within-cluster sum of squares, and W_1 the total sum of squares about the mean. X holds one half's centres: at level 1 the half's z-scored features; at level 2 the half's centres of one compartment, in the level-2 basis fitted on that half;
      - the reference is B = 20 draws from a Gaussian with X's mean and covariance and X's number of rows, with W*_k for each draw;
      - Gap(k) = mean_b log W*_k − log W_k and s_k = sd_b(log W*_k) · sqrt(1 + 1/B);
      - **the half's k** is the smallest k ≥ 2 with Gap(k) − Gap(1) ≥ 0.02 and Gap(k) ≥ Gap(k+1) − s_{k+1}; if there is none, it is 1. At k = k_max only the 0.02 condition applies.
      - In Claude's simulation (4,000 points, 20 dimensions) this gave k = 1 on a cluster-free elongated cloud (largest gain 0.006), on a skewed one and on a level-1-like cloud, and k = 2 and k = 3 on real two- and three-cluster mixtures (gains 0.044 and 0.115). A weak two-cluster split, 2.5 SD apart (gain 0.010), is missed.
    - **Both halves:** k = the smaller of the two halves' choices (Claude's call).
    - **Prediction strength** (Tibshirani and Walther, 2005) at that k must confirm recurrence. Otherwise k = 1.
      - In direction A→B, the basis (steps 6–7, level 2 only) and the training k-means are fitted on half A. Half B is scored in that basis and clustered on its own with the same k.
      - For each test cluster, the share of its pairs that the training centroids also put together; PS(k) is the smallest share over the test clusters.
      - Direction B→A is symmetric, and PS(k) ≥ 0.8 is required in both.
    - All of this is fixed in advance and applies at both levels (k_max = 10 for compartments, 6 for subdomains).
11. **Pairwise rung and density/composition.**
    - Per ball, the planar Delaunay triangulation of its cells (label-free).
    - The number of edges per unordered class pair (10 pairs, same-class included), minus its exact expectation under random labelling:
      - |E| · n_a n_b / C(n, 2) for a ≠ b;
      - |E| · C(n_a, 2) / C(n, 2) for a = b.
      
      These are 10 features, needing no shuffles.
    - Beside them sit the level-1 features (log10 count, four fractions).

## Checks fixed in advance

1. **Control (a gate, not a verdict).**
   - Labelling 1 stands in for the observed, and its residual is taken against labellings 2..19, with the same moderated scaling (its own s, the observed fitting set's s0).
   - It is projected into the observed bases (fPCA components and MFA weights, Claude's call), then run through the same compartments and the same rule for k (step 10). It asks whether those coordinates and that rule find structure in pure shuffle noise.
   - It must find k = 1 in every compartment. If it finds subdomains, the pipeline manufactures structure. Level 2 then stops and the report says so; the level-1 compartments still go to the atlas and to study 2.
   - **Pre-flight, before the 2-hour run:** the rule for k must return k = 1 on matched Gaussian clouds of the pilot's sizes, and the right k on synthetic two- and three-cluster mixtures.
2. **Pairwise guard (the verdict).** For each compartment with k_c ≥ 2 and each direction (train on one half, test on the other):
   - **Stability:** the test half's final subdomain labels predicted by the training half's own subdomain centroids, fitted on the training half alone in its own basis. Labels are matched by the Hungarian assignment on the training half. This gives the balanced accuracy BA_stab.
   - **Pairwise:** the test half's final subdomain labels predicted by a classifier trained on the training half's final labels from the 15 pairwise, density and composition features. The classifier is scikit-learn's `HistGradientBoostingClassifier` with default settings and a fixed seed: flexible, to give the pairwise features every chance. This gives BA_pw.
   - **Rule:** topology adds beyond pairwise contacts, density and composition in compartment c if BA_stab − BA_pw > 0 with its 95% interval above 0, in both directions.
     - The interval comes from 4000 bootstrap draws of the test half's slides (centres resampled by slide).
     - The study's verdict is "adds" if at least one compartment holding at least 10% of all centres passes, and "does not add" otherwise.
3. **Descriptive only:**
   - the same comparison for a classifier on the 8 domain blocks' scores alone: how much of the subdomains each class's own arrangement explains, and so what the between-class image adds;
   - gap and prediction-strength curves, per half and level;
   - cross-half adjusted Rand index;
   - per-subdomain mean residual curves and composition profiles;
   - spatial contiguity: the share of each centre's five nearest same-slide centres with its subdomain, against the exact expectation when subdomain labels are reassigned at random among the same compartment's centres on that slide.

## H&E atlas and viewer (after Simon downloads the 30 SVS files)

- **Atlas page.** For every compartment and subdomain, 12 centres drawn at random, at most 3 per slide.
  - Each centre gets a 400 µm square crop of the H&E at level 0, with the ball's outline and the cells as dots coloured by class (toggle).
  - **First pass:** the groups carry anonymous, randomly ordered IDs and no summaries. Simon describes each group in his own words (portal tract, central vein, septum, lobular parenchyma, capsule, artefact, ...), and his words go into `progress.md`.
  - **Second pass:** the IDs are revealed, with composition profiles and mean residual curves.
- **Viewer.** An adaptation of `tissuegeometry/web/cosine_viewer`: OpenSlide DeepZoom tiles plus OpenSeadragon and a WebGL overlay, without the tissuegraphs or Rust dependencies. It shows one pilot slide with:
  - centres coloured by compartment or subdomain;
  - cells by class;
  - the ball of a clicked centre.
- **The anatomy verdict is Simon's**, recorded in his words, with no number attached.

## Report, verification, ledger

- `report.html`, built like the siblings' (dataviz palette, quantized PNGs, every number in the prose computed and asserted in `build_report.py`). Sections:
  1. question and rulings;
  2. pilot slides and compute;
  3. compartments (profiles, gap and prediction strength, maps);
  4. subdomains per compartment (gap and prediction strength, mean residual curves, maps);
  5. the control;
  6. the pairwise guard and its verdict;
  7. contiguity;
  8. the H&E atlas, after Simon's review;
  9. methods.
- `verify_results.py`, independent of the main code path:
  - the port against chalc 15.0.1 on 200 sampled real balls, with faces to dimension 3 and to dimension 2;
  - the three Betti identities on the same balls;
  - the pairwise expectation against 10,000 brute-force shuffles on 20 balls;
  - the Gram matrix against topocyte's `betti_inner_product_matrix` for unit weights;
  - fPCA against `fpca_from_gram`, and MFA by eigendecomposition;
  - k-means, the gap statistics (from the saved reference seeds), prediction strength and balanced accuracies recomputed from saved labels;
  - the scaling: s² recomputed by brute force from the 19 reference curves on 20 balls, and s0 per block;
  - the verdict recomputed.
- `progress.md`: every ruling with its cost if wrong, and every result in order, as in the siblings.

## Compute

- **Six-packs:** 45,000 centres × 20 labellings = 900,000 complexes.
- **Time:** at the spike's medians (36 ms per complex in the healthy slide, 68 ms in the cirrhosis slide) about 12 h serial. Threads gave only 2.2–2.3× on 8 workers, because the numba kernels and the Python glue hold the GIL. So the run uses 8 processes: about 2 h.
- **Memory:** under 0.5 GB per process (spike peak 0.42 GB with 8 threads).
- **Storage:** a median of 449 finite domain and image bars per labelling (255–841 over 20 spike balls at δ = 150 µm), so about 0.14 MB per centre and about 6.5 GB in all as float64 npz per slide.
- **The rest:**
  - The 16 Gram matrices of up to 12,000 curves per fitting set (6,000 per half) and the scoring take minutes to tens of minutes per block.
  - The gap statistics (20 Gaussian draws × k = 1..k_max per half, level and compartment, on up to about 22,500 centres) and the readouts take tens of minutes in all.

## Left to the builder

- File layout, code structure, figures, wording.
- Speedups of the port that keep its results bit-identical: one boundary matrix per complex for the four inclusions, `nogil` kernels.
- The implementation of the exact weighted-bar inner products and of the scoring.
- The worker scheme.

## Builder check (2026-11-20)

A fresh reviewer (model opus) read this design as a builder would and asked
two questions. Both were brought to Simon and ruled the same day.

1. **Prediction strength cannot tell clusters from elongation.**
   - The reviewer's scratch simulation of the drafted rule on one cluster-free Gaussian:
     - PCA-like variances (1/i or 0.8^i, 20 dimensions) gave PS(2) = 0.86–0.97 at 1,000–10,000 points per half, and k = 4 at 10,000 points;
     - a level-1-like cloud (5 features, 22,500 points per half) gave PS 0.99 / 0.89 / 0.85 at k = 2 / 4 / 8, so the rule would pick 8 compartments in a cloud with no clusters;
     - only a perfectly round cloud gave k = 1.
   - Claude's independent check:
     - PS(2) = 0.98 for variances 1/i, 4,000 points per half.
     - Calibrating PS against matched Gaussians separates poorly: 0.99 for real clusters 4 SD apart against a null maximum of 0.98.
     - Hence the gap statistic of step 10 (Simon).
2. **The scaling of the residuals had not been ruled.**
   - The reviewer ran the port on 80 real 150 µm balls of the spike slides (R = 19, faces to dimension 2). Median scaled residuals were about 1.
   - The case "s = 0 but the observed curve differs" came up in up to 2 of 40 balls per block, all in image H1. The draft would have zeroed it.
   - Hence the moderated spread of step 5 (Simon).

## Appendix: timing spike (2026-11-20, throwaway)

- **Setup.**
  - Scratch venv: Python 3.13, chalc 15.0.1, phimaker 0.5.0, gudhi 3.13.0, numba 0.68.1.
  - The port copied from chalc-scaling `8f370c6`, with one boundary matrix per complex for all four inclusions.
  - Two development slides: GTEX-YJ89-1426 (healthy, 220,475 cells; shares epithelial 0.783, stromal 0.141, immune 0.053, other 0.023) and GTEX-NPJ8-0726 (cirrhosis, 203,167 cells; 0.361, 0.381, 0.098, 0.160).
- **Checks on 12 balls** (2 per slide and radius, δ = 100/150/200 µm, 44–440 cells):
  - six-packs (class t → all, every class present) equal to chalc 15.0.1's `SubChromaticInclusion`, all modules and degrees 0–1, by bar count and bottleneck distance within 1e-9 of the scale;
  - the three Betti identities with zero violations at every endpoint;
  - faces to dimension 2 giving the same six-packs as faces to dimension 3 (about 40% less six-pack time).
- **Timing, R = 19, faces to dimension 2** (12 balls per slide and radius; medians):

  | δ | slide | cells per ball (range) | per complex | per centre (1 + 19) | max per centre | phimaker share | Delaunay share |
  |---|---|---|---|---|---|---|---|
  | 100 µm | healthy | 66 (47–112) | 12 ms | 0.25 s | 0.71 s | 0.60 | 0.23 |
  | 100 µm | cirrhosis | 96 (58–190) | 28 ms | 0.56 s | 1.32 s | 0.54 | 0.34 |
  | 150 µm | healthy | 137 (97–223) | 36 ms | 0.77 s | 1.55 s | 0.59 | 0.29 |
  | 150 µm | cirrhosis | 198 (125–304) | 68 ms | 1.38 s | 2.38 s | 0.51 | 0.39 |
  | 200 µm | healthy | 242 (177–346) | 70 ms | 1.45 s | 2.60 s | 0.59 | 0.30 |
  | 200 µm | cirrhosis | 319 (244–500) | 119 ms | 2.51 s | 4.17 s | 0.52 | 0.39 |

- **Threads,** δ = 150 µm, 48 centres per slide, 8 threads: 0.24 s per centre wall clock against 0.54 s serial (healthy), and 0.63 s against 1.48 s (cirrhosis). That is 2.2–2.3×, so the run uses processes.
- **Peak RSS:** 0.28 GB serial, 0.42 GB with 8 threads.
