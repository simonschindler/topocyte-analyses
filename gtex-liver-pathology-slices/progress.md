Ledger: Betti-curve fPCA of density slices on the 594-slide GTEx liver
cohort, coloured by pathology. Started 2026-11-18 (session in worktree
nervous-shamir-3e400e). Untracked AI artifact; never commit it.

Task 1 (Simon, 2026-11-18): the cohort in
~/data/gtex_liver_cells/gtex_liver_cells (gtex_liver_cells.h5ad, 126,219,124
cells on 594 slides; liver_cohort.csv, 610 rows; 16 rows have no cells, among
them all 7 with no pathology note). Assign every slide one pathology group;
estimate 10 density slice levels from the pooled 10-NN density quantile
curve; per slice fit fPCA on the Betti curves and plot the FVE and the first
four components coloured by pathology.
Ruling (Simon, from a batched multiple-choice): one group per slide by
severity - cirrhosis > fibrosis > inflammation/hepatitis > necrosis >
steatosis > congestion - over the "Pathology Categories" column; the curated
label "healthy" (66 slides: note reports no finding, no_abnormalities,
clean_specimens) is its own group; the 34 slides with no category (finding
only in the free-text note, 25, or a preservation warning, 9) stay in the
fPCA as "unclassified", drawn grey. Category members, fixed by Claude when
the option was offered: fibrosis = {fibrosis, scarring, sclerotic};
inflammation/hepatitis = {inflammation, hepatitis}; necrosis = {necrosis,
infarction, ischemic_changes}; congestion = {congestion, hemorrhage}; every
other category (atrophy, nodularity, hyalinization, hyperplasia, pigment)
occurs only beside one of these. Expected sizes: steatosis 202, congestion
155, healthy 66, cirrhosis 57, fibrosis 50, unclassified 34, necrosis 18,
inflammation/hepatitis 12.
Ruling (Simon): four Betti-curve fPCA models per slice, H0 and H1, raw and
cosine, as in the 123-slide study.
Ruling (Simon): per slice and model, an FVE panel and a scatter for every
pair among PC1-PC4.
Ruling (Simon): a self-contained HTML report in this untracked folder,
copied to the main checkout's artifacts/.
Ruling (Claude, defaults carried over from the 123-slide study): density
rho = 10 / (pi r10^2) per cell from the exact 10-NN distance within its
slide, 0.4942 um/px assumed for every slide (units only); slice levels are
the pooled deciles q = 0, 0.1, ..., 0.9 (slice 0 = all cells), the same
absolute thresholds for every slide; a slice is the alpha PH (max_dim 1) of
the cells with rho >= threshold, per slide, with topocyte's existing
pipeline; every Betti curve is exact from the bar endpoints (no grid). Cost
if wrong: units only, the slicing and the fPCA do not depend on the pixel
size.
Ruling (Claude): nine components per model, fitted matrix-free (topocyte's
eigen_solver "auto" picks ARPACK for 594 curves and fewer than ten
components); the FVE is exact, each eigenvalue over the trace of the centred
Gram matrix. Cost if wrong: the FVE plot stops at nine components.
Ruling (Claude): the 16 GB machine has 29 GB free disk, so per-slice results
keep scores, FVE and kept-cell counts only, not the diagrams; one process per
slice, sequentially.
Ruling (Claude, 2026-11-18, before any slice ran in full): the study's home is
the topocyte-analyses repo (set up earlier today: new studies go there as
their own folder, not in topocyte's artifacts/). It is built in this
worktree's ignored artifacts/ and copied there at the end, uncommitted, in
that repo's layout (README, scripts, ledger, verifier, builder; outputs
ignored).
Ruling (Claude, after slice 0 under tc.tl ran out of room): slice 0 through
tc.tl.persistent_homology would hold an AnnData of 126 million cells, whose
string index of 126 million names exceeds this 16 GB machine. Each slice now
calls what tc.tl runs, directly: tc.ph.compute_persistent_homology per slide
(the slide's kept cells in row order, as float64, no unit scaling, as
tc.tl.persistent_homology does) in a thread pool, and tc.ph.FPCA with
tc.tl.fpca's defaults on the 594 results. Cost if wrong: none for the
numbers; the AnnData is not saved.

Task 2 (Simon, 2026-11-18, "can you add a umap in every fpca score space"):
a UMAP embedding of every slice's and model's fPCA score space, coloured like
the scatters. Rulings fixed before any compute:
Ruling: the score space is the nine-component score matrix of one model at
one slice (594 x 9), embedded as fitted - Euclidean distance on the scores,
the fPCA's own L2 geometry, so PC1 weighs most - with umap-learn 0.5.12 at
its defaults (n_neighbors 15, min_dist 0.1, 2 dimensions) and random_state 0,
which makes the run deterministic. A separate script (umap_scores.py) writes
results/umap.npz; the report only draws it. Cost if wrong: standardized
scores would weigh all nine components alike and show the small components'
structure more; the raw-score embedding follows what the fPCA itself
measures.
Ruling: each model's figure gets the UMAP as a seventh column, one panel per
facet row, the same colours and grey context; the caption says that UMAP
distances between clusters, and its axes, carry no units or scale.
Ruling (Claude, during the run, memory only): slices 6-9 peaked at 3.7-6.7 GB,
and slice 5 already held 6.2 GB while its alpha complexes were being built,
ten slides at a time. From slice 4 on (above 70 million cells) the script
builds five at a time, fits the two H1 models first, and drops the H1
diagrams before the two H0 fits. Cost if wrong: none for the numbers - the
same diagrams and fits, in another order and with fewer threads.
Task 1: complete. density 49.6 s (126,219,124 cells, peak 3.3 GB); slices 9
down to 0 in 51-640 s each (PH 19-312 s; every fit matrix-free, 5-90 s),
peak memory 3.7-7.7 GB; no slide skipped at any level.
Results:
- groups as ruled; 196 of the 594 slides carry more than one category.
- slice levels, the pooled deciles of rho: 0.2, 1202.8, 1454.5, 1665.3,
  1871.1, 2090.7, 2344.6, 2665.4, 3133.6, 4059.6 cells/mm^2; every slide keeps
  at least 496 cells at every level (13,120 at q = 0.2, 496 at q = 0.9).
- fractions of variance, PC1 / PC1-4 over the ten slices: H0 raw
  0.843-0.955 / at least 0.995; H1 raw 0.594-0.895 / at least 0.943; H0
  cosine 0.625-0.862 / at least 0.970; H1 cosine 0.435-0.800 / at least
  0.867. The H1 cosine model spreads its variance most, more so the denser
  the slice.
Task 2: complete. umap_scores.py ran in 21.9 s, 40 embeddings, and is
deterministic (a rerun of slice 9's H1 cosine embedding is identical).
Report: report.html (6.3 MB; the density curve, 10 scree plots and 40
seven-column figures; PNGs quantized to 128 colours), HTML parser clean.
Figures checked as PNGs (slices 0, 5 and 9), the text read back; not viewed
in a browser.
verify_results.py: ALL CHECKS PASSED (340 s, peak 5.7 GB) - groups by a
second implementation of the rule, rho for one slide through scikit-learn,
deciles and curve, kept counts and H0 bars, score structure against the FVE,
the 40 embeddings with one rerun identical, and slice 9's four models refitted
with the dense solver from recomputed diagrams, matching the saved fractions
and scores. Copied to topocyte-analyses/gtex-liver-pathology-slices with a row
in its README's Studies table; nothing committed.

Task 3 (Simon, 2026-11-18, "yes add it here and also do the One family
across all slices version as well"): combine the per-slice fPCA scores into
one score space. A scratch check on the saved scores came first: the 40
blocks (10 slices x 4 models, 9 scores each, 360 columns) differ in total
variance by up to 13 orders of magnitude (H0 raw up to 1e12, cosine below
0.5); concatenated as they are, PC1 takes 0.83 of the variance and puts 0.89
of its loading on the H0 raw blocks; with every column z-scored the spectrum
is flat (PC1 0.11, 20 components for 90%), the smallest components counting
as much as the first. Rulings fixed before the run:
Ruling: multiple factor analysis (MFA) weighting - each block divided by the
square root of the variance of its first score, so every block's leading
component has variance 1 and its internal proportions stay - then PCA (SVD)
of the centred concatenation. PCA of that concatenation is kernel PCA of the
sum of the blocks' centred Gram matrices, each divided by its first
eigenvalue and cut to its nine components (at least 0.867 of every block's
variance): one fPCA of a curve per slide that spans the slices. Cost if
wrong: an exact version would sum the full Gram matrices, which needs the
diagrams again (not saved); the cut drops at most 0.133 of a block.
Ruling: two kinds of space - "all", the 40 blocks; and one per model family
(H0 raw, H1 raw, H0 cosine, H1 cosine), its 10 slices' blocks (90
columns). Five spaces.
Ruling: per space, the FVE of every combined component, the scores of the
first 20, a UMAP of the weighted concatenation (Euclidean, the same as on
all its components; Task 2's parameters and seed), and, in the report, the
correlation of the first components with each slide's log10 median density
and log10 cell count, beside the 7-column figure of Task 2 (pairs of PC1-4
and the UMAP, three facet rows). Combined components are signed so that the
largest-magnitude entry of each left singular vector is positive. Script:
combined_spaces.py writing results/combined.npz; the report draws it.
Task 3: complete. combined_spaces.py ran in 5.7 s; verify_results.py checks
it against scikit-learn's PCA of the weighted blocks (fractions and scores,
sign aside), the weights, the loading shares and one UMAP rerun.
Results (descriptive, no test):
- spaces (PC1 / PC1-4 / components for 90%): all 40 blocks 0.575 / 0.890 /
  5; H0 raw 0.813 / 0.995 / 2; H1 raw 0.706 / 0.964 / 3; H0 cosine 0.711 /
  0.961 / 3; H1 cosine 0.527 / 0.897 / 5.
- PC1 is density in every space: |r| with log10 median density 0.77-0.95.
- PC2 hardly follows the median density (|r| <= 0.30) but follows the share
  of a slide's cells in the densest decile (|r| 0.45-0.58); part of it is the
  arch of PC1 (R^2 of PC2 on PC1 and PC1^2: 0.14-0.57), seen as a U in the
  PC1-PC2 panels.
- cirrhosis vs healthy along PC2: median gap 1.00-2.07 SD (largest in the H0
  cosine space), 1.24-1.97 SD with the arch removed; the remainder still
  follows the densest-decile share (|r| 0.36-0.63).
Caught before the copy: figure_scatter's new title parameter was shadowed by
the facet loop variable, which put the last facet's name over every figure
of that build; fixed, and the figures re-checked as PNGs (slice 6 H1 raw,
the all and H0 cosine spaces, the scree).
Report: section 5 (method, results, table, the five spaces' scree, five
seven-column figures with group-median tables), 7.1 MB, parser clean.
verify_results.py after Task 3: ALL CHECKS PASSED (324 s), the combined spaces included.

Task 4 (Simon, 2026-11-18, "Yes, check here" to "Check whether PC2's
cirrhosis gap survives removing the densest-decile share?"). Rulings fixed
before any compute:
Ruling: per combined space (the five of Task 3), PC2 is oriented so that
cirrhosis's median lies above healthy's; that orientation is kept for every
residual. Readout: the AUC of cirrhosis (57 slides) against healthy (66),
P(cirrhosis > healthy) over all pairs, with a 95% bootstrap interval
(4000 draws, each group resampled on its own, numpy default_rng(12345)),
and the median gap in standard deviations.
Ruling (primary): PC2 residualised on the densest-decile share (the share of
a slide's cells kept at slice 9) and its square, least squares over all 594
slides. The cirrhosis gap survives removing the share in a space if this
residual's AUC interval lies wholly above 0.5.
Ruling (secondary, stricter): PC2 residualised on the share and its square,
the arch (PC1 and PC1^2), log10 median density and log10 cell count. Also,
for scale: the AUC of the share alone, and of PC2 as fitted.
Cost if wrong: residualising on all slides and then bootstrapping only the
two groups treats the fit as fixed (six or seven coefficients from 594
slides, so the intervals are close but a little narrow); five spaces are
tested, so a single interval above 0.5 is weak evidence; and a nuisance
model that is too simple leaves density in the residual - the stricter
variant is there for that. Descriptive, not a classifier; nothing tuned.
Task 4: complete. pc2_density_check.py ran in 0.8 s; verify_results.py checks
the residuals against scikit-learn's regression, every AUC against
roc_auc_score and one bootstrap interval.
Results (AUC of cirrhosis vs healthy, 95% interval):
- the densest-decile share alone: 0.933 [0.883, 0.973], above PC2 as fitted
  in every space (0.742-0.918).
- share removed (the pre-registered rule): the gap survives in 2 of 5
  spaces - all 40 blocks 0.656 [0.554, 0.754], H0 cosine 0.761 [0.667,
  0.843]; not in H0 raw 0.423 [0.319, 0.527], H1 raw 0.499 [0.391, 0.602],
  H1 cosine 0.533 [0.430, 0.635].
- share, arch, log median density and log cell count removed: no space keeps
  any separation (0.395-0.509, every interval containing 0.5).
- scratch split, post-hoc, not saved and not in the report: in the two
  surviving spaces the remainder goes with median density or the PC1 arch
  (share + median density: AUC 0.330 all, 0.403 H0 cosine; share + arch:
  0.544, 0.476), not with cell count (share + log cells: 0.649, 0.748).
Conclusion written into the report: the separation along PC2 is the dense
tail of the density distribution, and what is left once that is removed
goes with the rest of density.
verify_results.py after Task 4: ALL CHECKS PASSED (361 s), the PC2 check included.
