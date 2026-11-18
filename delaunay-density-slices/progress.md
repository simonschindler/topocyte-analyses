# SDD ledger — plan: function-Delaunay density slices (brief pasted 2026-11-16, no plan file)

Ruling: the brief is the plan; it has no file. The ledger lives here, beside the
deliverables, rather than in `.superpowers/sdd/`, because the brief says every
output goes under `artifacts/delaunay-density-slices/`.
Cost if wrong: none — artifacts/ is gitignored and the ledger is process only.

Ruling: no commits, no merge. The brief says "Commit nothing" and "No change to
topocyte itself", so the finishing-a-development-branch step does not apply.
Cost if wrong: work stays uncommitted; Simon asked for exactly that.

Pre-flight interface scan (settled calls in the brief against the checkout at
7db4d26):
- `tc.tl.persistent_homology(groupby=...)` computes groups in
  `adata.obs[groupby].unique()` order; subsetting rows in store order keeps the
  tutorial-04 slide order. `microns_per_unit` is recorded only, changes no bar.
- `tc.tl.fpca` reads the grouped PH back itself; `n_components=None` keeps all
  123 components via the dense solver. Scores are in `uns[key]["scores"]` with
  row order `uns[key]["group_ids"]`.
- `tc.tl.fpca_classify` stores held-out `cv.accuracy`, `cv.confusion_matrix`,
  `y` and the fitted classifier under `uns[fpca_key]["classifiers"][name]`; it
  mirrors `_logistic(10, 5, "accuracy", 1000, seed)` and
  `StratifiedKFold(5, shuffle=True, random_state=seed)`.
- `tc.pl.fpca(color=...)` needs `obs[color]` constant within each group; our
  per-slice AnnData carries label per cell, constant by construction.
- `fpca_classify` rejects numeric label columns; we build `label` categorical.
- Store: zarr 3.2.1, obs/slide_id and obs/label are codes+categories,
  obsm/spatial is (30,836,186, 2) float32. Read with mode="r".

Task 1: verification contract (RED first) — verify_results.py encodes the
brief's checks (deciles, kept fractions, 123 slides, slice-0 tutorial
accuracies). It must fail before any result exists.

Task 1: RED confirmed (missing density.npz), then density computed.
Ruling: the brief prints deciles as whole cells and 1693.5 rounds either way;
verify accepts ±1 cell per decile instead of exact string/round equality.
Cost if wrong: a genuine 1-cell drift passes; kept fractions still pin the
slice membership.
Ruling: density.npz stores string arrays as Unicode, not Python objects, so
every result loads with allow_pickle=False. The first slice-0 run died on an
object array before PH; fixed and restarted. Cost if wrong: none.

Task 2: slice computation — slice 0 first, gate on the tutorial-04 reproduction
before the other nine slices run.
Task 2: slice 0 PASSED the gate: seed-0 accuracies 0.886 / 0.780 / 0.886 /
0.886 (H0 raw, H1 raw, H0 cosine, H1 cosine), exactly tutorial 04. 603 s total
(PH 85 s, 4 fPCA fits 44-73 s, 80 classifications 261 s). Control 0.537 ± 0,
the majority class, as expected from a constant kept fraction at slice 0.
Slices 1-9 launched sequentially.
Task 2: complete. All ten npz files written; verify_results.py passes every
check (deciles, cell/slide counts, kept fractions, min 212 cells at q0.9,
slice-0 tutorial match). Slice 1's H1-cosine fPCA took 1062 s during a
system-wide swap/load spike; slices 2-9 took 49-297 s each.
Task 3: report builder written, smoke-tested on duplicated slice 0, density
figure split into full-range + zoom panels, kept-fraction transpose fixed.
Slice 0 rerun in progress only to regenerate its figure title ("all cells").
Task 3: first real report.html built (3.1 MB, 13 embedded PNGs, HTML parser
clean). Verdict text computes "No": no slice beats both bars. Grammar fixed
("1 of the 36 combinations beats").
Ruling: abandoned the slice-0 figure rerun after the PH and three fPCA fits;
it was recomputing only a title. The saved slice-0 figure's title rounds the
q0 threshold 0.4 to 0; the HTML caption now states that explicitly, and the
saved numbers are unchanged. Cost if wrong: only the title text is rounded,
not any result.
Task 3: complete. Final report.html (3.1 MB, 13 embedded PNGs) rebuilt from
the saved results; verify_results.py passes every check. Verdict: No - no
slice beats both slice 0 and its density-only control by more than the noise.
No topocyte file touched, nothing committed.
Ruling: the report's noise bar is 2 * sqrt(SE_k^2 + SE_base^2 + SD_k^2 +
SD_base^2), as the brief asks for twice the binomial SE plus the seed spread;
conservative because it uses the seed SD, not its standard error.
Cost if wrong: the bar is wider, so the verdict is conservative.
Ruling: the density figure draws the empirical survival function at 800
log-spaced display points; no topological quantity is sampled - Betti curves,
fPCA and the deciles use exact bar endpoints and every cell. Cost if wrong:
the caption states the display sampling explicitly.
Ruling: scikit-learn emits ConvergenceWarnings (lbfgs status 2) inside the
inner CV folds; these come from the tutorial-04 estimator on separable folds,
not from a changed call, and the held-out accuracies reproduce tutorial 04.
Leave the estimator untouched. Cost if wrong: a fold's C may be at the edge of
the grid, but both the baseline and every slice use the same estimator.
Ruling: slice 0's figure title rounded the q=0 threshold (0.4043) to
"rho >= 0", which reads as an absolute threshold. Fixed the title to say
"all cells" for slice 0; the already-computed slice-0 figure predates the fix
and will be regenerated by rerunning slice 0 after slices 1-9 (same npz,
same numbers). Cost if wrong: 10 min of recomputation; ordering unchanged.

Task 4 (Simon, after the first report): concatenate the first 4 fPCA scores
of every slice and classify on that space.
Ruling: his wording admits two readings — one 160-feature vector pooling all
four families, or one 40-feature model per family. Chose per family: the four
families' scores differ by orders of magnitude (raw vs cosine, H0 vs H1
variances), and fpca_classify does not standardise, so a pooled vector would
be dominated by whichever family has the largest scale. 40 features per
family, 20 seeds, same estimator and folds; control concatenates the ten
kept fractions (10 features). Cost if wrong: the pooled reading is one more
640-way run over saved scores; nothing needs recomputing from PH.
Task 4: complete. combined_classify.py wrote results/combined.npz from the
saved per-slice scores (no PH or fPCA rerun). verify_results.py extended with
combined checks (40-feature blocks equal each slice's first 4 scores, control
columns equal kept fractions, shapes/sums) and passes. Report section 6 added
(dumbbell figure + table + verdict paragraph), rebuilt report.html 3.12 MB,
14 embedded PNGs, parser clean.
Result: no combined model beats both slice 0 and the combined density-only
control by more than the noise. Highest mean H0 cosine 0.896 ± 0.015 vs
control 0.883 ± 0.011 and its own slice-0 baseline 0.895; largest lift H1 raw
+0.107 over its baseline but 0.871, below the control. Overall verdict stays
No.

Task 5 (Simon, follow-up): how much PH adds beyond density. Three tests, all
from saved scores; no PH or fPCA rerun.
Ruling: the rich density baseline is ten features per slice — kept fraction,
log10 rho quantiles 0.05/0.10/0.25/0.50/0.75/0.90/0.95 among the kept cells,
and mean and sd of log10 rho. This answers the "kept-fraction control is too
weak" objection and covers both density notions: cross-slide level (fraction)
and within-slide density field (quantiles). A stronger arm A makes it harder
for PH to show an increment, so a positive result is decisive and a null is
conservative against PH. Cost if wrong: arm A carries 10 features vs 4 PH
scores, a dimensional disadvantage for the PH arms.
Ruling: no feature standardisation; the protocol's fpca_classify does not
standardise, and log10 rho features have modest spread (sd 0.07-0.5) against
fPCA score scales. Cost if wrong: the penalty treats large-scale features
differently; all arms would share the bias only partly.
Ruling: residualisation of the four scores on the density features is fitted
inside each outer training fold and applied to the test fold. No test-fold
information enters the transform. Cost if wrong: slightly noisier residuals
than a global fit; correctness of the comparison unaffected.
Ruling: uncertainty is a paired bootstrap over the 123 slides (4000 draws,
seed 12345) of per-slide correctness averaged over the 20 seeds. Seeds only
shuffle folds; they do not sample slides, so the earlier seed-based noise bar
is fold noise, not generalisation noise. Cost if wrong: none - the bootstrap
is reported alongside the seed SDs, not replacing them in the existing
sections.
Ruling: arms B = rich + first 4 scores; R = first 4 scores residualised on
rich; P1 = score 1 only; P24 = scores 2-4 only; each per slice and for the
ten-slice concatenation (combined.npz scores, already verified equal to the
per-slice blocks).
Task 5: verification contract extended (density_adds.npz: rich blocks equal
the slices and their kept fractions, quantiles monotone, accuracy equals mean
correctness, CI ordered) and confirmed RED before the compute. Smoke tests:
build_rich 2.2 s, monotone quantiles, slice 0 fractions all 1; residual CV
runs.
Ruling: the verification's own slice-0 fraction check initially read column 0
of each rich block; the kept fraction is feature 9 (quantiles 0-6, mean 7,
sd 8). Fixed the check, not the data - the per-slice columns were already
asserted equal to the saved kept_fraction. Cost if wrong: none.
Task 5: complete. Compute 1249 s; verify_results.py ALL CHECKS PASSED;
report.html rebuilt (3.19 MB, 15 embedded PNGs, parser and base64 clean).
Results, combined ten-slice models, mean ± SD over 20 seeds:
- density arm alone 0.909 ± 0.014 - above every PH-only model (0.871-0.896).
- density + first 4 scores: 0.869-0.911; best delta +0.002 (H0 cosine, 95%
  CI -0.022 to +0.028), i.e. no family adds measurable accuracy.
- scores residualised on density: 0.524-0.534, majority-class level; all 40
  per-slice residual CIs lie below the density arm.
- PC1 alone 0.824-0.911; PCs 2-4 alone 0.848-0.893, 0 of 40 intervals above
  the density arm and 36 below.
- density features explain median R2 0.86 of PC1's variance, 0.36 of PCs 2-4.
- per-slice nested arm: 3 of 40 intervals wholly above zero, 3 wholly below,
  best +0.045 (slice 6, H0 cosine) - not repeated in the combined model.
Conclusion written into the report: in this cohort PH acts as a density
proxy; it adds no measurable accuracy beyond a ten-feature density summary.

Task 6 (Simon, 2026-11-17, "Yes, fix it"): correct section 7 and the verdict.
A check of the saved npz found three flaws in Task 5: (1) the combined
residual arm is degenerate - 100 density features + intercept against about
98 training slides, so the regression interpolates and its 0.52-0.53 is an
artifact; (2) the residual arms were judged against the density arm, but
their null is chance; (3) slide size confounds: cirrhosis slides have more
cells (median 275,184 vs 183,952) and log10 n cells alone classifies at
0.717, yet the density arm did not contain it.
Ruling: residual arms per slice only (11 density features + intercept against
about 98 training slides). residual_cv_outputs raises when features + 1 reach
half the cohort, so the combined residual arm cannot come back. Cost if
wrong: no combined residual test; the per-slice arms answer the same question
slice by slice.
Ruling: the residual null is chance - the training fold's majority class for
accuracy (0.537) and its prevalence for log-loss (0.691) - computed per
held-out slide on the same folds. Cost if wrong: none; it is the classifier
that knows only the training labels.
Ruling: log10 n cells joins the density arm (11 features per slice, 101
combined), and slide size alone runs as its own arm S. A stronger density arm
is conservative against PH. Cost if wrong: raw scores lose the size signal
they carried; cosine scores never had it.
Ruling: out-of-fold probabilities give log-loss and AUC beside accuracy. C is
still tuned for accuracy (the protocol's _logistic), so log-loss judges
untuned probabilities; the report says so and keeps accuracy as the primary
metric. Cost if wrong: log-loss overstates the penalty of the larger arms.
Ruling: one shared bootstrap index matrix (default_rng(12345), 4000 x 123)
for every interval, so all differences are paired on the same resamples.
Cost if wrong: none.
Ruling: seeds run in parallel with joblib; compute 280.7 s against 1249 s.
Ruling: the 2026-11-16 npz, json, log and report move to superseded/, and the
verification checks that the score-only arms reproduce them exactly.
Ruling: every number in section 7 is read from the results files, and every
pattern claim in the prose is an assertion in build_report.py. Residual
claims say "linear": OLS residualisation shows only that the scores are not a
linear function of the density summary. Cost if wrong: none.
Task 6: verification contract extended (density arm = 10 features + slide
size, per-slice residual arms only, probabilities, log-loss and AUC
consistent, chance baselines exact, bootstrap CIs ordered) and confirmed RED
before the compute; GREEN after it, ALL CHECKS PASSED, including the exact
reproduction of the score-only arms.
Task 6: complete. report.html rebuilt (3.4 MB, 16 embedded PNGs, parser
clean): section 7 rewritten with a correction note, two new figures, tables
7a-7c, and a new verdict paragraph; section 8 updated. Figures checked as
PNGs; the file:// browser preview was refused.
Results, mean ± SD over 20 seeds, deltas with 95% paired bootstrap intervals:
- chance 0.537; slide size alone 0.717 ± 0.013 (AUC 0.775); density arm
  0.910 ± 0.011, log-loss 0.269, AUC 0.960.
- combined, density + four scores of every slice against density: H0 raw
  -0.015 [-0.057, +0.026], H1 raw -0.039 [-0.078, +0.002], H0 cosine +0.006
  [-0.009, +0.025], H1 cosine -0.024 [-0.051, +0.003]; no interval above
  zero; log-loss worse in all four (3 of 4 intervals wholly below zero), AUC
  -0.051 to -0.010.
- combined, PCs 2-4 alone: -0.061 to -0.017 against density, 2 of 4
  intervals below; H0 cosine PC1 alone 0.911 ± 0.018, level with density.
- per-slice density arms 0.842-0.921 (best slice 1). Nested: all 20 point
  estimates positive at slices 4-8, 8 of 40 intervals above zero (all at
  slices 4-8), 5 below (slices 0-1); gain vs density-arm accuracy r = -0.98;
  best middle-slice nested 0.915 (slice 5, H0 cosine) < 0.921 slice-1 density
  arm. Nested log-loss: none above zero, 10 below.
- per-slice residual against chance: 11 of 40 intervals above (7 of the 8 at
  slices 8-9; H1 raw at slices 0-2, H1 cosine at slice 0), none below;
  accuracy 0.530-0.644 at slices 0-7, 0.637-0.744 at slices 8-9; best AUC
  0.774 (slice 9, H0 cosine). Residual log-loss: no interval excludes zero.
- density features explain median R2 0.88 of PC1, 0.49 of PCs 2-4 (0.86 and
  0.36 without slide size).
Conclusion written into the report: PH is not a linear proxy for the density
summary - residual signal survives, mostly at slices 8-9 - but it adds
nothing usable beyond density. The combined model gains no accuracy and loses
log-loss and AUC; per slice, the scores lift the weaker density arms of
slices 4-8 only to the level the slice 0-1 density arms reach alone. The
overall verdict stays No.

Task 7 (Simon, 2026-11-17, "Run it here"): topology with density removed -
the H1 death/birth ratio, on all cells and on slices 6-9, nested against the
density arm. Rulings fixed before any compute:
Ruling: per H1 bar, pi = sqrt(death / birth), the ratio of death and birth
radii (alpha filtration values are squared radii; alpha diagrams equal Cech
diagrams). pi does not change when the cells are scaled, so a uniform change
of density, and the assumed 0.4942 um/px, leave it alone; for bars of noise
its distribution does not depend on the sampling density (Bobrowski &
Skraba, Sci. Rep. 2023). Cost if wrong: tissue is not iid noise, so the
ratio may still track density; the R2 check below measures how much.
Ruling: each slide's curve is the survival function S(t) = number of H1 bars
with log pi > t, t >= 0, which is the Betti curve of the bars [0, log pi).
topocyte.ph.FPCA(dim=1) fits it exactly from the bar endpoints; no grid. The
kernel is checked against the closed form sum_ij min(v_i, v_j) for every
pair of slides. Cost if wrong: none for exactness; the L2 geometry of S
weights each bar by its log pi, which favours long-lived cycles by design.
Ruling: both protocol normalizations. "none" keeps the bar count, which
scales with slide size; "cosine" removes it exactly (S = N F, so S/||S|| =
F/||F||). Cost if wrong: none; the density arm holds log10 cells anyway.
Ruling: slices 0, 6, 7, 8, 9. PH is recomputed as run_slices.py computes it
(nothing saved the bars). Gate: the H1 Betti-curve fPCA of the recomputed
diagrams must reproduce the saved first four scores of each slice, and the
recomputed density arm must equal density_adds.npz exactly.
Ruling: arms per slice and normalization under the protocol estimator, folds
and 20 seeds: Q = first four ratio scores alone; N = density arm of the slice
(10 features + log10 cells) + the four ratio scores. Intervals: N - A, Q -
chance, Q - A, in accuracy and log-loss, paired bootstrap with the shared
indices of density_adds.py. R2 of each ratio score on the density arm, next
to the Betti H1 values of section 7.
Ruling (decision rule, pre-registered): primary is slice 0, both
normalizations (2 comparisons). The ratio adds beyond density if a nested
accuracy interval lies wholly above zero. Slices 6-9 are secondary (8
comparisons) and exploratory. If it adds, the story is "topology beyond
density"; if not, topocyte presents the density control as part of the
method. Cost if wrong: a true gain below about 0.03 at n = 123 can go
undetected.
Task 7: complete. ratio_slices.py ran in 707.6 s, persistent homology
recomputed for slices 0 and 6-9. Gates held: the recomputed diagrams give
back the saved H1 Betti-curve scores of every slice bit for bit, the density
arm equals density_adds.npz exactly, and the fitted Gram matrix matches
sum_ij min(u_i, v_j) to 1.1e-14 (relative) for every pair of slides.
Results, mean ± SD over 20 seeds, deltas with 95% paired bootstrap intervals:
- primary, slice 0: density arm 0.915 ± 0.014; density + four ratio scores
  0.925 ± 0.016 raw, 0.915 ± 0.013 cosine; delta +0.010 [-0.025, +0.049] and
  -0.000 [-0.004, +0.002]. No interval above zero: by the pre-registered rule
  the ratio does not add beyond density. Log-loss reductions -0.026 [-0.194,
  +0.107] raw and +0.009 [+0.004, +0.014] cosine; log-loss is not the rule's
  metric, and the cosine gain does not survive the post-hoc check below.
- ratio alone, slice 0: 0.887 ± 0.012 raw (AUC 0.952), 0.815 ± 0.013 cosine
  (AUC 0.899); +0.351 and +0.279 above chance.
- secondary, slices 6-9: 1 of 8 nested intervals above zero, raw at slice 6,
  +0.045 [+0.007, +0.087]; it is the ratio-alone arm (both 0.887 ± 0.013).
  With the cosine scores the density arm's log-loss gets worse at slices 8
  and 9 (-0.008 [-0.017, -0.000], -0.018 [-0.028, -0.010]).
- density features explain median R2 0.87 of ratio PC1 and 0.34 of PCs 2-4
  (H1 Betti, same slices: 0.87 and 0.36); cosine PC1 0.85. Raw PC1 is the
  bar count (>= 99.8% of the variance, |r| >= 0.998 with the count).
- tails: at slices 6-9 cirrhosis has more H1 bars per cell and longer-lived
  ones (class medians of q99, q999 and max of log pi higher; median bar
  within 0.003); at slice 0 the tails nearly coincide. Beyond q99 lies at
  most 1.2% of any slide's squared norm of S, so the scores follow the bulk.
Finding, after the results were seen: the nested test is scale-sensitive.
The protocol's L2-penalized estimator sees every feature on its own scale. At
slice 0 the density features have SD 0.046-0.199 across slides, the cosine
ratio scores 0.0008-0.009, the raw ratio scores 68-21,870. Weight share of
the scores (sum |coef_j| SD_j over the score columns over the same sum over
all columns, training-fold SD): cosine ratio at slice 0 median 0.004, max
0.019, so the cosine nested arm is the density arm; raw ratio at slices 6-9
median 1.000, inner CV picks C <= 1e-3 in 25-45% of folds, so the raw nested
arm is the ratio-alone arm; section 7's raw H1 Betti scores (SD 13,420 to
258,451) median 0.82-0.91 at slices 6-9. The pre-registered test was weaker
than intended, and section 7's nested arms share the problem.
Ruling (post-hoc, not pre-registered, chosen after the results were seen):
scaled_nested.py refits the density arm and every nested arm - the ratio and
section 7's H1 Betti scores, both normalizations, slices 0 and 6-9 - with a
StandardScaler fitted on the outer training fold in front of the same
estimator, with the same folds, seeds and bootstrap indices (121.1 s). Gate:
without the scaler every refit equals the saved nested arm, seed for seed.
The pre-registered result stays primary; the report labels the check
post-hoc. Cost if wrong: a gain seen only after standardizing rests on a
choice made after the results; the report calls it a lead, not a finding.
Results, post-hoc, standardized nested arm against standardized density arm:
- standardizing the density arm alone moves it by -0.017 to +0.014 (no
  interval excludes zero); the scores now carry a median 0.05-0.40 of the
  weight.
- slice 0: ratio +0.016 [-0.016, +0.052] raw, +0.012 [-0.017, +0.042] cosine;
  H1 Betti -0.009 [-0.036, +0.019] raw, -0.001 [-0.020, +0.016] cosine; no
  interval above zero in accuracy or log-loss. The protocol's cosine log-loss
  gain becomes -0.015 [-0.179, +0.132].
- slices 6-9: 5 of 16 accuracy intervals above zero - ratio raw at slice 6
  (+0.068 [+0.029, +0.110]), Betti raw at slices 6, 7, 8 (+0.056, +0.041,
  +0.044) and Betti cosine at slice 8 (+0.042 [+0.009, +0.079]). No log-loss
  interval above zero; the Betti gains' mean log-loss gets worse (-0.098 to
  -0.310). Cosine ratio worse at slice 9 in accuracy (-0.012 [-0.023,
  -0.002]) and log-loss (-0.030 [-0.047, -0.011]).
- scratch check, not saved and not in the report: the protocol's cosine
  log-loss gain at slice 0 comes mostly from the choice of C. In 21 of 100
  fits the inner CV picks a larger C once the scores are added; those fits
  reduce log-loss by 0.0275 on average, the 79 with the same C by 0.0035.
Conclusion written into the report: by the pre-registered rule the ratio
does not add beyond density, so by Task 7's rule topocyte presents the
density control as part of the method rather than "topology beyond density".
Post-hoc, standardized H1 topology at slices 6-8 gains accuracy but no
log-loss: a lead for a pre-registered test. Section 7's other nested arms
(H0, slices 1-5, the combined models) share the scale problem and were not
refitted.
Task 7: report section 8 written (design, results table and figure, the
post-hoc subsection with figure and table, R2 table, tails figure); section
7 gains a note on feature scale; the verdict becomes section 9 and cites
section 8. report.html rebuilt (3.9 MB, 19 embedded PNGs, parser clean);
verify_results.py also checks ratio_slices and scaled_nested, ALL CHECKS
PASSED. Figures checked as PNGs and the page as text; not viewed in a
browser. Nothing committed (artifacts are untracked).

Task 8 (Simon, 2026-11-17, "Yes, refit here"): refit section 7's nested arms
with standardized features. Post-hoc, like Task 7's check: chosen after the
results were seen. Rulings fixed before any compute:
Ruling: scope is every nested arm of section 7 - the slice's density arm plus
its first four scores, for the four families (H0/H1 x raw/cosine) at slices
0-9, and the four combined models (101 density features + all 40 scores) -
and the eleven density arms, each with a StandardScaler fitted on the outer
training fold in front of the protocol estimator; same folds, seeds and
bootstrap indices (scaled_section7.py, reusing scaled_nested.fit_folds). The
ten H1 arms at slices 0 and 6-9 that scaled_nested.py refitted are refitted
again. Cost if wrong: none; 99 arm fits from saved scores.
Ruling: gates - refitted unscaled, every nested arm equals density_adds.npz's
B arm seed for seed; standardized, the overlapping H1 arms and density arms
equal scaled_nested.npz's.
Ruling: single-block arms (density alone, slide size, residual, PC1, PCs 2-4,
scores alone) are not refitted. Features sharing one spread s fit at C as
standardized ones fit at C s^2, so for one block the inner CV only shifts its
choice of C, as long as the C it needs lies in the grid (1e-4 to 1e4); the
nested problem is two blocks whose spreads differ by 1e3 to 1e6, which no
single C serves. Cost if wrong: spreads within a block still tilt those
fits, and a block with a tiny spread (the cosine ratio scores, SD about
0.001) may need a C beyond the grid; not checked here.
Ruling (reading, fixed before the run): the combined models are the main
comparison, as in section 7's verdict; a family adds beyond density there if
its standardized accuracy interval lies wholly above zero. Per-slice
intervals are tallied as in section 7 (above and below zero, accuracy and
log-loss): 40 uncorrected comparisons. The report labels the whole check
post-hoc, and section 7's protocol results stay as written.
Task 8: complete. scaled_section7.py ran in 166.4 s and both gates held:
every unscaled refit equals section 7's B arm seed for seed, and the
standardized H1 and density arms at slices 0 and 6-9 equal scaled_nested's.
verify_results.py checks the file (_check_scaled_section7: RED before the
run, GREEN after; ALL CHECKS PASSED).
Results, standardized nested arm against standardized density arm:
- why it mattered: raw scores SD 13,420 to 1,164,503 against density SD
  0.0025-0.211. In the protocol fits the raw scores carry a median 0.34-1.00
  of the weight per slice (H0 raw 0.89-1.00), the cosine scores 0.10-0.83;
  the combined raw models put all of it on the scores, with C <= 1e-3 in
  82-89% of folds, and their log-loss is 1.096 (H0) and 1.911 (H1) worse
  than the density arm's. Standardized, the scores carry 0.10-0.47.
- standardizing a density arm alone moves it by -0.017 to +0.017, no
  interval excluding zero; combined density arm 0.917 ± 0.010 (protocol
  0.910).
- combined models (the main comparison, fixed before the run): delta -0.022
  to +0.002, no interval above zero, H1 cosine's below (-0.022 [-0.043,
  -0.002]); log-loss worse in all four (-0.209 to -0.078), H1 cosine's
  interval wholly below. By that reading no family adds beyond density.
- per slice: 10 of 40 accuracy intervals above zero, all at slices 5-8
  (protocol: 8, at slices 4-8); 1 below (H0 raw, slice 1; protocol: 5, at
  slices 0-1). Gain against standardized density-arm accuracy r = -0.95; the
  best nested arm at slices 5-8, 0.916 (H0 cosine, slice 5), stays below the
  0.923 of slice 1's standardized density arm. Log-loss: none above zero, 5
  below (slices 1-2).
- context for Task 7's post-hoc lead: its standardized gains sit where the
  slice's density arm is weak (0.847-0.859 at slices 6-8), and the best, H1
  ratio raw at slice 6 (0.924), only matches slice 1's density arm (0.923).
- scratch peek, not saved and not in the report: section 8's cosine
  ratio-alone arm picks the grid's top C (1e4) in 43% of slice-0 folds (14% at
  slice 9); standardized, slice 0 reaches 0.844 against the protocol's 0.815
  (slice 9: 0.740 against 0.743). That single-block arm is over-penalized at
  slice 0; it is outside this task and was not refitted.
Conclusion written into the report: standardizing leaves section 7's
conclusion where it was - the scores lift the weaker density arms of single
slices, and with every slice's density summary in the model they add
nothing. Section 8's post-hoc lead narrows to "at most a lead for a
pre-registered test".
Task 8: report rebuilt (4.1 MB, 20 embedded PNGs, parser clean). Section 7's
note from Task 7 is replaced by a post-hoc subsection (text, figure, table);
section 8's post-hoc text and the verdict cite it. New figure checked as a
PNG, text read back; not viewed in a browser. Nothing committed.

Task 9 (Simon, 2026-11-17, "Yes, check here"): check section 8's ratio-alone
arms for the same scale problem. Post-hoc, prompted by Task 8's scratch peek
(cosine ratio alone at slice 0 picks the grid's top C in 43% of its folds).
Rulings fixed before any compute:
Ruling: scope is the ten ratio-alone arms Q of section 8 (raw and cosine,
slices 0 and 6-9), refitted unscaled to record each fold's C (gate: equal to
ratio_slices.npz's Q arms seed for seed) and with a StandardScaler fitted on
the outer training fold in front of the same estimator; same folds, seeds and
bootstrap indices (scaled_ratio_alone.py, reusing scaled_nested.fit_folds).
Section 7's single-block arms are not part of it. Cost if wrong: none; 20
arm fits from saved scores.
Ruling: indicator - the share of fitted folds whose chosen C is the grid's
top, 1e4. LogisticRegressionCV keeps the first best C, so a fold at the top
preferred it to every smaller C and might have gone further.
Ruling (reading, fixed before the run): an arm counts as held back by the
scale if its standardized accuracy interval against the protocol arm lies
wholly above zero. Standardizing also evens out the four scores' spreads
within the block (6- to 459-fold apart), so a change mixes the grid's reach
with that reweighting; the top-C share is reported beside it. The
pre-registered result concerns the nested arms and does not change.
Report: a paragraph and a table in section 8's post-hoc subsection, and a
pointer from the primary result's ratio-alone sentence.
Task 9: complete. scaled_ratio_alone.py ran in 16.0 s and its gate held:
every unscaled refit equals ratio_slices.npz's Q arm seed for seed.
verify_results.py checks the file (_check_scaled_ratio_alone: RED before the
run, GREEN after; ALL CHECKS PASSED).
Results, standardized ratio-alone arm against the protocol arm:
- the protocol cosine arms pick the top C (1e4) in 14-91% of folds (43% at
  slice 0, 91% at slice 6); the raw arms and the standardized cosine arms
  never do.
- accuracy changes by -0.013 to +0.029, no interval above zero: by the
  reading fixed before the run, no arm is held back by its scale. Largest at
  slice 0 cosine, 0.815 -> 0.844, +0.029 [-0.019, +0.080]; slice 6 cosine
  goes the other way, 0.871 -> 0.858.
- log-loss improves only at slice 0 cosine (+0.075 [+0.032, +0.119]; AUC
  0.899 -> 0.930); no other interval excludes zero.
- correction to Task 8's scratch peek: "over-penalized at slice 0" holds for
  the fit's C (pinned at the top) and its probabilities, not its accuracy.
Conclusion written into the report: the ratio-alone accuracies of the
results table stand; at slice 0 the cosine arm's probabilities are sharper
with standardized scores; standardizing also evens out the spreads within
the block, so the change is not the grid's reach alone.
Task 9: report rebuilt (4.1 MB, 20 embedded PNGs): a paragraph and a table
in section 8's post-hoc subsection, a pointer from the primary result's
ratio-alone sentence, the reproduction list. Text read back; not viewed in
a browser. Nothing committed.

Task 10 (Simon, 2026-11-17, "Yes, check here"): check section 7's residual
and PC arms for the same scale problem. Post-hoc, after Task 9. Rulings fixed
before any compute:
Ruling: scope is the arms the report states - the 40 per-slice residual arms
R and the eight combined PC arms (PC1 of every slice, PCs 2-4 of every
slice, four families each). The per-slice PC arms are not in the report and
are not refitted. Each arm is refitted unscaled to record each fold's C
(gate: equal to density_adds.npz's arm seed for seed) and with a
StandardScaler fitted on the outer training fold in front of the same
estimator - for R on the training fold's residuals, after the same in-fold
residualisation. Same folds, seeds and bootstrap indices
(scaled_single_block.py). Cost if wrong: none; 96 arm fits from saved
scores.
Ruling: indicator as in Task 9 - the share of fitted folds whose chosen C is
the grid's top, 1e4.
Ruling (reading, fixed before the run): an arm is held back by its scale if
its standardized accuracy interval against the protocol arm lies wholly
above zero. Section 7's claims are re-counted on the standardized arms: the
residual intervals above and below chance (accuracy) and against the
prevalence (log-loss), and PCs 2-4 and PC1 against Task 8's standardized
combined density arm. The claim most at risk is "in log-loss none of the 40
residual intervals clears the prevalence"; it fails if a standardized
residual log-loss interval lies wholly above zero.
Report: a paragraph and a table in section 7's post-hoc subsection; the
verdict changes only if a section-7 claim does.
Task 10: complete. scaled_single_block.py ran in 133.3 s and its gates held:
every unscaled refit equals density_adds.npz's R, P1 or P24 arm seed for
seed. verify_results.py checks the file (_check_scaled_single_block: RED
before the run, GREEN after; ALL CHECKS PASSED), including one standardized
residual arm through scikit-learn's cross_val_predict with an in-fold
residualiser.
Results, standardized against protocol:
- where C sat: the raw residuals (SD 11,647 to 448,576) lie beyond the
  grid's bottom - on them C = 1e-4 is no stronger a penalty than C = 1.4e4
  on standardized residuals - and their protocol fits pick C = 1e-4 in
  98-100% of folds; the cosine H0 residual fits pick the top C in up to 52%
  (cosine H1 at most 4%). Standardized: the top C in at most 2% of a
  residual arm's folds; the raw residual arms keep the smallest C in up to
  80%.
- residual arms: none held back (accuracy -0.050 to +0.010); two raw H1 arms
  lose (slice 1: -0.050 [-0.098, -0.003]; slice 2: -0.048 [-0.095,
  -0.002]). Section 7's residual claims come back unchanged: the same 11 of
  40 intervals above chance (7 of the 8 at slices 8-9), none below, none
  above the prevalence in log-loss - the claim at risk holds.
- combined PC arms: raw PCs 2-4 held back (+0.042 [+0.012, +0.076] H0,
  +0.042 [+0.004, +0.080] H1; log-loss 1.633 and 1.907 better; their
  protocol fits had picked the smallest C in 76-79% of folds). Against the
  standardized density arm all four PCs 2-4 arms stay below (-0.038 to
  -0.012; H0 cosine wholly below, none above; protocol: two below). H0
  cosine PC1 stays level (-0.005 [-0.041, +0.029]). Raw PC1 arms not held
  back (+0.032 [-0.003, +0.066], +0.025 [-0.004, +0.056]).
Conclusion written into the report: section 7's reading of its residual and
PC arms stands; the raw PCs 2-4 arms were held back by their scale but stay
below the density arm.
Open, not checked: sections 5-6's PH-only arms (per-slice four-score arms
and the combined 40-score models) use the same estimator on raw and cosine
scores; the raw combined models, like the raw PCs 2-4 arms, may be held back
by their scale.
Task 10: report rebuilt (4.1 MB, 20 embedded PNGs, parser clean): a
paragraph and a table in section 7's post-hoc subsection (renamed "Feature
scale (post-hoc)"), a sentence in the verdict, the reproduction list. Text
read back; not viewed in a browser. Nothing committed.

Task 11 (Simon, 2026-11-17, "Yes, check here"): check sections 5-6's
PH-only arms for the same scale problem. Post-hoc, after Tasks 8-10. Rulings
fixed before any compute:
Ruling: scope is every classification arm behind the verdict - the four
families' four-score arms at slices 0-9, the per-slice density-only controls
(kept fraction), the four combined 40-score models and the combined control
(ten kept fractions). Each is refitted unscaled through
scaled_nested.fit_folds (gate: its per-seed accuracy and confusion matrix
equal those fpca_classify wrote to slice_<k>.npz and combined_classify.py to
combined.npz) and with a StandardScaler fitted on the outer training fold in
front of the same estimator; same folds and seeds (scaled_ph_only.py).
fpca_classify uses raw scores by design, as its docstring says; this check
asks only whether the verdict depends on that and proposes no change to
topocyte. Cost if wrong: none; 110 arm fits from saved scores.
Ruling: indicator and per-arm reading as in Tasks 9-10 - the share of folds
at the grid's top C, and an arm is held back by its scale if its
standardized accuracy interval against the protocol arm lies wholly above
zero (paired bootstrap, shared indices).
Ruling (reading, fixed before the run): the verdict rule is re-applied
unchanged with every arm standardized - a (slice, model) pair or a combined
model counts if its mean beats both its slice-0 baseline and its control by
more than 2 sqrt(SE_binomial^2 + SD_seed^2) summed over the two arms. The
verdict "No" changes only if a standardized pair or combined model clears
both bars; the report then states it as a post-hoc finding beside the
protocol verdict, which stays as written.
Report: a post-hoc subsection at the end of section 6 (text and a table)
and one sentence in the verdict.
Task 11: complete. scaled_ph_only.py ran in 82.3 s and its gate held: every
unscaled refit gives back the per-seed accuracies and confusion matrices
that fpca_classify (slice_<k>.npz) and combined_classify.py (combined.npz)
stored. verify_results.py checks the file (_check_scaled_ph_only: RED before
the run, GREEN after; ALL CHECKS PASSED), including one unscaled arm through
fpca_classify's own cross_val_predict call.
Results:
- where C sat: the raw scores (SD 13,420 to 1,164,503) lie beyond the grid's
  bottom - the per-slice raw arms pick C = 1e-4 in 98-100% of folds, the
  combined raw models in 69-75%; the per-slice cosine arms pick the top C in
  up to 67%, the combined control in 36%.
- per-arm reading: one arm held back - the combined H1 raw model, 0.871 ->
  0.911, +0.040 [+0.007, +0.072], log-loss 1.506 better; the combined H0 raw
  model +0.011 [-0.023, +0.045], log-loss 0.850 better. Elsewhere -0.014 to
  +0.011, 5 intervals below zero (slice 1 H0 and H1 cosine, slice 2
  control, slices 5 and 7 H0 raw).
- the verdict rule re-applied with every arm standardized (the reading fixed
  before the run): no (slice, model) pair clears both bars; the closest is
  again slice 6 H1 raw, 0.850 ± 0.014, +0.097 over slice 0 (needs 0.111) and
  +0.087 over its control (needs 0.106), margin -0.019 (protocol -0.025); 2
  pairs beat slice 0 alone (protocol 1), 23 their control alone (23). No
  combined model clears both: the standardized combined H1 raw model, 0.911
  ± 0.012, clears the slice-0 bar (+0.159, needs 0.103) but not the combined
  control (0.882 ± 0.010; +0.029, needs 0.084).
Conclusion written into the report: the verdict does not rest on the
protocol's use of raw scores. With Tasks 7-11, every arm the report draws a
conclusion from has been refitted with standardized features, except slide
size alone (one feature); no conclusion changed. For fpca_classify's own
use, one model's four scores, standardizing moved accuracy by at most 0.015
here.
Task 11: report rebuilt (4.1 MB, 20 embedded PNGs, parser clean): a
post-hoc subsection at the end of section 6 (text and table), one sentence
in the verdict, the reproduction list. Text read back; not viewed in a
browser. Nothing committed.

Task 12 (Simon, 2026-11-18, "Yes, set it up"): move the study out of the
topocyte package repository, which is to become public, into a separate
private repository, topocyte-analyses
(/Users/simon/rep/lbi/reindero/topocyte-analyses).
Ruling: tracked there are the scripts, this ledger, the verifier, the report
builder, results/*.json and a README; *.npz, *.html, logs/, figures/ and
superseded/ are not (the scripts regenerate them; the copy the report was
built from stays in this folder, ignored). Cost if wrong: none; nothing is
deleted.
Ruling: run_slices.py takes topocyte from the checkout that PYTHONPATH names
first (it used the folder two levels up, the topocyte worktree that held
artifacts/) and still refuses any other copy; the docstrings say
<topocyte checkout> for <worktree>. No other code changed.
Task 12: complete. From the new folder, with PYTHONPATH at bd1d5c0, the guard
accepts that checkout and refuses an unset PYTHONPATH and one without
topocyte; verify_results.py: ALL CHECKS PASSED; build_report.py writes a
report identical to the 2026-11-17 one apart from its date. README.md names
the topocyte commits: 7db4d26 (density, slices, combined) and bd1d5c0
(density_adds.py onwards); neither was on GitHub on 2026-11-18. The old copy
in topocyte's ignored artifacts/ is left in place.
