# Ledger: microanatomical domains from local chromatic persistent homology

## Task 1: rulings (Simon, 2026-11-20)

Simon's answers to the brainstorm's batched questions, and his two rulings
after the builder check. `design.md` ("Rulings") has them in full.

- **Domains:** two levels.
  - Level 1, compartments: from the density and composition of each centre's neighbourhood (the cellular-neighbourhood recipe), with no topology.
  - Level 2, subdomains: inside each compartment, from chromatic topology features measured against a reference.
- **Pairwise guard, fixed in advance:** the subdomains must not be predictable from pairwise co-occurrence, that is, Delaunay edge counts per class pair against the same reference.
- **Scope:** a pilot of about 30 slides stratified by condition, about 1,500 sampled centre cells each. Pathology localisation is study 2.
- **Validation:**
  - an H&E atlas: random centres per domain, labels hidden first;
  - recurrence across two halves of the slides.
  
  Simon downloads the pilot's SVS files from the cluster when the study reaches the atlas.
- **Reference: random labelling.**
  - The observed chromatic PH always uses the true classes.
  - The reference keeps the ball's cells, positions and count per class, and shuffles the classes; each shuffle goes through the same chromatic PH.
  - A feature is the observed curve minus the mean reference curve: the shuffles are only the reference, never the feature.
- **Neighbourhood:** the closed δ-ball around each centre cell, δ = 150 µm.
- **Summaries:** per class, as Betti curves. Of the six-pack of the inclusion (class t → all cells), only the domain and image curves are kept.
- **Clustering:** k-means at both levels.
- **Choosing k (after the builder check):** the gap statistic against a Gaussian with the same mean and covariance; k is where the gap stops rising, counting only gains of at least 0.02. Prediction strength of at least 0.8 in both directions then confirms recurrence; otherwise k = 1. Both levels.
- **Scaling of residuals (after the builder check):** moderated spread. Each residual curve is divided by sqrt(s² + s0²), the centre's own reference spread plus the block's median reference spread, so a structure never seen in the shuffles keeps a finite, large value.
- **Cost:** measured by a timing spike before the design (`design.md`, appendix).
- **Commits:** local, in topocyte-analyses; Simon pushes. The design and the plan are committed before any result.

## Task 1: scaffold and pilot draw (2026-11-20)

- **topocyte:** a clean, detached checkout of `18e3aa5` beside the topocyte repository (`../../topocyte-18e3aa5`), named first on `PYTHONPATH`. `common.check_topocyte()` refuses a topocyte imported from elsewhere, another commit and a changed package. All three refusals were tried on a scratch checkout.
- **Venv:** `.venv/`, CPython 3.13.9, from `requirements.txt`. `results/venv_freeze.txt` lists the 53 installed packages. topocyte, chalc and phimaker import.
- **Tests:** `test_common.py`, 10 tests on synthetic inputs, pass without topocyte or the cohort.
- **Reader check:** `read_slide` reproduces the design spike's cell counts and class shares on its two slides:
  - GTEX-YJ89-1426: 220,475 cells; 0.783, 0.141, 0.053, 0.023;
  - GTEX-NPJ8-0726: 203,167 cells; 0.361, 0.381, 0.098, 0.160.
- **Eligible slides** (`run_pilot.py pilot`): the cell-type study's 476 development slides, minus the 20 flagged "preservation", in the six pilot conditions. Counts: healthy 53, congestion 118, steatosis 158, inflammation/hepatitis 9, fibrosis 40, cirrhosis 44, as in the design. The script refuses other counts.
- **Pilot** (`default_rng(20261120)`), in `results/pilot.csv`: 30 slides, 15 per half. Half A holds 3 healthy, 2 congestion, 3 steatosis, 2 inflammation/hepatitis, 3 fibrosis and 2 cirrhosis slides; half B the other 15.

| slide index | slide | condition | half |
|---|---|---|---|
| 0 | GTEX-RTLS-1326 | healthy | A |
| 1 | GTEX-1HCU7-1726 | healthy | A |
| 2 | GTEX-144GM-1326 | healthy | A |
| 3 | GTEX-1HBPI-1026 | healthy | B |
| 4 | GTEX-YJ89-1426 | healthy | B |
| 5 | GTEX-R55F-2026 | congestion | A |
| 6 | GTEX-12ZZW-1426 | congestion | A |
| 7 | GTEX-17F96-1226 | congestion | B |
| 8 | GTEX-1B8L1-0726 | congestion | B |
| 9 | GTEX-15RJE-1126 | congestion | B |
| 10 | GTEX-PWOO-0826 | steatosis | A |
| 11 | GTEX-1399R-1226 | steatosis | A |
| 12 | GTEX-13NZ8-1826 | steatosis | A |
| 13 | GTEX-RM2N-1926 | steatosis | B |
| 14 | GTEX-13FTY-0826 | steatosis | B |
| 15 | GTEX-145MF-0826 | inflammation/hepatitis | A |
| 16 | GTEX-139TS-1426 | inflammation/hepatitis | A |
| 17 | GTEX-Y3IK-0326 | inflammation/hepatitis | B |
| 18 | GTEX-1LGRB-0626 | inflammation/hepatitis | B |
| 19 | GTEX-ZPU1-0826 | inflammation/hepatitis | B |
| 20 | GTEX-1HT8W-0826 | fibrosis | A |
| 21 | GTEX-U8XE-1526 | fibrosis | A |
| 22 | GTEX-PWO3-1826 | fibrosis | A |
| 23 | GTEX-132AR-0426 | fibrosis | B |
| 24 | GTEX-111VG-0826 | fibrosis | B |
| 25 | GTEX-1IL2U-2026 | cirrhosis | A |
| 26 | GTEX-ZTTD-1526 | cirrhosis | A |
| 27 | GTEX-13YAN-1626 | cirrhosis | B |
| 28 | GTEX-1EX96-0326 | cirrhosis | B |
| 29 | GTEX-ZZPT-1726 | cirrhosis | B |

- The draw picked GTEX-YJ89-1426, the timing spike's healthy slide (slide index 4, half B). The builder check's 80 balls, which led to the moderated scaling, came from the spike's two slides. The other spike slide, GTEX-NPJ8-0726, was not drawn.

Claude's calls:
- **Seed keys** (corrected after the task review, before any centre was drawn).
  - **The rule:** the pilot draw uses `default_rng(SEED)`; every other generator uses a 4-entry key `SeedSequence([SEED, slide_index, centre_index, stream])` whose last entry is non-zero.
  - **Why:** SeedSequence pads short keys with zeros, so `default_rng(SEED)` is `SeedSequence([SEED, 0, 0, 0])`. The first version keyed the slide's centre draw `[SEED, slide_index, 0, 0]`, which for slide 0 is the pilot draw's stream. The review found it.
  - **Keys:**
    - the slide's centre draw: `[SEED, slide_index, 0, R + 1]` (stream 20);
    - labelling r of a centre: `[SEED, slide_index, centre_index, r]`, r = 1..19;
    - labelling 0 draws nothing.
  - **Checked,** padded:
    - the pilot's key, the 30 centre draws', the 855,000 labellings' and the 30 fit-sample keys the plan gives Task 6 are all distinct;
    - the fit-sample key, `[SEED, slide_index, 0xF17]`, has 3 entries and pads to a final 0, so it does not have the rule's form.
  - `results/pilot.csv` is unchanged, since the pilot draw did not move.
  - Cost if wrong: none while the keys stay fixed; changing one redraws every centre or labelling.
- **Row order of the pilot table:** within each condition, rows follow the permutation, so half A comes first. `slide_index` numbers the rows. Cost if wrong: none; only the numbering.
- **pytest pinned** at 9.2.0, the version that resolved: the plan pins every direct dependency but gave pytest no version. Cost: none.
- **`read_slide` checks** that the slide's first and last cohort rows belong to it, beside the coordinates' shape and finiteness. Cost: two reads of a slide code.

## Task 2: chromatic six-packs and pair counts (2026-11-20)

- **Module:** `chromatic.py`, the numba port from chalc-scaling `8f370c6` (`report/port/chromatic_port.py`), crediting chalc and di Montesano et al. (arXiv:2212.03128). The plan's changes:
  - values squared (px²) after the facet-max pass;
  - faces up to dimension 2, with a private `_max_face_dim=3` route for the test;
  - one boundary matrix per complex, reused for every class present;
  - class codes 0..3 in the class masks, and a lift over the classes present only.
- **The controller's rulings, applied:**
  - cells of one class at one position are one point of the lift: merged before the triangulation, and the cells mapped back to cell ids;
  - pair counts triangulate the distinct positions; every pair of cells across a Delaunay edge, and every pair at one position, is an edge;
  - numba kernels serial, phimaker on one thread;
  - the codomain once per ball, from the first class present.
- **Tests:** `test_chromatic.py`, 23 tests, pass in 6.0 s with warnings as errors and the pinned topocyte on `PYTHONPATH`.
  - **Against chalc 15.0.1:** all six modules of every class present, degrees 0–1, by bar count and bottleneck distance within 1e-9 of the scale. Run on the 5 random clouds and on 150 cells plus 50 copies of their positions with other classes.
  - **Exact checks:**
    - the three Betti identities at every bar endpoint, no violation;
    - faces to dimension 2 against 3, bar for bar, on 3 random clouds and a hex lattice;
    - same-class duplicates give the six-packs of the cloud without them, bar for bar.
  - **Against topocyte's alpha PH:** the codomain, and the domain of each class present (classes {0, 2}).
  - **Edge cases:** an absent class, a one-class ball, 1 and 2 cells, 5 collinear cells, and a 700-cell ball (0.44 s).
  - **Pair counts:**
    - the mean over all 180 labellings of a 6-cell cloud, with and without a shared position, equals `pair_expectation` as fractions;
    - on 200 random cells, `n_edges` = 3n − 3 − hull vertices;
    - with coincident cells, the counts per class pair equal a brute-force count over every pair of cells.
- **Timings** (M4, warm numba cache, medians of 5): chalc's six-packs against the port's on the test clouds.

  | cloud | chalc 15.0.1 | port, `full` | port |
  |---|---|---|---|
  | 30 cells, 1 class | 2.0 ms | 0.6 ms | 0.5 ms |
  | 60 cells, 2 classes | 15.3 ms | 2.8 ms | 2.4 ms |
  | 120 cells, 3 classes | 69.1 ms | 18.0 ms | 16.9 ms |
  | 200 cells, 4 classes | 337 ms | 96 ms | 93 ms |
  | 150 cells, 4 classes | 244 ms | 73 ms | 70 ms |

  - **The chalc tests in pytest:** 0.08–0.57 s per call. The first takes 1.8 s with chalc's import and the numba cache load.
  - **A 200-cell, 4-class ball** (synthetic: uniform positions in a 303.52 px disc, 12 balls per row, medians):
    - 54 ms per complex with the healthy spike slide's class shares;
    - 85 ms with the cirrhosis slide's shares;
    - 97 ms with equal shares.
  - **Where the time goes:** GUDHI's Delaunay of the 5D lift 22–61 ms, phimaker's four six-packs 30–33 ms, the rest about 3 ms. The spike's real balls took 36 ms (healthy, 137 cells) and 68 ms (cirrhosis, 198 cells).
  - **numba:** the first call takes 1.3 s with a cold cache and 0.3 s with a warm one.

Claude's calls:
- **Planar only:** the port's 3D miniball, Qhull engine, timing hooks and value snapping (off in the port) are left out. Cost if wrong: none for planar cells.
- **Input checks:** `ValueError` for positions that are not (n, 2), a class vector of another length, codes outside 0..3, and an empty ball in `ball_sixpacks`. Without the check, a code of −1 would silently count as class 3 in `pair_counts`. Cost: microseconds per call.
- **An absent class with `full`:** its kernel, cokernel and relative keys are empty too, as the plan's interface says for every key of an absent class. The inclusion of an empty complex would have cokernel and relative equal to the codomain.
  - Only domain and image enter the study, and both are empty for an absent class.
  - Cost if wrong: none for the features; a check must apply the identities to present classes only, as the tests do.
- **Tests beyond the plan's list**, each shown to fail on a deliberate break of the code it covers:
  - **the duplicate test** (the controller's ruling);
  - **a hex lattice in `test_dim2_equals_dim3`.** The facet-max pass never fires up to dimension 2; on a 12×12 lattice it raises 21 tetrahedra, and phimaker refuses the dimension-3 matrix without it;
  - **exact rounding of `pair_expectation`:** 300 random draws against `fractions.Fraction`;
  - **the brute-force pair count;**
  - **the input checks.**

  Deliberate breaks that no test can see, because they do not change any result:
  - dropping the explicit merge (GUDHI also drops duplicate points);
  - lifting over all four codes (any affinely independent placement of the colours gives the same Delaunay cells);
  - taking the codomain from another class.
- **Finding for the verifier (Task 8): exact ties.** On a 12×12 hex lattice (10 px spacing, 3 classes), the port and chalc differ by ulps in 1,320 of 4,950 simplex values.
  - Their kernel H0 diagrams then differ by 2–5 bars, all shorter than 1e-12 of the scale; the bottleneck distance is at most 3e-14 px². After both sides drop bars that short, every module matches.
  - The unchanged port at `8f370c6` shows the same. Its own 18×18 lattice check passed by the luck of its scale and seed.
  - The random clouds have no such ties, and neither do the real coordinates: float32 values on GTEX-YJ89-1426, with no two cells at one position.
  - The verifier's bar-count criterion could still trip on an exactly tied real ball. Dropping bars shorter than 1e-12 of the scale before comparing would make it robust.
  - Cost if ignored: a spurious verifier failure, not a wrong result.

## Task 3: exact weighted step functions (2026-11-20)

- **Module:** `curves.py`, exact algebra on signed, weighted step functions Σ w·1[b, d) of finite bars, held as flat arrays with offsets (`CurveSet`).
  - `curve_set`, `combine` and `scale` build them. `combine` keeps every bar, so a residual curve is exact; `scale` shares the bars with its input.
  - `inner_matrix`: exact L² inner products. Each curve's step table (sorted distinct breakpoints, and the value to the next) is built once; each inner product is a merged sweep over two tables, with numba `prange` over rows.
  - `spread2`: each centre's reference spread.
  - `components`: φ_k = Σ_i v_ik/√ρ_k · f_i from `fpca_from_gram`'s eigenpairs, as one step table.
  - `score`: ⟨g, φ_k⟩ − ⟨f̄, φ_k⟩, through the antiderivative Φ_k at each bar endpoint (binary search).
- **The controller's rulings, applied:**
  - numba `prange` in `inner_matrix` only, called from the main thread;
  - step tables built once per curve and reused across pairs;
  - compensated event sums for φ and its antiderivative;
  - scoring by binary search of Φ;
  - a curve without bars gives zero rows and columns, and the score −mean_term.
- **Tests:** `test_curves.py`, 10 tests, pass in 2.7 s with warnings as errors and the pinned topocyte on `PYTHONPATH`, cold numba cache included. The whole study suite: 44 tests in 4.9 s. Measured margins:
  - the unit-weight Gram matrix against topocyte's `betti_inner_product_matrix`: at most 7.4e-16 relative (bound 1e-12), with exact zeros where topocyte has them;
  - bilinearity with weights 1 and −1/19: 9.2e-16 of the terms (bound 1e-12);
  - `spread2` against mean_r ‖ref_r − mean‖², with the mean built by `combine`: 1.3e-15 (bound 1e-10);
  - the fit curves' scores against `fpca_from_gram`: 2.9e-15 of the largest (bound 1e-9);
  - scores of curves outside the fit against the kernel route (C − r̄)·V/√ρ: 2.0e-15 (bound 1e-9);
  - a residual's score against the observed score minus the mean reference score: 1.7e-16 (bound 1e-10);
  - φ on 2e6 events of ±1e8 and ±1e-8 against `math.fsum` per segment: 1.1–2.2e-16 over 6 seeds (bound 1e-12).
- **Timings** (M4, warm numba cache). The curves are residuals of 20 labellings × 28 bars, scaled per curve. `inner_matrix` and `fpca_from_gram` run on 10 threads; `components` and `score` are serial.

  | step, 6,000 curves | H1-like (1,120 breakpoints per curve) | H0-like (561) |
  |---|---|---|
  | `inner_matrix` | 23.2 s | 11.8 s |
  | `fpca_from_gram` (topocyte, dense) | 18.8 s | 18.7 s |
  | `components`, 9 | 2.7 s (6.7M breakpoints) | 1.5 s |
  | `score` of the fit curves | 1.3 s | 0.8 s |

  - **Threads:** 1,500 H1-like curves take 11.3 s on 1 thread, 3.0 s on 4 and 1.5 s on 10 (7.5×). numba's dynamic chunks (`parallel_chunksize(8)`) were 10% slower than its static split, whose steps pair rows t and n − 1 − t.
  - **The Gram cost grows with the square of the curves.** Extrapolated, not measured: about 95 s for an H1 block's 12,000-curve fit on all slides, and about half an hour for the 48 Gram matrices of the 16 blocks.
  - **`fpca_from_gram` is a dense eigendecomposition,** cubic in the curves: 18.8 s at 6,000 curves, so expect several times the Gram's cost at 12,000 (not measured).
  - **At the full block size** (H1-like, serial):
    - `score` of 45,000 curves (25.2M bars) takes 9.5 s;
    - `spread2` of 45,000 centres × 19 references takes 5.5 s;
    - `combine` of 20 sets takes 0.3 s;
    - peak RSS was 4.8 GB with a 6,000-curve fit in memory.

Claude's calls:
- **The spread from squared distances.**
  - `spread2` computes (1/(2R²))·Σ_{r,s} ‖ref_r − ref_s‖², each distance a merged sweep of (ref_r − ref_s)². It is the same number as the plan's (1/R)·Σ_r ‖ref_r‖² − ‖mean‖² from the R × R inner products.
  - **Why:** when the references agree, the inner-product form cancels to a rounding residue of either sign, while the design's s0 is the median over the centres with s > 0. The distance form is exactly 0 there and never negative.
  - `test_spread_zero_when_references_agree` fails with the inner-product form.
  - Cost if wrong: none; the quantity is the same.
- **Second-order Neumaier.**
  - On the plan's test events (2e6 of ±1e8 and ±1e-8), first-order Neumaier drifts by 2.1–2.2e-12 relative on every one of 6 seeds, above the 1e-12 bound. Second order is within 1.1–2.2e-16 on the same seeds, in `components` itself: the first step's rounding errors are summed by a second Neumaier step (Klein 2006, iterative Kahan–Babuška).
  - Every compensated sum uses it: the step values, φ, Φ and each curve's score.
  - Cost: two Neumaier steps per event instead of one; `components` takes 1.5–2.7 s for 6,000 fit curves with it.
- **The linearity test's identity.** The score ⟨g, φ_k⟩ − mean_term_k is affine. With coefficients summing to 0, a residual's score is score(obs) − mean_r score(ref_r) − mean_term; the plan's shorthand leaves out the mean_term. The test asserts the exact identity. Cost if wrong: none.
- **The compensated-sum test's input.**
  - The 1e8 bars come in pairs of opposite weight on one interval, so every segment's value is the 1e-8 bars alive there. Without the pairs, the segment values are dominated by the 1e8 events, and a plain running sum passes too.
  - The test asserts that `np.cumsum` fails on its input.
- **Tests beyond the plan's list**, each shown to fail on a deliberate break of the code it covers:
  - `test_spread_zero_when_references_agree` (the spread from inner products);
  - `test_scores_of_new_curves_match_kernel_route` (Φ read only at breakpoints). This break passed every other test: the fit curves' endpoints are all breakpoints of φ, and the broken score is still linear;
  - an odd-sized Gram matrix in `test_inner_matches_topocyte` (a balanced loop that misses the middle row);
  - `test_bad_inputs_refused` (the input checks).

  Ten deliberate breaks in all, each caught. The other six were: death events with +w, a sweep that adds after the update, scores without the mean_term, components without 1/√ρ, `combine` ignoring the earlier sets' rows, and a nonzero value after a curve's last breakpoint.
- **Input checks.**
  - `CurveSet` refuses non-finite values, d ≤ b and offsets that do not rise from 0 to the number of bars.
  - `curve_set` refuses NaN or infinite births and deaths below births.
  - `components` refuses a cosine fPCA, and more components than it kept.
  - Cost: one pass over the bars per set built.
- **The whole line.** The inner products integrate over the whole line, and topocyte's from 0. They agree for births ≥ 0, as squared radii are; `curve_set` does not check this. Cost if wrong: none for the study's values.
- **Serial `score` and `spread2`**, as the ruling names `inner_matrix` only: 9.5 s and 5.5 s per call at the full block size.
  - Cost: the observed and control scores take about 19 s per block and fitting set, about 15 min over the 48 pairs.
  - `prange` over curves would cut this several-fold, under the same main-thread condition.
- **The sweep kept branchy:** a branchless sweep gave identical bits but only 16% less time, since the loop is bound by its dependent loads, not by mispredictions.

## Task 4: rule for k and pre-flight (2026-11-21)

- **Module:** `domains.py`, the rule for k of design step 10 as pure functions, with numpy and scikit-learn only.
  - `kmeans_fit`: `KMeans(k, n_init=20, random_state=seed)`. For k = 1, one start at the column means, which Lloyd's step keeps.
  - `gap_choice`: one half's gap statistic against B = 20 draws from a Gaussian with the half's mean, covariance and number of rows, from one `default_rng(seed)` in order.
    - W_1 is the total sum of squares about the means and W_k the k-means inertia, for the data and every reference.
    - s_k = sd_b(log W*_k)·sqrt(1 + 1/B), with ddof 0.
  - `stop_rule`: the half's k from its curves (rulings below).
  - `ps_from_labels`, `prediction_strength`: Tibshirani and Walther's prediction strength, from an integer contingency table, never a loop over pairs. A test cluster of one row counts 0.
  - `choose_k`: the smaller half's k, kept only with prediction strength of at least 0.8 in both directions.
- **The controller's rulings, applied:**
  - R2's synthetic clouds for the tests: 4,000 × 20 from one `default_rng(1)`, in the order elongated, skewed, two clusters, three clusters;
  - the gap statistic as above, with gap and k-means seeds 0;
  - prediction strength from the contingency table.

### Two rulings on the rule for k (2026-11-21, before any study data)

1. **The floor is 0.03, not 0.02 (Simon).**
   - **Under 0.02, `test_gap_skewed_unimodal` failed.** The R2 skewed cloud (column 0 exp(N(0, 0.6)), the others variances 1/i, 4,000 × 20) gained −0.0038, 0.0169, 0.0214, 0.0159 and 0.0150 over Gap(1) at k = 2..6. The rule's first version picked k = 4.
   - **The design's "at most 0.016" was one draw.** On the test's draw, the controller's settings (B = 10, `n_init=10`) gave 0.0229 at k = 4, so the draw made the difference, not B or `n_init`.
   - **Ten fresh draws** gained 0.0169 ± 0.0030 at k = 4 and 0.0134 ± 0.0029 at k = 3, and 2 of the 10 picked k > 1. A draw of 22,500 × 20 gained at most 0.0174.
   - **Ruling:** Simon raised the floor to 0.03 and added skewed cluster-free clouds to the pre-flight, where k must be 1.
   - Cost if wrong: splits whose gain falls between 0.02 and 0.03 are missed (design step 10).
2. **Tibshirani's stopping rule from k = 1 (the controller's correction of the formal rule).**
   - **The first formal version** took the smallest k ≥ 2 with a gain at the floor and the stop condition. It skipped k = 1, while the ruling says k is "where the gap stops rising".
   - **The pre-flight's level-1 skewed cloud exposed it** (22,500 × 5: column 0 exp(N(0, 0.6)), the others variances [1.5, 0.8, 0.4, 0.2]).
     - Its gap falls at k = 2 and rises again from k = 5: gains 0.042 and 0.046 there, up to 0.055 at k = 9.
     - The first version picked k = 5 in both halves, and prediction strength confirmed it: 0.849 and 0.847, and also 0.98 at k = 2 and 0.88 at k = 4.
     - No floor could separate that cloud from the two-cluster mixture, whose gains are 0.041 and 0.045 in 20 dimensions.
   - **The rule now:**
     - k̂ is the smallest k ≥ 1 with Gap(k) ≥ Gap(k+1) − s_{k+1}, or k_max if there is none.
     - If k̂ ≥ 2, its gain Gap(k̂) − Gap(1) must reach 0.03; otherwise the half's k is 1.
     - `stop_rule` holds it, and the pre-flight adds a skew-dominant 5-d cloud (the lognormal column scaled to variance 3).
   - Cost if wrong: a structure whose gap does not rise from k = 1 to k = 2 is missed, the standard rule's known caution.

- **Tests:** `test_domains.py`, 30 tests (18 of them `test_stop_rule` cases), pass in 62 s with warnings as errors, without topocyte or the cohort. The whole study suite: 74 tests in 63 s.
  - **The R2 clouds** under the binding settings, gains Gap(k) − Gap(1) at k = 2..6:

    | cloud | gains | k |
    |---|---|---|
    | elongated | 0.0041, 0.0025, 0.0017, 0.0017, 0.0017 | 1 |
    | skewed | −0.0038, 0.0169, 0.0214, 0.0159, 0.0150 | 1 |
    | two clusters, 4 SD apart | 0.0429, 0.0205, 0.0191, 0.0174, 0.0167 | 2 |
    | three clusters | 0.0322, 0.1127, 0.0767, 0.0658, 0.0593 | 3 |

  - **`stop_rule` on hand-made curves:**
    - the controller's four cases;
    - the stop and the floor in detail;
    - the level-1 skewed half that broke the first version.
  - **The gap statistic** against its definition: 60 × 3, B = 4, to 1e-12.
  - **The binding settings:** B = 20, the 0.03 floor, PS 0.8, `n_init` 20, and `gap_choice`'s defaults.
  - **Prediction strength:**
    - separated blobs give 1.0;
    - the training centroids judge the test clusters, not the reverse;
    - `ps_from_labels` against a hand example, the singleton rule and an all-pairs count.
  - **`choose_k`,** with its parts patched.
  - **Deliberate breaks** of a scratch copy: 20 of 21 fail a test. The 21st, k = 1 through k-means++, changes no result.
- **Pre-flight** (`run_pilot.py preflight`, `results/preflight.json`): the full rule on synthetic halves of the pilot's sizes. One `default_rng(1)` draws them in case order, half A then half B. **Passed**, 9 of 9 cases, in 560 s (2,538 s CPU), peak RSS 0.27 GB.

  | case (rows per half × columns, k_max) | must be | k | k_A, k_B | largest gain, A / B | stop at 1, margin A / B | PS, A→B / B→A |
  |---|---|---|---|---|---|---|
  | level-1 Gaussian, variances [3, 1.5, 0.8, 0.4, 0.2] (22,500 × 5, 10) | 1 | 1 | 1, 1 | −0.0014 / 0.0006 | +0.0070 / +0.0059 | — |
  | level-2 Gaussian, variances 1/i (5,000 × 20, 6) | 1 | 1 | 1, 1 | 0.0029 / 0.0036 | +0.0049 / +0.0049 | — |
  | level-2 Gaussian, variances 1/i (22,500 × 20, 6) | 1 | 1 | 1, 1 | 0.0016 / 0.0004 | +0.0032 / +0.0031 | — |
  | two clusters, 4 SD apart (5,000 × 20, 6) | 2 | 2 | 2, 2 | 0.0412 / 0.0447 | −0.0374 / −0.0408 | 0.998 / 0.996 |
  | three clusters (5,000 × 20, 6) | 3 | 3 | 3, 3 | 0.1167 / 0.1111 | −0.0317 / −0.0272 | 0.994 / 0.992 |
  | level-2 skewed (5,000 × 20, 6) | 1 | 1 | 1, 1 | 0.0174 / 0.0176 | +0.0146 / +0.0125 | — |
  | level-2 skewed (22,500 × 20, 6) | 1 | 1 | 1, 1 | 0.0188 / 0.0205 | +0.0103 / +0.0076 | — |
  | level-1 skewed (22,500 × 5, 10) | 1 | 1 | 1, 1 | 0.0501 / 0.0553 | +0.0054 / +0.0070 | — |
  | level-1 skew-dominant (22,500 × 5, 10) | 1 | 1 | 1, 1 | 0.0514 / 0.0439 | +0.0359 / +0.0348 | — |

  - **The margin** is Gap(1) − (Gap(2) − s_2): where it is positive, the gap stops at k = 1. The mixtures' largest gains are at the k they chose.
  - **The skewed clouds' gains are not small, but they come after the gap has stopped at 1.** The 5-d ones reach 0.044–0.055 by k = 8–9, the 20-d ones about 0.02.
  - **The gap gains** Gap(k) − Gap(1), k = 2..k_max, per case and half:

    | case | half | gains |
    |---|---|---|
    | level-1 Gaussian | A | −0.0014, −0.0037, −0.0023, −0.0043, −0.0037, −0.0043, −0.0041, −0.0027, −0.0015 |
    | | B | −0.0003, −0.0007, 0.0006, 0.0004, 0.0000, 0.0002, −0.0010, −0.0018, −0.0018 |
    | level-2 Gaussian, 5,000 | A | 0.0012, −0.0005, 0.0014, 0.0015, 0.0029 |
    | | B | 0.0012, 0.0011, 0.0034, 0.0036, 0.0023 |
    | level-2 Gaussian, 22,500 | A | −0.0000, 0.0004, 0.0009, 0.0016, 0.0006 |
    | | B | −0.0000, −0.0001, 0.0004, −0.0005, −0.0013 |
    | two clusters | A | 0.0412, 0.0197, 0.0176, 0.0155, 0.0142 |
    | | B | 0.0447, 0.0216, 0.0205, 0.0184, 0.0178 |
    | three clusters | A | 0.0359, 0.1167, 0.0797, 0.0679, 0.0608 |
    | | B | 0.0313, 0.1111, 0.0755, 0.0638, 0.0571 |
    | level-2 skewed, 5,000 | A | −0.0087, 0.0135, 0.0159, 0.0134, 0.0174 |
    | | B | −0.0066, 0.0146, 0.0176, 0.0127, 0.0140 |
    | level-2 skewed, 22,500 | A | −0.0073, 0.0156, 0.0181, 0.0146, 0.0188 |
    | | B | −0.0046, 0.0185, 0.0205, 0.0173, 0.0189 |
    | level-1 skewed | A | −0.0004, −0.0023, −0.0013, 0.0419, 0.0432, 0.0407, 0.0433, 0.0501, 0.0477 |
    | | B | −0.0019, −0.0030, 0.0070, 0.0459, 0.0491, 0.0468, 0.0500, 0.0553, 0.0550 |
    | level-1 skew-dominant | A | −0.0303, 0.0215, 0.0195, 0.0273, 0.0371, 0.0411, 0.0508, 0.0514, 0.0490 |
    | | B | −0.0292, 0.0194, 0.0107, 0.0157, 0.0250, 0.0304, 0.0439, 0.0427, 0.0404 |

  - The pre-flight ran three times as the rulings came. The draws of every case are the same in all three runs, and the curves agree to four decimals. The table is the last run's, under the rule as it now stands.
- **Timings** (M4, 10 cores, scikit-learn's k-means threads):
  - one `gap_choice`: 12–16 s at 4,000 × 20 with k_max 6 (105 fits of 20 starts), and 30–36 s at 22,500 × 20;
  - a 22,500 × 5 pre-flight case with k_max 10: 91–111 s for both halves;
  - the test file: 62 s, of which the three gap tests take 14–27 s each.

Claude's calls:
- **`gap_A` and `gap_B` in `choose_k`'s result** are each half's whole `gap_choice` result: its k, and the curves `gap` and `s`. The plan says "the curves", and s gives the report's error bars. Cost if wrong: one key's shape.
- **`ps` is empty when min(k_A, k_B) = 1.** When prediction strength rejects, `ps` keeps the values at min(k_A, k_B), so the JSON shows why k fell to 1. Cost: none.
- **`choose_k` refuses anything but two directions,** so the "both directions" condition cannot be skipped. Cost: none for the study's calls.
- **The pre-flight's data:**
  - no z-scoring, and cases added later are appended, so earlier cases keep their draws;
  - the skew-dominant column is scaled by the lognormal's theoretical variance, (e^0.36 − 1)·e^0.36 ≈ 0.621, not a sample's.
  
  Cost if wrong: none; any fixed draw serves.
- **k = 1 in `kmeans_fit`** is one start at the mean instead of 20 k-means++ starts; the result is the same.
- **k-means inertia is not bit-reproducible.** scikit-learn's OpenMP reduction order moved s by 1e-16 between two calls on one input, and the gap curve not at all. The verifier (Task 8) should compare recomputed gap statistics with a tolerance. Cost if ignored: a spurious verifier failure.
- **`README.md`** is not updated: Task 8 owns its run table, and its "Nothing else has run" is now stale.

## Task 5: bars of the pilot (2026-11-22)

- **Subcommand:** `run_pilot.py bars [--slides ...] [--workers 8]` writes `results/bars/{slide_id}.npz` and `.json` for each pilot slide, in slide-index order.
  - **Balls:** `cKDTree(xy).query_ball_point(xy[centre], r=DELTA_PX)` over the slide's cells: closed, holding the centre, rows sorted ascending.
  - **Per centre:** labelling 0 and labellings 1..19 (`common.labelling`) go through `chromatic.ball_sixpacks`. Kept: the finite bars of the 16 blocks of every labelling, the finite codomain of labelling 0, and the class counts, pair counts and edge count of the observed classes.
  - **The npz:**
    - per centre: `centre_rows`, `centre_xy`, `n_cells` (cells in the ball), `class_counts`, `pair_counts`, `n_edges`, `seconds`;
    - `bars`, sorted by owner = (centre·20 + labelling)·16 + block (`common.encode_owner`), then birth and death, with `offsets` (480,001 entries);
    - `codomain`, sorted by `codomain_owner` = 2·centre + p.
    
    Essential bars are left out, as topocyte's reduced curves leave them out; values are the port's float64, untouched.
  - **The json:** wall, read and pool seconds, the pool's busy share, workers, ball sizes (minimum, median, maximum and its centre), seconds per centre (median, maximum and its centre, total), the workers' CPU seconds and peak RSS, the parent's peak RSS, bar counts, npz size, versions and the topocyte commit.
  - **Workers:** one spawn `ProcessPoolExecutor(8)` per slide over chunks of 100 centres.
    - The parent reads the slide, draws the centres, finds the balls and sends each chunk its balls' cells; each worker returns its chunk's arrays, and the parent assembles the slide.
    - numba is warmed once in the parent, and in each worker's initializer, so a worker's first centre is not timed with a cache load.
  - **Atomic writes and resume:**
    - the npz is written to `{slide}.npz.tmp` and synced, the json is written atomically, and the npz is renamed last;
    - a rerun skips a slide whose npz opens, holds every key (each read in full, so zipfile checks every CRC), has offsets of the right length, and has `offsets[-1] == len(bars) == len(owner)`.
  - **The `labelling` guard** (Task 1's carried minor): r must be 0 or 1..19. Stream 20 is the slide's centre draw, so r = 20 and anything else raise `ValueError`.
- **Tests:** `test_common.py`, 18 tests (7 new), pass in 20 s with warnings as errors; the study suite, 81 tests in 79 s.
  - **The guard,** at r = −1, 20 and 100.
  - **The owner code:** the extremes round-trip; the codes of all 480,000 triples are 0..479,999 in order; out-of-range arguments raise.
  - **`bars_complete`:** a missing file, a missing key, too few bars, offsets of the wrong length, owners and bars of different lengths, a truncated file and junk bytes are each incomplete.
  - **Resume:** a fake two-slide run (`read_slide` patched to 300-cell slides, `N_CENTRES` = 20). The complete slide is skipped, its mtime unchanged. The truncated one, with a stale `.tmp` beside it, is redone, bit for bit except `seconds`, and no `.tmp` is left.
  - **Every stored array of a fake slide against a direct computation:**
    - the closed ball by brute force;
    - the class and pair counts;
    - all 20 × 16 owners and the codomain of every centre, bit for bit.
  - The fake runs use a real spawn pool of 2 processes and chunks of 6 centres, so 20 centres make 4 chunks.
  - **Deliberate breaks:** 10 of 10 caught, on a scratch copy:
    - no finite filter;
    - a wrong owner formula;
    - `bars_complete` ignoring the offsets;
    - no skip;
    - a wrong seed key;
    - no `.tmp` rename;
    - no sort within an owner;
    - no guard;
    - pair counts of a shuffle;
    - chunks assembled in completion order.
- **One slide first** (GTEX-YJ89-1426, slide index 4, the timing spike's healthy slide; `logs/bars_first_slide.log`):
  - 246 s on 8 processes; balls of 11–232 cells (median 129; the spike's 12 balls had a median of 137);
  - **median 1.25 s per centre, 1.62× the spike's 0.77 s, inside the 2× gate;** maximum 2.63 s;
  - the pool was busy 92% of the time, workers' peak RSS 0.36 GB, 12.9M bars, 0.27 GB of npz.
  - **Spot checks** (a scratch script, reading the cohort read only):
    - on every centre: the centre rows are the slide's draw; the class counts add up to `n_cells`; the pair counts add up to `n_edges`; the codomain has (distinct positions in the ball) − 1 finite H0 bars;
    - on centre 0, the largest ball (232 cells) and the smallest (11): the ball equals a brute-force closed ball, and the stored codomain equals topocyte's alpha PH (`compute_persistent_homology(ball_xy, filtration="alpha", max_dim=1)`, finite bars) in bar count. Values agree to 1e-16–6e-14 of the scale (at most 3.5e-10 px²): the port's Welzl radii and GUDHI's alpha values are computed differently;
    - the largest ball recomputed through `ball_sixpacks`: all 20 × 16 owners (16,799 bars), the codomain, the pair counts and `n_edges`, bit for bit.
- **phimaker's threads (a finding).**
  - phimaker 0.5.0 runs up to 10 threads inside each six-pack call, though the port passes `num_threads=1`.
    - On 8 real balls of this slide, serially: 2.4 CPU-seconds per wall second inside phimaker, against 1.00 in GUDHI's Delaunay and in the port's numba code.
    - `ps` sees up to 10 threads; between calls the process has one, so the threads are scoped to the call.
    - `RAYON_NUM_THREADS=1` changes nothing.
  - **Results are unaffected:**
    - the pairing is unique for a fixed column order;
    - the resume test reproduces a slide bit for bit in other processes;
    - the spot check reproduces the largest ball bit for bit.
  - **Throughput is affected.** A centre of this slide takes 0.50 s serially on this machine (8 centres), against 1.25 s with 8 processes, so 8 processes give about 3.2× the serial throughput, not the ~6× the design's Compute section assumed.
  - The design's "phimaker on one thread" is true of the parameter passed, not of the library's threads.
- **Full run** (`logs/bars.log`; `logs/` stays out of git):
  - 29 slides computed and GTEX-YJ89-1426 skipped as complete, in 11,546 s; with the first slide, 11,792 s of slide wall time (3.3 h), 11:13–14:33 on 2026-11-22.
  - **Totals:** 447,050,046 finite domain and image bars and 13,355,202 codomain bars; 9.33 GB of npz in `results/bars/` (ignored by `*.npz`); the 30 jsons are tracked.
  - **Review Focus 4, dense balls:**
    - 7 slides held a ball of at least 500 cells, the largest 744 cells (GTEX-PWOO-0826, steatosis), whose centre took 12.36 s, the run's slowest;
    - every slide completed;
    - the workers' peak RSS was 0.33–0.45 GB per slide; the parent's, counted over the whole run, 1.50 GB at most.
  - **Pool:** busy 91–93% of each slide's time. The workers' CPU time was 1.29× their timed centre seconds (111,774 s against 86,809 s, or 24.1 centre-hours), from phimaker's threads and the workers' start-up.
  - **Against the spike's serial medians:** the healthy slides took 1.01–1.98 s per centre against 0.77 s, the cirrhosis slides 1.78–2.83 s against 1.38 s. The slides with the larger medians also have larger balls (medians of up to 212 cells against the spike's 137 and 198).
  - **Spot checks on all 30 slides** (the scratch script of the one-slide check):
    - every whole-file check holds on every centre;
    - 90 centres' codomains equal topocyte's alpha PH, to within 2.0e-13 of the scale;
    - each slide's largest ball is recomputed bit for bit, the 744-cell one included (50,779 bars);
    - no ball holds two cells at one position.

  | index | slide | condition | half | cells | ball cells, median (max) | s per centre, median (max) | wall | busy | workers' peak RSS | bars | npz |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 0 | GTEX-RTLS-1326 | healthy | A | 120,013 | 113 (188) | 1.01 (2.01) | 201 s | 0.92 | 0.36 GB | 11.2M | 0.23 GB |
  | 1 | GTEX-1HCU7-1726 | healthy | A | 150,352 | 124 (247) | 1.14 (2.91) | 226 s | 0.92 | 0.38 GB | 12.7M | 0.26 GB |
  | 2 | GTEX-144GM-1326 | healthy | A | 255,657 | 176 (254) | 1.98 (3.28) | 371 s | 0.93 | 0.42 GB | 17.8M | 0.37 GB |
  | 3 | GTEX-1HBPI-1026 | healthy | B | 274,842 | 106 (356) | 1.06 (4.23) | 221 s | 0.92 | 0.37 GB | 10.5M | 0.22 GB |
  | 4 | GTEX-YJ89-1426 | healthy | B | 220,475 | 129 (232) | 1.25 (2.63) | 246 s | 0.92 | 0.36 GB | 12.9M | 0.27 GB |
  | 5 | GTEX-R55F-2026 | congestion | A | 221,627 | 135 (509) | 1.52 (6.26) | 320 s | 0.92 | 0.38 GB | 14.2M | 0.30 GB |
  | 6 | GTEX-12ZZW-1426 | congestion | A | 214,488 | 154 (317) | 2.01 (4.13) | 410 s | 0.93 | 0.36 GB | 15.7M | 0.33 GB |
  | 7 | GTEX-17F96-1226 | congestion | B | 202,483 | 117 (301) | 1.19 (4.42) | 249 s | 0.92 | 0.33 GB | 12.1M | 0.25 GB |
  | 8 | GTEX-1B8L1-0726 | congestion | B | 190,664 | 123 (243) | 0.89 (3.60) | 217 s | 0.92 | 0.39 GB | 12.9M | 0.27 GB |
  | 9 | GTEX-15RJE-1126 | congestion | B | 294,129 | 188 (292) | 2.46 (4.76) | 482 s | 0.93 | 0.44 GB | 20.3M | 0.42 GB |
  | 10 | GTEX-PWOO-0826 | steatosis | A | 441,554 | 209 (744) | 3.02 (12.36) | 612 s | 0.93 | 0.43 GB | 21.9M | 0.46 GB |
  | 11 | GTEX-1399R-1226 | steatosis | A | 390,561 | 129 (379) | 1.75 (6.46) | 365 s | 0.92 | 0.40 GB | 12.5M | 0.26 GB |
  | 12 | GTEX-13NZ8-1826 | steatosis | A | 268,526 | 192 (316) | 2.73 (5.38) | 554 s | 0.93 | 0.44 GB | 20.0M | 0.41 GB |
  | 13 | GTEX-RM2N-1926 | steatosis | B | 246,310 | 121 (421) | 1.64 (6.88) | 346 s | 0.92 | 0.40 GB | 11.5M | 0.24 GB |
  | 14 | GTEX-13FTY-0826 | steatosis | B | 57,372 | 120 (264) | 1.50 (4.15) | 313 s | 0.92 | 0.39 GB | 11.9M | 0.25 GB |
  | 15 | GTEX-145MF-0826 | inflammation/hepatitis | A | 224,292 | 113 (465) | 1.00 (6.22) | 264 s | 0.91 | 0.41 GB | 12.6M | 0.26 GB |
  | 16 | GTEX-139TS-1426 | inflammation/hepatitis | A | 107,738 | 124 (418) | 1.61 (6.92) | 356 s | 0.92 | 0.41 GB | 13.0M | 0.27 GB |
  | 17 | GTEX-Y3IK-0326 | inflammation/hepatitis | B | 307,839 | 139 (542) | 2.08 (8.99) | 514 s | 0.92 | 0.44 GB | 15.4M | 0.32 GB |
  | 18 | GTEX-1LGRB-0626 | inflammation/hepatitis | B | 172,037 | 109 (375) | 1.49 (5.30) | 318 s | 0.92 | 0.33 GB | 10.7M | 0.22 GB |
  | 19 | GTEX-ZPU1-0826 | inflammation/hepatitis | B | 377,777 | 156 (341) | 2.28 (5.34) | 463 s | 0.93 | 0.41 GB | 14.9M | 0.31 GB |
  | 20 | GTEX-1HT8W-0826 | fibrosis | A | 240,788 | 135 (405) | 1.77 (6.86) | 377 s | 0.92 | 0.41 GB | 13.6M | 0.28 GB |
  | 21 | GTEX-U8XE-1526 | fibrosis | A | 156,224 | 150 (372) | 1.98 (6.00) | 435 s | 0.92 | 0.43 GB | 16.0M | 0.33 GB |
  | 22 | GTEX-PWO3-1826 | fibrosis | A | 279,524 | 212 (500) | 3.38 (8.35) | 714 s | 0.93 | 0.44 GB | 20.4M | 0.43 GB |
  | 23 | GTEX-132AR-0426 | fibrosis | B | 152,608 | 104 (374) | 1.41 (5.64) | 330 s | 0.92 | 0.39 GB | 10.6M | 0.22 GB |
  | 24 | GTEX-111VG-0826 | fibrosis | B | 317,274 | 108 (557) | 1.39 (7.79) | 322 s | 0.91 | 0.40 GB | 11.4M | 0.24 GB |
  | 25 | GTEX-1IL2U-2026 | cirrhosis | A | 316,834 | 165 (568) | 2.34 (9.72) | 532 s | 0.92 | 0.45 GB | 17.3M | 0.36 GB |
  | 26 | GTEX-ZTTD-1526 | cirrhosis | A | 260,538 | 172 (670) | 2.83 (9.45) | 644 s | 0.92 | 0.45 GB | 17.7M | 0.37 GB |
  | 27 | GTEX-13YAN-1626 | cirrhosis | B | 190,171 | 180 (360) | 2.54 (5.45) | 520 s | 0.93 | 0.44 GB | 18.8M | 0.39 GB |
  | 28 | GTEX-1EX96-0326 | cirrhosis | B | 376,833 | 148 (380) | 1.78 (5.28) | 348 s | 0.93 | 0.40 GB | 15.4M | 0.32 GB |
  | 29 | GTEX-ZZPT-1726 | cirrhosis | B | 239,163 | 199 (310) | 2.60 (5.30) | 522 s | 0.92 | 0.45 GB | 21.1M | 0.44 GB |

  - **Seconds per centre** are each centre's wall time in its worker while 8 workers ran: its 20 six-packs, its pair counts and its class counts.
  - **The run took 3.3 h, not the design's 2 h:** 8 processes gave about 3.2× the serial throughput (phimaker's threads, above), and several slides have larger balls than the spike's two.

Claude's calls:
- **The parent finds the balls** and sends each chunk its balls' cells; the controller allowed the workers to re-read the slide or to receive it through an initializer.
  - **Why:** the slide is read once, a worker holds only its chunk's cells, and a test's patched `read_slide` and `N_CENTRES` hold in spawned workers.
  - Cost if wrong: none; the same KD-tree call, and the rows are sorted either way.
- **Bars sorted within each owner** by birth, then death.
  - **Why:** phimaker returns its pairs in an order that changes from process to process. Without the sort, the resume test's redone slide comes back with the same bars in another order.
  - The sort makes a slide's npz reproducible bit for bit. Cost: one lexsort per chunk.
- **The owner code in `common.py`** (`encode_owner`, `decode_owner`), beside the constants it uses, because Task 6 reads the bars with it. Cost: none.
- **One pool per slide:** fresh workers give each slide's peak RSS exactly. The price is idle workers in each slide's last round of chunks: busy 91–93%, so about 7% of the 3.3 h. Cost if wrong: about 14 min.
- **The json before the npz's rename:** a slide counts as done only once its npz is in place, and its json is then already there. Cost: none.
- **`bars_complete` reads every array,** so zipfile checks every CRC: about 0.3 s per slide on a rerun.
- **numba warmed in each worker's initializer too,** so no worker's first centre is timed with a cache load. Cost: about 0.3 s per worker and slide.
- **Sleep.** On 2026-11-21 the Mac went into clamshell sleep on battery while the study's tests ran (28 min of wall clock for 92 s of work), and the session was cut off.
  - The runs of 2026-11-22 went on AC power inside `caffeinate -is`, and the session asked the app to keep the Mac awake. Neither changes a setting.
  - The lid was closed at 11:33, 13 min into the full run. caffeinate's system-sleep assertion held on AC power, so the Mac stayed in dark wake until the run ended. It slept at 14:32:41, the second the last slide was written.
  - The slides' start and finish times run back to back from 11:13:36 to 14:32:41, so no sleep fell inside the run. On battery, a closed lid would have stopped it.
  - Cost: none.
- **`main`'s dispatch:** the `preflight` branch no longer folds its exit into the `elif` condition, so `bars` reads as its own branch (Task 4's deferred minor, touched in passing). Cost: none.
- **The json carries more than the plan names:** condition, half, the slide's cells, read, pool and write seconds, the pool's busy share, the ball and timing extremes with their centres, the workers' CPU seconds, the parent's peak RSS, bar counts, the npz size, and start and finish times. Cost: none.
