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

## Task 6: features (2026-11-22)

- **Module:** `features.py`, run as `features.py` with no subcommand. It reads `results/pilot.csv` and `results/bars/*.npz` and writes `results/features.npz`: 45,000 rows, slides in slide-index order, then centre index 0..1499.
  - **Per centre:** `slide_index`, `half`, `centre_row`, `xy`, `fit` (in its slide's fit sample); `L1` (log10 cells, then the four class fractions); `P` (pair counts minus `chromatic.pair_expectation`, called on one ball's (4,) counts at a time).
  - **Per fitting set f (A, B, all):** `blk_obs_{f}` (N, 16, 9), `L2_obs_{f}` and `L2_ctl_{f}` (N, 20), `s0_{f}`, `kept_{f}`, `dropped_{f}` and `gate_scores_{f}` (16,), `evr_{f}` (16, 9), `mfa_evr_{f}` (20,).
  - **Diagnostics:** `zero_spread_nonzero` (16,) and `gate_gram` (16,).
  - **One block at a time, in two passes over the slides.**
    - Pass 1 reads each slide's bars and extracts the block's 20 labellings. It keeps every centre's observed and control s², its exact zero test, and the fit sample's residuals.
    - Then come the three fits.
    - Pass 2 reads each slide again and scores every centre's observed and control residuals in each fit's basis.
    - No copy of the bars is written.
- **The controller's rulings, applied:**
  - **Fit sample:** the key `[SEED, slide_index, 0, 0xF17]` draws 400 of a slide's 1,500 centres without replacement, sorted. Each slide has one sample: the halves fit on their 15 slides' samples (6,000 centres) and all on 12,000.
  - **s0:** s0² is the median positive observed s² over the fitting set's fit sample. The fit's s0, components and MFA weights are applied unchanged to all 45,000 centres.
  - **Dropped blocks:** a block with no fit centre of s > 0, or whose fit residuals are all zero, is dropped. Its `blk` columns are 0, it is left out of that fit's MFA, and `dropped_{f}` is set.
  - **Rank:** see Claude's calls.
  - **Control:** labelling 1 minus the mean of labellings 2..19, with its own spread (2..19) and the observed fit's s0. It is scored in the observed components and projected with the observed MFA's weights, centring and loadings.
  - **`zero_spread_nonzero`** counts over all 45,000 centres.
- **Tests:** `test_features.py`, 19 tests, pass in 4–8 s with warnings as errors and the pinned topocyte on `PYTHONPATH`. The study suite: 100 tests in 86 s.
  - **The brief's five:**
    - the level-1 features of [[50, 30, 15, 5]];
    - the control is lab1 − mean(lab2..19) by inner products, and the observed lab0 − mean(lab1..19); labelling 0 plays no part in the control;
    - s0² is the median of the positive s² only;
    - a zero-spread centre with a non-zero residual keeps a finite value through s0 (Review Focus 1);
    - the MFA projection of the fit sample equals its PCA scores to 1e-12.
  - **Beside them:**
    - the fit-sample key, against the 3-entry key it replaced;
    - `P` against per-row `pair_expectation` calls;
    - `take` and `concat`;
    - the labellings follow the owner code;
    - the moderated scaling;
    - `fit_block`: both drop rules, its scores against `fpca_from_gram` (gate 2), and a rank-2 block that keeps 2 components;
    - gate 1 against topocyte, and that it catches non-unit weights;
    - `mfa` refuses fewer than 20 components;
    - `level2` projects the control in the observed basis, and a dropped block plays no part;
    - `block_features` scales each residual with its own spread and the fit's s0, which is the median positive observed s² of that fit's sample;
    - `run` on four synthetic slides (24 centres, fit samples of 12): every key, shape and row, the slides' arrays, the dropped absent class, `zero_spread_nonzero`, the gates, and centred fit samples;
    - `run` again after changing a centre outside the fit samples: only that centre's features move, never s0, the fractions, the bases or the MFA, and a change of its labelling 0 leaves the control untouched.
  - **Deliberate breaks:** 19 of 19 caught, on a scratch copy of `features.py`:
    - control weights −1/19;
    - the control residual from labelling 0;
    - the control scaled with the observed spread;
    - the s0 median over every s², zeros included;
    - the 3-entry fit-sample key;
    - the zero test without the factor R, so a rounding residue counts as non-zero;
    - the control's MFA fitted on its own;
    - the kept count ignoring `fpca_from_gram`'s;
    - the kept count not capped at 9;
    - dropped blocks kept in the MFA;
    - the MFA weight from the second column;
    - no sign rule;
    - `take` sorting its index;
    - labellings off by one;
    - the halves fitted on misaligned rows;
    - level 1 with the natural log;
    - `P` with its sign flipped;
    - the fPCA fitted on every centre;
    - gate 1 on weighted curves.

    For two of these faults (the control scaled with the observed spread, the control's own MFA), the only failing test is one written after the first green run, when I saw no test would catch them.
- **Profile before the run** (a scratch script on block 1, epithelial domain H1, the block with the most distinct endpoints, over all 30 slides):
  - **Time:** 520 s for the block.
    - Pass 1 took 38 s and pass 2 146 s.
    - The all-slides fit took 244 s: the Gram matrix 77 s, `fpca_from_gram` 147 s, and the components with the fit sample's scores 20 s.
    - Each half took about 45 s.
    - Peak RSS was 7.07 GB.
  - **Gates:** gate 1 2.2e-15, gate 2 0.7–1.3e-14.
  - **Component tables:** 11.3M breakpoints for the all-slides fit (φ 0.81 GB) and 5.3–6.0M for a half.
  - **Φ is not cached.**
    - One rebuild of Φ takes 0.08 s for the all-slides fit and 0.04 s for a half.
    - So the 180 `score` calls of a block spend about 10 s on it, 2% of the block.
    - `curves.py` is unchanged.
  - **Scoring:** one slide's observed residuals (2.4M bars) score in 0.72 s, 0.3 µs per bar.
  - **`fpca_from_gram` alone,** on a synthetic Gram matrix handed over without a name: 145 s and 4.97 GB peak RSS at 12,000 curves, 17.5 s and 2.25 GB at 6,000.
- **The run** (`logs/features.log`): 22:45–23:53 on 2026-11-22, on AC power inside `caffeinate -is`.
  - **Time:** 4,059 s of wall time and 7,344 s of CPU.
    - `fpca_from_gram` took about 2,880 s of it: 2,303 s for the 16 all-slides fits and about 18 s per half fit.
    - Pass 2 took 506 s, the all-slides Gram matrices 254 s, and pass 1 159 s.
    - The blocks took 182–524 s each.
  - **Memory:**
    - Peak RSS was 9.85 GB, and the peak memory footprint 12.8 GB (`/usr/bin/time -l`), which also counts compressed pages.
    - Sampled between fits, RSS was 2.4 and 4.9 GB, and the footprint 8.6 GB and, a minute later, 5.0 GB. Of the 5.0 GB, 2.3 GB was malloc's cache of freed large regions, mostly swapped out.
    - System swap reached 3.4 GB. The block times matched the profile, so swap did not slow the run.
  - **Output:** `results/features.npz`, 206 MB, every array finite (`*.npz` is ignored).
  - **Gates:** gate 1 at most 2.2e-15 (bound 1e-10). Gate 2 at most 2.6e-13 (A), 2.2e-13 (B) and 5.9e-13 (all) (bound 1e-9). In all 48 block fits, the fit sample's pass-2 scores equal the gated scores bit for bit.
  - **Dropped blocks: none.** Every block has s > 0 and non-zero residuals in every fit sample, and every block keeps 9 components in every fit.
  - **`zero_spread_nonzero`:** 1,952 centre-blocks in all.
    - The four image H1 blocks hold almost all of them: epithelial 81, stromal 314, immune 715, other 837 (up to 1.9% of a block's centres).
    - Two domain H1 blocks hold the rest: immune 4, other 1.
    - The builder check's 80 balls found the case in image H1 only.
  - **MFA:** all 16 blocks in each fit.

    | fit | first 20 components explain | first five components | 20th |
    |---|---|---|---|
    | A | 0.624 | 0.115, 0.069, 0.061, 0.045, 0.043 | 0.012 |
    | B | 0.582 | 0.105, 0.070, 0.046, 0.040, 0.033 | 0.013 |
    | all | 0.603 | 0.110, 0.069, 0.054, 0.042, 0.038 | 0.012 |

  - **Per block:** s0 per fit, the fraction of variance explained by the 9 components (FVE) per fit, the first component's share in the all-slides fit, the zero-spread non-zero count, and the gates (gate 2 is the largest of the three fits).

    | block | class, module, degree | s0 A / B / all | FVE A / B / all | first component (all) | zero spread, non-zero | gate 1 | gate 2 | wall |
    |---|---|---|---|---|---|---|---|---|
    | 0 | epithelial domain H0 | 79.43 / 79.13 / 79.29 | 0.936 / 0.933 / 0.935 | 0.550 | 0 | 1.3e-15 | 4.8e-15 | 344 s |
    | 1 | epithelial domain H1 | 109.7 / 107.4 / 108.7 | 0.728 / 0.727 / 0.728 | 0.206 | 0 | 2.2e-15 | 1.3e-14 | 524 s |
    | 2 | epithelial image H0 | 40.71 / 41.06 / 40.84 | 0.959 / 0.958 / 0.959 | 0.668 | 0 | 1.5e-15 | 1.4e-14 | 298 s |
    | 3 | epithelial image H1 | 67.12 / 66.02 / 66.53 | 0.780 / 0.774 / 0.777 | 0.261 | 81 | 1.7e-15 | 1.0e-14 | 331 s |
    | 4 | stromal domain H0 | 122.1 / 123.0 / 122.6 | 0.910 / 0.909 / 0.909 | 0.373 | 0 | 7.4e-16 | 9.6e-15 | 261 s |
    | 5 | stromal domain H1 | 100.5 / 93.17 / 96.73 | 0.655 / 0.648 / 0.649 | 0.153 | 0 | 1.1e-15 | 4.4e-14 | 285 s |
    | 6 | stromal image H0 | 34.16 / 34.34 / 34.25 | 0.939 / 0.937 / 0.938 | 0.600 | 0 | 8.4e-16 | 9.9e-15 | 219 s |
    | 7 | stromal image H1 | 6.532 / 6.177 / 6.322 | 0.578 / 0.567 / 0.569 | 0.144 | 314 | 4.2e-16 | 8.9e-14 | 189 s |
    | 8 | immune domain H0 | 121.8 / 121.6 / 121.7 | 0.942 / 0.936 / 0.939 | 0.429 | 0 | 6.4e-16 | 5.7e-14 | 214 s |
    | 9 | immune domain H1 | 65.34 / 60.11 / 62.90 | 0.617 / 0.592 / 0.603 | 0.140 | 4 | 7.6e-16 | 4.5e-14 | 221 s |
    | 10 | immune image H0 | 21.78 / 22.17 / 21.95 | 0.969 / 0.965 / 0.967 | 0.736 | 0 | 9.8e-16 | 2.2e-14 | 201 s |
    | 11 | immune image H1 | 3.464 / 3.900 / 3.689 | 0.495 / 0.546 / 0.499 | 0.119 | 715 | 5.1e-16 | 5.9e-13 | 184 s |
    | 12 | other domain H0 | 119.6 / 119.4 / 119.4 | 0.937 / 0.927 / 0.932 | 0.359 | 0 | 6.3e-16 | 5.9e-14 | 206 s |
    | 13 | other domain H1 | 63.86 / 58.51 / 61.44 | 0.631 / 0.576 / 0.601 | 0.133 | 1 | 8.4e-16 | 1.1e-13 | 204 s |
    | 14 | other image H0 | 21.12 / 21.44 / 21.30 | 0.950 / 0.945 / 0.948 | 0.617 | 0 | 1.1e-15 | 6.8e-14 | 195 s |
    | 15 | other image H1 | 5.174 / 4.176 / 4.660 | 0.604 / 0.564 / 0.578 | 0.148 | 837 | 8.5e-16 | 1.9e-13 | 182 s |

    s0 is in px: a squared L² norm integrates a count over squared radii (px²).
  - **The halves against each other** (the larger s0 over the smaller):
    - s0 differs by at most 1.8% in the eight H0 blocks and by 1.7–2.2% in the epithelial H1 blocks;
    - by 5.7–9.2% in the stromal, immune and other domain H1 blocks and in stromal image H1;
    - by 12.6% and 23.9% in immune and other image H1, the two blocks with the fewest bars and the smallest s0 (3.5–5.2);
    - the 9 components' FVE differs by at most 0.054 (other domain H1).
  - **Descriptive only (Task 7 runs the control's rule):** in the all-slides basis, the control's level-2 columns have a mean SD of 0.765 against 1.115 for the observed, and its first column 0.59 against 2.26 (0.77 before the final-review fix wave, which corrected it).
  - **Single-class balls occur:** some balls are all epithelial or all stromal (a fraction of 1.0 in `L1`). The smallest ball holds 11 cells (log10 1.04), the largest 744 (log10 2.87).

Claude's calls:
- **The rank ruling's threshold never binds.** The ruling keeps a block's components with an eigenvalue above 1e-12 times its largest, at most 9.
  - `fpca_from_gram` has by then dropped every eigenvalue up to 1e-10 of the Gram matrix's trace (its default `rtol`), and the trace is at least the largest centred eigenvalue.
  - So every component it returns passes the ruling's threshold, and a block keeps the first 9 it returns, or all of them when there are fewer.
  - This showed when a deliberate break that removed the explicit 1e-12 filter passed every test. The filter is gone, and `features.py` says why.
  - To make the ruling's threshold bind, `rtol` would have to be smaller, close to eigh's rounding at 12,000 curves, so the default stays.
  - Cost if wrong: none in this run, where every block kept 9 components in every fit.
- **An exact zero test.** A residual is zero exactly when R·lab0 − Σ_r lab_r is, and that curve has integer step values. `nonzero` takes its squared norm as four times `spread2` of the two sets R·lab0 and Σ_r lab_r, which is exactly 0 when they agree.
  - **Why:** the residual's weights −1/19 are rounded, so a residual that should be zero keeps a residue of about 1e-17, and no test on its norm can tell it from a small one.
  - It decides `zero_spread_nonzero` and the second drop rule.
  - Cost: one more `spread2` of two sets per slide and block.
- **Keys beyond the brief's list:**
  - `fit`, the fit-sample rows: the verifier recomputes s0, the fPCA and the MFA from them;
  - `kept_{f}`, since the ruling says the count of components is recorded;
  - `gate_scores_{f}` and `gate_gram`, the gates' errors, for the report to assert.

  A dropped block's gate error, and the s0 of a block without a positive s², are 0, not NaN, so every saved array stays finite (Task 8 checks that); `dropped_{f}` says which. Cost: under 1 MB.
- **The MFA refuses fewer than 20 components** with a variance above 1e-12 of the first, instead of padding level 2 with rounding noise. That is why the synthetic run tests have 24 fit rows per half. Cost: none here (144 columns, 6,000 rows or more).
- **Fit samples are sorted.** The draw is the set; sorted, the fit rows are a subsequence of the row order. Cost: none.
- **Gate 1's 200 curves** are every 60th centre of the all-slides fit sample, spread over the 30 slides: each block's labelling-0 curves in its degree. Its error is taken entry by entry, relative, and every zero of topocyte's matrix must be exactly zero here too.
- **Gate 2 per block and fit,** relative to that fit's largest score.
- **A third check:** the fit sample's scores from pass 2 must equal the gate-checked scores of the fit bit for bit, or the run stops. It ties the written rows to the gated scores. Cost: none.
- **Two passes over the slides per block,** rereading the bars, rather than holding a whole block: the epithelial H0 blocks have 94M bars, 2.3 GB per copy of a residual set. Cost: a few seconds of reading per pass.
- **Memory of the fit:** the Gram matrix goes to `fpca_from_gram` without a name, so its symmetrized copy replaces it, and the result's n × n matrices are dropped right after. Cost: none.
- **The halves fit on their rows of the all-slides fit set,** taken from it, so pass 1 builds the fit sample's residuals once.
- **`pca` and `mfa` are copied** from the cell-type study's `common.py`, with their sign rule. `mfa` also returns its column means and loadings for the projection, and its `zip`s are strict.
- **The npz is written to `features.tmp.npz` and renamed;** `*.npz` ignores that name (Task 5's deferred minor about `.npz.tmp`).
- **One `BlockFit` outlives its block,** found while watching the run's memory. In `run`, the loop variable `fit` keeps the previous block's all-slides fit, with its component tables (up to 0.9 GB), through the next block. This is memory only and no result depends on it. It was left as it ran, so that the committed code is the code that wrote `features.npz`.

## Task 7: composition strata, subdomains, control and guard (2026-11-23)

- **Module:** `domains.py` gains its main, run as `domains.py` with no subcommand. It reads `results/features.npz` and writes `results/domains.npz` and `results/domains.json`, rows in the order of `features.npz`. Task 4's functions are unchanged.
  - **Level 1, the composition strata (`level1`).** `level1_inputs` z-scores `L1` with ddof 0: each half with its own means and SDs for its gap statistic, both halves with the training half's in each prediction-strength direction. The rule runs with k_max 10. The strata are `kmeans_fit` of all centres z-scored with all centres' means and SDs, or all 0 when k = 1.
  - **The control first (`level2` on `L2_ctl_*`).** Inside each stratum c, gap statistics on `L2_ctl_A[A∩c]` and `L2_ctl_B[B∩c]`, directions (`L2_ctl_A[A∩c]`, `L2_ctl_A[B∩c]`) and (`L2_ctl_B[B∩c]`, `L2_ctl_B[A∩c]`), k_max 6. A stratum with fewer than 500 centres in either half gets k = 1 without the rule, recorded as too small. Any control k > 1 stops level 2: no observed level-2 fit, `subdomain` −1 everywhere, no guard, and the verdict "not run: control failed".
  - **Level 2 (`level2` on `L2_obs_*`).** The same index pattern; then `kmeans_fit(L2_obs_all[c], k_c)`, and −1 where k_c = 1.
  - **The guard (`guard`),** in every stratum with k_c ≥ 2 and both directions:
    - stability (`stability_prediction`): the training half's own k-means, in its own basis, assigns the test half's centres in that basis to the nearest centroid, and `match_labels` (the Hungarian matching on the training half's contingency table) maps them to final labels;
    - pairwise (`classifier_prediction`): `HistGradientBoostingClassifier(random_state=0)` on [`P`, `L1`], raw;
    - BA_dom, descriptive: the same classifier on the 8 domain blocks' 72 scores, in the training half's basis for both halves;
    - balanced accuracy (`balanced_accuracy`): the mean recall over the classes present in the true labels, from confusion counts;
    - the bootstrap (`bootstrap_draws`, `resampled_ba`): a fresh `default_rng(12345)` per stratum and direction draws `integers(0, n, size=(4000, n))` over the n slides holding the test half's centres of the stratum, in slide-index order. Each draw sums its slides' confusion counts; the predictions stay fixed. The interval is the 2.5 and 97.5 percentiles, and a direction passes when the difference and its lower bound are both above 0.
  - **The verdict (`verdict`):** "adds", "does not add" or "not run: control failed", with a reason. "adds" needs a passing stratum holding at least 10% of all centres, compared exactly.
  - **Descriptives:** the prediction-strength curves at k = 2..k_max of every rule application (`ps_curve`), the cross-half adjusted Rand index where k ≥ 2 (`cross_half_ari`), and the contiguity (`contiguity`, with `same_slide_neighbours`).
  - **`results/domains.npz`:** `compartment`, `subdomain`, `control_k`, and the guard's 4000 bootstrap differences per stratum and direction, `boot_{c}_stab_minus_{pw,dom}_{AB,BA}` (none in this run). `*.npz` is ignored.
  - **`results/domains.json`:** level 1 (k, k_A, k_B, PS, both halves' gap curves with s, the PS curves, the ARI and each fitting set's z-scoring); per stratum its size, share, sizes and slides per half, mean `L1`, the control's and the level-2 choice with their curves, the guard and the contiguity; `control_k`, `control_passed`, the verdict, the settings, versions and timings.
- **`run_pilot.py`:** the composition null (`_composition`, Simon's strata ruling) runs after the gated cases, from the same generator, z-scored by `domains.level1_inputs`. It is recorded under `"descriptive"`, outside `"passed"`. `_half` became `domains.gap_summary`, which both JSON writers share.
- **The controller's rulings, applied:** R3 (level-1 z-scoring), T7-a (the control first, and a failed control is a result), T7-b (the verdict and its reasons), T7-c (the bootstrap), T7-d (the two predictions), T7-e (BA_dom), T7-f (the cross-half ARI), T7-g (contiguity), T7-h (the composition null), T7-i (the strata wording), T7-j (the stored draws), and the curves in `domains.json`.
- **Tests:** `test_domains.py`, 48 tests (18 new), and the study suite, 118 tests in 118 s, pass with warnings as errors.
  - **The brief's six:** the Hungarian matching; the guard on subdomains that are a function of `P` (no pass) and on noise `P` with clear level-2 subdomains (a pass); each bootstrap draw holds whole slides; the contiguity expectation equals the mean over every reassignment on a six-centre toy whose observed counts range from 0 to 4; 499 centres in one half give k = 1 and `subdomain` −1, and 500 in each run the rule.
  - **Beside them:**
    - the level-1 z-scoring of each half and direction, and the final fit on all centres' z-scores;
    - the balanced accuracy over the classes present, against scikit-learn's where every class is present;
    - a direction passes only with its lower bound above 0: hand-made predictions with a positive difference and a lower bound below 0. Its draws come from a fresh generator per direction, and the interval is their percentiles;
    - the final subdomains come from the all-slides basis;
    - the cross-half ARI compares the partitions prediction strength compares;
    - contiguity counts same-slide neighbours, whatever their stratum;
    - the verdict's rules, 10% exactly included;
    - the binding settings (k_max 10 and 6, 500 centres, 4000 draws, seed 12345, 10%, five neighbours, the domain blocks, the classifier's `random_state=0`);
    - two end-to-end runs on synthetic features: one stratum, a passing control, two subdomains and a passing guard; and a control with clusters, which stops level 2 after exactly two calls of the rule;
    - the composition null's construction, and its place in the pre-flight: after the gated cases, z-scored as level 1, outside `"passed"`.
  - **Deliberate breaks:** 33 of 34 caught, on scratch copies of `domains.py` and `run_pilot.py`. The 34th fits k-means for k = 1 instead of writing zeros, which gives the same labels. The guard test's percentile check was added when a break to the 5th and 95th percentiles passed: the hand-made predictions' draws take few values, so those percentiles coincided.
- **The pre-flight re-run** (`logs/preflight_task7.log`): 01:03–01:15 on 2026-11-23, on AC power inside `caffeinate -is`. **Passed**, 9 of 9 gated cases.
  - Every gated case gives the same k, k_A, k_B and prediction strength as Task 4's run, and its gap curves and s agree to 1.8e-15.
  - Time: 724 s of wall time, 2,714 s of user CPU; peak RSS 0.31 GB. The composition null took 151 s of it.
  - **The composition null (reported, not gated):** 22,500 × 5 per half, k = 5 (k_A 5, k_B 5), prediction strength 0.930 / 0.948. The Task 4 reviewer's draw, z-scored on each half alone, gave k = 5 with 0.91 / 0.93.

    | half | gains Gap(k) − Gap(1), k = 2..10 | stop at 1, margin |
    |---|---|---|
    | A | 0.0133, 0.0676, 0.1681, 0.2096, 0.2006, 0.1962, 0.2002, 0.2057, 0.2068 | −0.0080 |
    | B | 0.0112, 0.0671, 0.1653, 0.2043, 0.1994, 0.1957, 0.1995, 0.2062, 0.2058 | −0.0058 |

- **The run** (`logs/domains.log`): 01:15–01:20 on 2026-11-23, on AC power inside `caffeinate -is`, right after the pre-flight. `domains.py` had md5 `964d5082…` then and has not changed since.
  - **Time:** 300 s of wall time (level 1 117 s, the control 104 s, level 2 79 s, the guard none), 1,486 s of user CPU. Peak RSS 0.51 GB.
  - **Level 1: one composition stratum, k = 1.** Each half's gap rises at every k up to k_max, so k_A = k_B = 10. Prediction strength at k = 10 is 0.327 (A→B) and 0.276 (B→A), under 0.8, so k = 1: the stratum holds all 45,000 centres, 22,500 per half on 15 slides each. Its mean ball has log10 2.155 cells and fractions epithelial 0.720, stromal 0.151, immune 0.069 and other 0.060.
  - **The control: k = 1, passed.** Both halves stop at k = 1 (margins +0.0075 and +0.0053), so level 2 ran.
  - **Level 2: k = 1, no subdomains.** Each half's gap rises at every k up to k_max, so k_A = k_B = 6. Prediction strength at k = 6 is 0.338 (A→B) and 0.361 (B→A), so k = 1.
  - **Verdict: does not add**, with the reason "no compartment with subdomains". The guard did not run, so there are no bootstrap draws, BA_dom, cross-half ARI or contiguity: those fields are null in `domains.json`.
  - **The gap gains,** Gap(k) − Gap(1), and the margin Gap(1) − (Gap(2) − s_2) (positive: the gap stops at 1):

    | rule application | half | gains, k = 2..k_max | margin |
    |---|---|---|---|
    | level 1 (k_max 10) | A | 0.1020, 0.1518, 0.2124, 0.3004, 0.3521, 0.3726, 0.3928, 0.4252, 0.4572 | −0.0964 |
    | | B | 0.0461, 0.0584, 0.0835, 0.1549, 0.2336, 0.2624, 0.2738, 0.3040, 0.3241 | −0.0405 |
    | control (k_max 6) | A | −0.0051, −0.0066, −0.0054, −0.0023, 0.0035 | +0.0075 |
    | | B | −0.0029, −0.0017, 0.0091, 0.0183, 0.0271 | +0.0053 |
    | level 2 (k_max 6) | A | 0.0258, 0.0552, 0.0869, 0.1061, 0.1233 | −0.0232 |
    | | B | 0.0201, 0.0371, 0.0481, 0.0643, 0.0762 | −0.0175 |

  - **The prediction-strength curves (descriptive):**

    | rule application | direction | PS at k = 2..k_max |
    |---|---|---|
    | level 1 | A→B | 0.650, 0.516, 0.499, 0.509, 0.365, 0.427, 0.436, 0.312, 0.327 |
    | | B→A | 0.926, 0.516, 0.512, 0.571, 0.566, 0.298, 0.300, 0.264, 0.276 |
    | control | A→B | 0.790, 0.729, 0.721, 0.591, 0.626 |
    | | B→A | 0.691, 0.401, 0.601, 0.608, 0.441 |
    | level 2 | A→B | 0.737, 0.400, 0.415, 0.425, 0.338 |
    | | B→A | 0.926, 0.831, 0.526, 0.466, 0.361 |

  - **What the curves add.** The rule checks prediction strength only at the smaller half's k, here k_max at both levels. At no k from 2 to k_max does it reach 0.8 in both directions, at either level: the best is k = 2, with 0.650 / 0.926 at level 1 and 0.737 / 0.926 at level 2. So checking prediction strength at another k would not have changed either k. No rule, parameter or seed was changed after the run.
  - **The halves differ.** Half A's gains exceed half B's at both levels, and A→B is the weaker direction at k = 2 at both levels. Half A's balls hold more cells than half B's (mean log10 2.180 against 2.129) and more of class other (0.067 against 0.053).
  - **Against the composition null.** The null's gap peaks at k = 5 (0.21) and the rule takes 5, confirmed in both directions. The real compositions' gap keeps rising to k = 10, but no split of them recurs across the halves.
- **Checks after the run** (a scratch script): the strata recomputed from `L1` equal `compartment`; the JSON's sizes match; `subdomain` is −1 throughout; every saved array is finite.

Claude's calls:
- **The bootstrap resamples the slides holding the test half's centres of the stratum,** not all 15 slides of the test half. The two agree when a stratum covers every test slide. Over all 15, a draw of slides without the stratum would leave the balanced accuracy undefined, and the rulings give no rule for it. The JSON records the slides per direction. Cost: none in this run, where the guard did not run.
- **Prediction-strength curves at k = 2..k_max for every rule application,** for the curves the ruling asks for. They come from Task 4's `prediction_strength` with seed 0; only the rule's own value, at the smaller half's k, decides. Cost: 20–36 more k-means fits per application, a few seconds each.
- **The verdict's reason when every stratum has k = 1** is the ruling's wording, "no compartment with subdomains"; the other reasons name composition strata. Cost: one word.
- **`match_labels` builds a square table** over labels 0..m−1, m the larger maximum plus one, so every own label maps to a final label even if the training half lacks a subdomain.
- **The 10% share is compared exactly** (a `Fraction`), so a stratum of exactly 4,500 centres counts.
- **Kept beside the brief's keys:** each stratum's mean `L1` and slides per half, each fitting set's z-scoring, and each direction's test slides, for the report and the verifier (Task 8). The strata keep k-means' own numbering.
- **The composition null's multinomial is drawn in one vectorized call,** which equals the reviewer's loop draw for draw (checked on 50 rows).
- **`zscore` refuses a constant column,** and `same_slide_neighbours` a slide with five centres or fewer; neither occurs here.
- **The classifier's defaults include early stopping above 10,000 training rows** (scikit-learn's "auto"), with its validation split seeded by `random_state=0`. The design asks for default settings and a fixed seed.

## Task 8: verifier and report (2026-11-23)

- **Verifier:** `verify_results.py`, run with no arguments. It writes `results/verify.json` (tracked): every check with its sample, tolerance and measured value, the run's time, peak RSS and versions, and the sampled rows.
  - **The run** (`logs/verify.log`): 03:29–04:38 on 2026-11-23, after fix round 1 (below), on AC power inside `caffeinate -is`, from the committed file (md5 `dc89f50f…`, unchanged through the run). **ALL CHECKS PASSED**: 61 checks and 1 descriptive count, in 4,147 s (arrays 5 s, clustering 267 s, balls 256 s, blocks 3,618 s). Peak RSS 10.46 GB, peak memory footprint 13.75 GB.
  - **The first run,** 02:16–02:47, before the task review: 59 checks and 1 descriptive count passed in 1,833 s (peak RSS 9.17 GB), with the all-slides fPCA refitted for blocks 1 and 11 only and `L2_ctl_all` unchecked. The rerun replaces its `results/verify.json`.
  - **Trial runs before it,** on subsets: the arrays (6 s); the balls of two slides (14 s); blocks 7 (42 s) and 1 (353 s, peak RSS 8.52 GB, the heaviest block with its all-slides fit); the clustering (275 s). Every check passed in each.
  - **Routes** (the controller's rulings; the Coverage bullet below says which checks share code with what they check):
    - chalc 15.0.1 for the six-packs;
    - brute-force shuffles for the pair expectation;
    - topocyte's `betti_inner_product_matrix` for the inner products, with the weighted residual curves composed from the unit-weight inner products of each centre's 20 labelling curves;
    - an integer-count sweep and topocyte's 19 × 19 Gram matrices for the spreads;
    - `fpca_from_gram` for the components and scores;
    - a numpy eigendecomposition for the MFA and its sign rule;
    - the verifier's own gap statistic, stopping rule and prediction strength on scikit-learn's `KMeans`.
  - **Samples:**
    - 200 balls from one draw over all 45,000 rows, key `[SEED, 0xBA11, 0, 1]`, in draw order;
    - the first 20 of them for the pair counts and the brute-force spreads;
    - per block, the 10 of them with the most bars for the inner products;
    - one ball per slide, key `[SEED, 0xB17, slide_index, 1]`, recomputed bit for bit;
    - the j-th pair-count ball's 10,000 shuffles, key `[SEED, 0x5AFF, j, 1]`.

    Every key has 4 entries and a non-zero last entry; the second entry marks the purpose and lies above every slide index, so none meets a study key.

    | check | sample | tolerance | measured |
    |---|---|---|---|
    | every array of `features.npz` and `domains.npz` finite | 36 and 3 arrays | 0 arrays | 0 |
    | pilot: 30 slides, 15 per half, 5 per condition in both halves | `results/pilot.csv` | exact | 0 |
    | bars files: every array finite; owners, offsets, sort order, counts, centre draw and json | 30 slides, 447,050,046 bars and 13,355,202 codomain bars | 0 | 0 |
    | `features.npz`: slides, halves, centre rows, positions and fit samples redrawn; `L1` from the class counts; `P` from exact rational expectations | 45,000 rows | exact | 0 |
    | level 1, the control and level 2: k, k_A and k_B | three rule applications | exact | 0 |
    | level 1, the control and level 2: gap and s curves | k = 1..k_max | 1e-12 | 1.8e-15 at most |
    | level 1, the control and level 2: prediction-strength curves and the rule's values | k = 2..k_max, both directions | 1e-12 | 0 |
    | level 1: z-scoring of each fitting set; the strata labels | means and SDs; 45,000 rows | 1e-12; exact | 0 |
    | level 1: each stratum's sizes, slides per half and mean level-1 features, with the control's and level 2's sizes | 1 stratum | exact; means 1e-12 | 0 |
    | control k and passed; the verdict from the recomputed k's; the guard absent, and no contiguity or cross-half ARI where k = 1 | `domains.json`, `domains.npz` | exact | 0 |
    | balls by brute force against the saved cell and class counts | 200 sampled balls and one per slide | 0 | 0 |
    | saved labelling-0 domain, image and codomain bars against chalc 15.0.1 | 200 balls, 3,600 diagrams | equal counts, bottleneck ≤ 1e-9 of the ball's scale | 7.6e-16 |
    | the port, faces to dimension 3 and to dimension 2, every module, against chalc | 200 balls, 8,110 diagrams each | as above | 7.6e-16 |
    | saved codomain against topocyte's alpha PH | 200 balls, 400 diagrams | as above | 6.5e-14 |
    | diagrams whose raw bar counts differed before bars under 1e-12 of the scale were dropped | 20,220 diagrams | descriptive | 0 |
    | the three Betti identities at every endpoint | 200 balls, faces to dimension 2 and 3 | 0 | 0 |
    | one ball per slide recomputed with `ball_sixpacks`, bit for bit | 30 balls × 20 labellings × 16 blocks (287,965 bars), codomain, pair counts | 0 balls | 0 |
    | Qhull's Delaunay edges and class-pair counts; the expectation as a fraction | first 20 sampled balls | 0 balls | 0 |
    | the expectation against 10,000 shuffles, largest \|z\| | first 20 sampled balls × 10 pairs | 5 SE | 2.85 |
    | s0 per block and fit, from every fit centre's s² by the integer-count sweep | 16 blocks × 3 fits | 1e-12 relative | 5.3e-16 |
    | zero-spread non-zero residuals over every centre | 16 blocks × 45,000 centres | exact | 0 |
    | s² by brute force (topocyte's 19 × 19 Gram) against `curves.spread2` and the sweep; s² = 0 exactly when the references are one diagram | first 20 sampled balls × 16 blocks, observed and control | 1e-10 of the mean squared reference norm; 0 | 1.5e-15; 0 |
    | inner products: unit weights, cross matrices, composed scaled residuals, exact zeros | 10 sampled centres × 20 labellings × 16 blocks | 1e-10 | 2.5e-15, 2.3e-15, 1.4e-15, 0 |
    | fPCA A and B: fit-sample scores against `fpca_from_gram`; fractions of variance; 9 kept; the kernel route on the 200 sampled centres | 16 blocks each | 1e-9 of the largest score; 1e-9; 0 | 2.6e-13 (A), 2.2e-13 (B); 8.9e-16; 0; 9.8e-15 |
    | fPCA all: the same | 16 blocks | as above | 5.9e-13; 7.8e-16; 0; 4.4e-15 |
    | MFA A, B and all: every level-2 row and the fractions by eigendecomposition | 45,000 rows × 20 | 1e-9 | 1.1e-12 at most; 7.6e-16 |
    | MFA A, B and all: the control's level-2 rows of the sampled centres (kernel route, then the MFA) | 200 centres × 16 blocks | 1e-9 | 7.4e-14 (A), 2.3e-13 (B), 3.5e-13 (all) |

  - **Coverage** (as the final-review fix wave left it; the verifier's docstring, the README and the report's section 9 state it in their own words):
    - recomputed for every centre or fit: `L1` and `P` from the saved class and pair counts; every centre's spreads, s0 and the zero-spread counts; the block scores of every fit sample in all 48 fits, with every block's fractions of variance and kept count; every level-2 row from the saved block scores; `domains.json`'s curves, choices (each half's k, largest gains and the too-small flag included), strata, settings and verdict;
    - recomputed on samples: the bars of 230 balls (200 sampled, against chalc in labelling 0; one per slide, bit for bit in all 20 labellings; the two samples are checked to be disjoint), with their class counts; the pair counts of 50 of them (Qhull on the first 20 sampled, bit for bit on the 30); the other rows of `blk_obs_*` and the control's `L2_ctl_*` in all three bases on the 200 sampled centres;
    - shared code: the bit-for-bit bars rerun the study's port; the fPCA refits use `curves.inner_matrix`, as `features.py` did, checked against topocyte on 10 sampled centres per block only; the clustering checks reimplement the rule but call the same scikit-learn `KMeans` with the same seeds and reference draws. Those checks reproduce rather than verify;
    - not recomputed: `results/preflight.json` (the pre-flight), the pilot draw (`results/pilot.csv`, whose counts per half and condition are checked), the features run's gate diagnostics (`gate_gram`, `gate_scores_*`) and the runs' timings, sizes and versions;
    - the guard's balanced accuracies reduce to asserting the guard is absent (ruling T8-c).
- **Report:** `build_report.py` writes `report.html` (0.3 MB) in 79 s, peak RSS 6.92 GB, with no warning under `-W error` (`logs/build_report.log`; rebuilt after fix round 1). Sections follow ruling T8-a:
  1. question and rulings, with the k-rule correction, the 0.03 floor and the strata ruling dated;
  2. pilot slides and compute;
  3. the one composition stratum: its profile, both halves' gap and prediction-strength curves with the composition null for contrast, the halves' level-1 differences, the mechanism and the post-hoc random splits; no maps;
  4. no subdomains: the level-2 curves and the stratum's mean residual curves;
  5. the control;
  6. the guard did not run, and the verdict;
  7. contiguity not computed;
  8. the H&E atlas, Simon's decision;
  9. methods: the phimaker threads, the effective rank threshold, the per-block and MFA tables, the verification and the provenance.
  - **Mean residual curves (T8-b):** the exact mean, every bar kept with weight 1/n (`curves.combine`), of the 12,000 fit-sample centres' scaled residual curves per block, as `features.py` builds them. The step table comes from `curves._steps`; the curves are drawn with `drawstyle="steps-post"` and matplotlib's own path simplification, on topocyte's logarithmic filtration axis.
    - **Profile of block 1 first:** 23.0M bars, 11,289,801 breakpoints, built in 13 s and plotted in 1.5 s, peak RSS 5.3 GB.
    - **All 16 blocks:** 214,420 to 11,289,801 breakpoints (42.0M in all), 76 s. The 16-panel figure fits in memory, so it is one figure.
    - **What they show:**
      - in degree 0, the domain and image curves of the stromal, immune and other classes have negative integrals and dip furthest at α² of 154 to 315 px² (radii 6.1 to 8.8 µm);
      - in degree 1, the image curves of all four classes have positive integrals and peak at α² of 264 to 898 px² (radii 8.0 to 14.8 µm);
      - the domain H1 curves of all four classes rise above 0 (at 185 to 897 px²) before they fall below it (at 2,702 to 3,701 px²).
  - **The mechanism (T8-d),** both halves in the training half's z-coordinates at k = 2:

    | direction | \|cos\| | training half's minority | test half's own minority | midpoint offset, z-units | crossings | PS(2) |
    |---|---|---|---|---|---|---|
    | A→B | 0.993 | 17.5% of A | 15.7% of B (3,526) | +0.46 | 797 of B's 3,526 minority on A's majority side; 0 of 18,974 the other way | 0.650 |
    | B→A | 0.993 | 16.1% of B | 17.6% of A (3,964) | −0.51 | 711 of A's 18,536 majority on B's minority side; 0 of 3,964 the other way | 0.926 |

    - Every minority holds fuller balls with fewer epithelial and more stromal, immune and other cells: half B's own minority in A→B has log10 cells 2.216 against 2.113 and an epithelial fraction of 0.390 against 0.798.
    - The midpoint of half A's two centroids lies further toward the minority than the midpoint of half B's, in both coordinate systems. The midpoint offset is the training centroids' midpoint minus the test centroids' midpoint, projected on the training half's split direction (unit length); the code called it the boundary offset, and this ledger first said, wrongly, that it is where each boundary crosses that direction.
    - The task reviewer's numbers (|cos| 0.993, a 16% minority, 797 of 3,526) are reproduced, and PS(2) recomputed from the crossings equals the saved value in both directions.
  - **Post hoc, outside the rule (T8-e):** PS(2) on three random half/half splits of all 45,000 centres (keys `[SEED, 0x5B17, i, 1]`), each random half holding centres of all 30 slides:

    | halves | level 1 | level 2 (all-slides basis) |
    |---|---|---|
    | random split 1 | 0.977 / 0.997 | 0.986 / 0.976 |
    | random split 2 | 0.994 / 0.966 | 0.955 / 0.990 |
    | random split 3 | 0.990 / 0.952 | 0.957 / 0.992 |
    | slide halves, the rule's bases | 0.650 / 0.926 | 0.737 / 0.926 |
    | slide halves, the all-slides basis | — | 0.741 / 0.934 |

    High prediction strength on random halves is no evidence of a split: two random halves of one large sample give k-means a high PS(2) whether or not clusters exist, and cluster-free clouds give 0.98–0.99 (`design.md`, step 10 and the builder check; Task 4 above), the reason the rule asks the gap statistic first. What the contrast shows is that the two slide halves differ: half A's balls hold more cells, by 0.35 pooled SD in log10 cells, the largest of the level-1 differences. It cannot change the verdict. (Until the final-review fix wave this ledger said the split "moves between the two slide halves", which suggested a real split.)
  - **The look,** in the browser pane (the report served from a scratch copy on 127.0.0.1, dark theme), at 1,024 px and at the 375 px mobile preset: no horizontal page overflow, all five figures load, every table scrolls inside its wrapper. Fixed after the look:
    - the prediction-strength panels' threshold label sat on the curves at k = 2 and 3; it moved to the right end;
    - Helvetica Neue has no arrow glyph, so the legends say "trained on half A, tested on half B";
    - the verification table's first column wraps, and the per-block tables' labels are left-aligned.
- **Code edits beyond the two new files:** `chromatic.py`'s header comment only, checked identical below the docstring by its syntax tree. It now says that phimaker 0.5.0 runs up to 10 threads per call despite `num_threads=1`, and that the match with chalc holds up to bars shorter than about 1e-12 of the scale, which exactly tied clouds pair differently. No behaviour change anywhere; the edit invalidates numba's cache of `chromatic.py` once.
- **README:** the status is the outcome; the run table names every script with its code, topocyte commit and date; the venv recipe, the run order and the outputs are updated. `../README.md`'s Studies row has the verdict.
- **Tests:** the study suite, 118 tests in 123 s, pass with warnings as errors. Task 8 adds no test: the verifier is itself the check of the saved results. Lint: ruff (E, F, W, B, UP, SIM, I, line length 120) is clean and `ruff format` leaves both new files unchanged.

Claude's calls:
- **`results/verify.json` is tracked,** like the other small results; the report's verification table and methods read it. Cost: 15 kB.
- **The spreads' independent route is an integer-count sweep.** On each interval between consecutive endpoints of a centre's 19 references, R·Σβ_r² − (Σβ_r)² is an exact integer, so s² is 0 exactly when the references agree, as with `curves.spread2` and unlike the inner-product form. It matched `spread2` to about 3e-15 on one slide's blocks before the run, with identical zero sets, and gives s0 for every block and fit and the zero-spread counts over all 45,000 centres. Cost: about a second per slide and block.
- **Every fit is refitted with the Gram matrix of `curves.inner_matrix`,** itself checked against topocyte on 10 sampled centres per block; the components and scores come from `fpca_from_gram` alone. A topocyte Gram matrix of 6,000 weighted curves would need 120,000² unit-weight inner products.
- **The kernel route** scores the 200 sampled centres, observed and control, from `fpca_from_gram`'s eigenvectors and the cross inner products, apart from `curves.components` and `curves.score`.
- **Balls by brute force** (squared distance ≤ δ²) rather than a KD-tree. The 20 pair-count balls are checked to hold no two cells at one position before Qhull triangulates them.
- **Shuffles within 5 standard errors:** 200 tests, a false-alarm rate of about 1e-4; the seeded draws fix the result.
- **ECDFs, not histograms, for the halves' level-1 features:** exact steps at the data, no bins.
- **topocyte's private `_log_x_axis`** for the mean curves' logarithmic filtration axis: the topocyte figures' plain 1-2-5 labels (Simon's ruling for topocyte). The pinned checkout fixes it.
- **Constants in `build_report.py` without a results file,** named as such and checked where they can be: the provenance (the md5 sums of `features.py` and `domains.py`, the git blob, the commits); the serial time per centre measured in Task 5 (0.50 s) and phimaker's 10 threads; the design's estimates (12 h serial, 2 h on 8 processes); and, since fix round 1, the k rule's first floor (0.02) and the rank ruling's threshold (1e-12 of the largest eigenvalue).
- **`build_report.py` reads `logs/features.log`** for the features run's time, CPU and peak RSS, after checking that its first line names the md5 of this `features.py`. No results file holds them; the README says so.
- **T8-e adds the slide halves in the all-slides basis** as a reference row (0.741 / 0.934), so the random splits and the slide halves are compared in one basis at level 2.
- **T8-d adds the midpoint offset** (the boundary offset in the code), which says on which side the crossings fall.
- **The report's mean residual curves are recomputed at every build** (76 s): a cache of 670 MB served development only and went to the Trash.

### Task 8, fix round 1: the task review (2026-11-23)

The review found 2 Important issues; its Minor findings are deferred to the whole-study review. The controller's decisions: refit every all-slides block and check `L2_ctl_all`, and make every coverage statement true as written; assert the direction claim, fix the composition null's gains, format the typed numbers and assert the plurals.

- **Important 1, coverage.**
  - **Was:** the all-slides fPCA was refitted for blocks 1 and 11 only, `L2_ctl_all` was checked for finiteness only, and the README, the report's summary and section 9, and this section claimed every saved result or excused the gap with "no script reads it". That was wrong: the report's section 5 reads `L2_ctl_all`, and its random splits and per-block table read the all-slides fits.
  - **Now:** `verify_results.py` refits all 48 fits (`ALL_FIT_BLOCKS` is gone) and checks the control's level-2 rows of the 200 sampled centres in every basis, `L2_ctl_all` included (3.5e-13). Two checks were added on the way: each stratum's sizes, slides per half and mean level-1 features against `domains.json`, and no cross-half ARI where k = 1 in the guard-absent check.
  - **The statements:** the verifier's docstring has a Coverage paragraph, and the README, the report's summary and section 9, and the Coverage bullet above now say what is recomputed (the bars on 230 sampled balls, every fit, every centre's spreads, `L1` and `P`, every level-2 row, `domains.py`'s results) and name what is not: the pre-flight, the pilot draw, the features run's gate diagnostics, and the runs' timings, sizes and versions.
- **Important 2, prose.**
  - **The direction claim** is asserted (|cos| above 0.95 in both directions), and the summary now says "nearly the same direction".
  - **The composition null's gains** are now the gains at each half's stopping k, Gap(5) − Gap(1) = 0.210 and 0.204, tied to `null["k"]` by an assertion. Before, the report printed the largest gains, and half B's (0.206) lies at k = 9.
  - **Typed numbers** are now formatted from the saved files or the study's constants: the halves' 22,500 centres, the 12,000 fit-sample centres, the 30 slides with 5 per condition, the 3 + 2 split, alternating, with 15 per half, 150 µm, 303.52 px, 0.4942 µm/px, 144 columns, B = 20, `n_init`, the guard's 15 features, 95% interval and 10% share, the 5 neighbours, the composition null's construction (from `results/preflight.json`), the versions (from `results/venv_freeze.txt`), the face dimension, `fpca_from_gram`'s 1e-10, the outcome k's, and the table headers.
  - **The plurals** are asserted for both halves (sizes and slides per half) and for both random halves (all 30 slides in each).
- **The rerun:** `verify_results.py` ALL CHECKS PASSED (61 checks and 1 descriptive count, 4,147 s, peak RSS 10.46 GB; "The run" above). `build_report.py -W error` rebuilt `report.html` in 79 s with no warning. ruff check and `ruff format --check` are clean on both files. No study module changed, so the suite was not rerun.

### Final-review fix wave (2026-11-23)

The whole-study review (86179f0..1db991c) found 0 Critical, 2 Important and 5 new Minor issues, and triaged 4 carried minors to fix before hand-over. Nothing it found moves a saved result or the verdict. One fixer; the edits touch only `build_report.py`, `verify_results.py`, the study README, `../README.md`, this ledger and `../.gitignore`, so every study module keeps its md5.

- **I1, the post-hoc random splits.** Prediction strength of 0.95 or more on random halves was stated without what a cluster-free cloud gives. The report's summary and sections 3 and 4, `../README.md` and the T8-e note above now say that cluster-free clouds give 0.98–0.99 at k = 2 too, so the contrast shows that the slide halves differ (half A's balls hold more cells, by 0.35 pooled SD in log10 cells, asserted to be the largest difference), not that a split exists. The three records are quoted in `build_report.py` (`CLUSTER_FREE_PS2`) and found in `design.md` and this ledger at every build.
- **I2, the second study.** The summary, section 8 and the README status now say that the design's fallback, the level-1 strata as the second study's regions, does not hold either: one stratum, so no regions at either level, and where they come from is Simon's decision.
- **M1:** the README's run block creates `logs/`.
- **M2:** the summary says the control gives k = 1 in both halves, not that it finds no structure; section 5 contrasts it with the observed level 2 (the gaps rise at every k there, asserted).
- **M3:** section 1's note says that the rule was committed in 0b031a3 on 2026-11-21, before the bars began (2026-11-22), and ran unchanged, while the strata wording of `design.md` and `plan.md` came with the domains results in 381d0ce. `build_report.py` checks it from the git history: the design and the plan were added before the bars began, the rule's functions and constants stand verbatim in 0b031a3's `domains.py`, and the strata phrases first appear in 381d0ce.
- **M4:** Task 6's control SD is 0.765 (was 0.77).
- **M5:** `results/verify.json` records the md5 of `verify_results.py` and the sha256 of every gitignored file the verifier read (`features.npz`, `domains.npz`, the 30 bars files and the sibling's `density.npz`). `build_report.py` asserts the md5, and the sha256 of `features.npz` and the bars files it reads itself (3.8 s).
- **L22:** `*.tmp` in `../.gitignore`, so a temporary file left by an interrupted atomic write cannot be staged.
- **L44:** the report's section 9 and the verifier's docstring name the generators that are plain seeds: the pilot draw `default_rng(SEED)`, the gap references `default_rng(0)`, the pre-flight `default_rng(1)` and the guard's bootstrap `default_rng(12345)`. The report formats them from `domains.json`, `preflight.json` and `domains.py`.
- **L46:** the T8-d offset is the midpoint offset along the training split direction, in `split_mechanism`'s docstring, the report's prose and table header, and the T8-d note above.
- **L53, coverage.** The verifier's docstring, the README, the report's summary and section 9 and the Coverage bullet above say what is recomputed for every centre or fit, what only on samples, which checks share code with what they check, and what is not recomputed; "independent" is gone. New checks back the statements: each half's own k, its largest gain and the too-small flag of every rule application, the rule's settings and level-1 features in `domains.json` (`RULE` names them once in the verifier), and the disjointness of the 200 sampled and the 30 bit-for-bit balls. `verify.json` records the bit-for-bit rows and the size of the first sample, and the report asserts the disjointness too.
- **L45, L49, L52, L54:** `rtol=0` beside `atol=1e-12`; `FEATURES_MD5` is the md5 of blob f7ae634's content; every B→A prediction strength up to k = 3 is asserted to reach 0.8; the README says to run the verifier with nothing else heavy open.
- **The repeated rows (the review's recommendation):** section 4 says that in 9 of the 16 blocks at least 10% of the centres share one block-score row (all-slides fit), up to 71% in other image H1. Asserted for every block: those are exactly the centres with no bars in the block in any labelling, and in degree 0 exactly the balls with at most one cell of the class; the control's residual is zero there too, so it shares the rows.
- **Also:** "all four classes" and "the largest of the three fits" are no longer typed, and `build_report.py`'s docstring names every source of its constants.
- **Checks before the run:** ruff check and `ruff format --check` are clean on both files. The new clustering checks ran on their own (268 s, all passed), and `build_report.py -W error` built a scratch page from the previous `verify.json`, patched with the new fields and with the md5 and sha256 comparisons left out, in 86 s (peak RSS 6.63 GB) with no warning.
