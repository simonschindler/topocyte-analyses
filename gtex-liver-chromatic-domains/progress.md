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
