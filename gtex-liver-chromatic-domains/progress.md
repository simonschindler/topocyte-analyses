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
