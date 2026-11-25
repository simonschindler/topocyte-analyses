# Ledger: Betti-curve fPCA regression against the persistence-diagram kernel

## Request (Simon, 2026-11-24)

With the paper's PDF: "Benchmark fPCA regression using inner product space of betti curves against their reported
performances". Then: "whether Betti fPCA regression beats their kernel; download the repo. report in topocyte
analyses." Then he chose approach B of `design.md`.

## Rulings

### Simon (2026-11-24)

`design.md` ("Rulings") has them in full. The design was committed in 7075684, before any result.

- **The question:** does Betti fPCA regression beat their kernel?
- **Data:** download upstream (done: a scratch clone at `fe7d652`, 203 MB with its history).
- **Report:** in topocyte-analyses, as a study folder.
- **Approach B:** the FPCR head-to-head, as laid out on 2026-11-24 (including the 100 fluid splits), plus an
  exploratory arm that runs `PD_kernel`'s estimator on Betti-curve distances.

## Rulings during the build

Claude's calls while building (2026-11-24 and 2026-11-25), each with its reason, and Simon's confirmations of
2026-11-25 (items 8 and 9). None changes a number of `design.md`.

1. **The whitelist's round-trip test pickles at protocol 4,** as all 2,068 upstream pickles do; the whitelist stays
   the design's 11 globals (protocol 5, Python 3.14's default, needs a 12th, `numpy._core.numeric._frombuffer`).
2. **The test and verification oracles live in `oracles.py`,** shared by `test_study.py` and `verify_results.py`
   and apart from the code they check.
3. **Arm K's kernel keeps upstream's `np.e **` expression,** so that K rounds as upstream did (G1); **the
   closed-form LOO subtracts median(y) first,** which is exact, since every design has the intercept.
4. **The plan's rank rule σ_min ≤ 1e-12 σ_max is kept,** though `design.md` does not state it: it only refuses
   rank-deficient designs.
5. **Every stage checks the upstream pin (`common.check_upstream`),** as `design.md` asks: "refuse to use a clone at
   any other commit".
6. **The per-dataset function is `matrices.dataset_matrices(diagrams, shift=0.0, with_w2=True)`,** keys
   `{kind}_{h}`: the plan's test names it only as "the stage's per-dataset function".
7. **G4 is relative to each matrix's largest entry:** per time and dimension, max |G(s + 10) − G(s)| ≤ 1e-12 ×
   max |G(s)|, as G3 is relative to the largest score. `design.md`'s "1e-12 relative" names no reference, and
   entry by entry it would test the float64 rounding of the shifted endpoints on tiny overlaps, not a part clipped
   below 0, which is what G4 is for. Measured: at most 4.97e-15 of the largest entry, in the failed run and the
   passing one alike; entry by entry up to 2.98e-8, on entries of 3e-7 to 6e-5, where both Grams lie within
   3.4e-14 of the exact full-line values (rationals). `matrices.json` keeps the entry-wise worst for information.
8. **G3 compares the first min(K_MAX, rank) score columns per dimension,** the only ones arm F can use, relative to
   the largest score over them. Reason: topocyte's `FPCA.transform` is inaccurate in near-null components, by up to
   8.5e-6 of the largest score, and misses its own fitted scores by the same amounts, while the Gram projection
   returns the fitted scores to 5e-13. Over the compared columns the 7 G3 cells agreed to 4.5e-11 in Task 4's probe.
   `arms.json` keeps the worst over all columns for information. Ruled by the controller, confirmed by Simon on
   2026-11-25 ("Keep both").
9. **Ruling 7 (G4) confirmed by Simon** on 2026-11-25, with ruling 8 ("Keep both").
10. **The arms stage refuses `matrices.npz`** unless `matrices.json` records G2 and G4 passed and the npz's sha256
    equals the one it records (`matrices.load_matrices`), so that an npz left by an earlier run is never used.
11. **G3 also requires equal ranks and finite scores:** a cell whose ranks differ, or whose scores are not all
    finite, fails, its worsts inf. These are implementation checks, not rulings of Simon's. Unequal ranks cannot
    happen with bit-equal Gram matrices.
12. **The readouts stage refuses `arms.npz`** unless `arms.json` records G1 and G3 passed and the npz's sha256
    equals the one it records (`run_study.load_arms`), as item 10 does for `matrices.npz`. `readouts.json` records
    the sha256 of the bytes it loaded.
13. **The descriptive readouts' units and the tables' SDs** are interpretive choices that `design.md` does not state:
    it asks for "Δ by size and time, with the same interval recipes", so each keeps the unit of its task's recipe.
    - Interpolation by size resamples that size's 10 runs.
    - Fluid by time and by size resample the 100 splits, each by its mean Δ over that time's 5 sizes or that size's
      5 times.
    - Fluid on splits 0–4 resamples those 5 splits, each by its mean Δ over the 5 times × 5 sizes.
    - Every SD in the tables has ddof 1, as upstream's tables use: its fluid `R2_std` is the ddof-1 SD of its
      splits' R².

## Results

- 2026-11-24 13:48: `fetch_upstream.py --source <the design's scratch clone>`, in the foreground, 2.1 s wall. It
  cloned upstream into `upstream/` and checked out `fe7d652eff478ffe6081c7e6765458651ad3cd74` detached: the tree is
  clean, and `origin` names the GitHub URL. `results/upstream.json` records the SHA, the source's path and
  `provenance` (fetch_upstream.py md5 15ab6d0c…, common.py 18810fa1…, the committed files). A new line in the
  repository's `.gitignore` ignores `upstream/`. 5 tests pass with `-W error`.
  - Checked after the run (a scratch script, not kept): `safe_pickle.load_pickle` reads all 2,068 upstream pickles
    (2,042 DataFrames, 19 dicts, 6 arrays and 1 list) under `-W error` and refuses none. On scratch clones,
    `ensure_upstream` refuses an untracked file, a modified file, and an `upstream/` that is not a clone inside
    another repository.
  - **The whitelist test pickles its array at protocol 4,** as all 2,068 upstream pickles are (a static opcode
    scan). The plan's test pickled at the default protocol, which Python 3.14 raised to 5, and protocol 5 pickles a
    contiguous array through `numpy._core.numeric._frombuffer`, which the whitelist refuses. The whitelist stays the
    design's 11 globals.
- 2026-11-24 14:15: `inputs.py`, in the foreground, 5.4 s wall (3.8 s inside the stage), exit status 0. **Gate G5
  passed:** the 100 interpolation datasets equal their seed-0 regeneration; the Rips diagrams of all 10,600 clouds
  (10,500 interpolation, 100 curvature), H0 and H1, equal `ripser.ripser`'s as birth–death pairs, none differing;
  all 2,000 fluid diagrams are present. The fluid diagrams hold 3 to 6,493 bars (median 1,535). Their lowest birth is
  −25.093391940453845, so the shift is ⌈25.09…⌉ = 26 = `FLUID_SHIFT`. Every diagram has one essential bar in H0 and
  two in H1. Diagrams without a finite bar at t = 1, 2.5, 5, 7.5, 9: in H0 13, 6, 8, 9, 6; in H1 14, 10, 12, 8, 10.
  `results/inputs.json` records all of it. 15 tests pass with `-W error`.
  - Checked after the run (a scratch script, not kept): the md5s in `inputs.json` equal the files. `fluid_cached_w2`
    is complete, finite and symmetric with a zero diagonal at all five times in both dimensions, 50 random entries
    of each equal the store's, and no off-diagonal entry is 0. Diagrams with only essential bars, no finite bar in
    either dimension: 9, 3, 3, 2, 3 at the five times. The plan's Review Focus 1 gives 6–14 per time for these;
    that range is one dimension's, and an empty curve in one dimension already gives that dimension's zero Gram row.
- 2026-11-24 17:17: `matrices.py`, in the background, 545 s wall (9.1 min), peak RSS 0.81 GB, exit status 1: a gate
  failed, so `results/matrices.npz` was not written. `results/matrices.json` records the run.
  - **Gate G2 passed:** the scipy W2 equals upstream's cached fluid distances on 182 checks of 100 same-time pairs
    (`default_rng(2)`; 82 in H0, 100 in H1; n + m median 948, at most 3,857 finite bars), 19 of them large (more
    than 1,000 finite bars on both sides). The worst relative difference is 6.3e-16 (5.6e-16 among the large
    checks); none exceeds 1e-12. 279 s.
  - **Gate G4 failed as run, entry by entry,** while relative to each matrix's largest entry it holds. The fluid
    Gram matrices at shifts 26 and 36 differ, entry by entry, by up to 3.0e-8 of the entry (t = 1, H0), with 2,106
    to 5,061 entries above 1e-12 per matrix; relative to the largest entry, by at most 5.0e-15 (t = 9, H1). The
    worst entries are tiny (3e-7 to 6e-5, where the largest entries are 846 to 1.6e5): their bars overlap over
    1e-7 to 1e-5 only. Against the exact full-line inner product, computed in rationals from the unshifted bars,
    the worst entry of every time and dimension is off by at most 1.8e-14 at shift 26 and 3.4e-14 at shift 36:
    the rounding of the shifted endpoints, up to half an ulp each (3.6e-15 above 32). The endpoints are full
    float64 values, and (v + 26) − 26 gives v back for only 9–20% of them (5–15% at 36). No part of a finite bar
    lies below 0: the lowest finite birth is −24.9999 (t = 1, H0), 1.0 above 0 once shifted. Entry by entry, no
    shift of these endpoints can meet 1e-12, so G4's "1e-12 relative" needed another reference than the entry
    itself: ruling 7 takes the matrix's largest entry.
  - The rest ran: the 100 interpolation datasets in 41.7 s, curvature in 0.9 s, the five fluid times with G4 in
    220 s (41–48 s each).
- 2026-11-24 17:31: `matrices.py` again, after ruling 7, in the background, 524 s wall (8.7 min), peak RSS 0.81 GB,
  exit status 0. **Gates G2 and G4 passed.** G2 made the same 182 checks, with the same worst relative difference,
  6.3e-16. G4, relative to each matrix's largest entry: at most 4.97e-15 (t = 9, H1), against the tolerance
  1e-12; entry by entry, for information only, as before (worst 2.98e-8). `results/matrices.npz` holds the 738
  arrays: 96.0 MB, sha256 `6ac8e74e…`, both recorded in `results/matrices.json`. Parts: G2 264 s, interpolation
  40.8 s, curvature 0.9 s, the five fluid times with G4 215 s.
  - Checked after the run (a scratch script, not kept): the md5s in `matrices.json` equal the files. The npz holds
    exactly the specified keys. Every matrix is symmetric and finite, and every W2 and L² matrix has a zero
    diagonal. The interpolation sizes and targets, `curv_y` and `fluid_y` equal the loaders', and every fluid W2
    matrix equals `fluid_cached_w2`. At t = 1 the H0 Gram has 13 zero rows: the 13 empty H0 curves of the inputs
    stage.
- 2026-11-25 11:41: `run_study.py arms`, after rulings 8–11, in the background, 177.7 s wall (3.0 min), peak RSS
  1.58 GB, exit status 0. **Gates G1 and G3 passed;** arms F and B then ran.
  - **G1** compares arm K with upstream's published results, against the tolerance 1e-9 absolute:
    - interpolation's 20 values, each size's mean and SD: worst |difference| 1.25e-16;
    - curvature's MSE: 8.33e-17, which is pandas' default parse of the reported value, not a difference (see the
      readouts line);
    - fluid's 125 values (splits 0–4): 3.61e-16.
  - **G3** (ruling 8) compares the projected test scores with `FPCA.transform` in 14 cells × dimensions, ranks
    equal in all of them:
    - worst over the compared columns: 2.18e-11 (interpolation), 4.48e-11 (curvature) and 1.30e-11 (fluid), against
      1e-9;
    - over all columns, for information: 5.05e-9, 2.24e-9 and 8.53e-6 (fluid t = 9, H0), as in Task 4's probe.
  - Parts: loading 0.1 s, arm K 23.2 s, G1 under 0.1 s, G3 12.7 s, arm F 117.1 s, arm B 23.2 s. The fluid cells
    take nearly all of each arm's time.
  - Outputs: `results/arms.json`, and `results/arms.npz` with its 31 arrays (6.4 MB, sha256 `510348c6…`, recorded in
    the json). The md5s in `arms.json` equal the scripts committed in f94fea5.
- 2026-11-25 12:06, and again at 12:28 after the review's fixes: `run_study.py readouts`, in the foreground, 1.4 s
  and 1.5 s wall (0.1 s inside the stage), peak RSS 0.30 GB, exit status 0 both times. It read `results/arms.npz`
  through `run_study.load_arms` (ruling 12; sha256 `510348c6…`, the bytes loaded) and wrote `results/readouts.json`.
  - The rerun reads upstream's reported values exactly (below), adds the fluid readout by size, and records the
    sha256 that `load_arms` computed. Every other number of the 12:06 run is unchanged, bit for bit, the verdicts
    included.
  - **The verdicts,** by `design.md`'s rule fixed in advance: Δ > 0 favours F, and every interval is a 95%
    percentile bootstrap of 10,000 resamples from its own `default_rng(0)`. **Arm F is below `PD_kernel` on all
    three tasks** (copied from `readouts.json`, rounded to 4 significant digits):
    - interpolation: Δ = R²(F) − R²(K), mean −0.02401, 95% interval [−0.03837, −0.01141]: **below**;
    - curvature: Δ = MSE(K) − MSE(F), mean −0.2478, 95% interval [−0.4340, −0.09194]: **below**;
    - fluid: Δ = R²(F) − R²(K), mean −3.283, 95% interval [−5.812, −1.507]: **below**.
  - Exploratory, no verdict, with the same recipes:
    - B against K: interpolation −0.02512 [−0.03123, −0.01954], curvature −0.02964 [−0.08372, +0.02166], fluid
      −0.1388 [−0.1441, −0.1334];
    - F against B: interpolation +0.00111 [−0.01391, +0.01470], curvature −0.2182 [−0.4146, −0.04948], fluid
      −3.144 [−5.675, −1.367].
  - Descriptive, no verdict (`readouts.json` holds the rest; the units and SDs are ruling 13's):
    - fluid by training size, Δ(F, K): n 25 −15.34 [−28.01, −6.457]; n 50 −0.3164 [−0.3391, −0.2946]; n 100
      −0.2654 [−0.2746, −0.2562]; n 200 −0.2487 [−0.2574, −0.2397]; n 300 −0.2434 [−0.2521, −0.2349];
    - fluid on the paper's splits 0–4 alone: Δ(F, K) −4.142 [−8.839, −0.7448];
    - curvature's MSE: K 0.2258 (the reported `PD_kernel`), F 0.4736, B 0.2554;
    - F's chosen (K0, K1): (1, 2) for curvature; 45 distinct pairs over the 100 interpolation datasets and 239 over
      the 2,500 fluid cells, (1, 1) the most frequent in both, (0, 0) in 120 fluid cells.
  - **Reported values, read exactly.** pandas' default float parser misreads 52 of upstream's 101 reported values in
    their last bits (by at most 1.1e-16), so `readouts.py` reads them with `float_precision="round_trip"`, as
    Python's `float` reads their text.
    - G1's curvature gap of 8.33e-17 was this parse, not a difference: arm K's MSE equals the CSV's
      0.22575596585327898 bit for bit.
    - With the exact parse, G1's worsts would be 5.55e-17 (interpolation), 0 (curvature) and 4.44e-16 (fluid).
    - G1 keeps the default parser: at 1e-9 the parse cannot matter, and `arms.json`'s provenance stays that of the
      code that ran.
  - Checked after the runs (scratch scripts, not kept):
    - All nine primary and exploratory intervals (three pairs × three tasks) equal the brief's recipe, written out
      literally on `arms.npz`, bit for bit.
    - Every reported value in `readouts.json` equals the float of its CSV text.
    - Arm K's columns of the tables equal upstream's reported values to 2.2e-16: interpolation 1.1e-16, curvature
      0, fluid splits 0–4 2.2e-16. Arm K's ddof-1 SD over splits 0–4 matches the reported fluid `R2_std` to
      1.7e-16, where ddof 0 misses by 1.0e-2.
  - For information, not a readout: arm F fails badly in a few fluid cells, which make the size of fluid's mean Δ
    but not its sign.
    - F's fluid R² has median 0.085 (K 0.365, B 0.178); 755 of the 2,500 cells lie below 0, 96 below −1 and 51
      below −10.
    - All 51 below −10 have n 25, and 32 of them chose K0 + K1 = 23 = n − 2, the most the rule allows. The worst,
      −2,518 (t 2.5, split 92), chose (6, 17).
    - The readout by size shows it: Δ(F, K) is −15.34 at n 25, and −0.24 to −0.32 at the other four sizes, each
      interval below 0.
    - Δ(F, K) lies above 0 in 1.4% of the fluid cells, and its median is −0.25.
