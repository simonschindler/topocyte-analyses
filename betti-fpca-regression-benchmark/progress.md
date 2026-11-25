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

Claude's calls while building (2026-11-24), each with its reason. None changes a number of `design.md`.

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
