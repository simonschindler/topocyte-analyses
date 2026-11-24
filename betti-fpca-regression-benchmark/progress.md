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
