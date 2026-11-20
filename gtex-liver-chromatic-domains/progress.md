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
- **Seed keys.**
  - The slide's centre draw uses `SeedSequence([SEED, slide_index, 0, 0])`.
  - Labelling r of a centre uses `[SEED, slide_index, centre_index, r]`; labelling 0 draws nothing.
  - SeedSequence pads short keys with zeros, so the plan's fit-sample key `[SEED, slide_index, 0xF17]` equals `[SEED, slide_index, 0xF17, 0]`, and no key repeats.
  - Cost if wrong: none while the keys stay fixed; changing one redraws every centre or labelling.
- **Row order of the pilot table:** within each condition, rows follow the permutation, so half A comes first. `slide_index` numbers the rows. Cost if wrong: none; only the numbering.
- **pytest pinned** at 9.2.0, the version that resolved: the plan pins every direct dependency but gave pytest no version. Cost: none.
- **`read_slide` checks** that the slide's first and last cohort rows belong to it, beside the coordinates' shape and finiteness. Cost: two reads of a slide code.
