# Ledger: e-means clustering of every cohort cell's 150 µm neighbourhood

## Request (Simon, 2026-11-23)

"use emeans for clustering of composition, density profile, composition + density profile derived features of
the 150 um neigborhoods of every cell in the liver cohort. report gap statistics for k in [1,10]. save results."
Then: "no i mean e means clustering. its a very recently released new algorithm that is way faster:
https://arxiv.org/pdf/2611.09501v1"

## Simon's answers (2026-11-23)

- **Density profile:** "what we already used. fraction of cells preserved in a cells 150 um neighborhood after
  removing every cell that has below threshold 10-NN density computed at 10 thresholds." That is the pathology
  study's rho (density.npz: `rho = 10 · 10^6 / (π r10²)` cells per mm², r10 the distance to the 10th nearest
  cell in µm) and its ten thresholds, the deciles of rho at levels 0.0, 0.1, …, 0.9; a cell is kept at
  threshold j when `rho >= d_j` (`run_pathology_slices.py`, slice j).
- **Combined set:** equal weight per block (every column z-scored, each block divided by √(its columns)).
- **e-means:** the authors' code; download of faiss-cpu 1.16.0 from PyPI approved
  (faiss_cpu-1.16.0-cp310-abi3-macosx_15_0_arm64.whl, 5.3 MB).

## Rulings (Claude)

- Ruling: the ball is the chromatic pilot's, closed, of 150 µm (303.52 px at the assumed 0.4942 µm/px), the cell
  itself included, for both the class counts and the kept counts — why: "the 150 um neighborhoods" of the pilot
  — cost if wrong: one rerun of features.py (about 10 min).
- Ruling: threshold 0 is rho's minimum, so every cell is kept and its fraction is 1 for every cell; the column
  is stored implicitly (as the ball's cells) and left out of the clustering as constant, which leaves 9 density
  columns — why: a constant column carries nothing and has no z-score — cost if wrong: none.
- Ruling: the composition set is z-scored per column too (as the pilot's level 1) — why: the same scaling for
  all three sets; the fractions have different spreads — cost if wrong: a rerun of one set (about 30 min).
- Ruling: gap reference = the chromatic pilot's matched Gaussian (Simon's ruling of 2026-11-20 for the pilot's
  gap statistic), B = 20 — why: the same rule as the pilot; known to read any non-Gaussian shape, bounded
  compositions included, as structure (pilot, 2026-11-21) — cost if wrong: the reported k̂ may be high; the
  gap curves themselves are what to read.
- Ruling: W_k is the mean (not the sum) of squared distances, so the data's W_k uses every cell and each
  reference has 20M points — why: a 126M-point reference of 13 float32 columns with the data in memory exceeds
  the 16 GB machine; with means the sample size cancels in Gap(k) — cost if wrong: s_k reflects 20M-point
  references, slightly larger than 126M-point ones would give (conservative).
- Ruling: every cell's label is its nearest final centroid (e-means' `assign`), not e-means' last-pass label;
  W_k is computed from the same assignment — why: Tibshirani's W_k is the final partition's dispersion; the
  README of e-means says last-pass labels can differ — cost: one extra pass per fit.
- Ruling: e-means runs with its defaults (5 passes, its scaling laws for the batch and step, cost-based revival,
  "quality" batch at d <= 16), seeds from `SeedSequence([20261123, set, run, k])`; a check against scikit-learn's
  k-means (k-means++, 20 starts) on a 500,000-cell subsample at k = 4 and 10 is recorded per set — why: the paper
  tunes its laws on high-dimensional data and warns they are not optimal at low dimension — cost if wrong: a
  worse W_k than Lloyd's, which the check would show.
- Ruling: labels are stored two per byte (k <= 10 < 16) in compressed npz — why: 17 GB were free on the disk
  (97% full); 9 uint8 arrays of 126M per set would take 3.4 GB before compression — cost: an unpack step
  (`cluster.unpack`).

- Ruling: cluster.py's own k-means check is not what it says: it refits e-means to the 500,000 cells alone, a
  separate fit (e-means sizes batch and steps by the sample: 1,225 steps of 2,048 cells there, 2,410 of 262,144
  on every cell), so its ratios (up to 1.7% on composition) say nothing about the run's fits. `check_kmeans.py`
  measures the run's centroids on the same 500,000 cells against k-means++ (20 starts) fitted to them, and
  recomputes the gap with k-means++ on those cells and on 500,000 reference points — why: the stops at k = 3
  (composition + density) and k = 7 (density) rest on Gap dips of 0.009 and 0.008, the size of the in-run
  check's ratios — cost: 4 min; cluster.py's results are not changed, the in-run check stays in gap.json,
  labelled for what it is. (I first guessed 10 steps of 262,144 for the subsample fit; its recipe says 1,225 of
  2,048.)

## Runs

- 2026-11-23: faiss-cpu 1.16.0 installed into the study venv (Python 3.13.9; numpy, scipy, pandas, h5py,
  matplotlib, scikit-learn and pytest from the uv cache, pinned to the chromatic pilot's freeze).
- 2026-11-23: speed probe (synthetic Gaussian, k = 10, 13 columns): 20M points trained in 13.0 s (batch 65,536,
  α0 1.0), assigned in 0.3 s.
- 2026-11-23: features.py on the largest slide (GTEX-11TT1-1726, 725,038 cells): 3.1 s, max count 583, mean
  ball 212 cells. A first full run was stopped at slide 162 because `ruff format` had rewritten features.py
  while it ran; it was rerun from the formatted file (md5 ad8cfb3f…) so features.json names the code that ran.
- 2026-11-23: features.py on every slide: 458 s, peak RSS 4.5 GB; 126,219,124 balls, mean 151.2 cells, at most
  966 (`results/counts.npy`, 3.3 GB).
- 2026-11-23: cluster.py, all three sets: 2.49 h (composition 44 min, density 48 min, combined 57 min), peak RSS
  9.8 GB, peak memory footprint 10.7 GB. e-means fits on every cell took 66–122 s each (2,410 steps of 262,144),
  references 90–120 s each. Gap rises at every k on composition (k̂ 10 by both rules); density stops at k = 7
  (Gap 0.8104, then 0.8027 at k = 8); composition + density stops at k = 3 (Gap 0.1896, then 0.1805 at k = 4).
- 2026-11-23: verify_results.py: 40/40 checks pass in 844 s (peak RSS 7.2 GB): the counts' invariants, 1,200
  balls recounted by brute force, every label its nearest saved centroid (at most 16 float32 ties per k, none
  beyond), W_k from the labels within 3.5e-7, the cluster profiles, Gap and both rules, reference 1 redrawn and
  refitted within √(B − 1) SDs of the saved mean, and the summaries against the per-set files.
- 2026-11-23: check_kmeans.py: 3.6 min (k-means++ fits 22–41 s per set and sample). It rebuilt cluster.py's
  subsample exactly (k-means++ W_4 and W_10 as logged, within 1.9e-7; e-means' subsample fit as logged; total
  variance as the run's W_1 within 1.8e-12). The run's fits equal k-means++'s W_k within 0.03% at 23 of 27
  (set, k), and are worse at composition k = 4 (+0.79%), density k = 8 (+4.37%) and composition + density k = 6
  (+3.46%) and 9 (+1.24%). With k-means++ on both sides, Tibshirani's k̂ is 10 (composition), 10 (density, the
  e-means stop at 7 came from its k = 8 fit) and 3 (composition + density: the fits at 3 and 4 equal k-means++'s
  and both gaps fall from 3 to 4, by 0.0091 and 0.0094). Rerun once after a docstring fix (same numbers).
- 2026-11-23: build_report.py: report.html (0.1 MB), looked at in the browser pane.
