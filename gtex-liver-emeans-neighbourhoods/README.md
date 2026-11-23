# e-means clusters of every liver cell's 150 µm neighbourhood

**Status: ran on 2026-11-23; descriptive, no verdict.** Gap rises to k = 10 for composition; composition + density
stops at k = 3 (both rules), and a k-means++ check agrees; density stops at k = 7 only because e-means' fit at k =
8 is 4.4% worse than k-means++'s, and with k-means++ its gap rises to k = 10.

Simon's request (2026-11-23): "use emeans for clustering of composition, density profile, composition + density
profile derived features of the 150 um neigborhoods of every cell in the liver cohort. report gap statistics for k
in [1,10]. save results." e-means is the mini-batch k-means of Bessa et al. 2026
([arXiv:2611.09501](https://arxiv.org/abs/2611.09501)), run here with the authors' code,
`faiss.contrib.e_means.NumpyEmeans` from faiss-cpu 1.16.0.

Every cell of the 594 GTEx liver slides with cells (126,219,124 cells) gets the closed 150 µm ball around it (the
chromatic pilot's ball, the cell included), and three feature sets describe the ball:

- **composition:** the ball's four class fractions (epithelial, stromal, immune, other; the cell-type study's
  classes of the HistoPLUS types);
- **density:** the fraction of the ball's cells kept at each of the pathology study's density thresholds, the ten
  deciles of the 10-nearest-neighbour density over all cells, a cell kept at threshold j when its density is at
  least d_j (Simon: "fraction of cells preserved in a cells 150 um neighborhood after removing every cell that has
  below threshold 10-NN density computed at 10 thresholds"). Threshold 0 is the minimum density, which keeps every
  cell, so its column is 1 everywhere and is left out: 9 columns;
- **composition + density:** both blocks, each column z-scored and each block divided by √(its columns), so the
  two blocks weigh the same (Simon's choice).

Every column is z-scored over all cells. e-means clusters each set for k = 2..10, every cell gets its nearest final
centroid, and the gap statistic (Tibshirani, Walther and Hastie 2001) compares log W_k, the mean squared distance to
the nearest centroid, with its mean over 20 Gaussian references that have the set's mean and covariance, the
chromatic pilot's reference. `check_kmeans.py` measures the run's fits against k-means++ on 500,000 of the cells and recomputes the gap
with k-means++. `progress.md` has every ruling and what it costs if wrong; `report.html` has the gap
curves, the tables and every cluster's profile.

## Results

Gap(k) of the run (e-means on every cell, 20 references of 20M points; s_k in brackets) and of the k-means++
check (`check_kmeans.py`: scikit-learn's k-means++, 20 starts, on cluster.py's 500,000 cells and on 500,000
reference points; ± 1 SE):

| k | composition: Gap (s_k) | k-means++ Gap | density: Gap (s_k) | k-means++ Gap | composition + density: Gap (s_k) | k-means++ Gap |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0001 (0.0003) | 0.0008 ± 0.0032 | 0.0001 (0.0002) | −0.0001 ± 0.0023 | 0.0000 (0.0001) | 0.0001 ± 0.0022 |
| 2 | 0.1235 (0.0002) | 0.1214 ± 0.0033 | 0.0759 (0.0002) | 0.0747 ± 0.0024 | 0.0178 (0.0002) | 0.0181 ± 0.0021 |
| 3 | 0.1981 (0.0002) | 0.1977 ± 0.0031 | 0.2985 (0.0002) | 0.2976 ± 0.0023 | 0.1896 (0.0002) | 0.1894 ± 0.0025 |
| 4 | 0.2118 (0.0002) | 0.2196 ± 0.0026 | 0.5293 (0.0120) | 0.5221 ± 0.0022 | 0.1805 (0.0002) | 0.1800 ± 0.0027 |
| 5 | 0.3437 (0.0029) | 0.3423 ± 0.0028 | 0.6694 (0.0002) | 0.6688 ± 0.0021 | 0.2533 (0.0060) | 0.2498 ± 0.0025 |
| 6 | 0.3671 (0.0013) | 0.3676 ± 0.0028 | 0.7563 (0.0002) | 0.7547 ± 0.0021 | 0.2715 (0.0043) | 0.3004 ± 0.0023 |
| 7 | 0.3905 (0.0002) | 0.3910 ± 0.0029 | 0.8104 (0.0002) | 0.8116 ± 0.0021 | 0.3371 (0.0030) | 0.3358 ± 0.0024 |
| 8 | 0.3991 (0.0002) | 0.3984 ± 0.0030 | 0.8027 (0.0002) | 0.8464 ± 0.0021 | 0.3614 (0.0002) | 0.3617 ± 0.0025 |
| 9 | 0.4313 (0.0007) | 0.4322 ± 0.0028 | 0.8713 (0.0002) | 0.8709 ± 0.0021 | 0.3800 (0.0004) | 0.3912 ± 0.0023 |
| 10 | 0.4430 (0.0008) | 0.4434 ± 0.0027 | 0.9108 (0.0013) | 0.9091 ± 0.0020 | 0.4122 (0.0007) | 0.4126 ± 0.0021 |
| k̂ | **10** (pilot rule 10) | **10** | **7** (pilot rule 7) | **10** | **3** (pilot rule 3) | **3** |

k̂ is Tibshirani's rule (the smallest k with Gap(k) ≥ Gap(k + 1) − s_(k+1), else 10); the pilot's rule keeps it
only if Gap(k̂) − Gap(1) ≥ 0.03, which holds for every k̂ here. The check's rules use the run's s_k.

W_k of the run's centroids against k-means++ fitted to the same 500,000 cells (± paired SE):

| k | composition | density | composition + density |
|---|---:|---:|---:|
| 2 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| 3 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| 4 | +0.79% ± 0.17% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| 5 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| 6 | 0.00% ± 0.00% | −0.03% ± 0.02% | +3.46% ± 0.11% |
| 7 | 0.00% ± 0.01% | 0.00% ± 0.01% | 0.00% ± 0.00% |
| 8 | 0.00% ± 0.00% | +4.37% ± 0.12% | 0.00% ± 0.00% |
| 9 | +0.01% ± 0.00% | −0.01% ± 0.00% | +1.24% ± 0.09% |
| 10 | +0.01% ± 0.01% | 0.00% ± 0.00% | +0.01% ± 0.00% |

Reading:

- **composition + density stops at k = 3, and the check agrees.** The run's fits at k = 3 and 4 equal
  k-means++'s, and both gaps fall from k = 3 to 4 (by 0.0091 with e-means, 0.0094 with k-means++). The three
  clusters: 37% of cells in sparse parenchyma (105 cells per ball, 78% epithelial, 19% of the ball's cells at or
  above the cohort's median density), 48% in denser parenchyma (164 cells, 79% epithelial, 66% at or above the
  median) and 16% in dense mixed neighbourhoods (223 cells; 30% epithelial, 30% stromal, 17% immune, 22% other;
  81% at or above the median).
- **density stops at k = 7 only because e-means' fit at k = 8 is 4.4% above k-means++'s W_k.** With k-means++ its
  gap rises at every k up to 10.
- **composition rises at every k** with both fits.
- Against the matched Gaussian a rising gap says the cloud is not Gaussian (bounded, skewed fractions are not),
  not that it holds 10 groups; the chromatic pilot's compositions rose the same way.
- s_k is only the references' spread. In 4 of the 27 data fits e-means settled 0.8–4.4% above k-means++
  (composition k = 4, density k = 8, composition + density k = 6 and 9): those labels are a worse partition than
  k-means++'s, and the run's Gap(k) there is too low by about as much. s_k above 0.001 (composition k = 5–6,
  density k = 4 and 10, composition + density k = 5–7) is e-means settling differently across references.
- `report.html` has every cluster's profile at every k, the gap curves of both fits and the checks.

## Run

Needs:
- the cohort `~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad`, opened read only;
- `density.npz` of the sibling study [gtex-liver-pathology-slices](../gtex-liver-pathology-slices/README.md) in
  `../gtex-liver-pathology-slices/results` (each cell's 10-NN density, its deciles and the slides' row bounds);
- the study venv `.venv/` on Python 3.13, from `requirements.txt` (`results/venv_freeze.txt` lists every package
  the run used). The study does not import topocyte.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

| script | ran | wall time | peak RSS | md5 (named in its json) |
|---|---|---|---|---|
| `features.py` | 2026-11-23 | 7.6 min | 4.5 GB | ad8cfb3f563554a181cf9af12d984ac4 |
| `cluster.py` | 2026-11-23 | 2.5 h | 9.8 GB | 5d8c55b4ebbb2be32974294667396653 |
| `verify_results.py` | 2026-11-23 | 14 min | 7.2 GB | aa9d297bd78553121e4225d62055d3ec |
| `check_kmeans.py` | 2026-11-23 | 3.6 min | 6.3 GB | 54289aed319bfbbdbc8a8c7222bfd0e2 |

On an M4 with 10 cores and 16 GB; `build_report.py` checks each md5 against the script in this folder.

The long runs went on AC power inside `caffeinate -is`, with `/usr/bin/time -l` and their logs in `logs/`
(gitignored, so a fresh checkout creates it); `build_report.py` reads the runs' time and memory from the logs.
`cluster.py` keeps a finished set (its `gap_<set>.json` and `labels_<set>.npz`) and runs the others, so an
interrupted run resumes at the set it was in.

```bash
PY=.venv/bin/python
mkdir -p logs
$PY -m pytest -q -W error                                           # the study's tests, a few seconds
caffeinate -is /usr/bin/time -l $PY features.py > logs/features.log 2>&1   # about 8 min
caffeinate -is /usr/bin/time -l $PY cluster.py > logs/cluster.log 2>&1     # about 2.5 h
caffeinate -is /usr/bin/time -l $PY verify_results.py > logs/verify.log 2>&1     # about 14 min
caffeinate -is /usr/bin/time -l $PY check_kmeans.py > logs/check_kmeans.log 2>&1 # about 4 min
$PY build_report.py
```

## Outputs

- `results/counts.npy` (cells × 13, uint16, 3.3 GB, the cohort's row order): every cell's ball counted, the cells
  of each class, then the cells kept at thresholds 1..9 (`common.COUNT_COLUMNS`); the ball's cells are the sum of
  the first four;
- `results/labels_<set>.npz` (`<set>` is `composition`, `density` or `composition_density`): every cell's label
  for k = 2..10 as arrays `k2` … `k10`, two labels per byte, and `n_cells`; `cluster.unpack(z["k5"], int(z["n_cells"]))`
  gives the 126,219,124 labels of k = 5 in the cohort's row order. The fits at composition k = 4, density
  k = 8 and composition + density k = 6 and 9 are 0.8–4.4% above k-means++ in W_k (`kmeans_check.json`),
  so those labels are a worse partition than k-means++'s. The liver cell viewer (`../gtex-liver-chromatic-domains/viewer/serve.py`, "Cohort e-means clusters
  (saved)") colours any slide's cells by these labels;
- `results/gap.json`: the settings and, per set, log W_k, the references' mean and SD of log W*_k, Gap(k), s_k,
  both rules' k̂, cluster.py's own k-means check (a separate e-means fit on 500,000 cells, superseded by
  `kmeans_check.json`) and each fit's e-means recipe and time;
- `results/clusters.json`: per set and k, the centroids (z-scored, block-weighted units) and each cluster's cells,
  share, mean cells per ball and mean class and kept fractions;
- `results/gap_<set>.json`: one set's part of both, written as the set finished;
- `results/features.json`: the counting run's settings, thresholds, maximum and mean ball, time and versions;
- `results/kmeans_check.json`: per set, on cluster.py's 500,000 cells, W_k of the run's centroids and of
  k-means++ (20 starts) fitted to those cells, and the gap statistic from k-means++ on those cells and on
  500,000 reference points, with both rules' k̂;
- `results/verify.json`: every check of `verify_results.py` with its measured value and tolerance, the md5 of the
  script and the sha256 of every gitignored file it read.

Tracked: `results/*.json`, `results/venv_freeze.txt`. Not in git: `results/counts.npy`, `results/labels_*.npz`
(73–93 MB each), `logs/`, `report.html` and the `.venv/`. The scripts regenerate them; the copy the report was
built from is in this folder on Simon's machine, and `verify.json` holds its sha256 sums.
