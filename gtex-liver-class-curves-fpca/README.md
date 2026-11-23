# Class-density curves: does their fPCA beat density and composition?

**Status: ran on 2026-11-23; by the rule fixed in advance, level with D+C.** On the 476 development slides, the fPCA
of the class curves at the chosen k reaches macro AUC 0.750 against 0.756 for D+C trained on the same cleaned slides:
a difference of −0.0064, 95% interval [−0.0241, +0.0116], which contains 0. The 118-slide holdout, scored once, leans
the same way: 0.769 against 0.779 (−0.0101 [−0.0421, +0.0169]).

Simon's request (2026-11-23): "lets design a experiment that performs functional pca on these curves. first computing
the mean curve per cohort, filtering cohort outliers, fpca on inner product spaces as with betti curves normalized and
unnormalized, classification." Then: "cohort means pathology group. i want to know if it beats D+C."

For one slide and one of the four cell classes (epithelial, stromal, immune, other), the class curve n(t) counts the
class's cells whose 10-nearest-neighbour density is at or above t cells per mm². The liver cell viewer
(`../gtex-liver-chromatic-domains/viewer/serve.py`) draws them. The cell-type study
([gtex-liver-celltype-slices](../gtex-liver-celltype-slices/README.md)) found D+C the representation to beat: 51
features, the slide's density summaries and its cells of each class at each of the ten density deciles, where the
class curves give exactly those counts. This study asks whether a functional PCA of the whole curves separates the
eight pathology groups better.

- **Curves and Gram matrices** (`curves.py`): a class curve is the reduced Betti curve of the bars (0, ρ_i),
  topocyte's `BettiCurve` of dimension 0, so the inner product of two curves is exactly Σ_i Σ_j min(ρ_i, ρ_j). Each
  class's 594 × 594 Gram matrix is computed once, exactly, by `betti_inner_product_matrix`; no curve is evaluated on a
  grid.
- **In every outer fold** (`folds.py`, `run_study.py`), on the cell-type improvement phase's development folds (20
  repeats of stratified 5-fold):
  1. the outlier filter, on the training slides only: in each of the eight blocks (four classes × raw and cosine), a
     slide's squared distance to its group's mean curve, divided by the block's median and summed over the blocks; a
     slide above its group's median + 3 × 1.4826 × MAD is dropped;
  2. each block's fPCA on the kept slides (topocyte's `fpca_from_gram`, nine components), and an exact projection of
     the other slides;
  3. a multiple factor analysis of the eight blocks, up to 51 components;
  4. four arms, each the cell-type study's logistic regression: the curves at k chosen by inner 5-fold
     cross-validation from 4 to 51, the curves at k = 4, D+C on the same kept slides, and D+C on all the training
     slides (the gate against the cell-type study).
- **Decision rule, fixed in advance** (`design.md`, committed in e07c262 before any result): beats D+C if the
  development 95% interval of curves at the chosen k minus D+C lies above 0 and the holdout difference is positive;
  not confirmed if the interval lies above 0 and the holdout difference is not positive; level with D+C if the
  interval contains 0; below D+C if it lies below 0.

`design.md` holds the question, Simon's rulings and the rule; `plan.md` the implementation plan; `progress.md` the
ledger of what ran; `report.html` each group's mean curves, the dropped slides, the fractions of variance, the chosen
k and the readouts.

## Results

Macro AUC over the seven conditions (unclassified unscored). Development: from the out-of-fold probabilities averaged
over the 20 repeats (the pre-registered readout), and each repeat's, mean ± SD.

| arm | development | per repeat | holdout (scored once) |
|---|---:|---:|---:|
| curves at the chosen k | 0.750 | 0.722 ± 0.013 | 0.769 |
| curves at k = 4 | 0.740 | 0.729 ± 0.013 | 0.765 |
| D+C on the cleaned training slides | 0.756 | 0.749 ± 0.011 | 0.779 |
| D+C on all the training slides | 0.764 | 0.759 ± 0.009 | 0.774 |

Paired differences against D+C on the cleaned training slides, with 95% intervals over 4000 bootstrap draws of the
slides, stratified by group (seed 12345):

| comparison | development | holdout |
|---|---:|---:|
| curves at the chosen k − D+C (the verdict's) | −0.0064 [−0.0241, +0.0116] | −0.0101 [−0.0421, +0.0169] |
| curves at k = 4 − D+C (beside, no verdict) | −0.0164 [−0.0398, +0.0078] | −0.0144 [−0.0537, +0.0183] |

Reading:

- **Level with D+C.** The development interval contains 0: the curves' fPCA neither beats D+C nor falls below it. The
  holdout difference does not enter the rule then; it was negative too.
- **The outlier filter dropped 11.7% of the training slides** in the development folds: 4,460 of 38,080
  training-slide places, 36–56 per fold, from 4.5% of fibrosis's to 17.6% of necrosis's. Of the 476 development
  slides, 382 were never dropped and 37 in all 80 folds they trained in. **The holdout fit dropped 55 of the 476
  (11.6%):** healthy 7 of 53, congestion 15 of 124, steatosis 19 of 162, necrosis 3 of 14, inflammation/hepatitis 1
  of 10, fibrosis 1 of 40, cirrhosis 6 of 46, unclassified 3 of 27. Both arms train on the same cleaned slides, and
  the test slides are always scored.
- **Per repeat, the gap is larger:** 0.722 ± 0.013 for the curves against 0.749 ± 0.011 for D+C. The curves'
  probabilities vary more between repeats (an SD over the repeats of 0.054 on average, D+C's 0.035), and averaging
  over the repeats raises their macro AUC by 0.028, D+C's by 0.007.
- **k:** the inner cross-validation chose k = 4 or 6 in 62 of the 100 outer folds, and never 8, 40 or 51; the
  holdout fit chose k = 6.
- **The filter cost D+C** 0.008 of development macro AUC (0.756 against 0.764) and gained it 0.005 on the holdout
  (0.779 against 0.774).
- **The gates passed:** the unfiltered D+C equals the cell-type improvement phase's probabilities exactly, on the
  development slides and on the holdout (macro AUC 0.7645 and 0.7739).
- The holdout's groups of 2 (inflammation/hepatitis) and 4 (necrosis) slides make its intervals too narrow.

## Run

Needs:
- the cohort `~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad`, opened read only (h5py mode "r"),
  for the cells' classes;
- `density.npz` and `pathology.csv` of [gtex-liver-pathology-slices](../gtex-liver-pathology-slices/README.md) in its
  `results/`;
- [gtex-liver-celltype-slices](../gtex-liver-celltype-slices/README.md): its `common.py`, imported as `celltype`, and
  in its `results/` `spaces.npz`, `holdout.csv`, `composition.npz`, `improve_dev.npz` and `.json` and
  `improve_holdout.npz`;
- topocyte at `18e3aa5`, from a clean checkout named first on `PYTHONPATH`: every script calls
  `common.check_topocyte()` first and refuses anything else;
- no study venv: the scripts run in topocyte's main-checkout venv (Python 3.14.0), as the cell-type study's readouts
  did. `requirements.txt` records its versions; nothing is installed.

```bash
git -C /Users/simon/rep/lbi/reindero/topocyte worktree add --detach ../topocyte-18e3aa5 18e3aa5   # once
export PYTHONPATH=/Users/simon/rep/lbi/reindero/topocyte-18e3aa5
PY=/Users/simon/rep/lbi/reindero/topocyte/.venv/bin/python
mkdir -p logs
$PY -m pytest -q -W error test_study.py                                                  # a few seconds
caffeinate -is /usr/bin/time -l $PY curves.py spike > logs/spike.log 2>&1
caffeinate -is /usr/bin/time -l $PY curves.py gram > logs/gram.log 2>&1                  # about 10 min
caffeinate -is /usr/bin/time -l $PY run_study.py development > logs/development.log 2>&1 # under a minute
caffeinate -is /usr/bin/time -l $PY run_study.py readouts > logs/readouts.log 2>&1
caffeinate -is /usr/bin/time -l $PY run_study.py holdout > logs/holdout.log 2>&1         # once only: see below
caffeinate -is /usr/bin/time -l $PY verify_results.py > logs/verify.log 2>&1             # about 7 min
$PY build_report.py
```

**The holdout was scored once,** at 2026-11-23 20:07:22: `run_study.py holdout` wrote `results/holdout.npz` and then
`results/holdout.json`, opening the json with mode "x". The stage refuses to start when `holdout.json` exists, and
that is its only guard: it does not look at `holdout.npz`, and the `development` and `readouts` stages would rewrite
`results/development.json`, whose interval the verdict read. So none of the three stages may be run again in this
folder; a rerun belongs in a fresh copy. `verify_results.py` and `build_report.py` read the saved results only and
score nothing.

| stage | ran | wall time | peak RSS | md5 (named in its json) |
|---|---|---:|---:|---|
| `curves.py spike` | 2026-11-23 | 3.00 s | 3.02 GB | e014819eb1ac14bec1c463b90f12e46e |
| `curves.py gram` | 2026-11-23 | 617.63 s | 3.25 GB | e014819eb1ac14bec1c463b90f12e46e |
| `run_study.py development` | 2026-11-23 | 39.55 s | 0.32 GB | 319f1f1009cb27b2ec158e1506f63d95 (as committed in 727667e) |
| `run_study.py readouts` | 2026-11-23 | 6.90 s | 0.31 GB | 034e0ad569f7d6d79ea0f480f4ea00a5 (a58c812) |
| `run_study.py holdout` | 2026-11-23, once | 5.46 s | 0.33 GB | 034e0ad569f7d6d79ea0f480f4ea00a5 (a58c812) |
| `verify_results.py` | 2026-11-23 | 451.91 s | 5.19 GB | ea8bf2bdccf206e22fb6b6e46bdd5bf5 |

On an M4 with 10 cores and 16 GB. The development, readouts and holdout records also name `folds.py`
(c6b1d6546bbda28c9986d88204289133) and `common.py` (db4b4915cadc80c815832b6ce97c6b77), unchanged since.
`verify_results.py` checks every md5 against the script at its commit, and `build_report.py` checks that
`verify.json` comes from this folder's `verify_results.py` and that all its checks passed.

## Outputs

- `results/gram.npz` (11 MB): `gram_<class>`, each class's exact Gram matrix of the class curves over the 594 slides
  (float64, in the cell-type study's `spaces.npz` order), and `slide_ids`;
- `results/development.npz`: each arm's out-of-fold probabilities in every repeat (`curves`, `curves4`, `dc`,
  `dc_unfiltered`; 20 × 476 × 8, the columns in `GROUP_ORDER`), the chosen `k` (20 × 5), `dropped` (20 × 5 × 476:
  dropped from the training slides of repeat r, fold f), `folds` (20 × 476) and `slide_ids` (the 476 development
  slides);
- `results/holdout.npz`: each arm's holdout probabilities (118 × 8), `k`, `dropped` (476: the development slides the
  holdout fit dropped), `slide_ids`, `labels` and `dev_slide_ids`;
- `results/spike.json` and `results/gram.json`: the timing spike, and the Gram run with its decile gate;
- `results/development.json`: the development run's metadata (the D+C gate, the chosen k, the drops per fold) and its
  `readouts` (each arm's AUCs, per condition and per repeat, and the paired differences);
- `results/holdout.json`: the holdout's AUCs, differences, gate and verdict;
- `results/verify.json`: every check of `verify_results.py` with its measured value and tolerance, the md5 of the
  script and of the cell-type study's `common.py`, and the sha256 of every data file it read.

Tracked: `results/*.json`. Not in git: `results/*.npz`, `logs/` and `report.html`. A fresh copy regenerates them with
the scripts; the copy the report was built from is in this folder on Simon's machine, and `verify.json` holds its
sha256 sums.
