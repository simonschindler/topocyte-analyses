# Betti fPCA regression: does it beat the persistence-diagram kernel of Aggarwal, Nichols and Turner?

**Status: ran on 2026-11-24 and 2026-11-25. By the rule fixed in advance, Betti-curve fPCA regression (arm F) is
below `PD_kernel` (arm K) on all three tasks:** each 95% interval of the paired difference Δ, signed so that Δ > 0
favours F, lies below 0. Mean Δ and 95% interval: interpolation −0.02401 [−0.03837, −0.01141] (Δ = R²(F) − R²(K));
curvature −0.2478 [−0.4340, −0.09194] (Δ = MSE(K) − MSE(F)); fluid −3.283 [−5.812, −1.507] (Δ = R²(F) − R²(K)).
Arm K, `PD_kernel` reimplemented from the paper's public code (read, never run), matches the values published with
that code behind the paper's Tables 1–3 to 3.6e-16 at worst (4.4e-16 with the CSVs read exactly), so the comparison
stands on the paper's own footing.

Simon's request (2026-11-24), with the paper's PDF: "Benchmark fPCA regression using inner product space of betti
curves against their reported performances". Then: "whether Betti fPCA regression beats their kernel; download the
repo. report in topocyte analyses." Then he chose approach B of `design.md`.

The paper is S. Aggarwal, J. Nichols and K. Turner, *Kernel regression using persistence diagrams*, arXiv:2611.25295.
Its estimator, `PD_kernel`, is Nadaraya–Watson regression with a Gaussian kernel on the distance W2(H0) + W2(H1)
between persistence diagrams, its bandwidth chosen by leave-one-out (LOO) cross-validation. The paper argues that
working in diagram space, without a Hilbert embedding, predicts better, and tests that only against persistence images
(`PI_kernel`). topocyte's exact Betti-curve fPCA is a Hilbert embedding too: does functional principal component
regression (FPCR) on it beat `PD_kernel` on the paper's three tasks, both seeing the same diagrams, training sets and
test sets? The paper's code and data,
[kernel-regression-persistence-diagrams](https://github.com/saksham-aggarwal-2002/kernel-regression-persistence-diagrams)
(*upstream*), are cloned into `upstream/` at the pinned commit by `fetch_upstream.py` and used as data only: no upstream
code is imported or run, and its pickles are read by an unpickler that resolves only a whitelist of numpy and pandas
globals (`safe_pickle.py`).

- **Arms** (`estimators.py`, `run_study.py arms`):
  - **K, `PD_kernel` reimplemented:** Nadaraya–Watson on W2(H0) + W2(H1), transcribed from upstream's code, which was
    read and never run. W2 is exact (`wasserstein.py`, an optimal assignment by scipy, since POT is not installed); fluid
    uses upstream's cached distances, which gate G2 checks. The bandwidth is the first LOO minimizer over 50 values in
    [0.1, 2] × the median positive training distance.
  - **F, Betti FPCR (primary):** the reduced Betti curves of H0 and H1 (topocyte's `BettiCurve`, finite bars only);
    per dimension, topocyte's exact fPCA fitted on the training curves (normalization "none") and the test curves
    projected exactly (`betti.py`); then ordinary least squares of the target on the first K0 H0 and K1 H1 scores,
    with an intercept. (K0, K1) is chosen on the training set by closed-form LOO, with K_d ≤ 20, K0 + K1 ≤ n − 2 and
    every leverage below 1 − 1e-10. The inner product of topocyte's Betti curves integrates over [0, ∞), so every
    fluid bar is shifted by 26, the ceiling of minus the lowest birth (gate G4). No curve is evaluated on a grid.
  - **B, Betti NW (exploratory):** K's estimator on L2(H0) + L2(H1), the exact L² distances between the same Betti
    curves. Against K it isolates the representation, against F the estimator; it stays outside the verdict.
- **Tasks** (`inputs.py`), the paper's Tables 1–3:
  - **interpolation:** clouds of 20 points (cos θ + U, (1 − t) cos θ + t sin θ + U), the target t; 10 runs for each
    training size n = 10, 20, …, 100, each with its own training set and 50 test clouds; R²;
  - **curvature:** 50 training and 50 test clouds of 50 points sampled from the unit disk on a surface of constant
    curvature K, the target K; one dataset; mean squared error (MSE);
  - **fluid:** 400 simulations of a 2D unsteady flow ensemble, the target ln Re, each sample's diagrams the
    vorticity's sublevel persistence (upstream's), at five times; split s tests on 100 simulations and trains on
    n = 25, 50, 100, 200 or 300 of the others. The paper runs splits 0–4; this study runs 0–99, all of which feed the
    readout. R².
- **Gates, before any readout** (`inputs.py`, `matrices.py`, `run_study.py arms`): G1, arm K matches upstream's
  published results; G2, the scipy W2 equals upstream's cached fluid distances; G3, the test scores projected from the
  Gram matrices equal `FPCA.transform`'s; G4, the fluid Gram matrices do not change with the shift; G5, the inputs
  (the interpolation datasets regenerate from seed 0, the Rips diagrams equal `ripser`'s, every fluid diagram is
  there). Arms F and B ran only after K had passed G1.
- **Decision rule, fixed in advance** (`design.md`, committed in 7075684 before any result): per task, Δ(F, K) with a
  95% percentile bootstrap interval of 10,000 resamples from `default_rng(0)`; interpolation resamples the 10 runs
  within each size, curvature the 50 test clouds, fluid the 100 splits. F **beats** `PD_kernel` if the interval lies
  above 0, is **below** it if the interval lies below 0, and is **level** otherwise, which means that the readout
  cannot tell the two apart, not that they are equal. No pooled verdict.

`design.md` holds the question, Simon's rulings and the rule; `plan.md` the implementation plan; `progress.md` the
ledger: the 14 rulings made during the build, each with its reason, and every stage's results in the order they ran;
`report.html` the verdicts, the paper's three tables with the reported values beside K, F and B, R² against training
size as the paper's Figures 6 and 9, the exploratory and descriptive readouts, F's chosen (K0, K1), the gates, the
caveats and the runs.

## Results

Mean Δ and 95% interval per task and pair, Δ(X, Y) signed so that Δ > 0 favours X: R²(X) − R²(Y) for interpolation
and fluid, MSE(Y) − MSE(X) for curvature. Only Δ(F, K) has a verdict.

| task | Δ(F, K): primary | Δ(B, K): representation | Δ(F, B): estimator |
|---|---|---|---|
| interpolation | −0.02401 [−0.03837, −0.01141], **below** | −0.02512 [−0.03123, −0.01954] | +0.001110 [−0.01391, +0.01470] |
| curvature | −0.2478 [−0.4340, −0.09194], **below** | −0.02964 [−0.08372, +0.02166] | −0.2182 [−0.4146, −0.04948] |
| fluid | −3.283 [−5.812, −1.507], **below** | −0.1388 [−0.1441, −0.1334] | −3.144 [−5.675, −1.367] |

Reading:

- **Below `PD_kernel` on all three tasks, by the rule.** F's R² trails K's in interpolation and fluid, and its
  curvature MSE, 0.4736, is 2.1 times K's, 0.2258.
- **The comparison stands on the paper's own footing.** Arm K, a reimplementation that never ran upstream's code,
  equals upstream's published results, the 146 values behind the paper's Tables 1–3, to the last bits (the status line
  gives the worst differences), where gate G1 allowed 1e-9: the reimplementation reproduces the paper's `PD_kernel`
  rather than approximating it, so the verdict does not rest on a weaker version of it.
- **Exploratory, no verdict: the representation or the estimator?** Δ(B, K) compares the representations under one
  estimator, Δ(F, B) the estimators on one representation, and their means add up to Δ(F, K).
  - In interpolation the gap is the representation's: Δ(B, K) lies below 0, and F cannot be told apart from B. That
    is a cancellation: by size, Δ(F, B)'s interval lies above 0 at n = 40, 60, 70, 80 and 100 and below 0 at n = 30.
  - In curvature it is the estimator's: B cannot be told apart from K, and Δ(F, B) is most of F's gap. Curvature rests
    on one training set.
  - In fluid both lie below 0. Δ(F, B) is most of the mean, through the smallest training size; at the larger sizes
    the two parts are of the same order (`report.html`, section 3). Here the representation's part, Δ(B, K), also
    holds the essential births, which W2 compares and the Betti curves drop.
- **For information, not a readout: arm F fails badly in a few fluid cells.** All 51 cells where F's R² lies below −10
  have n = 25; 32 of them chose K0 + K1 = 23 = n − 2, the most the eligibility rule allows, and 44 chose 21 or more.
  The descriptive fluid readout by training size shows them: n = 25 −15.34 [−28.01, −6.457], and from −0.3164 to
  −0.2434 at the other sizes, every interval below 0. They make the size of fluid's mean Δ, not its sign: without
  them the mean Δ(F, K) is −0.3336, the median over all 2,500 cells is −0.2545, and F's R² lies above K's in 36 of
  the 2,500 cells (1.4%).
- **For information, not a readout: arm F's interpolation gap.** The median Δ(F, K) over the 100 datasets is
  −0.003795, and F's R² lies above K's in 36 of them. n = 30's mean Δ(F, K), −0.1211, comes 90% from datasets 21, 22
  and 25, whose K0 + K1 is 26, 27 and 28 against the cap of 28: near-saturated fits, as in fluid at n = 25.
- **A lesson for future work, not a change to this study.** With K0 + K1 ≤ n − 2, a saturated design leaves one
  residual degree of freedom; its closed-form LOO error is then essentially a single random number, and choosing among
  many saturated candidates picks one that extrapolates badly. This rule, binding here, produces F's catastrophic
  cells, at fluid n = 25 (above) and in interpolation: the 4 interpolation datasets with the lowest Δ(F, K) (21 at
  n = 30, 3 at n = 10, 22 at n = 30 and 25 at n = 30) all chose K0 + K1 within 2 of n − 2. A later FPCR should cap
  K0 + K1 at a fraction of n.
- **Beside the paper's other methods,** descriptively: F's mean R² lies above the reported `PI_kernel`'s at every
  interpolation size. In curvature, K's and B's MSE lie below all ten reported Turkeš et al. methods and F's below 6
  of them; 9 of those methods used geodesic distance matrices, not these inputs, and `point_net` the stored
  coordinates as 3D clouds, the 2D ones padded with z = 0.
- **Two gates needed a reference that `design.md` did not name** (rulings 7 and 8, confirmed by Simon on 2026-11-25):
  G3 compares the first min(20, rank) score columns per dimension, the only ones F can use, relative to the largest
  score over them; G4 is relative to each matrix's largest entry. Behind G3 lies **a follow-up for topocyte, not for
  this study:** topocyte's `FPCA.transform` is inaccurate in near-null components, up to 8.5e-6 of the largest score,
  where it also misses its own fitted scores; F's scores come from the exact Gram projection and use none of them.
- **Caveats** (`report.html`, section 9):
  - **the curvature inputs:** every arm reads the Euclidean Rips filtration of upstream's stored coordinates, 3D on
    the sphere for K > 0 and 2D in the Poincaré disk for K < 0, where 9 of the 10 Turkeš et al. baselines of Table 2
    used geodesic distance matrices and `point_net` the same coordinates as 3D clouds, the 2D ones padded with z = 0;
    every arm predicts the sign of K right for all 50 test clouds;
  - **the essential bars that F and B drop:** none that matters in the Rips tasks (only H0's, born at 0 in every
    cloud), but in fluid every diagram has one in H0 and two in H1, whose births W2 compares and the Betti curves do
    not hold;
  - **the single curvature training set:** its interval covers test sampling only;
  - **the 100 fluid splits share the same 400 simulations:** the fluid interval bounds the mean Δ over random splits
    of these simulations, not over new simulations;
  - **the paper's swap of line and circle:** in upstream's code, and so here, t = 0 is a line and t = 1 a circle; the
    paper's text swaps them.

## Run

Needs:
- upstream at `fe7d652eff478ffe6081c7e6765458651ad3cd74` (2026-11-15), cloned into `upstream/` by `fetch_upstream.py`
  and ignored by git; every stage refuses a clone at another commit or with a changed tree (`common.check_upstream`);
- topocyte at `18e3aa5`, from a clean checkout named first on `PYTHONPATH`: every script calls
  `common.check_topocyte()` first and refuses anything else;
- no study venv: the scripts run in topocyte's main-checkout venv (Python 3.14.0). `requirements.txt` records its
  versions; nothing is installed.

The run's commands, the stages in the order they ran (the tests ran throughout, as the code grew). They are a record,
not a recipe to repeat in this folder. A rerun needs three things:

- **A separate clone or worktree, not this folder.** `results/*.npz` and `logs/` are untracked: they are the kept copy
  that `verify.json` vouches for and the report reads. A rerun in place overwrites them, and `logs/` cannot be
  regenerated with the recorded wall times and peak RSS.
- **Each stage's json committed with the scripts that wrote it before `verify_results.py` runs.** Its provenance check
  requires every stage json to be unchanged since the commit that last changed it, and every md5 the json records to
  be that of the script in that commit.
- **This README's run table updated before `build_report.py` runs.** The report asserts the table against `logs/` and
  the jsons, and the README's other numbers against the results.

Each stage refuses an npz that its predecessor's json does not vouch for, and `build_report.py` refuses a `verify.json`
that is older than a stage's json or did not check the files it reads.

```bash
git -C /Users/simon/rep/lbi/reindero/topocyte worktree add --detach ../topocyte-18e3aa5 18e3aa5   # once, for the class-curve study
export PYTHONPATH=/Users/simon/rep/lbi/reindero/topocyte-18e3aa5
PY=/Users/simon/rep/lbi/reindero/topocyte/.venv/bin/python
mkdir -p logs
$PY fetch_upstream.py --source <a clone of upstream's GitHub repository>   # without --source, from GitHub
$PY -m pytest -q -W error test_study.py         # seconds; the tests that read upstream/ skip without it
$PY inputs.py > logs/inputs.log 2>&1                                              # seconds; gate G5
caffeinate -is /usr/bin/time -l $PY matrices.py > logs/matrices.log 2>&1          # about 9 min; gates G2, G4
caffeinate -is /usr/bin/time -l $PY run_study.py arms > logs/arms.log 2>&1        # about 3 min; K, G1, G3, F, B
/usr/bin/time -l $PY run_study.py readouts > logs/readouts.log 2>&1              # seconds; the verdicts
caffeinate -is /usr/bin/time -l $PY verify_results.py > logs/verify.log 2>&1      # about 20 s; 14 checks
$PY build_report.py                                                               # seconds; report.html
```

`matrices.py` ran twice: the first run's G4 failed entry by entry, before ruling 7 named its reference. The readouts
stage ran twice, the second time with upstream's reported values read exactly; `verify_results.py` ran three times,
the second after its checks were made to fail closed on values that are not finite, the third after a docstring in
`readouts.py` changed. `progress.md` has every run. The runs whose outputs the report reads:

| stage | finished | wall time | peak RSS | md5 its json names |
|---|---|---:|---:|---|
| `fetch_upstream.py` | 2026-11-24 13:48 | not logged | not logged | `fetch_upstream.py` 15ab6d0c4dce33418778cb3edf131f88 (as committed in 64ae02e) |
| `inputs.py` | 2026-11-24 14:15 | 3.8 s inside the stage | not logged | `inputs.py` 8ea3b6580b6b4f6d856ed332c6f0005d (as committed in 4fc8ec0) |
| `matrices.py` | 2026-11-24 17:31 | 524.17 s | 0.81 GB | `matrices.py` f6a19fbac567087c36b3df193245a3eb (as committed in e92b16f) |
| `run_study.py arms` | 2026-11-25 11:41 | 177.67 s | 1.58 GB | `run_study.py` d3ae19656cf7fc6e932b07aa18b726ca (as committed in 5473801) |
| `run_study.py readouts` | 2026-11-25 12:28 | 1.45 s | 0.30 GB | `readouts.py` 702be84742d64c382929164c44d47e26 (as committed in 36b0648) |
| `verify_results.py` | 2026-11-25 14:05 | 19.97 s | 2.40 GB | `verify_results.py` 9340d18dbfec57050c094417a860a3e1 |

On an M4 with 10 cores and 16 GB. Each json records the md5 of the study's scripts its stage ran, and
`verify_results.py` checks each one against the script in the commit that last changed that json; where a script has
changed since its stage ran, the table names that commit, and `progress.md` and the git log say what changed.

## Outputs

- `results/upstream.json`: the pinned clone's SHA, its source and date;
- `results/inputs.json`: gate G5, and the fluid diagrams' bars, lowest birth (the shift), essential bars and empty
  curves;
- `results/matrices.npz` (738 arrays, 96.0 MB): every pairwise matrix of every dataset and fluid time, W2, Betti L²
  and Betti Gram, H0 and H1, with the targets; `results/matrices.json`: gates G2 and G4, the shapes, the seconds and
  the npz's sha256;
- `results/arms.npz` (31 arrays, 6.4 MB): each arm's test predictions, R² or MSE, bandwidth or (K0, K1) in every cell,
  with the test targets and each fluid split's test set (`run_study.py`'s docstring lists the keys);
  `results/arms.json`: gates G1 and G3, the seconds per arm and task, the npz's sha256;
- `results/readouts.json`: the verdicts, the exploratory and descriptive readouts with their intervals, F's chosen
  (K0, K1), and the paper's three tables with the reported values beside K, F and B;
- `results/verify.json`: every check of `verify_results.py` with its worst value and tolerance, the md5 of the scripts
  and the sha256 of every data file it read;
- `report.html`.

Tracked: `results/*.json`. Not in git: `results/*.npz`, `logs/`, `report.html` and `upstream/`; the scripts
regenerate them. Each json records the md5 of the scripts that wrote it and the versions; `verify.json` holds the
sha256 of both npz files and of the upstream files the verifier read.
