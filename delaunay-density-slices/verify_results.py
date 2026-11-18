#!/usr/bin/env python
"""Check the density-slice results against the brief's already-measured numbers.

Run after ``run_slices.py`` has produced ``results/density.npz`` and one
``results/slice_<k>.npz`` per slice.  Every assertion is one of the checks the
brief requires before the report may be written:

* 30,836,186 cells on 123 slides, 57 cirrhosis and 66 healthy;
* the pooled density deciles, the cells-per-slide range, and no slide below
  100 cells in any slice;
* the median kept fractions and per-slide median densities measured on
  2026-11-16;
* slice 0, seed 0 reproduces tutorial 04's four held-out accuracies;
* the combined models' feature blocks equal the per-slice scores;
* section 7's density arm is ten density features plus log10 slide size,
  residual arms exist per slice only (a combined one is p >= n), every arm's
  probabilities, log-loss and AUC agree, and the chance baselines are the
  training fold's majority class and prevalence (corrections of 2026-11-17);
* section 8's recomputed diagrams give back the saved H1 scores, its density
  arm is section 7's, and its survival-curve Gram matrix equals the closed
  form;
* the post-hoc refits (``scaled_nested.py``, ``scaled_section7.py``,
  ``scaled_ratio_alone.py``, ``scaled_single_block.py``,
  ``scaled_ph_only.py``): refitted unscaled, every nested, ratio-alone,
  residual, combined PC and PH-only arm and every control equals the saved
  one; the standardized arms are consistent, agree where two scripts overlap,
  and match scikit-learn's own pipelines.

Exit status 0 iff every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

N_CELLS = 30_836_186
N_SLIDES = 123
N_CIRRHOSIS = 57
N_HEALTHY = 66
MIN_CELLS_PER_SLIDE = 28_530
MAX_CELLS_PER_SLIDE = 725_038
MEASURED_DECILES = np.array(
    [0, 1393, 1693, 1952, 2212, 2498, 2840, 3290, 3971, 5300], dtype=float
)
# Median kept fraction, cirrhosis / healthy, at q = 0.5, 0.8, 0.9.
MEASURED_KEPT = {
    5: (0.59, 0.30),
    8: (0.24, 0.05),
    9: (0.12, 0.02),
}
# Median over slides of each slide's median density, cells/mm^2.
MEASURED_SLIDE_MEDIAN = {"cirrhosis": 2793.0, "healthy": 2076.0}
MIN_KEPT_AT_Q09 = 212
MIN_KEPT_ANY_SLICE = 100
TUTORIAL_ACCURACY = {
    ("h0", "none"): 0.886,
    ("h1", "none"): 0.780,
    ("h0", "cosine"): 0.886,
    ("h1", "cosine"): 0.886,
}
SEEDS = np.arange(20)
MODELS = ("h0_none", "h1_none", "h0_cosine", "h1_cosine")
EPS = 1e-15


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def _log_loss(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-slide log-loss of the cirrhosis probability, clipped like sklearn's."""
    p = np.clip(proba, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def _check_ci(adds: np.lib.npyio.NpzFile, key: str, tag: str = "density_adds") -> None:
    if key not in adds.files:
        _fail(f"{tag}: {key} missing")
    ci = adds[key]
    if ci.shape != (3,) or not np.all(np.isfinite(ci)):
        _fail(f"{tag} {key}: shape {ci.shape} or non-finite")
    if ci[1] > ci[2]:
        _fail(f"{tag} {key}: lo > hi")


def main() -> None:
    if not (RESULTS / "density.npz").exists():
        _fail(f"{RESULTS / 'density.npz'} missing; run run_slices.py density")
    density = np.load(RESULTS / "density.npz", allow_pickle=False)
    rho = density["rho"]
    deciles = density["deciles"]
    slide_codes = density["slide_codes"]
    slide_categories = density["slide_categories"]
    slide_labels = density["slide_labels"]
    slide_counts = density["slide_counts"]
    slide_medians = density["slide_medians"]

    if rho.size != N_CELLS:
        _fail(f"n cells {rho.size} != {N_CELLS}")
    if slide_categories.size != N_SLIDES:
        _fail(f"n slides {slide_categories.size} != {N_SLIDES}")
    n_cirrhosis = int(np.sum(slide_labels == "cirrhosis"))
    n_healthy = int(np.sum(slide_labels == "healthy"))
    if (n_cirrhosis, n_healthy) != (N_CIRRHOSIS, N_HEALTHY):
        _fail(f"slide classes {n_cirrhosis}/{n_healthy} != {N_CIRRHOSIS}/{N_HEALTHY}")
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        _fail("density has non-finite or non-positive values")
    lo, hi = int(slide_counts.min()), int(slide_counts.max())
    if (lo, hi) != (MIN_CELLS_PER_SLIDE, MAX_CELLS_PER_SLIDE):
        _fail(f"cells per slide {lo}..{hi} != {MIN_CELLS_PER_SLIDE}..{MAX_CELLS_PER_SLIDE}")
    if np.any(slide_counts != np.bincount(slide_codes, minlength=N_SLIDES)):
        _fail("slide_counts does not match slide_codes")
    # The brief prints the deciles rounded to whole cells; accept the rounding
    # step, including the minimum, which it prints as 0.
    if np.any(np.abs(deciles - MEASURED_DECILES) > 1.0):
        _fail(f"deciles {np.array2string(deciles, precision=6)} do not match {MEASURED_DECILES} within 1")
    for label, expected in MEASURED_SLIDE_MEDIAN.items():
        got = float(np.median(slide_medians[slide_labels == label]))
        if abs(got - expected) > 1.0:
            _fail(f"median slide density for {label} is {got:.2f}, brief says {expected}")
    print(f"PASS density: {rho.size:,} cells, {n_healthy} healthy + {n_cirrhosis} cirrhosis")
    print(f"PASS deciles: {np.array2string(deciles, precision=1)}")
    print(f"PASS slide medians: cirrhosis {np.median(slide_medians[slide_labels == 'cirrhosis']):.0f},"
          f" healthy {np.median(slide_medians[slide_labels == 'healthy']):.0f} cells/mm^2")

    slice_files = sorted(RESULTS.glob("slice_*.npz"))
    if len(slice_files) != 10:
        _fail(f"found {len(slice_files)} slice result files, need 10")
    for k, path in enumerate(slice_files):
        data = np.load(path, allow_pickle=False)
        if int(data["slice_index"]) != k:
            _fail(f"{path.name} holds slice {int(data['slice_index'])}, expected {k}")
        group_ids = data["group_ids"]
        if group_ids.size != N_SLIDES:
            _fail(f"slice {k}: {group_ids.size} groups, need {N_SLIDES}")
        if set(group_ids.tolist()) != set(slide_categories.tolist()):
            _fail(f"slice {k}: group_ids are not the 123 slides")
        kept = data["kept_per_slide"]
        total = data["slide_total"]
        if not np.array_equal(total, slide_counts):
            _fail(f"slice {k}: slide_total differs from the cohort counts")
        if kept.size != N_SLIDES or kept.min() < MIN_KEPT_ANY_SLICE:
            _fail(f"slice {k}: a slide keeps fewer than {MIN_KEPT_ANY_SLICE} cells")
        if int(data["kept_total"]) != int(kept.sum()):
            _fail(f"slice {k}: kept_total disagrees with kept_per_slide")
        if k == 0:
            if int(kept.sum()) != N_CELLS or not np.array_equal(kept, total):
                _fail("slice 0 does not keep every cell")
            for dim in ("h0", "h1"):
                for norm in ("none", "cosine"):
                    acc = data[f"{dim}_{norm}_accuracy"]
                    if acc.shape != (len(SEEDS),):
                        _fail(f"slice 0 {dim}/{norm}: accuracy has shape {acc.shape}")
                    got = round(float(acc[0]), 3)
                    want = TUTORIAL_ACCURACY[(dim, norm)]
                    if got != want:
                        _fail(
                            f"slice 0 {dim}/{norm} seed 0: {acc[0]:.6f} (rounds to "
                            f"{got}), tutorial 04 says {want}"
                        )
        if k in MEASURED_KEPT:
            frac = data["kept_fraction"]
            labels = data["group_label"]
            for label, want in zip(("cirrhosis", "healthy"), MEASURED_KEPT[k]):
                got = float(np.median(frac[labels == label]))
                if abs(got - want) > 0.01:
                    _fail(
                        f"slice {k}: median kept fraction for {label} is {got:.3f},"
                        f" brief says {want:.2f}"
                    )
        if k == 9:
            if int(kept.min()) != MIN_KEPT_AT_Q09:
                _fail(f"slice 9: smallest kept slide {int(kept.min())} != {MIN_KEPT_AT_Q09}")
        for name in ("h0_none", "h1_none", "h0_cosine", "h1_cosine"):
            for field in ("accuracy", "confusion"):
                arr = data[f"{name}_{field}"]
                first = len(SEEDS) if field == "accuracy" else len(SEEDS)
                if arr.shape[0] != first:
                    _fail(f"slice {k} {name}: {field} has {arr.shape[0]} seeds")
        for field in ("control_accuracy", "control_confusion"):
            if data[field].shape[0] != len(SEEDS):
                _fail(f"slice {k}: {field} has {data[field].shape[0]} seeds")
        y = data["y"]
        if set(np.unique(y).tolist()) != {0, 1}:
            _fail(f"slice {k}: y is not healthy=0/cirrhosis=1")
        if data["h0_none_confusion"].shape[1:] != (2, 2):
            _fail(f"slice {k}: confusion matrix is not 2x2")
        if np.any(data["h0_none_confusion"].sum(axis=(1, 2)) != N_SLIDES):
            _fail(f"slice {k}: a confusion matrix does not hold all 123 slides")
        print(
            f"PASS slice {k}: d={float(data['threshold']):8.1f} cells/mm^2, "
            f"{int(data['kept_total']):>10,} cells kept, min {int(kept.min()):>6,}/slide, "
            f"tutorial match {data['h0_none_accuracy'][0]:.3f}/"
            f"{data['h1_none_accuracy'][0]:.3f}/"
            f"{data['h0_cosine_accuracy'][0]:.3f}/"
            f"{data['h1_cosine_accuracy'][0]:.3f}"
        )

    combined_path = RESULTS / "combined.npz"
    if not combined_path.exists():
        _fail(f"{combined_path} missing; run combined_classify.py")
    with np.load(combined_path, allow_pickle=False) as combined:
        with np.load(RESULTS / "slice_0.npz", allow_pickle=False) as slice0:
            if not np.array_equal(combined["group_ids"], slice0["group_ids"]):
                _fail("combined: group_ids differ from slice 0")
            if not np.array_equal(combined["y"], slice0["y"]):
                _fail("combined: y differs from slice 0")
        if combined["seeds"].size != len(SEEDS):
            _fail("combined: wrong number of seeds")
        for name in ("h0_none", "h1_none", "h0_cosine", "h1_cosine"):
            scores = combined[f"{name}_scores"]
            if scores.shape != (N_SLIDES, 10 * 4):
                _fail(f"combined {name}: scores shape {scores.shape}")
            for k in range(10):
                with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                    if not np.array_equal(scores[:, 4 * k : 4 * k + 4],
                                          one[f"{name}_scores"][:, :4]):
                        _fail(f"combined {name}: block {k} is not slice {k}'s first 4 scores")
            if combined[f"{name}_accuracy"].shape != (len(SEEDS),):
                _fail(f"combined {name}: accuracy shape")
            if np.any(combined[f"{name}_confusion"].sum(axis=(1, 2)) != N_SLIDES):
                _fail(f"combined {name}: confusion matrix does not hold all slides")
        control = combined["control_scores"]
        if control.shape != (N_SLIDES, 10):
            _fail(f"combined control: scores shape {control.shape}")
        for k in range(10):
            with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                if not np.array_equal(control[:, k], one["kept_fraction"]):
                    _fail(f"combined control: column {k} is not slice {k}'s kept fraction")
        if combined["control_accuracy"].shape != (len(SEEDS),):
            _fail("combined control: accuracy shape")
        if np.any(combined["control_confusion"].sum(axis=(1, 2)) != N_SLIDES):
            _fail("combined control: confusion matrix does not hold all slides")
        print(
            "PASS combined: 4 models x 40 features and a 10-feature control, "
            "20 seeds, blocks equal the per-slice scores"
        )

    adds_path = RESULTS / "density_adds.npz"
    if not adds_path.exists():
        _fail(f"{adds_path} missing; run density_adds.py")
    with np.load(adds_path, allow_pickle=False) as adds, np.load(
        RESULTS / "slice_0.npz", allow_pickle=False
    ) as slice0:
        y = slice0["y"]
        if not np.array_equal(adds["group_ids"], slice0["group_ids"]):
            _fail("density_adds: group_ids differ from slice 0")
        if not np.array_equal(adds["y"], y):
            _fail("density_adds: y differs from slice 0")
        # Slide size belongs to the density arm: raw Betti curves scale with
        # the number of cells, and cirrhosis slides have more of them.
        if "log_cells" not in adds.files:
            _fail("density_adds: log_cells missing; the density arm needs the slide size")
        log_cells = adds["log_cells"]
        if log_cells.shape != (N_SLIDES,) or not np.allclose(
            log_cells, np.log10(slice0["kept_per_slide"].astype(float)), rtol=0.0, atol=1e-12
        ):
            _fail("density_adds: log_cells is not log10 of slice 0's cells per slide")
        if adds["seeds"].size != len(SEEDS):
            _fail("density_adds: wrong number of seeds")
        rich_all = adds["rich_scores_all"]
        if rich_all.shape != (N_SLIDES, 10 * 10):
            _fail(f"density_adds: rich_scores_all shape {rich_all.shape}")
        for k in range(10):
            rich = adds[f"rich_scores_k{k}"]
            if rich.shape != (N_SLIDES, 10):
                _fail(f"density_adds: rich_scores_k{k} shape {rich.shape}")
            if not np.array_equal(rich_all[:, 10 * k : 10 * k + 10], rich):
                _fail(f"density_adds: rich block {k} differs from slice {k}")
            with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                if not np.array_equal(rich[:, 9], one["kept_fraction"]):
                    _fail(f"density_adds: slice {k} kept fraction column differs")
            if not np.all(np.isfinite(rich)):
                _fail(f"density_adds: slice {k} rich features not finite")
            if np.any(np.diff(rich[:, :7], axis=1) < -1e-12):
                _fail(f"density_adds: slice {k} quantiles not monotone")
            if np.any(rich[:, 8] < 0):
                _fail(f"density_adds: slice {k} negative sd")
        if not np.allclose(rich_all[:, 9], 1.0):
            _fail("density_adds: slice 0 keeps every cell but its fraction is not 1")

        from sklearn.metrics import roc_auc_score

        per_slice = [str(k) for k in range(10)]
        arms: dict[str, list[str]] = {"A": per_slice + ["all"], "S": ["all"]}
        for name in MODELS:
            arms[f"B_{name}"] = per_slice + ["all"]
            arms[f"R_{name}"] = per_slice
            arms[f"P1_{name}"] = per_slice + ["all"]
            arms[f"P24_{name}"] = per_slice + ["all"]
            # Residualising 40 scores on 101 density features with ~98
            # training slides interpolates: the training residuals vanish.
            for stale in (f"R_{name}_acc_all", f"ci_res_{name}_all", f"ci_res_chance_{name}_all"):
                if stale in adds.files:
                    _fail(f"density_adds: {stale} exists; a combined residual arm is p >= n")
        for field, suffixes in arms.items():
            for suffix in suffixes:
                tag = f"density_adds {field}/{suffix}"
                acc = adds[f"{field}_acc_{suffix}"]
                correct = adds[f"{field}_correct_{suffix}"]
                proba = adds[f"{field}_proba_{suffix}"]
                logloss = adds[f"{field}_logloss_{suffix}"]
                auc = adds[f"{field}_auc_{suffix}"]
                if acc.shape != (len(SEEDS),) or logloss.shape != (len(SEEDS),) or auc.shape != (len(SEEDS),):
                    _fail(f"{tag}: per-seed shapes {acc.shape}, {logloss.shape}, {auc.shape}")
                if correct.shape != (len(SEEDS), N_SLIDES) or proba.shape != (len(SEEDS), N_SLIDES):
                    _fail(f"{tag}: per-slide shapes {correct.shape}, {proba.shape}")
                if not np.allclose(correct.mean(axis=1), acc):
                    _fail(f"{tag}: accuracy != mean correctness")
                if np.any((proba < 0) | (proba > 1)):
                    _fail(f"{tag}: probability out of range")
                if not np.array_equal(correct, (proba > 0.5) == (y == 1)):
                    _fail(f"{tag}: predictions disagree with the probabilities")
                if not np.allclose(logloss, _log_loss(proba, y).mean(axis=1), rtol=0.0, atol=1e-12):
                    _fail(f"{tag}: log-loss != mean per-slide log-loss")
                want_auc = np.array([roc_auc_score(y, p) for p in proba])
                if not np.allclose(auc, want_auc, rtol=0.0, atol=1e-12):
                    _fail(f"{tag}: AUC != roc_auc_score of the stored probabilities")

        # Chance: the training fold's majority class and its prevalence.
        chance_correct = adds["chance_correct"]
        base_proba = adds["base_proba"]
        majority = int(y.mean() > 0.5)
        if chance_correct.shape != (len(SEEDS), N_SLIDES) or not np.all(
            chance_correct == (y == majority)
        ):
            _fail("density_adds: chance_correct is not the majority-class prediction")
        from sklearn.model_selection import StratifiedKFold

        for seed in SEEDS:
            want = np.empty(N_SLIDES)
            splits = StratifiedKFold(5, shuffle=True, random_state=int(seed)).split(y, y)
            for train, test in splits:
                want[test] = y[train].mean()
            if not np.allclose(base_proba[seed], want, rtol=0.0, atol=1e-12):
                _fail(f"density_adds: base_proba seed {seed} is not the training-fold prevalence")

        for name in MODELS:
            for suffix in per_slice + ["all"]:
                for kind in ("nested", "nested_ll", "pc24", "pc24_ll"):
                    _check_ci(adds, f"ci_{kind}_{name}_{suffix}")
            for suffix in per_slice:
                for kind in ("res_chance", "res_ll"):
                    _check_ci(adds, f"ci_{kind}_{name}_{suffix}")
        r2 = adds["pc_r2"]
        if r2.shape != (10, 4, 4):
            _fail(f"density_adds: pc_r2 shape {r2.shape}")
        if np.any((r2 < 0) | (r2 > 1)):
            _fail("density_adds: pc_r2 out of range")

        # Arm A is the documented recipe: ten density features plus slide size,
        # under the protocol estimator and folds (independent code path).
        from sklearn.model_selection import cross_val_predict
        from topocyte.tl.fpca import _logistic

        features = np.hstack([adds["rich_scores_k0"], log_cells[:, None]])
        want_proba = cross_val_predict(
            _logistic(10, 5, "accuracy", 1000, 0),
            features,
            y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
            method="predict_proba",
        )[:, 1]
        if not np.allclose(adds["A_proba_0"][0], want_proba, rtol=0.0, atol=1e-10):
            _fail("density_adds: arm A slice 0 seed 0 is not density + slide size")

        # The score-only arms did not change; they must reproduce the first run.
        superseded = HERE / "superseded" / "density_adds_2026-11-16.npz"
        if superseded.exists():
            with np.load(superseded, allow_pickle=False) as old:
                for name in MODELS:
                    for field in (f"P1_{name}", f"P24_{name}"):
                        for suffix in per_slice + ["all"]:
                            key = f"{field}_correct_{suffix}"
                            if not np.array_equal(adds[key], old[key]):
                                _fail(f"density_adds: {key} differs from the 2026-11-16 run")
            print("PASS density_adds: score-only arms reproduce the 2026-11-16 run exactly")
        else:
            print("SKIP density_adds: no superseded 2026-11-16 run to compare with")
        print(
            "PASS density_adds: density arm = 10 features + slide size, per-slice residual "
            "arms only, probabilities, log-loss and AUC consistent, chance baselines exact, "
            "bootstrap CIs ordered"
        )

    _check_ratio_slices()
    _check_scaled_nested()
    _check_scaled_section7()
    _check_scaled_ratio_alone()
    _check_scaled_single_block()
    _check_scaled_ph_only()
    print("ALL CHECKS PASSED")


RATIO_SLICES = (0, 6, 7, 8, 9)
RATIO_FAMILIES = ("none", "cosine")
RATIO_LEVELS = (0.5, 0.9, 0.99, 0.999)


def _check_ratio_slices() -> None:
    """Task 7: fPCA of the H1 death/birth ratio, nested against the density arm."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from topocyte.tl.fpca import _logistic

    path = RESULTS / "ratio_slices.npz"
    if not path.exists():
        _fail(f"{path} missing; run ratio_slices.py")
    with np.load(path, allow_pickle=False) as ratio, np.load(
        RESULTS / "density_adds.npz", allow_pickle=False
    ) as adds:
        y = adds["y"]
        if not np.array_equal(ratio["group_ids"], adds["group_ids"]):
            _fail("ratio_slices: group_ids differ from density_adds")
        if not np.array_equal(ratio["y"], y):
            _fail("ratio_slices: y differs from density_adds")
        if tuple(int(k) for k in ratio["slices"]) != RATIO_SLICES:
            _fail(f"ratio_slices: slices {ratio['slices']} are not {RATIO_SLICES}")
        for key in ("chance_correct", "base_proba"):
            if not np.array_equal(ratio[key], adds[key]):
                _fail(f"ratio_slices: {key} differs from density_adds")

        for k in RATIO_SLICES:
            with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                if not np.array_equal(ratio[f"kept_{k}"], one["kept_per_slide"]):
                    _fail(f"ratio_slices: slice {k} keeps other cells than run_slices.py")
                # Gate: the recomputed diagrams give the saved H1 fPCA back.
                for m in RATIO_FAMILIES:
                    got = ratio[f"betti_{m}_scores_{k}"]
                    saved = one[f"h1_{m}_scores"][:, :4]
                    if got.shape != saved.shape:
                        _fail(f"ratio_slices: betti_{m}_scores_{k} shape {got.shape}")
                    signs = np.sign(np.sum(got * saved, axis=0))
                    scale = np.max(np.abs(saved), axis=0)
                    deviation = np.max(np.abs(got * signs - saved), axis=0) / scale
                    if np.any(deviation > 1e-6):
                        _fail(
                            f"ratio_slices: slice {k} H1 {m} scores do not reproduce the "
                            f"saved ones (relative deviation {deviation.max():.2e})"
                        )
            # The density arm is recomputed and must equal section 7's.
            for field in ("correct", "proba"):
                if not np.array_equal(ratio[f"A_{field}_{k}"], adds[f"A_{field}_{k}"]):
                    _fail(f"ratio_slices: A_{field}_{k} differs from density_adds")

            n_bars = ratio[f"n_bars_{k}"]
            sums = ratio[f"sum_log_pi_{k}"]
            if n_bars.shape != (N_SLIDES,) or np.any(n_bars <= 0):
                _fail(f"ratio_slices: slice {k} has a slide without H1 bars")
            if np.any(sums <= 0) or not np.all(np.isfinite(sums)):
                _fail(f"ratio_slices: slice {k} log pi sums not positive and finite")
            quantiles = ratio[f"log_pi_quantiles_{k}"]
            if quantiles.shape != (N_SLIDES, len(RATIO_LEVELS)) or np.any(quantiles < 0):
                _fail(f"ratio_slices: slice {k} log pi quantiles shape or sign")
            if np.any(np.diff(quantiles, axis=1) < 0):
                _fail(f"ratio_slices: slice {k} log pi quantiles not monotone")
            if np.any(ratio[f"max_log_pi_{k}"] < quantiles[:, -1]):
                _fail(f"ratio_slices: slice {k} max log pi below its 0.999 quantile")

            # The survival curves' Gram matrix is exact: topocyte's kernel equals
            # the closed form sum_ij min(v_i, v_j) for every pair of slides.
            gram = ratio[f"ratio_gram_{k}"]
            if gram.shape != (N_SLIDES, N_SLIDES) or not np.allclose(gram, gram.T, rtol=1e-13, atol=0):
                _fail(f"ratio_slices: slice {k} Gram matrix shape or symmetry")
            if float(ratio[f"gram_check_rel_{k}"]) > 1e-10:
                _fail(
                    f"ratio_slices: slice {k} Gram differs from the closed form "
                    f"({float(ratio[f'gram_check_rel_{k}']):.2e})"
                )
            diagonal = np.diag(gram)
            if np.any(diagonal < sums * (1 - 1e-12)) or np.any(diagonal > n_bars * sums * (1 + 1e-12)):
                _fail(f"ratio_slices: slice {k} Gram diagonal outside [sum v, n sum v]")

            for m in RATIO_FAMILIES:
                scores = ratio[f"ratio_{m}_scores_{k}"]
                if scores.shape[0] != N_SLIDES or scores.shape[1] < 4 or not np.all(np.isfinite(scores)):
                    _fail(f"ratio_slices: ratio_{m}_scores_{k} shape {scores.shape} or non-finite")
                evr = ratio[f"ratio_{m}_evr_{k}"]
                if np.any(evr < 0) or evr.sum() > 1 + 1e-9:
                    _fail(f"ratio_slices: ratio_{m}_evr_{k} not a variance split")
                for field in (f"Q_{m}", f"N_{m}"):
                    tag = f"ratio_slices {field}/{k}"
                    correct = ratio[f"{field}_correct_{k}"]
                    proba = ratio[f"{field}_proba_{k}"]
                    if correct.shape != (len(SEEDS), N_SLIDES) or proba.shape != (len(SEEDS), N_SLIDES):
                        _fail(f"{tag}: per-slide shapes {correct.shape}, {proba.shape}")
                    if not np.allclose(correct.mean(axis=1), ratio[f"{field}_acc_{k}"]):
                        _fail(f"{tag}: accuracy != mean correctness")
                    if not np.array_equal(correct, (proba > 0.5) == (y == 1)):
                        _fail(f"{tag}: predictions disagree with the probabilities")
                    if not np.allclose(
                        ratio[f"{field}_logloss_{k}"], _log_loss(proba, y).mean(axis=1), rtol=0.0, atol=1e-12
                    ):
                        _fail(f"{tag}: log-loss != mean per-slide log-loss")
                    want_auc = np.array([roc_auc_score(y, p) for p in proba])
                    if not np.allclose(ratio[f"{field}_auc_{k}"], want_auc, rtol=0.0, atol=1e-12):
                        _fail(f"{tag}: AUC != roc_auc_score of the stored probabilities")
                for kind in ("nested", "nested_ll", "alone_chance", "alone_ll", "alone_density"):
                    _check_ci(ratio, f"ci_{kind}_{m}_{k}", "ratio_slices")

        # Independent code path for one nested and one ratio-only case: the arms
        # hold the documented features under the protocol estimator and folds.
        for field, X in (
            ("N_cosine", np.hstack([
                np.hstack([adds["rich_scores_k0"], adds["log_cells"][:, None]]),
                ratio["ratio_cosine_scores_0"][:, :4],
            ])),
            ("Q_none", ratio["ratio_none_scores_9"][:, :4]),
        ):
            suffix = "0" if field.startswith("N") else "9"
            want = cross_val_predict(
                _logistic(10, 5, "accuracy", 1000, 0),
                X,
                y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                method="predict_proba",
            )[:, 1]
            if not np.allclose(ratio[f"{field}_proba_{suffix}"][0], want, rtol=0.0, atol=1e-10):
                _fail(f"ratio_slices: {field} slice {suffix} seed 0 is not its documented arm")

        # The intervals use density_adds.py's shared bootstrap indices.
        boot_index = np.random.default_rng(12345).integers(0, N_SLIDES, (4000, N_SLIDES))
        delta = ratio["N_cosine_correct_0"].mean(axis=0) - ratio["A_correct_0"].mean(axis=0)
        draws = delta[boot_index].mean(axis=1)
        want_ci = np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])
        if not np.allclose(ratio["ci_nested_cosine_0"], want_ci, rtol=0.0, atol=1e-12):
            _fail("ratio_slices: ci_nested_cosine_0 is not the shared-index paired bootstrap")

        r2 = ratio["ratio_r2"]
        if r2.shape != (len(RATIO_SLICES), len(RATIO_FAMILIES), 4) or np.any((r2 < 0) | (r2 > 1)):
            _fail(f"ratio_slices: ratio_r2 shape {r2.shape} or out of range")
    print(
        "PASS ratio_slices: recomputed diagrams reproduce the saved H1 fPCA, density arm "
        "equals section 7's, survival-curve Gram exact against the closed form, arms "
        "consistent, shared-index bootstrap CIs"
    )


SCALED_SOURCES = ("ratio", "betti")
CS_GRID = np.logspace(-4, 4, 10)


def _check_arm(data: np.lib.npyio.NpzFile, field: str, suffix: str, y: np.ndarray, tag: str) -> None:
    """Per-slide outputs of one arm: shapes, accuracy, predictions, log-loss and AUC agree."""
    from sklearn.metrics import roc_auc_score

    correct = data[f"{field}_correct_{suffix}"]
    proba = data[f"{field}_proba_{suffix}"]
    if correct.shape != (len(SEEDS), N_SLIDES) or proba.shape != (len(SEEDS), N_SLIDES):
        _fail(f"{tag}: per-slide shapes {correct.shape}, {proba.shape}")
    if not np.allclose(correct.mean(axis=1), data[f"{field}_acc_{suffix}"]):
        _fail(f"{tag}: accuracy != mean correctness")
    if not np.array_equal(correct, (proba > 0.5) == (y == 1)):
        _fail(f"{tag}: predictions disagree with the probabilities")
    if not np.allclose(data[f"{field}_logloss_{suffix}"], _log_loss(proba, y).mean(axis=1), rtol=0.0, atol=1e-12):
        _fail(f"{tag}: log-loss != mean per-slide log-loss")
    want_auc = np.array([roc_auc_score(y, p) for p in proba])
    if not np.allclose(data[f"{field}_auc_{suffix}"], want_auc, rtol=0.0, atol=1e-12):
        _fail(f"{tag}: AUC != roc_auc_score of the stored probabilities")


def _check_scaled_nested() -> None:
    """Post-hoc: the nested arms refitted with features standardized in each training fold.

    ``P_*`` arms refit the protocol (unscaled) nested arms to record each fold's
    chosen C and the share of the classifier's weight that the four scores
    carry; they must equal the saved arms.  ``Z_*`` arms put a StandardScaler,
    fitted on the outer training fold, in front of the same estimator.
    """
    import warnings

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    path = RESULTS / "scaled_nested.npz"
    if not path.exists():
        _fail(f"{path} missing; run scaled_nested.py")
    with np.load(path, allow_pickle=False) as scaled, np.load(
        RESULTS / "ratio_slices.npz", allow_pickle=False
    ) as ratio, np.load(RESULTS / "density_adds.npz", allow_pickle=False) as adds:
        y = adds["y"]
        for key in ("y", "group_ids"):
            if not np.array_equal(scaled[key], adds[key]):
                _fail(f"scaled_nested: {key} differs from density_adds")
        if tuple(int(k) for k in scaled["slices"]) != RATIO_SLICES:
            _fail(f"scaled_nested: slices {scaled['slices']} are not {RATIO_SLICES}")

        for k in RATIO_SLICES:
            density = np.hstack([adds[f"rich_scores_k{k}"], adds["log_cells"][:, None]])
            if not np.allclose(scaled[f"density_sd_{k}"], density.std(axis=0, ddof=1), rtol=1e-12, atol=0.0):
                _fail(f"scaled_nested: density_sd_{k} is not the density arm's SD")
            with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                blocks = {
                    **{("ratio", m): ratio[f"ratio_{m}_scores_{k}"][:, :4] for m in RATIO_FAMILIES},
                    **{("betti", m): one[f"h1_{m}_scores"][:, :4] for m in RATIO_FAMILIES},
                }
            for (source, m), block in blocks.items():
                if not np.allclose(
                    scaled[f"score_sd_{source}_{m}_{k}"], block.std(axis=0, ddof=1), rtol=1e-12, atol=0.0
                ):
                    _fail(f"scaled_nested: score_sd_{source}_{m}_{k} is not the scores' SD")

            # Gate: the unscaled refits are the protocol arms sections 7 and 8 report.
            for m in RATIO_FAMILIES:
                for source, saved, key in (("ratio", ratio, f"N_{m}"), ("betti", adds, f"B_h1_{m}")):
                    field = f"P_{source}_{m}"
                    if not np.array_equal(scaled[f"{field}_correct_{k}"], saved[f"{key}_correct_{k}"]):
                        _fail(f"scaled_nested: {field}_correct_{k} is not the saved {key} arm")
                    if not np.allclose(
                        scaled[f"{field}_proba_{k}"], saved[f"{key}_proba_{k}"], rtol=0.0, atol=1e-12
                    ):
                        _fail(f"scaled_nested: {field}_proba_{k} is not the saved {key} arm")

            for field in ["Z_density"] + [f"Z_{s}_{m}" for s in SCALED_SOURCES for m in RATIO_FAMILIES]:
                _check_arm(scaled, field, str(k), y, f"scaled_nested {field}/{k}")
            for prefix in ("P", "Z"):
                for source in SCALED_SOURCES:
                    for m in RATIO_FAMILIES:
                        field = f"{prefix}_{source}_{m}"
                        chosen = scaled[f"{field}_C_{k}"]
                        share = scaled[f"{field}_share_{k}"]
                        if chosen.shape != (len(SEEDS), 5) or share.shape != (len(SEEDS), 5):
                            _fail(f"scaled_nested: {field} C or share shape at slice {k}")
                        gap = np.abs(np.log10(chosen)[..., None] - np.log10(CS_GRID)).min(axis=-1)
                        if np.any(gap > 1e-9):
                            _fail(f"scaled_nested: {field}_C_{k} not on the Cs grid")
                        if np.any((share < 0) | (share > 1)) or not np.all(np.isfinite(share)):
                            _fail(f"scaled_nested: {field}_share_{k} outside [0, 1]")
            for source in SCALED_SOURCES:
                for m in RATIO_FAMILIES:
                    for kind in ("nested", "nested_ll"):
                        _check_ci(scaled, f"ci_{kind}_{source}_{m}_{k}", "scaled_nested")
            _check_ci(scaled, f"ci_scaling_density_{k}", "scaled_nested")

        # Independent code path: one standardized arm under scikit-learn's own
        # pipeline, and one fold's C and weight share, unscaled and standardized.
        X = np.hstack([adds["rich_scores_k0"], adds["log_cells"][:, None], ratio["ratio_cosine_scores_0"][:, :4]])
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            want = cross_val_predict(
                make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0)),
                X,
                y,
                cv=folds,
                method="predict_proba",
            )[:, 1]
            if not np.allclose(scaled["Z_ratio_cosine_proba_0"][0], want, rtol=0.0, atol=1e-10):
                _fail("scaled_nested: Z_ratio_cosine slice 0 seed 0 is not the standardized pipeline")
            train, _ = next(folds.split(y, y))
            for prefix, model in (
                ("P", _logistic(10, 5, "accuracy", 1000, 0)),
                ("Z", make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0))),
            ):
                model.fit(X[train], y[train])
                seen = X[train] if prefix == "P" else model[0].transform(X[train])
                classifier = model if prefix == "P" else model[-1]
                weight = np.abs(classifier.coef_.ravel()) * seen.std(axis=0)
                want_share = weight[-4:].sum() / weight.sum()
                if not np.isclose(scaled[f"{prefix}_ratio_cosine_share_0"][0, 0], want_share, rtol=1e-9, atol=1e-12):
                    _fail(f"scaled_nested: {prefix}_ratio_cosine_share_0 is not the score block's |coef| x SD share")
                if not np.isclose(scaled[f"{prefix}_ratio_cosine_C_0"][0, 0], float(np.ravel(classifier.C_)[0])):
                    _fail(f"scaled_nested: {prefix}_ratio_cosine_C_0 is not the fold's chosen C")

        # The intervals use density_adds.py's shared bootstrap indices.
        boot_index = np.random.default_rng(12345).integers(0, N_SLIDES, (4000, N_SLIDES))
        density_ll = _log_loss(scaled["Z_density_proba_0"], y).mean(axis=0)
        nested_ll = _log_loss(scaled["Z_ratio_cosine_proba_0"], y).mean(axis=0)
        for key, delta in (
            ("ci_nested_ratio_cosine_0",
             scaled["Z_ratio_cosine_correct_0"].mean(axis=0) - scaled["Z_density_correct_0"].mean(axis=0)),
            ("ci_nested_ll_ratio_cosine_0", density_ll - nested_ll),
            ("ci_scaling_density_0",
             scaled["Z_density_correct_0"].mean(axis=0) - adds["A_correct_0"].mean(axis=0)),
        ):
            draws = delta[boot_index].mean(axis=1)
            want_ci = np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])
            if not np.allclose(scaled[key], want_ci, rtol=0.0, atol=1e-12):
                _fail(f"scaled_nested: {key} is not the shared-index paired bootstrap")
    print(
        "PASS scaled_nested: unscaled refits equal the saved nested arms, standardized arms "
        "consistent and equal to scikit-learn's pipeline, C on the grid, weight shares from "
        "|coef| x SD, shared-index bootstrap CIs"
    )


SECTION7_SUFFIXES = tuple(str(k) for k in range(10)) + ("all",)
SCALED_OVERLAP = ("0", "6", "7", "8", "9")


def _check_scaled_section7() -> None:
    """Post-hoc: every nested arm of section 7 refitted with standardized features.

    ``P_*`` arms refit section 7's nested arms unscaled and must equal them;
    ``Z_*`` arms put a StandardScaler, fitted on the outer training fold, in
    front of the same estimator.  Where ``scaled_nested.py`` refitted the same
    arms (H1 at slices 0 and 6-9, and the density arms there), both files must
    hold the same numbers.
    """
    import warnings

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    tag = "scaled_section7"
    path = RESULTS / "scaled_section7.npz"
    if not path.exists():
        _fail(f"{path} missing; run scaled_section7.py")
    with np.load(path, allow_pickle=False) as refit, np.load(
        RESULTS / "density_adds.npz", allow_pickle=False
    ) as adds, np.load(RESULTS / "combined.npz", allow_pickle=False) as combined, np.load(
        RESULTS / "scaled_nested.npz", allow_pickle=False
    ) as section8:
        y = adds["y"]
        for key in ("y", "group_ids"):
            if not np.array_equal(refit[key], adds[key]):
                _fail(f"{tag}: {key} differs from density_adds")
        if tuple(str(s) for s in refit["suffixes"]) != SECTION7_SUFFIXES:
            _fail(f"{tag}: suffixes {refit['suffixes']} are not {SECTION7_SUFFIXES}")
        if tuple(str(m) for m in refit["models"]) != MODELS:
            _fail(f"{tag}: models {refit['models']} are not {MODELS}")

        def density_arm(suffix: str) -> np.ndarray:
            rich = adds["rich_scores_all"] if suffix == "all" else adds[f"rich_scores_k{suffix}"]
            return np.hstack([rich, adds["log_cells"][:, None]])

        def score_block(name: str, suffix: str) -> np.ndarray:
            if suffix == "all":
                return combined[f"{name}_scores"]
            with np.load(RESULTS / f"slice_{suffix}.npz", allow_pickle=False) as one:
                return one[f"{name}_scores"][:, :4]

        for s in SECTION7_SUFFIXES:
            density = density_arm(s)
            if not np.allclose(refit[f"density_sd_{s}"], density.std(axis=0, ddof=1), rtol=1e-12, atol=0.0):
                _fail(f"{tag}: density_sd_{s} is not the density arm's SD")
            _check_arm(refit, "Z_density", s, y, f"{tag} Z_density/{s}")
            _check_ci(refit, f"ci_scaling_density_{s}", tag)
            for name in MODELS:
                block = score_block(name, s)
                if not np.allclose(refit[f"score_sd_{name}_{s}"], block.std(axis=0, ddof=1), rtol=1e-12, atol=0.0):
                    _fail(f"{tag}: score_sd_{name}_{s} is not the scores' SD")
                # Gate: the unscaled refits are the nested arms section 7 reports.
                if not np.array_equal(refit[f"P_{name}_correct_{s}"], adds[f"B_{name}_correct_{s}"]):
                    _fail(f"{tag}: P_{name}_correct_{s} is not the saved B arm")
                if not np.allclose(refit[f"P_{name}_proba_{s}"], adds[f"B_{name}_proba_{s}"], rtol=0.0, atol=1e-12):
                    _fail(f"{tag}: P_{name}_proba_{s} is not the saved B arm")
                _check_arm(refit, f"Z_{name}", s, y, f"{tag} Z_{name}/{s}")
                for prefix in ("P", "Z"):
                    field = f"{prefix}_{name}"
                    chosen = refit[f"{field}_C_{s}"]
                    share = refit[f"{field}_share_{s}"]
                    if chosen.shape != (len(SEEDS), 5) or share.shape != (len(SEEDS), 5):
                        _fail(f"{tag}: {field} C or share shape at {s}")
                    gap = np.abs(np.log10(chosen)[..., None] - np.log10(CS_GRID)).min(axis=-1)
                    if np.any(gap > 1e-9):
                        _fail(f"{tag}: {field}_C_{s} not on the Cs grid")
                    if np.any((share < 0) | (share > 1)) or not np.all(np.isfinite(share)):
                        _fail(f"{tag}: {field}_share_{s} outside [0, 1]")
                for kind in ("nested", "nested_ll"):
                    _check_ci(refit, f"ci_{kind}_{name}_{s}", tag)

            # Where section 8's post-hoc check refitted the same arms, the numbers agree.
            if s in SCALED_OVERLAP:
                pairs = [("Z_density", "Z_density", ("correct", "proba"))] + [
                    (f"Z_h1_{m}", f"Z_betti_{m}", ("correct", "proba", "C", "share"))
                    for m in RATIO_FAMILIES
                ]
                for mine, theirs, keys in pairs:
                    for key in keys:
                        a, b = refit[f"{mine}_{key}_{s}"], section8[f"{theirs}_{key}_{s}"]
                        same = np.array_equal(a, b) if key == "correct" else np.allclose(a, b, rtol=0.0, atol=1e-12)
                        if not same:
                            _fail(f"{tag}: {mine}_{key}_{s} differs from scaled_nested's {theirs}")
                for m in RATIO_FAMILIES:
                    for kind in ("nested", "nested_ll"):
                        if not np.allclose(
                            refit[f"ci_{kind}_h1_{m}_{s}"], section8[f"ci_{kind}_betti_{m}_{s}"], rtol=0.0, atol=1e-12
                        ):
                            _fail(f"{tag}: ci_{kind}_h1_{m}_{s} differs from scaled_nested's")
                if not np.allclose(refit[f"ci_scaling_density_{s}"], section8[f"ci_scaling_density_{s}"], rtol=0.0, atol=1e-12):
                    _fail(f"{tag}: ci_scaling_density_{s} differs from scaled_nested's")

        # Independent code path: the standardized combined H0 raw arm under
        # scikit-learn's own pipeline, and one fold's C and weight share, unscaled
        # and standardized, for an arm section 8 did not refit.
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            X = np.hstack([density_arm("all"), combined["h0_none_scores"]])
            want = cross_val_predict(
                make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0)),
                X,
                y,
                cv=folds,
                method="predict_proba",
            )[:, 1]
            if not np.allclose(refit["Z_h0_none_proba_all"][0], want, rtol=0.0, atol=1e-10):
                _fail(f"{tag}: Z_h0_none combined seed 0 is not the standardized pipeline")
            X = np.hstack([density_arm("3"), score_block("h0_cosine", "3")])
            train, _ = next(folds.split(y, y))
            for prefix, model in (
                ("P", _logistic(10, 5, "accuracy", 1000, 0)),
                ("Z", make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0))),
            ):
                model.fit(X[train], y[train])
                seen = X[train] if prefix == "P" else model[0].transform(X[train])
                classifier = model if prefix == "P" else model[-1]
                weight = np.abs(classifier.coef_.ravel()) * seen.std(axis=0)
                want_share = weight[-4:].sum() / weight.sum()
                if not np.isclose(refit[f"{prefix}_h0_cosine_share_3"][0, 0], want_share, rtol=1e-9, atol=1e-12):
                    _fail(f"{tag}: {prefix}_h0_cosine_share_3 is not the score block's |coef| x SD share")
                if not np.isclose(refit[f"{prefix}_h0_cosine_C_3"][0, 0], float(np.ravel(classifier.C_)[0])):
                    _fail(f"{tag}: {prefix}_h0_cosine_C_3 is not the fold's chosen C")

        # The intervals use density_adds.py's shared bootstrap indices.
        boot_index = np.random.default_rng(12345).integers(0, N_SLIDES, (4000, N_SLIDES))
        density_ll = _log_loss(refit["Z_density_proba_all"], y).mean(axis=0)
        nested_ll = _log_loss(refit["Z_h0_none_proba_all"], y).mean(axis=0)
        for key, delta in (
            ("ci_nested_h0_none_all",
             refit["Z_h0_none_correct_all"].mean(axis=0) - refit["Z_density_correct_all"].mean(axis=0)),
            ("ci_nested_ll_h0_none_all", density_ll - nested_ll),
            ("ci_scaling_density_all",
             refit["Z_density_correct_all"].mean(axis=0) - adds["A_correct_all"].mean(axis=0)),
        ):
            draws = delta[boot_index].mean(axis=1)
            want_ci = np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])
            if not np.allclose(refit[key], want_ci, rtol=0.0, atol=1e-12):
                _fail(f"{tag}: {key} is not the shared-index paired bootstrap")
    print(
        "PASS scaled_section7: unscaled refits equal section 7's 44 nested arms, standardized "
        "arms consistent, equal to scaled_nested's where they overlap and to scikit-learn's "
        "pipeline, C on the grid, weight shares from |coef| x SD, shared-index bootstrap CIs"
    )


def _check_scaled_ratio_alone() -> None:
    """Post-hoc: section 8's ratio-alone arms refitted with standardized features.

    ``P_Q_*`` refit the protocol ratio-alone arms unscaled to record each fold's
    C and must equal them; ``Z_Q_*`` put a StandardScaler, fitted on the outer
    training fold, in front of the same estimator.
    """
    import warnings

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    tag = "scaled_ratio_alone"
    path = RESULTS / "scaled_ratio_alone.npz"
    if not path.exists():
        _fail(f"{path} missing; run scaled_ratio_alone.py")
    with np.load(path, allow_pickle=False) as alone, np.load(
        RESULTS / "ratio_slices.npz", allow_pickle=False
    ) as ratio:
        y = ratio["y"]
        for key in ("y", "group_ids"):
            if not np.array_equal(alone[key], ratio[key]):
                _fail(f"{tag}: {key} differs from ratio_slices")
        if tuple(int(k) for k in alone["slices"]) != RATIO_SLICES:
            _fail(f"{tag}: slices {alone['slices']} are not {RATIO_SLICES}")
        if not np.allclose(alone["cs_grid"], CS_GRID, rtol=1e-12, atol=0.0):
            _fail(f"{tag}: cs_grid is not the protocol's logspace(-4, 4, 10)")
        for k in RATIO_SLICES:
            for m in RATIO_FAMILIES:
                block = ratio[f"ratio_{m}_scores_{k}"][:, :4]
                if not np.allclose(alone[f"score_sd_{m}_{k}"], block.std(axis=0, ddof=1), rtol=1e-12, atol=0.0):
                    _fail(f"{tag}: score_sd_{m}_{k} is not the scores' SD")
                # Gate: the unscaled refits are the ratio-alone arms section 8 reports.
                if not np.array_equal(alone[f"P_Q_{m}_correct_{k}"], ratio[f"Q_{m}_correct_{k}"]):
                    _fail(f"{tag}: P_Q_{m}_correct_{k} is not the saved Q arm")
                if not np.allclose(alone[f"P_Q_{m}_proba_{k}"], ratio[f"Q_{m}_proba_{k}"], rtol=0.0, atol=1e-12):
                    _fail(f"{tag}: P_Q_{m}_proba_{k} is not the saved Q arm")
                for field in (f"P_Q_{m}", f"Z_Q_{m}"):
                    _check_arm(alone, field, str(k), y, f"{tag} {field}/{k}")
                    chosen = alone[f"{field}_C_{k}"]
                    if chosen.shape != (len(SEEDS), 5):
                        _fail(f"{tag}: {field}_C_{k} shape {chosen.shape}")
                    gap = np.abs(np.log10(chosen)[..., None] - np.log10(CS_GRID)).min(axis=-1)
                    if np.any(gap > 1e-9):
                        _fail(f"{tag}: {field}_C_{k} not on the Cs grid")
                for kind in ("scaling", "scaling_ll", "alone_chance", "alone_ll"):
                    _check_ci(alone, f"ci_{kind}_{m}_{k}", tag)

        # Independent code path: one standardized arm under scikit-learn's own
        # pipeline, and one fold's chosen C, unscaled and standardized.
        X = ratio["ratio_cosine_scores_0"][:, :4]
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            want = cross_val_predict(
                make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0)),
                X,
                y,
                cv=folds,
                method="predict_proba",
            )[:, 1]
            if not np.allclose(alone["Z_Q_cosine_proba_0"][0], want, rtol=0.0, atol=1e-10):
                _fail(f"{tag}: Z_Q_cosine slice 0 seed 0 is not the standardized pipeline")
            train, _ = next(folds.split(y, y))
            for prefix, model in (
                ("P", _logistic(10, 5, "accuracy", 1000, 0)),
                ("Z", make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0))),
            ):
                model.fit(X[train], y[train])
                classifier = model if prefix == "P" else model[-1]
                if not np.isclose(alone[f"{prefix}_Q_cosine_C_0"][0, 0], float(np.ravel(classifier.C_)[0])):
                    _fail(f"{tag}: {prefix}_Q_cosine_C_0 is not the fold's chosen C")

        # The intervals use density_adds.py's shared bootstrap indices.
        boot_index = np.random.default_rng(12345).integers(0, N_SLIDES, (4000, N_SLIDES))
        protocol_ll = _log_loss(alone["P_Q_cosine_proba_0"], y).mean(axis=0)
        standard_ll = _log_loss(alone["Z_Q_cosine_proba_0"], y).mean(axis=0)
        for key, delta in (
            ("ci_scaling_cosine_0",
             alone["Z_Q_cosine_correct_0"].mean(axis=0) - alone["P_Q_cosine_correct_0"].mean(axis=0)),
            ("ci_scaling_ll_cosine_0", protocol_ll - standard_ll),
            ("ci_alone_chance_cosine_0",
             alone["Z_Q_cosine_correct_0"].mean(axis=0) - ratio["chance_correct"].mean(axis=0)),
        ):
            draws = delta[boot_index].mean(axis=1)
            want_ci = np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])
            if not np.allclose(alone[key], want_ci, rtol=0.0, atol=1e-12):
                _fail(f"{tag}: {key} is not the shared-index paired bootstrap")
    print(
        "PASS scaled_ratio_alone: unscaled refits equal section 8's ratio-alone arms, "
        "standardized arms consistent and equal to scikit-learn's pipeline, C on the grid, "
        "shared-index bootstrap CIs"
    )


def _check_scaled_single_block() -> None:
    """Post-hoc: section 7's residual and combined PC arms refitted with standardized features.

    ``P_*`` refit the protocol arms unscaled to record each fold's C and must
    equal them; ``Z_*`` put a StandardScaler, fitted on the outer training
    fold (for the residual arms on the training fold's residuals), in front of
    the same estimator.
    """
    import warnings

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    class Residualizer(TransformerMixin, BaseEstimator):
        """Residualise the last columns on the first ``n_density`` ones, column by column."""

        def __init__(self, n_density: int = 11) -> None:
            self.n_density = n_density

        def fit(self, X, y=None):
            d = self.n_density
            self.models_ = [LinearRegression().fit(X[:, :d], X[:, j]) for j in range(d, X.shape[1])]
            return self

        def transform(self, X):
            d = self.n_density
            return np.column_stack(
                [X[:, j] - m.predict(X[:, :d]) for j, m in zip(range(d, X.shape[1]), self.models_)]
            )

    tag = "scaled_single_block"
    path = RESULTS / "scaled_single_block.npz"
    if not path.exists():
        _fail(f"{path} missing; run scaled_single_block.py")
    with np.load(path, allow_pickle=False) as block, np.load(
        RESULTS / "density_adds.npz", allow_pickle=False
    ) as adds, np.load(RESULTS / "combined.npz", allow_pickle=False) as combined, np.load(
        RESULTS / "scaled_section7.npz", allow_pickle=False
    ) as refit7:
        y = adds["y"]
        for key in ("y", "group_ids"):
            if not np.array_equal(block[key], adds[key]):
                _fail(f"{tag}: {key} differs from density_adds")
        if tuple(str(m) for m in block["models"]) != MODELS:
            _fail(f"{tag}: models {block['models']} are not {MODELS}")
        if not np.allclose(block["cs_grid"], CS_GRID, rtol=1e-12, atol=0.0):
            _fail(f"{tag}: cs_grid is not the protocol's logspace(-4, 4, 10)")

        def arms(field: str, suffix: str, saved_field: str) -> None:
            if not np.array_equal(block[f"P_{field}_correct_{suffix}"], adds[f"{saved_field}_correct_{suffix}"]):
                _fail(f"{tag}: P_{field}_correct_{suffix} is not the saved {saved_field} arm")
            if not np.allclose(
                block[f"P_{field}_proba_{suffix}"], adds[f"{saved_field}_proba_{suffix}"], rtol=0.0, atol=1e-12
            ):
                _fail(f"{tag}: P_{field}_proba_{suffix} is not the saved {saved_field} arm")
            for prefix in ("P", "Z"):
                _check_arm(block, f"{prefix}_{field}", suffix, y, f"{tag} {prefix}_{field}/{suffix}")
                chosen = block[f"{prefix}_{field}_C_{suffix}"]
                if chosen.shape != (len(SEEDS), 5):
                    _fail(f"{tag}: {prefix}_{field}_C_{suffix} shape {chosen.shape}")
                gap = np.abs(np.log10(chosen)[..., None] - np.log10(CS_GRID)).min(axis=-1)
                if np.any(gap > 1e-9):
                    _fail(f"{tag}: {prefix}_{field}_C_{suffix} not on the Cs grid")

        for name in MODELS:
            for k in range(10):
                arms(f"R_{name}", str(k), f"R_{name}")
                density = np.hstack([adds[f"rich_scores_k{k}"], adds["log_cells"][:, None]])
                with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                    scores = one[f"{name}_scores"][:, :4]
                residual = np.column_stack(
                    [scores[:, j] - LinearRegression().fit(density, scores[:, j]).predict(density) for j in range(4)]
                )
                if not np.allclose(block[f"res_sd_{name}_{k}"], residual.std(axis=0, ddof=1), rtol=1e-9, atol=0.0):
                    _fail(f"{tag}: res_sd_{name}_{k} is not the in-sample residual SD")
                for kind in ("scaling_R", "scaling_ll_R", "res_chance", "res_ll"):
                    _check_ci(block, f"ci_{kind}_{name}_{k}", tag)
            every = combined[f"{name}_scores"]
            for part, saved_field, columns in (
                ("PC1", f"P1_{name}", every[:, 0::4]),
                ("PC24", f"P24_{name}", np.delete(every, np.s_[0::4], axis=1)),
            ):
                arms(f"{part}_{name}", "all", saved_field)
                if not np.allclose(block[f"score_sd_{part}_{name}_all"], columns.std(axis=0, ddof=1), rtol=1e-12, atol=0.0):
                    _fail(f"{tag}: score_sd_{part}_{name}_all is not the scores' SD")
                for kind in ("scaling", "scaling_ll", part.lower(), f"{part.lower()}_ll"):
                    _check_ci(block, f"ci_{kind}_{part}_{name}_all" if kind.startswith("scaling") else f"ci_{kind}_{name}_all", tag)

        # Independent code path: one standardized residual arm and one standardized
        # PC arm under scikit-learn's own pipeline, and one fold's chosen C.
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        density = np.hstack([adds["rich_scores_k8"], adds["log_cells"][:, None]])
        with np.load(RESULTS / "slice_8.npz", allow_pickle=False) as one:
            scores = one["h1_cosine_scores"][:, :4]
        X = np.hstack([density, scores])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            want = cross_val_predict(
                make_pipeline(Residualizer(density.shape[1]), StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0)),
                X,
                y,
                cv=folds,
                method="predict_proba",
            )[:, 1]
            if not np.allclose(block["Z_R_h1_cosine_proba_8"][0], want, rtol=0.0, atol=1e-10):
                _fail(f"{tag}: Z_R_h1_cosine slice 8 seed 0 is not the residualise-standardize pipeline")
            pc24 = np.delete(combined["h0_cosine_scores"], np.s_[0::4], axis=1)
            want = cross_val_predict(
                make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0)),
                pc24,
                y,
                cv=folds,
                method="predict_proba",
            )[:, 1]
            if not np.allclose(block["Z_PC24_h0_cosine_proba_all"][0], want, rtol=0.0, atol=1e-10):
                _fail(f"{tag}: Z_PC24_h0_cosine seed 0 is not the standardized pipeline")
            train, _ = next(folds.split(y, y))
            for prefix, model in (
                ("P", make_pipeline(Residualizer(density.shape[1]), _logistic(10, 5, "accuracy", 1000, 0))),
                ("Z", make_pipeline(Residualizer(density.shape[1]), StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0))),
            ):
                model.fit(X[train], y[train])
                if not np.isclose(block[f"{prefix}_R_h1_cosine_C_8"][0, 0], float(np.ravel(model[-1].C_)[0])):
                    _fail(f"{tag}: {prefix}_R_h1_cosine_C_8 is not the fold's chosen C")

        # The intervals use density_adds.py's shared bootstrap indices.
        boot_index = np.random.default_rng(12345).integers(0, N_SLIDES, (4000, N_SLIDES))
        base_ll = _log_loss(adds["base_proba"], y).mean(axis=0)
        z_res_ll = _log_loss(block["Z_R_h1_cosine_proba_8"], y).mean(axis=0)
        for key, delta in (
            ("ci_res_ll_h1_cosine_8", base_ll - z_res_ll),
            ("ci_scaling_R_h1_cosine_8",
             block["Z_R_h1_cosine_correct_8"].mean(axis=0) - block["P_R_h1_cosine_correct_8"].mean(axis=0)),
            ("ci_pc24_h0_cosine_all",
             block["Z_PC24_h0_cosine_correct_all"].mean(axis=0) - refit7["Z_density_correct_all"].mean(axis=0)),
        ):
            draws = delta[boot_index].mean(axis=1)
            want_ci = np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])
            if not np.allclose(block[key], want_ci, rtol=0.0, atol=1e-12):
                _fail(f"{tag}: {key} is not the shared-index paired bootstrap")
    print(
        "PASS scaled_single_block: unscaled refits equal section 7's residual and combined PC "
        "arms, standardized arms consistent and equal to scikit-learn's pipelines, C on the "
        "grid, shared-index bootstrap CIs"
    )


def _check_scaled_ph_only() -> None:
    """Post-hoc: the PH-only arms and density-only controls behind the verdict, standardized.

    ``P_*`` refit the arms that ``fpca_classify`` and ``combined_classify.py``
    scored and must give the same per-seed accuracy and confusion matrix;
    ``Z_*`` put a StandardScaler, fitted on the outer training fold, in front
    of the same estimator.
    """
    import warnings

    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    tag = "scaled_ph_only"
    path = RESULTS / "scaled_ph_only.npz"
    if not path.exists():
        _fail(f"{path} missing; run scaled_ph_only.py")
    arms = MODELS + ("control",)
    with np.load(path, allow_pickle=False) as refit, np.load(
        RESULTS / "combined.npz", allow_pickle=False
    ) as combined:
        slices = []
        for k in range(10):
            with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
                slices.append({key: one[key] for key in one.files})
        y = np.asarray(slices[0]["y"])
        for key in ("y", "group_ids"):
            if not np.array_equal(refit[key], slices[0][key]):
                _fail(f"{tag}: {key} differs from slice 0")
        if tuple(str(s) for s in refit["suffixes"]) != SECTION7_SUFFIXES:
            _fail(f"{tag}: suffixes {refit['suffixes']} are not {SECTION7_SUFFIXES}")
        if tuple(str(a) for a in refit["arms"]) != arms:
            _fail(f"{tag}: arms {refit['arms']} are not {arms}")
        if not np.allclose(refit["cs_grid"], CS_GRID, rtol=1e-12, atol=0.0):
            _fail(f"{tag}: cs_grid is not the protocol's logspace(-4, 4, 10)")

        def source(arm: str, suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Features, saved per-seed accuracy and saved confusion matrices of one arm."""
            if suffix == "all":
                field = "control" if arm == "control" else arm
                return combined[f"{field}_scores"], combined[f"{field}_accuracy"], combined[f"{field}_confusion"]
            one = slices[int(suffix)]
            if arm == "control":
                return (one["kept_per_slide"] / one["slide_total"])[:, None], one["control_accuracy"], one["control_confusion"]
            return one[f"{arm}_scores"][:, :4], one[f"{arm}_accuracy"], one[f"{arm}_confusion"]

        for s in SECTION7_SUFFIXES:
            for arm in arms:
                X, accuracy, confusion = source(arm, s)
                if not np.allclose(refit[f"feature_sd_{arm}_{s}"], X.std(axis=0, ddof=1), rtol=1e-12, atol=0.0, equal_nan=True):
                    _fail(f"{tag}: feature_sd_{arm}_{s} is not the features' SD")
                # Gate: the unscaled refit is the arm the verdict reports.
                if not np.array_equal(refit[f"P_{arm}_acc_{s}"], accuracy):
                    _fail(f"{tag}: P_{arm}_acc_{s} is not the saved per-seed accuracy")
                predicted = np.where(refit[f"P_{arm}_correct_{s}"], y, 1 - y)
                if not np.array_equal(
                    np.array([confusion_matrix(y, p, labels=[0, 1]) for p in predicted]), confusion
                ):
                    _fail(f"{tag}: P_{arm} {s} confusion matrices are not the saved ones")
                for prefix in ("P", "Z"):
                    _check_arm(refit, f"{prefix}_{arm}", s, y, f"{tag} {prefix}_{arm}/{s}")
                    chosen = refit[f"{prefix}_{arm}_C_{s}"]
                    if chosen.shape != (len(SEEDS), 5):
                        _fail(f"{tag}: {prefix}_{arm}_C_{s} shape {chosen.shape}")
                    gap = np.abs(np.log10(chosen)[..., None] - np.log10(CS_GRID)).min(axis=-1)
                    if np.any(gap > 1e-9):
                        _fail(f"{tag}: {prefix}_{arm}_C_{s} not on the Cs grid")
                for kind in ("scaling", "scaling_ll"):
                    _check_ci(refit, f"ci_{kind}_{arm}_{s}", tag)

        # Independent code path: a standardized combined model and an unscaled
        # per-slice arm through scikit-learn's cross_val_predict, as
        # fpca_classify calls it, and one fold's chosen C.
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            want = cross_val_predict(
                make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0)),
                combined["h1_none_scores"],
                y,
                cv=folds,
                method="predict_proba",
            )[:, 1]
            if not np.allclose(refit["Z_h1_none_proba_all"][0], want, rtol=0.0, atol=1e-10):
                _fail(f"{tag}: Z_h1_none combined seed 0 is not the standardized pipeline")
            held_out = cross_val_predict(
                _logistic(10, 5, "accuracy", 1000, 0), slices[3]["h0_cosine_scores"][:, :4], y, cv=folds
            )
            if not np.array_equal(refit["P_h0_cosine_correct_3"][0], held_out == y):
                _fail(f"{tag}: P_h0_cosine slice 3 seed 0 is not fpca_classify's held-out prediction")
            train, _ = next(folds.split(y, y))
            model = make_pipeline(StandardScaler(), _logistic(10, 5, "accuracy", 1000, 0))
            model.fit(combined["control_scores"][train], y[train])
            if not np.isclose(refit["Z_control_C_all"][0, 0], float(np.ravel(model[-1].C_)[0])):
                _fail(f"{tag}: Z_control_C_all is not the fold's chosen C")

        # The intervals use density_adds.py's shared bootstrap indices.
        boot_index = np.random.default_rng(12345).integers(0, N_SLIDES, (4000, N_SLIDES))
        for key, delta in (
            ("ci_scaling_h1_none_all",
             refit["Z_h1_none_correct_all"].mean(axis=0) - refit["P_h1_none_correct_all"].mean(axis=0)),
            ("ci_scaling_ll_h1_none_all",
             _log_loss(refit["P_h1_none_proba_all"], y).mean(axis=0)
             - _log_loss(refit["Z_h1_none_proba_all"], y).mean(axis=0)),
        ):
            draws = delta[boot_index].mean(axis=1)
            want_ci = np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])
            if not np.allclose(refit[key], want_ci, rtol=0.0, atol=1e-12):
                _fail(f"{tag}: {key} is not the shared-index paired bootstrap")
    print(
        "PASS scaled_ph_only: unscaled refits give the saved per-seed accuracies and confusion "
        "matrices of every PH-only arm and control, standardized arms consistent and equal to "
        "scikit-learn's pipeline, C on the grid, shared-index bootstrap CIs"
    )


if __name__ == "__main__":
    sys.exit(main())
