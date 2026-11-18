#!/usr/bin/env python
"""How much does persistent homology add beyond density?

Corrected on 2026-11-17; the first run of 2026-11-16 is kept in ``superseded/``.
All arms are answered from saved results (no PH or fPCA rerun) under the
protocol estimator and folds: LogisticRegressionCV via
``topocyte.tl.fpca._logistic`` inside StratifiedKFold(5, shuffle, seed).

* A, the density arm: ten density features per slice (log10 rho quantiles
  0.05 ... 0.95 among the kept cells, their mean and SD, the kept fraction)
  plus log10 of the slide's cell count.  Raw Betti curves scale with the
  number of cells and cirrhosis slides have more cells, so slide size belongs
  to the non-topological baseline.
* S: slide size alone.
* B, nested: A plus the first four fPCA scores of the slice.
* R, residual: the four scores residualised on A inside every outer training
  fold, per slice only.  A combined version would regress 40 scores on 101
  features with about 98 training slides; the fit interpolates, so it is not
  computed.  If PH were a density proxy, R would fall to chance.
* P1 and P24: the first score alone, and scores 2-4 alone.

Every arm stores, for 20 seeds, each slide's out-of-fold correctness and
probability of cirrhosis, and from them accuracy, log-loss and AUC.  Chance is
the training fold's majority class (accuracy) and its prevalence (log-loss).
Intervals are paired bootstraps over the 123 slides of seed-averaged
per-slide values, with one set of resampling indices shared by all
comparisons.

Writes ``results/density_adds.npz`` and ``results/density_adds.json``.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = np.arange(20)
MODELS = ("h0_none", "h1_none", "h0_cosine", "h1_cosine")
N_SLICES = 10
N_COMPONENTS = 4
QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
N_BOOT = 4000
BOOT_SEED = 12345
EPS = 1e-15
N_JOBS = min(len(SEEDS), os.cpu_count() or 1)


def log_loss(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-slide log-loss of the cirrhosis probability, clipped like sklearn's."""
    p = np.clip(proba, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def _folds(y: np.ndarray, seed: int):
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed)).split(y, y)


def _fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from topocyte.tl.fpca import _logistic

    model = _logistic(10, 5, "accuracy", 1000, int(seed)).fit(X_train, y_train)
    if list(model.classes_) != [0, 1]:
        raise RuntimeError(f"unexpected classes {model.classes_}")
    return model.predict(X_test), model.predict_proba(X_test)[:, 1]


def cv_outputs(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-slide held-out correctness and probability under the protocol folds."""
    correct = np.zeros(y.size, dtype=bool)
    proba = np.empty(y.size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="lbfgs failed to converge")
        for train, test in _folds(y, seed):
            predicted, proba[test] = _fit_predict(X[train], y[train], X[test], seed)
            correct[test] = predicted == y[test]
    return correct, proba


def residual_cv_outputs(
    density: np.ndarray, scores: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Classify scores after residualising them on density, fold by fold.

    The regression of each score on the density features is fitted on the
    outer training fold only, so no test-fold information enters the
    transformation.
    """
    from sklearn.linear_model import LinearRegression

    if density.shape[1] + 1 >= 0.5 * y.size:
        raise ValueError(
            f"{density.shape[1]} density features against {y.size} slides: "
            "the training-fold regression would (nearly) interpolate"
        )
    correct = np.zeros(y.size, dtype=bool)
    proba = np.empty(y.size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="lbfgs failed to converge")
        for train, test in _folds(y, seed):
            residuals = np.empty((y.size, scores.shape[1]))
            for column in range(scores.shape[1]):
                model = LinearRegression().fit(density[train], scores[train, column])
                residuals[:, column] = scores[:, column] - model.predict(density)
            predicted, proba[test] = _fit_predict(
                residuals[train], y[train], residuals[test], seed
            )
            correct[test] = predicted == y[test]
    return correct, proba


def chance_outputs(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The training fold's majority class and prevalence, per held-out slide."""
    correct = np.zeros(y.size, dtype=bool)
    proba = np.empty(y.size)
    for train, test in _folds(y, seed):
        prevalence = float(y[train].mean())
        correct[test] = y[test] == int(prevalence > 0.5)
        proba[test] = prevalence
    return correct, proba


def build_rich() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rich per-slide density features, one block of ten per slice."""
    density = np.load(RESULTS / "density.npz", allow_pickle=False)
    rho = density["rho"]
    slide_codes = density["slide_codes"]
    slide_categories = [str(label) for label in density["slide_categories"]]

    with np.load(RESULTS / "slice_0.npz", allow_pickle=False) as slice0:
        group_ids = [str(g) for g in slice0["group_ids"]]
    position = {label: i for i, label in enumerate(slide_categories)}
    reorder = np.array([position[label] for label in group_ids])
    if len(set(group_ids)) != len(group_ids):
        raise RuntimeError("group_ids are not unique")

    thresholds = np.empty(N_SLICES)
    kept_counts = np.empty((N_SLICES, len(group_ids)), dtype=np.int64)
    for k in range(N_SLICES):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            thresholds[k] = float(one["threshold"])
            kept_counts[k] = one["kept_per_slide"]

    n_slides = len(group_ids)
    rich = np.empty((N_SLICES, n_slides, 10))
    for i in range(n_slides):
        keep_rows = np.flatnonzero(slide_codes == reorder[i])
        logged = np.sort(np.log10(rho[keep_rows]))
        total = logged.size
        for k in range(N_SLICES):
            first = 0 if k == 0 else np.searchsorted(logged, np.log10(thresholds[k]), side="left")
            kept = logged[first:]
            if kept.size != kept_counts[k, i]:
                raise RuntimeError(
                    f"slide {i} slice {k}: kept {kept.size} cells, stored {kept_counts[k, i]}"
                )
            rich[k, i, :7] = np.quantile(kept, QUANTILES)
            rich[k, i, 7] = kept.mean()
            rich[k, i, 8] = kept.std(ddof=1)
            rich[k, i, 9] = kept.size / total
        if i % 25 == 0:
            print(f"  rich density {i + 1}/{n_slides} slides", flush=True)

    rich_all = np.hstack([rich[k] for k in range(N_SLICES)])
    return rich, rich_all, group_ids


def main() -> None:
    import sklearn
    import topocyte
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    started = time.time()
    rich, rich_all, group_ids = build_rich()
    with np.load(RESULTS / "slice_0.npz", allow_pickle=False) as slice0:
        y = np.asarray(slice0["y"], dtype=np.int8)
        cells_per_slide = np.asarray(slice0["kept_per_slide"], dtype=float)
    slices = []
    for k in range(N_SLICES):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            if [str(g) for g in one["group_ids"]] != group_ids:
                raise RuntimeError(f"slice {k}: group order differs from the density file")
            slices.append({key: one[key] for key in one.files})
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as combined:
        combined_scores = {name: combined[f"{name}_scores"] for name in MODELS}
        if not np.array_equal(combined["y"], y):
            raise RuntimeError("combined: y differs from the slices")

    log_cells = np.log10(cells_per_slide)
    density_k = [np.hstack([rich[k], log_cells[:, None]]) for k in range(N_SLICES)]
    density_all = np.hstack([rich_all, log_cells[:, None]])

    payload: dict[str, np.ndarray] = {
        "group_ids": np.array(group_ids),
        "y": y,
        "seeds": SEEDS,
        "rich_scores_all": rich_all,
        "log_cells": log_cells,
        "n_slices": np.int64(N_SLICES),
    }
    for k in range(N_SLICES):
        payload[f"rich_scores_k{k}"] = rich[k]

    chance = [chance_outputs(y, int(seed)) for seed in SEEDS]
    payload["chance_correct"] = np.array([c for c, _ in chance])
    payload["base_proba"] = np.array([p for _, p in chance])
    per_slide_ll: dict[str, np.ndarray] = {"base": log_loss(payload["base_proba"], y)}
    payload["base_logloss"] = per_slide_ll["base"].mean(axis=1)

    pool = Parallel(n_jobs=N_JOBS)

    def store(field: str, suffix: str, outputs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        correct = np.array([c for c, _ in outputs])
        proba = np.array([p for _, p in outputs])
        losses = log_loss(proba, y)
        payload[f"{field}_correct_{suffix}"] = correct
        payload[f"{field}_proba_{suffix}"] = proba
        payload[f"{field}_acc_{suffix}"] = correct.mean(axis=1)
        payload[f"{field}_logloss_{suffix}"] = losses.mean(axis=1)
        payload[f"{field}_auc_{suffix}"] = np.array([roc_auc_score(y, p) for p in proba])
        per_slide_ll[f"{field}_{suffix}"] = losses

    def run_arm(field: str, suffix: str, X: np.ndarray) -> None:
        store(field, suffix, pool(delayed(cv_outputs)(X, y, int(s)) for s in SEEDS))

    # --- density arm, slide size, nested ---------------------------------------------
    print("density and nested arms", flush=True)
    for k in range(N_SLICES):
        run_arm("A", str(k), density_k[k])
    run_arm("A", "all", density_all)
    run_arm("S", "all", log_cells[:, None])
    print(
        f"  density arm all ten {payload['A_acc_all'].mean():.3f}, "
        f"slide size alone {payload['S_acc_all'].mean():.3f}",
        flush=True,
    )
    for name in MODELS:
        for k in range(N_SLICES):
            scores = slices[k][f"{name}_scores"][:, :N_COMPONENTS]
            run_arm(f"B_{name}", str(k), np.hstack([density_k[k], scores]))
        run_arm(f"B_{name}", "all", np.hstack([density_all, combined_scores[name]]))
        print(f"  {name}: nested all ten {payload[f'B_{name}_acc_all'].mean():.3f}", flush=True)

    # --- residual, per slice only ------------------------------------------------------
    print("residual arms", flush=True)
    for name in MODELS:
        for k in range(N_SLICES):
            scores = slices[k][f"{name}_scores"][:, :N_COMPONENTS]
            outputs = pool(
                delayed(residual_cv_outputs)(density_k[k], scores, y, int(s)) for s in SEEDS
            )
            store(f"R_{name}", str(k), outputs)
        print(
            f"  {name}: residual per slice "
            + " ".join(f"{payload[f'R_{name}_acc_{k}'].mean():.3f}" for k in range(N_SLICES)),
            flush=True,
        )

    # --- PC breakdown ------------------------------------------------------------------
    print("PC breakdown", flush=True)
    for name in MODELS:
        for k in range(N_SLICES):
            scores = slices[k][f"{name}_scores"][:, :N_COMPONENTS]
            run_arm(f"P1_{name}", str(k), scores[:, :1])
            run_arm(f"P24_{name}", str(k), scores[:, 1:])
        scores = combined_scores[name]
        run_arm(f"P1_{name}", "all", scores[:, 0::N_COMPONENTS])
        run_arm(f"P24_{name}", "all", np.delete(scores, np.s_[0::N_COMPONENTS], axis=1))

    # --- descriptive: variance of each score explained by the density arm ---------------
    r2 = np.empty((N_SLICES, len(MODELS), N_COMPONENTS))
    for k in range(N_SLICES):
        for f, name in enumerate(MODELS):
            scores = slices[k][f"{name}_scores"][:, :N_COMPONENTS]
            for j in range(N_COMPONENTS):
                model = LinearRegression().fit(density_k[k], scores[:, j])
                residual = scores[:, j] - model.predict(density_k[k])
                total = scores[:, j] - scores[:, j].mean()
                r2[k, f, j] = 1.0 - residual @ residual / (total @ total)
    payload["pc_r2"] = r2

    # --- paired bootstrap intervals, one index set for every comparison ------------------
    boot_index = np.random.default_rng(BOOT_SEED).integers(0, y.size, (N_BOOT, y.size))

    def paired_ci(baseline: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Mean of other - baseline over slides, and its 95% bootstrap interval."""
        delta = other.mean(axis=0) - baseline.mean(axis=0)
        draws = delta[boot_index].mean(axis=1)
        return np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])

    chance_correct = payload["chance_correct"]
    for name in MODELS:
        for suffix in [str(k) for k in range(N_SLICES)] + ["all"]:
            density_correct = payload[f"A_correct_{suffix}"]
            density_ll = per_slide_ll[f"A_{suffix}"]
            # Accuracy differences: arm minus baseline.  Log-loss: baseline minus
            # arm, so that a positive value means the scores help in both.
            payload[f"ci_nested_{name}_{suffix}"] = paired_ci(
                density_correct, payload[f"B_{name}_correct_{suffix}"]
            )
            payload[f"ci_nested_ll_{name}_{suffix}"] = paired_ci(
                per_slide_ll[f"B_{name}_{suffix}"], density_ll
            )
            payload[f"ci_pc24_{name}_{suffix}"] = paired_ci(
                density_correct, payload[f"P24_{name}_correct_{suffix}"]
            )
            payload[f"ci_pc24_ll_{name}_{suffix}"] = paired_ci(
                per_slide_ll[f"P24_{name}_{suffix}"], density_ll
            )
        for k in range(N_SLICES):
            payload[f"ci_res_chance_{name}_{k}"] = paired_ci(
                chance_correct, payload[f"R_{name}_correct_{k}"]
            )
            payload[f"ci_res_ll_{name}_{k}"] = paired_ci(
                per_slide_ll[f"R_{name}_{k}"], per_slide_ll["base"]
            )

    np.savez(RESULTS / "density_adds.npz", **payload)
    meta = {
        "arms": {
            "A": "10 density features per slice (log10 rho quantiles "
            "0.05/0.10/0.25/0.50/0.75/0.90/0.95 among kept cells, mean, sd, kept "
            "fraction) + log10 cells per slide",
            "S": "log10 cells per slide alone",
            "B": "A + first 4 fPCA scores of the slice",
            "R": "first 4 fPCA scores residualised on A inside each outer training "
            "fold; per slice only (combined would be p >= n)",
            "P1": "first fPCA score only",
            "P24": "fPCA scores 2-4 only",
        },
        "chance": "training-fold majority class (accuracy) and prevalence (log-loss)",
        "models": list(MODELS),
        "seeds": len(SEEDS),
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "estimator": "LogisticRegressionCV(Cs=10, inner_cv=5, scoring=accuracy, max_iter=1000)",
        "cv": "StratifiedKFold(5, shuffle=True, random_state=seed)",
        "topocyte": str(topocyte.__version__),
        "sklearn": str(sklearn.__version__),
        "n_jobs": N_JOBS,
        "supersedes": "superseded/density_adds_2026-11-16.npz",
        "runtime_seconds": round(time.time() - started, 1),
    }
    (RESULTS / "density_adds.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/density_adds.npz in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
