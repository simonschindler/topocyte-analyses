#!/usr/bin/env python
"""Classify slides from the fPCA scores of all ten density slices at once.

For each fPCA family (H0/H1, raw/cosine), concatenate the first four scores
that slice ``k``'s model gives every slide, over the ten slices: 40 features
per slide.  Fit the same logistic regression and the same outer folds as
:func:`topocyte.tl.fpca_classify` did per slice, for seeds 0-19.

The density-only control is the analogous concatenation: the ten kept
fractions of every slide, one per slice.

Everything is read from ``results/slice_<k>.npz``; no persistent homology or
fPCA is recomputed.  Writes ``results/combined.npz`` and ``results/combined.json``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = np.arange(20)
MODELS = ("h0_none", "h1_none", "h0_cosine", "h1_cosine")
N_SLICES = 10
N_COMPONENTS = 4

warnings.filterwarnings("ignore", message="lbfgs failed to converge")


def load_slice(k: int) -> dict[str, np.ndarray]:
    with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def classify(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Held-out accuracy and confusion matrix per seed, mirroring fpca_classify."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from topocyte.tl.fpca import _logistic

    used = y >= 0
    used_X, used_y = X[used], y[used]
    accuracies = np.empty(len(SEEDS))
    confusions = np.empty((len(SEEDS), 2, 2), dtype=np.int64)
    for seed in SEEDS:
        held_out = cross_val_predict(
            _logistic(10, 5, "accuracy", 1000, int(seed)),
            used_X,
            used_y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed)),
        )
        accuracies[seed] = float(np.mean(held_out == used_y))
        confusions[seed] = confusion_matrix(used_y, held_out, labels=[0, 1])
    return accuracies, confusions


def main() -> None:
    started = time.time()
    slices = [load_slice(k) for k in range(N_SLICES)]
    group_ids = np.array([str(g) for g in slices[0]["group_ids"]])
    y = np.asarray(slices[0]["y"], dtype=np.int8)
    for k, one in enumerate(slices):
        if not np.array_equal(one["group_ids"], group_ids):
            raise RuntimeError(f"slice {k}: group order differs from slice 0")
        if not np.array_equal(one["y"], y):
            raise RuntimeError(f"slice {k}: y differs from slice 0")

    payload: dict[str, np.ndarray] = {
        "group_ids": group_ids,
        "y": y,
        "seeds": SEEDS,
        "n_slices": np.int64(N_SLICES),
        "n_components": np.int64(N_COMPONENTS),
    }
    for name in MODELS:
        X = np.hstack([one[f"{name}_scores"][:, :N_COMPONENTS] for one in slices])
        if X.shape != (group_ids.size, N_SLICES * N_COMPONENTS):
            raise RuntimeError(f"{name}: feature matrix shape {X.shape}")
        accuracies, confusions = classify(X, y)
        payload[f"{name}_scores"] = X
        payload[f"{name}_accuracy"] = accuracies
        payload[f"{name}_confusion"] = confusions
        print(
            f"{name:>10}: seed 0 {accuracies[0]:.3f}, "
            f"mean {accuracies.mean():.3f} ± {accuracies.std(ddof=1):.3f}, "
            f"features {X.shape[1]}",
            flush=True,
        )

    control = np.column_stack([one["kept_fraction"] for one in slices])
    if control.shape != (group_ids.size, N_SLICES):
        raise RuntimeError(f"control: feature matrix shape {control.shape}")
    control_accuracy, control_confusion = classify(control, y)
    payload["control_scores"] = control
    payload["control_accuracy"] = control_accuracy
    payload["control_confusion"] = control_confusion
    print(
        f"{'control':>10}: seed 0 {control_accuracy[0]:.3f}, "
        f"mean {control_accuracy.mean():.3f} ± {control_accuracy.std(ddof=1):.3f}, "
        f"features {control.shape[1]}",
        flush=True,
    )

    np.savez(RESULTS / "combined.npz", **payload)
    meta = {
        "models": list(MODELS),
        "features_per_model": N_SLICES * N_COMPONENTS,
        "control_features": N_SLICES,
        "seeds": len(SEEDS),
        "runtime_seconds": round(time.time() - started, 1),
        "estimator": "LogisticRegressionCV(Cs=10, inner_cv=5, scoring=accuracy, max_iter=1000)",
        "cv": "StratifiedKFold(5, shuffle=True, random_state=seed)",
    }
    (RESULTS / "combined.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/combined.npz in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
