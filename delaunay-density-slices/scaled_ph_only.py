#!/usr/bin/env python
"""Post-hoc: the PH-only arms behind the verdict refitted with standardized features.

Sections 5 and 6 classify slides from fPCA scores alone with
``topocyte.tl.fpca_classify``, which feeds the scores to an L2-penalized
logistic regression as they are, without standardisation (its docstring says
so), choosing C by inner cross-validation from ``logspace(-4, 4, 10)``.  The
post-hoc checks of sections 7 and 8 found arms held back where the features'
spread put the C they need off that grid.  This script refits every arm the
verdict rests on - the four families' four-score arms at slices 0-9, the
per-slice density-only controls (the kept fraction), the four combined
40-score models and the combined control (the ten kept fractions) - unscaled,
to record each fold's C and the held-out outputs, and with a StandardScaler,
fitted on the outer training fold only, in front of the same estimator.  It
was not pre-registered; the report labels it post-hoc.

``arm`` is one of the four families or ``control``; ``suffix`` a slice 0-9 or
``all`` (the combined models):

* ``P_{arm}``: the protocol arm refitted.  Gate: its per-seed accuracy and
  confusion matrix equal those that ``fpca_classify`` (``slice_<k>.npz``) and
  ``combined_classify.py`` (``combined.npz``) stored.
* ``Z_{arm}``: the same features, standardized.

Intervals: paired bootstrap over slides with ``density_adds.py``'s shared
indices - standardized minus protocol arm in accuracy (``scaling``) and, in
log-loss, protocol minus standardized (``scaling_ll``).

Reads only saved results.  Run with ``PYTHONPATH=<topocyte checkout>`` and the
checkout's interpreter.  Writes ``results/scaled_ph_only.npz`` and ``.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import run_slices  # noqa: F401  imports topocyte from the PYTHONPATH checkout and checks that it did
from density_adds import BOOT_SEED, MODELS, N_BOOT, N_COMPONENTS, N_JOBS, N_SLICES, SEEDS, log_loss
from scaled_nested import fit_folds

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUFFIXES = tuple(str(k) for k in range(N_SLICES)) + ("all",)
ARMS = MODELS + ("control",)
CS_GRID = np.logspace(-4, 4, 10)


def main() -> None:
    import sklearn
    import topocyte
    from sklearn.metrics import confusion_matrix, roc_auc_score

    started = time.time()
    features: dict[tuple[str, str], np.ndarray] = {}
    saved: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    y = None
    group_ids = None
    for k in range(N_SLICES):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            if y is None:
                y = np.asarray(one["y"], dtype=np.int8)
                group_ids = one["group_ids"]
            elif not (np.array_equal(one["y"], y) and np.array_equal(one["group_ids"], group_ids)):
                raise RuntimeError(f"slice {k}: slides differ from slice 0")
            for name in MODELS:
                features[(name, str(k))] = one[f"{name}_scores"][:, :N_COMPONENTS]
                saved[(name, str(k))] = (one[f"{name}_accuracy"], one[f"{name}_confusion"])
            features[("control", str(k))] = (one["kept_per_slide"] / one["slide_total"])[:, None]
            saved[("control", str(k))] = (one["control_accuracy"], one["control_confusion"])
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as combined:
        if not (np.array_equal(combined["y"], y) and np.array_equal(combined["group_ids"], group_ids)):
            raise RuntimeError("combined: slides differ from slice 0")
        for name in MODELS:
            features[(name, "all")] = combined[f"{name}_scores"]
            saved[(name, "all")] = (combined[f"{name}_accuracy"], combined[f"{name}_confusion"])
        features[("control", "all")] = combined["control_scores"]
        saved[("control", "all")] = (combined["control_accuracy"], combined["control_confusion"])
    if np.any(y < 0):
        raise RuntimeError("every slide is expected to carry a label")

    payload: dict[str, np.ndarray] = {
        "y": y,
        "group_ids": group_ids,
        "suffixes": np.array(SUFFIXES),
        "arms": np.array(ARMS),
        "seeds": SEEDS,
        "cs_grid": CS_GRID,
    }
    per_slide_ll: dict[str, np.ndarray] = {}
    pool = Parallel(n_jobs=N_JOBS)

    def run_arm(field: str, suffix: str, X: np.ndarray, scaled: bool) -> None:
        outputs = pool(delayed(fit_folds)(X, y, int(s), 0, scaled) for s in SEEDS)
        correct = np.array([o[0] for o in outputs])
        proba = np.array([o[1] for o in outputs])
        losses = log_loss(proba, y)
        payload[f"{field}_correct_{suffix}"] = correct
        payload[f"{field}_proba_{suffix}"] = proba
        payload[f"{field}_acc_{suffix}"] = correct.mean(axis=1)
        payload[f"{field}_logloss_{suffix}"] = losses.mean(axis=1)
        payload[f"{field}_auc_{suffix}"] = np.array([roc_auc_score(y, p) for p in proba])
        payload[f"{field}_C_{suffix}"] = np.array([o[2] for o in outputs])
        per_slide_ll[f"{field}_{suffix}"] = losses

    for s in SUFFIXES:
        clock = time.time()
        for arm in ARMS:
            X = features[(arm, s)]
            payload[f"feature_sd_{arm}_{s}"] = X.std(axis=0, ddof=1)
            run_arm(f"P_{arm}", s, X, False)
            # Gate: the refit is the arm the verdict reports, seed by seed.
            accuracy, confusion = saved[(arm, s)]
            correct = payload[f"P_{arm}_correct_{s}"]
            predicted = np.where(correct, y, 1 - y)
            refit_confusion = np.array([confusion_matrix(y, p, labels=[0, 1]) for p in predicted])
            if not (
                np.array_equal(payload[f"P_{arm}_acc_{s}"], accuracy)
                and np.array_equal(refit_confusion, confusion)
            ):
                raise RuntimeError(f"{arm} {s}: the refit is not the saved arm")
            run_arm(f"Z_{arm}", s, X, True)
        print(
            f"{s}: "
            + "; ".join(
                f"{arm} {payload[f'P_{arm}_acc_{s}'].mean():.3f} -> "
                f"{payload[f'Z_{arm}_acc_{s}'].mean():.3f}"
                for arm in ARMS
            )
            + f" [{time.time() - clock:.1f} s]",
            flush=True,
        )

    boot_index = np.random.default_rng(BOOT_SEED).integers(0, y.size, (N_BOOT, y.size))

    def paired_ci(baseline: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Mean of other - baseline over slides, and its 95% bootstrap interval."""
        delta = other.mean(axis=0) - baseline.mean(axis=0)
        draws = delta[boot_index].mean(axis=1)
        return np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])

    for s in SUFFIXES:
        for arm in ARMS:
            payload[f"ci_scaling_{arm}_{s}"] = paired_ci(
                payload[f"P_{arm}_correct_{s}"], payload[f"Z_{arm}_correct_{s}"]
            )
            payload[f"ci_scaling_ll_{arm}_{s}"] = paired_ci(
                per_slide_ll[f"Z_{arm}_{s}"], per_slide_ll[f"P_{arm}_{s}"]
            )

    out = RESULTS / "scaled_ph_only.npz"
    np.savez(out, **payload)
    meta = {
        "status": "post-hoc sensitivity analysis, not pre-registered",
        "suffixes": list(SUFFIXES),
        "arms": {
            "P": "the protocol arm refitted (features as they come); gate: per-seed "
            "accuracy and confusion matrix equal slice_<k>.npz's / combined.npz's",
            "Z": "the same features with a StandardScaler fitted on the outer training fold",
            "families": "first 4 fPCA scores of the slice (0-9) or the 40 scores of all "
            "ten slices (all)",
            "control": "kept fraction of the slice (0-9) or the ten kept fractions (all)",
        },
        "intervals": "paired bootstrap over slides of seed-averaged per-slide values, "
        "shared indices default_rng(12345).integers(0, 123, (4000, 123)); scaling: Z - P "
        "(accuracy), P - Z (log-loss)",
        "estimator": "LogisticRegressionCV(Cs=10, inner_cv=5, scoring=accuracy, max_iter=1000)",
        "cv": "StratifiedKFold(5, shuffle=True, random_state=seed)",
        "seeds": len(SEEDS),
        "topocyte": str(topocyte.__version__),
        "sklearn": str(sklearn.__version__),
        "n_jobs": N_JOBS,
        "runtime_seconds": round(time.time() - started, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
