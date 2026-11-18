#!/usr/bin/env python
"""Post-hoc: the nested arms refitted with every feature standardized.

Sections 7 and 8 test whether four fPCA scores add to the density arm by
fitting the protocol estimator, an L2-penalized logistic regression whose C
is chosen by inner cross-validation, on the density arm plus the scores, with
every feature on its own scale.  The L2 penalty is not scale-free.  The
cosine-normalized ratio scores vary by about 0.001 to 0.01 across slides and
the density features by about 0.05 to 0.2, so a fit can leave the scores out
at little cost; the unnormalized scores vary by 10^2 to 10^6, so a fit at the
smallest C keeps them and drops the density arm.  This script measures that
and repeats the nested test with a StandardScaler, fitted on the outer
training fold only, in front of the same estimator.  It was not
pre-registered: section 8 reports it as a sensitivity analysis next to the
pre-registered result, which stands.

Slices 0 and 6-9, the protocol folds, 20 seeds.  ``source`` is ``ratio``
(section 8's death/birth-ratio scores) or ``betti`` (section 7's H1
Betti-curve scores), ``m`` the normalization (``none`` or ``cosine``):

* ``P_{source}_{m}``: the protocol nested arm (density arm plus four scores),
  refitted to record each fold's chosen C and the score block's share of the
  weight.  Gate: its held-out outputs equal the saved arm's for every seed.
* ``Z_density``: the density arm, standardized.
* ``Z_{source}_{m}``: the nested arm, standardized.

The weight share of a fitted fold is ``sum_j |coef_j| SD_j`` over the four
score columns divided by the same sum over all columns, with ``SD_j`` the
training-fold standard deviation of column ``j`` as the classifier sees it
(raw for ``P``, standardized for ``Z``).

Intervals: paired bootstrap over slides with ``density_adds.py``'s shared
indices, standardized nested minus standardized density arm, in accuracy and
in log-loss (density minus nested, so positive favours the scores), and the
standardized minus the protocol density arm.

Reads only saved results (``density_adds.npz``, ``ratio_slices.npz``,
``slice_<k>.npz``).  Run with ``PYTHONPATH=<topocyte checkout>`` and the checkout's
interpreter.  Writes ``results/scaled_nested.npz`` and ``.json``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import run_slices  # noqa: F401  imports topocyte from the PYTHONPATH checkout and checks that it did
from density_adds import BOOT_SEED, N_BOOT, N_COMPONENTS, N_JOBS, SEEDS, _folds, log_loss

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SLICES = (0, 6, 7, 8, 9)
FAMILIES = ("none", "cosine")
SOURCES = ("ratio", "betti")


def fit_folds(
    X: np.ndarray, y: np.ndarray, seed: int, n_scores: int, scaled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Held-out correctness and probability, and each fold's C and score weight share."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    correct = np.zeros(y.size, dtype=bool)
    proba = np.empty(y.size)
    chosen = np.full(5, np.nan)
    share = np.full(5, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="lbfgs failed to converge")
        for fold, (train, test) in enumerate(_folds(y, seed)):
            classifier = _logistic(10, 5, "accuracy", 1000, int(seed))
            model = make_pipeline(StandardScaler(), classifier) if scaled else classifier
            model.fit(X[train], y[train])
            if list(classifier.classes_) != [0, 1]:
                raise RuntimeError(f"unexpected classes {classifier.classes_}")
            proba[test] = model.predict_proba(X[test])[:, 1]
            correct[test] = model.predict(X[test]) == y[test]
            seen = model[0].transform(X[train]) if scaled else X[train]
            weight = np.abs(classifier.coef_.ravel()) * seen.std(axis=0)
            if n_scores:
                share[fold] = weight[-n_scores:].sum() / weight.sum()
            chosen[fold] = float(np.ravel(classifier.C_)[0])
    return correct, proba, chosen, share


def main() -> None:
    import sklearn
    import topocyte
    from sklearn.metrics import roc_auc_score

    started = time.time()
    with np.load(RESULTS / "density_adds.npz", allow_pickle=False) as adds:
        y = adds["y"]
        group_ids = adds["group_ids"]
        density_k = {
            k: np.hstack([adds[f"rich_scores_k{k}"], adds["log_cells"][:, None]]) for k in SLICES
        }
        protocol_density = {k: adds[f"A_correct_{k}"] for k in SLICES}
        saved = {
            ("betti", m, k, field): adds[f"B_h1_{m}_{field}_{k}"]
            for m in FAMILIES
            for k in SLICES
            for field in ("correct", "proba")
        }
    blocks: dict[tuple[str, str, int], np.ndarray] = {}
    with np.load(RESULTS / "ratio_slices.npz", allow_pickle=False) as ratio:
        if not (np.array_equal(ratio["y"], y) and np.array_equal(ratio["group_ids"], group_ids)):
            raise RuntimeError("ratio_slices: slides differ from density_adds")
        for m in FAMILIES:
            for k in SLICES:
                blocks[("ratio", m, k)] = ratio[f"ratio_{m}_scores_{k}"][:, :N_COMPONENTS]
                for field in ("correct", "proba"):
                    saved[("ratio", m, k, field)] = ratio[f"N_{m}_{field}_{k}"]
    for k in SLICES:
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            if not np.array_equal(one["group_ids"], group_ids):
                raise RuntimeError(f"slice {k}: slides differ from density_adds")
            for m in FAMILIES:
                blocks[("betti", m, k)] = one[f"h1_{m}_scores"][:, :N_COMPONENTS]

    payload: dict[str, np.ndarray] = {
        "y": y,
        "group_ids": group_ids,
        "slices": np.array(SLICES),
        "seeds": SEEDS,
    }
    per_slide_ll: dict[str, np.ndarray] = {}
    pool = Parallel(n_jobs=N_JOBS)

    def run_arm(field: str, suffix: str, X: np.ndarray, n_scores: int, scaled: bool) -> None:
        outputs = pool(delayed(fit_folds)(X, y, int(s), n_scores, scaled) for s in SEEDS)
        correct = np.array([o[0] for o in outputs])
        proba = np.array([o[1] for o in outputs])
        losses = log_loss(proba, y)
        payload[f"{field}_correct_{suffix}"] = correct
        payload[f"{field}_proba_{suffix}"] = proba
        payload[f"{field}_acc_{suffix}"] = correct.mean(axis=1)
        payload[f"{field}_logloss_{suffix}"] = losses.mean(axis=1)
        payload[f"{field}_auc_{suffix}"] = np.array([roc_auc_score(y, p) for p in proba])
        if n_scores:
            payload[f"{field}_C_{suffix}"] = np.array([o[2] for o in outputs])
            payload[f"{field}_share_{suffix}"] = np.array([o[3] for o in outputs])
        per_slide_ll[f"{field}_{suffix}"] = losses

    for k in SLICES:
        payload[f"density_sd_{k}"] = density_k[k].std(axis=0, ddof=1)
        run_arm("Z_density", str(k), density_k[k], 0, True)
        for source in SOURCES:
            for m in FAMILIES:
                block = blocks[(source, m, k)]
                payload[f"score_sd_{source}_{m}_{k}"] = block.std(axis=0, ddof=1)
                X = np.hstack([density_k[k], block])
                run_arm(f"P_{source}_{m}", str(k), X, N_COMPONENTS, False)
                # Gate: the refit is the arm sections 7 and 8 report, seed by seed.
                if not (
                    np.array_equal(payload[f"P_{source}_{m}_correct_{k}"], saved[(source, m, k, "correct")])
                    and np.allclose(
                        payload[f"P_{source}_{m}_proba_{k}"], saved[(source, m, k, "proba")], rtol=0.0, atol=1e-12
                    )
                ):
                    raise RuntimeError(f"slice {k} {source} {m}: the refit is not the saved nested arm")
                run_arm(f"Z_{source}_{m}", str(k), X, N_COMPONENTS, True)
        print(
            f"slice {k}: standardized density {payload[f'Z_density_acc_{k}'].mean():.3f}; "
            + "; ".join(
                f"{source} {m}: {payload[f'Z_{source}_{m}_acc_{k}'].mean():.3f} "
                f"(score share {np.median(payload[f'P_{source}_{m}_share_{k}']):.3f} -> "
                f"{np.median(payload[f'Z_{source}_{m}_share_{k}']):.3f})"
                for source in SOURCES
                for m in FAMILIES
            ),
            flush=True,
        )

    boot_index = np.random.default_rng(BOOT_SEED).integers(0, y.size, (N_BOOT, y.size))

    def paired_ci(baseline: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Mean of other - baseline over slides, and its 95% bootstrap interval."""
        delta = other.mean(axis=0) - baseline.mean(axis=0)
        draws = delta[boot_index].mean(axis=1)
        return np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])

    for k in SLICES:
        density_correct = payload[f"Z_density_correct_{k}"]
        for source in SOURCES:
            for m in FAMILIES:
                payload[f"ci_nested_{source}_{m}_{k}"] = paired_ci(
                    density_correct, payload[f"Z_{source}_{m}_correct_{k}"]
                )
                payload[f"ci_nested_ll_{source}_{m}_{k}"] = paired_ci(
                    per_slide_ll[f"Z_{source}_{m}_{k}"], per_slide_ll[f"Z_density_{k}"]
                )
        payload[f"ci_scaling_density_{k}"] = paired_ci(protocol_density[k], density_correct)

    out = RESULTS / "scaled_nested.npz"
    np.savez(out, **payload)
    meta = {
        "status": "post-hoc sensitivity analysis, not pre-registered",
        "slices": list(SLICES),
        "sources": {
            "ratio": "first 4 death/birth-ratio fPCA scores (ratio_slices.npz)",
            "betti": "first 4 H1 Betti-curve fPCA scores (slice_<k>.npz, as section 7)",
        },
        "arms": {
            "P": "protocol nested arm refitted (density arm + 4 scores, unscaled); "
            "gate: equals the saved arm for every seed",
            "Z_density": "density arm with a StandardScaler fitted on the outer training fold",
            "Z": "nested arm with a StandardScaler fitted on the outer training fold",
        },
        "weight_share": "sum over the 4 score columns of |coef_j| * SD_j over the sum over "
        "all columns, SD_j the training-fold SD (ddof=0) of the column the classifier sees",
        "intervals": "paired bootstrap over slides of seed-averaged per-slide values, "
        "shared indices default_rng(12345).integers(0, 123, (4000, 123)); nested: Z - "
        "Z_density (accuracy), Z_density - Z (log-loss); scaling_density: Z_density - A",
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
