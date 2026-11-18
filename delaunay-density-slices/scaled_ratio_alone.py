#!/usr/bin/env python
"""Post-hoc: section 8's ratio-alone arms refitted with standardized features.

Section 8's ratio-alone arms fit the protocol estimator, an L2-penalized
logistic regression whose C is chosen by inner cross-validation from
``logspace(-4, 4, 10)``, on the first four death/birth-ratio scores as they
come.  With one block of features a common spread only shifts the C the fit
needs, but the cosine scores vary by about 0.001 to 0.04 across slides, and
at that spread even the grid's largest C penalizes them heavily: a scratch
check found the cosine arm at slice 0 choosing the grid's top C in 43% of its
folds.  This script refits the ten ratio-alone arms (raw and cosine, slices 0
and 6-9) unscaled, to record each fold's C, and with a StandardScaler, fitted
on the outer training fold only, in front of the same estimator.  It was not
pre-registered; the report labels it post-hoc.

* ``P_Q_{m}``: the protocol ratio-alone arm refitted to record each fold's
  chosen C.  Gate: its held-out outputs equal ``ratio_slices.npz``'s
  ``Q_{m}`` arm for every seed.
* ``Z_Q_{m}``: the same four scores, standardized.

Intervals: paired bootstrap over slides with ``density_adds.py``'s shared
indices - standardized minus protocol arm in accuracy (``scaling``) and, in
log-loss, protocol minus standardized (``scaling_ll``, positive favours
standardizing); the standardized arm against chance (the training fold's
majority class) and, in log-loss, against the training fold's prevalence.

Reads only ``ratio_slices.npz``.  Run with ``PYTHONPATH=<topocyte checkout>`` and the
checkout's interpreter.  Writes ``results/scaled_ratio_alone.npz`` and
``.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import run_slices  # noqa: F401  imports topocyte from the PYTHONPATH checkout and checks that it did
from density_adds import BOOT_SEED, N_BOOT, N_COMPONENTS, N_JOBS, SEEDS, log_loss
from scaled_nested import fit_folds

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SLICES = (0, 6, 7, 8, 9)
FAMILIES = ("none", "cosine")
CS_GRID = np.logspace(-4, 4, 10)


def main() -> None:
    import sklearn
    import topocyte
    from sklearn.metrics import roc_auc_score

    started = time.time()
    with np.load(RESULTS / "ratio_slices.npz", allow_pickle=False) as ratio:
        if tuple(int(k) for k in ratio["slices"]) != SLICES:
            raise RuntimeError(f"ratio_slices: slices {ratio['slices']} are not {SLICES}")
        y = ratio["y"]
        group_ids = ratio["group_ids"]
        chance_correct = ratio["chance_correct"]
        base_proba = ratio["base_proba"]
        blocks = {
            (m, k): ratio[f"ratio_{m}_scores_{k}"][:, :N_COMPONENTS] for m in FAMILIES for k in SLICES
        }
        saved = {
            (m, k, field): ratio[f"Q_{m}_{field}_{k}"]
            for m in FAMILIES
            for k in SLICES
            for field in ("correct", "proba")
        }

    payload: dict[str, np.ndarray] = {
        "y": y,
        "group_ids": group_ids,
        "slices": np.array(SLICES),
        "seeds": SEEDS,
        "cs_grid": CS_GRID,
    }
    per_slide_ll: dict[str, np.ndarray] = {"base": log_loss(base_proba, y)}
    pool = Parallel(n_jobs=N_JOBS)

    def run_arm(field: str, k: int, X: np.ndarray, scaled: bool) -> None:
        outputs = pool(delayed(fit_folds)(X, y, int(s), 0, scaled) for s in SEEDS)
        correct = np.array([o[0] for o in outputs])
        proba = np.array([o[1] for o in outputs])
        losses = log_loss(proba, y)
        payload[f"{field}_correct_{k}"] = correct
        payload[f"{field}_proba_{k}"] = proba
        payload[f"{field}_acc_{k}"] = correct.mean(axis=1)
        payload[f"{field}_logloss_{k}"] = losses.mean(axis=1)
        payload[f"{field}_auc_{k}"] = np.array([roc_auc_score(y, p) for p in proba])
        payload[f"{field}_C_{k}"] = np.array([o[2] for o in outputs])
        per_slide_ll[f"{field}_{k}"] = losses

    for k in SLICES:
        for m in FAMILIES:
            X = blocks[(m, k)]
            payload[f"score_sd_{m}_{k}"] = X.std(axis=0, ddof=1)
            run_arm(f"P_Q_{m}", k, X, False)
            # Gate: the unscaled refit is the ratio-alone arm section 8 reports.
            if not (
                np.array_equal(payload[f"P_Q_{m}_correct_{k}"], saved[(m, k, "correct")])
                and np.allclose(
                    payload[f"P_Q_{m}_proba_{k}"], saved[(m, k, "proba")], rtol=0.0, atol=1e-12
                )
            ):
                raise RuntimeError(f"slice {k} {m}: the refit is not the saved ratio-alone arm")
            run_arm(f"Z_Q_{m}", k, X, True)
        print(
            f"slice {k}: "
            + "; ".join(
                f"{m}: {payload[f'P_Q_{m}_acc_{k}'].mean():.3f} -> "
                f"{payload[f'Z_Q_{m}_acc_{k}'].mean():.3f} (top C in "
                f"{np.mean(np.isclose(payload[f'P_Q_{m}_C_{k}'], CS_GRID[-1])):.2f} -> "
                f"{np.mean(np.isclose(payload[f'Z_Q_{m}_C_{k}'], CS_GRID[-1])):.2f} of folds)"
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
        for m in FAMILIES:
            payload[f"ci_scaling_{m}_{k}"] = paired_ci(
                payload[f"P_Q_{m}_correct_{k}"], payload[f"Z_Q_{m}_correct_{k}"]
            )
            payload[f"ci_scaling_ll_{m}_{k}"] = paired_ci(
                per_slide_ll[f"Z_Q_{m}_{k}"], per_slide_ll[f"P_Q_{m}_{k}"]
            )
            payload[f"ci_alone_chance_{m}_{k}"] = paired_ci(
                chance_correct, payload[f"Z_Q_{m}_correct_{k}"]
            )
            payload[f"ci_alone_ll_{m}_{k}"] = paired_ci(
                per_slide_ll[f"Z_Q_{m}_{k}"], per_slide_ll["base"]
            )

    out = RESULTS / "scaled_ratio_alone.npz"
    np.savez(out, **payload)
    meta = {
        "status": "post-hoc sensitivity analysis, not pre-registered",
        "slices": list(SLICES),
        "arms": {
            "P_Q": "section 8's ratio-alone arm refitted (first 4 ratio scores, unscaled) to "
            "record each fold's C; gate: equals ratio_slices.npz's Q arm for every seed",
            "Z_Q": "the same scores with a StandardScaler fitted on the outer training fold",
        },
        "indicator": "share of fitted folds whose chosen C is the grid's top (1e4)",
        "intervals": "paired bootstrap over slides of seed-averaged per-slide values, "
        "shared indices default_rng(12345).integers(0, 123, (4000, 123)); scaling: Z_Q - "
        "P_Q (accuracy), scaling_ll: P_Q - Z_Q (log-loss); alone_chance: Z_Q - chance; "
        "alone_ll: prevalence - Z_Q (log-loss)",
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
