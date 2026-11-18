#!/usr/bin/env python
"""Post-hoc: section 7's residual and PC arms refitted with standardized features.

These arms hold one block of features, so a common spread only shifts the C
the protocol estimator needs - as long as that C lies on its grid,
``logspace(-4, 4, 10)``.  Section 8's ratio-alone check (``scaled_ratio_alone.py``)
found blocks with a small spread choosing the grid's top C.  This script
refits the arms of section 7 that the report states: the 40 per-slice
residual arms (the slice's four scores residualised on its density arm inside
every outer training fold) and the eight combined PC arms (PC1 of every
slice, and PCs 2-4 of every slice, for the four families).  Each is refitted
unscaled, to record each fold's C, and with a StandardScaler fitted on the
outer training fold only - for the residual arms on the training fold's
residuals, after the same residualisation.  It was not pre-registered; the
report labels it post-hoc.

* ``P_R_{model}``/``Z_R_{model}`` (suffix: slice 0-9): residual arm, protocol
  refit and standardized.  Gate: the protocol refit equals
  ``density_adds.npz``'s ``R_{model}`` arm for every seed.
* ``P_PC1_{model}``/``Z_PC1_{model}`` and ``P_PC24_{model}``/``Z_PC24_{model}``
  (suffix ``all``): the combined PC arms.  Gate: the protocol refits equal
  ``P1_{model}``/``P24_{model}``.

Intervals: paired bootstrap over slides with ``density_adds.py``'s shared
indices - standardized minus protocol arm (``scaling``, accuracy; in
log-loss protocol minus standardized); the standardized residual arm against
chance and, in log-loss, against the training fold's prevalence (``res`` and
``res_ll``, as section 7); the standardized PC arms against the standardized
combined density arm of ``scaled_section7.npz`` (``pc1``, ``pc24`` and their
log-loss versions).

Reads only saved results.  Run with ``PYTHONPATH=<topocyte checkout>`` and the
checkout's interpreter.  Writes ``results/scaled_single_block.npz`` and
``.json``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import run_slices  # noqa: F401  imports topocyte from the PYTHONPATH checkout and checks that it did
from density_adds import (
    BOOT_SEED,
    MODELS,
    N_BOOT,
    N_COMPONENTS,
    N_JOBS,
    N_SLICES,
    SEEDS,
    _folds,
    log_loss,
)
from scaled_nested import fit_folds

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CS_GRID = np.logspace(-4, 4, 10)


def fit_residual_folds(
    density: np.ndarray, scores: np.ndarray, y: np.ndarray, seed: int, scaled: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``density_adds.residual_cv_outputs``, optionally standardized, recording each fold's C."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from topocyte.tl.fpca import _logistic

    correct = np.zeros(y.size, dtype=bool)
    proba = np.empty(y.size)
    chosen = np.full(5, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="lbfgs failed to converge")
        for fold, (train, test) in enumerate(_folds(y, seed)):
            residuals = np.empty((y.size, scores.shape[1]))
            for column in range(scores.shape[1]):
                regression = LinearRegression().fit(density[train], scores[train, column])
                residuals[:, column] = scores[:, column] - regression.predict(density)
            classifier = _logistic(10, 5, "accuracy", 1000, int(seed))
            model = make_pipeline(StandardScaler(), classifier) if scaled else classifier
            model.fit(residuals[train], y[train])
            if list(classifier.classes_) != [0, 1]:
                raise RuntimeError(f"unexpected classes {classifier.classes_}")
            proba[test] = model.predict_proba(residuals[test])[:, 1]
            correct[test] = model.predict(residuals[test]) == y[test]
            chosen[fold] = float(np.ravel(classifier.C_)[0])
    return correct, proba, chosen


def main() -> None:
    import sklearn
    import topocyte
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    started = time.time()
    with np.load(RESULTS / "density_adds.npz", allow_pickle=False) as adds:
        y = adds["y"]
        group_ids = adds["group_ids"]
        log_cells = adds["log_cells"][:, None]
        density = {k: np.hstack([adds[f"rich_scores_k{k}"], log_cells]) for k in range(N_SLICES)}
        chance_correct = adds["chance_correct"]
        base_proba = adds["base_proba"]
        saved = {key: adds[key] for key in adds.files if key.startswith(("R_", "P1_", "P24_"))}
    scores = {}
    for k in range(N_SLICES):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            if not np.array_equal(one["group_ids"], group_ids):
                raise RuntimeError(f"slice {k}: slides differ from density_adds")
            for name in MODELS:
                scores[(name, k)] = one[f"{name}_scores"][:, :N_COMPONENTS]
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as combined:
        if not np.array_equal(combined["y"], y):
            raise RuntimeError("combined: y differs from density_adds")
        pc_blocks = {}
        for name in MODELS:
            every = combined[f"{name}_scores"]
            pc_blocks[("PC1", name)] = every[:, 0::N_COMPONENTS]
            pc_blocks[("PC24", name)] = np.delete(every, np.s_[0::N_COMPONENTS], axis=1)
    with np.load(RESULTS / "scaled_section7.npz", allow_pickle=False) as refit7:
        if not (np.array_equal(refit7["y"], y) and np.array_equal(refit7["group_ids"], group_ids)):
            raise RuntimeError("scaled_section7: slides differ from density_adds")
        z_density_all = {f: refit7[f"Z_density_{f}_all"] for f in ("correct", "proba")}

    payload: dict[str, np.ndarray] = {
        "y": y,
        "group_ids": group_ids,
        "models": np.array(MODELS),
        "seeds": SEEDS,
        "cs_grid": CS_GRID,
    }
    per_slide_ll: dict[str, np.ndarray] = {
        "base": log_loss(base_proba, y),
        "Z_density_all": log_loss(z_density_all["proba"], y),
    }
    pool = Parallel(n_jobs=N_JOBS)

    def store(field: str, suffix: str, outputs: list) -> None:
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

    def gate(field: str, suffix: str, saved_field: str) -> None:
        if not (
            np.array_equal(payload[f"{field}_correct_{suffix}"], saved[f"{saved_field}_correct_{suffix}"])
            and np.allclose(
                payload[f"{field}_proba_{suffix}"],
                saved[f"{saved_field}_proba_{suffix}"],
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError(f"{field} {suffix}: the refit is not the saved {saved_field} arm")

    def top_share(field: str, suffix: str) -> float:
        return float(np.mean(np.isclose(payload[f"{field}_C_{suffix}"], CS_GRID[-1])))

    # Residual arms, per slice.
    for k in range(N_SLICES):
        clock = time.time()
        for name in MODELS:
            block = scores[(name, k)]
            residual = np.empty_like(block)
            for column in range(block.shape[1]):
                regression = LinearRegression().fit(density[k], block[:, column])
                residual[:, column] = block[:, column] - regression.predict(density[k])
            payload[f"res_sd_{name}_{k}"] = residual.std(axis=0, ddof=1)  # in-sample, descriptive
            for prefix, scaled in (("P", False), ("Z", True)):
                outputs = pool(
                    delayed(fit_residual_folds)(density[k], block, y, int(s), scaled) for s in SEEDS
                )
                store(f"{prefix}_R_{name}", str(k), outputs)
            gate(f"P_R_{name}", str(k), f"R_{name}")
        print(
            f"residual slice {k}: "
            + "; ".join(
                f"{name} {payload[f'P_R_{name}_acc_{k}'].mean():.3f} -> "
                f"{payload[f'Z_R_{name}_acc_{k}'].mean():.3f} "
                f"(top C {top_share(f'P_R_{name}', str(k)):.2f})"
                for name in MODELS
            )
            + f" [{time.time() - clock:.1f} s]",
            flush=True,
        )

    # Combined PC arms.
    for (part, name), block in pc_blocks.items():
        payload[f"score_sd_{part}_{name}_all"] = block.std(axis=0, ddof=1)
        for prefix, scaled in (("P", False), ("Z", True)):
            outputs = pool(delayed(fit_folds)(block, y, int(s), 0, scaled) for s in SEEDS)
            store(f"{prefix}_{part}_{name}", "all", [o[:3] for o in outputs])
        gate(f"P_{part}_{name}", "all", "P1_" + name if part == "PC1" else "P24_" + name)
        print(
            f"combined {part} {name}: {payload[f'P_{part}_{name}_acc_all'].mean():.3f} -> "
            f"{payload[f'Z_{part}_{name}_acc_all'].mean():.3f} "
            f"(top C {top_share(f'P_{part}_{name}', 'all'):.2f})",
            flush=True,
        )

    boot_index = np.random.default_rng(BOOT_SEED).integers(0, y.size, (N_BOOT, y.size))

    def paired_ci(baseline: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Mean of other - baseline over slides, and its 95% bootstrap interval."""
        delta = other.mean(axis=0) - baseline.mean(axis=0)
        draws = delta[boot_index].mean(axis=1)
        return np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])

    for name in MODELS:
        for k in range(N_SLICES):
            p, z = f"P_R_{name}", f"Z_R_{name}"
            payload[f"ci_scaling_R_{name}_{k}"] = paired_ci(
                payload[f"{p}_correct_{k}"], payload[f"{z}_correct_{k}"]
            )
            payload[f"ci_scaling_ll_R_{name}_{k}"] = paired_ci(
                per_slide_ll[f"{z}_{k}"], per_slide_ll[f"{p}_{k}"]
            )
            payload[f"ci_res_chance_{name}_{k}"] = paired_ci(chance_correct, payload[f"{z}_correct_{k}"])
            payload[f"ci_res_ll_{name}_{k}"] = paired_ci(per_slide_ll[f"{z}_{k}"], per_slide_ll["base"])
        for part in ("PC1", "PC24"):
            p, z = f"P_{part}_{name}", f"Z_{part}_{name}"
            payload[f"ci_scaling_{part}_{name}_all"] = paired_ci(
                payload[f"{p}_correct_all"], payload[f"{z}_correct_all"]
            )
            payload[f"ci_scaling_ll_{part}_{name}_all"] = paired_ci(
                per_slide_ll[f"{z}_all"], per_slide_ll[f"{p}_all"]
            )
            key = part.lower()
            payload[f"ci_{key}_{name}_all"] = paired_ci(
                z_density_all["correct"], payload[f"{z}_correct_all"]
            )
            payload[f"ci_{key}_ll_{name}_all"] = paired_ci(
                per_slide_ll[f"{z}_all"], per_slide_ll["Z_density_all"]
            )

    out = RESULTS / "scaled_single_block.npz"
    np.savez(out, **payload)
    meta = {
        "status": "post-hoc sensitivity analysis, not pre-registered",
        "arms": {
            "P_R/Z_R": "per-slice residual arm (four scores residualised on the slice's "
            "density arm inside each outer training fold), protocol refit / standardized "
            "residuals; gate: P_R equals density_adds.npz's R arm for every seed",
            "P_PC1/Z_PC1, P_PC24/Z_PC24": "combined PC1 of every slice / PCs 2-4 of every "
            "slice, protocol refit / standardized; gate: equal to P1/P24 for every seed",
        },
        "indicator": "share of fitted folds whose chosen C is the grid's top (1e4)",
        "intervals": "paired bootstrap over slides of seed-averaged per-slide values, "
        "shared indices default_rng(12345).integers(0, 123, (4000, 123)); scaling: Z - P "
        "(accuracy), P - Z (log-loss); res_chance: Z_R - chance; res_ll: prevalence - Z_R "
        "(log-loss); pc1/pc24: Z_PC - Z_density_all of scaled_section7.npz (accuracy), "
        "Z_density_all - Z_PC (log-loss)",
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
