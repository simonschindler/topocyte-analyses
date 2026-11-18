#!/usr/bin/env python
"""Post-hoc: every nested arm of section 7 refitted with standardized features.

Section 8's post-hoc check (``scaled_nested.py``) found the nested test
sensitive to the scale of the features: the protocol estimator, an
L2-penalized logistic regression whose C is chosen by inner cross-validation,
sees every feature on its own scale, and a block of scores whose spread is
far below or far above the density features' gets almost none or almost all
of the weight.  That script refitted the H1 Betti arms of slices 0 and 6-9.
This one refits all 44 nested arms of section 7 - the slice's density arm
plus its first four fPCA scores, for the four families (H0 and H1, raw and
cosine) at slices 0-9, and the four combined models (the 101-feature density
arm plus all 40 scores) - and the eleven density arms, with a StandardScaler,
fitted on the outer training fold only, in front of the same estimator.  It
was not pre-registered; the report labels it post-hoc.

``model`` is one of ``density_adds.MODELS``, ``suffix`` a slice ``0``-``9``
or ``all``:

* ``P_{model}``: the protocol nested arm (density arm plus scores, unscaled),
  refitted to record each fold's chosen C and the scores' weight share.
  Gate: its held-out outputs equal the saved arm ``B_{model}`` of
  ``density_adds.npz`` for every seed.
* ``Z_density``: the density arm, standardized.
* ``Z_{model}``: the nested arm, standardized.  Gate: at slices 0 and 6-9 the
  H1 arms and the density arm equal ``scaled_nested.npz``'s, seed for seed.

The weight share is ``scaled_nested.fit_folds``'s: ``sum_j |coef_j| SD_j``
over the score columns divided by the same sum over all columns, ``SD_j`` the
training-fold standard deviation of the column as the classifier sees it.

Intervals: paired bootstrap over slides with ``density_adds.py``'s shared
indices - standardized nested minus standardized density arm in accuracy,
density minus nested in log-loss (positive favours the scores), and the
standardized minus the protocol density arm.

Reads only saved results (``density_adds.npz``, ``slice_<k>.npz``,
``combined.npz``, ``scaled_nested.npz``).  Run with ``PYTHONPATH=<topocyte checkout>``
and the checkout's interpreter.  Writes ``results/scaled_section7.npz`` and
``.json``.
"""

from __future__ import annotations

import json
import time
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
    log_loss,
)
from scaled_nested import fit_folds

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUFFIXES = tuple(str(k) for k in range(N_SLICES)) + ("all",)
# Slices whose H1 arms scaled_nested.py already refitted, and its name for them.
OVERLAP = ("0", "6", "7", "8", "9")
OVERLAP_NAMES = {"h1_none": "betti_none", "h1_cosine": "betti_cosine"}


def main() -> None:
    import sklearn
    import topocyte
    from sklearn.metrics import roc_auc_score

    started = time.time()
    with np.load(RESULTS / "density_adds.npz", allow_pickle=False) as adds:
        y = adds["y"]
        group_ids = adds["group_ids"]
        log_cells = adds["log_cells"][:, None]
        density = {
            str(k): np.hstack([adds[f"rich_scores_k{k}"], log_cells]) for k in range(N_SLICES)
        }
        density["all"] = np.hstack([adds["rich_scores_all"], log_cells])
        protocol_density = {s: adds[f"A_correct_{s}"] for s in SUFFIXES}
        saved = {
            (name, s, field): adds[f"B_{name}_{field}_{s}"]
            for name in MODELS
            for s in SUFFIXES
            for field in ("correct", "proba")
        }
    blocks: dict[tuple[str, str], np.ndarray] = {}
    for k in range(N_SLICES):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            if not np.array_equal(one["group_ids"], group_ids):
                raise RuntimeError(f"slice {k}: slides differ from density_adds")
            for name in MODELS:
                blocks[(name, str(k))] = one[f"{name}_scores"][:, :N_COMPONENTS]
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as combined:
        if not np.array_equal(combined["y"], y):
            raise RuntimeError("combined: y differs from density_adds")
        for name in MODELS:
            blocks[(name, "all")] = combined[f"{name}_scores"]
    with np.load(RESULTS / "scaled_nested.npz", allow_pickle=False) as section8:
        if not (np.array_equal(section8["y"], y) and np.array_equal(section8["group_ids"], group_ids)):
            raise RuntimeError("scaled_nested: slides differ from density_adds")
        reference = {key: section8[key] for key in section8.files if key.startswith("Z_")}

    payload: dict[str, np.ndarray] = {
        "y": y,
        "group_ids": group_ids,
        "suffixes": np.array(SUFFIXES),
        "models": np.array(MODELS),
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

    def same_arm(field: str, other: str, suffix: str, keys: tuple[str, ...]) -> bool:
        """True if two stored arms agree: correctness exactly, the rest to 1e-12."""
        for key in keys:
            mine, theirs = payload[f"{field}_{key}_{suffix}"], reference[f"{other}_{key}_{suffix}"]
            if key == "correct":
                if not np.array_equal(mine, theirs):
                    return False
            elif not np.allclose(mine, theirs, rtol=0.0, atol=1e-12):
                return False
        return True

    for s in SUFFIXES:
        clock = time.time()
        payload[f"density_sd_{s}"] = density[s].std(axis=0, ddof=1)
        run_arm("Z_density", s, density[s], 0, True)
        if s in OVERLAP and not same_arm("Z_density", "Z_density", s, ("correct", "proba")):
            raise RuntimeError(f"slice {s}: the standardized density arm is not scaled_nested's")
        for name in MODELS:
            block = blocks[(name, s)]
            payload[f"score_sd_{name}_{s}"] = block.std(axis=0, ddof=1)
            X = np.hstack([density[s], block])
            run_arm(f"P_{name}", s, X, block.shape[1], False)
            # Gate: the unscaled refit is section 7's nested arm, seed by seed.
            if not (
                np.array_equal(payload[f"P_{name}_correct_{s}"], saved[(name, s, "correct")])
                and np.allclose(
                    payload[f"P_{name}_proba_{s}"], saved[(name, s, "proba")], rtol=0.0, atol=1e-12
                )
            ):
                raise RuntimeError(f"{name} {s}: the refit is not the saved nested arm")
            run_arm(f"Z_{name}", s, X, block.shape[1], True)
            if s in OVERLAP and name in OVERLAP_NAMES:
                if not same_arm(
                    f"Z_{name}", f"Z_{OVERLAP_NAMES[name]}", s, ("correct", "proba", "C", "share")
                ):
                    raise RuntimeError(f"{name} {s}: the standardized arm is not scaled_nested's")
        print(
            f"{s}: standardized density {payload[f'Z_density_acc_{s}'].mean():.3f}; "
            + "; ".join(
                f"{name} {payload[f'Z_{name}_acc_{s}'].mean():.3f} "
                f"(share {np.median(payload[f'P_{name}_share_{s}']):.3f} -> "
                f"{np.median(payload[f'Z_{name}_share_{s}']):.3f})"
                for name in MODELS
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
        density_correct = payload[f"Z_density_correct_{s}"]
        for name in MODELS:
            payload[f"ci_nested_{name}_{s}"] = paired_ci(
                density_correct, payload[f"Z_{name}_correct_{s}"]
            )
            payload[f"ci_nested_ll_{name}_{s}"] = paired_ci(
                per_slide_ll[f"Z_{name}_{s}"], per_slide_ll[f"Z_density_{s}"]
            )
        payload[f"ci_scaling_density_{s}"] = paired_ci(protocol_density[s], density_correct)

    out = RESULTS / "scaled_section7.npz"
    np.savez(out, **payload)
    meta = {
        "status": "post-hoc sensitivity analysis, not pre-registered",
        "suffixes": list(SUFFIXES),
        "models": list(MODELS),
        "arms": {
            "P": "section 7's nested arm refitted (density arm + scores, unscaled); "
            "gate: equals density_adds.npz's B arm for every seed",
            "Z_density": "density arm with a StandardScaler fitted on the outer training fold",
            "Z": "nested arm with a StandardScaler fitted on the outer training fold; gate: "
            "the H1 arms and density arms at slices 0 and 6-9 equal scaled_nested.npz's",
        },
        "weight_share": "sum over the score columns of |coef_j| * SD_j over the sum over "
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
