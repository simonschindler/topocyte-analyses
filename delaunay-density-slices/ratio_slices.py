#!/usr/bin/env python
"""Topology with density removed: fPCA of the H1 death/birth ratio (Task 7).

For every finite H1 bar of a slide's alpha filtration, ``pi = sqrt(death /
birth)`` is the ratio of its death and birth radii: alpha filtration values
are squared radii, and alpha diagrams equal Cech diagrams.  Scaling the cells
leaves ``pi`` unchanged, so neither a uniform change of density nor the assumed
pixel size moves it, and for bars of noise its distribution does not depend on
the sampling density (Bobrowski & Skraba, Sci. Rep. 2023).

Each slide's curve is the survival function ``S(t)``, the number of its H1
bars with ``log pi > t`` for ``t >= 0``.  That is the Betti curve of the bars
``[0, log pi)``, so ``topocyte.ph.FPCA`` fits it exactly from the bar
endpoints, without a grid; the script checks its Gram matrix against the
closed form ``sum_ij min(v_i, v_j)`` for every pair of slides.

Slices 0 (all cells) and 6-9.  For each slice:

* persistent homology, recomputed as ``run_slices.py`` computes it (no run
  saved the bars).  Gate: the H1 Betti-curve fPCA of the recomputed diagrams
  gives the saved first four scores back.
* ratio fPCA with normalization ``"none"``, which keeps the bar count, and
  ``"cosine"``, which removes it exactly (``S = N F``, so ``S / ||S|| =
  F / ||F||``).
* arms under the protocol estimator and folds, 20 seeds: A, the slice's
  density arm (10 features + log10 cells), recomputed and equal to
  ``density_adds.npz``; Q, the first four ratio scores alone; N, A plus the
  four ratio scores.
* paired bootstrap intervals with the shared indices of ``density_adds.py``:
  N - A and Q - A against the density arm, Q against chance, in accuracy and
  log-loss.
* R2 of each ratio score on the density arm (in-sample OLS), as section 7
  reports for the Betti scores.

Run with ``PYTHONPATH=<topocyte checkout>`` and the checkout's interpreter.  Writes
``results/ratio_slices.npz`` and ``results/ratio_slices.json``; ``--slices``
and ``--out`` give a smoke run elsewhere.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import run_slices  # imports topocyte from the PYTHONPATH checkout and checks that it did
from density_adds import (
    BOOT_SEED,
    N_BOOT,
    N_COMPONENTS,
    N_JOBS,
    SEEDS,
    chance_outputs,
    cv_outputs,
    log_loss,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SLICES = (0, 6, 7, 8, 9)
FAMILIES = ("none", "cosine")
LEVELS = np.array([0.5, 0.9, 0.99, 0.999])
GATE_RTOL = 1e-6


def log_ratio(pairs: np.ndarray) -> np.ndarray:
    """Sorted ``log pi = log sqrt(death / birth)`` of the bars alive in a diagram."""
    births, deaths = pairs[:, 0], pairs[:, 1]
    alive = np.isfinite(deaths) & (deaths > births)
    births, deaths = births[alive], deaths[alive]
    if births.size == 0:
        raise RuntimeError("a slide has no finite H1 bar")
    if np.any(births <= 0):
        raise RuntimeError("an H1 bar is born at radius 0; its ratio is undefined")
    values = 0.5 * (np.log(deaths) - np.log(births))
    # A bar with log pi = 0 is [0, 0): never alive, and BettiCurve drops it too.
    return np.sort(values[values > 0])


def closed_form_gram(values: list[np.ndarray]) -> np.ndarray:
    """``sum_ij min(u_i, v_j)`` for every pair of sorted samples ``u``, ``v``."""
    cumsums = [np.concatenate([[0.0], np.cumsum(v)]) for v in values]
    n = len(values)
    gram = np.empty((n, n))
    for a in range(n):
        u = values[a]
        for b in range(a, n):
            v = values[b]
            below = np.searchsorted(v, u, side="right")  # the v_j <= u_i
            total = cumsums[b][below].sum() + (u * (v.size - below)).sum()
            gram[a, b] = gram[b, a] = total
    return gram


def slice_results(density: dict, k: int, group_ids: list[str]) -> tuple[list, np.ndarray]:
    """Persistence results of slice ``k`` in ``group_ids`` order, as run_slices.py computes them."""
    tc = run_slices.tc
    threshold = float(density["deciles"][k])
    n_cells = density["rho"].size
    keep = np.ones(n_cells, dtype=bool) if k == 0 else density["rho"] >= threshold
    sub = run_slices._slice_anndata(density, keep)
    started = time.time()
    tc.tl.persistent_homology(
        sub,
        filtration="alpha",
        max_dim=1,
        groupby="slide_id",
        microns_per_unit=run_slices.PIXEL_UM,
        n_jobs=-1,
        key_added="slide_ph",
    )
    groups = tc.tl.read_persistence(sub, "slide_ph")
    if len(groups) != len(group_ids) or groups.skipped:
        raise RuntimeError(f"slice {k}: {len(groups)} groups, skipped {list(groups.skipped)}")
    kept = sub.obs["slide_id"].value_counts().reindex(group_ids).to_numpy().astype(np.int64)
    results = [groups[g] for g in group_ids]
    print(
        f"slice {k}: d={threshold:.1f} cells/mm^2, {int(keep.sum()):,} cells, "
        f"PH in {time.time() - started:.1f} s",
        flush=True,
    )
    del sub, groups
    gc.collect()
    return results, kept


def main() -> None:
    import gudhi
    import sklearn
    import topocyte
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score
    from topocyte.ph import FPCA, BettiCurve

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slices", type=int, nargs="+", default=list(SLICES))
    parser.add_argument("--out", type=Path, default=RESULTS / "ratio_slices.npz")
    args = parser.parse_args()
    slices = tuple(args.slices)

    started = time.time()
    with np.load(RESULTS / "density_adds.npz", allow_pickle=False) as adds:
        group_ids = [str(g) for g in adds["group_ids"]]
        y = np.asarray(adds["y"], dtype=np.int8)
        log_cells = adds["log_cells"]
        density_k = {
            k: np.hstack([adds[f"rich_scores_k{k}"], log_cells[:, None]]) for k in slices
        }
    saved = {}
    for k in slices:
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as one:
            if [str(g) for g in one["group_ids"]] != group_ids:
                raise RuntimeError(f"slice {k}: group order differs from density_adds")
            saved[k] = {
                "kept": one["kept_per_slide"],
                **{m: one[f"h1_{m}_scores"][:, :N_COMPONENTS] for m in FAMILIES},
            }

    payload: dict[str, np.ndarray] = {
        "group_ids": np.array(group_ids),
        "y": y,
        "seeds": SEEDS,
        "slices": np.array(slices, dtype=np.int64),
        "families": np.array(FAMILIES),
        "levels": LEVELS,
    }
    chance = [chance_outputs(y, int(seed)) for seed in SEEDS]
    payload["chance_correct"] = np.array([c for c, _ in chance])
    payload["base_proba"] = np.array([p for _, p in chance])
    per_slide_ll: dict[str, np.ndarray] = {"base": log_loss(payload["base_proba"], y)}

    # --- persistent homology, gate, ratio curves and their fPCA ------------------------
    density = run_slices._load_density()
    ratio_scores: dict[tuple[int, str], np.ndarray] = {}
    for k in slices:
        results, kept = slice_results(density, k, group_ids)
        if not np.array_equal(kept, saved[k]["kept"]):
            raise RuntimeError(f"slice {k}: kept cells differ from run_slices.py")
        payload[f"kept_{k}"] = kept

        for m in FAMILIES:
            tick = time.time()
            betti = FPCA(dim=1, normalization=m, n_jobs=-1).fit(results)
            got = betti.result_.scores[:, :N_COMPONENTS]
            want = saved[k][m]
            signs = np.sign(np.sum(got * want, axis=0))
            deviation = np.max(np.abs(got * signs - want), axis=0) / np.max(np.abs(want), axis=0)
            if np.any(deviation > GATE_RTOL):
                raise RuntimeError(
                    f"gate: slice {k} H1 {m} scores deviate from the saved run by "
                    f"{deviation.max():.2e} (relative)"
                )
            payload[f"betti_{m}_scores_{k}"] = got
            print(
                f"  gate H1 {m}: saved scores reproduced (max relative deviation "
                f"{deviation.max():.1e}) in {time.time() - tick:.1f} s",
                flush=True,
            )
            del betti

        values = [log_ratio(result.diagrams[1]) for result in results]
        del results
        gc.collect()
        payload[f"n_bars_{k}"] = np.array([v.size for v in values], dtype=np.int64)
        payload[f"sum_log_pi_{k}"] = np.array([v.sum() for v in values])
        payload[f"max_log_pi_{k}"] = np.array([v[-1] for v in values])
        payload[f"log_pi_quantiles_{k}"] = np.array([np.quantile(v, LEVELS) for v in values])
        curves = [BettiCurve(np.column_stack([np.zeros(v.size), v]), dim=1) for v in values]

        raw_grams = []
        for m in FAMILIES:
            tick = time.time()
            model = FPCA(dim=1, normalization=m, n_jobs=-1).fit(curves)
            ratio_scores[(k, m)] = model.result_.scores
            payload[f"ratio_{m}_scores_{k}"] = model.result_.scores
            payload[f"ratio_{m}_evr_{k}"] = model.result_.explained_variance_ratio
            raw_grams.append(model.result_.raw_gram)
            print(
                f"  ratio fPCA {m} in {time.time() - tick:.1f} s, EVR[0:4]="
                f"{np.round(model.result_.explained_variance_ratio[:4], 3)}",
                flush=True,
            )
            del model
        if not np.array_equal(raw_grams[0], raw_grams[1]):
            raise RuntimeError(f"slice {k}: the two normalizations saw different raw Grams")
        tick = time.time()
        closed = closed_form_gram(values)
        relative = np.max(np.abs(raw_grams[0] - closed) / closed)
        payload[f"ratio_gram_{k}"] = raw_grams[0]
        payload[f"gram_check_rel_{k}"] = np.float64(relative)
        print(
            f"  Gram vs closed form: max relative deviation {relative:.1e} "
            f"({time.time() - tick:.1f} s); H1 bars per slide "
            f"{payload[f'n_bars_{k}'].min():,}..{payload[f'n_bars_{k}'].max():,}",
            flush=True,
        )
        del values, curves, raw_grams, closed
        gc.collect()
    del density
    gc.collect()

    # --- arms ---------------------------------------------------------------------------
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

    print("arms", flush=True)
    for k in slices:
        run_arm("A", str(k), density_k[k])
        for m in FAMILIES:
            scores = ratio_scores[(k, m)][:, :N_COMPONENTS]
            run_arm(f"Q_{m}", str(k), scores)
            run_arm(f"N_{m}", str(k), np.hstack([density_k[k], scores]))
        print(
            f"  slice {k}: density {payload[f'A_acc_{k}'].mean():.3f}; "
            + "; ".join(
                f"{m}: ratio alone {payload[f'Q_{m}_acc_{k}'].mean():.3f}, "
                f"density + ratio {payload[f'N_{m}_acc_{k}'].mean():.3f}"
                for m in FAMILIES
            ),
            flush=True,
        )

    # --- paired bootstrap intervals, density_adds.py's shared indices --------------------
    boot_index = np.random.default_rng(BOOT_SEED).integers(0, y.size, (N_BOOT, y.size))

    def paired_ci(baseline: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Mean of other - baseline over slides, and its 95% bootstrap interval."""
        delta = other.mean(axis=0) - baseline.mean(axis=0)
        draws = delta[boot_index].mean(axis=1)
        return np.array([delta.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)])

    for k in slices:
        density_correct = payload[f"A_correct_{k}"]
        density_ll = per_slide_ll[f"A_{k}"]
        for m in FAMILIES:
            # Accuracy: arm minus baseline.  Log-loss: baseline minus arm, so that
            # a positive value favours the ratio in both.
            payload[f"ci_nested_{m}_{k}"] = paired_ci(density_correct, payload[f"N_{m}_correct_{k}"])
            payload[f"ci_nested_ll_{m}_{k}"] = paired_ci(per_slide_ll[f"N_{m}_{k}"], density_ll)
            payload[f"ci_alone_density_{m}_{k}"] = paired_ci(
                density_correct, payload[f"Q_{m}_correct_{k}"]
            )
            payload[f"ci_alone_chance_{m}_{k}"] = paired_ci(
                payload["chance_correct"], payload[f"Q_{m}_correct_{k}"]
            )
            payload[f"ci_alone_ll_{m}_{k}"] = paired_ci(per_slide_ll[f"Q_{m}_{k}"], per_slide_ll["base"])

    # --- how much of each ratio score the density arm explains ---------------------------
    r2 = np.empty((len(slices), len(FAMILIES), N_COMPONENTS))
    for i, k in enumerate(slices):
        for f, m in enumerate(FAMILIES):
            scores = ratio_scores[(k, m)][:, :N_COMPONENTS]
            for j in range(N_COMPONENTS):
                model = LinearRegression().fit(density_k[k], scores[:, j])
                residual = scores[:, j] - model.predict(density_k[k])
                total = scores[:, j] - scores[:, j].mean()
                r2[i, f, j] = 1.0 - residual @ residual / (total @ total)
    payload["ratio_r2"] = r2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    meta = {
        "curve": "survival function of log pi over each slide's finite H1 bars, "
        "pi = sqrt(death / birth) of the alpha filtration (squared radii); the "
        "Betti curve of the bars [0, log pi), fitted by topocyte.ph.FPCA(dim=1)",
        "families": list(FAMILIES),
        "slices": list(slices),
        "arms": {
            "A": "density arm of the slice: 10 density features + log10 cells "
            "(as density_adds.py), recomputed",
            "Q": "first 4 ratio fPCA scores alone",
            "N": "A + first 4 ratio fPCA scores",
        },
        "intervals": "paired bootstrap over slides of seed-averaged per-slide values, "
        "shared indices default_rng(12345).integers(0, 123, (4000, 123))",
        "gate": f"H1 Betti-curve fPCA of the recomputed diagrams reproduces the saved "
        f"first 4 scores within {GATE_RTOL} (relative, up to sign)",
        "quantile_levels": LEVELS.tolist(),
        "estimator": "LogisticRegressionCV(Cs=10, inner_cv=5, scoring=accuracy, max_iter=1000)",
        "cv": "StratifiedKFold(5, shuffle=True, random_state=seed)",
        "seeds": len(SEEDS),
        "topocyte": str(topocyte.__version__),
        "gudhi": str(gudhi.__version__),
        "sklearn": str(sklearn.__version__),
        "n_jobs": N_JOBS,
        "runtime_seconds": round(time.time() - started, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out} in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
