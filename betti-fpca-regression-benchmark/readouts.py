"""The Betti FPCR benchmark's readouts: the paired differences between arms, their intervals and the verdicts.

``design.md``'s "Decision rule, fixed in advance" (committed in 7075684, before any result) defines them on the
arms' predictions in ``results/arms.npz``; ``run_study.py readouts`` writes them to ``results/readouts.json``.

Δ(X, Y) is the paired difference between arms X and Y, signed so that Δ > 0 favours X (``delta``):

- **interpolation:** R²(X) − R²(Y) per dataset, as (10 sizes, 10 runs). The readout is its mean over the 100
  datasets; its interval resamples the 10 runs within each size (``interval_interp``);
- **curvature:** se(Y) − se(X) per test cloud, se the squared error, so that its mean over the 50 test clouds is
  MSE(Y) − MSE(X). Its interval resamples the test clouds (``interval_units``), so it covers test sampling only:
  there is one training set;
- **fluid:** R²(X) − R²(Y) per time, split and size. The readout is its mean over the 5 times × 100 splits × 5
  sizes; its interval resamples the 100 splits (``interval_units``), each a unit with its mean Δ over the 5 times ×
  5 sizes. The splits all draw on the same 400 simulations, so the interval bounds the mean Δ over random splits of
  these simulations, not over new simulations.

Every interval holds the 2.5 and 97.5 percentiles of ``N_BOOT`` bootstrap means, drawn from a fresh
``np.random.default_rng(BOOT_SEED)``. The primary pair, F against K, gets a verdict per task (``verdict``); the
exploratory pairs, B against K (the same estimator on another representation) and F against B (the same
representation under another estimator), get the same readouts and no verdict. ``readouts`` gives them all, with
the descriptive readouts, the chosen (K0, K1) and the paper's tables, where upstream's reported values (its CSVs,
read as data) stand beside arms K, F and B.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

import common as c

TASKS = ("interp", "curv", "fluid")  # as run_study.TASKS
ARMS = ("K", "F", "B")  # as run_study.ARMS: the tables' arms
PRIMARY = ("F", "K")  # the only pair with a verdict
EXPLORATORY = (("B", "K"), ("F", "B"))
PERCENTILES = (2.5, 97.5)
RULE = (
    "design.md, 'Decision rule, fixed in advance' (committed in 7075684, before any result): per task, arm F against "
    "arm K, Delta signed so that Delta > 0 favours F; 'beats' (PD_kernel) if the 95% interval lies above 0, 'below' "
    "if it lies below 0, 'level' otherwise: the readout cannot tell the two apart, which does not mean they are "
    "equal. No pooled verdict; the exploratory pairs and the descriptive readouts carry none."
)
DELTA = {
    "interp": "Delta(X, Y) = R2(X) - R2(Y) per dataset (10 sizes x 10 runs); the mean over the 100 datasets; the "
    "interval resamples the 10 runs within each size",
    "curv": "Delta(X, Y) = se(Y) - se(X) per test cloud, se the squared error, so that the mean over the 50 test "
    "clouds is MSE(Y) - MSE(X); the interval resamples the 50 test clouds, so it covers test sampling only (one "
    "training set)",
    "fluid": "Delta(X, Y) = R2(X) - R2(Y) per time, split and size; the mean over 5 times x 100 splits x 5 sizes; the "
    "interval resamples the 100 splits, each by its mean Delta over the 5 times x 5 sizes. The splits all draw on the "
    "same 400 simulations, so it bounds the mean Delta over random splits of these simulations, not over new "
    "simulations",
    "interp_by_size": "a size's mean Delta over its 10 runs; the interval resamples those runs",
    "fluid_by_time": "a time's mean Delta over its 100 splits x 5 sizes; the interval resamples the 100 splits, each "
    "by its mean Delta over the 5 sizes",
    "fluid_by_size": "a training size's mean Delta over its 5 times x 100 splits; the interval resamples the 100 "
    "splits, each by its mean Delta over the 5 times",
    "fluid_paper_splits": "the fluid readout on the paper's splits 0-4 alone; the interval resamples those 5 splits",
}


def interval_interp(delta: np.ndarray) -> tuple[float, float, float]:
    """Interpolation's readout of the paired differences ``delta``, (sizes, runs): their mean and 95% interval.

    The bootstrap resamples the runs within each size: every one of ``N_BOOT`` draws takes, for each size, as many
    of its runs as it has, with replacement, and averages them all. The interval holds the draws' 2.5 and 97.5
    percentiles. Here ``delta`` is (10, 10).
    """
    sizes, runs = delta.shape
    rng = np.random.default_rng(c.BOOT_SEED)
    idx = rng.integers(0, runs, size=(c.N_BOOT, sizes, runs))
    draws = np.take_along_axis(np.broadcast_to(delta, (c.N_BOOT, sizes, runs)), idx, axis=2).mean(axis=(1, 2))
    return float(delta.mean()), *map(float, np.percentile(draws, PERCENTILES))


def interval_units(values: np.ndarray) -> tuple[float, float, float]:
    """The mean of ``values``, one per unit, and its 95% interval from a bootstrap over the units.

    Every one of ``N_BOOT`` draws takes ``len(values)`` units with replacement and averages them; the interval holds
    the draws' 2.5 and 97.5 percentiles. The units: a test cloud's se(Y) − se(X) (curvature) and a split's mean Δ
    over its times and sizes (fluid); for the descriptive readouts, one size's runs (interpolation), and for fluid one
    time's splits, each by its mean Δ over the sizes, one size's splits, each by its mean Δ over the times, and the
    paper's splits 0-4. ``ValueError`` unless ``values`` is 1-D: more axes would be resampled along the first and
    pooled, silently.
    """
    if np.ndim(values) != 1:
        raise ValueError(f"interval_units takes one value per unit, a 1-D array, not one of shape {np.shape(values)}")
    n = len(values)
    idx = np.random.default_rng(c.BOOT_SEED).integers(0, n, size=(c.N_BOOT, n))
    draws = values[idx].mean(1)
    return float(values.mean()), *map(float, np.percentile(draws, PERCENTILES))


def verdict(lo: float, hi: float) -> str:
    """The primary pair's verdict on its 95% interval [lo, hi], by the rule fixed in advance (``RULE``).

    ``"beats"`` if the interval lies above 0 (lo > 0), ``"below"`` if it lies below 0 (hi < 0), ``"level"`` otherwise.
    ``ValueError`` if a bound is not finite: that interval is no readout, though the comparisons alone would call a
    NaN one level.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"an interval's bounds must be finite, not [{lo}, {hi}]")
    if lo > 0:
        return "beats"
    if hi < 0:
        return "below"
    return "level"


def delta(arms: dict[str, np.ndarray], task: str, x: str, y: str) -> np.ndarray:
    """Δ(x, y) of ``task``, from the arrays of ``arms.npz``, signed so that Δ > 0 favours arm ``x``.

    - interpolation: R²(x) − R²(y), reshaped to (sizes, runs): the datasets come in size order, then run order;
    - curvature: se(y) − se(x) per test cloud, se the squared error of a prediction of ``curv_y_test``;
    - fluid: R²(x) − R²(y), (times, splits, sizes).

    ``ValueError`` on another task.
    """
    match task:
        case "interp":
            return (arms[f"interp_r2_{x}"] - arms[f"interp_r2_{y}"]).reshape(len(c.INTERP_SIZES), c.INTERP_RUNS)
        case "curv":
            target = arms["curv_y_test"]
            return (target - arms[f"curv_pred_{y}"]) ** 2 - (target - arms[f"curv_pred_{x}"]) ** 2
        case "fluid":
            return arms[f"fluid_r2_{x}"] - arms[f"fluid_r2_{y}"]
    raise ValueError(f"a task is one of {TASKS}, not {task!r}")


def _readout(interval: tuple[float, float, float], judged: bool = False) -> dict:
    """An interval's mean, lo and hi, and its verdict if ``judged``."""
    mean, lo, hi = interval
    return {"mean": mean, "lo": lo, "hi": hi} | ({"verdict": verdict(lo, hi)} if judged else {})


def pair_readout(arms: dict[str, np.ndarray], x: str, y: str, judged: bool = False) -> dict:
    """The readouts of Δ(x, y) on the three tasks (module docstring), with a verdict per task if ``judged``.

    Beside them, the descriptive readouts of ``design.md``, with the same recipes and never a verdict:

    - interpolation, ``by_size``: each size's mean Δ, its interval resampling that size's 10 runs;
    - fluid, ``by_time``: each time's mean Δ, its interval resampling the 100 splits, each by its mean Δ over the 5
      sizes;
    - fluid, ``by_size``: each training size's mean Δ, its interval resampling the 100 splits, each by its mean Δ over
      the 5 times;
    - fluid, ``paper_splits``: the readout on the paper's splits 0-4 alone, its interval resampling those 5 splits.
    """
    interp, curv, fluid = (delta(arms, task, x, y) for task in TASKS)
    splits = fluid.mean(axis=(0, 2))  # each split's mean Δ over the times and sizes
    by_size = [{"n_train": n} | _readout(interval_units(interp[si])) for si, n in enumerate(c.INTERP_SIZES)]
    by_time = [{"t": t} | _readout(interval_units(fluid[ti].mean(axis=1))) for ti, t in enumerate(c.FLUID_TIMES)]
    fluid_by_size = [
        {"n_train": n} | _readout(interval_units(fluid[:, :, si].mean(axis=0))) for si, n in enumerate(c.FLUID_SIZES)
    ]
    paper = {"splits": list(range(c.PAPER_SPLITS))} | _readout(interval_units(splits[: c.PAPER_SPLITS]))
    return {
        "pair": [x, y],
        "interp": _readout(interval_interp(interp), judged) | {"by_size": by_size},
        "curv": _readout(interval_units(curv), judged),
        "fluid": _readout(interval_units(splits), judged)
        | {"by_time": by_time, "by_size": fluid_by_size, "paper_splits": paper},
    }


def _counts(k: np.ndarray) -> list[dict]:
    """How often arm F chose each (K0, K1) in the cells ``k``, (..., 2), in (K0, K1) order."""
    counts = Counter(map(tuple, np.asarray(k).reshape(-1, 2).tolist()))
    return [{"k0": k0, "k1": k1, "count": n} for (k0, k1), n in sorted(counts.items())]


def chosen_k(arms: dict[str, np.ndarray]) -> dict:
    """Arm F's chosen (K0, K1), counted per task over its cells, and for fluid per time too."""
    return {
        "interp": _counts(arms["interp_k_F"]),
        "curv": _counts(arms["curv_k_F"]),
        "fluid": _counts(arms["fluid_k_F"]),
        "fluid_by_time": [{"t": t, "counts": _counts(arms["fluid_k_F"][ti])} for ti, t in enumerate(c.FLUID_TIMES)],
    }


def _mean_sd(values: np.ndarray) -> dict:
    """The mean and SD (ddof 1) of ``values``, as the paper's tables give R² over runs or splits."""
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def _same_cells(name: str, found: list, expected: list) -> None:
    """``ValueError`` unless upstream's ``name`` holds exactly the cells ``expected``, each once."""
    if sorted(found) != sorted(expected):
        raise ValueError(f"upstream's {name} holds {len(found)} cells, not exactly the {len(expected)} of the table")


def _reported(folder: str, name: str) -> pd.DataFrame:
    """Upstream's reported table ``folder/name``, read as a CSV with pandas and nothing else.

    ``float_precision="round_trip"`` reads every value as Python's ``float`` reads its text: pandas' default parser
    misreads some 17-digit values in their last bits, 52 of upstream's 101 reported values.
    """
    return pd.read_csv(c.UPSTREAM / folder / name, float_precision="round_trip")


def tables(arms: dict[str, np.ndarray]) -> dict:
    """The paper's three tables in its format: upstream's reported values beside arms K, F and B.

    - interpolation, per training size: the mean and SD of R² over the size's 10 runs for K, F and B, beside the
      reported ``PD_kernel`` and ``PI_kernel`` (``interpolation/results_interpolation.csv``);
    - curvature: the MSE of K, F and B, beside every reported method, the ten of Turkeš et al. and ``PD_kernel``, in
      the order of ``curvature/results_curvature.csv``;
    - fluid, per time and training size: the mean and SD of R² over the paper's splits 0-4 (its protocol) for K, F
      and B, beside the reported ``R2_mean`` and ``R2_std`` (``fluid/results_fluid.csv``), and the same over all 100
      splits.

    Every SD has ddof 1, as upstream's: its fluid ``R2_std`` is the ddof-1 SD of its splits' R². The CSVs are read
    by ``_reported``, exactly. ``ValueError`` if a CSV does not hold exactly the table's cells, or if the curvature
    table names a method twice or lacks ``PD_kernel``.
    """
    table = _reported("interpolation", "results_interpolation.csv")
    _same_cells("results_interpolation.csv", table["n_train"].tolist(), list(c.INTERP_SIZES))
    reported = table.set_index("n_train")
    r2 = {arm: arms[f"interp_r2_{arm}"].reshape(len(c.INTERP_SIZES), c.INTERP_RUNS) for arm in ARMS}
    interp = [
        {"n_train": n}
        | {
            "reported": {
                method: {stat: float(reported.loc[n, f"{method}_R2_{stat}"]) for stat in ("mean", "std")}
                for method in ("PD_kernel", "PI_kernel")
            }
        }
        | {arm: _mean_sd(r2[arm][si]) for arm in ARMS}
        for si, n in enumerate(c.INTERP_SIZES)
    ]
    table = _reported("curvature", "results_curvature.csv")
    methods = [str(method) for method in table["method"]]
    if len(set(methods)) != len(methods) or "PD_kernel" not in methods:
        raise ValueError(
            f"upstream's results_curvature.csv must name each method once, PD_kernel among them: {methods}"
        )
    curv = {"reported": dict(zip(methods, table["MSE"].astype(float).tolist(), strict=True))}
    curv |= {arm: float(arms[f"curv_mse_{arm}"]) for arm in ARMS}
    table = _reported("fluid", "results_fluid.csv")
    found = [(float(t), int(n)) for t, n in table[["t", "n_train"]].itertuples(index=False)]
    _same_cells("results_fluid.csv", found, [(t, n) for t in c.FLUID_TIMES for n in c.FLUID_SIZES])
    means, stds = table["R2_mean"].astype(float).tolist(), table["R2_std"].astype(float).tolist()
    reported = {cell: {"mean": m, "std": s} for cell, m, s in zip(found, means, stds, strict=True)}
    fluid = [
        {"t": t, "n_train": n, "reported": reported[t, n]}
        | {"paper_splits": {arm: _mean_sd(arms[f"fluid_r2_{arm}"][ti, : c.PAPER_SPLITS, si]) for arm in ARMS}}
        | {"all_splits": {arm: _mean_sd(arms[f"fluid_r2_{arm}"][ti, :, si]) for arm in ARMS}}
        for ti, t in enumerate(c.FLUID_TIMES)
        for si, n in enumerate(c.FLUID_SIZES)
    ]
    return {"interp": interp, "curv": curv, "fluid": fluid}


def readouts(arms: dict[str, np.ndarray]) -> dict:
    """Every readout of the arms' arrays ``arms`` (``run_study.load_arms``), as plain JSON values.

    - ``primary``: Δ(F, K) per task with its interval and verdict (``pair_readout``, judged), and its descriptive
      readouts;
    - ``exploratory``: Δ(B, K) and Δ(F, B), keyed ``B_vs_K`` and ``F_vs_B``, the same readouts without a verdict;
    - ``chosen_k``: arm F's chosen (K0, K1) (``chosen_k``);
    - ``tables``: the paper's three tables, the reported values beside K, F and B (``tables``);

    with the rule, the bootstrap's recipe and the definition of every Δ.
    """
    return {
        "rule": RULE,
        "bootstrap": {"n_boot": c.N_BOOT, "seed": c.BOOT_SEED, "percentiles": list(PERCENTILES)},
        "delta": DELTA,
        "primary": pair_readout(arms, *PRIMARY, judged=True),
        "exploratory": {f"{x}_vs_{y}": pair_readout(arms, x, y) for x, y in EXPLORATORY},
        "chosen_k": chosen_k(arms),
        "tables": tables(arms),
    }
