#!/usr/bin/env python
"""The Betti FPCR benchmark's arms K, F and B, and the gates between them, G1 and G3 (``design.md``).

Stage ``arms``, from this folder::

    caffeinate -is /usr/bin/time -l $PY run_study.py arms > logs/arms.log 2>&1

It reads ``results/matrices.npz`` through ``matrices.load_matrices``, which refuses an npz that
``results/matrices.json`` does not vouch for, and then, in this order:

1. arm K on every task, then gate G1: K must reproduce upstream's published results to ``G1_TOL``. If G1 fails,
   the stage writes ``results/arms.json`` and exits with status 1; arms F and B never run;
2. gate G3: the test scores that arm F projects from the Gram matrices (``betti.fpca_scores``) against topocyte's
   ``FPCA.fit(...).transform(...)`` of curves rebuilt from ``inputs``. On failure, the same;
3. arm F, then arm B, on every task; then ``results/arms.npz`` and ``results/arms.json``.

The arms (``design.md``, "Pipeline"):

- **K:** Nadaraya–Watson (``estimators.nw_fit_predict``) on D = W2(H0) + W2(H1);
- **F:** FPCR (``estimators.fpcr_fit_predict``) on the Gram matrices of the H0 and H1 Betti curves;
- **B:** Nadaraya–Watson on D = L2(H0) + L2(H1) between the Betti curves.

They see the same cells:

- **interpolation:** each of the 100 datasets, in ``inputs.interpolation_datasets()`` order (training size
  ascending, then runs 0-9), trains on its first n samples and tests on its last 50; R² by ``common.r2``;
- **curvature:** trains on clouds 0-49 and tests on 50-99; the mean squared error;
- **fluid:** ``fluid_cells``, each time, split 0-99 and size n: ``test, pool = inputs.fluid_split(s)`` and
  ``train = pool[:n]``, positions in ``inputs.fluid_sim_ids()`` order; R² by ``common.r2``.

``results/arms.npz`` holds, for each arm X: ``interp_pred_X`` (100, 50) and ``interp_r2_X`` (100,); ``curv_pred_X``
(50,) and ``curv_mse_X`` (); ``fluid_pred_X`` (5, 100, 5, 100), indexed [time, split, size, test], and
``fluid_r2_X`` (5, 100, 5); K's and B's bandwidths as ``{task}_h_X``, and F's (K0, K1) as ``interp_k_F`` (100, 2),
``curv_k_F`` (2,) and ``fluid_k_F`` (5, 100, 5, 2). Beside them: ``interp_y_test`` (100, 50), ``curv_y_test`` (50,),
``fluid_y`` (400,) and ``fluid_test`` (100, 100), each split's test positions. ``results/arms.json`` holds the
gates, the seconds per arm and task, the npz's size and sha256, and ``provenance``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import topocyte as tc

import betti
import common as c
import estimators as est
import inputs as inp
import matrices as mat

ARMS = ("K", "F", "B")  # the stage runs K, then G1 and G3, then F and B
TASKS = ("interp", "curv", "fluid")
_DISTANCES = {"K": "w2", "B": "l2"}  # each Nadaraya-Watson arm's distance: that kind's H0 matrix plus its H1 matrix
CURV_TRAIN, CURV_TEST = np.arange(50), np.arange(50, 100)  # upstream's split, which inputs.curvature_data checks
G3_INTERP = (0, 45, 99)  # G3's interpolation datasets
G3_FLUID = ((0, 0, 25), (2, 7, 100), (4, 99, 300))  # G3's fluid cells: (time index, split, training size)
_SCRIPTS = (
    "run_study.py",
    "common.py",
    "inputs.py",
    "safe_pickle.py",
    "matrices.py",
    "wasserstein.py",
    "betti.py",
    "estimators.py",
)


def fluid_cells() -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]:
    """Every fluid cell, ``(time index, split, size index, train, test)``: times, then splits, then sizes.

    ``test, pool = inputs.fluid_split(s)`` and ``train = pool[:n]``: a split's training sets are nested, the largest
    is its whole pool, and every time has the same splits.
    """
    for ti in range(len(c.FLUID_TIMES)):
        for s in range(c.FLUID_SPLITS):
            test, pool = inp.fluid_split(s)
            for si, n in enumerate(c.FLUID_SIZES):
                yield ti, s, si, pool[:n], test


def _ykey(task: str, idx: int | None) -> str:
    """The npz key of a dataset's targets: ``interp_{idx:02d}_y``, ``curv_y`` or ``fluid_y``."""
    return f"interp_{idx:02d}_y" if task == "interp" else f"{task}_y"


def _grid(task: str) -> tuple[tuple[int, ...], int]:
    """The shape of ``task``'s cells in the outputs, and each cell's number of test samples."""
    if task == "interp":
        return (len(c.INTERP_SIZES) * c.INTERP_RUNS,), c.INTERP_N_TEST
    if task == "curv":
        return (), CURV_TEST.size
    return (len(c.FLUID_TIMES), c.FLUID_SPLITS, len(c.FLUID_SIZES)), c.FLUID_N_TEST


def _cells(task: str, M: dict[str, np.ndarray]) -> Iterator[tuple[tuple[int, ...], int | None, np.ndarray, np.ndarray]]:
    """Every cell of ``task``: ``(position in the outputs, dataset index, train, test)``, as the module docstring says.

    An interpolation dataset's targets in ``M`` give its size: its n training samples, then ``INTERP_N_TEST`` tests.
    """
    if task == "interp":
        for idx in range(len(c.INTERP_SIZES) * c.INTERP_RUNS):
            n = M[_ykey(task, idx)].size - c.INTERP_N_TEST
            yield (idx,), idx, np.arange(n), np.arange(n, n + c.INTERP_N_TEST)
    elif task == "curv":
        yield (), None, CURV_TRAIN, CURV_TEST
    else:
        for ti, s, si, train, test in fluid_cells():
            yield (ti, s, si), ti, train, test


def _arm_matrices(arm: str, task: str, M: dict[str, np.ndarray], idx: int | None) -> tuple | np.ndarray:
    """Arm ``arm``'s matrices of one dataset: (G0, G1) for F; for K and B, the distance of H0 plus that of H1."""
    if arm == "F":
        return M[mat.mkey(task, "gram", 0, idx)], M[mat.mkey(task, "gram", 1, idx)]
    kind = _DISTANCES[arm]
    return M[mat.mkey(task, kind, 0, idx)] + M[mat.mkey(task, kind, 1, idx)]


def _fit_predict(
    arm: str, A: tuple | np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, float | tuple[int, int]]:
    """One cell of arm ``arm``, on its matrices ``A``: the test predictions, and h (K, B) or (K0, K1) (F)."""
    if arm == "F":
        pred, k0, k1 = est.fpcr_fit_predict(*A, y[train], train, test)
        return pred, (k0, k1)
    return est.nw_fit_predict(A[np.ix_(train, train)], y[train], A[np.ix_(test, train)])


def run_arm(arm: str, task: str, M: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Arm ``arm`` on every cell of ``task``, from the matrices and targets ``M``: its arrays of ``arms.npz``.

    ``M`` is what ``matrices.load_matrices`` gives. The arrays: ``{task}_pred_{arm}``, the test predictions;
    ``{task}_r2_{arm}``, their R² (``common.r2``), or for curvature ``curv_mse_{arm}``, their mean squared error; and
    each cell's choice, made on its training samples only: the bandwidth ``{task}_h_{arm}`` (K, B) or (K0, K1)
    ``{task}_k_F`` (F). ``ValueError`` on another arm or task.
    """
    if arm not in ARMS or task not in TASKS:
        raise ValueError(f"an arm is one of {ARMS} and a task one of {TASKS}, not {arm!r} and {task!r}")
    shape, n_test = _grid(task)
    pred, score = np.empty((*shape, n_test)), np.empty(shape)
    choice = np.empty((*shape, 2), dtype=int) if arm == "F" else np.empty(shape)
    dataset: dict = {}  # the current dataset's matrices and targets
    for at, idx, train, test in _cells(task, M):
        if idx not in dataset:
            dataset = {idx: (_arm_matrices(arm, task, M, idx), M[_ykey(task, idx)])}
        A, y = dataset[idx]
        pred[at], choice[at] = _fit_predict(arm, A, y, train, test)
        score[at] = np.mean((y[test] - pred[at]) ** 2) if task == "curv" else c.r2(y[test], pred[at])
    metric, chosen = "mse" if task == "curv" else "r2", "k" if arm == "F" else "h"
    return {f"{task}_pred_{arm}": pred, f"{task}_{metric}_{arm}": score, f"{task}_{chosen}_{arm}": choice}


def _shared(M: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The arrays of ``arms.npz`` that no arm owns: every cell's test targets, ``fluid_y`` and the fluid test sets."""
    return {
        "interp_y_test": np.array([M[_ykey("interp", idx)][test] for _, idx, _, test in _cells("interp", M)]),
        "curv_y_test": M["curv_y"][CURV_TEST],
        "fluid_y": M["fluid_y"],
        "fluid_test": np.array([inp.fluid_split(s)[0] for s in range(c.FLUID_SPLITS)]),
    }


def _same_cells(name: str, found: list, expected: list) -> None:
    """``ValueError`` unless upstream's ``name`` holds exactly the cells ``expected``, each once."""
    if sorted(found) != sorted(expected):
        raise ValueError(
            f"upstream's {name} holds {len(found)} cells, not exactly the {len(expected)} that G1 compares"
        )


def _g1_task(rows: list[dict], cells: dict) -> dict:
    """One task's part of G1, from its compared values, each a dict with ``rerun`` and ``reported``.

    A value fails if its |difference| is not at most ``G1_TOL``, a NaN included.
    """
    for row in rows:
        row["abs_diff"] = abs(row["rerun"] - row["reported"])
    failing = [row for row in rows if not row["abs_diff"] <= c.G1_TOL]
    worst = max(rows, key=lambda row: row["abs_diff"])
    return {
        "passed": not failing,
        "cells": cells,
        "compared": len(rows),
        "worst_abs_diff": worst["abs_diff"],
        "worst": worst,
        "failing": failing,
    }


def g1(k: dict[str, np.ndarray]) -> dict:
    """Gate G1: arm K's results ``k`` against upstream's published ones; it passes if every |difference| ≤ ``G1_TOL``.

    ``k`` holds ``run_arm("K", task, M)`` of the three tasks. The values compared:

    - interpolation: per training size, the mean and the SD (ddof 1) of ``interp_r2_K`` over its 10 runs, by pandas'
      ``groupby(...).agg(["mean", "std"])`` as upstream's, against ``PD_kernel_R2_mean`` and ``PD_kernel_R2_std`` of
      ``interpolation/results_interpolation.csv``;
    - curvature: ``curv_mse_K`` against the ``PD_kernel`` row of ``curvature/results_curvature.csv``;
    - fluid: ``fluid_r2_K`` at splits 0-4 against ``fluid/results_fluid_splits.csv``, keyed by
      ``(t, split, n_train)``.

    Per task, the record holds the cells compared, how many values, the worst |difference| and its cell, and every
    value above the tolerance. ``ValueError`` if a CSV does not hold exactly the cells compared.
    """
    runs = pd.DataFrame({"n_train": np.repeat(c.INTERP_SIZES, c.INTERP_RUNS), "R2": k["interp_r2_K"]})
    stats = runs.groupby("n_train")["R2"].agg(["mean", "std"])
    table = pd.read_csv(c.UPSTREAM / "interpolation" / "results_interpolation.csv")
    _same_cells("results_interpolation.csv", table["n_train"].tolist(), list(c.INTERP_SIZES))
    reported = table.set_index("n_train")
    interp = [
        {"n_train": n, "statistic": stat, "rerun": float(stats.loc[n, stat])}
        | {"reported": float(reported.loc[n, f"PD_kernel_R2_{stat}"])}
        for n in c.INTERP_SIZES
        for stat in ("mean", "std")
    ]
    table = pd.read_csv(c.UPSTREAM / "curvature" / "results_curvature.csv")
    mse = table.loc[table["method"] == "PD_kernel", "MSE"].tolist()
    _same_cells("results_curvature.csv", ["PD_kernel"] * len(mse), ["PD_kernel"])
    curv = [{"method": "PD_kernel", "rerun": float(k["curv_mse_K"]), "reported": float(mse[0])}]
    table = pd.read_csv(c.UPSTREAM / "fluid" / "results_fluid_splits.csv")
    found = [(float(t), int(s), int(n)) for t, s, n in table[["t", "split", "n_train"]].itertuples(index=False)]
    keys = [(t, s, n) for t in c.FLUID_TIMES for s in range(c.PAPER_SPLITS) for n in c.FLUID_SIZES]
    _same_cells("results_fluid_splits.csv", found, keys)
    reported = dict(zip(found, table["R2"].astype(float).tolist(), strict=True))
    fluid = [
        {"t": t, "split": s, "n_train": n, "rerun": float(k["fluid_r2_K"][ti, s, si]), "reported": reported[t, s, n]}
        for ti, t in enumerate(c.FLUID_TIMES)
        for s in range(c.PAPER_SPLITS)
        for si, n in enumerate(c.FLUID_SIZES)
    ]
    tasks = {
        "interp": _g1_task(interp, {"n_train": list(c.INTERP_SIZES), "statistic": ["mean", "std"]}),
        "curv": _g1_task(curv, {"method": ["PD_kernel"]}),
        "fluid": _g1_task(
            fluid, {"t": list(c.FLUID_TIMES), "split": list(range(c.PAPER_SPLITS)), "n_train": list(c.FLUID_SIZES)}
        ),
    }
    for task, part in tasks.items():
        print(
            f"G1 {task}: {part['compared']} values, worst |difference| {part['worst_abs_diff']:.2e}, "
            f"{len(part['failing'])} above {c.G1_TOL:g}: {'passed' if part['passed'] else 'FAILED'}",
            flush=True,
        )
    return {"passed": all(part["passed"] for part in tasks.values()), "tolerance": c.G1_TOL, **tasks}


def _relative_worst(S: np.ndarray, T: np.ndarray) -> float:
    """max |S − T| / max |T|: 0 if both are 0 or empty, inf if only the difference is not 0."""
    largest, diff = float(np.abs(T).max(initial=0.0)), float(np.abs(S - T).max(initial=0.0))
    return diff / largest if largest > 0 else (math.inf if diff > 0 else 0.0)


def _g3_measures(S: np.ndarray, T: np.ndarray) -> dict:
    """G3 in one cell and dimension: the projected test scores ``S`` against ``FPCA.transform``'s ``T``.

    The ruling after Task 4 (put to Simon): only the first min(``K_MAX``, rank) columns, the only ones arm F can use,
    are compared, and they pass if their worst |difference| is at most ``G3_TOL`` times their largest |score| in
    ``T``. ``transform`` misses in the near-null components, where it does not give the training curves their own
    fitted scores back either; the worst over all columns is kept for information. Ranks that differ fail.
    """
    rank, k = S.shape[1], min(c.K_MAX, S.shape[1])
    row = {"rank": rank, "rank_transform": T.shape[1], "compared_columns": k}
    if T.shape != S.shape:
        return row | {"passed": False, "worst_relative": math.inf, "worst_relative_all_columns": math.inf}
    worst = _relative_worst(S[:, :k], T[:, :k])
    return row | {
        "passed": worst <= c.G3_TOL,
        "worst_relative": worst,
        "worst_relative_all_columns": _relative_worst(S, T),
    }


def _g3_cells() -> Iterator[tuple[dict, int | None, list, float, np.ndarray, np.ndarray]]:
    """G3's cells, ``(label, dataset index, diagrams, shift, train, test)``, the diagrams rebuilt from ``inputs``."""
    datasets = inp.interpolation_datasets()
    for idx in G3_INTERP:
        n = datasets[idx]["n_train"]
        diagrams = [inp.rips_diagram(X) for X in datasets[idx]["clouds"]]
        yield {"task": "interp", "dataset": idx}, idx, diagrams, 0.0, np.arange(n), np.arange(n, n + c.INTERP_N_TEST)
    diagrams = [inp.rips_diagram(X) for X in inp.curvature_data()["clouds"]]
    yield {"task": "curv"}, None, diagrams, 0.0, CURV_TRAIN, CURV_TEST
    for ti, s, n in G3_FLUID:
        test, pool = inp.fluid_split(s)
        label = {"task": "fluid", "time": c.FLUID_TIMES[ti], "split": s, "n_train": n}
        yield label, ti, inp.fluid_diagrams(c.FLUID_TIMES[ti]), c.FLUID_SHIFT, pool[:n], test


def g3(M: dict[str, np.ndarray]) -> dict:
    """Gate G3: arm F's projected test scores against topocyte's own fit and transform, by ``_g3_measures``' rule.

    In each cell of ``_g3_cells`` (``G3_INTERP``, curvature, ``G3_FLUID``) and dimension: the test rows of
    ``betti.fpca_scores`` on the Gram matrix in ``M`` against
    ``tc.ph.FPCA(dim=h, eigen_solver="dense").fit(train curves).transform(test curves)``, the curves rebuilt from
    the diagrams of ``inputs``, shifted as the matrices' (``FLUID_SHIFT`` for fluid). It passes if every cell does.
    """
    rows = []
    for label, idx, diagrams, shift, train, test in _g3_cells():
        for h in (0, 1):
            curves = betti.curves([d[h] for d in diagrams], h, shift)
            S = betti.fpca_scores(M[mat.mkey(label["task"], "gram", h, idx)], train, test)[1]
            model = tc.ph.FPCA(dim=h, eigen_solver="dense").fit([curves[i] for i in train])
            row = label | {"h": h} | _g3_measures(S, model.transform([curves[i] for i in test]))
            rows.append(row)
            print(
                f"G3 {', '.join(f'{key} {value}' for key, value in label.items())}, H{h}: rank {row['rank']}, the "
                f"first {row['compared_columns']} columns {row['worst_relative']:.2e} of the largest score "
                f"({'passed' if row['passed'] else 'FAILED'}); all columns {row['worst_relative_all_columns']:.2e}",
                flush=True,
            )
    return {
        "passed": all(row["passed"] for row in rows),
        "rule": "per cell and dimension, the test scores' first min(K_MAX, rank) columns, the only ones arm F can "
        "use: max |fpca_scores - FPCA.transform| <= tolerance * max |FPCA.transform| over those columns, and equal "
        "ranks (the controller's ruling after Task 4, put to Simon); worst_relative_all_columns is information only",
        "tolerance": c.G3_TOL,
        "k_max": c.K_MAX,
        "worst_relative": {task: max(row["worst_relative"] for row in rows if row["task"] == task) for task in TASKS},
        "worst_relative_all_columns": {
            task: max(row["worst_relative_all_columns"] for row in rows if row["task"] == task) for task in TASKS
        },
        "cells": rows,
    }


def _run(arm: str, M: dict[str, np.ndarray], seconds: dict) -> dict[str, np.ndarray]:
    """``run_arm`` of ``arm`` on every task, each task's seconds into ``seconds[arm]``."""
    out, seconds[arm] = {}, {}
    for task in TASKS:
        started = time.perf_counter()
        out |= run_arm(arm, task, M)
        seconds[arm][task] = round(time.perf_counter() - started, 1)
        print(f"arm {arm}, {task}: {seconds[arm][task]} s", flush=True)
    return out


def _write(record: dict, started: float) -> None:
    """``results/arms.json``, with the stage's total seconds and the date."""
    record["seconds"]["total"] = round(time.perf_counter() - started, 1)
    record["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    c.RESULTS.mkdir(exist_ok=True)
    (c.RESULTS / "arms.json").write_text(json.dumps(record, indent=1) + "\n")


def _arms(provenance: dict) -> None:
    """The stage ``arms``: K, G1, G3, F, B, in that order (module docstring); exits with status 1 if a gate fails."""
    started = time.perf_counter()
    M = mat.load_matrices()
    seconds = {"load": round(time.perf_counter() - started, 1)}
    record = {
        "description": "arms K, F and B on the three tasks (design.md, Runs), with gate G1 (arm K against "
        "upstream's published results) after K, and gate G3 (the test scores projected from the Gram matrices "
        "against FPCA.transform) before F and B",
        "passed": False,
        "g1": None,
        "g3": None,
        "npz": None,
        "seconds": seconds,
        "date": None,
        "provenance": provenance,
    }
    results = {"K": _run("K", M, seconds)}
    for gate, check, argument in (("g1", g1, results["K"]), ("g3", g3, M)):
        part = time.perf_counter()
        record[gate] = check(argument)
        seconds[gate] = round(time.perf_counter() - part, 1)
        if not record[gate]["passed"]:
            _write(record, started)
            print(f"wrote results/arms.json: {gate.upper()} FAILED, so arms F and B did not run", flush=True)
            sys.exit(1)
    for arm in ("F", "B"):
        results[arm] = _run(arm, M, seconds)
    path = c.RESULTS / "arms.npz"
    arrays = _shared(M) | results["K"] | results["F"] | results["B"]
    np.savez(path, **arrays)
    record["npz"] = {
        "arrays": len(arrays),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    record["passed"] = True
    _write(record, started)
    print(f"wrote results/arms.json and results/arms.npz in {record['seconds']['total']} s: passed", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    c.check_topocyte()
    c.check_upstream()
    provenance = c.provenance(*(c.HERE / name for name in _SCRIPTS))  # at the start: the code that runs
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("stage", choices=("arms",))
    parser.parse_args(argv)
    _arms(provenance)


if __name__ == "__main__":
    main()
