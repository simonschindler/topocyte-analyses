#!/usr/bin/env python
"""Check the Betti FPCR benchmark's saved results by routes other than the code that made them.

Run from this folder after every stage, in the background (about 20 s; a peak RSS of 2.4 to 2.7 GB, from the
bar-overlap sums of the largest fluid diagrams and the fPCA of a fluid cell with 300 training curves)::

    caffeinate -is /usr/bin/time -l $PY verify_results.py > logs/verify.log 2>&1

It calls ``common.check_topocyte`` and ``common.check_upstream`` first. It then reads, read only: the stages' jsons in
``results/``; ``results/matrices.npz`` and ``results/arms.npz`` through ``matrices.load_matrices`` and
``run_study.load_arms``, which refuse an npz that its stage's json does not vouch for; upstream's four result CSVs,
with ``float_precision="round_trip"`` (pandas' default parser misreads about half of their values in the last bit);
and, through the study's loaders in ``inputs``, upstream's interpolation and curvature clouds and its fluid diagrams.
It runs no stage. Of the study's code it calls only ``common`` (the two checks, the constants and ``provenance``),
those loaders and their file paths, ``inputs.fluid_sim_ids`` (the order of the fluid files it hashes),
``matrices.mkey``, ``oracles``, and ``betti.fpca_scores`` for the fpcr check's information alone, to measure how far
its scores lie from arm F's. It writes ``results/verify.json``: every check with its family, its worst measured value,
its tolerance and its details; a summary per family; ``provenance`` (the md5 of this script and of every study module
it imports, and the versions); ``sha256``, the sha256 of every other data file it read, keyed by its path in this
folder; ``fluid_diagrams_sha256``, for each fluid time, one sha256 of its 400 diagram pickles read one after the other
in ``fluid_sim_ids()`` order; the seeds; the seconds; and the peak RSS. It exits with status 1 if a check fails.

The checks, each by its own route:

- **w2:** 30 entries per task of the interpolation and curvature W2 matrices, drawn with ``W2_SEED``. The two Rips
  diagrams come from ``ripser.ripser``, upstream's own call, and W2 from the transport LP of the diagrams augmented
  with the diagonal, the (n + 1) × (m + 1) problem that gudhi hands to POT, solved by ``scipy.optimize.linprog``
  (HiGHS's dual simplex) rather than by ``linear_sum_assignment``; relative ``TOL_W2``. The fluid W2 matrices are
  upstream's own distances, which gate G2 checked.
- **gram:** 30 entries per task (25 off the diagonal, 5 on it; ``GRAM_SEED``) as ``oracles.overlap``, the sum of
  the bar overlaps, of the two diagrams, the fluid ones shifted by ``FLUID_SHIFT`` as the matrices' curves are;
  relative ``TOL_GRAM``. **l2:** the 25 off the diagonal as √(⟨f, f⟩ + ⟨g, g⟩ − 2⟨f, g⟩) from the same sums;
  relative ``TOL_L2``.
- **arrays:** ``arms.npz`` holds exactly its 31 arrays in their shapes, every prediction finite; its targets equal
  ``matrices.npz``'s; each fluid split's test set is upstream's recipe, ``default_rng(s).permutation(400)[:100]``,
  written out here. **scores:** every saved R² and MSE of every arm equals its recomputation from the saved
  predictions with ``math.fsum``, to ``TOL_SCORE`` × max(1, |value|).
- **g1:** gate G1 again: arm K's R² and MSE recomputed from its predictions in ``arms.npz``, and interpolation's
  mean and SD (ddof 1) over the runs of each size, against upstream's CSVs read exactly; absolute ``G1_TOL``.
- **nw:** 5 cells per task and arm (curvature has one), K and B, drawn with ``NW_SEED``: ``oracles.upstream_nw``, a
  loop transcription of upstream's ``kernel_estimator``, on the saved matrices gives the saved bandwidth exactly and
  the saved predictions to ``TOL_NW`` × (1 + the cell's largest |prediction|).
- **fpcr:** 5 cells per task (curvature has one) and one interpolation cell with ``LOO_N_TRAIN`` training samples,
  drawn with ``FPCR_SEED``. topocyte's ``FPCA(dim, eigen_solver="dense")``, fitted on the cell's training curves
  rebuilt from the diagrams, scores its test curves with ``transform``, which integrates the component functions over
  the bars rather than projecting rows of the saved Gram matrix; scikit-learn's ``LinearRegression`` on the first
  (K0, K1) scores that arm F chose predicts the test targets, to ``TOL_FPCR`` × (1 + the cell's largest
  |prediction|). For information: the condition number of each cell's design [1, scores], and the largest
  difference between its scores and arm F's, relative to the largest of arm F's.
- **loo:** in the cell with ``LOO_N_TRAIN`` training samples, (K0, K1) is chosen again among the candidates of
  ``design.md``, each candidate's LOO error by refitting the regression without each sample in turn and its
  eligibility computed anew (rank by ``scipy.linalg.svdvals``, leverages by a QR factorization); it must equal arm
  F's choice.
- **readouts:** every interval in ``readouts.json`` (the primary pair, the exploratory pairs and the descriptive
  readouts) from a bootstrap written here with ``design.md``'s recipe: Δ from ``arms.npz``, a fresh
  ``default_rng(BOOT_SEED)`` for each interval, ``N_BOOT`` resamples of the units, the 2.5 and 97.5 percentiles;
  equal to the bit. **verdicts:** each primary verdict again from its recomputed interval by the rule fixed in
  advance, and no verdict elsewhere.
- **tables:** the reported values in ``readouts.json``'s tables equal upstream's CSVs to the bit; the arms' means and
  SDs recomputed with ``math.fsum`` to ``TOL_SCORE``, and the chosen (K0, K1) counted again.
- **provenance:** each stage's json is unchanged since the commit that last changed it, and every md5 it records is
  that of the script in that commit (``git show``, read only); its pins are the study's and its gates passed; the
  two npz files are the ones the jsons vouch for.

Not redone: the Rips diagrams (gate G5 compared them with ``ripser.ripser``'s), gates G2 to G4, the matrices' other
entries, the arms' other cells, the LOO choice of the other cells, and the runtimes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import ripser
import scipy.linalg
import sklearn
import topocyte as tc
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression

import betti
import common as c
import inputs as inp
import matrices as mat
import oracles
import run_study as rs

W2_SEED = 5  # draws the W2 entries
FPCR_SEED = 6  # draws the FPCR cells and the LOO cell
GRAM_SEED = 7  # draws the Gram and L2 entries
NW_SEED = 8  # draws the NW cells
N_W2 = 30  # W2 entries per task, interpolation and curvature
N_OFF, N_DIAG = 25, 5  # Gram entries per task, off and on the diagonal; the L2 entries are those off it
N_CELLS = 5  # NW and FPCR cells per task; curvature has one cell
LOO_N_TRAIN = 10  # the training size of the cell whose (K0, K1) is chosen again by refitting
TOL_W2 = 1e-9  # relative
TOL_GRAM = 1e-12  # relative
TOL_L2 = 1e-9  # relative: from the sums, ⟨f, f⟩ + ⟨g, g⟩ cancels against 2⟨f, g⟩
TOL_NW = 1e-12  # times 1 + the cell's largest |prediction|; the bandwidth must be the same
TOL_FPCR = 1e-8  # times 1 + the cell's largest |prediction|
TOL_SCORE = 1e-12  # times max(1, |value|): an R², MSE, mean or SD recomputed with math.fsum
SIGMA_RTOL = 1e-12  # ruling 4 in progress.md: a design with σ_min ≤ SIGMA_RTOL σ_max is not eligible
TASKS = ("interp", "curv", "fluid")
ARMS = ("K", "F", "B")
FLUID_GRID = (len(c.FLUID_TIMES), c.FLUID_SPLITS, len(c.FLUID_SIZES))  # a fluid cell: (time, split, size) indices
N_FLUID = 400  # the fluid simulations
PERCENTILES = (2.5, 97.5)
CSV = {
    "interp": c.UPSTREAM / "interpolation" / "results_interpolation.csv",
    "curv": c.UPSTREAM / "curvature" / "results_curvature.csv",
    "fluid_splits": c.UPSTREAM / "fluid" / "results_fluid_splits.csv",
    "fluid": c.UPSTREAM / "fluid" / "results_fluid.csv",
}
STAGE_JSONS = ("upstream.json", "inputs.json", "matrices.json", "arms.json", "readouts.json")
# Every study module this script imports, directly or through another: its code is what runs.
_SCRIPTS = (
    "verify_results.py",
    "oracles.py",
    "common.py",
    "inputs.py",
    "safe_pickle.py",
    "matrices.py",
    "wasserstein.py",
    "betti.py",
    "run_study.py",
    "estimators.py",
    "readouts.py",
)

checks: list[dict] = []
read_files: list[Path] = []  # the data files read, for their sha256


def plain(value):
    """``value`` with numpy scalars and arrays turned into Python's, for the json."""
    if isinstance(value, dict):
        return {str(key): plain(v) for key, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def record(family: str, name: str, passed: bool, worst, tolerance, measured: dict) -> None:
    """One check: its family, what it checks, whether it passed, its worst measured value and the tolerance."""
    row = {"family": family, "check": name, "passed": bool(passed), "worst": worst, "tolerance": tolerance}
    checks.append(plain(row | {"measured": measured}))
    print(
        f"{'ok  ' if passed else 'FAIL'} [{family}] {name}: worst {worst!r} (tolerance {tolerance!r}); "
        f"{json.dumps(plain(measured), ensure_ascii=False)}",
        flush=True,
    )


def relative(a: float, b: float) -> float:
    """|a − b| / |b|: 0 where a equals b, also where both are 0; inf where only b is 0."""
    diff = abs(a - b)
    if diff == 0:
        return 0.0
    return math.inf if b == 0 else diff / abs(b)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    """A read-only git command, run in this folder."""
    return subprocess.run(["git", "-C", str(c.HERE), *args], capture_output=True, check=False)


def read_csv(name: str) -> pd.DataFrame:
    """Upstream's result CSV ``name`` (a key of ``CSV``), each value read as Python's ``float`` reads its text."""
    read_files.append(CSV[name])
    return pd.read_csv(CSV[name], float_precision="round_trip")


def fsum_r2(y: np.ndarray, p: np.ndarray) -> float:
    """1 − SSE/SST of predictions ``p`` of ``y``, SST about the mean of ``y``, every sum by ``math.fsum``."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    centre = math.fsum(y.tolist()) / y.size
    return 1.0 - math.fsum(((y - p) ** 2).tolist()) / math.fsum(((y - centre) ** 2).tolist())


def fsum_mse(y: np.ndarray, p: np.ndarray) -> float:
    return math.fsum(((np.asarray(y, dtype=float) - p) ** 2).tolist()) / len(y)


def fsum_mean_sd(values) -> tuple[float, float]:
    """The mean of ``values`` and their SD with ddof 1, every sum by ``math.fsum``."""
    values = [float(v) for v in values]
    mean = math.fsum(values) / len(values)
    return mean, math.sqrt(math.fsum([(v - mean) ** 2 for v in values]) / (len(values) - 1))


def score_gap(mine: float, saved: float) -> float:
    """|mine − saved| / max(1, |saved|): ``TOL_SCORE``'s measure."""
    return abs(mine - saved) / max(1.0, abs(saved))


# ---------------------------------------------------------------- the data and the cells


class Data:
    """What the checks read: the saved matrices and arms, the clouds and the fluid diagrams, each loaded once."""

    def __init__(self, M: dict[str, np.ndarray], arms: dict[str, np.ndarray]) -> None:
        self.M, self.arms = M, arms
        self.interp = inp.interpolation_datasets()
        self.curv = inp.curvature_data()
        self.fluid = [inp.fluid_diagrams(t) for t in c.FLUID_TIMES]
        read_files.extend(
            [inp._INTERPOLATION, *(inp._CURVATURE / name for name in ("point_clouds.pkl", "labels.pkl"))]
            + [inp._CURVATURE / name for name in ("train_indices.pkl", "test_indices.pkl")]
        )
        self._rips: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def size(self, task: str, idx: int | None) -> int:
        """The number of samples of a dataset: an interpolation dataset's clouds, 100 curvature clouds, 400 fluid."""
        if task == "interp":
            return len(self.interp[idx]["clouds"])
        return len(self.curv["clouds"]) if task == "curv" else N_FLUID

    def diagram(self, task: str, idx: int | None, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(H0, H1) of sample ``i``: for a cloud, ``ripser.ripser(X)["dgms"]``, upstream's call; for fluid, upstream's
        diagram from ``inputs.fluid_diagrams``. Unshifted."""
        if task == "fluid":
            return self.fluid[idx][i]
        key = (task, idx, i)
        if key not in self._rips:
            X = self.interp[idx]["clouds"][i] if task == "interp" else self.curv["clouds"][i]
            h0, h1 = ripser.ripser(X)["dgms"]
            self._rips[key] = (np.asarray(h0, dtype=float), np.asarray(h1, dtype=float))
        return self._rips[key]


def draw_cells(rng: np.random.Generator, task: str) -> list[tuple[int, ...]]:
    """``N_CELLS`` distinct cells of ``task``: interpolation datasets ``(idx,)`` or fluid ``(time, split, size)``
    indices; curvature's one cell, ``()``, draws nothing."""
    if task == "interp":
        return [(int(i),) for i in rng.choice(len(c.INTERP_SIZES) * c.INTERP_RUNS, N_CELLS, replace=False)]
    if task == "curv":
        return [()]
    flat = rng.choice(math.prod(FLUID_GRID), N_CELLS, replace=False)
    return [tuple(int(v) for v in np.unravel_index(int(k), FLUID_GRID)) for k in flat]


def cell_split(
    M: dict[str, np.ndarray], task: str, cell: tuple[int, ...]
) -> tuple[int | None, np.ndarray, np.ndarray, np.ndarray, dict]:
    """A cell's dataset index, targets, training and test positions, and label.

    Interpolation: the dataset's first n samples train and its last 50 test. Curvature: clouds 0-49 train, 50-99
    test. Fluid split s, by upstream's recipe written out here: ``perm = default_rng(s).permutation(400)``, the test
    set ``perm[:100]`` and the training set the next n.
    """
    if task == "interp":
        (idx,) = cell
        y = M[f"interp_{idx:02d}_y"]
        n = y.size - c.INTERP_N_TEST
        return idx, y, np.arange(n), np.arange(n, y.size), {"dataset": idx, "n_train": n}
    if task == "curv":
        return None, M["curv_y"], np.arange(50), np.arange(50, 100), {}
    ti, s, si = cell
    perm = np.random.default_rng(s).permutation(N_FLUID)
    n = c.FLUID_SIZES[si]
    train, test = perm[c.FLUID_N_TEST : c.FLUID_N_TEST + n], perm[: c.FLUID_N_TEST]
    return ti, M["fluid_y"], train, test, {"t": c.FLUID_TIMES[ti], "split": s, "n_train": n}


# ---------------------------------------------------------------- W2, Gram and L2 entries


def w2_lp(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """W2 between two diagrams of one dimension, (n, 2) each, from a transport LP solved by HiGHS; and its status.

    The finite bars: the transport problem between the diagrams augmented with the diagonal, as gudhi poses it for
    POT. Its n + 1 sources are ``a``'s n finite points, of mass 1 each, and the diagonal, of mass m; its m + 1 sinks
    are ``b``'s m finite points and the diagonal, of mass n. A point goes to a point at their squared Euclidean
    distance and to the diagonal at its own squared distance to it, (d − b)² / 2; the diagonal goes to itself for
    nothing. Its optimum, Σ cost × flow over the solution by ``math.fsum``, is the finite cost. The essential bars:
    matched by sorted birth, at the squared differences. W2 = √(finite cost + essential cost); ``inf`` if the
    numbers of essential bars differ, and NaN if the LP does not end optimal (status 0).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    fa, fb = a[np.isfinite(a[:, 1])], b[np.isfinite(b[:, 1])]
    ea, eb = np.sort(a[np.isinf(a[:, 1]), 0]), np.sort(b[np.isinf(b[:, 1]), 0])
    if ea.size != eb.size:
        return math.inf, 0
    n, m = len(fa), len(fb)
    cost = np.zeros((n + 1, m + 1))
    cost[:n, :m] = (fa[:, None, 0] - fb[None, :, 0]) ** 2 + (fa[:, None, 1] - fb[None, :, 1]) ** 2
    cost[:n, m] = (fa[:, 1] - fa[:, 0]) ** 2 / 2
    cost[n, :m] = (fb[:, 1] - fb[:, 0]) ** 2 / 2
    # the flow x[i, j], flattened row by row: each row sums to its source's mass, each column to its sink's
    rows = np.kron(np.eye(n + 1), np.ones((1, m + 1)))
    columns = np.kron(np.ones((1, n + 1)), np.eye(m + 1))
    masses = np.concatenate((np.ones(n), [m], np.ones(m), [n]))
    result = linprog(
        cost.ravel(),
        A_eq=np.vstack((rows, columns)),
        b_eq=masses,
        bounds=(0, None),
        method="highs-ds",
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    if result.status != 0:
        return math.nan, int(result.status)
    finite = math.fsum((cost.ravel() * result.x).tolist())
    return math.sqrt(finite + math.fsum(((ea - eb) ** 2).tolist())), 0


def check_w2(data: Data) -> None:
    rng = np.random.default_rng(W2_SEED)
    rows = []
    for task in ("interp", "curv"):
        for _ in range(N_W2):
            idx = int(rng.integers(len(data.interp))) if task == "interp" else None
            h = int(rng.integers(2))
            p, q = sorted(int(v) for v in rng.choice(data.size(task, idx), 2, replace=False))
            a, b = data.diagram(task, idx, p)[h], data.diagram(task, idx, q)[h]
            lp, status = w2_lp(a, b)
            saved = float(data.M[mat.mkey(task, "w2", h, idx)][p, q])
            finite_bars = [int(np.isfinite(d[:, 1]).sum()) for d in (a, b)]
            rows.append(
                {"task": task, "dataset": idx, "h": h, "p": p, "q": q, "finite_bars": finite_bars, "status": status}
                | {"lp": lp, "saved": saved, "relative": relative(saved, lp)}
            )
    worst = max(row["relative"] for row in rows)
    record(
        "w2",
        f"{N_W2} entries per task of the interpolation and curvature W2 matrices (default_rng({W2_SEED})) against the "
        "optimum of the transport LP of the diagrams augmented with the diagonal, by scipy.optimize.linprog (HiGHS "
        "dual simplex), on ripser.ripser's diagrams (|saved - LP| / LP)",
        all(row["status"] == 0 for row in rows) and worst <= TOL_W2,
        worst,
        TOL_W2,
        {
            "entries": {task: sum(row["task"] == task for row in rows) for task in ("interp", "curv")},
            "worst_relative": {
                task: max(r["relative"] for r in rows if r["task"] == task) for task in ("interp", "curv")
            },
            "lp_not_optimal": sum(row["status"] != 0 for row in rows),
            "entries_zero": sum(row["lp"] == 0 for row in rows),
            "largest_n_plus_m": max(sum(row["finite_bars"]) for row in rows),
            "worst_entry": max(rows, key=lambda row: row["relative"]),
        },
    )


def check_gram_l2(data: Data) -> None:
    rng = np.random.default_rng(GRAM_SEED)
    gram_rows, l2_rows = [], []
    for task in TASKS:
        for k in range(N_OFF + N_DIAG):
            if task == "interp":
                idx = int(rng.integers(len(data.interp)))
            else:
                idx = None if task == "curv" else int(rng.integers(len(c.FLUID_TIMES)))
            n, h = data.size(task, idx), int(rng.integers(2))
            if k < N_OFF:
                p, q = (int(v) for v in rng.choice(n, 2, replace=False))
            else:
                p = q = int(rng.integers(n))
            shift = c.FLUID_SHIFT if task == "fluid" else 0.0  # inf + shift stays inf
            a, b = data.diagram(task, idx, p)[h] + shift, data.diagram(task, idx, q)[h] + shift
            ab = oracles.overlap(a, b)
            saved = float(data.M[mat.mkey(task, "gram", h, idx)][p, q])
            cell = {"task": task, "dataset": idx, "h": h, "p": p, "q": q}
            gram_rows.append(cell | {"overlap": ab, "saved": saved, "relative": relative(saved, ab)})
            if p != q:
                l2 = math.sqrt(max(0.0, math.fsum([oracles.overlap(a, a), oracles.overlap(b, b), -2.0 * ab])))
                saved = float(data.M[mat.mkey(task, "l2", h, idx)][p, q])
                l2_rows.append(cell | {"from_sums": l2, "saved": saved, "relative": relative(saved, l2)})
    for family, rows, tolerance, what in (
        (
            "gram",
            gram_rows,
            TOL_GRAM,
            f"{N_OFF + N_DIAG} Gram entries per task ({N_OFF} off the diagonal, {N_DIAG} on it; "
            f"default_rng({GRAM_SEED})) against oracles.overlap, the sum of the bar overlaps by math.fsum, of the two "
            "diagrams (fluid's shifted by FLUID_SHIFT, Rips' not) (|saved - sum| / sum)",
        ),
        (
            "l2",
            l2_rows,
            TOL_L2,
            f"the {N_OFF} L2 entries per task off the diagonal, from the same sums: "
            "sqrt(<f, f> + <g, g> - 2 <f, g>) (|saved - from sums| / from sums)",
        ),
    ):
        worst = max(row["relative"] for row in rows)
        record(
            family,
            what,
            worst <= tolerance,
            worst,
            tolerance,
            {
                "entries": {task: sum(row["task"] == task for row in rows) for task in TASKS},
                "worst_relative": {task: max(r["relative"] for r in rows if r["task"] == task) for task in TASKS},
                "entries_zero": sum(row["saved"] == 0 for row in rows),
                "worst_entry": max(rows, key=lambda row: row["relative"]),
            },
        )


# ---------------------------------------------------------------- the saved arrays, their scores and G1


def expected_arrays() -> dict[str, tuple[int, ...]]:
    """The shape of every array of ``arms.npz`` (``run_study``'s module docstring)."""
    n_interp = len(c.INTERP_SIZES) * c.INTERP_RUNS
    shapes = {
        "interp_y_test": (n_interp, c.INTERP_N_TEST),
        "curv_y_test": (50,),
        "fluid_y": (N_FLUID,),
        "fluid_test": (c.FLUID_SPLITS, c.FLUID_N_TEST),
    }
    for arm in ARMS:
        choice = ("k", (2,)) if arm == "F" else ("h", ())
        shapes |= {
            f"interp_pred_{arm}": (n_interp, c.INTERP_N_TEST),
            f"interp_r2_{arm}": (n_interp,),
            f"interp_{choice[0]}_{arm}": (n_interp, *choice[1]),
            f"curv_pred_{arm}": (50,),
            f"curv_mse_{arm}": (),
            f"curv_{choice[0]}_{arm}": choice[1],
            f"fluid_pred_{arm}": (*FLUID_GRID, c.FLUID_N_TEST),
            f"fluid_r2_{arm}": FLUID_GRID,
            f"fluid_{choice[0]}_{arm}": (*FLUID_GRID, *choice[1]),
        }
    return shapes


def check_arrays(data: Data) -> None:
    M, arms = data.M, data.arms
    shapes = expected_arrays()
    wrong_shapes = sorted(key for key in shapes if key in arms and arms[key].shape != shapes[key])
    n_train = [M[f"interp_{idx:02d}_y"].size - c.INTERP_N_TEST for idx in range(len(data.interp))]
    differing = {
        "keys_missing_or_extra": sorted(set(shapes) ^ set(arms)),
        "shapes": wrong_shapes,
        "predictions_not_finite": [
            key for key in shapes if "_pred_" in key and key in arms and not np.isfinite(arms[key]).all()
        ],
        "interp_sizes": n_train != list(np.repeat(c.INTERP_SIZES, c.INTERP_RUNS)),
        "interp_y_test": [
            idx
            for idx in range(len(n_train))
            if not np.array_equal(arms["interp_y_test"][idx], M[f"interp_{idx:02d}_y"][n_train[idx] :])
        ],
        "curv_y_test": not np.array_equal(arms["curv_y_test"], M["curv_y"][50:]),
        "fluid_y": not np.array_equal(arms["fluid_y"], M["fluid_y"]),
        "fluid_test_splits": [
            s
            for s in range(c.FLUID_SPLITS)
            if not np.array_equal(
                arms["fluid_test"][s], np.random.default_rng(s).permutation(N_FLUID)[: c.FLUID_N_TEST]
            )
        ],
    }
    problems = sum(len(v) if isinstance(v, list) else int(v) for v in differing.values())
    record(
        "arrays",
        "arms.npz holds exactly its 31 arrays in their shapes, every prediction finite; the interpolation sizes come "
        "in the order of the readouts' reshape; its test targets equal matrices.npz's; each fluid split's test set is "
        "default_rng(s).permutation(400)[:100] (how many differ)",
        problems == 0,
        problems,
        0,
        differing | {"arrays": len(arms)},
    )
    gaps = {}
    for arm in ARMS:
        interp = [
            score_gap(
                fsum_r2(arms["interp_y_test"][idx], arms[f"interp_pred_{arm}"][idx]), arms[f"interp_r2_{arm}"][idx]
            )
            for idx in range(len(n_train))
        ]
        curv = [score_gap(fsum_mse(arms["curv_y_test"], arms[f"curv_pred_{arm}"]), float(arms[f"curv_mse_{arm}"]))]
        fluid = []
        for ti, s, si in itertools.product(*map(range, FLUID_GRID)):
            test = arms["fluid_test"][s]
            mine = fsum_r2(arms["fluid_y"][test], arms[f"fluid_pred_{arm}"][ti, s, si])
            fluid.append(score_gap(mine, float(arms[f"fluid_r2_{arm}"][ti, s, si])))
        gaps[arm] = {
            "interp": max(interp),
            "curv": max(curv),
            "fluid": max(fluid),
            "values": len(interp + curv + fluid),
        }
    worst = max(gap[task] for gap in gaps.values() for task in TASKS)
    record(
        "scores",
        "every saved R2 (interpolation, fluid) and MSE (curvature) of arms K, F and B against its recomputation from "
        "the saved predictions and test targets by math.fsum (|saved - recomputed| / max(1, |saved|))",
        worst <= TOL_SCORE,
        worst,
        TOL_SCORE,
        gaps,
    )


def check_g1(data: Data) -> None:
    M, arms = data.M, data.arms
    rows = []
    table = read_csv("interp")
    sizes_ok = sorted(table["n_train"].tolist()) == list(c.INTERP_SIZES)
    reported = table.set_index("n_train")
    r2_k = {}
    for idx in range(len(data.interp)):
        n = M[f"interp_{idx:02d}_y"].size - c.INTERP_N_TEST
        r2_k.setdefault(n, []).append(fsum_r2(arms["interp_y_test"][idx], arms["interp_pred_K"][idx]))
    for n in c.INTERP_SIZES:
        for stat, value in zip(("mean", "std"), fsum_mean_sd(r2_k[n]), strict=True):
            theirs = float(reported.loc[n, f"PD_kernel_R2_{stat}"])
            rows.append({"task": "interp", "n_train": n, "statistic": stat, "rerun": value, "reported": theirs})
    table = read_csv("curv")
    mse = table.loc[table["method"] == "PD_kernel", "MSE"].tolist()
    rows.append(
        {"task": "curv", "method": "PD_kernel", "rerun": fsum_mse(arms["curv_y_test"], arms["curv_pred_K"])}
        | {"reported": float(mse[0]) if len(mse) == 1 else math.nan}
    )
    table = read_csv("fluid_splits")
    found = [(float(t), int(s), int(n)) for t, s, n in table[["t", "split", "n_train"]].itertuples(index=False)]
    keys = [(t, s, n) for t in c.FLUID_TIMES for s in range(c.PAPER_SPLITS) for n in c.FLUID_SIZES]
    reported = dict(zip(found, table["R2"].tolist(), strict=True))
    for ti, s, si in itertools.product(range(len(c.FLUID_TIMES)), range(c.PAPER_SPLITS), range(len(c.FLUID_SIZES))):
        test = np.random.default_rng(s).permutation(N_FLUID)[: c.FLUID_N_TEST]
        t, n = c.FLUID_TIMES[ti], c.FLUID_SIZES[si]
        mine = fsum_r2(M["fluid_y"][test], arms["fluid_pred_K"][ti, s, si])
        rows.append(
            {
                "task": "fluid",
                "t": t,
                "split": s,
                "n_train": n,
                "rerun": mine,
                "reported": reported.get((t, s, n), math.nan),
            }
        )
    for row in rows:
        row["abs_diff"] = abs(row["rerun"] - row["reported"])
    cells_ok = sizes_ok and len(mse) == 1 and sorted(found) == sorted(keys)
    worst = max(rows, key=lambda row: math.inf if math.isnan(row["abs_diff"]) else row["abs_diff"])
    record(
        "g1",
        "gate G1 again from arms.npz: arm K's R2 and MSE recomputed from its predictions by math.fsum, "
        "interpolation's mean and SD (ddof 1) over each size's runs, against upstream's results_interpolation.csv, "
        "results_curvature.csv and results_fluid_splits.csv (splits 0-4), read exactly (|rerun - reported|)",
        cells_ok and all(row["abs_diff"] <= c.G1_TOL for row in rows),
        worst["abs_diff"],
        c.G1_TOL,
        {
            "csv_cells_as_expected": cells_ok,
            "compared": {task: sum(row["task"] == task for row in rows) for task in TASKS},
            "worst_abs_diff": {task: max(r["abs_diff"] for r in rows if r["task"] == task) for task in TASKS},
            "worst": worst,
        },
    )


# ---------------------------------------------------------------- the estimators: NW, FPCR and the LOO choice


def check_nw(data: Data) -> None:
    rng = np.random.default_rng(NW_SEED)
    rows = []
    for arm, kind in (("K", "w2"), ("B", "l2")):
        for task in TASKS:
            for cell in draw_cells(rng, task):
                idx, y, train, test, label = cell_split(data.M, task, cell)
                D = data.M[mat.mkey(task, kind, 0, idx)] + data.M[mat.mkey(task, kind, 1, idx)]
                pred, h = oracles.upstream_nw(D[np.ix_(train, train)], y[train], D[np.ix_(test, train)])
                saved, saved_h = data.arms[f"{task}_pred_{arm}"][cell], float(data.arms[f"{task}_h_{arm}"][cell])
                scale = 1.0 + float(np.abs(saved).max())
                rows.append(
                    {"arm": arm, "task": task}
                    | label
                    | {"h": h, "same_h": h == saved_h, "max_abs_diff": float(np.abs(pred - saved).max())}
                    | {"tolerance": TOL_NW * scale, "measure": float(np.abs(pred - saved).max()) / scale}
                )
    worst = max(row["measure"] for row in rows)
    record(
        "nw",
        f"{N_CELLS} cells per task and arm (curvature has one; default_rng({NW_SEED})), arms K and B: "
        "oracles.upstream_nw, a loop transcription of upstream's kernel_estimator, on the saved matrices gives the "
        "saved bandwidth exactly and the saved predictions (max |difference| / (1 + max |prediction|))",
        all(row["same_h"] for row in rows) and worst <= TOL_NW,
        worst,
        TOL_NW,
        {"cells": len(rows), "bandwidths_differing": sum(not row["same_h"] for row in rows), "per_cell": rows},
    )


def fpca_route(data: Data, task: str, idx: int | None, train: np.ndarray, test: np.ndarray):
    """Per dimension, topocyte's dense FPCA fitted on the training curves: the training scores of the fit, the test
    scores of ``transform``. The curves are the diagrams' finite bars, shifted by ``FLUID_SHIFT`` for fluid."""
    shift = c.FLUID_SHIFT if task == "fluid" else 0.0
    fitted, projected = [], []
    for h in (0, 1):
        curves = {}
        for i in np.concatenate((train, test)).tolist():
            bars = data.diagram(task, idx, i)[h]
            curves[i] = tc.ph.BettiCurve(bars[np.isfinite(bars[:, 1])] + shift, dim=h)
        model = tc.ph.FPCA(dim=h, eigen_solver="dense").fit([curves[i] for i in train.tolist()])
        fitted.append(model.result_.scores)
        projected.append(model.transform([curves[i] for i in test.tolist()]))
    return fitted, projected


def ols_predict(S: list[np.ndarray], T: list[np.ndarray], k: tuple[int, int], y: np.ndarray) -> np.ndarray:
    """scikit-learn's OLS, with its intercept, of ``y`` on the first k0 H0 and k1 H1 training scores, at the test
    scores; with no score, the training mean."""
    X = np.column_stack((S[0][:, : k[0]], S[1][:, : k[1]]))
    if X.shape[1] == 0:
        return np.full(len(T[0]), y.mean())
    return LinearRegression().fit(X, y).predict(np.column_stack((T[0][:, : k[0]], T[1][:, : k[1]])))


def score_discrepancy(S, T, G: tuple[np.ndarray, np.ndarray], train, test, k) -> float:
    """For information: the largest difference between the fPCA route's scores and arm F's (``betti.fpca_scores`` on
    the saved Gram matrices) in the first k_d columns of each dimension, signs aligned per column, relative to the
    largest of arm F's scores there."""
    worst = 0.0
    for h in (0, 1):
        if k[h] == 0:
            continue
        S_gram, T_gram = (scores[:, : k[h]] for scores in betti.fpca_scores(G[h], train, test))
        sign = np.where(np.sum(S[h][:, : k[h]] * S_gram, axis=0) < 0, -1.0, 1.0)
        diff = max(np.abs(S[h][:, : k[h]] - sign * S_gram).max(), np.abs(T[h][:, : k[h]] - sign * T_gram).max())
        worst = max(worst, float(diff / max(np.abs(S_gram).max(), np.abs(T_gram).max())))
    return worst


def check_fpcr(data: Data) -> tuple[list[np.ndarray], np.ndarray, tuple[int, int], dict]:
    """The fpcr check; returns the LOO cell's training scores, targets, arm F's choice and label for ``check_loo``."""
    rng = np.random.default_rng(FPCR_SEED)
    cells = [(task, cell) for task in TASKS for cell in draw_cells(rng, task)]
    small = [idx for idx in range(len(data.interp)) if data.interp[idx]["n_train"] == LOO_N_TRAIN]
    loo_idx = int(rng.choice([idx for idx in small if ("interp", (idx,)) not in cells]))
    cells.append(("interp", (loo_idx,)))
    rows, loo = [], None
    for task, cell in cells:
        started = time.perf_counter()
        idx, y, train, test, label = cell_split(data.M, task, cell)
        k = tuple(int(v) for v in data.arms[f"{task}_k_F"][cell])
        S, T = fpca_route(data, task, idx, train, test)
        components = [s.shape[1] for s in S]
        saved = data.arms[f"{task}_pred_F"][cell]
        scale = 1.0 + float(np.abs(saved).max())
        fits = k[0] <= components[0] and k[1] <= components[1]  # else arm F chose more components than this fit has
        diff = float(np.abs(ols_predict(S, T, k, y[train]) - saved).max()) if fits else math.inf
        design = np.column_stack((np.ones(len(train)), S[0][:, : k[0]], S[1][:, : k[1]]))
        G = tuple(data.M[mat.mkey(task, "gram", h, idx)] for h in (0, 1))
        rows.append(
            {"task": task}
            | label
            | {"k": list(k), "components": components, "max_abs_diff": diff, "tolerance": TOL_FPCR * scale}
            | {"measure": diff / scale, "design_condition": float(np.linalg.cond(design))}
            | {"score_discrepancy": score_discrepancy(S, T, G, train, test, k) if fits else None}
            | {"seconds": round(time.perf_counter() - started, 1)}
        )
        if cell == (loo_idx,) and task == "interp":
            loo = (S, y[train], k, label)
    worst = max(row["measure"] for row in rows)
    record(
        "fpcr",
        f"{N_CELLS} cells per task (curvature has one) and one interpolation cell with {LOO_N_TRAIN} training samples "
        f"(default_rng({FPCR_SEED})): topocyte's FPCA(dim, eigen_solver='dense') fitted on the training curves and its "
        "transform of the test curves, then scikit-learn's LinearRegression on the first (K0, K1) scores arm F chose, "
        "against arm F's saved predictions (max |difference| / (1 + max |prediction|)); the condition number of the "
        "design [1, scores] and the scores' largest difference from arm F's (betti.fpca_scores, relative to its "
        "largest score) are information",
        worst <= TOL_FPCR,
        worst,
        TOL_FPCR,
        {"cells": len(rows), "per_cell": rows},
    )
    return loo


def check_loo(loo: tuple) -> None:
    """(K0, K1) chosen again in one cell: every candidate's LOO error by refitting without each sample in turn."""
    (S0, S1), y, saved, label = loo
    n = len(y)
    # design.md's candidates: 0 <= K_d <= min(K_MAX, components), K0 + K1 <= n - 2; ties to the smaller K0 + K1, then K0
    pairs = itertools.product(range(min(c.K_MAX, S0.shape[1]) + 1), range(min(c.K_MAX, S1.shape[1]) + 1))
    candidates = sorted((k for k in pairs if sum(k) <= n - 2), key=lambda k: (sum(k), k[0]))
    errors = []
    for k0, k1 in candidates:
        X = np.column_stack((S0[:, :k0], S1[:, :k1]))
        Z = np.column_stack((np.ones(n), X))
        sigma = scipy.linalg.svdvals(Z)
        leverage = (np.linalg.qr(Z)[0] ** 2).sum(axis=1)
        if sigma[-1] <= SIGMA_RTOL * sigma[0] or leverage.max() >= c.LEVERAGE_CAP:
            errors.append(math.inf)
            continue
        squared = []
        for i in range(n):
            keep = np.arange(n) != i
            if X.shape[1] == 0:
                guess = y[keep].mean()
            else:
                guess = LinearRegression().fit(X[keep], y[keep]).predict(X[i : i + 1])[0]
            squared.append((y[i] - guess) ** 2)
        errors.append(math.fsum(squared) / n)
    best = min(range(len(candidates)), key=lambda j: (errors[j], j))  # the first of least error
    runner_up = min((j for j in range(len(candidates)) if j != best), key=lambda j: (errors[j], j))
    chosen = candidates[best]
    record(
        "loo",
        f"in the interpolation cell with {LOO_N_TRAIN} training samples, (K0, K1) chosen again on the fPCA route's "
        "scores: each candidate's LOO error by refitting LinearRegression without each sample in turn, its eligibility "
        "(rank, leverage) computed anew; it must equal arm F's choice (pairs differing)",
        chosen == saved,
        int(chosen != saved),
        0,
        label
        | {"chosen": list(chosen), "saved": list(saved), "loo_mse": errors[best]}
        | {"runner_up": list(candidates[runner_up]), "runner_up_loo_mse": errors[runner_up]}
        | {"candidates": len(candidates), "ineligible": sum(math.isinf(e) for e in errors)},
    )


# ---------------------------------------------------------------- the readouts, the verdicts and the tables


def bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    """The mean of ``values`` and the 2.5 and 97.5 percentiles of ``N_BOOT`` bootstrap means: design.md's recipe.

    1-D ``values`` are units, resampled with replacement. 2-D ``values`` are strata by units, the units resampled
    within each stratum, each resample's mean taken over all of them (interpolation: the runs within each size). The
    resampling indices of all ``N_BOOT`` draws come from one call of a fresh ``default_rng(BOOT_SEED)``, in draw,
    stratum, unit order: a generator gives the same integers only in calls of the same size.
    """
    strata, units = (1, values.size) if values.ndim == 1 else values.shape
    picks = np.random.default_rng(c.BOOT_SEED).integers(0, units, size=c.N_BOOT * strata * units)
    resampled = values.reshape(strata, units)[np.arange(strata)[:, None], picks.reshape(c.N_BOOT, strata, units)]
    lo, hi = np.percentile(resampled.reshape(c.N_BOOT, strata * units).mean(axis=1), PERCENTILES)
    return float(values.mean()), float(lo), float(hi)


def rule(lo: float, hi: float) -> str:
    """design.md's verdict, fixed in advance: above 0 beats, below 0 is below, otherwise level."""
    return "beats" if lo > 0 else "below" if hi < 0 else "level"


def my_readouts(arms: dict[str, np.ndarray], x: str, y: str) -> dict[str, tuple[float, float, float]]:
    """Every readout of Δ(x, y), Δ > 0 favouring arm ``x``, by name: the three tasks and the descriptive ones."""
    interp = (arms[f"interp_r2_{x}"] - arms[f"interp_r2_{y}"]).reshape(len(c.INTERP_SIZES), c.INTERP_RUNS)
    target = arms["curv_y_test"]
    curv = (target - arms[f"curv_pred_{y}"]) ** 2 - (target - arms[f"curv_pred_{x}"]) ** 2
    fluid = arms[f"fluid_r2_{x}"] - arms[f"fluid_r2_{y}"]  # (times, splits, sizes)
    per_split = fluid.mean(axis=(0, 2))
    out = {"interp": bootstrap(interp), "curv": bootstrap(curv), "fluid": bootstrap(per_split)}
    out |= {f"interp n {n}": bootstrap(interp[si]) for si, n in enumerate(c.INTERP_SIZES)}
    out |= {f"fluid t {t:g}": bootstrap(fluid[ti].mean(axis=1)) for ti, t in enumerate(c.FLUID_TIMES)}
    out |= {f"fluid n {n}": bootstrap(fluid[:, :, si].mean(axis=0)) for si, n in enumerate(c.FLUID_SIZES)}
    out["fluid splits 0-4"] = bootstrap(per_split[: c.PAPER_SPLITS])
    return out


def saved_readouts(pair: dict) -> dict[str, tuple[float, float, float]]:
    """The readouts of one pair in ``readouts.json``, by the names of ``my_readouts``; ``KeyError`` on a label that
    is not the expected one."""
    leaf = {}
    for task in TASKS:
        leaf[task] = pair[task]
    expected = {
        "by_size": [f"interp n {n}" for n in c.INTERP_SIZES],
        "by_time": [f"fluid t {t:g}" for t in c.FLUID_TIMES],
        "fluid_by_size": [f"fluid n {n}" for n in c.FLUID_SIZES],
    }
    found = {
        "by_size": [f"interp n {r['n_train']}" for r in pair["interp"]["by_size"]],
        "by_time": [f"fluid t {r['t']:g}" for r in pair["fluid"]["by_time"]],
        "fluid_by_size": [f"fluid n {r['n_train']}" for r in pair["fluid"]["by_size"]],
    }
    if found != expected or pair["fluid"]["paper_splits"]["splits"] != list(range(c.PAPER_SPLITS)):
        raise KeyError(f"readouts.json's descriptive labels are not the expected ones: {found}")
    rows = pair["interp"]["by_size"] + pair["fluid"]["by_time"] + pair["fluid"]["by_size"]
    leaf |= dict(zip(expected["by_size"] + expected["by_time"] + expected["fluid_by_size"], rows, strict=True))
    leaf["fluid splits 0-4"] = pair["fluid"]["paper_splits"]
    return {name: (r["mean"], r["lo"], r["hi"]) for name, r in leaf.items()}


def verdict_keys(value, path: str = "") -> list[str]:
    """Every path in ``value`` at which a ``verdict`` key sits."""
    if isinstance(value, dict):
        here = [path] if "verdict" in value else []
        return here + [p for key, v in value.items() for p in verdict_keys(v, f"{path}/{key}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in verdict_keys(v, f"{path}/{i}")]
    return []


def check_readouts(arms: dict[str, np.ndarray], saved: dict) -> None:
    pairs = {"primary": (("F", "K"), saved["primary"])}
    pairs |= {
        name: ((x, y), saved["exploratory"][f"{x}_vs_{y}"])
        for name, (x, y) in (("B_vs_K", ("B", "K")), ("F_vs_B", ("F", "B")))
    }
    compared, unequal, worst, mine_all = 0, [], 0.0, {}
    for name, ((x, y), pair) in pairs.items():
        mine, theirs = my_readouts(arms, x, y), saved_readouts(pair)
        mine_all[name] = mine
        for key, values in mine.items():
            compared += 3
            gaps = [abs(a - b) for a, b in zip(values, theirs[key], strict=True)]
            worst = max(worst, *gaps)
            if list(values) != list(theirs[key]):
                unequal.append({"pair": f"{x} vs {y}", "readout": key, "mine": values, "saved": theirs[key]})
        if pair["pair"] != [x, y]:
            unequal.append({"pair": f"{x} vs {y}", "readout": "pair", "saved": pair["pair"]})
    settings = saved["bootstrap"] == {"n_boot": c.N_BOOT, "seed": c.BOOT_SEED, "percentiles": list(PERCENTILES)}
    record(
        "readouts",
        "every interval in readouts.json, the primary pair F vs K, the exploratory pairs B vs K and F vs B, and the "
        "descriptive readouts (interpolation by size, fluid by time, by size and on splits 0-4), from a bootstrap "
        f"written here: Delta from arms.npz, a fresh default_rng({c.BOOT_SEED}) per interval, {c.N_BOOT} resamples of "
        "the units, the 2.5 and 97.5 percentiles; equal to the bit (largest |difference| over mean, lo and hi)",
        settings and not unequal and worst == 0.0,
        worst,
        0.0,
        {
            "bootstrap_settings_as_recorded": settings,
            "values_compared": compared,
            "unequal": unequal,
            "primary": {task: mine_all["primary"][task] for task in TASKS},
        },
    )
    verdicts = {task: rule(*mine_all["primary"][task][1:]) for task in TASKS}
    recorded = {task: saved["primary"][task].get("verdict") for task in TASKS}
    elsewhere = verdict_keys(saved["exploratory"]) + [
        p for p in verdict_keys(saved["primary"]) if p not in {f"/{task}" for task in TASKS}
    ]
    record(
        "verdicts",
        "the primary verdicts by design.md's rule fixed in advance, on the recomputed intervals (beats if lo > 0, "
        "below if hi < 0, level otherwise), against readouts.json's; no verdict in the exploratory pairs or the "
        "descriptive readouts (verdicts differing or misplaced)",
        verdicts == recorded and not elsewhere,
        sum(verdicts[task] != recorded[task] for task in TASKS) + len(elsewhere),
        0,
        {"recomputed": verdicts, "recorded": recorded, "verdicts_elsewhere": elsewhere},
    )


def count_k(k: np.ndarray) -> list[dict]:
    """The chosen (K0, K1) of the cells ``k``, (..., 2), counted with ``np.unique``, in (K0, K1) order."""
    pairs, counts = np.unique(np.asarray(k).reshape(-1, 2), axis=0, return_counts=True)
    return [{"k0": int(a), "k1": int(b), "count": int(n)} for (a, b), n in zip(pairs, counts, strict=True)]


def check_tables(arms: dict[str, np.ndarray], saved: dict) -> None:
    tables, mismatches, gaps = saved["tables"], [], []
    table = read_csv("interp").set_index("n_train")
    for si, n in enumerate(c.INTERP_SIZES):
        row = tables["interp"][si]
        if row["n_train"] != n:
            mismatches.append({"table": "interp", "n_train": row["n_train"], "expected": n})
            continue
        for method, stat in itertools.product(("PD_kernel", "PI_kernel"), ("mean", "std")):
            if row["reported"][method][stat] != float(table.loc[n, f"{method}_R2_{stat}"]):
                mismatches.append({"table": "interp", "n_train": n, "method": method, "statistic": stat})
        runs = arms["interp_r2_K"].size // len(c.INTERP_SIZES)
        for arm in ARMS:
            mean, sd = fsum_mean_sd(arms[f"interp_r2_{arm}"][si * runs : (si + 1) * runs])
            gaps += [score_gap(mean, row[arm]["mean"]), score_gap(sd, row[arm]["std"])]
    table = read_csv("curv")
    reported = list(zip(table["method"].astype(str), table["MSE"].tolist(), strict=True))
    if list(tables["curv"]["reported"].items()) != reported:
        mismatches.append({"table": "curv", "reported": "differs from results_curvature.csv, its values or order"})
    mismatches += [
        {"table": "curv", "arm": arm} for arm in ARMS if tables["curv"][arm] != float(arms[f"curv_mse_{arm}"])
    ]
    table = read_csv("fluid")
    reported = {
        (float(t), int(n)): {"mean": m, "std": s}
        for t, n, m, s in table[["t", "n_train", "R2_mean", "R2_std"]].itertuples(index=False)
    }
    cells = [(t, n) for t in c.FLUID_TIMES for n in c.FLUID_SIZES]
    if len(table) != len(cells) or [(row["t"], row["n_train"]) for row in tables["fluid"]] != cells:
        mismatches.append({"table": "fluid", "cells": "not the 25 (t, n_train) of the table"})
    for ti, si in itertools.product(range(len(c.FLUID_TIMES)), range(len(c.FLUID_SIZES))):
        row = tables["fluid"][ti * len(c.FLUID_SIZES) + si]
        if row["reported"] != reported.get((row["t"], row["n_train"])):
            mismatches.append({"table": "fluid", "t": row["t"], "n_train": row["n_train"]})
        for arm, (part, splits) in itertools.product(ARMS, (("paper_splits", c.PAPER_SPLITS), ("all_splits", None))):
            mean, sd = fsum_mean_sd(arms[f"fluid_r2_{arm}"][ti, :splits, si])
            gaps += [score_gap(mean, row[part][arm]["mean"]), score_gap(sd, row[part][arm]["std"])]
    record(
        "tables",
        "readouts.json's tables: every reported value equals upstream's CSV, read exactly (results_interpolation.csv, "
        "results_curvature.csv in its order, results_fluid.csv), and the curvature MSEs equal arms.npz's (values "
        "differing)",
        not mismatches,
        len(mismatches),
        0,
        {"mismatches": mismatches},
    )
    k_saved = saved["chosen_k"]
    k_mine = {
        "interp": count_k(arms["interp_k_F"]),
        "curv": count_k(arms["curv_k_F"]),
        "fluid": count_k(arms["fluid_k_F"]),
    }
    k_mine["fluid_by_time"] = [{"t": t, "counts": count_k(arms["fluid_k_F"][ti])} for ti, t in enumerate(c.FLUID_TIMES)]
    k_differing = [key for key in k_mine if k_mine[key] != k_saved.get(key)]
    worst = max(gaps)
    record(
        "tables",
        "readouts.json's tables: each arm's mean and SD (ddof 1) of R2 over each interpolation size's runs and over "
        "each fluid (t, n_train)'s splits 0-4 and 0-99, recomputed by math.fsum (|saved - recomputed| / max(1, "
        "|saved|)); and arm F's chosen (K0, K1) counted again with np.unique, per task and fluid time, equal",
        worst <= TOL_SCORE and not k_differing,
        worst,
        TOL_SCORE,
        {"means_and_sds": len(gaps), "chosen_k_differing": k_differing},
    )


# ---------------------------------------------------------------- the provenance of the stages' jsons


def check_provenance(matrices_sha: str, arms_sha: str) -> None:
    rows, problems = [], []
    stage = {}
    for name in STAGE_JSONS:
        path = f"results/{name}"
        commit = git("log", "-1", "--format=%H", "--", path).stdout.decode().strip()
        unchanged = bool(commit) and git("diff", "--quiet", "HEAD", "--", path).returncode == 0
        stage[name] = json.loads((c.RESULTS / name).read_text())
        provenance = stage[name]["provenance"]
        if not unchanged:
            problems.append(f"{name}: not committed, or changed since its last commit")
        if (provenance["topocyte_commit"], provenance["upstream_sha"]) != (c.TOPOCYTE_COMMIT, c.UPSTREAM_SHA):
            problems.append(f"{name}: pins {provenance['topocyte_commit']}, {provenance['upstream_sha']}")
        if stage[name].get("passed", True) is not True:
            problems.append(f"{name}: a gate failed")
        for script, recorded in provenance["md5"].items():
            shown = git("show", f"{commit}:./{script}")
            at_commit = md5(shown.stdout) if shown.returncode == 0 and commit else None
            now = md5((c.HERE / script).read_bytes())
            rows.append([name, commit[:7], script, recorded[:8], at_commit == recorded, now == recorded])
            if at_commit != recorded:
                problems.append(f"{name}: {script}'s md5 {recorded[:8]} is not that of the script in {commit[:7]}")
    vouched = {
        "matrices.json": stage["matrices.json"]["npz"]["sha256"] == matrices_sha,
        "arms.json": stage["arms.json"]["npz"]["sha256"] == arms_sha,
        "readouts.json": stage["readouts.json"]["arms_npz_sha256"] == arms_sha,
    }
    problems += [f"{name}: records another npz sha256" for name, ok in vouched.items() if not ok]
    record(
        "provenance",
        "each stage's json (upstream, inputs, matrices, arms, readouts) is unchanged since the commit that last "
        "changed it; every md5 it records is that of the script in that commit (git show); its pins are the study's "
        "and its gates passed; matrices.npz and arms.npz are the files the jsons vouch for (problems; rows: json, "
        "commit, script, md5, equal at the commit, equal in the working tree)",
        not problems,
        len(problems),
        0,
        {"md5s_compared": len(rows), "problems": problems, "npz_vouched": vouched, "rows": rows},
    )


# ---------------------------------------------------------------- the run


def fluid_digests() -> dict[str, str]:
    """For each fluid time, the sha256 of its 400 diagram pickles' bytes in ``fluid_sim_ids()`` order, one after the
    other: what ``inputs.fluid_diagrams`` read."""
    out = {}
    for t in c.FLUID_TIMES:
        digest = hashlib.sha256()
        for i in inp.fluid_sim_ids():
            digest.update(inp._diagram_path(i, t).read_bytes())
        out[f"{t:g}"] = digest.hexdigest()
    return out


def main() -> None:
    c.check_topocyte()
    c.check_upstream()
    provenance = c.provenance(*(c.HERE / name for name in _SCRIPTS))  # at the start: the code that runs
    provenance["versions"] |= {"scikit-learn": sklearn.__version__, "python": platform.python_version()}
    started = time.perf_counter()
    seconds: dict[str, float] = {}

    def timed(family: str, check: Callable, *args):
        part = time.perf_counter()
        out = check(*args)
        seconds[family] = round(time.perf_counter() - part, 1)
        return out

    M = mat.load_matrices()
    arms, arms_sha = rs.load_arms()
    read_files.extend([c.RESULTS / "matrices.npz", c.RESULTS / "arms.npz"])
    readouts = json.loads((c.RESULTS / "readouts.json").read_text())
    data = timed("load", Data, M, arms)
    print(f"loaded the matrices, the arms, the clouds and the fluid diagrams in {seconds['load']} s", flush=True)
    timed("w2", check_w2, data)
    timed("gram_l2", check_gram_l2, data)
    timed("arrays_scores", check_arrays, data)
    timed("g1", check_g1, data)
    timed("nw", check_nw, data)
    loo = timed("fpcr", check_fpcr, data)
    timed("loo", check_loo, loo)
    timed("readouts_verdicts", check_readouts, arms, readouts)
    timed("tables", check_tables, arms, readouts)
    matrices_sha = sha256(c.RESULTS / "matrices.npz")
    timed("provenance", check_provenance, matrices_sha, arms_sha)

    families = {}
    for check in checks:
        family = families.setdefault(check["family"], {"checks": 0, "passed": 0, "worst": [], "tolerance": []})
        family["checks"] += 1
        family["passed"] += check["passed"]
        family["worst"].append(check["worst"])
        family["tolerance"].append(check["tolerance"])
    sha = {str(path.relative_to(c.HERE)): sha256(path) for path in dict.fromkeys(read_files)}  # in order, once each
    out = {
        "description": "the Betti FPCR benchmark's saved results checked by routes other than the code that made them "
        "(verify_results.py's docstring): the W2, Gram and L2 entries, arms.npz and its scores, gate G1, arms K, B and "
        "F on sampled cells, F's LOO choice in one cell, every readout interval and verdict, the tables, and the "
        "stages' provenance",
        "passed": all(check["passed"] for check in checks),
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "families": families,
        "checks": checks,
        "seeds": {"w2": W2_SEED, "fpcr": FPCR_SEED, "gram": GRAM_SEED, "nw": NW_SEED, "bootstrap": c.BOOT_SEED},
        "sha256": sha,
        "fluid_diagrams_sha256": fluid_digests(),
        "upstream_clean_at": c.UPSTREAM_SHA,
        "seconds": seconds | {"total": round(time.perf_counter() - started, 1)},
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),  # bytes on macOS
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance,
        "platform": platform.platform(),
    }
    (c.RESULTS / "verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(
        f"wrote results/verify.json: {out['checks_passed']}/{out['checks_total']} checks pass in "
        f"{out['seconds']['total']} s",
        flush=True,
    )
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
