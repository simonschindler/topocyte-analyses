#!/usr/bin/env python
"""Every cohort cell's 150 µm neighbourhood, counted: its cells of each class, and its cells that survive each of
the pathology study's density thresholds.

For every cell of the 594 slides, the closed ball of ``RADIUS_UM`` around it (the cell included, as the
chromatic pilot's balls) is counted twice over:

- by class: the ball's cells of each of the four classes (their sum is the ball's cells);
- by density: the ball's cells with a 10-NN density ``rho >= d_j`` for the thresholds j = 1..9, where ``d_j`` are
  the pathology study's deciles of rho (density.npz); threshold 0, the minimum of rho, keeps every cell.

Writes ``results/counts.npy`` (cells × ``COUNT_COLUMNS``, uint16, cohort row order) and ``results/features.json``.
Reads, read only, the cohort .h5ad (h5py mode "r") and the pathology study's density.npz.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree

import common as c


def ball_counts(xy: np.ndarray, classes: np.ndarray, rho: np.ndarray, thresholds: np.ndarray, radius_px: float):
    """(n, 4 + len(thresholds) - 1) counts per cell's closed ball: cells of each class, then the cells with
    ``rho >= thresholds[j]`` for j >= 1 (threshold 0 is taken to keep every cell and is not counted)."""
    n = len(xy)
    counts = np.zeros((n, len(c.CLASSES) + len(thresholds) - 1), np.int64)
    subsets = [classes == k for k in range(len(c.CLASSES))] + [rho >= t for t in thresholds[1:]]
    for column, members in enumerate(subsets):
        if members.any():
            counts[:, column] = cKDTree(xy[members]).query_ball_point(xy, r=radius_px, return_length=True, workers=-1)
    return counts


def check_counts(counts: np.ndarray, rho: np.ndarray, thresholds: np.ndarray, slide_id: str) -> None:
    """Every ball holds its own cell; kept counts fall with the threshold, never above the ball's cells; a cell
    above a threshold is kept in its own ball."""
    n_ball = counts[:, : len(c.CLASSES)].sum(axis=1)
    kept = np.column_stack([n_ball, counts[:, len(c.CLASSES) :]])
    problems = []
    if n_ball.min() < 1:
        problems.append("a ball without its own cell")
    if np.any(np.diff(kept, axis=1) > 0):
        problems.append("a kept count rises with the threshold")
    own = rho[:, None] >= thresholds[None, 1:]
    if np.any(own & (counts[:, len(c.CLASSES) :] < 1)):
        problems.append("a kept cell missing from its own ball")
    if counts.max() >= np.iinfo(np.uint16).max:
        problems.append(f"a count of {counts.max()} does not fit uint16")
    if problems:
        raise RuntimeError(f"{slide_id}: {'; '.join(problems)}")


def main() -> None:
    import h5py

    started = time.perf_counter()
    with np.load(c.SIB / "density.npz", allow_pickle=False) as density:
        rho_all = density["rho"]
        thresholds = np.asarray(density["deciles"], dtype=np.float64)
        bounds = np.asarray(density["bounds"], dtype=np.int64)
        slide_ids = [str(s) for s in density["slide_ids"]]
        pixel_um, k_nn = float(density["pixel_um"]), int(density["k_nn"])
    if len(thresholds) != c.N_THRESHOLDS or thresholds[0] != rho_all.min() or pixel_um != c.PX_UM:
        raise RuntimeError("density.npz's thresholds or pixel size are not the ones this study expects")
    n_cells = int(bounds[-1])
    c.RESULTS.mkdir(exist_ok=True)
    partial = c.RESULTS / "counts.npy.partial"
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint16, shape=(n_cells, len(c.COUNT_COLUMNS)))
    with h5py.File(c.H5AD, "r") as f:
        if f["obsm/spatial"].shape[0] != n_cells:
            raise RuntimeError("the cohort and density.npz hold different numbers of cells")
        names = c.strings(f["obs/slide_id/categories"][:])
        owners = f["obs/slide_id/codes"]
        class_of = c.class_lookup(c.strings(f["obs/cell_type/categories"][:]))
        for index, slide_id in enumerate(slide_ids):
            t0 = time.perf_counter()
            start, end = int(bounds[index]), int(bounds[index + 1])
            if names[int(owners[start])] != slide_id or names[int(owners[end - 1])] != slide_id:
                raise RuntimeError(f"cohort rows {start}:{end} are not all {slide_id}")
            xy = np.asarray(f["obsm/spatial"][start:end], dtype=np.float64)
            codes = np.asarray(f["obs/cell_type/codes"][start:end])
            if codes.min() < 0 or not np.all(np.isfinite(xy)):
                raise RuntimeError(f"{slide_id}: a cell without a type or a finite position")
            rho = rho_all[start:end]
            counts = ball_counts(xy, class_of[codes], rho, thresholds, c.RADIUS_PX)
            check_counts(counts, rho, thresholds, slide_id)
            out[start:end] = counts.astype(np.uint16)
            print(
                f"{index + 1:>3}/{len(slide_ids)} {slide_id} {end - start:>8,} cells {time.perf_counter() - t0:6.1f} s",
                flush=True,
            )
    out.flush()
    del out
    os.replace(partial, c.RESULTS / "counts.npy")
    counts = np.load(c.RESULTS / "counts.npy", mmap_mode="r")
    summary = {
        "description": "every cohort cell's closed 150 µm ball: cells per class, and cells kept at density thresholds",
        "columns": list(c.COUNT_COLUMNS),
        "n_cells": n_cells,
        "n_slides": len(slide_ids),
        "radius_um": c.RADIUS_UM,
        "radius_px": c.RADIUS_PX,
        "px_um": c.PX_UM,
        "k_nn": k_nn,
        "thresholds_cells_per_mm2": thresholds.tolist(),
        "max_count": int(counts.max()),
        "mean_cells_per_ball": float(counts[:, : len(c.CLASSES)].sum(axis=1, dtype=np.float64).mean()),
        "seconds": round(time.perf_counter() - started, 1),
        "script_md5": c.md5(c.HERE / "features.py"),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    (c.RESULTS / "features.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_cells", "max_count", "mean_cells_per_ball", "seconds")}))


if __name__ == "__main__":
    main()
