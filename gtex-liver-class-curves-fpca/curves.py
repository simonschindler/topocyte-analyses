#!/usr/bin/env python
"""Each slide's class curves over the 10-NN density, and each class's exact Gram matrix over the 594 slides.

For one slide and one cell class, the curve counts the class's cells whose density ``rho`` (cells per mm², the
pathology study's ``density.npz``) lies above ``t``: the reduced Betti curve of the bars ``(0, rho_i)``, topocyte's
``BettiCurve`` of dimension 0. A bar is alive on ``[0, rho_i)``, so the curve counts ``rho > t`` where the viewer
counts ``rho >= t``; the two differ only at the points ``rho_i``, which no integral sees. The L2 inner product of two
such curves is exactly ``sum_i sum_j min(rho_i, rho_j)``.

Stages
------
``spike``
    The Gram matrix of the first ``SPIKE_SLIDES`` slides per class, timed and extrapolated to the 594 slides by pair
    count: ``results/spike.json``.
``gram``
    Gate: each class's cells at or above each of the ten deciles, counted per slide with ``rho >= d_j``, equal the
    cell-type study's ``composition.npz`` ``kept_by_group``, else ``RuntimeError``. Then, one class at a time, the
    594 curves and ``betti_inner_product_matrix(curves, dim=0, n_jobs=-1)``: ``results/gram.npz`` (``gram_<class>``,
    float64 594 × 594, and ``slide_ids``) and ``results/gram.json``.

Run from this folder: ``PYTHONPATH=<topocyte checkout at 18e3aa5> <python> curves.py spike|gram``. It reads
``density.npz``, the cohort .h5ad (h5py mode "r") and ``composition.npz`` read only, and writes only to ``results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from topocyte.ph import BettiCurve, betti_inner_product_matrix

import common as c

SPIKE_SLIDES = 20  # the spike's Gram covers the first 20 slides of the study's order, per class


def sum_of_minima(a: np.ndarray, b: np.ndarray) -> float:
    """``sum_i sum_j min(a_i, b_j)`` in O(n log n), from sorted ``b`` and its prefix sums.

    For each ``a_i``, the ``b_j`` below it add themselves and the others add ``a_i``. The tests and the verifier
    check the Gram matrices with it; the Gram never uses it.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.sort(np.asarray(b, dtype=np.float64))
    prefix = np.concatenate([[0.0], np.cumsum(b)])
    below = np.searchsorted(b, a, side="left")
    return float(np.sum(prefix[below] + a * (b.size - below)))


def class_curve(rho: np.ndarray) -> BettiCurve:
    """The class curve of one slide's cells of one class: the dimension-0 Betti curve of the bars ``(0, rho_i)``."""
    return BettiCurve(np.column_stack([np.zeros(len(rho)), rho]), dim=0)


def decile_counts(rho: np.ndarray, classes: np.ndarray, bounds: np.ndarray, deciles: np.ndarray) -> np.ndarray:
    """(deciles, slides, classes) int64: each slide's cells of each class with ``rho >= d_j``.

    Slide ``s`` is the cohort rows ``bounds[s]:bounds[s + 1]``; ``classes`` indexes ``CLASSES``.
    """
    counts = np.zeros((len(deciles), len(bounds) - 1, len(c.CLASSES)), dtype=np.int64)
    for s in range(len(bounds) - 1):
        r, k = rho[bounds[s] : bounds[s + 1]], classes[bounds[s] : bounds[s + 1]]
        for j, d in enumerate(deciles):
            counts[j, s] = np.bincount(k[r >= d], minlength=len(c.CLASSES))
    return counts


def decile_gate(counts: np.ndarray, slide_ids: list[str]) -> None:
    """``RuntimeError`` unless ``counts`` equal the cell-type study's ``kept_by_group``, slide for slide."""
    with np.load(c.CELLTYPE / "results/composition.npz", allow_pickle=False) as composition:
        if [str(s) for s in composition["slide_ids"]] != slide_ids:
            raise RuntimeError("composition.npz lists the slides differently from spaces.npz")
        kept = composition["kept_by_group"]
    if not np.array_equal(counts, kept):
        raise RuntimeError("the decile counts differ from the cell-type study's kept_by_group")


def read_cells(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every cohort cell's density ``rho`` (float64, finite) and class code (int8), in cohort row order."""
    with np.load(c.SIB_DENSITY, allow_pickle=False) as density:
        rho = density["rho"]
    classes, _ = c.celltype.read_cell_groups()
    if rho.dtype != np.float64 or rho.shape != classes.shape or rho.size != bounds[-1]:
        raise RuntimeError(f"rho {rho.dtype} {rho.shape}, classes {classes.shape} and bounds[-1] {bounds[-1]} disagree")
    if not np.isfinite(rho).all():
        raise RuntimeError("a cell's density is not finite")
    return rho, classes


def slide_curves(rho: np.ndarray, classes: np.ndarray, bounds: np.ndarray, k: int) -> list[BettiCurve]:
    """The class-``k`` curve of every slide that ``bounds`` delimits."""
    return [
        class_curve(rho[start:end][classes[start:end] == k]) for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def run_spike(inputs: dict, rho: np.ndarray, classes: np.ndarray) -> dict:
    """Time each class's Gram matrix of the first ``SPIKE_SLIDES`` slides; extrapolate it by pair count."""
    n_all = len(inputs["slide_ids"])
    pairs_spike, pairs_all = SPIKE_SLIDES * (SPIKE_SLIDES + 1) // 2, n_all * (n_all + 1) // 2
    cells_all = np.bincount(classes, minlength=len(c.CLASSES))
    per_class, hours = {}, 0.0
    for k, name in enumerate(c.CLASSES):
        t0 = time.perf_counter()
        curves = slide_curves(rho, classes, inputs["bounds"][: SPIKE_SLIDES + 1], k)
        t1 = time.perf_counter()
        betti_inner_product_matrix(curves, dim=0, n_jobs=-1)
        t2 = time.perf_counter()
        bars = sum(curve.deaths.size for curve in curves)
        class_hours = (t2 - t1) * pairs_all / pairs_spike / 3600
        hours += class_hours
        # A pair's cost grows with its bars: the two means show whether the first slides are typical.
        per_class[name] = {
            "bars": bars,
            "mean_bars_per_slide_spike": round(bars / SPIKE_SLIDES, 1),
            "mean_bars_per_slide_all": round(int(cells_all[k]) / n_all, 1),
            "build_seconds": round(t1 - t0, 2),
            "gram_seconds": round(t2 - t1, 2),
            "extrapolated_hours": round(class_hours, 3),
        }
        print(
            f"{name}: {bars:,} bars on {SPIKE_SLIDES} slides, Gram in {t2 - t1:.1f} s "
            f"-> {class_hours:.2f} h for {n_all} slides",
            flush=True,
        )
        del curves
    return {
        "description": f"each class's Gram of the first {SPIKE_SLIDES} slides, timed, extrapolated by pair count",
        "slides": SPIKE_SLIDES,
        "pairs": pairs_spike,
        "pairs_all": pairs_all,
        "classes": per_class,
        "extrapolated_hours": round(hours, 3),
    }


def run_gram(inputs: dict, rho: np.ndarray, classes: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """The decile gate, then each class's exact Gram matrix over every slide, one class's curves at a time."""
    t0 = time.perf_counter()
    counts = decile_counts(rho, classes, inputs["bounds"], inputs["deciles"])
    decile_gate(counts, inputs["slide_ids"])
    gate = {
        "equal": True,
        "shape": list(counts.shape),
        "cells": int(counts[0].sum()),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"decile gate passed: {gate['cells']:,} cells at decile 0 in {gate['seconds']} s", flush=True)
    grams, per_class = {}, {}
    for k, name in enumerate(c.CLASSES):
        t0 = time.perf_counter()
        curves = slide_curves(rho, classes, inputs["bounds"], k)
        t1 = time.perf_counter()
        grams[f"gram_{name}"] = betti_inner_product_matrix(curves, dim=0, n_jobs=-1)
        t2 = time.perf_counter()
        bars = [curve.deaths.size for curve in curves]
        del curves
        per_class[name] = {
            "bars": int(sum(bars)),
            "min_bars_per_slide": int(min(bars)),
            "build_seconds": round(t1 - t0, 1),
            "gram_seconds": round(t2 - t1, 1),
        }
        print(f"{name}: {sum(bars):,} bars, Gram in {t2 - t1:.1f} s", flush=True)
    grams["slide_ids"] = np.array(inputs["slide_ids"])
    summary = {
        "description": "each class's exact Gram matrix of the class curves over the 594 slides",
        "n_slides": len(inputs["slide_ids"]),
        "classes": per_class,
        "decile_gate": gate,
    }
    return grams, summary


def main(argv: Sequence[str] | None = None) -> None:
    c.check_topocyte()
    script_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()  # at the start: the code that runs
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("stage", choices=("spike", "gram"))
    stage = parser.parse_args(argv).stage
    started = time.perf_counter()
    inputs = c.load_inputs()
    rho, classes = read_cells(inputs["bounds"])
    print(f"read {rho.size:,} cells in {time.perf_counter() - started:.1f} s", flush=True)
    c.RESULTS.mkdir(exist_ok=True)
    if stage == "spike":
        summary = run_spike(inputs, rho, classes)
    else:
        grams, summary = run_gram(inputs, rho, classes)
        np.savez(c.RESULTS / "gram.npz", **grams)
    summary.update(
        {
            "seconds": round(time.perf_counter() - started, 1),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),  # bytes on macOS
            "workers": os.process_cpu_count(),
            "script_md5": script_md5,
            "topocyte_commit": c.TOPOCYTE_COMMIT,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__},
            "platform": platform.platform(),
        }
    )
    (c.RESULTS / f"{stage}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote results/{stage}.json in {summary['seconds']} s, peak RSS {summary['peak_rss_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
