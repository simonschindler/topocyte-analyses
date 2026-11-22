#!/usr/bin/env python
"""Run script of the chromatic-domains pilot.

Subcommands
-----------
``pilot``
    Draw the 30 pilot slides from the eligible development slides (design
    step 1) and write ``results/pilot.csv``: slide, condition, half and slide
    index. It refuses inputs whose eligible counts per condition differ from
    the design's.
``preflight``
    Check the rule for k (design step 10, ``domains.choose_k``) before the
    run, on synthetic halves of the pilot's sizes (design, "Checks fixed in
    advance", 1). Matched Gaussians with PCA-like variances must give k = 1:
    22,500 × 5 with k_max 10 (level 1), and 5,000 and 22,500 × 20 with k_max
    6 (level 2). So must skewed cluster-free clouds of the same sizes, whose
    first column is lognormal (σ = 0.6), and a 5-d one whose lognormal column
    carries the most variance (3). Two- and three-cluster mixtures of 5,000 ×
    20 must give k = 2 and 3. Writes ``results/preflight.json`` and exits 1 if
    any case fails.
``bars [--slides SLIDE_ID ...] [--workers 8]``
    The six-packs of every centre (design steps 2–4) and its pair counts
    (step 11), for the pilot's slides in slide-index order (default all 30).
    A centre's ball is every cell of its slide within ``DELTA_PX`` of it,
    boundary and centre included, in ascending row order. For labelling 0
    (the observed classes) and labellings 1..R (``common.labelling``), the
    domain and image diagrams of every block of ``common.BLOCKS``, and for
    labelling 0 the codomain, are kept as their finite bars (essential bars
    are left out, as topocyte's reduced Betti curves leave them out). Balls
    go to ``--workers`` spawned processes in chunks of ``CHUNK`` centres;
    each slide is written as ``results/bars/{slide_id}.npz`` (``BARS_KEYS``)
    and ``.json`` (timings, sizes, peak RSS, versions). A slide whose npz is
    already complete (``bars_complete``) is skipped, so a rerun resumes.

Run from this folder with the study venv and
``PYTHONPATH=<topocyte checkout at 18e3aa5>``. Every subcommand first
refuses a topocyte from anywhere else, or a checkout whose package differs
from the commit (``common.check_topocyte``). The cohort is opened read only,
and nothing is written outside this folder.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import resource
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import cKDTree

import chromatic
import common as c
import domains

# Design step 1: the eligible slides per condition.
EXPECTED_ELIGIBLE = {
    "healthy": 53,
    "congestion": 118,
    "steatosis": 158,
    "inflammation/hepatitis": 9,
    "fibrosis": 40,
    "cirrhosis": 44,
}

# The pre-flight's synthetic halves (the controller's constructions, ruling R2, and the skewed cases of Simon's ruling
# of 2026-11-21). One default_rng(PREFLIGHT_SEED) draws the cases in PREFLIGHT_CASES order, half A then half B.
PREFLIGHT_SEED = 1
LEVEL1_VARIANCES = np.array([3.0, 1.5, 0.8, 0.4, 0.2])  # PCA-like variances of the five level-1 features
LEVEL2_VARIANCES = 1 / np.arange(1, 21)  # PCA-like variances of the 20 level-2 components
LOGNORMAL_VARIANCE = (np.exp(0.36) - 1) * np.exp(0.36)  # the variance of exp(N(0, 0.6)), about 0.621


def _gaussian(variances: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """n rows of a centred Gaussian with independent columns of the given variances."""
    return rng.normal(size=(n, variances.size)) * np.sqrt(variances)


def _skewed(variances: np.ndarray, rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    """n rows of a skewed cluster-free cloud: column 0 scale * exp(N(0, 0.6)), the rest Gaussian of these variances."""
    return np.column_stack([scale * np.exp(rng.normal(0, 0.6, n)), _gaussian(variances, rng, n)])


def _two_clusters(rng: np.random.Generator, n: int) -> np.ndarray:
    """N(0, I) in 20 dimensions with column 0 shifted by -2 or +2 at random: centres 4 SD apart."""
    X = rng.normal(size=(n, 20))
    X[:, 0] += np.where(rng.random(n) < 0.5, -2.0, 2.0)
    return X


def _three_clusters(rng: np.random.Generator, n: int) -> np.ndarray:
    """N(0, I) in 20 dimensions with centres (-3, 0), (0, 3) and (3, 0) on columns 0 and 1, drawn at random."""
    X = rng.normal(size=(n, 20))
    X[:, :2] += np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])[rng.integers(0, 3, n)]
    return X


# (case, rows per half, k_max, the k the rule must give, construction)
PREFLIGHT_CASES = (
    ("level-1 Gaussian, 22,500 x 5", 22_500, 10, 1, partial(_gaussian, LEVEL1_VARIANCES)),
    ("level-2 Gaussian, 5,000 x 20", 5_000, 6, 1, partial(_gaussian, LEVEL2_VARIANCES)),
    ("level-2 Gaussian, 22,500 x 20", 22_500, 6, 1, partial(_gaussian, LEVEL2_VARIANCES)),
    ("two clusters, 5,000 x 20", 5_000, 6, 2, _two_clusters),
    ("three clusters, 5,000 x 20", 5_000, 6, 3, _three_clusters),
    # Appended, so that the cases above keep their draws.
    ("level-2 skewed, 5,000 x 20", 5_000, 6, 1, partial(_skewed, 1 / np.arange(2, 21))),
    ("level-2 skewed, 22,500 x 20", 22_500, 6, 1, partial(_skewed, 1 / np.arange(2, 21))),
    ("level-1 skewed, 22,500 x 5", 22_500, 10, 1, partial(_skewed, np.array([1.5, 0.8, 0.4, 0.2]))),
    # The lognormal column scaled to variance 3, the largest (the corrected rule, 2026-11-21).
    (
        "level-1 skew-dominant, 22,500 x 5",
        22_500,
        10,
        1,
        partial(_skewed, np.array([1.5, 0.8, 0.4, 0.2]), scale=np.sqrt(3 / LOGNORMAL_VARIANCE)),
    ),
)


def cmd_pilot() -> None:
    eligible = c.eligible_slides(c.CT / "holdout.csv", c.SIB / "pathology.csv")
    counts = {condition: int((eligible["condition"] == condition).sum()) for condition in c.CONDITIONS}
    print("eligible slides: " + ", ".join(f"{condition} {n}" for condition, n in counts.items()), flush=True)
    if counts != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"eligible slides per condition are {counts}, not the design's {EXPECTED_ELIGIBLE}")
    pilot = c.draw_pilot(eligible)
    pilot.to_csv(c.RESULTS / "pilot.csv", index=False)
    print(pd.crosstab(pilot["condition"], pilot["half"]).reindex(list(c.CONDITIONS)).to_string())
    halves = pilot["half"].value_counts()
    print(f"pilot: {len(pilot)} slides, half A {halves['A']}, half B {halves['B']}; wrote {c.RESULTS / 'pilot.csv'}")


def _half(result: dict) -> dict:
    """One half's gap curves for the JSON, with its largest gain Gap(k) - Gap(1) over k >= 2."""
    gap, s = result["gap"], result["s"]
    return {"k": result["k"], "gap": gap.tolist(), "s": s.tolist(), "largest_gain": float(np.max(gap[1:] - gap[0]))}


def cmd_preflight() -> bool:
    """Run the full rule for k on each pre-flight case, write ``results/preflight.json``; True if every case passes."""
    rng = np.random.default_rng(PREFLIGHT_SEED)
    cases, start = [], time.perf_counter()
    for name, n, k_max, expected, draw in PREFLIGHT_CASES:
        A, B = draw(rng, n), draw(rng, n)
        t = time.perf_counter()
        result = domains.choose_k(A, B, [(A, B), (B, A)], k_max)
        case = {
            "case": name,
            "rows_per_half": n,
            "columns": A.shape[1],
            "k_max": k_max,
            "expected_k": expected,
            "k": result["k"],
            "k_A": result["k_A"],
            "k_B": result["k_B"],
            "ps": result["ps"],
            "gap_A": _half(result["gap_A"]),
            "gap_B": _half(result["gap_B"]),
            "seconds": round(time.perf_counter() - t, 1),
            "passed": result["k"] == expected,
        }
        cases.append(case)
        print(
            f"{name}: k {case['k']} (expected {expected}), k_A {case['k_A']}, k_B {case['k_B']}, "
            f"largest gain {case['gap_A']['largest_gain']:.4f} / {case['gap_B']['largest_gain']:.4f}, "
            f"PS {[round(p, 3) for p in case['ps']]}, {'pass' if case['passed'] else 'FAIL'}, {case['seconds']} s",
            flush=True,
        )
    passed = all(case["passed"] for case in cases)
    out = {
        "passed": passed,
        "rule": {
            "stop": "Tibshirani's stopping rule from k = 1, then the gain floor on the k it stops at",
            "B": domains.N_REFERENCES,
            "min_gain": domains.MIN_GAIN,
            "ps_min": domains.PS_MIN,
            "n_init": domains.N_INIT,
            "seed": 0,
        },
        "data": f"one default_rng({PREFLIGHT_SEED}) draws the cases in order, half A then half B",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "seconds": round(time.perf_counter() - start, 1),
        "cases": cases,
    }
    (c.RESULTS / "preflight.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"pre-flight {'passed' if passed else 'FAILED'} in {out['seconds']} s; wrote {c.RESULTS / 'preflight.json'}")
    return passed


# ---------------------------------------------------------------------------------------------------------------- bars

CHUNK = 100  # centres per worker task
# The arrays of results/bars/{slide_id}.npz, centre i in row i:
# - per centre: centre_rows (the slide rows of draw_centres), centre_xy, n_cells (cells in the ball), class_counts
#   (N, 4) and pair_counts (N, 10) and n_edges of the observed classes, seconds (the centre's six-packs and counts,
#   timed in its worker);
# - bars (M, 2), finite only, in px², sorted by owner = common.encode_owner(centre, labelling, block), then birth and
#   death, with offsets (N·(R + 1)·len(BLOCKS) + 1,): owner o holds rows offsets[o]:offsets[o + 1];
# - codomain (K, 2), finite only, of labelling 0, sorted by codomain_owner = 2·centre + p, then birth and death.
BARS_KEYS = (
    "centre_rows", "centre_xy", "n_cells", "class_counts", "pair_counts", "n_edges", "seconds",
    "bars", "owner", "offsets", "codomain", "codomain_owner",
)  # fmt: skip
VERSIONED = ("numpy", "scipy", "numba", "gudhi", "phimaker", "h5py", "pandas")


def _peak_rss() -> int:
    """This process's peak resident set size so far, in bytes (``ru_maxrss`` counts bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else 1024 * int(peak)


def _warm() -> None:
    """One small ``ball_sixpacks`` and ``pair_counts`` call, which compiles or loads every numba kernel of the port.

    The ball is a fixed 40-cell spiral of four classes, so no generator is drawn.
    """
    t = np.arange(1.0, 41.0)
    xy, cls = np.column_stack([t * np.cos(t), t * np.sin(t)]), (np.arange(40) % 4).astype(np.int8)
    chromatic.ball_sixpacks(xy, cls)
    chromatic.pair_counts(xy, cls)


def _by_owner(pieces: list[np.ndarray], codes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The finite bars of ``pieces`` (piece j owned by ``codes[j]``), sorted by owner, then birth, then death."""
    owner = np.repeat(np.asarray(codes, dtype=np.int32), [len(piece) for piece in pieces])
    bars = np.concatenate(pieces) if pieces else np.empty((0, 2))
    order = np.lexsort((bars[:, 1], bars[:, 0], owner))
    return bars[order], owner[order]


def _finite(diagram: np.ndarray) -> np.ndarray:
    return diagram[np.isfinite(diagram[:, 1])]


def _bars_chunk(
    slide_index: int, centres: np.ndarray, ball_offsets: np.ndarray, ball_xy: np.ndarray, ball_cls: np.ndarray
) -> dict:
    """One worker task: the six-packs and counts of the balls of ``centres`` (centre indices, ascending).

    Ball j is ``ball_xy[ball_offsets[j]:ball_offsets[j + 1]]`` with its observed
    classes, rows in ascending slide order. Each centre's 20 labellings go
    through ``chromatic.ball_sixpacks``; its pair counts are of the observed
    classes. The returned arrays follow ``BARS_KEYS``, plus the worker's pid,
    peak RSS (bytes) and CPU seconds so far.
    """
    k = len(centres)
    class_counts = np.zeros((k, len(c.CLASSES)), np.int64)
    pair_counts = np.zeros((k, len(c.PAIRS)), np.int64)
    n_edges = np.zeros(k, np.int64)
    seconds = np.zeros(k)
    pieces, codes, cod_pieces, cod_codes = [], [], [], []
    started = time.time()
    for j, centre in enumerate(centres.tolist()):
        xy, cls = ball_xy[ball_offsets[j] : ball_offsets[j + 1]], ball_cls[ball_offsets[j] : ball_offsets[j + 1]]
        start = time.perf_counter()
        try:
            for r in range(c.R + 1):
                packs = chromatic.ball_sixpacks(xy, c.labelling(cls, slide_index, centre, r))
                owners = c.encode_owner(centre, r, np.arange(len(c.BLOCKS)))
                for block, (t, module, p) in enumerate(c.BLOCKS):
                    pieces.append(_finite(packs[module, t, p]))
                    codes.append(int(owners[block]))
                if r == 0:
                    for p in (0, 1):
                        cod_pieces.append(_finite(packs["codomain", -1, p]))
                        cod_codes.append(2 * centre + p)
            pair_counts[j], n_edges[j] = chromatic.pair_counts(xy, cls)
            class_counts[j] = np.bincount(cls, minlength=len(c.CLASSES))
        except Exception as error:
            raise RuntimeError(f"slide index {slide_index}, centre {centre} ({len(cls)} cells): {error!r}") from error
        seconds[j] = time.perf_counter() - start
    bars, owner = _by_owner(pieces, codes)
    codomain, codomain_owner = _by_owner(cod_pieces, cod_codes)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "centres": centres,
        "n_cells": np.diff(ball_offsets).astype(np.int64),
        "class_counts": class_counts,
        "pair_counts": pair_counts,
        "n_edges": n_edges,
        "seconds": seconds,
        "bars": bars,
        "owner": owner,
        "codomain": codomain,
        "codomain_owner": codomain_owner,
        "pid": os.getpid(),
        "peak_rss": _peak_rss(),
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "started": started,
        "finished": time.time(),
    }


def bars_complete(path: Path) -> bool:
    """Whether ``path`` holds a whole slide: it opens, every one of ``BARS_KEYS`` reads, and the offsets cover the bars.

    Reading every array lets zipfile check each member's CRC. The offsets must
    have one entry per owner of ``common.N_CENTRES`` centres plus one, and
    end at the number of bars and of owners. The file is opened here, not by
    ``np.load``, which leaves its own handle open when a truncated zip fails.
    """
    n_owners = c.N_CENTRES * (c.R + 1) * len(c.BLOCKS)
    try:
        with open(path, "rb") as f, np.load(f, allow_pickle=False) as z:
            if not set(BARS_KEYS) <= set(z.files):
                return False
            arrays = {key: z[key] for key in BARS_KEYS}
    except Exception:  # missing, truncated or not an npz: redo the slide
        return False
    offsets = arrays["offsets"]
    return offsets.shape == (n_owners + 1,) and int(offsets[-1]) == len(arrays["bars"]) == len(arrays["owner"])


def _write_atomically(path: Path, write) -> None:
    """``write(file)`` into ``path.tmp``, synced, then renamed over ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _chunk_tasks(slide_index: int, xy: np.ndarray, cls: np.ndarray, balls: list[np.ndarray], chunk: int) -> list[tuple]:
    """``_bars_chunk``'s arguments for consecutive runs of ``chunk`` centres, with each ball's cells concatenated."""
    tasks = []
    for start in range(0, len(balls), chunk):
        centres = np.arange(start, min(start + chunk, len(balls)))
        rows = [balls[i] for i in centres]
        offsets = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
        members = np.concatenate(rows)
        tasks.append((slide_index, centres, offsets, xy[members], cls[members]))
    return tasks


def _bars_slide(slide: pd.Series, out: Path, workers: int, chunk: int) -> dict:
    """Compute, assemble and write one slide's bars; returns its summary (the ``.json``)."""
    started, t0 = datetime.now(), time.perf_counter()
    slide_id, slide_index = str(slide["slide_id"]), int(slide["slide_index"])
    xy, cls = c.read_slide(slide_id)
    t_read = time.perf_counter() - t0
    centres = c.draw_centres(len(xy), slide_index)
    tree = cKDTree(xy)
    balls = [np.sort(np.asarray(rows, dtype=np.int64)) for rows in tree.query_ball_point(xy[centres], r=c.DELTA_PX)]
    tasks = _chunk_tasks(slide_index, xy, cls, balls, chunk)
    t_pool = time.perf_counter()
    results: list[dict | None] = [None] * len(tasks)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_warm) as pool:
        futures = {pool.submit(_bars_chunk, *task): k for k, task in enumerate(tasks)}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                k = futures[future]
                results[k] = future.result()
                n = len(results[k]["centres"])
                print(
                    f"  {slide_id}: chunk {k + 1}/{len(tasks)} done ({done} of {len(tasks)}), {n} centres in "
                    f"{results[k]['seconds'].sum():.0f} s, pid {results[k]['pid']}, {time.perf_counter() - t0:.0f} s",
                    flush=True,
                )
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    t_pool = time.perf_counter() - t_pool

    def joined(key: str) -> np.ndarray:
        return np.concatenate([result[key] for result in results])

    if not np.array_equal(joined("centres"), np.arange(len(centres))):
        raise RuntimeError(f"{slide_id}: the chunks do not cover centres 0..{len(centres) - 1} in order")
    bars, owner = joined("bars"), joined("owner")
    n_owners = len(centres) * (c.R + 1) * len(c.BLOCKS)
    if np.any(np.diff(owner) < 0) or (len(owner) and owner[-1] >= n_owners):
        raise RuntimeError(f"{slide_id}: owners out of order or out of range")
    arrays = {
        "centre_rows": centres,
        "centre_xy": xy[centres],
        **{key: joined(key) for key in ("n_cells", "class_counts", "pair_counts", "n_edges", "seconds")},
        "bars": bars,
        "owner": owner,
        "offsets": np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=n_owners))]).astype(np.int64),
        "codomain": joined("codomain"),
        "codomain_owner": joined("codomain_owner"),
    }
    if not np.array_equal(arrays["class_counts"].sum(axis=1), arrays["n_cells"]):
        raise RuntimeError(f"{slide_id}: class counts do not add up to the balls' cells")
    workers_cpu: dict[int, float] = {}
    for result in results:
        workers_cpu[result["pid"]] = max(workers_cpu.get(result["pid"], 0.0), result["cpu_seconds"])
    chunk_seconds = sum(result["finished"] - result["started"] for result in results)
    n_cells, seconds = arrays["n_cells"], arrays["seconds"]
    t_write = time.perf_counter()
    npz = out / f"{slide_id}.npz"
    tmp = npz.with_name(npz.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    summary = {
        "slide_id": slide_id,
        "slide_index": slide_index,
        "condition": str(slide["condition"]),
        "half": str(slide["half"]),
        "slide_cells": len(xy),
        "centres": len(centres),
        "labellings": c.R + 1,
        "blocks": len(c.BLOCKS),
        "delta_px": c.DELTA_PX,
        "workers": workers,
        "chunk": chunk,
        "start_method": "spawn",
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "read_seconds": round(t_read, 1),
        "pool_seconds": round(t_pool, 1),
        "pool_busy_share": round(chunk_seconds / (workers * t_pool), 3),
        "ball_cells": {
            "min": int(n_cells.min()),
            "median": float(np.median(n_cells)),
            "max": int(n_cells.max()),
            "max_centre": int(np.argmax(n_cells)),
        },
        "seconds_per_centre": {
            "median": round(float(np.median(seconds)), 3),
            "max": round(float(seconds.max()), 3),
            "max_centre": int(np.argmax(seconds)),
            "total": round(float(seconds.sum()), 1),
        },
        "worker_cpu_seconds": round(sum(workers_cpu.values()), 1),
        "worker_peak_rss_gb": round(max(result["peak_rss"] for result in results) / 1e9, 3),
        "parent_peak_rss_gb": round(_peak_rss() / 1e9, 3),
        "bars": len(bars),
        "codomain_bars": len(arrays["codomain"]),
        "npz_bytes": tmp.stat().st_size,
        "versions": {"python": platform.python_version(), **{name: version(name) for name in VERSIONED}},
        "topocyte_commit": c.TOPOCYTE_COMMIT,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    summary["write_seconds"] = round(time.perf_counter() - t_write, 1)
    _write_atomically(out / f"{slide_id}.json", lambda f: f.write((json.dumps(summary, indent=1) + "\n").encode()))
    os.replace(tmp, npz)  # last: a slide counts as done only once its npz is in place
    return summary


def cmd_bars(
    slides: Sequence[str] | None = None,
    workers: int = 8,
    *,
    pilot: pd.DataFrame | None = None,
    out: Path | None = None,
    chunk: int = CHUNK,
) -> dict[str, str]:
    """Write ``results/bars/{slide_id}.npz`` and ``.json`` for pilot ``slides`` (default all), skipping complete ones.

    Returns each slide's status, "skipped" or "computed". ``pilot`` and
    ``out`` default to ``results/pilot.csv`` and ``results/bars``.
    """
    pilot = pd.read_csv(c.RESULTS / "pilot.csv") if pilot is None else pilot
    out = c.RESULTS / "bars" if out is None else Path(out)
    if slides is not None:
        unknown = sorted(set(slides) - set(pilot["slide_id"]))
        if unknown:
            raise ValueError(f"not pilot slides: {unknown}")
        pilot = pilot[pilot["slide_id"].isin(slides)]
    out.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    todo = []
    for _, slide in pilot.sort_values("slide_index").iterrows():
        if bars_complete(out / f"{slide['slide_id']}.npz"):
            status[slide["slide_id"]] = "skipped"
            print(f"{slide['slide_id']}: complete, skipped", flush=True)
        else:
            todo.append(slide)
    if todo:
        _warm()  # compile or load the numba kernels once here, so that the workers only load them
    start = time.perf_counter()
    for n, slide in enumerate(todo, 1):
        where = f"slide index {slide['slide_index']}, {slide['condition']}"
        print(f"{slide['slide_id']} ({where}): {n} of {len(todo)}", flush=True)
        s = _bars_slide(slide, out, workers, chunk)
        status[s["slide_id"]] = "computed"
        balls, timing = s["ball_cells"], s["seconds_per_centre"]
        print(
            f"{s['slide_id']}: {s['slide_cells']} cells; balls {balls['min']}-{balls['max']} cells "
            f"(median {balls['median']:.0f}); {timing['median']:.2f} s per centre (max {timing['max']:.2f}); "
            f"wall {s['wall_seconds']:.0f} s, busy {s['pool_busy_share']:.2f}; "
            f"worker peak RSS {s['worker_peak_rss_gb']:.2f} GB; {s['bars']} bars, {s['npz_bytes'] / 1e9:.2f} GB; "
            f"{time.perf_counter() - start:.0f} s so far",
            flush=True,
        )
    elapsed = time.perf_counter() - start
    print(f"bars: {len(todo)} computed, {len(status) - len(todo)} skipped, in {elapsed:.0f} s", flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pilot", help="draw the 30 pilot slides: results/pilot.csv")
    sub.add_parser("preflight", help="check the rule for k on synthetic halves: results/preflight.json")
    bars = sub.add_parser("bars", help="six-packs and pair counts of every centre: results/bars/{slide_id}.npz, .json")
    bars.add_argument("--slides", nargs="+", metavar="SLIDE_ID", help="pilot slides to run (default: all 30)")
    bars.add_argument("--workers", type=int, default=8, help="spawned worker processes (default 8)")
    args = parser.parse_args()
    c.check_topocyte()
    c.RESULTS.mkdir(exist_ok=True)
    if args.command == "pilot":
        cmd_pilot()
    elif args.command == "preflight":
        if not cmd_preflight():
            raise SystemExit(1)
    elif args.command == "bars":
        cmd_bars(args.slides, args.workers)


if __name__ == "__main__":
    main()
