"""Every pairwise matrix of the Betti FPCR benchmark, computed once, and the stage that checks them: gates G2 and G4.

The arms (``design.md``, "Pipeline") read, for each dataset of the three tasks and each dimension h ∈ {0, 1}:

- ``w2``: arm K's 2-Wasserstein distances between the diagrams (``wasserstein.w2_matrix``); for fluid, upstream's
  cached distances (``inputs.fluid_cached_w2``), which gate G2 checks;
- ``l2``: arm B's exact L² distances between the reduced Betti curves (``betti.l2_matrix``);
- ``gram``: arm F's exact Gram matrix of the same curves (``betti.gram``).

The Rips curves (interpolation, curvature) are not shifted; the fluid curves are shifted by ``FLUID_SHIFT``, which
gate G4 checks. ``results/matrices.npz`` holds the matrices under ``mkey``'s keys and the targets under
``interp_{idx:02d}_y``, ``curv_y`` and ``fluid_y``:

- interpolation dataset idx (0-99, in ``inputs.interpolation_datasets()`` order): (n_train + 50)², training clouds
  first, and their targets t;
- curvature: 100², training clouds 0-49 first, and the curvatures K;
- fluid time idx (the index in ``FLUID_TIMES``): 400², in ``inputs.fluid_sim_ids()`` order; one target, ln Re.

The stage, from this folder (about 9 min, under 1 GB)::

    caffeinate -is /usr/bin/time -l $PY matrices.py > logs/matrices.log 2>&1

first checks gate G2: ``wasserstein.w2`` against upstream's cached fluid distances on ``g2_sample``'s checks,
relative ``G2_TOL``. A failure goes to Simon (``design.md``, "Gates"), and nothing else is computed. It then computes
every matrix, and with the fluid ones gate G4: each fluid Gram matrix at ``FLUID_SHIFT`` against one at
``FLUID_SHIFT + 10``, whose largest difference must be at most ``G4_TOL`` times the matrix's largest entry (ruling 7
in ``progress.md``, as G3 is relative to the largest score). It writes ``results/matrices.json`` (the gates, the
shapes, the npz's size and sha256, the runtimes and ``provenance``), and the npz only if both gates pass; otherwise it
exits with status 1. ``load_matrices`` gives the later stages the npz only if this json vouches for it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Sequence

import numpy as np

import betti
import common as c
import inputs as inp
import wasserstein as ws

KINDS = ("w2", "l2", "gram")
# G2's sample (design.md: at least 100 random same-time pairs, at least 10 checks with more than 1,000 finite bars).
G2_PAIRS = 100
G2_LARGE = 10
G2_LARGE_BARS = 1_000  # a large check has more than this many finite bars on both sides
G2_MAX_BARS = 4_000  # a check of more finite bars, n + m, is not made: its cost matrix holds (n + m)² floats
G2_SEED = 2
G4_EXTRA_SHIFT = 10.0
_SCRIPTS = ("matrices.py", "common.py", "inputs.py", "safe_pickle.py", "wasserstein.py", "betti.py")


def mkey(task: str, kind: str, h: int, idx: int | None = None) -> str:
    """The npz key of a matrix: ``interp_{idx:02d}_{kind}_{h}``, ``curv_{kind}_{h}`` or ``fluid_{idx}_{kind}_{h}``.

    ``task`` is ``"interp"``, ``"curv"`` or ``"fluid"``, and ``kind`` one of ``KINDS``. ``idx`` is the interpolation
    dataset (0-99) or the fluid time's index in ``FLUID_TIMES``; curvature has none. ``ValueError`` on another task
    or kind.
    """
    if kind not in KINDS:
        raise ValueError(f"a matrix's kind is one of {KINDS}, not {kind!r}")
    match task:
        case "interp":
            return f"interp_{idx:02d}_{kind}_{h}"
        case "curv":
            return f"curv_{kind}_{h}"
        case "fluid":
            return f"fluid_{idx}_{kind}_{h}"
    raise ValueError(f"a task is 'interp', 'curv' or 'fluid', not {task!r}")


def dataset_matrices(
    diagrams: Sequence[inp.Diagram], shift: float = 0.0, with_w2: bool = True
) -> dict[str, np.ndarray]:
    """The matrices of one dataset's ``diagrams``, keyed ``"{kind}_{h}"``: ``w2_h``, ``l2_h`` and ``gram_h``.

    In each dimension h: W2 between the diagrams (serial ``wasserstein.w2_matrix``, essential bars kept) unless
    ``with_w2`` is false, and the L² distances and the Gram matrix of their reduced Betti curves, every endpoint
    plus ``shift``.
    """
    out = {}
    for h in (0, 1):
        bars = [d[h] for d in diagrams]
        if with_w2:
            out[f"w2_{h}"] = ws.w2_matrix(bars, n_jobs=1)
        curves = betti.curves(bars, h, shift)
        out[f"l2_{h}"] = betti.l2_matrix(curves, h)
        out[f"gram_{h}"] = betti.gram(curves, h)
    return out


def _keyed(task: str, matrices: dict[str, np.ndarray], idx: int | None = None) -> dict[str, np.ndarray]:
    """``matrices``, a ``dataset_matrices`` dict with all three kinds, under their npz keys."""
    return {mkey(task, kind, h, idx): matrices[f"{kind}_{h}"] for kind in KINDS for h in (0, 1)}


def g2_sample(bars: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    """Gate G2's checks, (time index, h, p, q) with p < q, from the finite-bar counts ``bars[t, h, i]``.

    Same-time pairs are drawn uniformly without replacement: ``rng.permutation`` of every (time, p < q). Each pair is
    checked in every dimension, except where its n + m finite bars exceed ``G2_MAX_BARS``. The sample ends at the
    first pair at which ``G2_PAIRS`` pairs have a check and ``G2_LARGE`` checks are large (more than
    ``G2_LARGE_BARS`` finite bars on both sides): 100 pairs, then more until ten checks are large. ``ValueError`` if
    every pair is drawn first.
    """
    n_times, n_dims, n_samples = bars.shape
    first, second = np.triu_indices(n_samples, 1)
    checks, n_pairs, n_large = [], 0, 0
    for k in rng.permutation(n_times * first.size):
        t, j = divmod(int(k), first.size)
        p, q = int(first[j]), int(second[j])
        kept = [h for h in range(n_dims) if bars[t, h, p] + bars[t, h, q] <= G2_MAX_BARS]
        checks += [(t, h, p, q) for h in kept]
        n_pairs += bool(kept)
        n_large += sum(int(min(bars[t, h, p], bars[t, h, q]) > G2_LARGE_BARS) for h in kept)
        if n_pairs >= G2_PAIRS and n_large >= G2_LARGE:
            return checks
    raise ValueError(
        f"all {n_times * first.size} same-time pairs give {n_pairs} pairs with a check and {n_large} large checks; "
        f"G2 needs {G2_PAIRS} and {G2_LARGE}"
    )


def _relative(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """|a − b| / |b|, entry by entry: 0 where a equals b (also where both are 0), inf where only b is 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.abs(a - b)
    return np.divide(diff, np.abs(b), out=np.where(diff > 0, np.inf, 0.0), where=b != 0)


def _g2(fluid: list[list[inp.Diagram]], cached: list[list[np.ndarray]]) -> dict:
    """Gate G2: ``wasserstein.w2`` against upstream's cached distances ``cached[t][h]`` on ``g2_sample``'s checks.

    ``fluid[t]`` holds the 400 diagrams at ``FLUID_TIMES[t]``, in ``fluid_sim_ids()`` order.
    """
    started = time.perf_counter()
    ids = inp.fluid_sim_ids()
    bars = np.array([[[np.isfinite(d[h][:, 1]).sum() for d in diagrams] for h in (0, 1)] for diagrams in fluid])
    checks = g2_sample(bars, np.random.default_rng(G2_SEED))
    rows = []
    for k, (t, h, p, q) in enumerate(checks, 1):
        mine, theirs = ws.w2(fluid[t][p][h], fluid[t][q][h]), float(cached[t][h][p, q])
        rows.append(
            {
                "time": c.FLUID_TIMES[t],
                "h": h,
                "p": p,
                "q": q,
                "sims": [ids[p], ids[q]],
                "finite_bars": [int(bars[t, h, p]), int(bars[t, h, q])],
                "w2": mine,
                "cached": theirs,
                "relative": float(_relative(mine, theirs)),
            }
        )
        if k % 50 == 0:
            print(f"G2: {k} of {len(checks)} checks ({time.perf_counter() - started:.0f} s)", flush=True)
    relative = np.array([row["relative"] for row in rows])
    large = np.array([min(row["finite_bars"]) > G2_LARGE_BARS for row in rows])
    sizes = np.array([sum(row["finite_bars"]) for row in rows])
    record = {
        "passed": bool(relative.max() <= c.G2_TOL),
        "tolerance": c.G2_TOL,
        "seed": G2_SEED,
        "pairs": len({(row["time"], row["p"], row["q"]) for row in rows}),
        "checks": len(rows),
        "large_checks": int(large.sum()),
        "n_plus_m": {"median": float(np.median(sizes)), "max": int(sizes.max())},
        "worst_relative": float(relative.max()),
        "worst_relative_large": float(relative[large].max()),
        "worst_check": rows[int(np.argmax(relative))],
        "failing_checks": [row for row in rows if row["relative"] > c.G2_TOL],
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(
        f"G2: {record['checks']} checks of {record['pairs']} pairs, {record['large_checks']} large; n + m median "
        f"{record['n_plus_m']['median']:g}, max {record['n_plus_m']['max']}; worst relative difference "
        f"{record['worst_relative']:.2e} (large {record['worst_relative_large']:.2e}), "
        f"{len(record['failing_checks'])} above {c.G2_TOL:g}: {'passed' if record['passed'] else 'FAILED'} "
        f"({record['seconds']} s)",
        flush=True,
    )
    return record


def _rips_matrices() -> dict[str, np.ndarray]:
    """The matrices and targets of the 100 interpolation datasets and of curvature: Rips diagrams, curves unshifted."""
    started = time.perf_counter()
    M = {}
    for idx, ds in enumerate(inp.interpolation_datasets()):
        M |= _keyed("interp", dataset_matrices([inp.rips_diagram(X) for X in ds["clouds"]]), idx)
        M[f"interp_{idx:02d}_y"] = ds["y"]
    print(f"interpolation: 100 datasets in {time.perf_counter() - started:.1f} s", flush=True)
    started = time.perf_counter()
    data = inp.curvature_data()
    M |= _keyed("curv", dataset_matrices([inp.rips_diagram(X) for X in data["clouds"]]))
    M["curv_y"] = data["y"]
    print(f"curvature: {time.perf_counter() - started:.1f} s", flush=True)
    return M


def _g4_measures(G: np.ndarray, G10: np.ndarray, bars: np.ndarray) -> dict:
    """Gate G4 in one fluid time and dimension: the Gram ``G`` at ``FLUID_SHIFT`` against ``G10``, at the shift + 10.

    It passes if the largest difference is at most ``G4_TOL`` times the largest entry of ``G`` (ruling 7). A part of a
    bar lost below 0 would change its curve's diagonal entry by at least the lost length, so the gate sees any loss
    above ``G4_TOL`` times the largest entry. ``entrywise``, for information only, holds the worst difference of an
    entry relative to itself: on the tiny entries of curves whose bars barely overlap, that is the rounding of the
    shifted endpoints. ``bars`` are the curves' finite-bar counts.
    """
    largest, diff = float(np.abs(G).max()), float(np.abs(G10 - G).max())
    worst = diff / largest if largest > 0 else (math.inf if diff > 0 else 0.0)
    relative = _relative(G10, G)
    p, q = (int(i) for i in np.unravel_index(np.argmax(relative), relative.shape))
    return {
        "passed": worst <= c.G4_TOL,
        "relative_to_largest_entry": worst,
        "largest_entry": largest,
        "entrywise": {
            "worst_relative": float(relative.max()),
            "entries_above_tolerance": int((relative > c.G4_TOL).sum()),
            "worst_entry": {"p": p, "q": q, "gram": float(G[p, q]), "finite_bars": [int(bars[p]), int(bars[q])]},
        },
    }


def _fluid_matrices(fluid: list[list[inp.Diagram]], cached: list[list[np.ndarray]]) -> tuple[dict, dict]:
    """The fluid matrices and targets, and gate G4: the curves shifted by ``FLUID_SHIFT``, W2 from ``cached``."""
    M, rows = {}, []
    for idx, (t, diagrams) in enumerate(zip(c.FLUID_TIMES, fluid, strict=True)):
        started = time.perf_counter()
        matrices = dataset_matrices(diagrams, shift=c.FLUID_SHIFT, with_w2=False)
        for h in (0, 1):
            matrices[f"w2_{h}"] = cached[idx][h]
            bars = [d[h] for d in diagrams]
            G10 = betti.gram(betti.curves(bars, h, c.FLUID_SHIFT + G4_EXTRA_SHIFT), h)
            counts = np.array([np.isfinite(a[:, 1]).sum() for a in bars])
            rows.append({"time": t, "h": h, **_g4_measures(matrices[f"gram_{h}"], G10, counts)})
        M |= _keyed("fluid", matrices, idx)
        print(
            f"fluid t = {t:g}: "
            + "; ".join(
                f"G4 H{row['h']} {row['relative_to_largest_entry']:.2e} of the largest entry "
                f"({'passed' if row['passed'] else 'FAILED'}; entry by entry {row['entrywise']['worst_relative']:.2e})"
                for row in rows[-2:]
            )
            + f" ({time.perf_counter() - started:.1f} s)",
            flush=True,
        )
    M["fluid_y"] = inp.fluid_ln_re(np.asarray(inp.fluid_sim_ids()))
    g4 = {
        "passed": all(row["passed"] for row in rows),
        "rule": "per time and dimension, max |G(s + 10) - G(s)| <= tolerance * max |G(s)| (ruling 7 in progress.md); "
        "entrywise is information only",
        "tolerance": c.G4_TOL,
        "shifts": [c.FLUID_SHIFT, c.FLUID_SHIFT + G4_EXTRA_SHIFT],
        "per_time_and_dimension": rows,
    }
    return M, g4


def _shapes(M: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """For each task, how many of ``M``'s arrays have each shape."""
    shapes: dict[str, Counter] = {}
    for key, a in M.items():
        shapes.setdefault(key.split("_")[0], Counter())["x".join(map(str, a.shape))] += 1
    return {task: dict(counts) for task, counts in shapes.items()}


def load_matrices() -> dict[str, np.ndarray]:
    """Every array of ``results/matrices.npz`` by its key, if ``results/matrices.json`` vouches for the file.

    ``RuntimeError`` unless the json records gates G2 and G4 passed and the npz's sha256 equals the one it records,
    so that an npz left by an earlier run, after a later run failed, is refused. The arrays are loaded, with
    ``allow_pickle=False``, from the bytes that were hashed.
    """
    data = (c.RESULTS / "matrices.npz").read_bytes()
    record = json.loads((c.RESULTS / "matrices.json").read_text())
    gates = {gate: (record.get(gate) or {}).get("passed") is True for gate in ("g2", "g4")}
    recorded, actual = (record.get("npz") or {}).get("sha256"), hashlib.sha256(data).hexdigest()
    if not all(gates.values()) or actual != recorded:
        raise RuntimeError(
            f"results/matrices.json does not vouch for results/matrices.npz: gates passed {gates}, npz sha256 "
            f"{actual[:12]}… against the recorded {str(recorded)[:12]}…; rerun matrices.py"
        )
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def main() -> None:
    c.check_topocyte()
    c.check_upstream()
    provenance = c.provenance(*(c.HERE / name for name in _SCRIPTS))  # at the start: the code that runs
    started = time.perf_counter()
    fluid = [inp.fluid_diagrams(t) for t in c.FLUID_TIMES]
    cached = [[inp.fluid_cached_w2(t, h) for h in (0, 1)] for t in c.FLUID_TIMES]
    seconds = {"fluid_inputs": round(time.perf_counter() - started, 1)}
    part = time.perf_counter()
    g2 = _g2(fluid, cached)
    seconds["g2"] = round(time.perf_counter() - part, 1)
    M, g4 = {}, None
    if g2["passed"]:
        part = time.perf_counter()
        M = _rips_matrices()
        seconds["interpolation_and_curvature"] = round(time.perf_counter() - part, 1)
        part = time.perf_counter()
        fluid_matrices, g4 = _fluid_matrices(fluid, cached)
        M |= fluid_matrices
        seconds["fluid_and_g4"] = round(time.perf_counter() - part, 1)
    passed = bool(g2["passed"] and g4 is not None and g4["passed"])
    c.RESULTS.mkdir(exist_ok=True)
    npz = None
    if passed:
        path = c.RESULTS / "matrices.npz"
        np.savez(path, **M)
        npz = {"arrays": len(M), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    seconds["total"] = round(time.perf_counter() - started, 1)
    record = {
        "description": "every pairwise matrix of the three tasks (W2, Betti L2, Betti Gram), computed once, and "
        "gates G2 (the scipy W2 against upstream's cached fluid distances) and G4 (the fluid Gram matrices at the "
        "shift and at the shift + 10) (design.md)",
        "passed": passed,
        "g2": g2,
        "g4": g4,
        "shapes": _shapes(M) if M else None,
        "npz": npz,
        "seconds": seconds,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance,
    }
    (c.RESULTS / "matrices.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"wrote results/matrices.json{' and results/matrices.npz' if passed else ''} in {seconds['total']} s: "
        f"{'passed' if passed else 'a gate FAILED'}",
        flush=True,
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
