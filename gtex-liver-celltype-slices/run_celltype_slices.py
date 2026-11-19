#!/usr/bin/env python
"""Per-cell-group alpha persistent homology and Betti-curve fPCA at the sibling study's density slices.

The sibling study (``../gtex-liver-pathology-slices``) cut every slide at ten
thresholds of the all-cell 10-nearest-neighbour density ``rho``; this study
keeps the same cells per slice and splits them by HistoPLUS cell type into
four groups (``common.GROUP_MEMBERS``).

Subcommands
-----------
``groups``
    Count every slide's cells of each group kept at each slice:
    ``results/composition.npz`` (the counts must add up to the sibling's
    ``kept_per_slide``).
``slice K``
    For each group, alpha persistent homology of every slide's cells of that
    group in slice ``K``, then the four Betti-curve fPCA models (H0/H1,
    raw/cosine), nine components each, fitted exactly as the sibling's
    ``run_pathology_slices.py`` fits them: ``results/slice_K.npz`` and
    ``.json``. A cosine model that a zero curve makes undefined is recorded
    in ``dropped_blocks`` instead of fitted.
``gate K``
    The same code path on all cells of slice ``K``, compared with the
    sibling's saved scores: ``results/gate_K.json``.

Run with ``PYTHONPATH=<topocyte checkout at bd1d5c0>`` and that checkout's
interpreter; the script refuses a topocyte from anywhere else or a checkout
whose package differs from the commit. It opens the cohort read only and
writes only below its own directory.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import common as c

TOPOCYTE_COMMIT = "bd1d5c0"
_CHECKOUT = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
if not _CHECKOUT:
    raise RuntimeError("run with PYTHONPATH=<topocyte checkout>; see README.md")
TOPOCYTE_SRC = Path(_CHECKOUT).expanduser().resolve()
import topocyte as tc  # noqa: E402

if not Path(tc.__file__).resolve().is_relative_to(TOPOCYTE_SRC):
    raise RuntimeError(f"imported topocyte from {tc.__file__}, not from the checkout {TOPOCYTE_SRC}")
_HEAD = subprocess.run(["git", "-C", str(TOPOCYTE_SRC), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
_DIRTY = subprocess.run(
    ["git", "-C", str(TOPOCYTE_SRC), "status", "--porcelain", "--", "topocyte"], capture_output=True, text=True, check=True
).stdout.strip()
if not _HEAD.startswith(TOPOCYTE_COMMIT) or _DIRTY:
    raise RuntimeError(f"topocyte checkout is at {_HEAD[:7]} (package changed: {bool(_DIRTY)}), not a clean {TOPOCYTE_COMMIT}")

warnings.filterwarnings("ignore", message="Transforming to str index", category=Warning)
MAX_CELLS_ALL_WORKERS = 70_000_000
GATE_MIN_ABS_CORR = 0.999999
GATE_MAX_FVE_DIFF = 1e-6


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def _density(index: int) -> tuple[float, list[str], np.ndarray, np.ndarray]:
    with np.load(c.SOURCE / "density.npz", allow_pickle=False) as density:
        threshold = float(density["deciles"][index])
        slides = [str(s) for s in density["slide_ids"]]
        bounds = density["bounds"]
        keep = c.slice_mask(density["rho"], density["deciles"], index)
    return threshold, slides, bounds, keep


def _per_slide(mask: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.array([int(np.count_nonzero(mask[bounds[s]:bounds[s + 1]])) for s in range(c.N_SLIDES)])


def fit_point_set(coords: np.ndarray, mask: np.ndarray, bounds: np.ndarray, label: str) -> tuple[dict, dict, list[str]]:
    """PH of every slide's cells in ``mask``, then the four fPCA models; the sibling's code path."""
    kept = _per_slide(mask, bounds)
    workers = os.cpu_count() if int(kept.sum()) <= MAX_CELLS_ALL_WORKERS else 5

    def compute(code: int) -> object:
        start, end = bounds[code], bounds[code + 1]
        cloud = coords[start:end][mask[start:end]].astype(np.float64)
        return tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)

    clock = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compute, range(c.N_SLIDES)))
    ph_seconds = time.time() - clock
    n_bars = np.array([[len(r.diagrams[d]) for d in (0, 1)] for r in results], dtype=np.int64)
    empty_h1 = int(np.sum(n_bars[:, 1] == 0))
    print(f"  {label}: {int(kept.sum()):,} cells, PH in {ph_seconds:.1f} s ({workers} workers); "
          f"bars H0 {n_bars[:, 0].sum():,}, H1 {n_bars[:, 1].sum():,}; slides without H1 bars {empty_h1}", flush=True)
    arrays: dict[str, np.ndarray] = {f"{label}_n_bars": n_bars, f"{label}_kept": kept}
    timings: dict[str, object] = {"ph_seconds": round(ph_seconds, 1), "ph_workers": workers, "slides_without_h1": empty_h1}
    dropped: list[str] = []
    # H1 first; once both H1 models are fitted the H1 diagrams are dropped (sibling order).
    fit_order = sorted(c.MODELS, key=lambda model: -model[1])
    for position, (name, dim, normalization) in enumerate(fit_order):
        if position and fit_order[position - 1][1] != dim:
            for result in results:
                result.diagrams.pop(fit_order[position - 1][1], None)
        clock = time.time()
        try:
            model = tc.ph.FPCA(dim=dim, n_components=c.N_COMPONENTS, normalization=normalization, n_jobs=-1).fit(results)
        except ValueError as error:
            if normalization == "cosine" and "zero-norm" in str(error):
                dropped.append(f"{label}:{name}")
                print(f"  {label} {name}: dropped, {error}", flush=True)
                continue
            raise
        result = model.result_
        arrays[f"{label}_{name}_scores"] = np.asarray(result.scores, dtype=np.float64)
        arrays[f"{label}_{name}_evr"] = np.asarray(result.explained_variance_ratio, dtype=np.float64)
        timings[f"{name}_seconds"] = round(time.time() - clock, 1)
        timings[f"{name}_solver"] = str(result.params.get("eigen_solver", ""))
        print(f"  {label} {name} ({timings[f'{name}_solver']}) in {time.time() - clock:.1f} s: "
              f"FVE[0:4]={arrays[f'{label}_{name}_evr'][:4].round(3)}", flush=True)
        del model, result
    return arrays, timings, dropped


def cmd_groups() -> None:
    started = time.time()
    slides, bounds = c.read_slides()
    groups, categories = c.read_cell_groups()
    with np.load(c.SOURCE / "density.npz", allow_pickle=False) as density:
        if [str(s) for s in density["slide_ids"]] != slides or not np.array_equal(density["bounds"], bounds):
            raise RuntimeError("the sibling's density.npz lists the slides differently")
        rho, deciles, sibling_kept = density["rho"], density["deciles"], density["kept_per_slide"]
    kept = np.zeros((c.N_SLICES, c.N_SLIDES, len(c.CELL_GROUPS)), dtype=np.int64)
    for index in range(c.N_SLICES):
        keep = c.slice_mask(rho, deciles, index)
        for s in range(c.N_SLIDES):
            start, end = bounds[s], bounds[s + 1]
            kept[index, s] = np.bincount(groups[start:end][keep[start:end]], minlength=len(c.CELL_GROUPS))
    if not np.array_equal(kept.sum(axis=2), sibling_kept):
        raise RuntimeError("group counts do not add up to the sibling's kept_per_slide")
    np.savez(
        c.RESULTS / "composition.npz",
        kept_by_group=kept,
        group_names=np.array(c.CELL_GROUPS),
        slide_ids=np.array(slides),
        categories=np.array(categories),
        category_group=c.group_codes(categories),
    )
    shares = kept[0].sum(axis=0) / kept[0].sum()
    print("slice 0 shares: " + ", ".join(f"{g} {100 * v:.2f}%" for g, v in zip(c.CELL_GROUPS, shares)))
    for index in range(c.N_SLICES):
        row = ", ".join(f"{g} min {kept[index, :, j].min():,} median {int(np.median(kept[index, :, j])):,}"
                        for j, g in enumerate(c.CELL_GROUPS))
        print(f"  slice {index}: {row}")
    meta = {"runtime_seconds": round(time.time() - started, 1), "min_cells_per_group_and_slide": int(kept.min())}
    (c.RESULTS / "composition.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/composition.npz in {time.time() - started:.1f} s")


def cmd_slice(index: int) -> None:
    started = time.time()
    threshold, slides, bounds, keep = _density(index)
    with np.load(c.RESULTS / "composition.npz", allow_pickle=False) as composition:
        expected = composition["kept_by_group"][index]
    groups, _ = c.read_cell_groups()
    coords = c.read_coords()
    print(f"slice {index}: d={threshold:.1f} cells/mm^2, {int(np.count_nonzero(keep)):,} cells", flush=True)
    payload: dict[str, np.ndarray] = {
        "slice_index": np.int64(index),
        "threshold": np.float64(threshold),
        "slide_ids": np.array(slides),
    }
    meta: dict[str, object] = {"slice": index, "threshold_cells_per_mm2": threshold, "topocyte": str(tc.__version__),
                               "topocyte_commit": _HEAD}
    dropped: list[str] = []
    for j, group in enumerate(c.CELL_GROUPS):
        mask = keep & (groups == j)
        if not np.array_equal(_per_slide(mask, bounds), expected[:, j]):
            raise RuntimeError(f"slice {index}, {group}: kept cells differ from composition.npz")
        arrays, timings, lost = fit_point_set(coords, mask, bounds, group)
        payload.update(arrays)
        meta[group] = timings
        dropped += lost
        del mask, arrays
    payload["dropped_blocks"] = np.array(dropped, dtype=str)
    np.savez(c.RESULTS / f"slice_{index}.npz", **payload)
    meta.update({"dropped_blocks": dropped, "peak_rss_gb": round(_peak_rss_gb(), 2),
                 "runtime_seconds": round(time.time() - started, 1)})
    (c.RESULTS / f"slice_{index}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/slice_{index}.npz in {time.time() - started:.1f} s; peak RSS {_peak_rss_gb():.1f} GB", flush=True)


def cmd_gate(index: int) -> None:
    started = time.time()
    _, slides, bounds, keep = _density(index)
    coords = c.read_coords()
    arrays, timings, dropped = fit_point_set(coords, keep, bounds, "all")
    report: dict[str, object] = {"slice": index, "topocyte_commit": _HEAD, "timings": timings, "dropped": dropped}
    passed = not dropped
    with np.load(c.SOURCE / f"slice_{index}.npz", allow_pickle=False) as sibling:
        if [str(s) for s in sibling["slide_ids"]] != slides:
            raise RuntimeError("the sibling's slice lists the slides differently")
        bars_equal = bool(np.array_equal(sibling["n_bars"], arrays["all_n_bars"]))
        passed &= bars_equal
        report["n_bars_equal"] = bars_equal
        for name in c.MODEL_NAMES:
            ours, theirs = arrays[f"all_{name}_scores"], sibling[f"{name}_scores"]
            corr = [abs(float(np.corrcoef(ours[:, k], theirs[:, k])[0, 1])) for k in range(c.N_COMPONENTS)]
            fve = float(np.max(np.abs(arrays[f"all_{name}_evr"] - sibling[f"{name}_evr"])))
            ok = min(corr) >= GATE_MIN_ABS_CORR and fve <= GATE_MAX_FVE_DIFF
            passed &= ok
            report[name] = {"min_abs_corr": min(corr), "abs_corr": corr, "max_fve_diff": fve, "passed": ok}
    report.update({"passed": bool(passed), "runtime_seconds": round(time.time() - started, 1)})
    (c.RESULTS / f"gate_{index}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"gate slice {index}: {'PASSED' if passed else 'FAILED'}; "
          + ", ".join(f"{n} min|r| {report[n]['min_abs_corr']:.9f} dFVE {report[n]['max_fve_diff']:.1e}"
                      for n in c.MODEL_NAMES if n in report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("groups")
    for command in ("slice", "gate"):
        one = sub.add_parser(command)
        one.add_argument("index", type=int, choices=range(c.N_SLICES))
    args = parser.parse_args()
    c.RESULTS.mkdir(exist_ok=True)
    if args.command == "groups":
        cmd_groups()
    elif args.command == "slice":
        cmd_slice(args.index)
    else:
        cmd_gate(args.index)


if __name__ == "__main__":
    main()
