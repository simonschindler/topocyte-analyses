#!/usr/bin/env python
"""Density slices of the 594-slide GTEx liver cohort, Betti-curve fPCA by pathology.

The function-Delaunay bifiltration of Alonso et al. (SODA 2024) is
topologically equivalent to the sublevel-offset bifiltration, so the slice at
density threshold ``d`` has the barcode of the alpha filtration of the cells
with ``rho >= d``.  Each slice is computed with topocyte's existing alpha
pipeline, as in the 123-slide density-slice study.

Subcommands
-----------
``groups``
    Assign every slide one pathology group, by severity, from the cohort CSV:
    ``results/pathology.csv``.
``density``
    The 10-nearest-neighbour density of every cell, the pooled deciles and a
    fine quantile curve: ``results/density.npz``.
``slice K``
    Alpha persistent homology of the cells with ``rho`` at or above decile
    ``K`` (0..9; slice 0 keeps every cell), and the four Betti-curve fPCA
    models (H0/H1, raw/cosine) with nine components each:
    ``results/slice_K.npz`` and ``.json``.

Run with ``PYTHONPATH=<topocyte checkout>`` and an interpreter with topocyte's
dependencies; the script checks that it imported ``topocyte`` from that
checkout (README.md names the commit).  It opens the cohort read-only (h5py
mode "r", pandas for the CSV) and writes only below its own directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# topocyte comes from the source checkout that PYTHONPATH names first, not from
# whatever the interpreter has installed; README.md names the commit.
_CHECKOUT = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
if not _CHECKOUT:
    raise RuntimeError("run with PYTHONPATH=<topocyte checkout>; see README.md")
TOPOCYTE_SRC = Path(_CHECKOUT).expanduser().resolve()
import topocyte as tc  # noqa: E402

if not Path(tc.__file__).resolve().is_relative_to(TOPOCYTE_SRC):
    raise RuntimeError(
        f"imported topocyte from {tc.__file__}, not from the checkout {TOPOCYTE_SRC} that PYTHONPATH names"
    )

COHORT = Path("/Users/simon/data/gtex_liver_cells/gtex_liver_cells")
H5AD = COHORT / "gtex_liver_cells.h5ad"
CSV = COHORT / "liver_cohort.csv"
N_CELLS = 126_219_124
N_SLIDES = 594
PIXEL_UM = 0.4942  # level-0 pixel size assumed for every slide; units only
K_NN = 10
DECILES = np.arange(0.0, 1.0, 0.1)
CURVE_LEVELS = np.linspace(0.0, 1.0, 1001)
N_COMPONENTS = 9
MODELS = (
    ("h0_none", 0, "none"),
    ("h1_none", 1, "none"),
    ("h0_cosine", 0, "cosine"),
    ("h1_cosine", 1, "cosine"),
)
# Most severe first; a slide takes the first group any of its categories maps to.
SEVERITY = (
    ("cirrhosis", ("cirrhosis",)),
    ("fibrosis", ("fibrosis", "scarring", "sclerotic")),
    ("inflammation/hepatitis", ("inflammation", "hepatitis")),
    ("necrosis", ("necrosis", "infarction", "ischemic_changes")),
    ("steatosis", ("steatosis",)),
    ("congestion", ("congestion", "hemorrhage")),
)
GROUP_ORDER = (
    "healthy",
    "congestion",
    "steatosis",
    "necrosis",
    "inflammation/hepatitis",
    "fibrosis",
    "cirrhosis",
    "unclassified",
)

warnings.filterwarnings("ignore", message="Transforming to str index", category=Warning)


def _strings(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _read_slides() -> tuple[np.ndarray, list[str], np.ndarray]:
    """Slide code of every cell, the slide names, and each slide's first row (plus the end)."""
    import h5py

    with h5py.File(H5AD, "r") as f:
        codes = np.asarray(f["obs/slide_id/codes"][:], dtype=np.int16)
        slides = _strings(f["obs/slide_id/categories"][:])
    if codes.size != N_CELLS or len(slides) != N_SLIDES:
        raise RuntimeError(f"expected {N_CELLS:,} cells on {N_SLIDES} slides, got {codes.size:,} on {len(slides)}")
    if np.any(np.diff(codes) < 0):
        raise RuntimeError("cells are not grouped by slide code")
    bounds = np.searchsorted(codes, np.arange(len(slides) + 1), side="left")
    if np.any(np.diff(bounds) <= 0):
        raise RuntimeError("a slide has no cells")
    return codes, slides, bounds


def _read_coords() -> np.ndarray:
    import h5py

    with h5py.File(H5AD, "r") as f:
        coords = np.asarray(f["obsm/spatial"][:], dtype=np.float32)
    if coords.shape != (N_CELLS, 2) or not np.all(np.isfinite(coords)):
        raise RuntimeError(f"obsm/spatial has shape {coords.shape} or non-finite values")
    return coords


def assign_group(categories: str | float, label: str) -> str:
    """The slide's pathology group: curated healthy, the most severe category, or unclassified."""
    if label == "healthy":
        return "healthy"
    present = {c.strip() for c in str(categories).split(",") if c.strip()} if isinstance(categories, str) else set()
    for group, members in SEVERITY:
        if present & set(members):
            return group
    if present:
        raise RuntimeError(f"categories {sorted(present)} map to no group")
    return "unclassified"


def cmd_groups() -> None:
    _, slides, bounds = _read_slides()
    cohort = pd.read_csv(CSV)
    rows = cohort.set_index("Tissue Sample ID").reindex(slides)
    if rows["label"].isna().any():
        raise RuntimeError("a slide with cells has no row in the cohort CSV")
    table = pd.DataFrame(
        {
            "slide_id": slides,
            "group": [assign_group(c, l) for c, l in zip(rows["Pathology Categories"], rows["label"])],
            "label": rows["label"].to_numpy(),
            "label_reason": rows["label_reason"].to_numpy(),
            "categories": rows["Pathology Categories"].fillna("").to_numpy(),
            "notes": rows["Pathology Notes"].fillna("").to_numpy(),
            "quality_flag": rows["quality_flag"].fillna("").to_numpy(),
            "n_cells": np.diff(bounds),
        }
    )
    if not set(table["group"]) <= set(GROUP_ORDER):
        raise RuntimeError(f"unexpected groups {set(table['group']) - set(GROUP_ORDER)}")
    table.to_csv(RESULTS / "pathology.csv", index=False)
    counts = table["group"].value_counts().reindex(GROUP_ORDER)
    print(counts.to_string())
    print(f"wrote {RESULTS / 'pathology.csv'}: {len(table)} slides")


def cmd_density() -> None:
    from scipy.spatial import cKDTree

    started = time.time()
    codes, slides, bounds = _read_slides()
    coords = _read_coords()
    rho = np.empty(N_CELLS, dtype=np.float64)
    medians = np.empty(N_SLIDES)
    for code in range(N_SLIDES):
        start, end = bounds[code], bounds[code + 1]
        points = np.ascontiguousarray(coords[start:end], dtype=np.float64)
        distances, _ = cKDTree(points).query(points, k=K_NN + 1, workers=-1)
        r10_um = distances[:, K_NN] * PIXEL_UM  # column 0 is the cell itself
        if not np.all(np.isfinite(r10_um)) or np.any(r10_um <= 0):
            raise RuntimeError(f"slide {slides[code]}: r10 is not finite and positive")
        # K_NN cells per disk of radius r10; one mm^2 is (1000 um)^2.
        rho[start:end] = K_NN * 1e6 / (np.pi * r10_um**2)
        medians[code] = np.median(rho[start:end])
        if code % 50 == 0:
            print(f"  {code + 1}/{N_SLIDES} slides, {time.time() - started:.0f} s", flush=True)
    del coords
    deciles = np.quantile(rho, DECILES)
    curve = np.quantile(rho, CURVE_LEVELS)
    kept = np.array(
        [[int(np.count_nonzero(rho[bounds[c]:bounds[c + 1]] >= d)) for c in range(N_SLIDES)] for d in deciles]
    )
    np.savez(
        RESULTS / "density.npz",
        rho=rho,
        slide_codes=codes,
        slide_ids=np.array(slides),
        bounds=bounds,
        deciles=deciles,
        decile_levels=DECILES,
        curve=curve,
        curve_levels=CURVE_LEVELS,
        kept_per_slide=kept,
        slide_medians=medians,
        pixel_um=np.float64(PIXEL_UM),
        k_nn=np.int64(K_NN),
    )
    meta = {"runtime_seconds": round(time.time() - started, 1), "n_cells": N_CELLS, "n_slides": N_SLIDES}
    (RESULTS / "density.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{N_CELLS:,} cells on {N_SLIDES} slides in {time.time() - started:.1f} s")
    for k, d in enumerate(deciles):
        print(
            f"  q={DECILES[k]:.1f} d={d:8.1f} cells/mm^2: kept {kept[k].sum():>11,}, "
            f"per slide min {kept[k].min():,} median {int(np.median(kept[k])):,}, "
            f"slides under 100 cells {int(np.sum(kept[k] < 100))}",
            flush=True,
        )


def cmd_slice(index: int) -> None:
    """PH per slide with topocyte's core, then the four cohort fPCA models.

    ``tc.tl.persistent_homology`` maps ``tc.ph.compute_persistent_homology``
    over the groups' coordinates (in row order, as float64, without unit
    scaling) and ``tc.tl.fpca`` fits ``tc.ph.FPCA`` on the groups' results;
    this calls the same two directly.  An AnnData of all 126 million cells
    would build a string index of 126 million names, more than this machine
    holds.
    """
    from concurrent.futures import ThreadPoolExecutor

    started = time.time()
    with np.load(RESULTS / "density.npz", allow_pickle=False) as density:
        threshold = float(density["deciles"][index])
        slides = [str(s) for s in density["slide_ids"]]
        bounds = density["bounds"]
        kept_per_slide = density["kept_per_slide"][index]
        keep = np.ones(N_CELLS, dtype=bool) if index == 0 else density["rho"] >= threshold
    if not np.array_equal([int(np.count_nonzero(keep[bounds[c]:bounds[c + 1]])) for c in range(N_SLIDES)], kept_per_slide):
        raise RuntimeError(f"slice {index}: kept cells differ from density.npz")
    coords = _read_coords()
    print(f"slice {index}: d={threshold:.1f} cells/mm^2, {int(kept_per_slide.sum()):,} cells", flush=True)

    def compute(code: int) -> object:
        start, end = bounds[code], bounds[code + 1]
        cloud = coords[start:end][keep[start:end]].astype(np.float64)
        return tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)

    # Each worker holds one slide's alpha complex; on this 16 GB machine the
    # slices above 70 million cells run five at a time instead of ten.
    workers = os.cpu_count() if int(kept_per_slide.sum()) <= 70_000_000 else 5
    clock = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compute, range(N_SLIDES)))
    ph_seconds = time.time() - clock
    coords = keep = None  # free them; compute() is done with both
    n_bars = np.array([[len(r.diagrams[d]) for d in (0, 1)] for r in results], dtype=np.int64)
    print(f"  PH in {ph_seconds:.1f} s; bars H0 {n_bars[:, 0].sum():,}, H1 {n_bars[:, 1].sum():,}", flush=True)

    payload: dict[str, np.ndarray] = {
        "slice_index": np.int64(index),
        "threshold": np.float64(threshold),
        "slide_ids": np.array(slides),
        "kept_per_slide": kept_per_slide,
        "n_bars": n_bars,
        "skipped": np.array([], dtype=str),
    }
    timings: dict[str, object] = {"ph_seconds": round(ph_seconds, 1), "ph_workers": workers}
    # H1 first; once a dimension's two models are fitted its diagrams are
    # dropped, so the H0 fits, the largest, run without the H1 bars in memory.
    fit_order = sorted(MODELS, key=lambda model: -model[1])
    for position, (name, dim, normalization) in enumerate(fit_order):
        if position and fit_order[position - 1][1] != dim:
            for result in results:
                result.diagrams.pop(fit_order[position - 1][1], None)
        clock = time.time()
        model = tc.ph.FPCA(
            dim=dim, n_components=N_COMPONENTS, normalization=normalization, n_jobs=-1
        ).fit(results)
        result = model.result_
        payload[f"{name}_group_ids"] = np.array(slides)
        payload[f"{name}_scores"] = np.asarray(result.scores, dtype=np.float64)
        payload[f"{name}_evr"] = np.asarray(result.explained_variance_ratio, dtype=np.float64)
        solver = str(result.params.get("eigen_solver", ""))
        timings[f"{name}_seconds"] = round(time.time() - clock, 1)
        timings[f"{name}_solver"] = solver
        print(
            f"  fPCA {name} ({solver}) in {time.time() - clock:.1f} s: "
            f"{payload[f'{name}_scores'].shape[0]} slides, FVE[0:4]={payload[f'{name}_evr'][:4].round(3)}",
            flush=True,
        )
        del model, result
    np.savez(RESULTS / f"slice_{index}.npz", **payload)
    meta = {
        "slice": index,
        "threshold_cells_per_mm2": threshold,
        "n_cells_kept": int(kept_per_slide.sum()),
        "n_components": N_COMPONENTS,
        "pixel_um": PIXEL_UM,
        "topocyte": str(tc.__version__),
        "runtime_seconds": round(time.time() - started, 1),
        **timings,
    }
    (RESULTS / f"slice_{index}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/slice_{index}.npz in {time.time() - started:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("groups")
    sub.add_parser("density")
    one = sub.add_parser("slice")
    one.add_argument("index", type=int, choices=range(10))
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.command == "groups":
        cmd_groups()
    elif args.command == "density":
        cmd_density()
    else:
        cmd_slice(args.index)


if __name__ == "__main__":
    main()
