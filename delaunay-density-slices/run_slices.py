#!/usr/bin/env python
"""Density slices of the GTEx liver cohort, one slice per invocation.

The function-Delaunay bifiltration of Alonso et al. (SODA 2024) is
topologically equivalent to the sublevel-offset bifiltration, so the slice at
density threshold ``d`` has the barcode of the alpha filtration of the cells
with ``rho >= d``.  This script therefore computes each slice with topocyte's
existing alpha pipeline; no bifiltration library is used.

Subcommands
-----------
``density``
    Compute the 10-nearest-neighbour density of every cell, the pooled
    deciles, and the per-slide summaries in ``results/density.npz``.
``slice K``
    Compute persistent homology, the four fPCA models, the 20-seeded
    classifications and the density-only control for slice ``K`` (0..9), and
    save ``results/slice_K.npz`` and ``figures/slice_K.png``.

Run with ``PYTHONPATH=<topocyte checkout>`` and an interpreter with topocyte's
dependencies; the script asserts that it imported ``topocyte`` from that
checkout (README.md names the commits).  It reads the shared zarr store
read-only and writes only below its own directory.
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
import zarr
import anndata as ad
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# topocyte comes from the source checkout that PYTHONPATH names first, not from
# whatever the interpreter has installed; README.md names the commits the
# results were made with.
_CHECKOUT = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
if not _CHECKOUT:
    raise RuntimeError("run with PYTHONPATH=<topocyte checkout>; see README.md")
TOPOCYTE_SRC = Path(_CHECKOUT).expanduser().resolve()
import topocyte as tc

if not Path(tc.__file__).resolve().is_relative_to(TOPOCYTE_SRC):
    raise RuntimeError(
        f"imported topocyte from {tc.__file__}, not from the checkout "
        f"{TOPOCYTE_SRC} that PYTHONPATH names"
    )

STORE = Path(
    "/Users/simon/data/topocyte_data/data/gtex_liver_cells_healthy_cirrhosis/"
    "gtex_liver_cells_healthy_cirrhosis.zarr"
)
PIXEL_UM = 0.4942  # level-0 pixel size assumed for every slide; see the report
K_NN = 10
DECILES = np.arange(0.0, 1.0, 0.1)
SEEDS = np.arange(20)
MODELS = (
    ("h0_none", 0, "none"),
    ("h1_none", 1, "none"),
    ("h0_cosine", 0, "cosine"),
    ("h1_cosine", 1, "cosine"),
)
CLASSES = ["healthy", "cirrhosis"]

warnings.filterwarnings(
    "ignore", message="Transforming to str index", category=Warning
)


def _read_cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Read coordinates and the two obs columns straight from the store."""
    group = zarr.open_group(store=str(STORE), mode="r")
    slide_codes = np.asarray(group["obs/slide_id/codes"][:], dtype=np.int16)
    slide_categories = [str(c) for c in group["obs/slide_id/categories"][:]]
    label_codes = np.asarray(group["obs/label/codes"][:], dtype=np.int8)
    label_categories = [str(c) for c in group["obs/label/categories"][:]]
    coords = np.ascontiguousarray(group["obsm/spatial"][:], dtype=np.float32)
    if coords.shape[0] != slide_codes.size:
        raise RuntimeError("obsm/spatial and obs have different lengths")
    return coords, slide_codes, label_codes, slide_categories, label_categories


def _group_bounds(codes: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Row indices of every code, in code order, keeping the row order inside."""
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    return [
        order[
            np.searchsorted(sorted_codes, code, side="left") :
            np.searchsorted(sorted_codes, code, side="right")
        ]
        for code in range(n_groups)
    ]


def cmd_density() -> None:
    started = time.time()
    coords, slide_codes, label_codes, slide_categories, label_categories = _read_cohort()
    n_slides = len(slide_categories)
    n_cells = coords.shape[0]
    rho = np.empty(n_cells, dtype=np.float64)
    slide_counts = np.zeros(n_slides, dtype=np.int64)
    slide_medians = np.zeros(n_slides, dtype=np.float64)
    slide_labels = np.empty(n_slides, dtype=object)

    for code, rows in enumerate(_group_bounds(slide_codes, n_slides)):
        points = np.ascontiguousarray(coords[rows], dtype=np.float64)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=K_NN + 1, workers=-1)
        r10_px = distances[:, K_NN]  # column 0 is the cell itself
        if not np.all(np.isfinite(r10_px)) or np.any(r10_px <= 0):
            raise RuntimeError(
                f"slide {slide_categories[code]}: r10 is not finite and positive"
            )
        r10_um = r10_px * PIXEL_UM
        # K_NN cells per disk of radius r10; one mm^2 is (1000 um)^2.
        rho[rows] = K_NN * 1e6 / (np.pi * r10_um**2)
        slide_counts[code] = rows.size
        slide_medians[code] = np.median(rho[rows])
        labels = np.unique(label_codes[rows])
        if labels.size != 1:
            raise RuntimeError(f"slide {slide_categories[code]} has mixed labels")
        slide_labels[code] = label_categories[int(labels[0])]
        print(
            f"  {slide_categories[code]:>18}  {rows.size:>8,} cells  "
            f"median {slide_medians[code]:8.1f} cells/mm^2  {slide_labels[code]}",
            flush=True,
        )

    deciles = np.quantile(rho, DECILES)
    np.savez(
        RESULTS / "density.npz",
        rho=rho,
        deciles=deciles,
        slide_codes=slide_codes,
        slide_categories=np.array(slide_categories),
        label_codes=label_codes,
        label_categories=np.array(label_categories),
        slide_labels=np.array([str(label) for label in slide_labels]),
        slide_counts=slide_counts,
        slide_medians=slide_medians,
        pixel_um=np.float64(PIXEL_UM),
        k_nn=np.int64(K_NN),
    )

    print(f"\n{rho.size:,} cells on {n_slides} slides in {time.time() - started:.1f} s")
    print("pooled density deciles [cells/mm^2]:")
    print("  " + np.array2string(deciles, precision=1))
    print(f"cells per slide: {slide_counts.min():,} .. {slide_counts.max():,}")
    for k, d in enumerate(deciles):
        kept = rho >= d
        fractions = [
            float(np.mean(kept[rows])) for rows in _group_bounds(slide_codes, n_slides)
        ]
        fractions = np.array(fractions)
        line = (
            f"  q={DECILES[k]:.1f} d={d:8.1f}  kept {kept.sum():>10,} "
            f"({kept.mean():5.1%})  median slide fraction cirrhosis/healthy "
            f"{np.median(fractions[slide_labels == 'cirrhosis']):.2f}/"
            f"{np.median(fractions[slide_labels == 'healthy']):.2f}  "
            f"min kept/slide {min(kept[rows].sum() for rows in _group_bounds(slide_codes, n_slides)):,}"
        )
        print(line)


def _load_density() -> dict[str, np.ndarray]:
    with np.load(RESULTS / "density.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _slice_anndata(
    density: dict[str, np.ndarray], keep: np.ndarray
) -> ad.AnnData:
    slide_categories = [str(c) for c in density["slide_categories"]]
    label_categories = [str(c) for c in density["label_categories"]]
    label_strings = np.array(label_categories, dtype=object)[density["label_codes"]]
    obs = pd.DataFrame(
        {
            "slide_id": pd.Categorical.from_codes(
                density["slide_codes"][keep], slide_categories
            ),
            # Plot both cohorts in this order; fpca_classify is given the pair
            # explicitly, so the column's category order does not code the classes.
            "label": pd.Categorical(
                label_strings[keep], categories=CLASSES, ordered=False
            ),
        }
    )
    coords = np.ascontiguousarray(
        zarr.open_group(store=str(STORE), mode="r")["obsm/spatial"][:][keep],
        dtype=np.float32,
    )
    return ad.AnnData(obs=obs, obsm={"spatial": coords})


def _fit_and_classify(sub: ad.AnnData) -> dict[str, np.ndarray]:
    started = time.time()
    tc.tl.persistent_homology(
        sub,
        filtration="alpha",
        max_dim=1,
        groupby="slide_id",
        microns_per_unit=PIXEL_UM,
        n_jobs=-1,
        key_added="slide_ph",
    )
    print(f"  PH done in {time.time() - started:.1f} s", flush=True)
    slides = tc.tl.read_persistence(sub, "slide_ph")
    if len(slides) != 123 or slides.skipped:
        raise RuntimeError(
            f"expected 123 computed slides, got {len(slides)} "
            f"(skipped: {list(slides.skipped)})"
        )

    group_ids = None
    out: dict[str, np.ndarray] = {}
    for name, dim, normalization in MODELS:
        key = f"fpca_{name}"
        started = time.time()
        tc.tl.fpca(
            sub,
            ph_key="slide_ph",
            dim=dim,
            normalization=normalization,
            key_added=key,
            n_jobs=-1,
        )
        model = sub.uns[key]
        model_group_ids = np.array([str(g) for g in model["group_ids"]])
        if group_ids is None:
            group_ids = model_group_ids
        elif not np.array_equal(group_ids, model_group_ids):
            raise RuntimeError(f"{key}: group_ids differ from the first model")
        out[f"{name}_scores"] = np.asarray(model["scores"], dtype=np.float64)
        out[f"{name}_evr"] = np.asarray(
            model["explained_variance_ratio"], dtype=np.float64
        )
        print(
            f"  fPCA {name} done in {time.time() - started:.1f} s, "
            f"EVR[0:4]={out[f'{name}_evr'][:4].round(3)}",
            flush=True,
        )

    accuracies = {name: np.empty(len(SEEDS)) for name, _, _ in MODELS}
    confusions = {name: np.empty((len(SEEDS), 2, 2), dtype=np.int64) for name, _, _ in MODELS}
    y = None
    started = time.time()
    for name, _, _ in MODELS:
        key = f"fpca_{name}"
        for seed in SEEDS:
            tc.tl.fpca_classify(
                sub,
                fpca_key=key,
                label_key="label",
                classes=CLASSES,
                n_components=4,
                scoring="accuracy",
                random_state=int(seed),
                key_added=f"label_s{seed}",
            )
            entry = sub.uns[key]["classifiers"][f"label_s{seed}"]
            accuracies[name][seed] = float(entry["cv"]["accuracy"])
            confusions[name][seed] = np.asarray(
                entry["cv"]["confusion_matrix"], dtype=np.int64
            )
            if y is None:
                y = np.asarray(entry["y"], dtype=np.int8)
            elif not np.array_equal(y, np.asarray(entry["y"], dtype=np.int8)):
                raise RuntimeError(f"{key} seed {seed}: y differs between models")
    print(f"  classification done in {time.time() - started:.1f} s", flush=True)
    out["group_ids"] = group_ids
    out["y"] = y
    for name, _, _ in MODELS:
        out[f"{name}_accuracy"] = accuracies[name]
        out[f"{name}_confusion"] = confusions[name]
    return out


def _control(
    kept_per_slide: np.ndarray, slide_total: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The density-only control, mirroring fpca_classify's cross-validation."""
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from topocyte.tl.fpca import _logistic

    X = (kept_per_slide / slide_total)[:, None]
    used = y >= 0
    accuracies = np.empty(len(SEEDS))
    confusions = np.empty((len(SEEDS), 2, 2), dtype=np.int64)
    for seed in SEEDS:
        held_out = cross_val_predict(
            _logistic(10, 5, "accuracy", 1000, int(seed)),
            X[used],
            y[used],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed)),
        )
        accuracies[seed] = float(np.mean(held_out == y[used]))
        confusions[seed] = confusion_matrix(y[used], held_out, labels=[0, 1])
    return accuracies, confusions


def cmd_slice(index: int) -> None:
    started = time.time()
    density = _load_density()
    n_cells = density["rho"].size
    threshold = float(density["deciles"][index])
    keep = np.ones(n_cells, dtype=bool) if index == 0 else density["rho"] >= threshold
    kept_total = int(keep.sum())
    print(
        f"slice {index}: d={threshold:.4f} cells/mm^2, keeps {kept_total:,} "
        f"of {n_cells:,} cells",
        flush=True,
    )
    sub = _slice_anndata(density, keep)
    del density
    n_groups = sub.obs["slide_id"].nunique()
    if n_groups != 123:
        raise RuntimeError(f"slice {index} keeps cells of {n_groups} slides, not 123")
    counts = sub.obs["slide_id"].value_counts()
    if counts.min() < 100:
        raise RuntimeError(f"slice {index}: a slide keeps fewer than 100 cells")

    out = _fit_and_classify(sub)
    group_ids = out["group_ids"]
    with np.load(RESULTS / "density.npz", allow_pickle=False) as density_file:
        label_by_slide = dict(
            zip(
                [str(c) for c in density_file["slide_categories"]],
                [str(c) for c in density_file["slide_labels"]],
            )
        )
        totals_lookup = dict(
            zip(
                [str(c) for c in density_file["slide_categories"]],
                map(int, density_file["slide_counts"]),
            )
        )
    kept_per_slide = counts.reindex(group_ids).to_numpy().astype(np.int64)
    slide_total = np.array([totals_lookup[slide] for slide in group_ids], dtype=np.int64)
    kept_fraction = kept_per_slide / slide_total
    group_label = np.array([label_by_slide[slide] for slide in group_ids], dtype=object)
    if not np.array_equal(
        np.array([0 if label == "healthy" else 1 for label in group_label]),
        out["y"],
    ):
        raise RuntimeError(f"slice {index}: y does not match the group labels")

    control_accuracy, control_confusion = _control(kept_per_slide, slide_total, out["y"])
    print(
        f"  control: mean {control_accuracy.mean():.3f} ± {control_accuracy.std():.3f}",
        flush=True,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, dim in enumerate((0, 1)):
        for col, normalization in enumerate(("none", "cosine")):
            name = f"h{dim}_{normalization}"
            tc.pl.fpca(
                sub,
                key=f"fpca_{name}",
                color="label",
                components=(0, 1),
                ax=axes[row, col],
            )
            axes[row, col].set_title(f"H{dim}, {normalization}")
    if index == 0:
        title = f"Density slice 0: all cells — {kept_total:,} of {n_cells:,} cells"
    else:
        title = (
            f"Density slice {index}: rho >= {threshold:.0f} cells/mm^2 — "
            f"{kept_total:,} of {n_cells:,} cells"
        )
    fig.suptitle(title)
    fig.savefig(FIGURES / f"slice_{index:02d}.png", dpi=140)
    plt.close(fig)

    payload = {
        "slice_index": np.int64(index),
        "threshold": np.float64(threshold),
        "kept_total": np.int64(kept_total),
        "n_cells_total": np.int64(n_cells),
        "group_ids": np.array(group_ids),
        "group_label": np.array([str(label) for label in group_label]),
        "slide_total": slide_total,
        "kept_per_slide": kept_per_slide,
        "kept_fraction": kept_fraction,
        "seeds": SEEDS,
        "control_accuracy": control_accuracy,
        "control_confusion": control_confusion,
        **out,
    }
    np.savez(RESULTS / f"slice_{index}.npz", **payload)
    meta = {
        "slice": index,
        "threshold_cells_per_mm2": threshold,
        "kept_total": kept_total,
        "n_cells_total": n_cells,
        "runtime_seconds": round(time.time() - started, 1),
        "pixel_um": PIXEL_UM,
        "n_jobs": -1,
        "seeds": int(len(SEEDS)),
        "n_components": 4,
        "scoring": "accuracy",
        "classes": CLASSES,
        "models": [name for name, _, _ in MODELS],
        "sklearn": __import__("sklearn").__version__,
        "topocyte": tc.__version__ if hasattr(tc, "__version__") else "unknown",
    }
    (RESULTS / f"slice_{index}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"slice {index} done in {time.time() - started:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("density", "slice"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.command == "density":
        cmd_density()
    else:
        if args.index is None or not 0 <= args.index <= 9:
            parser.error("slice needs an index 0..9")
        cmd_slice(args.index)


if __name__ == "__main__":
    main()
