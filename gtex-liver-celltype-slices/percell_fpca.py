#!/usr/bin/env python
"""Improvement idea: raw Betti curves per cell (beta / n), fitted exactly and matrix-free.

A raw H0 Betti curve starts at the number of points, so a group's raw model
mixes its topology with its cell count. The cosine model removes the curve's
whole magnitude. Dividing each slide's curve by the number of points in its
cloud keeps the magnitude relative to the cells: H0 becomes the fraction of
cells still apart at each scale, H1 the loops per cell.

The Gram matrix of the scaled curves is ``K = D^-1 G D^-1``, with ``G`` the
raw Gram matrix and ``D`` the diagonal of point counts. The fit needs only:
- products ``K x = D^-1 G (D^-1 x)``, which topocyte's exact matrix-free product with ``G`` (its ``_EndpointList``, from the bar endpoints) gives;
- the squared norms ``diag(G) / n^2``.

topocyte's own ``_fit_matrix_free`` (ARPACK, as in its raw fits) does the
rest. These are private functions of the pinned commit (bd1d5c0, which
``run_celltype_slices.py`` enforces on import). The dense route through
``betti_inner_product_matrix`` takes about 60 s per Gram matrix at slice 9
alone, which would be some four hours in all.

``slice K``: for each point set (all cells and the four groups), alpha
persistent homology of every slide's cloud as in ``run_celltype_slices.py``,
then the per-cell H0 and H1 fits, nine components each. At slice 9 the same
path with ``D = I`` must reproduce the main run's raw scores (the sibling's
for all cells). Writes ``results/percell_slice_K.npz`` and ``.json``.

``collect``: gathers the blocks into ``results/percell_blocks.npz`` (keys
``<set>:h<d>_percell:<slice>``), which ``improve.py`` reads.

Exploratory: written and run after the main readouts existed (``progress.md``).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common as c
import run_celltype_slices as run  # checks the topocyte checkout on import

tc = run.tc
from topocyte.ph.decomposition import _as_curves, _betti_operator, _fit_matrix_free  # noqa: E402

GATE_MIN_ABS_CORR = 0.999999


def scaled_fit(results: list, dim: int, scale: np.ndarray) -> object:
    """Matrix-free fPCA (nine components) of the curves divided by ``scale``, by topocyte's own solver."""
    squared, make_product = _betti_operator(_as_curves(results, dim))

    def make_scaled() -> object:
        product = make_product()
        return lambda x: product(np.asarray(x, dtype=np.float64) / scale) / scale

    result, _ = _fit_matrix_free(squared / scale**2, make_scaled, c.N_COMPONENTS, "none", "n", 1e-10)
    return result


def cmd_slice(index: int) -> None:
    started = time.time()
    _, slides, bounds, keep = run._density(index)
    groups, _ = c.read_cell_groups()
    coords = c.read_coords()
    payload: dict[str, np.ndarray] = {"slide_ids": np.array(slides)}
    meta: dict[str, object] = {"slice": index, "topocyte_commit": run._HEAD}
    gate: dict[str, float] = {}
    with np.load(c.SOURCE / f"slice_{index}.npz", allow_pickle=False) as sibling, \
            np.load(c.RESULTS / f"slice_{index}.npz", allow_pickle=False) as own:
        saved = {**{f"all_{m}_scores": sibling[f"{m}_scores"] for m in ("h0_none", "h1_none")},
                 **{key: own[key] for key in own.files if key.endswith("_none_scores")}}
    for label in c.POINT_SETS:
        mask = keep if label == "all" else keep & (groups == c.CELL_GROUPS.index(label))
        kept = run._per_slide(mask, bounds)
        workers = 10 if int(kept.sum()) <= run.MAX_CELLS_ALL_WORKERS else 5

        def compute(code: int, mask: np.ndarray = mask) -> object:
            start, end = bounds[code], bounds[code + 1]
            return tc.ph.compute_persistent_homology(coords[start:end][mask[start:end]].astype(np.float64),
                                                     filtration="alpha", max_dim=1)

        clock = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(compute, range(c.N_SLIDES)))
        timings: dict[str, float] = {"ph_seconds": round(time.time() - clock, 1)}
        n = kept.astype(np.float64)
        payload[f"{label}_n"] = kept
        for dim in (1, 0):
            clock = time.time()
            result = scaled_fit(results, dim, n)
            payload[f"{label}_h{dim}_percell_scores"] = np.asarray(result.scores, dtype=np.float64)
            payload[f"{label}_h{dim}_percell_evr"] = np.asarray(result.explained_variance_ratio, dtype=np.float64)
            timings[f"h{dim}_seconds"] = round(time.time() - clock, 1)
            if index == 9:  # gate: D = I is the main run's raw fit
                raw = np.asarray(scaled_fit(results, dim, np.ones_like(n)).scores)
                theirs = saved[f"{label}_h{dim}_none_scores"]
                gate[f"{label}:h{dim}"] = min(abs(float(np.corrcoef(raw[:, j], theirs[:, j])[0, 1]))
                                              for j in range(c.N_COMPONENTS))
            for item in results:
                item.diagrams.pop(dim, None)
        meta[label] = timings
        print(f"  slice {index} {label}: {timings}", flush=True)
        del results
    if gate:
        meta["gate_min_abs_corr"] = gate
        if min(gate.values()) < GATE_MIN_ABS_CORR:
            raise RuntimeError(f"identity scaling does not reproduce the raw fits: {gate}")
    np.savez(c.RESULTS / f"percell_slice_{index}.npz", **payload)
    meta["runtime_seconds"] = round(time.time() - started, 1)
    (c.RESULTS / f"percell_slice_{index}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/percell_slice_{index}.npz in {time.time() - started:.1f} s", flush=True)


def cmd_collect() -> None:
    blocks = {}
    for k in range(c.N_SLICES):
        with np.load(c.RESULTS / f"percell_slice_{k}.npz", allow_pickle=False) as data:
            for label in c.POINT_SETS:
                for dim in (0, 1):
                    blocks[f"{label}:h{dim}_percell:{k}"] = data[f"{label}_h{dim}_percell_scores"]
                    blocks[f"{label}:h{dim}_percell:{k}__evr"] = data[f"{label}_h{dim}_percell_evr"]
    np.savez(c.RESULTS / "percell_blocks.npz", **blocks)
    print(f"wrote results/percell_blocks.npz: {len(blocks) // 2} blocks")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("slice")
    one.add_argument("index", type=int, choices=range(c.N_SLICES))
    sub.add_parser("collect")
    args = parser.parse_args()
    if args.command == "slice":
        cmd_slice(args.index)
    else:
        cmd_collect()


if __name__ == "__main__":
    main()
