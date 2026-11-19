#!/usr/bin/env python
"""How close the random-feature kernel is to gudhi's exact PersistenceScaleSpaceKernel.

On slice 9 (the densest decile, the smallest diagrams), with the bandwidths
and random features that ``run_pssk_slices.py`` saved:

* ``equivalence``: for each dimension, the kernel of two diagrams (the first
  300 finite bars of the two slides with the fewest kept cells) through
  gudhi's own ``kernel_approx`` route and through this study's once-per-slide
  features;
* ``exact``: the six slides with the fewest kept cells, their diagrams
  recomputed; each slide's features recomputed against its saved row, and
  the 6 x 6 approximate kernel (from the saved rows) against gudhi's exact
  kernel: the relative error of every entry, the absolute error of the
  cosine-normalized off-diagonal entries, and how full the exact entries are
  (their median share of the largest value an entry of two diagrams of these
  sizes can take, every Gaussian at its peak);
* ``feature_count``: the H1 error's median over the same 36 entries with
  1000, 4000, 16000 and 64000 features, five draws each (random_state 0-4);
* ``stability``: kernel PCA of all 594 slides at slice 9 from three other
  draws (1000 features with random_state 1; 4000 features with random_state
  0 and 1), for each dimension, raw and cosine: the absolute correlation of
  PC1-4 with the saved scores, and the fractions of variance of PC1-4.

Writes ``results/validation.json``.  Run with ``PYTHONPATH=<topocyte
checkout>`` after ``run_pssk_slices.py slice 9``; it opens the cohort read
only.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import run_pssk_slices as r

SLICE = 9
N_SMALL = 6
N_EQUIVALENCE = 300
FEATURE_COUNTS = (1000, 4000, 16000, 64000)
SEEDS = range(5)
DRAWS = ((1000, 1), (4000, 0), (4000, 1))
NAMES = {(0, "none"): "h0_none", (1, "none"): "h1_none", (0, "cosine"): "h0_cosine", (1, "cosine"): "h1_cosine"}


def sampler_for(bandwidth: float, n_features: int, seed: int):
    from sklearn.kernel_approximation import RBFSampler

    return RBFSampler(gamma=1.0 / (2.0 * bandwidth**2), n_components=n_features, random_state=seed).fit(np.zeros((1, 2)))


def exact_kernel(diagrams: list[np.ndarray], bandwidth: float) -> np.ndarray:
    """gudhi's exact kernel matrix, each pair once."""
    from gudhi.representations.kernel_methods import PersistenceScaleSpaceKernel

    kernel = PersistenceScaleSpaceKernel(bandwidth=bandwidth)
    n = len(diagrams)
    matrix = np.empty((n, n))
    for a in range(n):
        for b in range(a, n):
            matrix[a, b] = matrix[b, a] = kernel(diagrams[a], diagrams[b])
    return matrix


def entry_errors(approx: np.ndarray, exact: np.ndarray, ceiling: np.ndarray | None = None) -> dict[str, float]:
    """Errors of the approximate kernel matrix; with ``ceiling``, also the exact entries' median share of it."""
    relative = np.abs(approx - exact) / np.abs(exact)
    off = ~np.eye(len(exact), dtype=bool)
    cos_exact = exact / np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
    cos_approx = approx / np.sqrt(np.outer(np.diag(approx), np.diag(approx)))
    return {
        "relative_error_median": float(np.median(relative)),
        "relative_error_max": float(relative.max()),
        "cosine_abs_error_max": float(np.abs(cos_approx - cos_exact)[off].max()),
        "exact_cosine_min": float(cos_exact[off].min()),
        "exact_cosine_max": float(cos_exact[off].max()),
        **({} if ceiling is None else {"fill_median": float(np.median(exact / ceiling))}),
    }


def main() -> None:
    import gudhi
    from gudhi.representations.kernel_methods import PersistenceScaleSpaceKernel

    started = time.time()
    with np.load(r.RESULTS / f"slice_{SLICE}.npz", allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    with np.load(r.SOURCE / "density.npz", allow_pickle=False) as density:
        bounds = density["bounds"]
        keep = density["rho"] >= density["deciles"][SLICE]
        slide_ids = [str(s) for s in density["slide_ids"]]
    if [str(s) for s in saved["slide_ids"]] != slide_ids:
        raise RuntimeError("slice 9 and density.npz list the slides differently")
    kept = saved["kept_per_slide"]
    small = np.argsort(kept, kind="stable")[:N_SMALL]

    coords = r.read_coords()

    def compute(code: int) -> list[np.ndarray]:
        start, end = bounds[code], bounds[code + 1]
        cloud = coords[start:end][keep[start:end]].astype(np.float64)
        result = r.tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)
        finite = []
        for dim in (0, 1):
            diagram = np.asarray(result.diagrams[dim], dtype=np.float64)
            finite.append(diagram[np.isfinite(diagram[:, 1])])
        return finite

    clock = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        diagrams = list(pool.map(compute, range(r.N_SLIDES)))
    coords = keep = None
    ph_seconds = time.time() - clock
    if not np.array_equal([[len(d) for d in pair] for pair in diagrams], saved["n_finite"]):
        raise RuntimeError("recomputed finite bar counts differ from the saved ones")
    print(f"PH of all slides at slice {SLICE} in {ph_seconds:.1f} s; finite bars as saved", flush=True)

    report: dict[str, object] = {
        "slice": SLICE,
        "gudhi": str(gudhi.__version__),
        "small_slides": {
            "slide_ids": [slide_ids[c] for c in small],
            "kept_cells": [int(kept[c]) for c in small],
            "h0_finite_bars": [int(saved["n_finite"][c, 0]) for c in small],
            "h1_finite_bars": [int(saved["n_finite"][c, 1]) for c in small],
        },
    }
    for dim in (0, 1):
        clock = time.time()
        bandwidth = float(saved["bandwidth"][dim])
        constant = 0.5 / (np.sqrt(2.0 * np.pi) * bandwidth)
        sampler = r.make_sampler(bandwidth)
        if not (np.array_equal(sampler.random_weights_, saved[f"h{dim}_random_weights"])
                and np.array_equal(sampler.random_offset_, saved[f"h{dim}_random_offset"])):
            raise RuntimeError(f"H{dim}: the sampler does not draw the saved random features")
        pieces = [diagrams[c][dim] for c in small]

        first, second = pieces[0][:N_EQUIVALENCE], pieces[1][:N_EQUIVALENCE]
        through_gudhi = float(PersistenceScaleSpaceKernel(bandwidth=bandwidth, kernel_approx=sampler)(first, second))
        ours = float(constant * r.features(first, sampler) @ r.features(second, sampler))

        rows = np.array([np.sqrt(constant) * r.features(piece, sampler) for piece in pieces])
        row_deviation = float(np.max(np.abs(rows - saved[f"h{dim}_features"][small]) / np.abs(rows).max()))

        exact = exact_kernel(pieces, bandwidth)
        approx = saved[f"h{dim}_features"][small] @ saved[f"h{dim}_features"][small].T
        # The largest an entry can be: every one of the 2|D| x 2|E| Gaussians at its peak, all of one sign.
        sizes = np.array([len(piece) for piece in pieces], dtype=np.float64)
        ceiling = 2.0 * np.outer(sizes, sizes) / (np.sqrt(2.0 * np.pi) * bandwidth)
        report[f"h{dim}"] = {
            "bandwidth_px2": bandwidth,
            "equivalence": {
                "bars_each": [len(first), len(second)],
                "gudhi_kernel_approx": through_gudhi,
                "features": ours,
                "relative_difference": abs(through_gudhi - ours) / abs(through_gudhi),
            },
            "saved_rows_relative_deviation": row_deviation,
            "exact": entry_errors(approx, exact, ceiling),
        }
        print(f"H{dim}: {json.dumps(report[f'h{dim}'])} [{time.time() - clock:.1f} s]", flush=True)

        if dim == 1:
            clock = time.time()
            sweep = {}
            for n_features in FEATURE_COUNTS:
                errors = []
                for seed in SEEDS:
                    draw = sampler_for(bandwidth, n_features, seed)
                    psi = np.array([np.sqrt(constant) * r.features(piece, draw) for piece in pieces])
                    errors.append(entry_errors(psi @ psi.T, exact)["relative_error_median"])
                sweep[str(n_features)] = {"per_seed": errors, "mean": float(np.mean(errors))}
                print(f"  H1, {n_features} features: median relative error per draw {np.round(errors, 4)}", flush=True)
            report["h1"]["feature_count"] = sweep
            print(f"  [{time.time() - clock:.1f} s]", flush=True)

    stability: dict[str, list] = {name: [] for name in NAMES.values()}
    for dim in (0, 1):
        bandwidth = float(saved["bandwidth"][dim])
        constant = 0.5 / (np.sqrt(2.0 * np.pi) * bandwidth)
        finite = [pair[dim] for pair in diagrams]
        for n_features, seed in DRAWS:
            clock = time.time()
            draw = sampler_for(bandwidth, n_features, seed)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                phi = np.array(list(pool.map(lambda d: r.features(d, draw), finite)))
            psi = np.sqrt(constant) * phi
            for normalization in ("none", "cosine"):
                name = NAMES[(dim, normalization)]
                rows = psi if normalization == "none" else psi / np.linalg.norm(psi, axis=1, keepdims=True)
                scores, evr = r.kernel_pca(rows)
                corr = [abs(float(np.corrcoef(scores[:, j], saved[f"{name}_scores"][:, j])[0, 1])) for j in range(4)]
                stability[name].append({
                    "n_features": n_features,
                    "seed": seed,
                    "abs_corr_pc1_4": corr,
                    "fve_pc1_4": [float(v) for v in evr[:4]],
                })
                print(f"  {name}, {n_features} features, seed {seed}: |corr| PC1-4 {np.round(corr, 3)}, "
                      f"FVE {np.round(evr[:4], 3)}", flush=True)
            print(f"  [{time.time() - clock:.1f} s]", flush=True)
    report["stability"] = {
        "saved": {name: [float(v) for v in saved[f"{name}_evr"][:4]] for name in NAMES.values()},
        "draws": stability,
    }
    report["ph_seconds"] = round(ph_seconds, 1)
    report["runtime_seconds"] = round(time.time() - started, 1)
    (r.RESULTS / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote results/validation.json in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
