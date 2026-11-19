#!/usr/bin/env python
"""Density slices of the 594-slide GTEx liver cohort with the persistence scale-space kernel.

The sibling study ``../gtex-liver-pathology-slices`` fitted Betti-curve fPCA
at ten density slices; this one repeats it with gudhi's
``PersistenceScaleSpaceKernel`` (Reininghaus et al., CVPR 2015) in place of
the Betti-curve inner product.  It reuses that study's groups
(``results/pathology.csv``), densities and slice levels
(``results/density.npz``) and recomputes persistent homology as it did.

gudhi 3.13 defines the kernel of two diagrams as

    k(D1, D2) = 0.5 * sum_{p in DD1, q in DD2} w(p) w(q) exp(-|p - q|^2 / (2 h^2)) / (sqrt(2 pi) h)

with ``DD = D u mirror(D)`` and ``w = +1`` on or above the diagonal, ``-1``
below.  An exact entry costs ``|DD1| |DD2|`` Gaussians (about 10^11 here), so
the kernel goes through gudhi's own approximation route, random Fourier
features (scikit-learn's ``RBFSampler``, ``gamma = 1 / (2 h^2)``, 1000
features): each slide gets ``phi(D) = sum_{p in DD} w(p) z(p)``, and
``k(D1, D2) ~ c phi(D1) . phi(D2)`` with ``c = 0.5 / (sqrt(2 pi) h)``, which is
what gudhi returns with ``kernel_approx``, but computed once per slide instead
of once per pair.

``slice K``: PH of the cells with ``rho`` at or above decile K, the bandwidth
``h`` (per dimension, the median persistence of the finite bars pooled over
all slides), the features, and kernel PCA (raw and cosine) with nine
components saved: ``results/slice_K.npz`` and ``.json``.

Run with ``PYTHONPATH=<topocyte checkout>``; the script checks that it
imported ``topocyte`` from that checkout.  It opens the cohort read only and
writes only below its own directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = HERE.parent / "gtex-liver-pathology-slices" / "results"

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
N_CELLS = 126_219_124
N_SLIDES = 594
N_FEATURES = 1000
SEED = 0
N_COMPONENTS = 9
CHUNK = 10_000
MODELS = (
    ("h0_none", 0, "none"),
    ("h1_none", 1, "none"),
    ("h0_cosine", 0, "cosine"),
    ("h1_cosine", 1, "cosine"),
)


def read_coords() -> np.ndarray:
    import h5py

    with h5py.File(H5AD, "r") as f:
        coords = np.asarray(f["obsm/spatial"][:], dtype=np.float32)
    if coords.shape != (N_CELLS, 2) or not np.all(np.isfinite(coords)):
        raise RuntimeError(f"obsm/spatial has shape {coords.shape} or non-finite values")
    return coords


def make_sampler(bandwidth: float):
    """The fitted RBFSampler of gudhi's kernel_approx route for this bandwidth."""
    from sklearn.kernel_approximation import RBFSampler

    sampler = RBFSampler(gamma=1.0 / (2.0 * bandwidth**2), n_components=N_FEATURES, random_state=SEED)
    return sampler.fit(np.zeros((1, 2)))


def features(diagram: np.ndarray, sampler) -> np.ndarray:
    """phi(D) = sum over D and its mirror of w(p) z(p), w = +1 on or above the diagonal, else -1."""
    total = np.zeros(sampler.n_components)
    for start in range(0, len(diagram), CHUNK):
        block = diagram[start:start + CHUNK]
        for points in (block, block[:, ::-1]):
            weight = np.where(points[:, 1] >= points[:, 0], 1.0, -1.0)
            total += weight @ sampler.transform(points)
    return total


def kernel_pca(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scores and fractions of variance of the centred rows, signs fixed."""
    centred = psi - psi.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    signs[signs == 0] = 1.0
    variance = s**2
    return (u * signs) * s, variance / variance.sum()


def cmd_slice(index: int) -> None:
    started = time.time()
    with np.load(SOURCE / "density.npz", allow_pickle=False) as density:
        threshold = float(density["deciles"][index])
        slides = [str(s) for s in density["slide_ids"]]
        bounds = density["bounds"]
        kept_per_slide = density["kept_per_slide"][index]
        keep = np.ones(N_CELLS, dtype=bool) if index == 0 else density["rho"] >= threshold
    with np.load(SOURCE / f"slice_{index}.npz", allow_pickle=False) as betti:
        betti_bars = betti["n_bars"]
        if [str(s) for s in betti["slide_ids"]] != slides:
            raise RuntimeError("the Betti study lists the slides differently")
    coords = read_coords()
    print(f"slice {index}: d={threshold:.1f} cells/mm^2, {int(kept_per_slide.sum()):,} cells", flush=True)

    def compute(code: int) -> object:
        start, end = bounds[code], bounds[code + 1]
        cloud = coords[start:end][keep[start:end]].astype(np.float64)
        return tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)

    workers = os.cpu_count() if int(kept_per_slide.sum()) <= 70_000_000 else 5
    clock = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compute, range(N_SLIDES)))
    ph_seconds = time.time() - clock
    coords = keep = None  # free them; compute() is done with both
    n_bars = np.array([[len(r.diagrams[d]) for d in (0, 1)] for r in results], dtype=np.int64)
    # Gate: the same diagrams as the Betti study, slide by slide.
    if not np.array_equal(n_bars, betti_bars):
        raise RuntimeError(f"slice {index}: bar counts differ from the Betti study's")
    print(f"  PH in {ph_seconds:.1f} s; bars equal the Betti study's", flush=True)

    payload: dict[str, np.ndarray] = {
        "slice_index": np.int64(index),
        "threshold": np.float64(threshold),
        "slide_ids": np.array(slides),
        "kept_per_slide": kept_per_slide,
        "n_bars": n_bars,
        "skipped": np.array([], dtype=str),
    }
    timings: dict[str, object] = {"ph_seconds": round(ph_seconds, 1), "ph_workers": workers}
    bandwidths = np.empty(2)
    n_finite = np.empty((N_SLIDES, 2), dtype=np.int64)
    psi = {}
    for dim in (0, 1):
        clock = time.time()
        finite = []
        for code, result in enumerate(results):
            diagram = np.asarray(result.diagrams[dim], dtype=np.float64)
            diagram = diagram[np.isfinite(diagram[:, 1])]
            if np.any(diagram[:, 1] <= diagram[:, 0]):
                raise RuntimeError(f"slice {index} H{dim}: a finite bar without positive persistence")
            finite.append(diagram)
            n_finite[code, dim] = len(diagram)
            result.diagrams.pop(dim)  # the finite copy is all that is needed from here on
        persistence = np.concatenate([d[:, 1] - d[:, 0] for d in finite])
        bandwidth = float(np.median(persistence))
        del persistence
        bandwidths[dim] = bandwidth
        sampler = make_sampler(bandwidth)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            phi = np.array(list(pool.map(lambda d: features(d, sampler), finite)))
        del finite
        constant = 0.5 / (np.sqrt(2.0 * np.pi) * bandwidth)
        psi[dim] = np.sqrt(constant) * phi  # K = psi psi^T approximates gudhi's kernel
        payload[f"h{dim}_features"] = psi[dim]
        payload[f"h{dim}_random_weights"] = sampler.random_weights_
        payload[f"h{dim}_random_offset"] = sampler.random_offset_
        timings[f"h{dim}_feature_seconds"] = round(time.time() - clock, 1)
        print(f"  H{dim}: h = {bandwidth:.4g} px^2, features in {time.time() - clock:.1f} s", flush=True)
    payload["bandwidth"] = bandwidths
    payload["n_finite"] = n_finite

    for name, dim, normalization in MODELS:
        rows = psi[dim]
        if normalization == "cosine":
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        scores, evr = kernel_pca(rows)
        payload[f"{name}_group_ids"] = np.array(slides)
        payload[f"{name}_scores"] = scores[:, :N_COMPONENTS]
        payload[f"{name}_evr"] = evr[:N_COMPONENTS]
        payload[f"{name}_evr_full"] = evr
        print(f"  kernel PCA {name}: FVE[0:4]={evr[:4].round(3)}", flush=True)

    np.savez(RESULTS / f"slice_{index}.npz", **payload)
    meta = {
        "slice": index,
        "threshold_cells_per_mm2": threshold,
        "n_cells_kept": int(kept_per_slide.sum()),
        "kernel": "gudhi PersistenceScaleSpaceKernel through random Fourier features (RBFSampler)",
        "n_features": N_FEATURES,
        "seed": SEED,
        "bandwidth_px2": {"h0": float(bandwidths[0]), "h1": float(bandwidths[1])},
        "n_components": N_COMPONENTS,
        "topocyte": str(tc.__version__),
        "runtime_seconds": round(time.time() - started, 1),
        **timings,
    }
    (RESULTS / f"slice_{index}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/slice_{index}.npz in {time.time() - started:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("slice")
    one.add_argument("index", type=int, choices=range(10))
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cmd_slice(args.index)


if __name__ == "__main__":
    main()
