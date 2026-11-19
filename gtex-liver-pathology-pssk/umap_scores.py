#!/usr/bin/env python
"""UMAP of every kernel PCA score space: one 2-D embedding per slice and model.

Reads ``results/slice_<k>.npz`` and embeds each model's nine-component score
matrix (one row per slide, in the slides' order) with umap-learn: Euclidean
distance on the scores as fitted, the library's defaults (``n_neighbors=15``,
``min_dist=0.1``) and ``random_state=0``, which makes the run deterministic.
Writes ``results/umap.npz`` (keys ``<model>_<slice>``, 594 x 2) and
``results/umap.json``.  Needs no topocyte import.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
N_SLICES = 10
MODELS = ("h0_none", "h1_none", "h0_cosine", "h1_cosine")
PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "n_components": 2, "random_state": 0}


def embed(scores: np.ndarray) -> np.ndarray:
    """The 2-D UMAP of one score matrix."""
    import umap

    with warnings.catch_warnings():
        # random_state forces one thread, and umap-learn says so each time.
        warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
        return np.asarray(umap.UMAP(**PARAMS).fit_transform(scores), dtype=np.float64)


def main() -> None:
    import umap

    started = time.time()
    payload: dict[str, np.ndarray] = {}
    slide_ids = None
    for k in range(N_SLICES):
        clock = time.time()
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
            ids = data["slide_ids"]
            if slide_ids is None:
                slide_ids = ids
            elif not np.array_equal(ids, slide_ids):
                raise RuntimeError(f"slice {k}: slides differ from slice 0")
            for name in MODELS:
                if not np.array_equal(data[f"{name}_group_ids"], slide_ids):
                    raise RuntimeError(f"slice {k} {name}: score rows are not in the slides' order")
                scores = np.asarray(data[f"{name}_scores"], dtype=np.float64)
                payload[f"{name}_{k}"] = embed(scores)
        print(f"slice {k}: four embeddings in {time.time() - clock:.1f} s", flush=True)
    payload["slide_ids"] = slide_ids
    np.savez(RESULTS / "umap.npz", **payload)
    meta = {
        "embedding": "UMAP of each model's nine-component kernel PCA score matrix, Euclidean, scores as fitted",
        "params": PARAMS,
        "umap_learn": str(umap.__version__),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (RESULTS / "umap.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/umap.npz in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
