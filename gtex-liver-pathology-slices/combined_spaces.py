#!/usr/bin/env python
"""Combine the per-slice fPCA scores into one score space by multiple factor analysis.

Every slice's four models give a block of nine scores per slide: 40 blocks.
The blocks' scales differ by up to 13 orders of magnitude (raw H0 scores reach
10^6, cosine scores stay below 1), so each block is divided by the square
root of the variance of its first score (multiple factor analysis): its
leading component then has variance 1 and its internal proportions stay.
PCA (SVD) of the centred concatenation of the weighted blocks is the combined
space.  It equals kernel PCA of the sum of the blocks' centred Gram matrices,
each divided by its first eigenvalue and cut to its nine components.

Spaces: ``all`` (the 40 blocks, 360 columns) and one per model family
(``h0_none``, ``h1_none``, ``h0_cosine``, ``h1_cosine``: that model's ten
slices, 90 columns).  For each, saves the blocks' weights, the fraction of
variance of every combined component, the scores of the first 20 (each
component signed so that the largest-magnitude entry of its left singular
vector is positive), each block's share of every saved component's squared
loading, and a UMAP of the weighted concatenation (``umap_scores.embed``).
Also saves, for the report, the two weightings MFA replaces on the ``all``
space: plain concatenation and every column z-scored.

Reads ``results/slice_<k>.npz``; writes ``results/combined.npz`` and
``results/combined.json``.  Needs no topocyte import.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from umap_scores import PARAMS, embed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
N_SLICES = 10
MODELS = ("h0_none", "h1_none", "h0_cosine", "h1_cosine")
SPACES = ("all",) + MODELS
N_KEEP = 20


def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, fractions of variance and components of the centred matrix, signs fixed."""
    centred = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    # Sign: the largest-magnitude entry of each left singular vector is positive.
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    signs[signs == 0] = 1.0
    u, vt = u * signs, vt * signs[:, None]
    variance = s**2
    return u * s, variance / variance.sum(), vt


def main() -> None:
    import umap

    started = time.time()
    blocks: dict[tuple[int, str], np.ndarray] = {}
    slide_ids = None
    for k in range(N_SLICES):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
            if slide_ids is None:
                slide_ids = data["slide_ids"]
            elif not np.array_equal(data["slide_ids"], slide_ids):
                raise RuntimeError(f"slice {k}: slides differ from slice 0")
            for name in MODELS:
                if not np.array_equal(data[f"{name}_group_ids"], slide_ids):
                    raise RuntimeError(f"slice {k} {name}: score rows are not in the slides' order")
                blocks[(k, name)] = np.asarray(data[f"{name}_scores"], dtype=np.float64)

    payload: dict[str, np.ndarray] = {"slide_ids": slide_ids, "spaces": np.array(SPACES)}
    summary = {}
    for space in SPACES:
        keys = [(k, m) for k in range(N_SLICES) for m in MODELS if space in ("all", m)]
        weights = np.array([1.0 / np.sqrt(blocks[key][:, 0].var()) for key in keys])
        weighted = np.hstack([blocks[key] * w for key, w in zip(keys, weights)])
        scores, evr, vt = pca(weighted)
        width = blocks[keys[0]].shape[1]
        share = (vt[:N_KEEP] ** 2).reshape(N_KEEP, len(keys), width).sum(axis=2).T  # blocks x components
        payload[f"{space}_blocks"] = np.array([f"{k}:{m}" for k, m in keys])
        payload[f"{space}_weights"] = weights
        payload[f"{space}_evr"] = evr
        payload[f"{space}_scores"] = scores[:, :N_KEEP]
        payload[f"{space}_block_share"] = share
        payload[f"{space}_umap"] = embed(weighted)
        n90 = int(np.searchsorted(np.cumsum(evr), 0.9)) + 1
        summary[space] = {"blocks": len(keys), "columns": weighted.shape[1], "pc1": round(float(evr[0]), 4), "components_for_90": n90}
        print(f"{space}: {len(keys)} blocks, {weighted.shape[1]} columns; FVE[0:5]={np.round(evr[:5], 3)}; {n90} for 90%", flush=True)

    # The two weightings MFA replaces, on the all space, for the report.
    plain = np.hstack([blocks[(k, m)] for k in range(N_SLICES) for m in MODELS])
    _, evr_plain, vt_plain = pca(plain)
    loading = (vt_plain[0] ** 2).reshape(N_SLICES * len(MODELS), -1).sum(axis=1)
    h0_raw = [i for i, (k, m) in enumerate((k, m) for k in range(N_SLICES) for m in MODELS) if m == "h0_none"]
    payload["plain_evr"] = evr_plain
    payload["plain_pc1_h0_none_share"] = np.float64(loading[h0_raw].sum())
    zscored = (plain - plain.mean(axis=0)) / plain.std(axis=0)
    _, evr_z, _ = pca(zscored)
    payload["zscore_evr"] = evr_z

    np.savez(RESULTS / "combined.npz", **payload)
    meta = {
        "weighting": "multiple factor analysis: each block divided by the square root of its first score's variance",
        "spaces": summary,
        "n_keep": N_KEEP,
        "umap": {"params": PARAMS, "umap_learn": str(umap.__version__)},
        "runtime_seconds": round(time.time() - started, 1),
    }
    (RESULTS / "combined.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/combined.npz in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
