#!/usr/bin/env python
"""Blocks, combined spaces and baselines of the cell-group study.

A block is one point set's nine fPCA scores for one model at one slice: the
all-cell blocks come from the sibling study's ``slice_<k>.npz``, the four
groups' from this study's. Spaces:

* ``block:<point set>:<model>:<slice>``: one block (200, less any dropped);
* ``mfa:<point sets>:<models>``: multiple factor analysis (``common.mfa``)
  of the blocks of a non-empty subset of the five point sets and a non-empty
  subset of the four models, over the ten slices (465 spaces, 20 scores
  kept);
* ``D``, ``C``, ``D+C``: PCA of the z-scored density summaries,
  composition, or both (``common.baseline_features``), every component kept.

Aliases: ``all-cell`` (the sibling's combined ``all`` space, which it must
reproduce), ``groups`` (the four groups, all models) and ``everything`` (all
five point sets, all models). UMAPs (the sibling's parameters, on the
weighted concatenation) of the key spaces: ``all-cell``, each group over all
models, ``everything``.

Reads ``results/slice_<k>.npz``, ``results/composition.npz`` and the
sibling's results; writes ``results/spaces.npz``, ``results/spaces.json``
and ``results/umap.npz``. Needs no topocyte import.
"""

from __future__ import annotations

import itertools
import json
import time
import warnings

import numpy as np
import pandas as pd

import common as c

ALL_MODELS = "+".join(c.MODEL_NAMES)
ALIASES = {
    "all-cell": f"mfa:all:{ALL_MODELS}",
    "groups": f"mfa:{'+'.join(c.CELL_GROUPS)}:{ALL_MODELS}",
    "everything": f"mfa:{'+'.join(c.POINT_SETS)}:{ALL_MODELS}",
}
KEY_SPACES = ("all-cell",) + tuple(f"mfa:{g}:{ALL_MODELS}" for g in c.CELL_GROUPS) + ("everything",)
N_KEEP = 20
UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "n_components": 2, "random_state": 0}
SIBLING_MIN_ABS_CORR = 0.999999


def labels_and_slides() -> tuple[np.ndarray, list[str]]:
    table = pd.read_csv(c.SOURCE / "pathology.csv")
    return np.array([c.GROUP_ORDER.index(g) for g in table["group"]]), [str(s) for s in table["slide_id"]]


def load_blocks(slides: list[str]) -> tuple[dict, dict, list[str]]:
    """Every available block and its fractions of variance, keyed (point set, model, slice), and the dropped ones."""
    blocks: dict[tuple[str, str, int], np.ndarray] = {}
    evrs: dict[tuple[str, str, int], np.ndarray] = {}
    dropped: list[str] = []
    for k in range(c.N_SLICES):
        with np.load(c.SOURCE / f"slice_{k}.npz", allow_pickle=False) as sibling, \
                np.load(c.RESULTS / f"slice_{k}.npz", allow_pickle=False) as own:
            for data in (sibling, own):
                if [str(s) for s in data["slide_ids"]] != slides:
                    raise RuntimeError(f"slice {k}: a results file lists the slides differently from pathology.csv")
            for m in c.MODEL_NAMES:
                blocks[("all", m, k)] = sibling[f"{m}_scores"]
                evrs[("all", m, k)] = sibling[f"{m}_evr"]
            dropped += [f"{d}:{k}" for d in own["dropped_blocks"]]
            for g in c.CELL_GROUPS:
                for m in c.MODEL_NAMES:
                    if f"{g}_{m}_scores" in own.files:
                        blocks[(g, m, k)] = own[f"{g}_{m}_scores"]
                        evrs[(g, m, k)] = own[f"{g}_{m}_evr"]
    return blocks, evrs, dropped


def subsets(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [combo for r in range(1, len(items) + 1) for combo in itertools.combinations(items, r)]


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def main() -> None:
    started = time.time()
    labels, slides = labels_and_slides()
    blocks, evrs, dropped = load_blocks(slides)
    names: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    def add(name: str, scores: np.ndarray, evr: np.ndarray, **extra: np.ndarray) -> None:
        i = len(names)
        names.append(name)
        arrays[f"s{i}__scores"] = scores
        arrays[f"s{i}__evr"] = evr
        for key, value in extra.items():
            arrays[f"s{i}__{key}"] = value

    for (p, m, k), block in blocks.items():
        add(f"block:{p}:{m}:{k}", block, evrs[(p, m, k)])
    umap_inputs: dict[str, np.ndarray] = {}
    key_names = {ALIASES.get(name, name) for name in KEY_SPACES}
    for point_sets in subsets(c.POINT_SETS):
        for models in subsets(c.MODEL_NAMES):
            keys = [(p, m, k) for p in point_sets for m in models for k in range(c.N_SLICES) if (p, m, k) in blocks]
            name = f"mfa:{'+'.join(point_sets)}:{'+'.join(models)}"
            scores, evr, weights = c.mfa([blocks[key] for key in keys], n_keep=N_KEEP)
            add(name, scores, evr, weights=weights, blocks=np.array([f"{p}:{m}:{k}" for p, m, k in keys]))
            if name in key_names:
                umap_inputs[name] = np.hstack([blocks[key] * w for key, w in zip(keys, weights)])

    with np.load(c.RESULTS / "composition.npz", allow_pickle=False) as composition, \
            np.load(c.SOURCE / "density.npz", allow_pickle=False) as density:
        if [str(s) for s in composition["slide_ids"]] != slides:
            raise RuntimeError("composition.npz lists the slides differently from pathology.csv")
        D, C, d_names, c_names = c.baseline_features(density["slide_medians"], composition["kept_by_group"])
    for name, features in (("D", D), ("C", C), ("D+C", np.hstack([D, C]))):
        scores, evr, _ = c.pca(zscore(features))
        add(name, scores, evr)

    # The all-cell space must be the sibling's combined "all" space.
    ours = arrays[f"s{names.index(ALIASES['all-cell'])}__scores"]
    with np.load(c.SOURCE / "combined.npz", allow_pickle=False) as combined:
        theirs = combined["all_scores"]
    sibling_corr = [abs(float(np.corrcoef(ours[:, j], theirs[:, j])[0, 1])) for j in range(N_KEEP)]
    if min(sibling_corr) < SIBLING_MIN_ABS_CORR:
        raise RuntimeError(f"all-cell space differs from the sibling's combined space: min |r| {min(sibling_corr)}")

    import umap

    embeddings = {}
    for name, X in umap_inputs.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
            embeddings[name] = np.asarray(umap.UMAP(**UMAP_PARAMS).fit_transform(X), dtype=np.float64)

    np.savez(
        c.RESULTS / "spaces.npz",
        names=np.array(names),
        labels=labels,
        group_order=np.array(c.GROUP_ORDER),
        slide_ids=np.array(slides),
        D_features=D,
        C_features=C,
        D_names=np.array(d_names),
        C_names=np.array(c_names),
        dropped_blocks=np.array(dropped, dtype=str),
        **arrays,
    )
    np.savez(c.RESULTS / "umap.npz", names=np.array(list(embeddings)), **{f"u{i}": e for i, e in enumerate(embeddings.values())})
    meta = {
        "n_spaces": len(names),
        "n_blocks": len(blocks),
        "n_mfa": sum(n.startswith("mfa:") for n in names),
        "aliases": ALIASES,
        "key_spaces": [ALIASES.get(n, n) for n in KEY_SPACES],
        "dropped_blocks": dropped,
        "all_cell_vs_sibling_min_abs_corr": min(sibling_corr),
        "umap_params": UMAP_PARAMS,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (c.RESULTS / "spaces.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(names)} spaces ({meta['n_blocks']} blocks, {meta['n_mfa']} MFA, 3 baselines); dropped {dropped}; "
          f"all-cell vs sibling min |r| {min(sibling_corr):.9f}; {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
