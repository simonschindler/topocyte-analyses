#!/usr/bin/env python
"""Check the saved results of run_pathology_slices.py.

* ``pathology.csv``: one row per slide with cells, the groups recomputed from
  the cohort CSV by a separate implementation of the severity rule, and the
  group sizes the ledger records;
* ``density.npz``: the cell count, rho finite and positive, the deciles and
  the quantile curve equal to numpy's quantiles of rho, the kept counts, and
  one slide's rho recomputed with scikit-learn's nearest neighbours;
* ``slice_<k>.npz``: every slide in every model, kept counts equal to the
  density file's, one H0 bar per kept cell, scores centred and uncorrelated
  with variances in the ratio of the fractions of variance, the fractions
  positive, ordered and below one in sum; and slice 9 refitted with the dense
  solver from recomputed diagrams, which must give the matrix-free fit's
  fractions and scores back;
* ``umap.npz``: every slice's and model's embedding, in the slides' order,
  with the ledger's parameters, and one of them rerun bit for bit;
* ``combined.npz``: the five MFA spaces - block weights, a spectrum summing
  to one, scores and fractions equal to scikit-learn's PCA of the weighted
  blocks, block shares, and one UMAP rerun bit for bit;
* ``pc2_check.npz``: PC2's orientation, the residuals against scikit-learn's
  regression, every AUC against ``roc_auc_score`` and one bootstrap interval
  recomputed.

Run with ``PYTHONPATH=<topocyte checkout>``.  Exit status 0 iff every check
passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COHORT = Path("/Users/simon/data/gtex_liver_cells/gtex_liver_cells")
N_CELLS = 126_219_124
N_SLIDES = 594
N_COMPONENTS = 9
EXPECTED_GROUPS = {
    "steatosis": 202,
    "congestion": 155,
    "healthy": 66,
    "cirrhosis": 57,
    "fibrosis": 50,
    "unclassified": 34,
    "necrosis": 18,
    "inflammation/hepatitis": 12,
}
MODELS = (("h0_none", 0, "none"), ("h1_none", 1, "none"), ("h0_cosine", 0, "cosine"), ("h1_cosine", 1, "cosine"))


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def _group(categories: str, label: str) -> str:
    """The severity rule, written again from the ledger."""
    if label == "healthy":
        return "healthy"
    found = {c.strip() for c in categories.split(",")} - {""}
    order = [
        ("cirrhosis", {"cirrhosis"}),
        ("fibrosis", {"fibrosis", "scarring", "sclerotic"}),
        ("inflammation/hepatitis", {"inflammation", "hepatitis"}),
        ("necrosis", {"necrosis", "infarction", "ischemic_changes"}),
        ("steatosis", {"steatosis"}),
        ("congestion", {"congestion", "hemorrhage"}),
    ]
    for name, members in order:
        if found & members:
            return name
    return "unclassified" if not found else "?"


def check_groups(slides: list[str], counts: np.ndarray) -> dict[str, str]:
    table = pd.read_csv(RESULTS / "pathology.csv", keep_default_na=False)
    if list(table["slide_id"]) != slides:
        _fail("pathology.csv: slides differ from the cohort's, or are in another order")
    cohort = pd.read_csv(COHORT / "liver_cohort.csv", keep_default_na=False).set_index("Tissue Sample ID")
    want = [_group(cohort.at[s, "Pathology Categories"], cohort.at[s, "label"]) for s in slides]
    if want != list(table["group"]):
        _fail("pathology.csv: a group differs from the severity rule")
    sizes = table["group"].value_counts().to_dict()
    if sizes != EXPECTED_GROUPS:
        _fail(f"pathology.csv: group sizes {sizes} are not the ledger's {EXPECTED_GROUPS}")
    if not np.array_equal(table["n_cells"].to_numpy(), counts):
        _fail("pathology.csv: cells per slide differ from the cohort's")
    print(f"PASS groups: {len(slides)} slides, severity rule reproduced, sizes as in the ledger")
    return dict(zip(table["slide_id"], table["group"]))


def check_density(codes: np.ndarray, slides: list[str]) -> dict[str, np.ndarray]:
    import h5py
    from sklearn.neighbors import NearestNeighbors

    with np.load(RESULTS / "density.npz", allow_pickle=False) as data:
        density = {key: data[key] for key in data.files}
    rho = density["rho"]
    if rho.shape != (N_CELLS,) or not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        _fail("density.npz: rho is not finite and positive for every cell")
    if not np.array_equal(density["slide_codes"], codes):
        _fail("density.npz: slide codes differ from the cohort's")
    bounds = density["bounds"]
    if not np.array_equal(bounds, np.searchsorted(codes, np.arange(N_SLIDES + 1))):
        _fail("density.npz: slide bounds do not follow the slide codes")
    if not np.array_equal(density["deciles"], np.quantile(rho, density["decile_levels"])):
        _fail("density.npz: deciles are not numpy's quantiles of rho")
    if np.any(np.diff(density["deciles"]) <= 0) or not np.allclose(density["decile_levels"], np.arange(10) / 10):
        _fail("density.npz: decile levels or thresholds out of order")
    if not np.array_equal(density["curve"], np.quantile(rho, density["curve_levels"])):
        _fail("density.npz: quantile curve is not numpy's quantiles of rho")
    kept = np.array(
        [[int(np.count_nonzero(rho[bounds[c]:bounds[c + 1]] >= d)) for c in range(N_SLIDES)] for d in density["deciles"]]
    )
    if not np.array_equal(kept, density["kept_per_slide"]):
        _fail("density.npz: kept counts differ from rho")
    # One slide's density through scikit-learn instead of scipy's cKDTree.
    code = int(np.argmin(np.diff(bounds)))
    start, end = bounds[code], bounds[code + 1]
    with h5py.File(COHORT / "gtex_liver_cells.h5ad", "r") as f:
        points = np.asarray(f["obsm/spatial"][start:end], dtype=np.float64)
    distances, _ = NearestNeighbors(n_neighbors=11).fit(points).kneighbors(points)
    r10_um = distances[:, 10] * float(density["pixel_um"])
    want = 10 * 1e6 / (np.pi * r10_um**2)
    if not np.allclose(rho[start:end], want, rtol=1e-12, atol=0.0):
        _fail(f"density.npz: rho of slide {slides[code]} differs from scikit-learn's 10-NN distances")
    print(
        f"PASS density: {N_CELLS:,} cells, deciles and curve are rho's quantiles, kept counts exact, "
        f"slide {slides[code]} recomputed with scikit-learn"
    )
    return density


def check_slices(density: dict[str, np.ndarray], slides: list[str]) -> None:
    for k in range(10):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        tag = f"slice_{k}"
        if [str(s) for s in arrays["slide_ids"]] != slides or arrays["skipped"].size:
            _fail(f"{tag}: slides differ or some were skipped")
        if not np.array_equal(arrays["kept_per_slide"], density["kept_per_slide"][k]):
            _fail(f"{tag}: kept counts differ from density.npz")
        if float(arrays["threshold"]) != float(density["deciles"][k]):
            _fail(f"{tag}: threshold is not decile {k}")
        if not np.array_equal(arrays["n_bars"][:, 0], arrays["kept_per_slide"]):
            _fail(f"{tag}: H0 bars differ from the kept cells")
        for name, _, _ in MODELS:
            scores, evr = arrays[f"{name}_scores"], arrays[f"{name}_evr"]
            if [str(s) for s in arrays[f"{name}_group_ids"]] != slides:
                _fail(f"{tag} {name}: group ids are not the slides")
            if scores.shape != (N_SLIDES, N_COMPONENTS) or not np.all(np.isfinite(scores)):
                _fail(f"{tag} {name}: scores shape {scores.shape} or non-finite")
            if evr.shape != (N_COMPONENTS,) or np.any(evr <= 0) or np.any(np.diff(evr) > 1e-15) or evr.sum() > 1 + 1e-12:
                _fail(f"{tag} {name}: fractions of variance not positive, ordered and below one")
            scale = float(np.abs(scores).max())
            if np.abs(scores.mean(axis=0)).max() > 1e-9 * scale:
                _fail(f"{tag} {name}: scores are not centred")
            cov = scores.T @ scores
            off = cov - np.diag(np.diag(cov))
            if np.abs(off).max() > 1e-8 * np.diag(cov).max():
                _fail(f"{tag} {name}: scores are correlated")
            if not np.allclose(np.diag(cov) / cov[0, 0], evr / evr[0], rtol=1e-8, atol=1e-12):
                _fail(f"{tag} {name}: score variances are not in the ratio of the fractions of variance")
    print("PASS slices: every slide in every model, kept counts and H0 bars exact, scores centred and uncorrelated, FVE consistent")


def check_dense_refit(density: dict[str, np.ndarray]) -> None:
    """Slice 9 again: recomputed diagrams, the dense solver, the same answers."""
    import h5py
    from concurrent.futures import ThreadPoolExecutor
    import os
    import topocyte as tc

    k = 9
    bounds = density["bounds"]
    keep = density["rho"] >= density["deciles"][k]
    with h5py.File(COHORT / "gtex_liver_cells.h5ad", "r") as f:
        coords = np.asarray(f["obsm/spatial"][:], dtype=np.float32)

    def compute(code: int):
        start, end = bounds[code], bounds[code + 1]
        cloud = coords[start:end][keep[start:end]].astype(np.float64)
        return tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(compute, range(N_SLIDES)))
    with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    for name, dim, normalization in MODELS:
        dense = tc.ph.FPCA(
            dim=dim, n_components=N_COMPONENTS, normalization=normalization, n_jobs=-1, eigen_solver="dense"
        ).fit(results).result_
        if not np.allclose(dense.explained_variance_ratio, saved[f"{name}_evr"], rtol=1e-6, atol=0.0):
            _fail(f"slice {k} {name}: the dense fit's fractions of variance differ from the saved ones")
        got = np.asarray(dense.scores)
        want = saved[f"{name}_scores"]
        signs = np.sign(np.sum(got * want, axis=0))
        if not np.allclose(got[:, :4] * signs[:4], want[:, :4], rtol=0.0, atol=1e-6 * np.abs(want).max()):
            _fail(f"slice {k} {name}: the dense fit's first four scores differ from the saved ones")
    print("PASS dense refit: slice 9 from recomputed diagrams with the dense solver gives the saved fractions and scores")


def check_umap(slides: list[str]) -> None:
    """Every embedding present, in the slides' order, and one rerun identical."""
    import json

    from umap_scores import embed

    with np.load(RESULTS / "umap.npz", allow_pickle=False) as data:
        embeddings = {key: data[key] for key in data.files}
    meta = json.loads((RESULTS / "umap.json").read_text())
    if [str(s) for s in embeddings["slide_ids"]] != slides:
        _fail("umap.npz: slides differ from the cohort's")
    if meta["params"] != {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "n_components": 2, "random_state": 0}:
        _fail(f"umap.json: parameters {meta['params']} are not the ledger's")
    for k in range(10):
        for name, _, _ in MODELS:
            emb = embeddings.get(f"{name}_{k}")
            if emb is None or emb.shape != (N_SLIDES, 2) or not np.all(np.isfinite(emb)):
                _fail(f"umap.npz: {name}_{k} missing, misshapen or non-finite")
    with np.load(RESULTS / "slice_9.npz", allow_pickle=False) as data:
        scores = np.asarray(data["h1_cosine_scores"], dtype=np.float64)
    if not np.array_equal(embed(scores), embeddings["h1_cosine_9"]):
        _fail("umap.npz: rerunning slice 9's H1 cosine embedding does not give the saved one")
    print("PASS umap: 40 embeddings of 594 slides in the slides' order, one rerun identical")


def check_combined(slides: list[str]) -> None:
    """The MFA spaces: weights, spectrum, scores against scikit-learn's PCA, shares, one UMAP rerun."""
    from sklearn.decomposition import PCA

    from umap_scores import embed

    with np.load(RESULTS / "combined.npz", allow_pickle=False) as data:
        combined = {key: data[key] for key in data.files}
    if [str(s) for s in combined["slide_ids"]] != slides:
        _fail("combined.npz: slides differ from the cohort's")
    blocks = {}
    for k in range(10):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
            for name, _, _ in MODELS:
                blocks[(k, name)] = np.asarray(data[f"{name}_scores"], dtype=np.float64)
    models = [name for name, _, _ in MODELS]
    if [str(s) for s in combined["spaces"]] != ["all"] + models:
        _fail(f"combined.npz: spaces {combined['spaces']}")
    for space in ["all"] + models:
        keys = [(k, m) for k in range(10) for m in models if space in ("all", m)]
        if [str(b) for b in combined[f"{space}_blocks"]] != [f"{k}:{m}" for k, m in keys]:
            _fail(f"combined.npz: {space} blocks are not the expected ones in order")
        want_weights = np.array([1 / np.sqrt(blocks[key][:, 0].var()) for key in keys])
        if not np.allclose(combined[f"{space}_weights"], want_weights, rtol=1e-12, atol=0.0):
            _fail(f"combined.npz: {space} weights are not 1 / sqrt(variance of each block's first score)")
        weighted = np.hstack([blocks[key] * w for key, w in zip(keys, want_weights)])
        evr = combined[f"{space}_evr"]
        if abs(evr.sum() - 1) > 1e-12 or np.any(np.diff(evr) > 1e-15):
            _fail(f"combined.npz: {space} fractions of variance do not sum to one in order")
        reference = PCA(n_components=20, svd_solver="full").fit(weighted)
        if not np.allclose(evr[:20], reference.explained_variance_ratio_, rtol=1e-9, atol=1e-14):
            _fail(f"combined.npz: {space} fractions of variance differ from scikit-learn's PCA")
        got = combined[f"{space}_scores"]
        want = reference.transform(weighted)
        signs = np.sign(np.sum(got * want, axis=0))
        if not np.allclose(got * signs, want, rtol=0.0, atol=1e-9 * np.abs(want).max()):
            _fail(f"combined.npz: {space} scores differ from scikit-learn's PCA beyond sign")
        share = combined[f"{space}_block_share"]
        if share.shape != (len(keys), 20) or not np.allclose(share.sum(axis=0), 1.0, atol=1e-12):
            _fail(f"combined.npz: {space} block shares do not sum to one per component")
        if combined[f"{space}_umap"].shape != (N_SLIDES, 2):
            _fail(f"combined.npz: {space} UMAP shape")
    keys = [(k, "h1_cosine") for k in range(10)]
    weighted = np.hstack([blocks[key] / np.sqrt(blocks[key][:, 0].var()) for key in keys])
    if not np.array_equal(embed(weighted), combined["h1_cosine_umap"]):
        _fail("combined.npz: rerunning the H1 cosine space's UMAP does not give the saved one")
    print("PASS combined: 5 MFA spaces, weights, spectra and scores match scikit-learn's PCA, shares, one UMAP rerun identical")


def check_pc2(slides: list[str]) -> None:
    """Task 4: residuals through scikit-learn, AUCs through roc_auc_score, one interval recomputed."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    with np.load(RESULTS / "pc2_check.npz", allow_pickle=False) as data:
        check = {key: data[key] for key in data.files}
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as data:
        combined = {key: data[key] for key in data.files}
    with np.load(RESULTS / "density.npz", allow_pickle=False) as data:
        kept = data["kept_per_slide"]
        log_median = np.log10(data["slide_medians"])
        log_cells = np.log10(np.diff(data["bounds"]))
    if [str(s) for s in check["slide_ids"]] != slides:
        _fail("pc2_check.npz: slides differ from the cohort's")
    group = pd.read_csv(RESULTS / "pathology.csv", keep_default_na=False)["group"].to_numpy()
    pair = np.isin(group, ["cirrhosis", "healthy"])
    y = (group[pair] == "cirrhosis").astype(int)
    share = kept[9] / kept[0]
    if not np.allclose(check["share"], share, rtol=0, atol=0):
        _fail("pc2_check.npz: the densest-decile share is not kept[9] / kept[0]")

    def roc(values: np.ndarray) -> float:
        return float(roc_auc_score(y, values[pair]))

    if not np.isclose(check["share_auc"][0], roc(share), rtol=0, atol=1e-12):
        _fail("pc2_check.npz: the share's AUC differs from roc_auc_score")
    for space in [str(s) for s in check["spaces"]]:
        scores = combined[f"{space}_scores"]
        pc1, pc2 = scores[:, 0], float(check[f"{space}_sign"]) * scores[:, 1]
        if np.median(pc2[group == "cirrhosis"]) < np.median(pc2[group == "healthy"]):
            _fail(f"pc2_check.npz: {space} PC2 is not oriented with cirrhosis above healthy")
        designs = {
            "raw": None,
            "share": np.column_stack([share, share**2]),
            "strict": np.column_stack([share, share**2, pc1, pc1**2, log_median, log_cells]),
        }
        for variant, design in designs.items():
            want = pc2 if design is None else pc2 - LinearRegression().fit(design, pc2).predict(design)
            got = check[f"{space}_{variant}_values"]
            if not np.allclose(got, want, rtol=0, atol=1e-9 * np.abs(pc2).max()):
                _fail(f"pc2_check.npz: {space} {variant} residual differs from scikit-learn's regression")
            readout = check[f"{space}_{variant}"]
            if not np.isclose(readout[0], roc(got), rtol=0, atol=1e-12) or not readout[1] <= readout[0] <= readout[2]:
                _fail(f"pc2_check.npz: {space} {variant} AUC or its interval is inconsistent")
    # One interval recomputed with the same draws.
    rng = np.random.default_rng(12345)
    c, h = check["all_share_values"][group == "cirrhosis"], check["all_share_values"][group == "healthy"]
    draws_c = rng.integers(0, c.size, (4000, c.size))
    draws_h = rng.integers(0, h.size, (4000, h.size))
    boot = [float(roc_auc_score(np.r_[np.ones(c.size), np.zeros(h.size)], np.r_[c[i], h[j]])) for i, j in zip(draws_c, draws_h)]
    if not np.allclose(check["all_share"][1:3], np.quantile(boot, [0.025, 0.975]), rtol=0, atol=1e-12):
        _fail("pc2_check.npz: the all-space share interval is not the stated bootstrap")
    print("PASS pc2 check: residuals match scikit-learn's regression, AUCs match roc_auc_score, one interval recomputed")


def main() -> None:
    import h5py

    with h5py.File(COHORT / "gtex_liver_cells.h5ad", "r") as f:
        codes = np.asarray(f["obs/slide_id/codes"][:], dtype=np.int16)
        slides = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/slide_id/categories"][:]]
    if codes.size != N_CELLS or len(slides) != N_SLIDES:
        _fail(f"cohort: {codes.size:,} cells on {len(slides)} slides")
    counts = np.bincount(codes, minlength=N_SLIDES)
    check_groups(slides, counts)
    density = check_density(codes, slides)
    check_slices(density, slides)
    check_umap(slides)
    check_combined(slides)
    check_pc2(slides)
    check_dense_refit(density)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
