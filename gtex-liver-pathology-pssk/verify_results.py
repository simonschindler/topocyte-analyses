#!/usr/bin/env python
"""Check the saved results of the PSSK study.

* inputs: the slides of the cohort, of the sibling study's
  ``pathology.csv`` and ``density.npz``, in the same order, and the group
  sizes its ledger records;
* ``slice_<k>.npz``: every slide, kept counts and thresholds equal to the
  density file's, bar counts equal to the Betti study's (the gate), one H0 bar
  per kept cell and one of them essential, every H1 bar finite; the random
  features the stated sampler draws; kernel PCA recomputed from the saved
  features by an eigendecomposition of the centred Gram matrix, which must
  give the saved fractions and scores back; scores centred and uncorrelated;
* slice 9 recomputed: diagrams from the cohort, the bandwidths (median
  persistence), and every slide's features through a separate implementation
  of the feature map; and gudhi's own ``kernel_approx`` route on two small
  diagrams;
* ``validation.json``: present and consistent;
* ``umap.npz``, ``combined.npz`` and ``pc2_check.npz``: as in the Betti
  study's verifier - one UMAP rerun bit for bit, the MFA spaces against
  scikit-learn's PCA, the residuals against scikit-learn's regression and the
  AUCs against ``roc_auc_score``.

Run with ``PYTHONPATH=<topocyte checkout>``.  Exit status 0 iff every check
passes.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = HERE.parent / "gtex-liver-pathology-slices" / "results"
COHORT = Path("/Users/simon/data/gtex_liver_cells/gtex_liver_cells")
N_CELLS = 126_219_124
N_SLIDES = 594
N_COMPONENTS = 9
N_FEATURES = 1000
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


def check_inputs(slides: list[str]) -> dict[str, np.ndarray]:
    table = pd.read_csv(SOURCE / "pathology.csv", keep_default_na=False)
    if list(table["slide_id"]) != slides:
        _fail("pathology.csv: slides differ from the cohort's, or are in another order")
    sizes = table["group"].value_counts().to_dict()
    if sizes != EXPECTED_GROUPS:
        _fail(f"pathology.csv: group sizes {sizes} are not the ledger's {EXPECTED_GROUPS}")
    with np.load(SOURCE / "density.npz", allow_pickle=False) as data:
        density = {key: data[key] for key in data.files if key not in ("rho", "slide_codes")}
    if [str(s) for s in density["slide_ids"]] != slides:
        _fail("density.npz: slides differ from the cohort's")
    print(f"PASS inputs: the sibling study's groups and densities list the cohort's {len(slides)} slides in order")
    return density


def kernel_pca_by_eigh(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kernel PCA through the centred Gram matrix's eigendecomposition: scores and fractions."""
    centred = rows - rows.mean(axis=0)
    gram = centred @ centred.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    top = np.clip(values[:N_COMPONENTS], 0.0, None)
    return vectors[:, :N_COMPONENTS] * np.sqrt(top), values[:N_COMPONENTS] / np.trace(gram)


def check_slices(density: dict[str, np.ndarray], slides: list[str]) -> None:
    from sklearn.kernel_approximation import RBFSampler

    for k in range(10):
        with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        with np.load(SOURCE / f"slice_{k}.npz", allow_pickle=False) as data:
            betti_bars = data["n_bars"]
        tag = f"slice_{k}"
        if [str(s) for s in arrays["slide_ids"]] != slides or arrays["skipped"].size:
            _fail(f"{tag}: slides differ or some were skipped")
        if not np.array_equal(arrays["kept_per_slide"], density["kept_per_slide"][k]):
            _fail(f"{tag}: kept counts differ from density.npz")
        if float(arrays["threshold"]) != float(density["deciles"][k]):
            _fail(f"{tag}: threshold is not decile {k}")
        if not np.array_equal(arrays["n_bars"], betti_bars):
            _fail(f"{tag}: bar counts differ from the Betti study's")
        if not np.array_equal(arrays["n_bars"][:, 0], arrays["kept_per_slide"]):
            _fail(f"{tag}: H0 bars differ from the kept cells")
        if not (np.array_equal(arrays["n_finite"][:, 0], arrays["kept_per_slide"] - 1)
                and np.array_equal(arrays["n_finite"][:, 1], arrays["n_bars"][:, 1])):
            _fail(f"{tag}: finite bars are not every bar but one essential H0 class")
        bandwidth = arrays["bandwidth"]
        if bandwidth.shape != (2,) or not np.all(np.isfinite(bandwidth)) or np.any(bandwidth <= 0):
            _fail(f"{tag}: bandwidths {bandwidth}")
        psi = {}
        for dim in (0, 1):
            sampler = RBFSampler(gamma=1 / (2 * bandwidth[dim] ** 2), n_components=N_FEATURES, random_state=0)
            sampler.fit(np.zeros((1, 2)))
            if not (np.array_equal(arrays[f"h{dim}_random_weights"], sampler.random_weights_)
                    and np.array_equal(arrays[f"h{dim}_random_offset"], sampler.random_offset_)):
                _fail(f"{tag} H{dim}: saved random features are not the stated sampler's")
            psi[dim] = arrays[f"h{dim}_features"]
            if psi[dim].shape != (N_SLIDES, N_FEATURES) or not np.all(np.isfinite(psi[dim])):
                _fail(f"{tag} H{dim}: features shape {psi[dim].shape} or non-finite")
        for name, dim, normalization in MODELS:
            scores, evr = arrays[f"{name}_scores"], arrays[f"{name}_evr"]
            if [str(s) for s in arrays[f"{name}_group_ids"]] != slides:
                _fail(f"{tag} {name}: group ids are not the slides")
            if scores.shape != (N_SLIDES, N_COMPONENTS) or not np.all(np.isfinite(scores)):
                _fail(f"{tag} {name}: scores shape {scores.shape} or non-finite")
            if evr.shape != (N_COMPONENTS,) or np.any(evr <= 0) or np.any(np.diff(evr) > 1e-15) or evr.sum() > 1 + 1e-12:
                _fail(f"{tag} {name}: fractions of variance not positive, ordered and below one")
            full = arrays[f"{name}_evr_full"]
            if abs(full.sum() - 1) > 1e-12 or not np.array_equal(full[:N_COMPONENTS], evr):
                _fail(f"{tag} {name}: the full spectrum does not sum to one or does not start with the saved fractions")
            scale = float(np.abs(scores).max())
            if np.abs(scores.mean(axis=0)).max() > 1e-9 * scale:
                _fail(f"{tag} {name}: scores are not centred")
            cov = scores.T @ scores
            off = cov - np.diag(np.diag(cov))
            if np.abs(off).max() > 1e-8 * np.diag(cov).max():
                _fail(f"{tag} {name}: scores are correlated")
            rows = psi[dim] if normalization == "none" else psi[dim] / np.linalg.norm(psi[dim], axis=1, keepdims=True)
            want_scores, want_evr = kernel_pca_by_eigh(rows)
            if not np.allclose(evr, want_evr, rtol=1e-8, atol=1e-14):
                _fail(f"{tag} {name}: fractions of variance differ from the Gram matrix's eigenvalues")
            signs = np.sign(np.sum(scores * want_scores, axis=0))
            if not np.allclose(scores[:, :4] * signs[:4], want_scores[:, :4], rtol=0.0, atol=1e-7 * scale):
                _fail(f"{tag} {name}: the first four scores differ from the Gram matrix's eigenvectors")
    print("PASS slices: gate bar counts, finite bars, samplers, kernel PCA recomputed through the Gram matrix for every slice and model")


def features_direct(diagram: np.ndarray, weights: np.ndarray, offset: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """The feature map written out: sqrt(2/m) * sum over bars of cos at (b, d) minus cos at (d, b)."""
    total = np.zeros(weights.shape[1])
    for start in range(0, len(diagram), chunk):
        b, d = diagram[start:start + chunk, 0:1], diagram[start:start + chunk, 1:2]
        above = np.cos(b * weights[0] + d * weights[1] + offset)
        below = np.cos(d * weights[0] + b * weights[1] + offset)
        total += (above - below).sum(axis=0)
    return np.sqrt(2.0 / weights.shape[1]) * total


def check_slice9_recomputed(slides: list[str]) -> None:
    """Slice 9 from the cohort again: diagrams, bandwidths, features, gudhi's kernel_approx route."""
    import h5py
    import topocyte as tc
    from gudhi.representations.kernel_methods import PersistenceScaleSpaceKernel
    from sklearn.kernel_approximation import RBFSampler

    k = 9
    with np.load(SOURCE / "density.npz", allow_pickle=False) as data:
        bounds = data["bounds"]
        keep = data["rho"] >= data["deciles"][k]
    with h5py.File(COHORT / "gtex_liver_cells.h5ad", "r") as f:
        coords = np.asarray(f["obsm/spatial"][:], dtype=np.float32)
    with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}

    def compute(code: int) -> list[np.ndarray]:
        start, end = bounds[code], bounds[code + 1]
        cloud = coords[start:end][keep[start:end]].astype(np.float64)
        result = tc.ph.compute_persistent_homology(cloud, filtration="alpha", max_dim=1)
        return [(lambda x: x[np.isfinite(x[:, 1])])(np.asarray(result.diagrams[d], dtype=np.float64)) for d in (0, 1)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        diagrams = list(pool.map(compute, range(N_SLIDES)))
    coords = keep = None
    for dim in (0, 1):
        finite = [pair[dim] for pair in diagrams]
        bandwidth = float(np.median(np.concatenate([d[:, 1] - d[:, 0] for d in finite])))
        if bandwidth != float(saved["bandwidth"][dim]):
            _fail(f"slice 9 H{dim}: median persistence {bandwidth} is not the saved bandwidth")
        weights, offset = saved[f"h{dim}_random_weights"], saved[f"h{dim}_random_offset"]
        constant = 0.5 / (np.sqrt(2 * np.pi) * bandwidth)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            phi = np.array(list(pool.map(lambda d: features_direct(d, weights, offset), finite)))
        want = np.sqrt(constant) * phi
        if not np.allclose(saved[f"h{dim}_features"], want, rtol=0.0, atol=1e-9 * np.abs(want).max()):
            _fail(f"slice 9 H{dim}: saved features differ from the feature map written out")
        sampler = RBFSampler(gamma=1 / (2 * bandwidth**2), n_components=N_FEATURES, random_state=0).fit(np.zeros((1, 2)))
        small = np.argsort([len(d) for d in finite], kind="stable")[:2]
        first, second = finite[small[0]][:200], finite[small[1]][:200]
        through_gudhi = PersistenceScaleSpaceKernel(bandwidth=bandwidth, kernel_approx=sampler)(first, second)
        ours = constant * features_direct(first, weights, offset) @ features_direct(second, weights, offset)
        if abs(through_gudhi - ours) > 1e-9 * abs(through_gudhi):
            _fail(f"slice 9 H{dim}: gudhi's kernel_approx gives {through_gudhi}, the features {ours}")
    print("PASS slice 9 recomputed: diagrams, bandwidths, every slide's features, and gudhi's kernel_approx route on two diagrams")


def check_validation(density: dict[str, np.ndarray]) -> None:
    report = json.loads((RESULTS / "validation.json").read_text())
    kept = density["kept_per_slide"][9]
    small = np.argsort(kept, kind="stable")[:6]
    if report["small_slides"]["kept_cells"] != [int(kept[c]) for c in small]:
        _fail("validation.json: the six slides are not those with the fewest kept cells")
    for dim in (0, 1):
        part = report[f"h{dim}"]
        if part["equivalence"]["relative_difference"] > 1e-10:
            _fail(f"validation.json H{dim}: gudhi's kernel_approx and the features disagree")
        if part["saved_rows_relative_deviation"] > 1e-12:
            _fail(f"validation.json H{dim}: recomputed rows differ from the saved ones")
        exact = part["exact"]
        if not 0 <= exact["relative_error_median"] <= exact["relative_error_max"]:
            _fail(f"validation.json H{dim}: error summary out of order")
    if not report["h0"]["exact"]["relative_error_median"] < report["h1"]["exact"]["relative_error_median"]:
        _fail("validation.json: the H0 kernel is not the more accurate one")
    sweep = report["h1"]["feature_count"]
    means = [sweep[m]["mean"] for m in ("1000", "4000", "16000", "64000")]
    if not all(a > b for a, b in zip(means, means[1:])):
        _fail(f"validation.json: the H1 error does not fall with the number of features ({means})")
    for name, _, _ in MODELS:
        draws = report["stability"]["draws"][name]
        if len(draws) != 3 or not all(0 <= c <= 1 + 1e-12 for draw in draws for c in draw["abs_corr_pc1_4"]):
            _fail(f"validation.json: stability of {name}")
    print("PASS validation: equivalence, saved rows, error summaries, H1 error falling with the number of features, stability draws")


def check_umap(slides: list[str]) -> None:
    """Every embedding present, in the slides' order, and one rerun identical."""
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
    """Residuals through scikit-learn, AUCs through roc_auc_score, one interval recomputed."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score

    with np.load(RESULTS / "pc2_check.npz", allow_pickle=False) as data:
        check = {key: data[key] for key in data.files}
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as data:
        combined = {key: data[key] for key in data.files}
    with np.load(SOURCE / "density.npz", allow_pickle=False) as data:
        kept = data["kept_per_slide"]
        log_median = np.log10(data["slide_medians"])
        log_cells = np.log10(np.diff(data["bounds"]))
    if [str(s) for s in check["slide_ids"]] != slides:
        _fail("pc2_check.npz: slides differ from the cohort's")
    group = pd.read_csv(SOURCE / "pathology.csv", keep_default_na=False)["group"].to_numpy()
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
        slides = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs/slide_id/categories"][:]]
        n_cells = f["obs/slide_id/codes"].shape[0]
    if n_cells != N_CELLS or len(slides) != N_SLIDES:
        _fail(f"cohort: {n_cells:,} cells on {len(slides)} slides")
    density = check_inputs(slides)
    check_slices(density, slides)
    check_validation(density)
    check_umap(slides)
    check_combined(slides)
    check_pc2(slides)
    check_slice9_recomputed(slides)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
