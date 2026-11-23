#!/usr/bin/env python
"""e-means clustering of every cohort cell's 150 µm neighbourhood for k = 1..10, with gap statistics.

Feature sets, from ``results/counts.npy``:

- ``composition``: the ball's four class fractions;
- ``density``: the ball's fractions of cells kept at the pathology study's ten density thresholds (threshold 0
  keeps every cell, so its column is constant and left out);
- ``composition+density``: both blocks.

Every column is z-scored over all cells (ddof 0) and a constant column is left out; in the combined set each
block is then divided by √(its columns), so the two blocks carry equal total variance.

Clustering is e-means (Bessa et al. 2026, arXiv:2611.09501) with the authors' code,
``faiss.contrib.e_means.NumpyEmeans(d, k, nepochs=5, seed=…)`` from faiss-cpu 1.16.0, whose scaling laws set the
batch size and the step; every cell then gets its nearest final centroid (``assign``). k = 1 is the mean.

The gap statistic (Tibshirani, Walther and Hastie 2001) uses the chromatic pilot's reference, a Gaussian with
the features' mean and covariance. W_k is the mean squared distance to the nearest centroid, so the sample size
cancels: the data's W_k is over every cell, each of the B = 20 references has 20M points and is clustered as the
data. Gap(k) = mean_b log W*_k − log W_k and s_k = sd_b(log W*_k) √(1 + 1/B). Reported: Tibshirani's k̂ (the
smallest k with Gap(k) >= Gap(k+1) − s_{k+1}, k_max if none) and the pilot's rule (k̂, then 1 unless
Gap(k̂) − Gap(1) >= 0.03).

Writes, per set, ``results/labels_<set>.npz`` (every cell's label for k = 2..10, two labels per byte,
compressed) and ``results/gap_<set>.json``; then ``results/gap.json`` and ``results/clusters.json``. A set whose
two files exist is not rerun.
"""

from __future__ import annotations

import json
import platform
import sys
import time

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

import common as c

CHUNK = 4_000_000  # rows per chunk when reading counts or drawing references
CHECK_ROWS, CHECK_KS = 500_000, (4, 10)  # e-means against scikit-learn's k-means on a random subsample


def fractions(counts: np.ndarray, name: str) -> tuple[np.ndarray, list[str]]:
    """(rows, columns) float64 fractions of a chunk of counts for one feature set, with the column names."""
    n_class = len(c.CLASSES)
    n_ball = counts[:, :n_class].sum(axis=1, dtype=np.float64)[:, None]
    comp = counts[:, :n_class] / n_ball
    dens = np.column_stack([np.ones(len(counts)), counts[:, n_class:] / n_ball])
    comp_names = [f"{k} fraction" for k in c.CLASSES]
    dens_names = [f"kept fraction, threshold {j}" for j in range(c.N_THRESHOLDS)]
    if name == "composition":
        return comp, comp_names
    if name == "density":
        return dens, dens_names
    return np.column_stack([comp, dens]), comp_names + dens_names


def blocks_of(name: str, columns: list[str]) -> list[str]:
    return ["composition" if "fraction" in col and "kept" not in col else "density" for col in columns]


def feature_matrix(counts: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    """The set's z-scored, block-weighted float32 matrix over every cell, and how it was built."""
    n = len(counts)
    total = total2 = None
    for start in range(0, n, CHUNK):
        x, columns = fractions(np.asarray(counts[start : start + CHUNK]), name)
        total = x.sum(axis=0) if total is None else total + x.sum(axis=0)
        total2 = (x * x).sum(axis=0) if total2 is None else total2 + (x * x).sum(axis=0)
    mean = total / n
    sd = np.sqrt(np.maximum(total2 / n - mean * mean, 0.0))
    used = [j for j in range(len(columns)) if sd[j] > 1e-12]
    blocks = blocks_of(name, columns)
    weight = np.ones(len(columns))
    if name == "composition+density":
        for block in ("composition", "density"):
            members = [j for j in used if blocks[j] == block]
            weight[members] = 1 / np.sqrt(len(members))
    X = np.empty((n, len(used)), np.float32)
    for start in range(0, n, CHUNK):
        x = fractions(np.asarray(counts[start : start + CHUNK]), name)[0]
        X[start : start + CHUNK] = ((x[:, used] - mean[used]) / sd[used] * weight[used]).astype(np.float32)
    built = {
        "columns": [columns[j] for j in used],
        "left_out_constant": [columns[j] for j in range(len(columns)) if j not in used],
        "column_means": [float(mean[j]) for j in used],
        "column_sds": [float(sd[j]) for j in used],
        "column_weights": [float(weight[j]) for j in used],
    }
    return X, built


def column_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance (ddof 0) of the rows of X, accumulated in float64 by chunks."""
    n, d = X.shape
    total, cross = np.zeros(d), np.zeros((d, d))
    for start in range(0, n, CHUNK):
        x = X[start : start + CHUNK].astype(np.float64)
        total += x.sum(axis=0)
        cross += x.T @ x
    mean = total / n
    return mean, cross / n - np.outer(mean, mean)


def mse_about_mean(X: np.ndarray) -> float:
    mean, cov = column_moments(X)
    return float(np.trace(cov))


def fit(X: np.ndarray, k: int, seed: int) -> dict:
    """e-means with k clusters, then every row's nearest final centroid: labels (uint8), W_k (mean squared
    distance), centroids and e-means' recipe. k = 1 is the mean."""
    if k == 1:
        return {"labels": None, "mse": mse_about_mean(X), "centroids": [column_moments(X)[0].tolist()]}
    em = NumpyEmeans(X.shape[1], k, nepochs=c.N_EPOCHS, seed=seed, capture_assignments=False)
    em.train(X)
    distances, labels = em.assign(X)
    recipe = {key: (value if isinstance(value, (int, float, str)) else str(value)) for key, value in em.recipe.items()}
    return {
        "labels": labels.astype(np.uint8),
        "mse": float(np.mean(distances, dtype=np.float64)),
        "centroids": np.asarray(em.centroids, dtype=np.float64).tolist(),
        "recipe": recipe,
        "revived": int(getattr(em, "n_splits", 0)),
    }


def seed_of(*key: int) -> int:
    return int(np.random.SeedSequence([c.SEED, *key]).generate_state(1)[0] % (2**31 - 1))


def reference(mean: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points of the Gaussian with this mean and covariance (float32), drawn by chunks."""
    out = np.empty((n, len(mean)), np.float32)
    for start in range(0, n, CHUNK):
        rows = min(CHUNK, n - start)
        out[start : start + rows] = rng.multivariate_normal(mean, cov, size=rows, method="eigh")
    return out


def tibshirani(gap: np.ndarray, s: np.ndarray) -> int:
    """The smallest k with Gap(k) >= Gap(k+1) - s_{k+1}; k_max if none (entry k - 1 is k clusters)."""
    return next((k for k in range(1, len(gap)) if gap[k - 1] >= gap[k] - s[k]), len(gap))


def pilot_rule(gap: np.ndarray, s: np.ndarray, min_gain: float = c.MIN_GAIN) -> int:
    """Tibshirani's k̂, then 1 unless Gap(k̂) - Gap(1) reaches ``min_gain`` (the chromatic pilot's stop_rule)."""
    k_hat = tibshirani(gap, s)
    return k_hat if k_hat == 1 or gap[k_hat - 1] - gap[0] >= min_gain else 1


def pack(labels: np.ndarray) -> np.ndarray:
    """Two labels below 16 per byte: the even rows in the high half, the odd rows in the low half."""
    if labels.max(initial=0) >= 16:
        raise ValueError("labels must be below 16 to pack two per byte")
    padded = np.append(labels, np.uint8(0)) if len(labels) % 2 else labels
    return (padded[0::2] << 4) | padded[1::2]


def unpack(packed: np.ndarray, n: int) -> np.ndarray:
    labels = np.empty(2 * len(packed), np.uint8)
    labels[0::2], labels[1::2] = packed >> 4, packed & 15
    return labels[:n]


def profiles(counts: np.ndarray, labels: np.ndarray, k: int) -> list[dict]:
    """Each cluster's cells, mean cells per ball, mean class fractions and mean kept fractions."""
    n_class = len(c.CLASSES)
    size = np.zeros(k)
    sums = np.zeros((k, 1 + n_class + c.N_THRESHOLDS - 1))
    for start in range(0, len(counts), CHUNK):
        chunk = np.asarray(counts[start : start + CHUNK], dtype=np.float64)
        lab = labels[start : start + CHUNK]
        n_ball = chunk[:, :n_class].sum(axis=1)
        values = np.column_stack([n_ball, chunk / n_ball[:, None]])
        size += np.bincount(lab, minlength=k)
        for j in range(values.shape[1]):
            sums[:, j] += np.bincount(lab, weights=values[:, j], minlength=k)
    means = sums / np.maximum(size, 1)[:, None]
    return [
        {
            "n": int(size[j]),
            "share": float(size[j] / size.sum()),
            "cells_per_ball": float(means[j, 0]),
            "class_fractions": dict(zip(c.CLASSES, means[j, 1 : 1 + n_class].round(4).tolist(), strict=True)),
            "kept_fractions": [1.0, *means[j, 1 + n_class :].round(4).tolist()],
        }
        for j in range(k)
    ]


def check_against_kmeans(X: np.ndarray, rng: np.random.Generator) -> dict:
    """e-means' W_k against scikit-learn k-means (k-means++, 20 starts) on a random subsample."""
    from sklearn.cluster import KMeans

    sample = X[np.sort(rng.choice(len(X), size=min(CHECK_ROWS, len(X)), replace=False))]
    out = {}
    for k in CHECK_KS:
        e = fit(sample, k, seed_of(99, k))["mse"]
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(sample)
        out[str(k)] = {
            "e_means": e,
            "kmeans": float(km.inertia_ / len(sample)),
            "ratio": e / (km.inertia_ / len(sample)),
        }
    return out


def run_set(counts: np.ndarray, index: int, name: str) -> dict:
    started = time.perf_counter()
    X, built = feature_matrix(counts, name)
    print(
        f"{name}: {X.shape[0]:,} cells × {X.shape[1]} columns, built in {time.perf_counter() - started:.0f} s",
        flush=True,
    )
    check = check_against_kmeans(X, np.random.default_rng([c.SEED, index, 0xC4]))
    print(f"  e-means vs k-means W_k on {CHECK_ROWS:,} cells: {check}", flush=True)
    data, labels = [], {}
    for k in range(1, c.K_MAX + 1):
        t0 = time.perf_counter()
        result = fit(X, k, seed_of(index, 0, k))
        if result["labels"] is not None:
            labels[k] = result.pop("labels")
        result.update(k=k, seconds=round(time.perf_counter() - t0, 1))
        data.append(result)
        print(f"  data k={k:>2}: W_k {result['mse']:.6f} in {result['seconds']} s", flush=True)
    mean, cov = column_moments(X)
    del X
    log_ref = np.empty((c.N_REFERENCES, c.K_MAX))
    for b in range(c.N_REFERENCES):
        t0 = time.perf_counter()
        R = reference(mean, cov, c.N_REFERENCE_POINTS, np.random.default_rng([c.SEED, index, b, 0xF00]))
        log_ref[b] = [np.log(fit(R, k, seed_of(index, b + 1, k))["mse"]) for k in range(1, c.K_MAX + 1)]
        print(f"  reference {b + 1:>2}/{c.N_REFERENCES}: {time.perf_counter() - t0:.0f} s", flush=True)
        del R
    log_w = np.log([d["mse"] for d in data])
    gap = log_ref.mean(axis=0) - log_w
    s = log_ref.std(axis=0) * np.sqrt(1 + 1 / c.N_REFERENCES)
    packed = {f"k{k}": pack(lab) for k, lab in labels.items()}
    np.savez_compressed(c.RESULTS / f"labels_{name.replace('+', '_')}.npz", n_cells=np.int64(len(counts)), **packed)
    clusters = {str(k): profiles(counts, lab, k) for k, lab in labels.items()}
    out = {
        "set": name,
        "features": built,
        "n_cells": int(len(counts)),
        "check_against_kmeans": check,
        "k": list(range(1, c.K_MAX + 1)),
        "log_w": log_w.tolist(),
        "log_w_reference_mean": log_ref.mean(axis=0).tolist(),
        "log_w_reference_sd": log_ref.std(axis=0).tolist(),
        "gap": gap.tolist(),
        "s": s.tolist(),
        "k_tibshirani": tibshirani(gap, s),
        "k_pilot_rule": pilot_rule(gap, s),
        "fits": [{key: value for key, value in d.items() if key != "centroids"} for d in data],
        "centroids": {str(d["k"]): d["centroids"] for d in data},
        "clusters": clusters,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (c.RESULTS / f"gap_{name.replace('+', '_')}.json").write_text(json.dumps(out) + "\n")
    print(
        f"  {name}: Tibshirani k = {out['k_tibshirani']}, pilot rule k = {out['k_pilot_rule']}; {out['seconds']:.0f} s",
        flush=True,
    )
    return out


def main() -> None:
    counts = np.load(c.RESULTS / "counts.npy", mmap_mode="r")
    results = {}
    for index, name in enumerate(c.SETS):
        stem = name.replace("+", "_")
        done = c.RESULTS / f"gap_{stem}.json"
        if done.exists() and (c.RESULTS / f"labels_{stem}.npz").exists():
            results[name] = json.loads(done.read_text())
            print(f"{name}: done before, kept", flush=True)
            continue
        results[name] = run_set(counts, index, name)
    settings = {
        "radius_um": c.RADIUS_UM,
        "k_max": c.K_MAX,
        "n_epochs": c.N_EPOCHS,
        "n_references": c.N_REFERENCES,
        "n_reference_points": c.N_REFERENCE_POINTS,
        "min_gain": c.MIN_GAIN,
        "seed": c.SEED,
        "reference": "Gaussian with the set's mean and covariance (numpy multivariate_normal, eigh)",
        "w_k": "mean squared distance to the nearest final centroid (e-means assign); k = 1: to the mean",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "faiss": faiss.__version__,
            "platform": platform.platform(),
        },
        "script_md5": c.md5(c.HERE / "cluster.py"),
    }
    gap = {
        name: {
            key: r[key]
            for key in (
                "features",
                "n_cells",
                "check_against_kmeans",
                "k",
                "log_w",
                "log_w_reference_mean",
                "log_w_reference_sd",
                "gap",
                "s",
                "k_tibshirani",
                "k_pilot_rule",
                "fits",
                "seconds",
            )
        }
        for name, r in results.items()
    }
    (c.RESULTS / "gap.json").write_text(json.dumps({"settings": settings, "sets": gap}, indent=1) + "\n")
    clusters = {name: {"centroids": r["centroids"], "clusters": r["clusters"]} for name, r in results.items()}
    (c.RESULTS / "clusters.json").write_text(json.dumps(clusters, indent=1) + "\n")
    for name, r in results.items():
        print(name, "gap", np.round(r["gap"], 4).tolist(), "k̂", r["k_tibshirani"], r["k_pilot_rule"])


if __name__ == "__main__":
    main()
