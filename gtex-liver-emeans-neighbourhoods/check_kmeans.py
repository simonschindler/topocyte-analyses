#!/usr/bin/env python
"""How close the run's e-means fits are to k-means++, and the gap statistic with k-means++ fits on both sides.

cluster.py's own check refits e-means to a 500,000-cell subsample, a separate fit (e-means sizes its batch and
steps by the sample: 1,225 steps of 2,048 cells there, 2,410 of 262,144 on every cell), so it says nothing about
the run's fits. This script measures them. For each feature set:

- on the same 500,000 cells (cluster.py's subsample, rebuilt from ``counts.npy`` and the run's column moments,
  checked by reproducing the run's k-means values at k = 4 and 10), W_k of the run's e-means centroids, which
  were fitted to every cell, and of scikit-learn's k-means (k-means++, 20 starts, as cluster.py) fitted to the
  subsample, both as mean squared distance in float64, k = 1..10;
- on 500,000 points of the set's Gaussian reference (its mean and covariance over every cell, as cluster.py's
  references), W*_k of k-means fitted the same way;
- the gap statistic from those two k-means curves, log W*_k − log W_k, and both rules' k̂ with the run's s_k,
  with the standard error of each Gap from the two samples' spread.

Writes ``results/kmeans_check.json``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import sklearn
from sklearn.cluster import KMeans

import cluster
import common as c

ROWS = cluster.CHECK_ROWS  # cluster.py's subsample of cells
REFERENCE_POINTS = 500_000
N_INIT = 20
TOL_REPRODUCE = 1e-6  # relative; the same cells and the same k-means give the same W_k


def subsample_rows(n: int, index: int) -> np.ndarray:
    """cluster.check_against_kmeans' rows for the set with this index."""
    rng = np.random.default_rng([c.SEED, index, 0xC4])
    return np.sort(rng.choice(n, size=min(ROWS, n), replace=False))


def rebuild(counts: np.ndarray, name: str, features: dict) -> np.ndarray:
    """The run's float32 feature rows for these counts, from its saved columns and column moments."""
    x, columns = cluster.fractions(counts, name)
    used = [columns.index(col) for col in features["columns"]]
    mean, sd, weight = (np.array(features[key]) for key in ("column_means", "column_sds", "column_weights"))
    return ((x[:, used] - mean) / sd * weight).astype(np.float32)


def moments(counts: np.ndarray, name: str, features: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the run's feature matrix, summed by cluster.py's chunks."""
    n, d = len(counts), len(features["columns"])
    total, cross = np.zeros(d), np.zeros((d, d))
    for start in range(0, n, cluster.CHUNK):
        x = rebuild(np.asarray(counts[start : start + cluster.CHUNK]), name, features).astype(np.float64)
        total += x.sum(axis=0)
        cross += x.T @ x
    mean = total / n
    return mean, cross / n - np.outer(mean, mean)


def nearest_sq(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Each row's squared distance to its nearest centroid, in float64."""
    X = X.astype(np.float64)
    best = np.full(len(X), np.inf)
    for centre in np.asarray(centroids, dtype=np.float64):
        np.minimum(best, ((X - centre) ** 2).sum(axis=1), out=best)
    return best


def kmeans_curve(X: np.ndarray) -> tuple[list[np.ndarray], list[float], dict]:
    """Per k = 1..10: each row's squared distance to its nearest k-means centroid, the fit's time, and W_k as
    scikit-learn's inertia (float32) at the k cluster.py checked."""
    d2, seconds, inertia = [nearest_sq(X, [X.astype(np.float64).mean(axis=0)])], [0.0], {}
    for k in range(2, c.K_MAX + 1):
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=0).fit(X)
        seconds.append(round(time.perf_counter() - t0, 1))
        d2.append(nearest_sq(X, km.cluster_centers_))
        if k in cluster.CHECK_KS:
            inertia[str(k)] = float(km.inertia_ / len(X))
    return d2, seconds, inertia


def se_of_log_mean(d2: np.ndarray) -> float:
    return float(d2.std() / (np.sqrt(len(d2)) * d2.mean()))


def check_set(counts: np.ndarray, index: int, name: str, run: dict, centroids: dict) -> dict:
    started = time.perf_counter()
    features = run["features"]
    X = rebuild(np.asarray(counts[subsample_rows(len(counts), index)]), name, features)
    small = cluster.fit(X, 4, cluster.seed_of(99, 4))  # cluster.py's subsample e-means fit, for its step count
    d2, km_seconds, inertia = kmeans_curve(X)
    logged = run["check_against_kmeans"]
    reproduced = {k: [logged[k]["kmeans"], inertia[k]] for k in inertia}
    for k, (was, now) in reproduced.items():
        if abs(now / was - 1) > TOL_REPRODUCE:
            raise RuntimeError(f"{name}: k-means W_{k} {now} on the rebuilt subsample, the run logged {was}")
    d2_run = [nearest_sq(X, centroids[str(k)]) for k in range(1, c.K_MAX + 1)]
    print(
        f"  {name}: subsample rebuilt, k-means at k = 4 and 10 reproduced; k-means {sum(km_seconds):.0f} s", flush=True
    )

    mean, cov = moments(counts, name, features)
    R = cluster.reference(mean, cov, REFERENCE_POINTS, np.random.default_rng([c.SEED, index, 0xC4, 1]))
    d2_ref, ref_seconds, _ = kmeans_curve(R)
    print(f"  {name}: reference k-means {sum(ref_seconds):.0f} s", flush=True)

    w_km = np.array([v.mean() for v in d2])
    w_run = np.array([v.mean() for v in d2_run])
    w_ref = np.array([v.mean() for v in d2_ref])
    gap_km = np.log(w_ref) - np.log(w_km)
    se_data = np.array([se_of_log_mean(v) for v in d2])
    se_ref = np.array([se_of_log_mean(v) for v in d2_ref])
    paired_se = np.array([float((a - b).std() / (np.sqrt(len(a)) * b.mean())) for a, b in zip(d2_run, d2, strict=True)])
    s = np.array(run["s"])
    return {
        "rows": len(X),
        "reference_points": REFERENCE_POINTS,
        "reproduces_run_kmeans": reproduced,
        "subsample_emeans_k4": {
            "logged_mse": logged["4"]["e_means"],
            "mse": small["mse"],
            "batch_size": small["recipe"]["batch_size"],
            "total_steps": small["recipe"]["total_steps"],
        },
        "trace_check": {"run_w1": float(np.exp(run["log_w"][0])), "rebuilt_w1": float(np.trace(cov))},
        "k": list(range(1, c.K_MAX + 1)),
        "w_run_centroids": w_run.tolist(),
        "w_kmeans": w_km.tolist(),
        "log_ratio_run_to_kmeans": np.log(w_run / w_km).tolist(),
        "log_ratio_se": paired_se.tolist(),
        "w_reference_kmeans": w_ref.tolist(),
        "gap_kmeans": gap_km.tolist(),
        "gap_kmeans_se": np.sqrt(se_data**2 + se_ref**2).tolist(),
        "k_tibshirani": cluster.tibshirani(gap_km, s),
        "k_pilot_rule": cluster.pilot_rule(gap_km, s),
        "kmeans_seconds": {"cells": km_seconds, "reference": ref_seconds},
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    gap = json.loads((c.RESULTS / "gap.json").read_text())
    clusters = json.loads((c.RESULTS / "clusters.json").read_text())
    counts = np.load(c.RESULTS / "counts.npy", mmap_mode="r")
    sets = {}
    for index, name in enumerate(c.SETS):
        sets[name] = check_set(counts, index, name, gap["sets"][name], clusters[name]["centroids"])
        r = sets[name]
        print(
            f"{name}: log(W run / W k-means) {np.round(r['log_ratio_run_to_kmeans'], 4).tolist()}\n"
            f"  k-means gap {np.round(r['gap_kmeans'], 4).tolist()} k̂ {r['k_tibshirani']} {r['k_pilot_rule']}",
            flush=True,
        )
    settings = {
        "rows": ROWS,
        "reference_points": REFERENCE_POINTS,
        "kmeans": f"scikit-learn KMeans(n_clusters=k, n_init={N_INIT}, random_state=0), k-means++",
        "w_k": "mean squared distance to the nearest centroid, float64",
        "s_k": "the run's (gap.json), for both rules",
        "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "script_md5": c.md5(c.HERE / "check_kmeans.py"),
        "cluster_md5": c.md5(c.HERE / "cluster.py"),
    }
    (c.RESULTS / "kmeans_check.json").write_text(json.dumps({"settings": settings, "sets": sets}, indent=1) + "\n")


if __name__ == "__main__":
    main()
