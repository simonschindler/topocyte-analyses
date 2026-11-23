#!/usr/bin/env python
"""Check the saved results of the e-means neighbourhood study against the cohort, the counts and each other.

Run from this folder with the study venv after ``features.py`` and ``cluster.py``, with nothing else heavy open.
Reads, read only, the cohort .h5ad (h5py mode "r"), the pathology study's density.npz and this study's
``results/``. Writes ``results/verify.json``: every check with its measured value and tolerance, the md5 of this
script and the sha256 of every gitignored file it read. Exits with status 1 if a check fails.

Checks, most by another route than the code they check:

- ``counts.npy``: its shape and type; in every ball the cell itself, kept counts that fall with the threshold and
  never exceed the ball's cells, and a cell at or above a threshold kept in its own ball; its maximum and mean
  ball as ``features.json`` says;
- the balls of 1,200 sampled cells (50 on each of 24 sampled slides) recounted from every distance to the slide's
  cells, without a tree;
- per feature set, from the counts in float64: the columns' means and SDs (merged by chunks, Chan's update, not
  sums of squares), the constant column left out and the block weights; every cell's saved label is its nearest
  saved centroid, up to float32 ties; W_k for k = 1..10 from the labels; every cluster's size, cells per ball and
  mean class and kept fractions;
- the gap arithmetic: Gap(k), s_k and both rules' k̂ from the saved W_k and the references' mean and SD, by loops of
  its own; ``gap.json`` and ``clusters.json`` as the per-set files; the md5 of ``features.py`` and ``cluster.py``;
- per set, the first reference redrawn and refitted with ``cluster.py``'s own functions and seeds (this reproduces
  rather than verifies): its log W*_k within √(B − 1) SDs of the saved mean, the farthest any one of B references
  can lie from their mean.

Not redone: the data's e-means fits (their centroids are taken as saved), the other 19 references of each set, the
check against scikit-learn's k-means, and the runs' timings and versions.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import cluster
import common as c

CHUNK = 2_000_000  # rows per chunk of counts
SAMPLE_SLIDES, SAMPLE_CELLS = 24, 50  # the brute-force recount
SAMPLE_KEY = 0x7E51  # SeedSequence([SEED, SAMPLE_KEY]) draws the recount's slides and cells
BOUNDARY = 1e-12  # the recount's radius shrunk and grown by this share: cKDTree's rounding at the ball's edge
TOL_MOMENT = 1e-9  # relative: the columns' means and SDs, float64 in another order
TOL_W = 1e-5  # relative: W_k from float64 features, against cluster.py's float32 features and faiss distances
TIE = 1e-5  # a saved label's distance may exceed the nearest by this share of |x|² + max |c|² + 1 (float32 ties)
TOL_ROUNDED = 5e-5 + 1e-12  # the profiles' fractions are saved rounded to 4 decimals
TOL_EXACT = 1e-12
REFERENCE_SLACK = 1e-4  # the redrawn reference's moments come from float64 features, cluster.py's from float32

checks: list[dict] = []


def record(name: str, ok: bool, measured, tolerance=None) -> bool:
    checks.append({"check": name, "ok": bool(ok), "measured": measured, "tolerance": tolerance})
    tail = "" if tolerance is None else f" (tolerance {tolerance})"
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {measured}{tail}", flush=True)
    return bool(ok)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def set_columns(x: np.ndarray, name: str) -> tuple[np.ndarray, list[str]]:
    """A chunk of counts (int64) as one set's float64 columns, before any is left out, with their names."""
    n_class = len(c.CLASSES)
    n_ball = x[:, :n_class].sum(axis=1, keepdims=True).astype(np.float64)
    comp = (x[:, :n_class] / n_ball, [f"{k} fraction" for k in c.CLASSES])
    dens = (
        np.hstack([np.ones_like(n_ball), x[:, n_class:] / n_ball]),
        [f"kept fraction, threshold {j}" for j in range(c.N_THRESHOLDS)],
    )
    if name == "composition":
        return comp
    if name == "density":
        return dens
    return np.hstack([comp[0], dens[0]]), comp[1] + dens[1]


class Moments:
    """Running mean and sum of squared deviations per column, merged chunk by chunk (Chan, Golub and LeVeque)."""

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: np.ndarray) -> None:
        n_b, mean_b = len(x), x.mean(axis=0)
        m2_b = ((x - mean_b) ** 2).sum(axis=0)
        delta, n = mean_b - self.mean, self.n + n_b
        self.m2 = self.m2 + m2_b + delta**2 * self.n * n_b / n
        self.mean = self.mean + delta * n_b / n
        self.n = n

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(self.m2 / self.n)


def brute_ball(xy: np.ndarray, classes: np.ndarray, rho: np.ndarray, thresholds: np.ndarray, centre: int):
    """One cell's closed-ball counts from every distance, no tree: rows for the radius, shrunk and grown by
    ``BOUNDARY``; each row is the cells per class, then the cells with ``rho >= thresholds[j]``, j >= 1."""
    d2 = ((xy - xy[centre]) ** 2).sum(axis=1)
    rows = []
    for r2 in (c.RADIUS_PX**2, c.RADIUS_PX**2 * (1 - BOUNDARY), c.RADIUS_PX**2 * (1 + BOUNDARY)):
        inside = d2 <= r2
        row = [np.count_nonzero(inside & (classes == k)) for k in range(len(c.CLASSES))]
        row += [np.count_nonzero(inside & (rho >= t)) for t in thresholds[1:]]
        rows.append(np.array(row, dtype=np.int64))
    return rows


def label_check(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Each row's squared distance to its saved centroid, how many saved labels are not the float64 nearest
    centroid, and how many of those lie farther from it than float32 ties allow (``TIE``)."""
    d2 = np.empty((len(X), len(centroids)))
    for j, centre in enumerate(centroids):
        d2[:, j] = ((X - centre) ** 2).sum(axis=1)
    own = d2[np.arange(len(X)), labels]
    best = d2.min(axis=1)
    differ = own > best
    scale = (X * X).sum(axis=1) + (centroids * centroids).sum(axis=1).max() + 1
    return own, int(differ.sum()), int((differ & (own - best > TIE * scale)).sum())


def tibshirani(gap: np.ndarray, s: np.ndarray) -> int:
    for k in range(1, len(gap)):
        if gap[k - 1] >= gap[k] - s[k]:
            return k
    return len(gap)


def check_counts_file(counts: np.ndarray, feats: dict, rho: np.ndarray, thresholds: np.ndarray) -> None:
    n_class = len(c.CLASSES)
    record(
        "counts.npy: cells × columns, uint16",
        counts.shape == (feats["n_cells"], len(c.COUNT_COLUMNS)) and counts.dtype == np.uint16 and len(rho) == len(counts),
        [list(counts.shape), str(counts.dtype)],
    )
    empty = rising = above_ball = missing = total = 0
    maximum = 0
    for start in range(0, len(counts), CHUNK):
        x = np.asarray(counts[start : start + CHUNK], dtype=np.int64)
        n_ball = x[:, :n_class].sum(axis=1)
        kept = np.column_stack([n_ball, x[:, n_class:]])
        empty += int(np.count_nonzero(n_ball < 1))
        rising += int(np.count_nonzero((np.diff(kept, axis=1) > 0).any(axis=1)))
        above_ball += int(np.count_nonzero((x[:, n_class:] > n_ball[:, None]).any(axis=1)))
        own = rho[start : start + CHUNK, None] >= thresholds[None, 1:]
        missing += int(np.count_nonzero((own & (x[:, n_class:] < 1)).any(axis=1)))
        maximum = max(maximum, int(x.max()))
        total += int(n_ball.sum())
    record(
        "counts.npy: balls without their cell, kept counts rising or above the ball's, kept cells missing from their ball",
        empty == rising == above_ball == missing == 0,
        [empty, rising, above_ball, missing],
    )
    mean = total / len(counts)
    record(
        "counts.npy: maximum count and mean cells per ball as features.json",
        maximum == feats["max_count"] and abs(mean - feats["mean_cells_per_ball"]) <= TOL_EXACT * mean,
        {"max": maximum, "mean": mean},
        TOL_EXACT,
    )
    record(
        "features.json: thresholds as density.npz's deciles",
        feats["thresholds_cells_per_mm2"] == thresholds.tolist() and feats["radius_px"] == c.RADIUS_PX,
        len(thresholds),
    )


def recount_sample(counts, rho, thresholds, bounds, slide_ids) -> None:
    import h5py

    rng = np.random.default_rng([c.SEED, SAMPLE_KEY])
    picks = np.sort(rng.choice(len(slide_ids), size=SAMPLE_SLIDES, replace=False))
    exact = edge = wrong = 0
    with h5py.File(c.H5AD, "r") as f:
        class_of = c.class_lookup(c.strings(f["obs/cell_type/categories"][:]))
        for i in picks:
            start, end = int(bounds[i]), int(bounds[i + 1])
            xy = np.asarray(f["obsm/spatial"][start:end], dtype=np.float64)
            classes = class_of[np.asarray(f["obs/cell_type/codes"][start:end])]
            for j in np.sort(rng.choice(end - start, size=SAMPLE_CELLS, replace=False)):
                at, low, high = brute_ball(xy, classes, rho[start:end], thresholds, int(j))
                saved = np.asarray(counts[start + j], dtype=np.int64)
                if np.array_equal(saved, at):
                    exact += 1
                elif np.all(low <= saved) and np.all(saved <= high):
                    edge += 1
                else:
                    wrong += 1
    record(
        f"ball counts of {SAMPLE_SLIDES * SAMPLE_CELLS:,} sampled cells recounted from every distance "
        "(exact, within the boundary's rounding, wrong)",
        wrong == 0,
        [exact, edge, wrong],
        BOUNDARY,
    )


def verify_set(counts: np.ndarray, name: str) -> dict:
    """Every check of one feature set against the counts; returns the moments the reference redraw needs."""
    stem = name.replace("+", "_")
    g = json.loads((c.RESULTS / f"gap_{stem}.json").read_text())
    built, n = g["features"], len(counts)
    K = c.K_MAX

    moments, names = Moments(), None
    for start in range(0, n, CHUNK):
        x, names = set_columns(np.asarray(counts[start : start + CHUNK], dtype=np.int64), name)
        moments.add(x)
    sd = moments.sd
    used = [j for j in range(len(names)) if sd[j] > 1e-12]
    left_out = [names[j] for j in range(len(names)) if j not in used]
    record(
        f"{name}: columns used and the constant ones left out",
        [names[j] for j in used] == built["columns"] and left_out == built["left_out_constant"],
        {"used": len(used), "left out": left_out, "smallest SD used": float(sd[used].min())},
    )
    mean_err = float(np.max(np.abs(moments.mean[used] - built["column_means"]) / np.abs(moments.mean[used])))
    sd_err = float(np.max(np.abs(sd[used] - built["column_sds"]) / sd[used]))
    record(f"{name}: column means and SDs (relative)", max(mean_err, sd_err) <= TOL_MOMENT, [mean_err, sd_err], TOL_MOMENT)
    blocks = ["density" if col.startswith("kept") else "composition" for col in built["columns"]]
    expected = [
        1 / math.sqrt(blocks.count(b)) if name == "composition+density" else 1.0 for b in blocks
    ]
    weight_err = float(np.max(np.abs(np.array(expected) - built["column_weights"])))
    record(f"{name}: block weights 1/√(block columns) in the combined set, else 1", weight_err <= TOL_EXACT, weight_err)

    with np.load(c.RESULTS / f"labels_{stem}.npz") as z:
        keys = sorted(z.files)
        n_saved = int(z["n_cells"])
        labels = {k: cluster.unpack(z[f"k{k}"], n) for k in range(2, K + 1)}
    record(
        f"labels_{stem}.npz: every cell, k = 2..{K}, each label below its k",
        n_saved == n
        and keys == sorted(["n_cells", *(f"k{k}" for k in range(2, K + 1))])
        and all(int(lab.max()) < k for k, lab in labels.items()),
        {"cells": n_saved, "arrays": len(keys)},
    )

    mean_s, sd_s, w_s = (np.array(built[key]) for key in ("column_means", "column_sds", "column_weights"))
    centroids = {k: np.array(g["centroids"][str(k)]) for k in range(2, K + 1)}
    d = len(used)
    sum_x, cross, sq = np.zeros(d), np.zeros((d, d)), dict.fromkeys(range(2, K + 1), 0.0)
    differ, beyond = dict.fromkeys(range(2, K + 1), 0), dict.fromkeys(range(2, K + 1), 0)
    size = {k: np.zeros(k) for k in range(2, K + 1)}
    sums = {k: np.zeros((k, 1 + len(c.COUNT_COLUMNS))) for k in range(2, K + 1)}
    n_class = len(c.CLASSES)
    for start in range(0, n, CHUNK):
        raw = np.asarray(counts[start : start + CHUNK], dtype=np.int64)
        X = (set_columns(raw, name)[0][:, used] - mean_s) / sd_s * w_s
        sum_x += X.sum(axis=0)
        cross += X.T @ X
        n_ball = raw[:, :n_class].sum(axis=1).astype(np.float64)
        values = np.column_stack([n_ball, raw / n_ball[:, None]])
        for k in range(2, K + 1):
            lab = labels[k][start : start + CHUNK]
            own, n_differ, n_beyond = label_check(X, centroids[k], lab)
            sq[k] += float(own.sum())
            differ[k] += n_differ
            beyond[k] += n_beyond
            size[k] += np.bincount(lab, minlength=k)
            sums[k] += np.stack([np.bincount(lab, weights=values[:, j], minlength=k) for j in range(values.shape[1])], 1)
    del labels
    record(
        f"{name}: every saved label is its nearest saved centroid, up to float32 ties (k: relabelled, beyond ties)",
        all(v == 0 for v in beyond.values()),
        {k: [differ[k], beyond[k]] for k in differ},
        TIE,
    )
    mean = sum_x / n
    cov = cross / n - np.outer(mean, mean)
    w = np.array([float(np.trace(cov)), *(sq[k] / n for k in range(2, K + 1))])
    w_err = np.abs(w / np.exp(g["log_w"]) - 1)
    record(f"{name}: W_k, k = 1..{K}, from the labels (relative)", float(w_err.max()) <= TOL_W, w_err.round(9).tolist(), TOL_W)
    fit_err = max(abs(math.log(f["mse"]) - lw) for f, lw in zip(g["fits"], g["log_w"], strict=True))
    record(f"{name}: log W_k as the fits' mse", fit_err <= TOL_EXACT, fit_err)

    worst = {"n": 0, "share": 0.0, "cells per ball": 0.0, "fractions": 0.0}
    for k in range(2, K + 1):
        saved = g["clusters"][str(k)]
        means = sums[k] / np.maximum(size[k], 1)[:, None]
        for j, row in enumerate(saved):
            worst["n"] = max(worst["n"], abs(row["n"] - int(size[k][j])))
            worst["share"] = max(worst["share"], abs(row["share"] - size[k][j] / n))
            worst["cells per ball"] = max(worst["cells per ball"], abs(row["cells_per_ball"] / means[j, 0] - 1))
            fractions = [*row["class_fractions"].values(), *row["kept_fractions"][1:]]
            worst["fractions"] = max(worst["fractions"], float(np.max(np.abs(np.array(fractions) - means[j, 1:]))))
            worst["fractions"] = max(worst["fractions"], abs(row["kept_fractions"][0] - 1.0))
    record(
        f"{name}: cluster sizes, shares, cells per ball and fractions (largest error)",
        worst["n"] == 0 and worst["share"] <= TOL_EXACT and worst["cells per ball"] <= TOL_MOMENT
        and worst["fractions"] <= TOL_ROUNDED,
        worst,
        {"share": TOL_EXACT, "cells per ball": TOL_MOMENT, "fractions": TOL_ROUNDED},
    )

    log_w, ref_mean, ref_sd = (np.array(g[key]) for key in ("log_w", "log_w_reference_mean", "log_w_reference_sd"))
    gap = ref_mean - log_w
    s = ref_sd * math.sqrt(1 + 1 / c.N_REFERENCES)
    gap_err = float(max(np.max(np.abs(gap - g["gap"])), np.max(np.abs(s - g["s"]))))
    record(f"{name}: Gap(k) and s_k from the saved W_k and references", gap_err <= TOL_EXACT, gap_err)
    k_hat = tibshirani(gap, s)
    k_pilot = k_hat if k_hat == 1 or gap[k_hat - 1] - gap[0] >= c.MIN_GAIN else 1
    record(
        f"{name}: k̂, Tibshirani's rule and the pilot's",
        k_hat == g["k_tibshirani"] and k_pilot == g["k_pilot_rule"],
        [k_hat, k_pilot],
    )
    return {"mean": mean, "cov": cov, "ref_mean": ref_mean, "ref_sd": ref_sd}


def redraw_reference(index: int, name: str, moments: dict) -> None:
    """Reference 1 of the set redrawn and refitted with cluster.py's functions and seeds (reproduces)."""
    R = cluster.reference(
        moments["mean"], moments["cov"], c.N_REFERENCE_POINTS, np.random.default_rng([c.SEED, index, 0, 0xF00])
    )
    redrawn = np.array([math.log(cluster.fit(R, k, cluster.seed_of(index, 1, k))["mse"]) for k in range(1, c.K_MAX + 1)])
    del R
    bound = moments["ref_sd"] * math.sqrt(c.N_REFERENCES - 1) + REFERENCE_SLACK
    deviation = np.abs(redrawn - moments["ref_mean"])
    record(
        f"{name}: reference 1 redrawn, log W*_k within √(B − 1) SDs of the saved mean",
        bool(np.all(deviation <= bound)),
        {"deviation": deviation.round(6).tolist(), "bound": bound.round(6).tolist()},
        REFERENCE_SLACK,
    )


def check_summaries() -> None:
    gap_all = json.loads((c.RESULTS / "gap.json").read_text())
    clusters_all = json.loads((c.RESULTS / "clusters.json").read_text())
    same = True
    for name in c.SETS:
        g = json.loads((c.RESULTS / f"gap_{name.replace('+', '_')}.json").read_text())
        same &= all(g[key] == value for key, value in gap_all["sets"][name].items())
        same &= clusters_all[name] == {"centroids": g["centroids"], "clusters": g["clusters"]}
    record("gap.json and clusters.json as the per-set files", same and list(gap_all["sets"]) == list(c.SETS), len(c.SETS))
    feats = json.loads((c.RESULTS / "features.json").read_text())
    record(
        "md5 of features.py and cluster.py as the results name them",
        feats["script_md5"] == c.md5(c.HERE / "features.py") and gap_all["settings"]["script_md5"] == c.md5(c.HERE / "cluster.py"),
        [feats["script_md5"], gap_all["settings"]["script_md5"]],
    )


def main() -> None:
    started = time.perf_counter()
    feats = json.loads((c.RESULTS / "features.json").read_text())
    counts = np.load(c.RESULTS / "counts.npy", mmap_mode="r")
    with np.load(c.SIB / "density.npz", allow_pickle=False) as density:
        rho = np.asarray(density["rho"])
        thresholds = np.asarray(density["deciles"], dtype=np.float64)
        bounds = np.asarray(density["bounds"], dtype=np.int64)
        slide_ids = [str(s) for s in density["slide_ids"]]
    check_counts_file(counts, feats, rho, thresholds)
    recount_sample(counts, rho, thresholds, bounds, slide_ids)
    del rho
    check_summaries()
    for index, name in enumerate(c.SETS):
        redraw_reference(index, name, verify_set(counts, name))
    files = {
        str(path): sha256(path)
        for path in (
            c.RESULTS / "counts.npy",
            *(c.RESULTS / f"labels_{name.replace('+', '_')}.npz" for name in c.SETS),
            c.SIB / "density.npz",
            c.H5AD,
        )
    }
    out = {
        "all_ok": all(check["ok"] for check in checks),
        "checks": checks,
        "script_md5": c.md5(c.HERE / "verify_results.py"),
        "sha256": files,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (c.RESULTS / "verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{sum(ch['ok'] for ch in checks)}/{len(checks)} checks pass; {out['seconds']:.0f} s", flush=True)
    sys.exit(0 if out["all_ok"] else 1)


if __name__ == "__main__":
    main()
