#!/usr/bin/env python
"""The rule for k (design step 10), and the strata, subdomains, control and guard of the chromatic-domains pilot.

The rule for k
--------------
Both levels cluster with k-means (scikit-learn ``KMeans``, ``n_init=20``,
``random_state`` from the caller's seed). The number of clusters is fixed in
advance by Simon's rule:

- **Gap statistic** (Tibshirani, Walther and Hastie 2001) on each half of the
  slides, against B = 20 draws from a Gaussian with the half's mean and
  covariance and its number of rows. W_1 is the total sum of squares about
  the column means and W_k (k >= 2) the k-means inertia, for the data and for
  every reference alike. Gap(k) = mean_b log W*_k - log W_k, and
  s_k = sd_b(log W*_k) * sqrt(1 + 1/B) with the SD taken with ddof 0.
- **A half's k** is where the gap stops rising: Tibshirani's stopping rule
  from k = 1 (``stop_rule``). k̂ is the smallest k >= 1 with
  Gap(k) >= Gap(k+1) - s_{k+1}, or k_max if no k < k_max has it. If k̂ >= 2,
  its gain Gap(k̂) - Gap(1) must reach 0.03, else the half's k is 1.
  Two corrections were made on 2026-11-21, before any study data, both on
  skewed cluster-free clouds:
  - Simon raised the floor from 0.02 to 0.03, because those clouds reached
    0.02.
  - The first formal version took the smallest k >= 2 with the gain and the
    stop, skipping k = 1. On those clouds in 5-d the gap falls at k = 2 and
    rises again from k = 5, so that version split them.
- **Both halves:** k is the smaller of the two halves' choices.
- **Prediction strength** (Tibshirani and Walther 2005) must then be at least
  0.8 in both directions between the halves, or k = 1. With k = 1 there is
  nothing to confirm.

Why the gap statistic and not prediction strength alone: prediction strength
measures how stable a split is, not whether clusters exist, and it splits
cluster-free elongated clouds (design, "Builder check").

Seeds are small integers from the caller (default 0): one seeds the
reference draws (``default_rng(seed)``) and every k-means fit alike. They are
separate from the study's ``SeedSequence`` keys.

The main (Task 7)
-----------------
Run from this folder with the study venv and
``PYTHONPATH=<topocyte checkout at 18e3aa5>``; it refuses any other topocyte
(``common.check_topocyte``). It reads ``results/features.npz`` and writes
``results/domains.npz`` and ``results/domains.json``, rows in the order of
``features.npz``. Every rule application below uses seed 0.

1. **Composition strata (level 1, design step 8; ``level1``).** ``L1``, the
   log10 of a ball's cells and its four class fractions, is z-scored with
   ddof 0: each half with its own means and SDs for its gap statistic, both
   halves with the training half's in each prediction-strength direction
   (``level1_inputs``), and all centres with all centres' for the final fit.
   The rule runs with k_max 10, and ``compartment`` is
   ``kmeans_fit(z_all, k).labels_``, all 0 when k = 1. The strata are bins of
   a continuum of compositions that recur across the halves, nothing more
   (Simon, 2026-11-21): the rule splits cluster-free composition clouds too.
2. **The control first ("Checks fixed in advance", 1; ``level2``).** Inside
   each stratum c, the rule with k_max 6 on the control's level-2 scores: gap
   statistics on ``L2_ctl_A[A∩c]`` and ``L2_ctl_B[B∩c]``, directions
   (``L2_ctl_A[A∩c]``, ``L2_ctl_A[B∩c]``) and (``L2_ctl_B[B∩c]``,
   ``L2_ctl_B[A∩c]``). A stratum with fewer than 500 centres in either half
   gets k = 1 without the rule, recorded as too small. If the control finds
   k > 1 in any stratum, level 2 stops: no observed level-2 fit,
   ``subdomain`` -1 everywhere, no guard, and the verdict "not run: control
   failed". That is a result of the study, not a fault.
3. **Subdomains (level 2, step 9; ``level2``).** The same on ``L2_obs_*``,
   then ``subdomain[c] = kmeans_fit(L2_obs_all[c], k_c).labels_``, and -1
   where k_c = 1.
4. **The pairwise guard ("Checks fixed in advance", 2; ``guard``)** in every
   stratum with k_c >= 2, both directions, with BA_dom beside it.
5. **The verdict (``verdict``).**
6. **Descriptives:** the gap and prediction-strength curves of every rule
   application (``ps_curve``), the cross-half adjusted Rand index
   (``cross_half_ari``) and the spatial contiguity (``contiguity``).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score

import common as c

N_INIT = 20  # k-means starts per fit
N_REFERENCES = 20  # B, reference draws of the gap statistic
MIN_GAIN = 0.03  # the smallest gain Gap(k) - Gap(1) that counts (Simon, 2026-11-21; first 0.02)
PS_MIN = 0.8  # the prediction strength needed in both directions

K_MAX_STRATA = 10  # k_max at level 1, the composition strata
K_MAX_SUBDOMAINS = 6  # k_max at level 2, the subdomains
MIN_HALF = 500  # centres of a stratum needed in each half for subdomains (design step 9)
N_DRAWS = 4000  # the guard's bootstrap draws of the test half's slides
BOOT_SEED = 12345  # a fresh default_rng(BOOT_SEED) per stratum and direction
MIN_SHARE = Fraction(1, 10)  # the share of all centres a passing stratum needs for the verdict "adds"
K_NN = 5  # the nearest same-slide centres of the contiguity
HALVES = ("A", "B")
DIRECTIONS = (("A", "B"), ("B", "A"))  # (training half, test half), named "A->B" and "B->A"
DOMAIN_BLOCKS = tuple(b for b, (_, module, _) in enumerate(c.BLOCKS) if module == "domain")  # the 8 domain blocks
# The arrays of results/features.npz the main reads.
FEATURE_KEYS = (
    "half", "slide_index", "xy", "L1", "P", "L2_obs_A", "L2_obs_B", "L2_obs_all", "L2_ctl_A", "L2_ctl_B",
    "blk_obs_A", "blk_obs_B",
)  # fmt: skip


def _matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 2 or not np.all(np.isfinite(X)):
        raise ValueError(f"need a finite (n, d) matrix with n >= 2, got shape {X.shape}")
    return X


def kmeans_fit(X: np.ndarray, k: int, seed: int = 0) -> KMeans:
    """k-means of the rows of ``X``: ``KMeans(k, n_init=20, random_state=seed)``, fitted.

    For k = 1 the single centroid is the mean: one start at the column means,
    which Lloyd's step keeps.
    """
    X = _matrix(X)
    if k == 1:
        return KMeans(n_clusters=1, init=X.mean(axis=0, keepdims=True), n_init=1, random_state=seed).fit(X)
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)


def _log_w(X: np.ndarray, k_max: int, seed: int) -> np.ndarray:
    """log W_k for k = 1..k_max: the total sum of squares about the column means, then k-means inertias."""
    centred = X - X.mean(axis=0)
    log_w = [np.log(np.sum(centred * centred))]
    log_w += [np.log(kmeans_fit(X, k, seed).inertia_) for k in range(2, k_max + 1)]
    return np.array(log_w)


def stop_rule(gap: np.ndarray, s: np.ndarray, min_gain: float) -> int:
    """A half's k from its gap curve and its s, both indexed k = 1..k_max (entry k - 1 for k clusters).

    Tibshirani's stopping rule from k = 1: k̂ is the smallest k >= 1 with
    Gap(k) >= Gap(k+1) - s_{k+1}, or k_max if no k < k_max has it. If
    k̂ >= 2, its gain Gap(k̂) - Gap(1) must reach ``min_gain``, else the
    half's k is 1.
    """
    gap, s = np.asarray(gap, dtype=np.float64), np.asarray(s, dtype=np.float64)
    if gap.ndim != 1 or gap.shape != s.shape or gap.size == 0:
        raise ValueError(f"need gap and s of one shape (k_max,), got {gap.shape} and {s.shape}")
    k_max = gap.size
    k_hat = next((k for k in range(1, k_max) if gap[k - 1] >= gap[k] - s[k]), k_max)
    return k_hat if k_hat == 1 or gap[k_hat - 1] - gap[0] >= min_gain else 1


def gap_choice(
    X: np.ndarray, k_max: int, *, B: int = N_REFERENCES, min_gain: float = MIN_GAIN, seed: int = 0
) -> dict:
    """One half's gap statistic against a matched Gaussian, and its choice of k (design step 10).

    The B references are ``rng.multivariate_normal(X.mean(0), np.cov(X,
    rowvar=False), size=len(X))`` from one ``default_rng(seed)``, drawn in
    order, and every fit, on ``X`` and on each reference, is
    ``kmeans_fit(·, k, seed)``. The half's k is ``stop_rule(gap, s,
    min_gain)``.

    Returns ``k``, and ``gap`` and ``s`` of shape (k_max,), entry k - 1 for k
    clusters.
    """
    X = _matrix(X)
    if not 2 <= k_max <= len(X):
        raise ValueError(f"k_max must be 2..{len(X)} (the rows), got {k_max}")
    rng = np.random.default_rng(seed)
    mean, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    observed = _log_w(X, k_max, seed)
    references = np.array([_log_w(rng.multivariate_normal(mean, cov, size=len(X)), k_max, seed) for _ in range(B)])
    gap = references.mean(axis=0) - observed
    s = references.std(axis=0) * np.sqrt(1 + 1 / B)
    return {"k": stop_rule(gap, s, min_gain), "gap": gap, "s": s}


def ps_from_labels(test_labels: np.ndarray, train_assign: np.ndarray) -> float:
    """The smallest share, over the test clusters, of a cluster's pairs of rows that ``train_assign`` puts together.

    For test cluster i, the share is sum_j C(n_ij, 2) / C(n_i, 2), where n_ij
    counts its rows with training label j: a contingency table, never a loop
    over pairs. A test cluster of one row counts as 0. Labels are any
    integers.
    """
    test_labels, train_assign = np.asarray(test_labels), np.asarray(train_assign)
    if test_labels.ndim != 1 or test_labels.shape != train_assign.shape or test_labels.size == 0:
        raise ValueError(f"need two label vectors of one length > 0, got {test_labels.shape} and {train_assign.shape}")
    _, i = np.unique(test_labels, return_inverse=True)
    _, j = np.unique(train_assign, return_inverse=True)
    n_test, n_train = int(i.max()) + 1, int(j.max()) + 1
    table = np.bincount(i * n_train + j, minlength=n_test * n_train).reshape(n_test, n_train).astype(np.int64)
    size = table.sum(axis=1)
    pairs = size * (size - 1) // 2
    together = (table * (table - 1) // 2).sum(axis=1)
    shares = np.where(pairs > 0, together / np.maximum(pairs, 1), 0.0)
    return float(shares.min())


def prediction_strength(train: np.ndarray, test: np.ndarray, k: int, *, seed: int = 0) -> float:
    """Prediction strength of k clusters from ``train`` to ``test`` (Tibshirani and Walther 2005).

    Both halves are clustered on their own with ``kmeans_fit(·, k, seed)``. The
    test clusters are then judged by the training centroids' assignment of the
    test rows (``ps_from_labels``).
    """
    train_model = kmeans_fit(train, k, seed)
    test_model = kmeans_fit(test, k, seed)
    return ps_from_labels(test_model.labels_, train_model.predict(_matrix(test)))


def choose_k(
    gap_A: np.ndarray,
    gap_B: np.ndarray,
    directions: Sequence[tuple[np.ndarray, np.ndarray]],
    k_max: int,
    *,
    seed: int = 0,
) -> dict:
    """The rule for k across the two halves (design step 10).

    ``gap_A`` and ``gap_B`` are the matrices of each half's gap statistic, and
    ``directions`` the two (train, test) pairs of prediction strength, A→B and
    B→A, each test half in its training half's coordinates. k is the smaller of
    the halves' ``gap_choice`` (B = 20, minimum gain 0.03); if it is at least
    2, it stands only with a prediction strength of at least 0.8 in both
    directions, else k = 1.

    Returns ``k``, ``k_A``, ``k_B``; ``ps``, the prediction strength per
    direction at min(k_A, k_B), empty when that is 1; and ``gap_A`` and
    ``gap_B``, the halves' ``gap_choice`` results with their curves ``gap``
    and ``s``.
    """
    if len(directions) != 2:
        raise ValueError(f"the rule needs both directions, A→B and B→A; got {len(directions)}")
    half_A = gap_choice(gap_A, k_max, seed=seed)
    half_B = gap_choice(gap_B, k_max, seed=seed)
    k = min(half_A["k"], half_B["k"])
    ps = [] if k == 1 else [prediction_strength(train, test, k, seed=seed) for train, test in directions]
    if any(p < PS_MIN for p in ps):
        k = 1
    return {"k": k, "k_A": half_A["k"], "k_B": half_B["k"], "ps": ps, "gap_A": half_A, "gap_B": half_B}


# ---------------------------------------------------------------------------------------- Task 7: the inputs and curves


def _name(train: str, test: str) -> str:
    return f"{train}->{test}"


def zscore(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``X``'s columns z-scored with ``ref``'s column means and SDs (ddof 0); refuses a constant column of ``ref``."""
    X, ref = np.asarray(X, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    sd = ref.std(axis=0)
    if np.any(sd == 0):
        raise ValueError("a constant column cannot be z-scored")
    return (X - ref.mean(axis=0)) / sd


def level1_inputs(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """``choose_k``'s inputs at level 1 (design step 8) from the two halves' raw features.

    Each half is z-scored with its own means and SDs for its gap statistic,
    and in each prediction-strength direction both halves are z-scored with
    the training half's. Returns ``gap_A``, ``gap_B`` and ``directions``.
    """
    zA, zB = zscore(A, A), zscore(B, B)
    return zA, zB, [(zA, zscore(B, A)), (zB, zscore(A, B))]


def ps_curve(directions: Sequence[tuple[np.ndarray, np.ndarray]], k_max: int, *, seed: int = 0) -> list[list[float]]:
    """Per direction, ``prediction_strength`` at k = 2..k_max (entry k - 2 for k clusters), for the curves."""
    return [[prediction_strength(train, test, k, seed=seed) for k in range(2, k_max + 1)] for train, test in directions]


def cross_half_ari(directions: Sequence[tuple[np.ndarray, np.ndarray]], k: int, *, seed: int = 0) -> list[float]:
    """Per direction, the adjusted Rand index between the test half's own k-means labels and the training centroids'
    assignment of the test half: the two partitions that prediction strength compares (descriptive)."""
    out = []
    for train, test in directions:
        assignment = kmeans_fit(train, k, seed).predict(_matrix(test))
        out.append(float(adjusted_rand_score(kmeans_fit(test, k, seed).labels_, assignment)))
    return out


def gap_summary(result: dict) -> dict:
    """One half's ``gap_choice`` result for the JSON, with its largest gain Gap(k) - Gap(1) over k >= 2."""
    gap, s = result["gap"], result["s"]
    largest_gain = float(np.max(gap[1:] - gap[0]))
    return {"k": int(result["k"]), "gap": gap.tolist(), "s": s.tolist(), "largest_gain": largest_gain}


def _choice(
    gap_A: np.ndarray, gap_B: np.ndarray, directions: list[tuple[np.ndarray, np.ndarray]], k_max: int, seed: int
) -> dict:
    """``choose_k`` on these inputs, with both halves' gap curves and the prediction-strength curves, for the JSON."""
    result = choose_k(gap_A, gap_B, directions, k_max, seed=seed)
    curves = ps_curve(directions, k_max, seed=seed)
    return {
        "k": int(result["k"]),
        "k_A": int(result["k_A"]),
        "k_B": int(result["k_B"]),
        "ps": [float(p) for p in result["ps"]],
        "gap_A": gap_summary(result["gap_A"]),
        "gap_B": gap_summary(result["gap_B"]),
        "ps_curve": {_name(*d): curve for d, curve in zip(DIRECTIONS, curves, strict=True)},
    }


def _ari(directions: list[tuple[np.ndarray, np.ndarray]], k: int, seed: int) -> dict[str, float] | None:
    if k < 2:
        return None
    return {_name(*d): a for d, a in zip(DIRECTIONS, cross_half_ari(directions, k, seed=seed), strict=True)}


# ------------------------------------------------------------------------------------------- Task 7: strata, subdomains


def level1(L1: np.ndarray, half: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, dict]:
    """The composition strata (design step 8): the rule for k on the halves' z-scored ``L1``, then the final fit.

    Returns ``compartment`` (N,), the labels 0..k-1 of ``kmeans_fit(zscore(L1,
    L1), k)`` and all 0 when k = 1, and the choice (``_choice``) with the
    cross-half ARI where k >= 2 and the z-scoring of each fitting set.
    """
    L1 = np.asarray(L1, dtype=np.float64)
    gap_A, gap_B, directions = level1_inputs(L1[half == "A"], L1[half == "B"])
    choice = _choice(gap_A, gap_B, directions, K_MAX_STRATA, seed)
    k = choice["k"]
    compartment = np.zeros(len(L1), np.int64)
    if k >= 2:
        compartment = kmeans_fit(zscore(L1, L1), k, seed).labels_.astype(np.int64)
    choice["ari"] = _ari(directions, k, seed)
    sets = {"A": L1[half == "A"], "B": L1[half == "B"], "all": L1}
    choice["zscore"] = {f: {"mean": X.mean(axis=0).tolist(), "sd": X.std(axis=0).tolist()} for f, X in sets.items()}
    return compartment, choice


def _too_small(n_A: int, n_B: int) -> dict:
    empty = dict.fromkeys(("k_A", "k_B", "ps", "gap_A", "gap_B", "ps_curve"))
    return {"k": 1, "too_small": True, "n_A": n_A, "n_B": n_B} | empty


def level2(
    X: Mapping[str, np.ndarray],
    half: np.ndarray,
    compartment: np.ndarray,
    n_strata: int,
    X_all: np.ndarray | None = None,
    *,
    seed: int = 0,
) -> tuple[list[dict], np.ndarray | None]:
    """The rule for k inside each composition stratum (design step 9), on level-2 scores ``X[h]`` in half h's basis.

    Stratum c's gap statistics run on ``X["A"][A∩c]`` and ``X["B"][B∩c]``,
    and its directions are (``X["A"][A∩c]``, ``X["A"][B∩c]``) and
    (``X["B"][B∩c]``, ``X["B"][A∩c]``), with k_max 6. A stratum with fewer than
    ``MIN_HALF`` centres in either half gets k = 1 without the rule, recorded
    as too small. Given ``X_all``, the all-slides basis, the subdomains are
    fitted too: ``kmeans_fit(X_all[c], k_c).labels_``, and -1 where k_c = 1;
    each choice then carries the cross-half ARI (None where k_c = 1).

    Returns the choice per stratum and the subdomains (None without ``X_all``).
    """
    subdomain = None if X_all is None else np.full(len(half), -1, np.int64)
    choices = []
    for comp in range(n_strata):
        in_c = compartment == comp
        A, B = in_c & (half == "A"), in_c & (half == "B")
        n_A, n_B = int(A.sum()), int(B.sum())
        if min(n_A, n_B) < MIN_HALF:
            choices.append(_too_small(n_A, n_B) | ({} if X_all is None else {"ari": None}))
            continue
        directions = [(X["A"][A], X["A"][B]), (X["B"][B], X["B"][A])]
        choice = {"too_small": False, "n_A": n_A, "n_B": n_B} | _choice(
            X["A"][A], X["B"][B], directions, K_MAX_SUBDOMAINS, seed
        )
        if subdomain is not None:
            if choice["k"] >= 2:
                subdomain[in_c] = kmeans_fit(X_all[in_c], choice["k"], seed).labels_
            choice["ari"] = _ari(directions, choice["k"], seed)
        choices.append({"k": choice.pop("k")} | choice)
    return choices, subdomain


# ---------------------------------------------------------------------------------------------------- Task 7: the guard


def match_labels(own: np.ndarray, final: np.ndarray) -> dict[int, int]:
    """The one-to-one map from ``own`` to ``final`` labels of the same rows that keeps the most rows in agreement.

    Labels are integers 0..m-1, m the larger maximum plus one. The m × m
    contingency table counts the rows with own label i and final label j, and
    ``scipy.optimize.linear_sum_assignment`` (maximize) picks the assignment:
    the Hungarian matching. Returns {own label: final label} for 0..m-1.
    """
    own, final = np.asarray(own), np.asarray(final)
    if own.ndim != 1 or own.shape != final.shape or own.size == 0:
        raise ValueError(f"need two label vectors of one length > 0, got {own.shape} and {final.shape}")
    integers = np.issubdtype(own.dtype, np.integer) and np.issubdtype(final.dtype, np.integer)
    if not integers or min(own.min(), final.min()) < 0:
        raise ValueError("labels must be integers >= 0")
    m = int(max(own.max(), final.max())) + 1
    table = np.bincount(own.astype(np.int64) * m + final, minlength=m * m).reshape(m, m)
    rows, cols = linear_sum_assignment(table, maximize=True)
    return {int(i): int(j) for i, j in zip(rows, cols, strict=True)}


def balanced_accuracy(conf: np.ndarray) -> np.ndarray:
    """Balanced accuracy from confusion counts ``conf[..., i, j]`` (true label i, predicted j), batched in front.

    The mean recall over the classes present in the true labels, computed
    without warnings: a class that is predicted but never true counts for
    nothing. A table without any true label is refused.
    """
    conf = np.asarray(conf)
    support = conf.sum(axis=-1)
    present = support > 0
    n_present = present.sum(axis=-1)
    if np.any(n_present == 0):
        raise ValueError("a confusion table without any true label")
    hits = np.diagonal(conf, axis1=-2, axis2=-1)
    recall = np.divide(hits, support, out=np.zeros(hits.shape), where=present)
    return recall.sum(axis=-1) / n_present


def confusion_by_slide(truth: np.ndarray, pred: np.ndarray, slide: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The slides in increasing order, and their confusion counts (n_slides, k, k): ``conf[s, i, j]`` counts slide s's
    centres with true label i and prediction j. Labels are 0..k-1."""
    truth, pred = np.asarray(truth, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    if truth.shape != pred.shape or min(truth.min(), pred.min()) < 0 or max(truth.max(), pred.max()) >= k:
        raise ValueError(f"need true and predicted labels of one shape in 0..{k - 1}")
    slides, position = np.unique(slide, return_inverse=True)
    flat = (position * k + truth) * k + pred
    return slides, np.bincount(flat, minlength=slides.size * k * k).reshape(slides.size, k, k)


def bootstrap_draws(n_slides: int, *, n_draws: int = N_DRAWS, seed: int = BOOT_SEED) -> np.ndarray:
    """The guard's bootstrap of a test half's slides: ``default_rng(seed).integers(0, n_slides, size=(n_draws,
    n_slides))``. Row d holds the positions (0..n_slides-1, slides in slide-index order) drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n_slides, size=(n_draws, n_slides))


def resampled_ba(conf: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """The balanced accuracy of each draw: the confusion counts of its slides (``conf``'s first axis, indexed by the
    draw's positions) summed with their multiplicities. The predictions stay fixed; only the evaluation set moves."""
    n_draws, n = draws.shape
    if conf.shape[0] != n:
        raise ValueError(f"draws over {n} slides, confusion counts of {conf.shape[0]}")
    multiplicity = np.bincount((np.arange(n_draws)[:, None] * n + draws).ravel(), minlength=n_draws * n)
    k = conf.shape[1]
    summed = multiplicity.reshape(n_draws, n) @ conf.reshape(n, k * k)
    return balanced_accuracy(summed.reshape(n_draws, k, k))


def stability_prediction(
    train: np.ndarray, test: np.ndarray, k: int, train_final: np.ndarray, *, seed: int = 0
) -> np.ndarray:
    """The guard's stability prediction (design "Checks fixed in advance", 2; ruling T7-d).

    The training half's own k-means, ``kmeans_fit(train, k, seed)``, assigns
    each test row (in the training basis) to its nearest centroid, and the
    assignment is mapped to final labels by ``match_labels(own labels,
    train_final)``, the Hungarian matching on the training half.
    """
    model = kmeans_fit(train, k, seed)
    mapping = match_labels(model.labels_, train_final)
    return np.array([mapping[label] for label in range(k)])[model.predict(_matrix(test))]


def classifier_prediction(train_F: np.ndarray, train_y: np.ndarray, test_F: np.ndarray) -> np.ndarray:
    """``HistGradientBoostingClassifier(random_state=0)``, scikit-learn's defaults otherwise, fitted on the training
    rows; its predictions for ``test_F``."""
    return HistGradientBoostingClassifier(random_state=0).fit(train_F, train_y).predict(test_F)


def guard(
    X: Mapping[str, np.ndarray],
    F: np.ndarray,
    D: Mapping[str, np.ndarray],
    final: np.ndarray,
    half: np.ndarray,
    slide: np.ndarray,
    k: int,
    *,
    n_draws: int = N_DRAWS,
    seed: int = BOOT_SEED,
) -> dict:
    """The pairwise guard of one composition stratum ("Checks fixed in advance", 2; rulings T7-c to T7-e).

    The arguments hold the stratum's centres: ``X[h]``, their level-2 scores
    in half h's basis; ``F``, the 15 pairwise, density and composition
    features; ``D[h]``, the 8 domain blocks' 72 scores in half h's basis;
    ``final``, the final subdomain labels 0..k-1; and each centre's ``half``
    and ``slide``.

    In each direction, training half t and test half u, three predictions of
    the test half's final labels:
    - stability: ``stability_prediction(X[t][t's rows], X[t][u's rows], k,
      final[t's rows])``;
    - pairwise: ``classifier_prediction`` on ``F``, trained on t's final labels;
    - domain blocks (descriptive): the same classifier on ``D[t]`` for both halves.

    Each is scored by balanced accuracy on the whole test half, and on each of
    ``n_draws`` bootstrap draws of the slides holding the test half's centres
    (``bootstrap_draws`` from a fresh ``default_rng(seed)``, one index matrix
    for the three; ``resampled_ba``). A difference's interval is its 2.5 and
    97.5 percentiles over the draws (numpy's default, linear). A direction
    passes when BA_stab - BA_pw > 0 and its lower bound > 0; the stratum
    passes when both directions do.

    Returns, per direction ("A->B", "B->A"), the sizes and test slides, BA_stab,
    BA_pw, BA_dom, both differences with their intervals and "passed"; the
    stratum's "passed"; and "draws": the bootstrap differences
    ``stab_minus_{pw,dom}_{tu}``, (n_draws,) each.
    """
    out: dict = {}
    draws_out = {}
    for t, u in DIRECTIONS:
        train, test = half == t, half == u
        predictions = {
            "stab": stability_prediction(X[t][train], X[t][test], k, final[train]),
            "pw": classifier_prediction(F[train], final[train], F[test]),
            "dom": classifier_prediction(D[t][train], final[train], D[t][test]),
        }
        slides = np.unique(slide[test])
        draws = bootstrap_draws(slides.size, n_draws=n_draws, seed=seed)
        point, boot = {}, {}
        for kind, pred in predictions.items():
            _, conf = confusion_by_slide(final[test], pred, slide[test], k)
            point[kind] = float(balanced_accuracy(conf.sum(axis=0)))
            boot[kind] = resampled_ba(conf, draws)
        entry = {
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "n_test_slides": int(slides.size),
            "test_slides": slides.tolist(),
            "BA_stab": point["stab"],
            "BA_pw": point["pw"],
            "BA_dom": point["dom"],
        }
        for kind in ("pw", "dom"):
            difference = boot["stab"] - boot[kind]
            low, high = np.percentile(difference, [2.5, 97.5])
            entry[f"stab_minus_{kind}"] = point["stab"] - point[kind]
            entry[f"stab_minus_{kind}_interval"] = [float(low), float(high)]
            draws_out[f"stab_minus_{kind}_{t}{u}"] = difference
        entry["passed"] = entry["stab_minus_pw"] > 0 and entry["stab_minus_pw_interval"][0] > 0
        out[_name(t, u)] = entry
    out["passed"] = all(out[_name(t, u)]["passed"] for t, u in DIRECTIONS)
    out["draws"] = draws_out
    return out


def _strata(strata: Sequence[int]) -> str:
    listed = ", ".join(str(s) for s in strata)
    return f"composition stratum {listed}" if len(strata) == 1 else f"composition strata {listed}"


def verdict(
    sizes: Sequence[int],
    control_k: Sequence[int],
    level2_k: Sequence[int] | None = None,
    guard_passed: Sequence[bool | None] | None = None,
) -> dict:
    """The study's verdict ("Checks fixed in advance", 2; ruling T7-b) with its reason.

    "not run: control failed" if the control found k > 1 in any stratum. Else
    "adds" if a stratum holding at least ``MIN_SHARE`` of all centres
    (compared exactly) passes the guard, and "does not add" otherwise.
    ``level2_k`` and ``guard_passed`` (None where k = 1) are per stratum.
    """
    failed = [comp for comp, k in enumerate(control_k) if k > 1]
    if failed:
        reason = f"the control found subdomains (k > 1) in {_strata(failed)}"
        return {"verdict": "not run: control failed", "reason": reason}
    with_subdomains = [comp for comp, k in enumerate(level2_k) if k >= 2]
    if not with_subdomains:
        return {"verdict": "does not add", "reason": "no compartment with subdomains"}
    passing = [comp for comp in with_subdomains if guard_passed[comp]]
    large = [comp for comp in passing if sizes[comp] >= MIN_SHARE * sum(sizes)]
    if large:
        return {"verdict": "adds", "reason": f"{_strata(large)}, holding at least 10% of all centres, passed the guard"}
    if passing:
        reason = f"only {_strata(passing)} passed the guard, holding under 10% of all centres"
        return {"verdict": "does not add", "reason": reason}
    reason = "no composition stratum with subdomains passed the guard in both directions"
    return {"verdict": "does not add", "reason": reason}


# ------------------------------------------------------------------------------------------ Task 7: the descriptives


def same_slide_neighbours(xy: np.ndarray, slide: np.ndarray, k_nn: int = K_NN) -> np.ndarray:
    """Each centre's ``k_nn`` nearest other centres of its own slide, as row indices (N, k_nn), nearest first.

    Per slide, ``cKDTree(xy).query(xy, k=k_nn + 1)`` with the centre itself
    dropped; if tied positions keep it out of its own k_nn + 1, the farthest
    of them goes instead.
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = np.empty((len(xy), k_nn), np.int64)
    for s in np.unique(slide):
        rows = np.flatnonzero(slide == s)
        if rows.size <= k_nn:
            raise ValueError(f"slide {s} has {rows.size} centres, not more than k_nn = {k_nn}")
        _, index = cKDTree(xy[rows]).query(xy[rows], k=k_nn + 1)
        drop = index == np.arange(rows.size)[:, None]
        drop[~drop.any(axis=1), -1] = True
        out[rows] = rows[index[~drop].reshape(rows.size, k_nn)]
    return out


def _ratio(observed: float, expected: float) -> float | None:
    return float(observed / expected) if expected > 0 else None


def contiguity(
    xy: np.ndarray, slide: np.ndarray, compartment: np.ndarray, subdomain: np.ndarray, k_nn: int = K_NN
) -> dict[int, dict]:
    """The subdomains' spatial contiguity ("Checks fixed in advance", 3; ruling T7-g), per stratum with subdomains.

    Each centre's neighbours are its ``k_nn`` nearest centres of the same
    slide, whatever their stratum (``same_slide_neighbours``). Observed: the
    directed neighbour pairs (i, j) with i in the stratum and j of the same
    stratum and the same subdomain. Expected, when the subdomain labels are
    reassigned uniformly at random within each (slide, stratum) group g: per
    subdomain s, the sum over g of m_g · n_gs (n_gs - 1) / (n_g (n_g - 1)),
    where m_g counts the directed neighbour pairs within g, n_g its centres
    and n_gs those of subdomain s. A group with n_g < 2 adds nothing.

    Returns {stratum: ...} for every stratum whose labels are >= 0 (k >= 2):
    its observed and expected counts and their ratio (None where nothing is
    expected), its centres' neighbour pairs and those within the stratum,
    and the same counts per subdomain with its size.
    """
    slide, compartment, subdomain = (np.asarray(v) for v in (slide, compartment, subdomain))
    neighbours = same_slide_neighbours(xy, slide, k_nn)
    i, j = np.repeat(np.arange(len(subdomain)), k_nn), neighbours.ravel()
    slides, group = np.unique(slide, return_inverse=True)  # a centre's group within its stratum: its slide's position
    n_groups = slides.size
    out = {}
    for comp in np.unique(compartment[subdomain >= 0]).tolist():
        in_c = compartment == comp
        labels = subdomain[in_c]
        if labels.min() < 0:
            raise ValueError(f"stratum {comp} mixes subdomain labels and -1")
        k = int(labels.max()) + 1
        within = in_c[i] & (compartment[j] == comp)
        m = np.bincount(group[i[within]], minlength=n_groups)
        n_g = np.bincount(group[in_c], minlength=n_groups)
        n_gs = np.bincount(group[in_c] * k + labels, minlength=n_groups * k).reshape(n_groups, k)
        pairs = n_g * (n_g - 1)
        expected = np.divide(m, pairs, out=np.zeros(n_groups), where=pairs > 0) @ (n_gs * (n_gs - 1))
        agree = within & (subdomain[i] == subdomain[j])
        observed = np.bincount(subdomain[i[agree]], minlength=k)
        out[comp] = {
            "observed": int(observed.sum()),
            "expected": float(expected.sum()),
            "ratio": _ratio(observed.sum(), expected.sum()),
            "neighbour_pairs": int(in_c.sum()) * k_nn,
            "within_stratum_pairs": int(within.sum()),
            "subdomains": [
                {
                    "subdomain": s,
                    "size": int(np.count_nonzero(labels == s)),
                    "observed": int(observed[s]),
                    "expected": float(expected[s]),
                    "ratio": _ratio(observed[s], expected[s]),
                }
                for s in range(k)
            ],
        }
    return out


# ------------------------------------------------------------------------------------------------- Task 7: the run

L1_NAMES = ("log10 cells", *c.CLASSES)
PAIR_NAMES = tuple(f"{c.CLASSES[a]}-{c.CLASSES[b]}" for a, b in c.PAIRS)
DESCRIPTION = {
    "compartment": (
        "the level-1 composition strata: k-means clusters of the centres by their balls' z-scored log10 cell count and "
        "four class fractions; bins of a continuum of compositions that recur across the halves (design step 8; "
        "Simon, 2026-11-21)"
    ),
    "subdomain": "the level-2 subdomain, an index within its composition stratum; -1 where the stratum has none",
    "control_k": (
        "the control's k per composition stratum (labelling 1 against labellings 2..19, in the observed bases); "
        "level 2 runs only if every one is 1"
    ),
    "boot_{c}_stab_minus_pw_{AB,BA}": (
        "domains.npz: the guard's 4000 bootstrap differences BA_stab - BA_pw in composition stratum c, training "
        "half first"
    ),
    "boot_{c}_stab_minus_dom_{AB,BA}": "domains.npz: the same for BA_stab - BA_dom (descriptive)",
    "gap_A, gap_B": "a half's gap curve and s at k = 1..k_max (entry k - 1), its k and largest gain Gap(k) - Gap(1)",
    "ps": "prediction strength per direction at the smaller of the halves' k; empty when that is 1",
    "ps_curve": "prediction strength per direction at k = 2..k_max (entry k - 2), descriptive",
    "ari": "the cross-half adjusted Rand index per direction where k >= 2, descriptive",
}


def _settings() -> dict:
    return {
        "rule": {
            "B": N_REFERENCES,
            "min_gain": MIN_GAIN,
            "ps_min": PS_MIN,
            "n_init": N_INIT,
            "seed": 0,
            "k_max_strata": K_MAX_STRATA,
            "k_max_subdomains": K_MAX_SUBDOMAINS,
            "min_half": MIN_HALF,
        },
        "level1_features": list(L1_NAMES),
        "zscore": (
            "ddof 0: each half's own means and SDs for its gap statistic, the training half's in each "
            "prediction-strength direction, all centres' for the final fit"
        ),
        "guard": {
            "classifier": "HistGradientBoostingClassifier(random_state=0), scikit-learn's defaults otherwise",
            "pairwise_features": [*(f"P {name}" for name in PAIR_NAMES), *L1_NAMES],
            "domain_blocks": list(DOMAIN_BLOCKS),
            "balanced_accuracy": "the mean recall over the classes present in the true labels",
            "n_draws": N_DRAWS,
            "bootstrap": (
                f"a fresh default_rng({BOOT_SEED}) per composition stratum and direction draws integers(0, n, "
                f"size=({N_DRAWS}, n)) over the n slides holding the test half's centres of the stratum, in "
                "slide-index order; the predictions stay fixed"
            ),
            "interval": [2.5, 97.5],
            "min_share": float(MIN_SHARE),
        },
        "contiguity_k_nn": K_NN,
    }


def _peak_rss_gb() -> float:
    """This process's peak resident set size so far, in GB (``ru_maxrss`` counts bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (peak if sys.platform == "darwin" else 1024 * peak) / 1e9


def _choice_line(choice: dict) -> str:
    if choice.get("too_small"):
        return f"k 1 (too small: {choice['n_A']} centres in A, {choice['n_B']} in B)"
    gains = f"{choice['gap_A']['largest_gain']:.4f} / {choice['gap_B']['largest_gain']:.4f}"
    ps = "[" + ", ".join(f"{p:.3f}" for p in choice["ps"]) + "]"
    line = f"k {choice['k']} (k_A {choice['k_A']}, k_B {choice['k_B']}; largest gains {gains}; PS {ps})"
    if choice.get("ari"):
        line += "; cross-half ARI " + ", ".join(f"{name} {a:.3f}" for name, a in choice["ari"].items())
    return line


def _guard_line(entry: dict) -> str:
    pw, dom = entry["stab_minus_pw_interval"], entry["stab_minus_dom_interval"]
    return (
        f"BA_stab {entry['BA_stab']:.3f}, BA_pw {entry['BA_pw']:.3f}, difference {entry['stab_minus_pw']:+.3f} "
        f"[{pw[0]:+.3f}, {pw[1]:+.3f}], {'pass' if entry['passed'] else 'no pass'}; BA_dom {entry['BA_dom']:.3f}, "
        f"BA_stab - BA_dom {entry['stab_minus_dom']:+.3f} [{dom[0]:+.3f}, {dom[1]:+.3f}] "
        f"({entry['n_test']} test centres on {entry['n_test_slides']} slides)"
    )


def run(features: Mapping[str, np.ndarray], log: Callable[[str], None] = print) -> tuple[dict[str, np.ndarray], dict]:
    """The composition strata, the control's gate, the subdomains, the guard, the verdict and the descriptives.

    ``features`` holds ``FEATURE_KEYS`` of ``results/features.npz``. Returns
    the arrays of ``results/domains.npz`` (all finite) and the summary of
    ``results/domains.json`` (JSON-ready).
    """
    start = time.perf_counter()
    half, slide, xy, L1 = (np.asarray(features[key]) for key in ("half", "slide_index", "xy", "L1"))
    if not np.all(np.isin(half, HALVES)):
        raise ValueError("every centre belongs to half A or half B")
    N = len(half)
    seconds = {}

    t = time.perf_counter()
    compartment, level1_choice = level1(L1, half)
    n_strata = level1_choice["k"]
    seconds["level1"] = round(time.perf_counter() - t, 1)
    log(f"level 1, composition strata: {_choice_line(level1_choice)}; {seconds['level1']} s")
    sizes = np.bincount(compartment, minlength=n_strata)
    strata = []
    for comp in range(n_strata):
        in_c = compartment == comp
        record = {"compartment": comp, "size": int(sizes[comp]), "share": float(sizes[comp] / N)}
        for h in HALVES:
            record[f"size_{h}"] = int(np.count_nonzero(in_c & (half == h)))
            record[f"slides_{h}"] = int(np.unique(slide[in_c & (half == h)]).size)
        record["mean_L1"] = dict(zip(L1_NAMES, L1[in_c].mean(axis=0).tolist(), strict=True))
        strata.append(record)
        profile = ", ".join(f"{name} {value:.3f}" for name, value in record["mean_L1"].items())
        log(
            f"composition stratum {comp}: {record['size']} centres ({record['share']:.3f}), A {record['size_A']} on "
            f"{record['slides_A']} slides, B {record['size_B']} on {record['slides_B']}; mean {profile}"
        )

    # The control first (ruling T7-a): level 2 runs only if it finds k = 1 in every stratum.
    t = time.perf_counter()
    control, _ = level2({h: features[f"L2_ctl_{h}"] for h in HALVES}, half, compartment, n_strata)
    control_k = np.array([choice["k"] for choice in control], np.int64)
    control_passed = bool(np.all(control_k == 1))
    seconds["control"] = round(time.perf_counter() - t, 1)
    for comp, choice in enumerate(control):
        log(f"composition stratum {comp}, control: {_choice_line(choice)}")
    log(f"control: {'passed' if control_passed else 'FAILED'}, k {control_k.tolist()}; {seconds['control']} s")

    arrays = {"compartment": compartment, "control_k": control_k}
    observed: list[dict | None] = [None] * n_strata
    guards: list[dict | None] = [None] * n_strata
    contiguous: dict[int, dict] = {}
    if control_passed:
        t = time.perf_counter()
        obs = {h: features[f"L2_obs_{h}"] for h in HALVES}
        observed, subdomain = level2(obs, half, compartment, n_strata, X_all=features["L2_obs_all"])
        seconds["level2"] = round(time.perf_counter() - t, 1)
        for comp, choice in enumerate(observed):
            log(f"composition stratum {comp}, subdomains: {_choice_line(choice)}")
        t = time.perf_counter()
        F = np.column_stack([features["P"], L1])
        D = {h: np.asarray(features[f"blk_obs_{h}"])[:, DOMAIN_BLOCKS].reshape(N, -1) for h in HALVES}
        for comp, choice in enumerate(observed):
            if choice["k"] < 2:
                continue
            rows = compartment == comp
            result = guard(
                {h: obs[h][rows] for h in HALVES},
                F[rows],
                {h: D[h][rows] for h in HALVES},
                subdomain[rows],
                half[rows],
                slide[rows],
                choice["k"],
            )
            for key, difference in result.pop("draws").items():
                arrays[f"boot_{comp}_{key}"] = difference
            guards[comp] = result
            for direction in DIRECTIONS:
                name = _name(*direction)
                log(f"composition stratum {comp}, guard {name}: {_guard_line(result[name])}")
        seconds["guard"] = round(time.perf_counter() - t, 1)
        contiguous = contiguity(xy, slide, compartment, subdomain)
        for comp, record in contiguous.items():
            ratios = [(s["subdomain"], s["ratio"]) for s in record["subdomains"] if s["ratio"] is not None]
            per = ", ".join(f"{s}: {ratio:.3f}" for s, ratio in ratios)
            log(
                f"composition stratum {comp}, contiguity: observed/expected {record['ratio']:.3f} "
                f"({record['observed']} / {record['expected']:.1f}); per subdomain {per}"
            )
        outcome = verdict(
            sizes.tolist(),
            control_k.tolist(),
            [choice["k"] for choice in observed],
            [None if g is None else g["passed"] for g in guards],
        )
    else:
        subdomain = np.full(N, -1, np.int64)
        log("level 2 stops: the control found subdomains, so there is no observed level-2 fit and no guard")
        outcome = verdict(sizes.tolist(), control_k.tolist())
    arrays["subdomain"] = subdomain
    for comp, record in enumerate(strata):
        record |= {"control": control[comp], "level2": observed[comp], "guard": guards[comp]}
        record["contiguity"] = contiguous.get(comp)
    seconds["total"] = round(time.perf_counter() - start, 1)
    log(f"verdict: {outcome['verdict']} ({outcome['reason']}); {seconds['total']} s")
    summary = {
        "description": DESCRIPTION,
        "settings": _settings(),
        "n_centres": N,
        "level1": level1_choice,
        "compartments": strata,
        "control_k": control_k.tolist(),
        "control_passed": control_passed,
        "verdict": outcome,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "seconds": seconds,
        "peak_rss_gb": round(_peak_rss_gb(), 3),
    }
    return arrays, summary


def _print(line: str) -> None:
    print(line, flush=True)


def main() -> None:
    c.check_topocyte()
    t = time.perf_counter()
    with np.load(c.RESULTS / "features.npz", allow_pickle=False) as z:
        features = {key: z[key] for key in FEATURE_KEYS}
    _print(f"read {len(features['half'])} centres from {c.RESULTS / 'features.npz'} in {time.perf_counter() - t:.0f} s")
    arrays, summary = run(features, log=_print)
    npz = c.RESULTS / "domains.npz"
    tmp = npz.with_name("domains.tmp.npz")  # matched by *.npz in .gitignore
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, npz)
    out = c.RESULTS / "domains.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, out)
    _print(f"wrote {npz} and {out}; peak RSS {_peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
