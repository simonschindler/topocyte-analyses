"""The rule for k of the chromatic-domains pilot (design step 10), as pure functions.

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
separate from the study's ``SeedSequence`` keys. Task 7 adds the main.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans

N_INIT = 20  # k-means starts per fit
N_REFERENCES = 20  # B, reference draws of the gap statistic
MIN_GAIN = 0.03  # the smallest gain Gap(k) - Gap(1) that counts (Simon, 2026-11-21; first 0.02)
PS_MIN = 0.8  # the prediction strength needed in both directions


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
