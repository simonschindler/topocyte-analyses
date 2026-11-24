"""Independent oracles of the Betti FPCR benchmark, for its tests and its verifier.

Each oracle computes, by a route of its own, a quantity that a module of the study computes another way.
``test_study.py`` and ``verify_results.py`` check the study against them; no stage uses them.
"""

from __future__ import annotations

import math

import numpy as np


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """The full-line L² inner product of the reduced Betti curves of the diagrams ``a`` and ``b``, (n, 2) each.

    A reduced Betti curve is the sum of the indicators of its diagram's finite bars [b, d), so the inner product of
    two of them is Σᵢⱼ max(0, min(dᵢ, d'ⱼ) − max(bᵢ, b'ⱼ)), the overlaps of every finite bar (bᵢ, dᵢ) of ``a`` with
    every finite bar (b'ⱼ, d'ⱼ) of ``b``, summed with ``math.fsum``. Essential bars (death ``inf``) are left out. It
    integrates over the whole line, where topocyte's inner product integrates over [0, ∞) only.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a[:, 1])], b[np.isfinite(b[:, 1])]
    lengths = np.minimum.outer(a[:, 1], b[:, 1]) - np.maximum.outer(a[:, 0], b[:, 0])
    return math.fsum(np.maximum(lengths, 0.0).ravel())


# _ker, _kernel_estimator_given_h and upstream_nw transcribe upstream/kernel_regression.py (read as text, never run)
# statement for statement, with upstream's names. They are kept apart from estimators.py on purpose: the oracle must
# not share code with the estimator it checks.
def _ker(z):
    """Upstream's ``ker``: the standard normal density, ``(1/√(2π)) e^(−z²/2)``."""
    return (1 / (np.sqrt(2 * np.pi))) * np.e ** (-0.5 * (z**2))


def _kernel_estimator_given_h(distances, y_vals, h):
    """Upstream's ``kernel_estimator_given_h``: the weighted mean of ``y_vals``, their mean if every weight is 0."""
    distances = np.asarray(distances)
    y_vals = np.asarray(y_vals)
    weights = _ker(distances / h)

    if np.sum(weights) == 0:
        return np.mean(y_vals)

    return np.sum(weights * y_vals) / np.sum(weights)


def upstream_nw(D_train: np.ndarray, y: np.ndarray, D_test: np.ndarray) -> tuple[np.ndarray, float]:
    """Upstream's ``kernel_estimator`` on precomputed distances, loop for loop: the test predictions and the bandwidth.

    Upstream computes the distances from the point clouds; here they are given: ``D_train`` between the n training
    samples (symmetric, zero diagonal) with targets ``y``, and ``D_test`` from each test sample to them (m × n). For
    each of 50 bandwidths ``h`` in ``linspace(0.1 m, 2 m, 50)``, m the median of the positive entries of ``D_train``,
    each training sample in turn is left out with ``np.delete`` and estimated from the others; the bandwidth is the
    first minimizer (``np.argmin``) of the mean squared error. Each test sample is then estimated from all training
    samples. A sample whose weights all underflow to 0 gets the plain mean of the targets it is estimated from.
    ``estimators.nw_fit_predict`` follows upstream's vectorised ``fluid/nw_regression.py`` instead.
    """
    bandwidth_range = np.linspace(0.1 * np.median(D_train[D_train > 0]), 2.0 * np.median(D_train[D_train > 0]), 50)

    avg_mses = []
    for h in bandwidth_range:
        mses = []
        for left_out_index in range(len(D_train)):
            distances = np.delete(D_train[left_out_index], left_out_index)
            y_vals = np.delete(y, left_out_index)
            left_out_y_val = y[left_out_index]
            estimate = _kernel_estimator_given_h(distances, y_vals, h)
            mse = (left_out_y_val - estimate) ** 2
            mses.append(mse)
        avg_mses.append(np.mean(mses))

    h_optimal = bandwidth_range[np.argmin(avg_mses)]

    final_estimates = [_kernel_estimator_given_h(final_distances, y, h_optimal) for final_distances in D_test]
    return np.array(final_estimates), float(h_optimal)
