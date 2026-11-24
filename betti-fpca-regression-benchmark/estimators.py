"""The arms' estimators, on precomputed matrices (``design.md``, arms K, F and B).

- **Nadaraya–Watson** (arms K and B): ``nw_fit_predict`` is upstream's ``PD_kernel`` estimator on a distance matrix,
  as upstream's ``fluid/nw_regression.py`` vectorises ``kernel_regression.kernel_estimator`` (both read as text,
  never run). The kernel is the standard normal density, and the bandwidth is chosen by leave-one-out (LOO) over 50
  values in [0.1, 2] × the median positive training distance. Its operations are upstream's, in upstream's order, so
  that arm K rounds as upstream's results did: gate G1 compares them to 1e-9.
- **FPCR** (arm F): ``fpcr_fit_predict`` fits topocyte's fPCA of each dimension on the training curves
  (``betti.fpca_scores``), chooses how many H0 and H1 scores to use, (K0, K1), by the closed-form LOO error of
  ``loo_select``, and predicts the test curves by OLS on those scores.

Every choice, the bandwidth and (K0, K1), uses the training data only.
"""

from __future__ import annotations

import itertools

import numpy as np

import betti
from common import K_MAX, LEVERAGE_CAP, N_BANDWIDTHS

SIGMA_RTOL = 1e-12  # FPCR: a (K0, K1) whose design has σ_min ≤ SIGMA_RTOL σ_max is not eligible


def _ker(z: np.ndarray) -> np.ndarray:
    """Upstream's kernel ``ker``, operation for operation: ``(1/√(2π)) e^(−z²/2)``."""
    return (1 / (np.sqrt(2 * np.pi))) * np.e ** (-0.5 * (z**2))


def nw_fit_predict(D_train: np.ndarray, y: np.ndarray, D_test: np.ndarray) -> tuple[np.ndarray, float]:
    """Nadaraya–Watson regression on precomputed distances (arms K and B): the test predictions and the bandwidth h.

    ``D_train`` holds the distances between the n training samples (symmetric, zero diagonal), ``y`` their targets,
    and ``D_test`` the distances from each test sample to them (test by train). As upstream's ``nw_predict``:

    - the bandwidths are ``linspace(0.1 m, 2 m, 50)``, m the median of the positive entries of ``D_train``, so that
      identical training samples (distance 0) do not count;
    - the weights are ``ker(D / h)``, the standard normal density of D / h; distance 0 has the kernel's maximum;
    - each bandwidth's LOO error is the mean squared error of every training target's weighted mean of the others
      (the diagonal weights set to 0). A sample whose weights all underflow to 0 gets the mean of the others,
      ``(y.sum() − yᵢ) / (n − 1)``;
    - h is the first bandwidth of least LOO error (``np.argmin``);
    - a test sample gets the weighted mean of ``y`` at h, or ``y.mean()`` if its weights all underflow to 0.
    """
    n = len(y)
    m = np.median(D_train[D_train > 0])
    hs = np.linspace(0.1 * m, 2.0 * m, N_BANDWIDTHS)
    mse = np.empty(N_BANDWIDTHS)
    for k, h in enumerate(hs):
        W = _ker(D_train / h)
        np.fill_diagonal(W, 0.0)
        s = W.sum(1)
        loo = np.where(s > 0, (W @ y) / np.where(s > 0, s, 1), (y.sum() - y) / (n - 1))
        mse[k] = np.mean((y - loo) ** 2)
    h = hs[np.argmin(mse)]
    W = _ker(D_test / h)
    s = W.sum(1)
    return np.where(s > 0, (W @ y) / np.where(s > 0, s, 1), y.mean()), float(h)


def _design(S0: np.ndarray, S1: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """FPCR's design matrix [1, S0[:, :k0], S1[:, :k1]]: an intercept, the first k0 H0 and the first k1 H1 scores."""
    return np.column_stack((np.ones(len(S0)), S0[:, :k0], S1[:, :k1]))


def _loo_mse(Z: np.ndarray, y: np.ndarray) -> float:
    """The closed-form LOO mean squared error of the OLS of ``y`` on the columns of ``Z``; ``inf`` if not eligible.

    With the thin SVD Z = U Σ Wᵀ, the leverages are hᵢ = Σⱼ Uᵢⱼ² and the residuals e = y − U Uᵀ y, and the LOO error
    is mean((e / (1 − h))²), which equals refitting without each sample in turn. ``Z`` is not eligible if
    σ_min ≤ ``SIGMA_RTOL`` σ_max or if a leverage is at least ``LEVERAGE_CAP``.

    ``y`` enters less its median. Z's first column is the constant 1, so this leaves e unchanged, and a constant
    ``y`` gets residuals of exactly 0 in every design, which then tie. Unshifted, rounding residuals of order
    ε |y| would decide among them.
    """
    U, sigma, _ = np.linalg.svd(Z, full_matrices=False)
    if sigma[-1] <= SIGMA_RTOL * sigma[0]:
        return np.inf
    h = (U**2).sum(axis=1)
    if h.max() >= LEVERAGE_CAP:
        return np.inf
    y = y - np.median(y)
    e = y - U @ (U.T @ y)
    return float(np.mean((e / (1 - h)) ** 2))


def loo_select(S0: np.ndarray, S1: np.ndarray, y: np.ndarray, k_max: int = K_MAX) -> tuple[int, int, float]:
    """FPCR's numbers of H0 and H1 scores, (K0, K1), and their LOO mean squared error, from the training set alone.

    ``S0`` and ``S1`` are the training scores of each dimension, n rows each, and ``y`` the training targets. The
    candidates are every (K0, K1) with 0 ≤ K_d ≤ min(``k_max``, S_d.shape[1]) and K0 + K1 ≤ n − 2, in the order
    (K0 + K1, K0). A candidate's design is [1, S0[:, :K0], S1[:, :K1]], and its LOO error the closed form of
    ``_loo_mse``; an ineligible candidate counts as ``inf``. The first candidate of least error wins, so ties go to
    the smaller K0 + K1, then the smaller K0. (0, 0), the training mean, is always eligible. A dimension without
    components (``S_d.shape[1] == 0``) offers only K_d = 0.
    """
    pairs = itertools.product(range(min(k_max, S0.shape[1]) + 1), range(min(k_max, S1.shape[1]) + 1))
    candidates = sorted((k for k in pairs if sum(k) <= len(y) - 2), key=lambda k: (sum(k), k[0]))
    mse = [_loo_mse(_design(S0, S1, k0, k1), y) for k0, k1 in candidates]
    best = int(np.argmin(mse))
    return *candidates[best], mse[best]


def fpcr_fit_predict(
    G0: np.ndarray, G1: np.ndarray, y_train: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Arm F on one training set: the test predictions and the chosen (K0, K1).

    ``G0`` and ``G1`` are the Gram matrices of the H0 and H1 curves of a set of samples, which ``train`` and ``test``
    index; ``y_train`` holds the targets of ``train``, in its order. ``betti.fpca_scores`` fits the fPCA of each
    dimension on the training curves and projects the test curves onto it; ``loo_select`` chooses (K0, K1) on the
    training scores; the OLS of ``y_train`` on [1, the first K0 H0 scores, the first K1 H1 scores], by
    ``np.linalg.lstsq``, predicts the test rows.
    """
    (S0, T0), (S1, T1) = betti.fpca_scores(G0, train, test), betti.fpca_scores(G1, train, test)
    k0, k1, _ = loo_select(S0, S1, y_train)
    beta = np.linalg.lstsq(_design(S0, S1, k0, k1), y_train, rcond=None)[0]
    return _design(T0, T1, k0, k1) @ beta, k0, k1
