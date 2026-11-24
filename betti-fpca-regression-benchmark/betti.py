"""Reduced Betti curves, their exact Gram and L² matrices, and the exact fPCA projection (``design.md``, arms F, B).

- **Curves:** topocyte's reduced Betti curve of a diagram counts the diagram's finite bars [b, d) alive at each
  value. Essential bars (death ``inf``) are left out, so arms F and B do not see the fluid diagrams' essential
  births, which W2 uses.
- **The shift:** topocyte's inner product integrates over [0, ∞) only (``topocyte/ph/metrics.py``, ``_levels``): a
  bar born below 0 counts nothing below 0. Adding one shift to every endpoint leaves every full-line inner product
  unchanged, so a shift that lifts every value to 0 or above makes topocyte's inner products the full-line ones. The
  fluid curves are shifted by ``FLUID_SHIFT`` (gate G4). Rips values are never negative and are not shifted.
- **Matrices:** the Gram matrix of exact inner products (arm F) and the exact L² distances (arm B). Each distance
  comes from the difference of its two step functions, without the cancellation of ⟨f, f⟩ + ⟨g, g⟩ − 2⟨f, g⟩.
- **fPCA:** ``fpca_from_gram`` centres the training block K of the Gram matrix as K_c = H K H, with
  H = I − 11ᵀ/n, and its kept eigenpairs (ρ, V) give the training scores V √ρ. A test curve f with the inner products
  g = (⟨f, f_j⟩)ⱼ with the training curves has the centred row g − mean(g) − mean(K, axis=0) + mean(K), the inner
  products of f − f̄ with every f_j − f̄, f̄ the mean training curve. Its exact scores are that row times V / √ρ: the
  projections of f − f̄ onto the component functions. A training curve gets its own score back, since
  K_c V / √ρ = V √ρ. ``FPCA.transform`` (gate G3) leaves out the row's two constant terms, which vanish when V's
  columns are orthogonal to 1. The computed eigenvectors of the smallest eigenvalues are not orthogonal to 1 to
  working precision, so in those components the two differ by more than rounding. There this projection still
  gives the training curves their fitted scores back, and ``transform`` does not: on the fluid cells of G3 it missed
  them by up to 8e-6 of the largest score.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import topocyte as tc


def curves(diagrams: Sequence[np.ndarray], dim: int, shift: float = 0.0) -> list[tc.ph.BettiCurve]:
    """The reduced Betti curves of ``diagrams``, all of dimension ``dim``: each diagram's finite bars, shifted.

    Each diagram is an (n, 2) array of (birth, death) rows. Its essential bars (death ``inf``) are dropped, and
    ``shift`` is added to every endpoint of the others. ``ValueError`` from ``tc.ph.BettiCurve`` for a bar it refuses,
    such as a NaN death.
    """
    out = []
    for a in diagrams:
        a = np.asarray(a, dtype=float)
        out.append(tc.ph.BettiCurve(a[a[:, 1] != np.inf] + shift, dim=dim))
    return out


def gram(curves: Sequence[tc.ph.BettiCurve], dim: int) -> np.ndarray:
    """The exact Gram matrix of ``curves``, of dimension ``dim``: topocyte's inner products over [0, ∞).

    Computed with ``n_jobs=-1``, one thread per CPU.
    """
    return tc.ph.betti_inner_product_matrix(curves, dim=dim, n_jobs=-1)


def l2_matrix(curves: Sequence[tc.ph.BettiCurve], dim: int) -> np.ndarray:
    """The exact L² distance between every two of ``curves``: a symmetric matrix with a zero diagonal.

    Each entry above the diagonal is ``tc.ph.betti_l2_distance`` of its two curves, mirrored below it. The pairs run
    serially: on the 400 fluid curves at t = 5 that took 9.0 s in H0 and 2.8 s in H1, where 10 threads took 11.3 s
    and 6.8 s.
    """
    D = np.zeros((len(curves), len(curves)))
    for p, q in itertools.combinations(range(len(curves)), 2):
        D[p, q] = tc.ph.betti_l2_distance(curves[p], curves[q], dim=dim)
    return D + D.T


def fpca_scores(G: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fPCA scores of the curves ``train`` and ``test`` on an fPCA of the ``train`` curves, from their Gram ``G``.

    ``G`` is the Gram matrix of a set of curves, and ``train`` and ``test`` index them. The fPCA is topocyte's
    ``fpca_from_gram`` of the training block, raw curves (normalization ``"none"``), with its other defaults; the
    test scores are the exact projection of the module docstring. Returns the training scores, (len(train), rank),
    and the test scores, (len(test), rank). If no training curve has a finite bar, the rank is 0 and the scores have
    no columns.
    """
    K, g = G[np.ix_(train, train)], G[np.ix_(test, train)]
    r = tc.ph.fpca_from_gram(K, normalization="none")
    S_test = (g - g.mean(axis=1, keepdims=True) - K.mean(axis=0) + K.mean()) @ (
        r.eigenvectors / np.sqrt(r.gram_eigenvalues)
    )
    return r.scores, S_test
