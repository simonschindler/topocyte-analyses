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
