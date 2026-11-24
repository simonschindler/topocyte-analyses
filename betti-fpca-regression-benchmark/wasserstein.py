"""The exact 2-Wasserstein distance between persistence diagrams: arm K's distance (``design.md``, "Arm K").

Upstream computed W2 with gudhi's ``wasserstein_distance(X, Y, order=2, internal_p=2)``, essential bars kept
(``gudhi/wasserstein/wasserstein.py``, read as text), which solves a transport problem with POT's ``emd2``. POT is
not installed here, so ``w2`` solves the same problem as an assignment problem, exactly, with
``scipy.optimize.linear_sum_assignment``:

- **finite bars:** n in one diagram, m in the other, give the (n + m) × (n + m) cost matrix C. Its rows are the n
  bars, then m diagonal slots; its columns are the m bars, then n diagonal slots. ``C[:n, :m]`` holds the squared
  Euclidean distances between the bars' (birth, death) points; ``C[:n, m:]`` holds row i's squared distance to the
  diagonal, (d_i − b_i)² / 2, in every column; ``C[n:, :m]`` holds column j's in every row; ``C[n:, m:]`` is 0.
  gudhi's (n + 1) × (m + 1) transport problem merges the m diagonal rows into one of mass m and the n diagonal
  columns into one of mass n, which leaves the optimal cost unchanged. The finite cost is the sum of C over its
  optimal assignment;
- **essential bars** (death ``inf``) are matched only among themselves, by sorted birth, at the squared differences
  of the births. Unequal counts give ``inf``. Births must be finite, so gudhi's other kinds of infinite point cannot
  occur;
- **W2** = √(finite cost + essential cost).

On 40 random fluid pairs at t = 5 this rule agreed with upstream's cached distances to 1.1e-15 relative
(``design.md``); gate G2 checks it again in the run.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

# What w2 needs of a diagram: its finite bars (k, 2), their squared distances to the diagonal (k,), and its essential
# births, sorted.
_Parts = tuple[np.ndarray, np.ndarray, np.ndarray]

_SHARED: list[_Parts] = []  # in a worker process of w2_matrix: the parts of all its diagrams


def _parts(a: np.ndarray) -> _Parts:
    """The parts of the diagram ``a`` that ``w2`` needs.

    ``ValueError`` unless ``a`` is an (n, 2) array of (birth, death) rows whose births are finite and whose deaths are
    finite or ``inf``.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"a diagram is an (n, 2) array of (birth, death) rows, not an array of shape {a.shape}")
    births, deaths = a[:, 0], a[:, 1]
    essential = deaths == np.inf
    if not (np.isfinite(births).all() and (np.isfinite(deaths) | essential).all()):
        raise ValueError("a diagram's births must be finite, and its deaths finite or inf")
    finite = a[~essential]
    return finite, (finite[:, 1] - finite[:, 0]) ** 2 / 2, np.sort(births[essential])


def _w2(x: _Parts, y: _Parts) -> float:
    """W2 between two diagrams given by their parts (the module docstring's rule)."""
    (a, diagonal_a, essential_a), (b, diagonal_b, essential_b) = x, y
    if len(essential_a) != len(essential_b):
        return math.inf
    n, m = len(a), len(b)
    C = np.zeros((n + m, m + n))
    C[:n, :m] = (a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2
    C[:n, m:] = diagonal_a[:, None]
    C[n:, :m] = diagonal_b
    rows, cols = linear_sum_assignment(C)
    return math.sqrt(C[rows, cols].sum() + ((essential_a - essential_b) ** 2).sum())


def w2(a: np.ndarray, b: np.ndarray) -> float:
    """The 2-Wasserstein distance between the diagrams ``a`` and ``b`` of one dimension, (n, 2), deaths maybe ``inf``.

    gudhi's ``wasserstein_distance(a, b, order=2, internal_p=2)`` with the essential parts kept: ``inf`` if the two
    numbers of essential bars differ. ``ValueError`` on a non-finite birth and on a death neither finite nor ``inf``.
    """
    return _w2(_parts(a), _parts(b))


def _row(parts: Sequence[_Parts], p: int) -> np.ndarray:
    """Row p of the W2 matrix right of its diagonal: W2 between diagram p and each later one."""
    return np.array([_w2(parts[p], parts[q]) for q in range(p + 1, len(parts))])


def _share(parts: list[_Parts]) -> None:
    """A worker's initializer: keep every diagram's parts, so that a task sends only its row number."""
    _SHARED[:] = parts


def _shared_row(p: int) -> np.ndarray:
    """``_row`` in a worker process, on the parts that ``_share`` kept."""
    return _row(_SHARED, p)


def w2_matrix(diagrams: Sequence[np.ndarray], n_jobs: int = 1) -> np.ndarray:
    """W2 between every two of ``diagrams``, all of one dimension: a symmetric matrix with a zero diagonal.

    Each entry above the diagonal is ``w2`` of its two diagrams, mirrored below it. ``n_jobs`` processes compute the
    rows, -1 one per CPU; every entry comes from the same computation in any process, so the matrix does not depend
    on ``n_jobs``. The processes may import the calling script again (they do on macOS), so a script that passes
    ``n_jobs`` other than 1 must keep its top level under ``if __name__ == "__main__":``. ``ValueError`` as ``w2``,
    and on an ``n_jobs`` below 1 other than -1.
    """
    parts = [_parts(a) for a in diagrams]
    workers = (os.process_cpu_count() or 1) if n_jobs == -1 else n_jobs
    if workers < 1:
        raise ValueError(f"n_jobs is a number of processes, at least 1, or -1 for one per CPU; not {n_jobs}")
    if workers == 1:
        rows = [_row(parts, p) for p in range(len(parts))]
    else:
        with ProcessPoolExecutor(workers, initializer=_share, initargs=(parts,)) as pool:
            rows = list(pool.map(_shared_row, range(len(parts))))
    D = np.zeros((len(parts), len(parts)))
    for p, row in enumerate(rows):
        D[p, p + 1 :] = row
    return D + D.T
