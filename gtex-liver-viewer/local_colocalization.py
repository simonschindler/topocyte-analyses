"""Single-cell hypergeometric count excess of each cell's k nearest neighbours (``LocalColocalizationExtractor``).

For a focal cell i of type c_i with neighbours N(i), the count o_{i,A} of neighbours of type A is compared with
what random labelling of the cloud's other N - 1 cells gives, holding the cell's position, label and degree
fixed (the conditional random labelling of spicyR v2): o_{i,A} is hypergeometric with population N - 1,
n'_{i,A} = n_A - [c_i = A] successes and k_i draws, so

    q = n' / (N - 1),   e = k q,   v = k q (1 - q) (N - 1 - k) / (N - 2)   (v = 0 for N <= 2).

The excess is divided by the room R it has: k (1 - q) on the gain scale, k q on the loss scale, chosen per
(focal type, neighbour type) pair by the sign of the summed excess over the training clouds and never cell by cell
or cloud by cloud. Y = (o - e) / R is left unclipped, so that its null mean stays 0, and s^2 = v / R^2. An
empirical Bayes step shrinks each cell's Y toward its cloud's pooled excess over room, by a DerSimonian-Laird
between-cell variance, so sparse estimates collapse to the prior.

Adapted for the liver cell viewer from a radius-graph specification: the neighbourhood of a cell is its k nearest
other cells (Simon, 2026-11-24), so k_i = k in every cloud of more than k cells and N - 1 otherwise, where v = 0
makes every cell invalid. Cells at one position stay each other's neighbours: a cell is left out of its own
neighbourhood by index, never by distance. Only numpy and scipy, no loop over cells.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

Cloud = tuple[np.ndarray, Sequence[Any]]
ScaleMode = Literal["pair", "target", "gain", "loss"]
ShrinkTarget = Literal["image_mean", "zero"]

TINY = 1e-12  # v and R at or below this are degenerate: the cell-type pair has no valid estimate
WEIGHT_FLOOR = 1e-8  # the smallest s^2 a weight 1 / s^2 is taken at
MIN_PAIR_CELLS = 5  # tau^2 of a (focal type, neighbour type) group needs this many valid cells
MIN_ROOM = 1.0  # a cloud's pooled prior needs this much room, else it is 0
CHUNK = 100_000  # cells per k-d tree query, to bound the memory of the neighbour table


@dataclass
class CellColocalizationResult:
    """One cloud's cell-level excess for its N cells and the K modelled types."""

    Y_shrunk: np.ndarray  # (N, K), no NaN
    Y_raw: np.ndarray  # (N, K), NaN where not valid
    s2: np.ndarray  # (N, K), inf where not valid
    valid_mask: np.ndarray  # (N, K) bool
    degree: np.ndarray  # (N,) int32, the realised k_i
    homotypic_excess_shrunk: np.ndarray  # (N,), Y_shrunk at the cell's own type; NaN for a background cell


def neighbour_indices(xy: np.ndarray, k: int) -> np.ndarray:
    """(N, min(k, N - 1)): each cell's nearest other cells, nearest first, the cell itself left out by index."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = min(int(k), n - 1)
    if k <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    tree = cKDTree(xy)
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, CHUNK):  # chunks of cells, not single cells: each query is vectorised
        stop = min(start + CHUNK, n)
        idx = np.asarray(tree.query(xy[start:stop], k=k + 1, workers=-1)[1]).reshape(stop - start, k + 1)
        others = idx != np.arange(start, stop)[:, None]
        others[others.all(axis=1), -1] = False  # k + 1 others at one distance: the cell itself fell out
        out[start:stop] = idx[others].reshape(stop - start, k)
    return out


def _null_mask(labels: np.ndarray) -> np.ndarray:
    if labels.dtype.kind == "f":
        return np.isnan(labels)
    if labels.dtype.kind == "O":
        is_null = np.frompyfunc(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), 1, 1)
        return is_null(labels).astype(bool)
    return np.zeros(len(labels), dtype=bool)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def encode_labels(cell_types: Sequence[Any], vocabulary: Sequence[Any]) -> np.ndarray:
    """Each cell's index in ``vocabulary`` (int64), -1 for a label outside it and for None or NaN (background)."""
    labels = np.asarray(cell_types)
    codes = np.full(len(labels), -1, dtype=np.int64)
    keep = ~_null_mask(labels)
    if keep.any():
        lookup = {_scalar(v): j for j, v in enumerate(vocabulary)}
        uniques, inverse = np.unique(labels[keep], return_inverse=True)
        mapped = np.array([lookup.get(_scalar(u), -1) for u in uniques], dtype=np.int64)  # one per label, not cell
        codes[keep] = mapped[inverse]
    return codes


def infer_vocabulary(clouds: Iterable[Cloud]) -> list[Any]:
    """The sorted distinct labels of the clouds, without None, NaN or negative integers (background)."""
    seen: set[Any] = set()
    for _, cell_types in clouds:
        labels = np.asarray(cell_types)
        labels = labels[~_null_mask(labels)]
        if labels.dtype.kind in "iu":
            labels = labels[labels >= 0]
        seen.update(_scalar(u) for u in np.unique(labels))
    return sorted(seen)


def _one_hot(codes: np.ndarray, width: int) -> sparse.csr_matrix:
    """(N, width) indicator of the codes; a negative code has an empty row."""
    rows = np.flatnonzero(codes >= 0)
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, codes[rows])), shape=(len(codes), width))


@dataclass
class _Counts:
    observed: np.ndarray  # (N, K) o
    expected: np.ndarray  # (N, K) e
    variance: np.ndarray  # (N, K) v
    q: np.ndarray  # (N, K)
    degree: np.ndarray  # (N,) int32


def _counts(xy: np.ndarray, codes: np.ndarray, n_types: int, k: int) -> _Counts:
    """o, e, v and q of one cloud: G H with G the (N, N) k-nearest-neighbour graph and H the (N, K) one-hot."""
    n = len(codes)
    nbrs = neighbour_indices(xy, k) if n else np.zeros((0, 0), dtype=np.int64)
    k_eff = nbrs.shape[1]
    degree = np.full(n, k_eff, dtype=np.int32)
    graph = sparse.csr_matrix(  # row i holds its k_eff neighbours, so the row pointers step by k_eff
        (np.ones(n * k_eff), nbrs.ravel(), np.arange(n + 1, dtype=np.int64) * k_eff), shape=(n, n)
    )
    one_hot = _one_hot(codes, n_types)
    observed = np.asarray((graph @ one_hot).todense(), dtype=np.float64)
    totals = np.bincount(codes[codes >= 0], minlength=n_types).astype(np.float64)
    available = totals[None, :] - np.asarray(one_hot.todense())  # n'_{i,A}: the focal cell leaves its own type
    q = available / (n - 1) if n >= 2 else np.zeros((n, n_types))
    expected = degree[:, None] * q
    if n >= 3:
        variance = np.maximum(expected * (1 - q) * (n - 1 - degree[:, None]) / (n - 2), 0.0)
    else:
        variance = np.zeros((n, n_types))
    return _Counts(observed, expected, variance, q, degree)


def excess_sums(xy: np.ndarray, codes: np.ndarray, n_types: int, ks: Sequence[int],
                nbrs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For every k in ``ks``: Σ (o - e) over the cells of each focal type B for each neighbour type A, shape
    (len(ks), K, K), and over every cell, background included, shape (len(ks), K): what ``fit`` sums, from one
    query of max(ks) neighbours (the first k of a cell's nearest neighbours are its k nearest). ``nbrs``: that
    query's ``neighbour_indices`` when the caller has it already."""
    codes = np.asarray(codes, dtype=np.int64)
    n = len(codes)
    pair = np.zeros((len(ks), n_types, n_types))
    target = np.zeros((len(ks), n_types))
    if n < 2:
        return pair, target
    groups = np.where(codes >= 0, codes, n_types)  # the focal cell's type, background as type K
    if nbrs is None:
        nbrs = neighbour_indices(xy, max(ks))
    if nbrs.shape[1] < min(max(ks), n - 1):
        raise ValueError(f"{nbrs.shape[1]} neighbours per cell, fewer than the {max(ks)} asked for")
    cumulative = np.zeros((nbrs.shape[1] + 1, n_types + 1, n_types))
    for rank in range(nbrs.shape[1]):  # over neighbour ranks, each vectorised over every cell
        neighbour = codes[nbrs[:, rank]]
        ok = neighbour >= 0
        counts = np.bincount(groups[ok] * n_types + neighbour[ok], minlength=(n_types + 1) * n_types)
        cumulative[rank + 1] = cumulative[rank] + counts.reshape(n_types + 1, n_types)
    totals = np.bincount(codes[codes >= 0], minlength=n_types).astype(np.float64)
    focal = np.bincount(groups, minlength=n_types + 1).astype(np.float64)
    available = np.vstack([totals[None, :] - np.eye(n_types), totals[None, :]])  # (K + 1, K): n'_{B,A}
    for j, k in enumerate(ks):
        k_eff = min(int(k), n - 1)
        excess = cumulative[k_eff] - k_eff * focal[:, None] * available / (n - 1)
        pair[j] = excess[:n_types]
        target[j] = excess.sum(axis=0)
    return pair, target


def dersimonian_laird(y: np.ndarray, s2: np.ndarray, valid: np.ndarray, groups: np.ndarray, n_groups: int,
                      min_cells: int) -> np.ndarray:
    """(n_groups, K): the between-cell variance tau^2 >= 0 of each group's valid Y, by DerSimonian-Laird moments
    with weights 1 / max(s^2, WEIGHT_FLOOR); 0 where the group has fewer than ``min_cells`` (at least 2) valid cells
    or C <= TINY."""
    member = _one_hot(groups, n_groups).T.tocsr()  # (G, N)
    w = np.where(valid, 1.0 / np.maximum(np.where(valid, s2, 1.0), WEIGHT_FLOOR), 0.0)
    y0 = np.where(valid, y, 0.0)
    m = member @ valid.astype(np.float64)
    sum_w = member @ w
    mean = np.divide(member @ (w * y0), sum_w, out=np.zeros_like(sum_w), where=sum_w > 0)
    q_stat = member @ (w * (y0 - mean[groups]) ** 2)
    c = sum_w - np.divide(member @ (w * w), sum_w, out=np.zeros_like(sum_w), where=sum_w > 0)
    ok = (c > TINY) & (m >= max(min_cells, 2))
    tau2 = np.divide(q_stat - (m - 1), c, out=np.zeros_like(c), where=ok)
    return np.maximum(tau2, 0.0)


def shrink(y: np.ndarray, s2: np.ndarray, valid: np.ndarray, prior: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """lambda Y + (1 - lambda) prior with lambda = tau^2 / (tau^2 + s^2) where valid, the prior elsewhere."""
    lam = np.divide(tau2, tau2 + s2, out=np.zeros(np.broadcast_shapes(np.shape(tau2), np.shape(s2))),
                    where=valid & (tau2 + s2 > 0))
    return np.where(valid, lam * np.where(valid, y, 0.0) + (1 - lam) * prior, prior)


class LocalColocalizationExtractor:
    """Hypergeometric count excess of every cell's k nearest neighbours, per neighbour type, on a gain/loss scale
    fitted on training clouds, with empirical Bayes shrinkage within each cloud.

    ``fit`` freezes ``vocabulary_``, the summed excess (``pair_excess_``, ``target_excess_``) and from them
    ``gain_mask_`` ((K + 1, K): one row per focal type, the last for background cells); ``transform`` only reads them.
    """

    def __init__(self, n_neighbors: int = 15, scale_mode: ScaleMode = "pair", shrink_target: ShrinkTarget = "image_mean",
                 tau2: float | Literal["auto"] = "auto", vocabulary: Sequence[Any] | None = None,
                 dtype: np.dtype | type = np.float64) -> None:
        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be at least 1")
        if scale_mode not in ("pair", "target", "gain", "loss"):
            raise ValueError(f"scale_mode must be pair, target, gain or loss, not {scale_mode!r}")
        if shrink_target not in ("image_mean", "zero"):
            raise ValueError(f"shrink_target must be image_mean or zero, not {shrink_target!r}")
        if tau2 != "auto" and not (isinstance(tau2, int | float) and tau2 >= 0):
            raise ValueError("tau2 must be 'auto' or a number >= 0")
        self.n_neighbors = int(n_neighbors)
        self.scale_mode = scale_mode
        self.shrink_target = shrink_target
        self.tau2 = tau2
        self.vocabulary = None if vocabulary is None else list(vocabulary)
        self.dtype = np.dtype(dtype)

    # -- fitting ---------------------------------------------------------------------------------------------

    def fit(self, clouds: Sequence[Cloud]) -> Self:
        """Sum (o - e) over the training clouds' cells by focal and neighbour type, and fix the gain/loss scale."""
        clouds = list(clouds)
        vocabulary = self.vocabulary if self.vocabulary is not None else infer_vocabulary(clouds)
        n_types = len(vocabulary)
        pair = np.zeros((n_types, n_types))
        target = np.zeros(n_types)
        for xy, cell_types in clouds:
            codes = encode_labels(cell_types, vocabulary)
            p, t = excess_sums(np.asarray(xy, dtype=np.float64).reshape(-1, 2), codes, n_types, [self.n_neighbors])
            pair += p[0]
            target += t[0]
        return self._set_fit(vocabulary, pair, target)

    @classmethod
    def fitted(cls, vocabulary: Sequence[Any], pair_excess: np.ndarray, target_excess: np.ndarray, **kwargs: Any
               ) -> LocalColocalizationExtractor:
        """An extractor fitted elsewhere, from its summed excess (``excess_sums`` over the training clouds)."""
        extractor = cls(vocabulary=vocabulary, **kwargs)
        return extractor._set_fit(list(vocabulary), np.asarray(pair_excess, dtype=np.float64),
                                  np.asarray(target_excess, dtype=np.float64))

    def _set_fit(self, vocabulary: list[Any], pair: np.ndarray, target: np.ndarray) -> Self:
        n_types = len(vocabulary)
        if pair.shape != (n_types, n_types) or target.shape != (n_types,):
            raise ValueError(f"summed excess of shapes {pair.shape} and {target.shape} for {n_types} types")
        self.vocabulary_ = vocabulary
        self.pair_excess_ = pair
        self.target_excess_ = target
        by_target = np.broadcast_to(target >= 0, (n_types + 1, n_types))
        mask = {
            "pair": np.vstack([pair >= 0, (target >= 0)[None, :]]),  # background focal cells: the target's sign
            "target": by_target,
            "gain": np.ones((n_types + 1, n_types), dtype=bool),
            "loss": np.zeros((n_types + 1, n_types), dtype=bool),
        }[self.scale_mode]
        self.gain_mask_ = np.array(mask, dtype=bool)
        return self

    # -- transforming ----------------------------------------------------------------------------------------

    def transform(self, clouds: Sequence[Cloud]) -> list[CellColocalizationResult]:
        """Each cloud's cell-level excess on the fitted scale."""
        if not hasattr(self, "gain_mask_"):
            raise ValueError("fit the extractor first")
        return [self._transform_one(xy, cell_types) for xy, cell_types in clouds]

    def fit_transform(self, clouds: Sequence[Cloud]) -> list[CellColocalizationResult]:
        clouds = list(clouds)
        return self.fit(clouds).transform(clouds)

    def _transform_one(self, xy: np.ndarray, cell_types: Sequence[Any]) -> CellColocalizationResult:
        n_types = len(self.vocabulary_)
        codes = encode_labels(cell_types, self.vocabulary_)
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if len(xy) != len(codes):
            raise ValueError(f"{len(xy)} positions for {len(codes)} labels")
        counts = _counts(xy, codes, n_types, self.n_neighbors)
        k = counts.degree[:, None].astype(np.float64)
        focal = np.where(codes >= 0, codes, n_types)
        gain = self.gain_mask_[focal]
        room = np.where(gain, k * (1 - counts.q), k * counts.q)
        valid = (counts.variance > TINY) & (room > TINY)
        excess = counts.observed - counts.expected
        y = np.full(excess.shape, np.nan)
        np.divide(excess, room, out=y, where=valid)
        s2 = np.full(excess.shape, np.inf)
        np.divide(counts.variance, room * room, out=s2, where=valid)
        # groups for the prior and tau^2: (focal type, neighbour type) on the pair scale, else the neighbour type
        groups = focal if self.scale_mode == "pair" else np.zeros(len(codes), dtype=np.int64)
        n_groups = n_types + 1 if self.scale_mode == "pair" else 1
        member = _one_hot(groups, n_groups).T.tocsr()
        if self.shrink_target == "image_mean":
            num = member @ np.where(valid, excess, 0.0)
            den = member @ np.where(valid, room, 0.0)
            prior = np.divide(num, den, out=np.zeros_like(num), where=den >= MIN_ROOM)
        else:
            prior = np.zeros((n_groups, n_types))
        if self.tau2 == "auto":
            min_cells = MIN_PAIR_CELLS if self.scale_mode == "pair" else 2
            tau2 = dersimonian_laird(y, s2, valid, groups, n_groups, min_cells)
        else:
            tau2 = np.full((n_groups, n_types), float(self.tau2))
        shrunk = shrink(y, s2, valid, prior[groups], tau2[groups])
        homotypic = np.full(len(codes), np.nan)
        modelled = codes >= 0
        homotypic[modelled] = shrunk[np.flatnonzero(modelled), codes[modelled]]
        return CellColocalizationResult(
            Y_shrunk=shrunk.astype(self.dtype),
            Y_raw=y.astype(self.dtype),
            s2=s2.astype(self.dtype),
            valid_mask=valid,
            degree=counts.degree,
            homotypic_excess_shrunk=homotypic.astype(self.dtype),
        )

    # -- summaries -------------------------------------------------------------------------------------------

    def summarize_cloud(self, result: CellColocalizationResult, cell_types: Sequence[Any]) -> dict[str, np.ndarray]:
        """A cloud's cell-level excess pooled by (focal type B, neighbour type A), for cloud-level features.

        ``pooled_excess``: the precision-weighted mean Σ Y / s^2 / Σ 1 / s^2 of the valid Y_raw of B's cells;
        ``pooled_variance``: its sampling variance 1 / Σ 1 / s^2 (both NaN without a valid cell);
        ``quantiles``: (3, K, K) quantiles 0.1, 0.5, 0.9 of B's cells' Y_shrunk; ``n_cells``: (K,) cells of each B.
        """
        n_types = len(self.vocabulary_)
        codes = encode_labels(cell_types, self.vocabulary_)
        valid = result.valid_mask
        y = np.where(valid, np.asarray(result.Y_raw, dtype=np.float64), 0.0)
        s2 = np.asarray(result.s2, dtype=np.float64)
        w = np.where(valid, 1.0 / np.maximum(np.where(valid, s2, 1.0), WEIGHT_FLOOR), 0.0)
        member = _one_hot(codes, n_types).T.tocsr()  # background cells are in no group
        sum_w = member @ w
        pooled = np.divide(member @ (w * y), sum_w, out=np.full_like(sum_w, np.nan), where=sum_w > 0)
        variance = np.divide(1.0, sum_w, out=np.full_like(sum_w, np.nan), where=sum_w > 0)
        levels = np.array([0.1, 0.5, 0.9])
        quantiles = np.full((3, n_types, n_types), np.nan)
        shrunk = np.asarray(result.Y_shrunk, dtype=np.float64)
        for b in range(n_types):  # over the K focal types, each vectorised over its cells
            rows = codes == b
            if rows.any():
                quantiles[:, b, :] = np.quantile(shrunk[rows], levels, axis=0)
        return {
            "pooled_excess": pooled,
            "pooled_variance": variance,
            "quantiles": quantiles,
            "quantile_levels": levels,
            "n_cells": np.bincount(codes[codes >= 0], minlength=n_types),
        }
