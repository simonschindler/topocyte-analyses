"""Exact algebra on signed, weighted step functions of persistence bars.

A curve is a finite sum f = Σ w·1[b, d) over bars [b, d) with real weights w.
A reduced Betti curve has unit weights. The study's residual curves have
weights 1/a and -1/(R·a): the observed labelling's curve minus the mean of R
reference curves, divided by a per-curve scale a. Every operation is exact
arithmetic on the bar endpoints, up to floating-point rounding: no curve is
ever evaluated on a grid, and nothing is integrated by quadrature. The module
does not care about units; the study's values are squared radii in px².

- ``CurveSet`` holds n curves as flat bar arrays with offsets. ``curve_set``,
  ``combine`` and ``scale`` build them.
- ``inner_matrix`` gives the exact L² inner products. Each curve's step
  function is built once: its sorted distinct breakpoints, and its value from
  each breakpoint to the next. The inner product of two curves is then a merged
  sweep over their breakpoints, Σ f·g·width. With unit weights it equals
  topocyte's ``betti_inner_product_matrix``, which integrates from 0; the
  integral here runs over the whole line, which is the same thing for births
  >= 0, as squared radii are.
- ``spread2`` gives each centre's reference spread, mean_r ‖ref_r − mean‖².
- ``components`` gives the component functions of an fPCA from
  ``topocyte.ph.fpca_from_gram``, and ``score`` the scores of any curve on
  them.

Rounding. A step value is a running sum of events, +w at each birth and −w at
each death, in position order. The component functions sum millions of events
of mixed sign, so the running sums are Neumaier-compensated to second order
(``_add``), as are the antiderivative that scoring reads and each curve's
score. The value after a curve's last breakpoint is set to exactly 0: every
bar has ended there, and its events sum to 0 exactly. The inner products sum
f·g·width plainly: ``_dot`` and ``_sqdist`` see one pair of curves at a time,
a thousand or so terms.

Components. ``fpca_from_gram`` centres the Gram matrix algebraically
(H G H), so each eigenvector v_k sums to 0 and the k-th component function is
φ_k = Σ_i v_ik/√ρ_k · f_i, with no mean term. A curve g scores
⟨g, φ_k⟩ − ⟨f̄, φ_k⟩, which reproduces ``fpca_from_gram``'s scores
√ρ_k·v_ik on the fit curves. ⟨g, φ_k⟩ = Σ_bars w·(Φ_k(d) − Φ_k(b)), with
Φ_k the antiderivative of φ_k, exact at g's bar endpoints wherever they fall.

Threads. ``inner_matrix`` runs its sweeps in numba ``prange`` on ``n_jobs``
threads. Call it from one Python thread only: numba's workqueue layer, the one
the study venv has, aborts when parallel code is entered from several threads
at once. Everything else here is serial.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from topocyte.ph import FPCAResult


@dataclass
class CurveSet:
    """n curves: curve i is Σ w·1[b, d) over its rows ``offsets[i]:offsets[i + 1]``.

    ``b``, ``d`` and ``w`` are float64 (N,) arrays of finite bars with d > b;
    ``offsets`` is int64 (n + 1,), rising from 0 to N.
    """

    b: np.ndarray
    d: np.ndarray
    w: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        self.b, self.d, self.w = (np.ascontiguousarray(a, dtype=np.float64) for a in (self.b, self.d, self.w))
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)
        n_bars = self.b.size
        if any(a.ndim != 1 or a.size != n_bars for a in (self.b, self.d, self.w)):
            raise ValueError("b, d and w must be 1-D arrays of one length")
        o = self.offsets
        if o.ndim != 1 or o.size == 0 or o[0] != 0 or o[-1] != n_bars or np.any(np.diff(o) < 0):
            raise ValueError("offsets must rise from 0 to the number of bars")
        if not (np.all(np.isfinite(self.b)) and np.all(np.isfinite(self.d)) and np.all(np.isfinite(self.w))):
            raise ValueError("bars and weights must be finite")
        if np.any(self.d <= self.b):
            raise ValueError("every bar needs d > b")

    def __len__(self) -> int:
        return self.offsets.size - 1


@dataclass
class Components:
    """The first K component functions of an fPCA, as one step table.

    ``phi[m, k]`` is φ_k on [x[m], x[m + 1]); φ_k is 0 before x[0] and from
    x[M - 1] on, so ``phi[M - 1]`` is 0. ``mean_term[k]`` is ⟨f̄, φ_k⟩, with
    f̄ the mean fit curve.
    """

    x: np.ndarray
    phi: np.ndarray
    mean_term: np.ndarray


# ------------------------------------------------------------ building sets


def curve_set(diagrams: Sequence[np.ndarray]) -> CurveSet:
    """One unit-weight curve per diagram: topocyte's reduced Betti curve.

    A diagram is an (m, 2) array of (birth, death) rows. Essential bars (death
    inf) and bars with death == birth are left out, as
    ``topocyte.ph.BettiCurve`` leaves them out.
    """
    arrays = []
    for diagram in diagrams:
        a = np.asarray(diagram, dtype=np.float64)
        if a.size == 0:
            a = a.reshape(0, 2)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"a diagram must be an (m, 2) array; got shape {a.shape}")
        arrays.append(a)
    bars = np.concatenate(arrays) if arrays else np.empty((0, 2))
    b, d = bars[:, 0], bars[:, 1]
    if not np.all(np.isfinite(b)) or np.any(np.isnan(d)) or np.any(d < b):
        raise ValueError("every bar needs a finite birth and a death >= its birth (inf for an essential bar)")
    keep = np.isfinite(d) & (d > b)
    owner = np.repeat(np.arange(len(arrays)), np.array([len(a) for a in arrays], dtype=np.int64))
    counts = np.bincount(owner[keep], minlength=len(arrays))
    return CurveSet(b[keep], d[keep], np.ones(int(keep.sum())), np.concatenate([[0], np.cumsum(counts)]))


def combine(sets: Sequence[CurveSet], coeffs: Sequence[float]) -> CurveSet:
    """Curve i is Σ_j coeffs[j]·sets[j][i]: the bars of every set, with weights times the set's coefficient.

    Bars are kept as they are, never merged, so the result is exact. Curve i's
    rows hold set 0's bars of curve i, then set 1's, and so on.
    """
    sets = list(sets)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if not sets or coeffs.shape != (len(sets),):
        raise ValueError("combine needs at least one set and one coefficient per set")
    n = len(sets[0])
    if any(len(s) != n for s in sets):
        raise ValueError("every set must hold the same number of curves")
    lengths = np.stack([np.diff(s.offsets) for s in sets])  # (J, n)
    offsets = np.concatenate([[0], np.cumsum(lengths.sum(axis=0))])
    before = np.cumsum(lengths, axis=0) - lengths  # rows of the earlier sets in each output curve
    b, d, w = (np.empty(int(offsets[-1])) for _ in range(3))
    for j, s in enumerate(sets):
        dest = np.repeat(offsets[:-1] + before[j] - s.offsets[:-1], lengths[j]) + np.arange(s.b.size)
        b[dest], d[dest], w[dest] = s.b, s.d, coeffs[j] * s.w
    return CurveSet(b, d, w, offsets)


def scale(cs: CurveSet, factors: np.ndarray) -> CurveSet:
    """Curve i times ``factors[i]``. The bars and offsets are shared with ``cs``; only the weights are new."""
    factors = np.asarray(factors, dtype=np.float64)
    if factors.shape != (len(cs),):
        raise ValueError(f"scale needs one factor per curve ({len(cs)}); got shape {factors.shape}")
    return CurveSet(cs.b, cs.d, cs.w * np.repeat(factors, np.diff(cs.offsets)), cs.offsets)


# ------------------------------------------------------------ step tables


@nb.njit(cache=True)
def _add(s: float, cs: float, ccs: float, x: float) -> tuple[float, float, float]:
    """One step of second-order Neumaier summation: the running sum is s + (cs + ccs).

    Neumaier's step keeps the rounding error of s + x exactly; here those
    errors are summed by a second Neumaier step into cs, whose own rounding
    errors go to ccs (Klein 2006, iterative Kahan–Babuška). First order alone
    drifts by about 2e-12 of the result over 2e6 events of ±1e8 and ±1e-8.
    """
    t = s + x
    if abs(s) >= abs(x):
        c = (s - t) + x
    else:
        c = (x - t) + s
    u = cs + c
    if abs(cs) >= abs(c):
        cc = (cs - u) + c
    else:
        cc = (c - u) + cs
    return t, u, ccs + cc


@nb.njit(cache=True)
def _steps(b, d, w, offsets):
    """Each curve's step function: sorted distinct breakpoints x and the value v from each to the next.

    Curve i's table is rows ``2·offsets[i]`` to ``2·offsets[i] + count[i]`` of
    x and v. Its value after the last breakpoint is 0.
    """
    n = offsets.size - 1
    x = np.empty(2 * b.size)
    v = np.empty(2 * b.size)
    count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo, hi = offsets[i], offsets[i + 1]
        if hi == lo:
            continue
        pos = np.concatenate((b[lo:hi], d[lo:hi]))
        wt = np.concatenate((w[lo:hi], -w[lo:hi]))
        start = 2 * lo
        k = start - 1
        s = 0.0
        cs = 0.0
        ccs = 0.0
        for e in np.argsort(pos, kind="mergesort"):
            if k < start or pos[e] != x[k]:
                if k >= start:
                    v[k] = s + (cs + ccs)
                k += 1
                x[k] = pos[e]
            s, cs, ccs = _add(s, cs, ccs, wt[e])
        v[k] = 0.0
        count[i] = k + 1 - start
    return x, v, count


@nb.njit(cache=True)
def _dot(x1, v1, s1, n1, x2, v2, s2, n2):
    """∫ f·g of two step tables, by a merged sweep over their breakpoints.

    Before its first breakpoint and after its last a curve is 0, so the sweep
    starts at the first breakpoint of either and stops when one table ends.
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    p, q = s1, s2
    e1, e2 = s1 + n1, s2 + n2
    f = 0.0
    g = 0.0
    acc = 0.0
    prev = min(x1[p], x2[q])
    while p < e1 and q < e2:
        a = x1[p]
        c = x2[q]
        t = min(a, c)
        acc += f * g * (t - prev)
        prev = t
        if a == t:
            f = v1[p]
            p += 1
        if c == t:
            g = v2[q]
            q += 1
    return acc


@nb.njit(cache=True)
def _sqdist(x1, v1, s1, n1, x2, v2, s2, n2):
    """∫ (f − g)² of two step tables, by a merged sweep that runs to the end of both."""
    p, q = s1, s2
    e1, e2 = s1 + n1, s2 + n2
    f = 0.0
    g = 0.0
    acc = 0.0
    prev = 0.0
    while p < e1 or q < e2:
        a = x1[p] if p < e1 else np.inf
        c = x2[q] if q < e2 else np.inf
        t = min(a, c)
        acc += (f - g) * (f - g) * (t - prev)  # f = g = 0 before the first breakpoint
        prev = t
        if a == t:
            f = v1[p]
            p += 1
        if c == t:
            g = v2[q]
            q += 1
    return acc


@nb.njit(cache=True, parallel=True)
def _gram(x, v, start, count, out):
    """The symmetric Gram matrix; each prange step fills rows t and n − 1 − t, so every step does n + 1 sweeps."""
    n = count.size
    for t in nb.prange((n + 1) // 2):
        for j in range(t, n):
            value = _dot(x, v, start[t], count[t], x, v, start[j], count[j])
            out[t, j] = value
            out[j, t] = value
        i = n - 1 - t
        if i != t:
            for j in range(i, n):
                value = _dot(x, v, start[i], count[i], x, v, start[j], count[j])
                out[i, j] = value
                out[j, i] = value


@nb.njit(cache=True, parallel=True)
def _cross(xa, va, sa, ca, xb, vb, sb, cb, out):
    """Every ⟨a_i, b_j⟩; each prange step fills one row."""
    for i in nb.prange(ca.size):
        for j in range(cb.size):
            out[i, j] = _dot(xa, va, sa[i], ca[i], xb, vb, sb[j], cb[j])


@nb.njit(cache=True)
def _spread(x, v, start, count):
    """Per curve i, (1/R²) Σ_{r<s} ∫ (ref_r − ref_s)² over the R step tables ``start[r, i]``."""
    R, n = start.shape
    out = np.empty(n)
    for i in range(n):
        acc = 0.0
        for r in range(R):
            for s in range(r + 1, R):
                acc += _sqdist(x, v, start[r, i], count[r, i], x, v, start[s, i], count[s, i])
        out[i] = acc / (R * R)
    return out


# ------------------------------------------------------------ inner products


def _threads(n_jobs: int) -> int:
    limit = nb.config.NUMBA_NUM_THREADS
    integer = isinstance(n_jobs, (int, np.integer)) and not isinstance(n_jobs, bool)
    if integer and n_jobs == -1:
        return limit
    if integer and n_jobs >= 1:
        return min(int(n_jobs), limit)
    raise ValueError(f"n_jobs must be -1 or a positive integer; got {n_jobs!r}")


def inner_matrix(a: CurveSet, b: CurveSet | None = None, n_jobs: int = -1) -> np.ndarray:
    """Exact L² inner products ⟨a_i, b_j⟩, (len(a), len(b)); the Gram matrix of ``a`` when ``b`` is None.

    Each entry is one merged sweep over two step tables, so the result does
    not depend on ``n_jobs`` (-1: all of numba's threads), and the Gram matrix
    is exactly symmetric. A curve without bars gives a row of zeros.
    """
    threads = _threads(n_jobs)
    xa, va, ca = _steps(a.b, a.d, a.w, a.offsets)
    sa = 2 * a.offsets[:-1]
    previous = nb.get_num_threads()
    nb.set_num_threads(threads)
    try:
        if b is None:
            out = np.empty((len(a), len(a)))
            _gram(xa, va, sa, ca, out)
        else:
            xb, vb, cb = _steps(b.b, b.d, b.w, b.offsets)
            out = np.empty((len(a), len(b)))
            _cross(xa, va, sa, ca, xb, vb, 2 * b.offsets[:-1], cb, out)
    finally:
        nb.set_num_threads(previous)
    return out


def spread2(refs: Sequence[CurveSet]) -> np.ndarray:
    """Per curve index i, the spread (1/R)·Σ_r ‖ref_r[i]‖² − ‖mean_r ref_r[i]‖² of R reference sets.

    It is computed as (1/(2R²))·Σ_{r,s} ‖ref_r[i] − ref_s[i]‖², the same
    number from the R × R exact squared distances. Each distance is a merged
    sweep of (ref_r − ref_s)², so the spread is never negative, and it is
    exactly 0 when the references agree. The inner-product form cancels to a
    rounding residue of either sign there.
    """
    refs = list(refs)
    if not refs:
        raise ValueError("spread2 needs at least one reference set")
    n = len(refs[0])
    if any(len(r) != n for r in refs):
        raise ValueError("every reference set must hold the same number of curves")
    tables = [_steps(r.b, r.d, r.w, r.offsets) for r in refs]
    base = np.concatenate([[0], np.cumsum([x.size for x, _, _ in tables])])
    x = np.concatenate([t[0] for t in tables])
    v = np.concatenate([t[1] for t in tables])
    start = np.stack([base[k] + 2 * r.offsets[:-1] for k, r in enumerate(refs)])
    count = np.stack([t[2] for t in tables])
    return _spread(x, v, start, count)


# ------------------------------------------------------------ components and scores


@nb.njit(cache=True)
def _component_steps(pos, order, w, owner, coef, n_positions):
    """φ_k = Σ_i coef[i, k]·f_i as a step table: every event of every fit curve, in position order, compensated.

    ``pos`` holds the births then the deaths of the N fit bars, ``order`` sorts
    it, and event e of bar p adds ±w[p]·coef[owner[p], k].
    """
    N = w.size
    K = coef.shape[1]
    x = np.empty(n_positions)
    phi = np.empty((n_positions, K))
    s = np.zeros(K)
    cs = np.zeros(K)
    ccs = np.zeros(K)
    m = -1
    for e in order:
        if m < 0 or pos[e] != x[m]:
            if m >= 0:
                for k in range(K):
                    phi[m, k] = s[k] + (cs[k] + ccs[k])
            m += 1
            x[m] = pos[e]
        p = e if e < N else e - N
        sign = 1.0 if e < N else -1.0
        i = owner[p]
        for k in range(K):
            s[k], cs[k], ccs[k] = _add(s[k], cs[k], ccs[k], sign * (w[p] * coef[i, k]))
    if m >= 0:
        phi[m, :] = 0.0
    return x, phi


@nb.njit(cache=True)
def _antiderivative(x, phi):
    """Φ_k(x[m]) = Σ_{m' < m} φ_k on [x[m'], x[m' + 1]) times its width, compensated; Φ_k(x[0]) = 0."""
    M, K = phi.shape
    Phi = np.empty((M, K))
    s = np.zeros(K)
    cs = np.zeros(K)
    ccs = np.zeros(K)
    for m in range(M):
        for k in range(K):
            Phi[m, k] = s[k] + (cs[k] + ccs[k])
        if m + 1 < M:
            width = x[m + 1] - x[m]
            for k in range(K):
                s[k], cs[k], ccs[k] = _add(s[k], cs[k], ccs[k], phi[m, k] * width)
    return Phi


@nb.njit(cache=True)
def _locate(x, t):
    """The largest m with x[m] <= t, or -1 when t < x[0]: binary search."""
    lo = 0
    hi = x.size
    while lo < hi:
        mid = (lo + hi) >> 1
        if x[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@nb.njit(cache=True)
def _inner_with(b, d, w, offsets, x, phi, Phi):
    """⟨g_i, φ_k⟩ = Σ_bars w·(Φ_k(d) − Φ_k(b)) for every curve g_i, compensated per curve.

    Φ_k(t) = Φ_k(x[m]) + φ_k[m]·(t − x[m]) for x[m] <= t < x[m + 1], and 0
    before x[0]. The two cumulative values are subtracted first, so a bar
    inside one segment costs no cancellation.
    """
    n = offsets.size - 1
    K = phi.shape[1]
    out = np.zeros((n, K))
    s = np.zeros(K)
    cs = np.zeros(K)
    ccs = np.zeros(K)
    for i in range(n):
        s[:] = 0.0
        cs[:] = 0.0
        ccs[:] = 0.0
        for p in range(offsets[i], offsets[i + 1]):
            mb = _locate(x, b[p])
            md = _locate(x, d[p])
            for k in range(K):
                cum = Phi[md, k] if md >= 0 else 0.0
                local = phi[md, k] * (d[p] - x[md]) if md >= 0 else 0.0
                if mb >= 0:
                    cum -= Phi[mb, k]
                    local -= phi[mb, k] * (b[p] - x[mb])
                s[k], cs[k], ccs[k] = _add(s[k], cs[k], ccs[k], w[p] * (cum + local))
        for k in range(K):
            out[i, k] = s[k] + (cs[k] + ccs[k])
    return out


def components(fit: CurveSet, result: FPCAResult, n_components: int) -> Components:
    """The first ``n_components`` component functions φ_k = Σ_i v_ik/√ρ_k · f_i of an fPCA of ``fit``.

    ``result`` is ``topocyte.ph.fpca_from_gram`` of ``inner_matrix(fit)``. It
    must have ``normalization="none"``, and only its eigenvectors v and Gram
    eigenvalues ρ enter φ. ``mean_term`` is ⟨f̄, φ_k⟩, the mean of
    ⟨f_i, φ_k⟩ over the fit curves.
    """
    if result.normalization != "none":
        raise ValueError(f"components need an fPCA with normalization 'none'; got {result.normalization!r}")
    V = np.asarray(result.eigenvectors, dtype=np.float64)
    rho = np.asarray(result.gram_eigenvalues, dtype=np.float64)
    if V.shape[0] != len(fit):
        raise ValueError(f"the fPCA has {V.shape[0]} curves but the fit set {len(fit)}")
    K = n_components
    if not (isinstance(K, (int, np.integer)) and not isinstance(K, bool) and 1 <= K <= rho.size):
        raise ValueError(f"n_components must be 1..{rho.size}, the components the fPCA kept; got {K!r}")
    coef = np.ascontiguousarray(V[:, :K] / np.sqrt(rho[:K]))
    pos = np.concatenate([fit.b, fit.d])
    order = np.argsort(pos, kind="stable")
    n_positions = int(np.count_nonzero(np.diff(pos[order]))) + 1 if pos.size else 0
    owner = np.repeat(np.arange(len(fit)), np.diff(fit.offsets))
    x, phi = _component_steps(pos, order, fit.w, owner, coef, n_positions)
    raw = _inner_with(fit.b, fit.d, fit.w, fit.offsets, x, phi, _antiderivative(x, phi))
    mean_term = np.array([math.fsum(raw[:, k]) / len(fit) for k in range(K)])
    return Components(x, phi, mean_term)


def score(cs: CurveSet, comp: Components) -> np.ndarray:
    """The scores (len(cs), K) of ``cs`` on the components: ⟨g, φ_k⟩ − mean_term_k.

    ⟨g, φ_k⟩ is exact at g's bar endpoints, through the antiderivative of
    φ_k. A curve without bars scores −mean_term.
    """
    Phi = _antiderivative(comp.x, comp.phi)
    return _inner_with(cs.b, cs.d, cs.w, cs.offsets, comp.x, comp.phi, Phi) - comp.mean_term
