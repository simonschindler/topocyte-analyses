"""Unit tests of ``curves.py`` on synthetic diagrams.

topocyte is the oracle of the inner products and the scores: its
``betti_inner_product_matrix`` for unit-weight curves, and its
``fpca_from_gram`` for the scores of the fit curves. The tests therefore need
the pinned topocyte on ``PYTHONPATH``. ``math.fsum`` is the oracle of the
compensated event sums.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import topocyte as tc

import curves

R = 19  # reference labellings per centre, as in the study


def diagram(rng: np.random.Generator, n_bars: int, h0: bool) -> np.ndarray:
    """A random diagram in squared radii (px²), births >= 0, rounded to 0.1 so that endpoints tie across curves.

    An H0-like diagram has every birth at 0. Each diagram also holds an
    essential bar and a zero-length bar, which the reduced curve leaves out;
    rounding can make more bars zero-length.
    """
    births = np.zeros(n_bars) if h0 else np.round(rng.uniform(0, 1e4, n_bars), 1)
    deaths = np.round(births + rng.exponential(2.5e3, n_bars), 1)
    dropped = np.array([[0.0, np.inf], [5e3, 5e3]])
    return np.vstack([np.column_stack([births, deaths]), dropped])


def diagrams(rng: np.random.Generator, n: int, lo: int = 0, hi: int = 30) -> list[np.ndarray]:
    """n random diagrams of lo..hi bars each, H0-like or not at random."""
    return [diagram(rng, int(rng.integers(lo, hi + 1)), bool(rng.integers(2))) for _ in range(n)]


def residuals(rng: np.random.Generator, n: int) -> curves.CurveSet:
    """n curves of the study's form: an observed curve minus the mean of R reference curves, scaled per curve."""
    observed = curves.curve_set(diagrams(rng, n))
    refs = [curves.curve_set(diagrams(rng, n)) for _ in range(R)]
    residual = curves.combine([observed, *refs], [1.0] + [-1.0 / R] * R)
    return curves.scale(residual, rng.uniform(0.5, 2.0, n))


def fpca_result(eigenvectors: np.ndarray, gram_eigenvalues: np.ndarray) -> tc.ph.FPCAResult:
    """An fPCA result with given eigenpairs; ``components`` reads nothing else."""
    k = gram_eigenvalues.size
    return tc.ph.FPCAResult(
        scores=np.zeros(eigenvectors.shape),
        eigenvalues=gram_eigenvalues.copy(),
        gram_eigenvalues=gram_eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=np.full(k, 1.0 / k),
        cumulative_explained_variance=np.arange(1, k + 1) / k,
    )


def test_inner_matches_topocyte():
    """With unit weights the Gram matrix is topocyte's, entry for entry; the cross matrix and any n_jobs give the same."""
    rng = np.random.default_rng(0)
    diags = diagrams(rng, 50)
    cs = curves.curve_set(diags)
    G = curves.inner_matrix(cs)
    T = tc.ph.betti_inner_product_matrix([tc.ph.BettiCurve(d, 0) for d in diags], dim=0)
    assert np.allclose(G, T, rtol=1e-12, atol=0)
    assert np.array_equal(G, G.T)
    assert np.array_equal(curves.inner_matrix(cs, n_jobs=1), G)
    head, tail = curves.curve_set(diags[:31]), curves.curve_set(diags[31:])
    assert np.array_equal(curves.inner_matrix(head, tail), G[:31, 31:])
    assert np.array_equal(curves.inner_matrix(head), G[:31, :31])  # an odd count: the middle row of the balanced loop


def test_signed_weights_bilinear():
    """<a - b/19, c> = <a, c> - <b, c>/19 and <s·a, c> = s<a, c>, to 1e-12 of the terms; the sweep is symmetric."""
    rng = np.random.default_rng(1)
    a, b, c = (curves.curve_set(diagrams(rng, 40)) for _ in range(3))
    ac, bc = curves.inner_matrix(a, c), curves.inner_matrix(b, c)
    combined = curves.combine([a, b], [1.0, -1.0 / R])
    lhs = curves.inner_matrix(combined, c)
    assert np.all(np.abs(lhs - (ac - bc / R)) <= 1e-12 * (np.abs(ac) + np.abs(bc) / R))
    assert np.array_equal(curves.inner_matrix(c, combined), lhs.T)
    factors = rng.uniform(0.5, 2.0, 40)
    assert np.allclose(curves.inner_matrix(curves.scale(a, factors), c), factors[:, None] * ac, rtol=1e-12, atol=0)


def test_spread_matches_bruteforce():
    """spread2 equals mean_r ||ref_r - mean||², with the mean curve built by combine and the norms by inner_matrix."""
    rng = np.random.default_rng(2)
    refs = [curves.curve_set(diagrams(rng, 25)) for _ in range(R)]
    mean = curves.combine(refs, [1.0 / R] * R)
    deviations = [np.diag(curves.inner_matrix(curves.combine([ref, mean], [1.0, -1.0]))) for ref in refs]
    assert np.allclose(curves.spread2(refs), np.mean(deviations, axis=0), rtol=1e-10, atol=0)


def test_spread_zero_when_references_agree():
    """References that agree, bar for bar in any row order, have a spread of exactly 0, with no rounding residue.

    The design's s0 is the median of s² over the centres with s > 0, so a
    spread that should be 0 must not come out as a positive ulp.
    """
    rng = np.random.default_rng(3)
    base = diagrams(rng, 20, lo=1)
    refs = [curves.curve_set([d[rng.permutation(len(d))] for d in base]) for _ in range(R)]
    assert np.array_equal(curves.spread2(refs), np.zeros(20))
    assert np.all(curves.spread2([refs[0], curves.curve_set(diagrams(rng, 20, lo=1))]) > 0)


def test_scores_reproduce_fpca_from_gram():
    """The scores of the fit curves on their components are fpca_from_gram's, to 1e-9 of the largest."""
    rng = np.random.default_rng(4)
    cs = residuals(rng, 300)
    res = tc.ph.fpca_from_gram(curves.inner_matrix(cs))
    S = curves.score(cs, curves.components(cs, res, 9))
    assert S.shape == (300, 9)
    assert np.max(np.abs(S - res.scores[:, :9])) <= 1e-9 * np.max(np.abs(res.scores[:, :9]))


def test_scores_of_new_curves_match_kernel_route():
    """Curves outside the fit score (<g, f_i> - <f̄, f_i>)·v_ik/√ρ_k, from their inner products with the fit curves.

    Most of their bar endpoints fall inside segments of phi, where the
    antiderivative is read between breakpoints; the fit curves' never do.
    """
    rng = np.random.default_rng(8)
    fit = residuals(rng, 200)
    G = curves.inner_matrix(fit)
    res = tc.ph.fpca_from_gram(G)
    new = residuals(rng, 150)
    kernel = (curves.inner_matrix(new, fit) - G.mean(axis=0)) @ (res.eigenvectors[:, :9] / np.sqrt(res.gram_eigenvalues[:9]))
    S = curves.score(new, curves.components(fit, res, 9))
    assert np.max(np.abs(S - kernel)) <= 1e-9 * np.max(np.abs(kernel))


def test_score_is_linear():
    """A residual's score is the observed score minus the mean reference score, to 1e-10 of the scores.

    score(g) = <g, phi_k> - mean_term_k is affine. The coefficients 1, -1/R,
    ..., -1/R sum to 0, so the residual's score also loses one mean_term:
    score(obs - mean refs) = score(obs) - mean_r score(ref_r) - mean_term.
    """
    rng = np.random.default_rng(5)
    fit = residuals(rng, 200)
    comp = curves.components(fit, tc.ph.fpca_from_gram(curves.inner_matrix(fit)), 9)
    observed = curves.curve_set(diagrams(rng, 100))
    refs = [curves.curve_set(diagrams(rng, 100)) for _ in range(R)]
    lhs = curves.score(curves.combine([observed, *refs], [1.0] + [-1.0 / R] * R), comp)
    obs_score = curves.score(observed, comp)
    rhs = obs_score - np.mean([curves.score(ref, comp) for ref in refs], axis=0) - comp.mean_term
    assert np.max(np.abs(lhs - rhs)) <= 1e-10 * np.max(np.abs(obs_score))


def test_component_sum_compensated():
    """phi equals math.fsum of its events per segment, to 1e-12, for 2e6 events of ±1e8 and ±1e-8.

    The 1e8 bars come in pairs of opposite weight on one interval, so on every
    segment they cancel exactly and phi is 1e-8 times the number of small bars
    alive there. A running sum without compensation loses the small events
    under the large ones; the last assertion shows that this input makes it
    fail.
    """
    rng = np.random.default_rng(6)
    pool = np.sort(rng.choice(np.arange(1.0, 1e4), 24, replace=False))
    n_pairs, n_small = 250_000, 500_000
    first = rng.integers(0, pool.size - 1, n_pairs + n_small)
    second = rng.integers(first + 1, pool.size)
    b, d = pool[first], pool[second]
    b = np.concatenate([b[:n_pairs], b])
    d = np.concatenate([d[:n_pairs], d])
    w = np.concatenate([np.full(n_pairs, 1e8), np.full(n_pairs, -1e8), np.full(n_small, 1e-8)])
    order = rng.permutation(w.size)
    fit = curves.CurveSet(b[order], d[order], w[order], np.array([0, w.size]))
    comp = curves.components(fit, fpca_result(np.ones((1, 1)), np.ones(1)), 1)

    positions = np.concatenate([fit.b, fit.d])
    events = np.concatenate([fit.w, -fit.w])
    by_position = np.argsort(positions, kind="stable")
    ends = np.searchsorted(positions[by_position], pool, side="right")
    ordered = events[by_position].tolist()
    expected = np.array([math.fsum(ordered[:end]) for end in ends])
    assert positions.size == 2_000_000
    assert np.array_equal(comp.x, pool)
    assert np.all(np.abs(comp.phi[:, 0] - expected) <= 1e-12 * np.abs(expected))
    assert expected[-1] == 0 and np.all(expected[:-1] > 0)
    naive = np.cumsum(events[by_position])[ends - 1]
    assert np.max(np.abs(naive[:-1] - expected[:-1]) / expected[:-1]) > 1e-3


def test_empty_curves():
    """A curve without bars has zero rows and columns in the Gram matrix and the score -mean_term."""
    rng = np.random.default_rng(7)
    diags = diagrams(rng, 30, lo=1)
    empty = [3, 10, 11, 29]
    diags[3] = np.empty((0, 2))
    diags[10] = np.array([[0.0, np.inf]])
    diags[11] = np.array([[7.5, 7.5], [2.0, 2.0]])
    diags[29] = np.empty((0, 2))
    cs = curves.curve_set(diags)
    assert np.array_equal(np.diff(cs.offsets)[empty], np.zeros(4))
    G = curves.inner_matrix(cs)
    assert np.all(G[empty] == 0) and np.all(G[:, empty] == 0)
    comp = curves.components(cs, tc.ph.fpca_from_gram(G), 5)
    assert np.array_equal(curves.score(cs, comp)[empty], np.tile(-comp.mean_term, (4, 1)))

    none = curves.curve_set([])
    assert len(none) == 0
    assert curves.inner_matrix(none).shape == (0, 0)
    assert curves.score(none, comp).shape == (0, 5)
    blank = curves.curve_set([np.empty((0, 2))] * 3)
    assert np.array_equal(curves.inner_matrix(blank, cs), np.zeros((3, 30)))
    assert np.array_equal(curves.spread2([blank] * R), np.zeros(3))


def test_bad_inputs_refused():
    """Malformed bars, mismatched sets, a cosine fit, too many components and a bad n_jobs raise ValueError."""
    one = curves.curve_set([np.array([[0.0, 1.0]])])
    two = curves.curve_set([np.array([[0.0, 1.0]]), np.array([[0.5, 2.0]])])
    with pytest.raises(ValueError, match="death"):
        curves.curve_set([np.array([[2.0, 1.0]])])
    with pytest.raises(ValueError, match="birth"):
        curves.curve_set([np.array([[np.nan, 1.0]])])
    with pytest.raises(ValueError, match=r"\(m, 2\)"):
        curves.curve_set([np.zeros((2, 3))])
    with pytest.raises(ValueError, match="d > b"):
        curves.CurveSet(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="same number"):
        curves.combine([one, two], [1.0, 1.0])
    with pytest.raises(ValueError, match="factor"):
        curves.scale(two, np.ones(3))
    with pytest.raises(ValueError, match="n_jobs"):
        curves.inner_matrix(two, n_jobs=0)
    res = tc.ph.fpca_from_gram(curves.inner_matrix(two))
    with pytest.raises(ValueError, match="n_components"):
        curves.components(two, res, 2)
    with pytest.raises(ValueError, match="normalization"):
        curves.components(two, tc.ph.fpca_from_gram(curves.inner_matrix(two), normalization="cosine"), 1)
