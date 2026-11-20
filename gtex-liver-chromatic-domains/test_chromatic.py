"""Unit tests of ``chromatic.py`` on synthetic clouds.

chalc 15.0.1 is the oracle for the six-packs: its ``SubChromaticInclusion``
of each class present, with its radii squared. topocyte's alpha persistent
homology (α², squared radii) is the oracle for the codomain and for each
class's domain, so ``test_codomain_is_alpha_ph`` and
``test_absent_class_is_empty`` need the pinned topocyte on ``PYTHONPATH``.
"""

from __future__ import annotations

import itertools
import time
from fractions import Fraction
from math import comb

import numpy as np
import pytest

import chromatic
from common import PAIRS

NAMES = ("kernel", "domain", "image", "codomain", "cokernel", "relative")
CLASS_MODULES = ("kernel", "domain", "image", "cokernel", "relative")

CLOUDS = [(n, s, seed) for seed, (n, s) in enumerate([(30, 1), (60, 2), (120, 3), (200, 4), (150, 4)])]


def cloud(n: int, s: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n points uniform in [0, 300)², classes uniform in 0..s-1 (int8, as ``common.read_slide`` gives them)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 300, size=(n, 2)), rng.integers(0, s, n).astype(np.int8)


def keys(full: bool) -> set[tuple[str, int, int]]:
    modules = CLASS_MODULES if full else ("domain", "image")
    return {(m, t, p) for m in modules for t in range(4) for p in (0, 1)} | {("codomain", -1, p) for p in (0, 1)}


def assert_valid(packs: dict, full: bool) -> None:
    """Every key is there, and every value is (n, 2) float64 with finite births >= 0 and death > birth."""
    assert set(packs) == keys(full)
    for key, bars in packs.items():
        assert bars.dtype == np.float64 and bars.ndim == 2 and bars.shape[1] == 2, key
        assert np.all(np.isfinite(bars[:, 0])) and np.all(bars[:, 0] >= 0), key
        assert np.all(bars[:, 1] > bars[:, 0]), key


def sort_bars(bars: np.ndarray) -> np.ndarray:
    return bars[np.lexsort((bars[:, 1], bars[:, 0]))]


def same_diagram(a: np.ndarray, b: np.ndarray, rtol: float = 1e-9) -> bool:
    """Same number of bars and of essential bars, and bottleneck distance within ``rtol`` of the scale.

    From chalc-scaling ``8f370c6:report/port/validate.py``. Comparing sorted
    rows breaks on ties, where equal values can pair differently; the
    bottleneck distance does not.
    """
    import gudhi

    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    if np.isinf(a[:, 1]).sum() != np.isinf(b[:, 1]).sum():
        return False
    finite = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    scale = max(1.0, float(np.max(np.abs(finite))))
    return bool(gudhi.bottleneck_distance(a, b) <= rtol * scale)


def chalc_sixpacks(xy: np.ndarray, cls: np.ndarray) -> dict[tuple[str, int, int], np.ndarray]:
    """chalc 15.0.1's six-pack of (class t -> all cells) for each class present, radii squared, keyed like ours.

    chalc wants colours 0..s-1, so the classes present map to their rank.
    """
    import chalc
    from chalc.morphisms import SubChromaticInclusion

    present = np.unique(cls)
    f = chalc.chromatic.delaunay_cech(np.ascontiguousarray(xy.T), np.searchsorted(present, cls).tolist(), 0)
    out = {}
    for rank, t in enumerate(present.tolist()):
        pack = SubChromaticInclusion(f, [rank]).sixpack(1)
        for name in NAMES:
            for p in (0, 1):
                bars = np.asarray(getattr(pack, name).intervals(p), dtype=np.float64).reshape(-1, 2) ** 2
                out[name, t, p] = bars[bars[:, 1] > bars[:, 0]]
    return out


def assert_matches_chalc(xy: np.ndarray, cls: np.ndarray) -> None:
    ours = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(ours, full=True)
    theirs = chalc_sixpacks(xy, cls)
    assert {t for _, t, _ in theirs} == set(np.unique(cls).tolist())
    bad = [
        f"{name} class {t} H{p}: {len(ours[name, -1 if name == 'codomain' else t, p])} vs {len(bars)} bars"
        for (name, t, p), bars in theirs.items()
        if not same_diagram(ours[name, -1 if name == "codomain" else t, p], bars)
    ]
    assert not bad


def betti_at(bars: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """The number of bars [b, d) alive at each t of ``ts``: exact counts, no grid."""
    b = np.sort(bars[:, 0])
    d = np.sort(bars[:, 1])
    return np.searchsorted(b, ts, side="right") - np.searchsorted(d, ts, side="right")


def identity_violation(packs: dict, t: int) -> int:
    """The largest violation of dom = ker + im, cod = im + cok and rel_p = cok_p + ker_{p-1} at any bar endpoint.

    The Betti numbers are right-continuous steps that change only at bar
    endpoints, so the endpoints (and 0) are every value that matters.
    """
    worst = 0
    for p in (0, 1):
        bars = {name: packs[name, t, p] for name in CLASS_MODULES} | {"codomain": packs["codomain", -1, p]}
        ker_prev = packs["kernel", t, 0] if p == 1 else np.empty((0, 2))
        ends = np.concatenate([a.ravel() for a in bars.values()] + [ker_prev.ravel(), np.zeros(1)])
        ts = np.unique(ends[np.isfinite(ends)])
        B = {name: betti_at(a, ts) for name, a in bars.items()}
        K = betti_at(ker_prev, ts)
        worst = max(
            worst,
            int(np.abs(B["domain"] - B["kernel"] - B["image"]).max()),
            int(np.abs(B["codomain"] - B["image"] - B["cokernel"]).max()),
            int(np.abs(B["relative"] - B["cokernel"] - K).max()),
        )
    return worst


def alpha_diagrams(xy: np.ndarray) -> dict[int, np.ndarray]:
    import topocyte as tc

    return tc.ph.compute_persistent_homology(xy, filtration="alpha", max_dim=1).diagrams


@pytest.mark.parametrize("n,s,seed", CLOUDS)
def test_matches_chalc(n, s, seed):
    assert_matches_chalc(*cloud(n, s, seed))


@pytest.mark.parametrize("n,s,seed", CLOUDS)
def test_betti_identities(n, s, seed):
    xy, cls = cloud(n, s, seed)
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(packs, full=True)
    assert [identity_violation(packs, t) for t in np.unique(cls).tolist()] == [0] * s


def hex_lattice(side: int = 12, spacing: float = 10.0) -> np.ndarray:
    """A hexagonal lattice: many cocircular points, whose floating-point balls tie to within ulps."""
    return np.array([(i + 0.5 * (j % 2), j * np.sqrt(3) / 2) for i in range(side) for j in range(side)]) * spacing


def test_dim2_equals_dim3():
    # On the lattice, floating-point Welzl puts some tetrahedra an ulp below a facet; the facet-max pass
    # restores the order, without which phimaker refuses the dimension-3 boundary matrix.
    lattice = (hex_lattice(), np.random.default_rng(5).integers(0, 3, 144).astype(np.int8))
    for xy, cls in [cloud(*c) for c in CLOUDS[2:]] + [lattice]:
        dim2 = chromatic.ball_sixpacks(xy, cls, full=True)
        dim3 = chromatic.ball_sixpacks(xy, cls, full=True, _max_face_dim=3)
        assert set(dim2) == set(dim3)
        for key in dim2:
            assert np.array_equal(sort_bars(dim2[key]), sort_bars(dim3[key])), key


def test_codomain_is_alpha_ph():
    xy, cls = cloud(300, 3, 11)
    packs = chromatic.ball_sixpacks(xy, cls)
    assert_valid(packs, full=False)
    alpha = alpha_diagrams(xy)
    for p in (0, 1):
        assert same_diagram(packs["codomain", -1, p], alpha[p]), p
    assert np.isinf(packs["codomain", -1, 0][:, 1]).sum() == 1


def test_absent_class_is_empty():
    xy, cls = cloud(120, 2, 7)
    cls = (2 * cls).astype(np.int8)  # classes {0, 2}
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(packs, full=True)
    for name in CLASS_MODULES:
        for t in (1, 3):
            for p in (0, 1):
                assert packs[name, t, p].shape == (0, 2), (name, t, p)
    # The classes present keep their codes: each one's domain is the alpha PH of its own cells.
    for t in (0, 2):
        alpha = alpha_diagrams(xy[cls == t])
        for p in (0, 1):
            assert same_diagram(packs["domain", t, p], alpha[p]), (t, p)
        assert identity_violation(packs, t) == 0


def test_single_class_ball():
    xy, _ = cloud(80, 1, 5)
    cls = np.full(80, 3, dtype=np.int8)
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(packs, full=True)
    for p in (0, 1):
        codomain = sort_bars(packs["codomain", -1, p])
        assert np.array_equal(sort_bars(packs["domain", 3, p]), codomain)
        assert np.array_equal(sort_bars(packs["image", 3, p]), codomain)
        for name in ("kernel", "cokernel", "relative"):
            assert packs[name, 3, p].shape == (0, 2)
        for t in (0, 1, 2):
            for name in CLASS_MODULES:
                assert packs[name, t, p].shape == (0, 2)
    assert len(packs["codomain", -1, 0]) == 80 and np.isinf(packs["codomain", -1, 0][:, 1]).sum() == 1


def test_coincident_points_match_chalc():
    rng = np.random.default_rng(12)
    base = rng.uniform(0, 300, size=(150, 2))
    base_cls = rng.integers(0, 4, 150)
    pick = rng.choice(150, size=50, replace=False)
    copy_cls = (base_cls[pick] + rng.integers(1, 4, 50)) % 4  # never the original's class
    xy = np.vstack([base, base[pick]])
    cls = np.concatenate([base_cls, copy_cls]).astype(np.int8)
    assert len(np.unique(xy, axis=0)) == 150
    assert_matches_chalc(xy, cls)


def test_same_class_duplicates_are_merged():
    xy, cls = cloud(120, 4, 13)
    rng = np.random.default_rng(13)
    pick = rng.choice(120, size=40, replace=False)
    rows = np.concatenate([np.arange(120), pick, pick[:10]])  # 30 cells twice, 10 three times
    rng.shuffle(rows)
    with_duplicates = chromatic.ball_sixpacks(xy[rows], cls[rows], full=True)
    without = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(with_duplicates, full=True)
    for key in without:
        assert np.array_equal(sort_bars(with_duplicates[key]), sort_bars(without[key])), key


def test_tiny_and_collinear_balls():
    # One cell: an essential H0 bar in its class's domain and image and in the codomain, nothing finite, no edges.
    xy, cls = np.array([[10.0, 20.0]]), np.array([2], dtype=np.int8)
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(packs, full=True)
    assert all(np.all(np.isinf(bars[:, 1])) for bars in packs.values())
    for key in (("codomain", -1, 0), ("domain", 2, 0), ("image", 2, 0)):
        assert packs[key].tolist() == [[0.0, np.inf]], key
    counts, n_edges = chromatic.pair_counts(xy, cls)
    assert counts.shape == (len(PAIRS),) and counts.sum() == 0 and n_edges == 0

    # Two cells of two classes, 10 px apart: they join at (10 / 2)² px².
    xy, cls = np.array([[0.0, 0.0], [6.0, 8.0]]), np.array([0, 1], dtype=np.int8)
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(packs, full=True)
    codomain = sort_bars(packs["codomain", -1, 0])
    assert codomain[:, 0].tolist() == [0.0, 0.0] and codomain[0, 1] == pytest.approx(25.0, rel=1e-15)
    assert np.isinf(codomain[1, 1]) and packs["codomain", -1, 1].shape == (0, 2)
    assert [identity_violation(packs, t) for t in (0, 1)] == [0, 0]
    counts, n_edges = chromatic.pair_counts(xy, cls)
    assert n_edges == 1 and counts.tolist() == [int(pair == (0, 1)) for pair in PAIRS]

    # Five collinear cells of three classes (exactly collinear: integer coordinates on one line).
    k = np.array([0.0, 3.0, 7.0, 12.0, 20.0])
    xy, cls = np.column_stack([3 * k + 5, 4 * k + 1]), np.array([0, 1, 0, 2, 1], dtype=np.int8)
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    assert_valid(packs, full=True)
    codomain = packs["codomain", -1, 0]
    assert len(codomain) == 5 and np.isinf(codomain[:, 1]).sum() == 1
    assert packs["codomain", -1, 1].shape == (0, 2)
    assert [identity_violation(packs, t) for t in (0, 1, 2)] == [0, 0, 0]
    counts, n_edges = chromatic.pair_counts(xy, cls)
    assert n_edges == 4 and counts.sum() == 4


def test_dense_ball():
    chromatic.ball_sixpacks(*cloud(40, 4, 99), full=True)  # compile or load the numba kernels first
    xy, cls = cloud(700, 4, 21)
    start = time.perf_counter()
    packs = chromatic.ball_sixpacks(xy, cls, full=True)
    seconds = time.perf_counter() - start
    assert_valid(packs, full=True)
    assert [identity_violation(packs, t) for t in range(4)] == [0, 0, 0, 0]
    assert seconds < 5, seconds


# A fixed 6-point cloud in general position, and the same cloud with cells 2 and 5 at one position.
SIX = np.array([[0.0, 0.0], [10.0, 1.0], [4.0, 9.0], [13.0, 11.0], [-3.0, 8.0], [6.0, 4.0]])
SIX_COINCIDENT = SIX.copy()
SIX_COINCIDENT[5] = SIX[2]


@pytest.mark.parametrize("xy", [SIX, SIX_COINCIDENT], ids=["distinct", "coincident"])
def test_pair_expectation_exhaustive(xy):
    cls = np.array([0, 0, 1, 1, 2, 3], dtype=np.int8)
    labellings = sorted(set(itertools.permutations(cls.tolist())))
    assert len(labellings) == 180
    totals = np.zeros(len(PAIRS), dtype=np.int64)
    edge_counts = set()
    for labels in labellings:
        counts, n_edges = chromatic.pair_counts(xy, np.array(labels, dtype=np.int8))
        assert counts.sum() == n_edges
        totals += counts
        edge_counts.add(n_edges)
    (n_edges,) = edge_counts  # the edge set does not depend on the labels
    class_counts = np.bincount(cls, minlength=4)
    expected = chromatic.pair_expectation(class_counts, n_edges)
    assert expected.shape == (len(PAIRS),) and expected.dtype == np.float64
    for k, (a, b) in enumerate(PAIRS):
        mean = Fraction(int(totals[k]), len(labellings))
        same = comb(int(class_counts[a]), 2) if a == b else int(class_counts[a] * class_counts[b])
        assert mean == Fraction(n_edges * same, comb(len(cls), 2)), (a, b)
        assert expected[k] == float(mean), (a, b)


def test_pair_counts_sum():
    from scipy.spatial import ConvexHull, Delaunay

    xy, cls = cloud(200, 4, 31)
    counts, n_edges = chromatic.pair_counts(xy, cls)
    assert counts.dtype == np.int64 and counts.shape == (len(PAIRS),)
    assert counts.sum() == n_edges == 3 * 200 - 3 - len(ConvexHull(xy).vertices)  # Euler, general position

    # Coincident cells: every pair across a Delaunay edge of the positions, and every pair at one position.
    rows = np.concatenate([np.arange(200), np.arange(30), np.arange(5)])  # 5 positions hold 3 cells, 25 hold 2
    rng = np.random.default_rng(31)
    rng.shuffle(rows)
    shuffled = rng.integers(0, 4, len(rows)).astype(np.int8)
    counts, n_edges = chromatic.pair_counts(xy[rows], shuffled)
    m = np.bincount(rows, minlength=200)
    triangles = Delaunay(xy).simplices
    edges = np.unique(np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [0, 2]], triangles[:, [1, 2]]]), axis=1), axis=0)
    assert counts.sum() == n_edges == int(np.sum(m[edges[:, 0]] * m[edges[:, 1]]) + np.sum(m * (m - 1) // 2))
    adjacent = set(map(tuple, edges.tolist()))
    brute = np.zeros(len(PAIRS), dtype=np.int64)  # every pair of cells, by brute force
    for i, j in itertools.combinations(range(len(rows)), 2):
        u, v = sorted((int(rows[i]), int(rows[j])))
        if u == v or (u, v) in adjacent:
            brute[PAIRS.index(tuple(sorted((int(shuffled[i]), int(shuffled[j])))))] += 1
    assert counts.tolist() == brute.tolist()


def test_pair_expectation_is_correctly_rounded():
    rng = np.random.default_rng(41)
    for _ in range(300):
        class_counts = rng.integers(0, 300, 4)
        n, n_edges = int(class_counts.sum()), int(rng.integers(0, 900))
        expected = chromatic.pair_expectation(class_counts, n_edges)
        for k, (a, b) in enumerate(PAIRS):
            same = comb(int(class_counts[a]), 2) if a == b else int(class_counts[a] * class_counts[b])
            assert expected[k] == float(Fraction(n_edges * same, comb(n, 2))), (class_counts, n_edges, a, b)
    assert chromatic.pair_expectation(np.array([1, 0, 0, 0]), 0).tolist() == [0.0] * len(PAIRS)


def test_bad_input_raises():
    xy, cls = cloud(10, 2, 0)
    for bad_cls in (np.full(10, 4), np.full(10, -1), cls[:9]):
        for f in (chromatic.ball_sixpacks, chromatic.pair_counts):
            with pytest.raises(ValueError):
                f(xy, bad_cls)
    with pytest.raises(ValueError, match="planar"):
        chromatic.pair_counts(np.zeros((10, 3)), cls)
    with pytest.raises(ValueError, match="at least one cell"):
        chromatic.ball_sixpacks(np.empty((0, 2)), np.empty(0, dtype=np.int8))
    counts, n_edges = chromatic.pair_counts(np.empty((0, 2)), np.empty(0, dtype=np.int8))
    assert counts.tolist() == [0] * len(PAIRS) and n_edges == 0
