"""Tests of local_colocalization: the hypergeometric null by Monte Carlo, self-exclusion, small clouds, cells at one
position, background cells, the frozen fit, shrinkage, the cohort sums and the cloud summary."""

import numpy as np
import pytest
from scipy import sparse

from local_colocalization import (
    CellColocalizationResult,
    LocalColocalizationExtractor,
    _counts,
    dersimonian_laird,
    encode_labels,
    excess_sums,
    neighbour_indices,
    shrink,
)


def irregular_cloud(rng: np.random.Generator, n: int = 120, n_types: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A dense cluster, a ring around a hole and scattered cells, so the neighbours' distances vary widely."""
    cluster = rng.normal([0.0, 0.0], 1.0, size=(n // 2, 2))
    angle = rng.uniform(0, 2 * np.pi, n // 4)
    radius = rng.uniform(4, 6, n // 4)
    ring = np.c_[radius * np.cos(angle), radius * np.sin(angle)] + [8.0, 0.0]
    scatter = rng.uniform(-6, 14, size=(n - n // 2 - n // 4, 2))
    return np.vstack([cluster, ring, scatter]), rng.integers(0, n_types, n)


def test_neighbour_indices_are_the_k_nearest_others():
    xy = np.random.default_rng(1).uniform(0, 50, size=(300, 2))
    distances = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert np.array_equal(neighbour_indices(xy, 7), np.argsort(distances, axis=1)[:, :7])
    assert neighbour_indices(xy[:4], 7).shape == (4, 3)  # at most the N - 1 others
    assert neighbour_indices(xy[:1], 7).shape == (1, 0)


def test_counts_are_hypergeometric_under_random_relabelling_and_y_is_unbiased():
    rng = np.random.default_rng(7)
    n, n_types, k, draws = 120, 3, 10, 200_000
    xy, labels = irregular_cloud(rng, n, n_types)
    counts = _counts(xy, labels, n_types, k)
    nbrs = neighbour_indices(xy, k)
    for i in [int(np.flatnonzero(labels == b)[0]) for b in range(n_types)]:  # one focal cell of each type
        others = np.delete(np.arange(n), i)
        shuffled = rng.permuted(np.tile(labels[others].astype(np.int8), (draws, 1)), axis=1)
        picked = shuffled[:, np.searchsorted(others, nbrs[i])]  # the labels the relabelling gives i's neighbours
        o = np.stack([(picked == a).sum(axis=1) for a in range(n_types)], axis=1).astype(np.float64)
        e, v, q = counts.expected[i], counts.variance[i], counts.q[i]
        assert np.all(np.abs(o.mean(axis=0) - e) < 1e-2), (i, o.mean(axis=0), e)
        assert np.all(np.abs(o.var(axis=0) / v - 1) < 0.05), (i, o.var(axis=0), v)
        for room in (k * (1 - q), k * q):  # the gain and the loss scale: unclipped Y has null mean 0 on both
            assert np.all(np.abs(((o - e) / room).mean(axis=0)) < 2e-2)


def test_the_extractor_gives_y_and_s2_from_the_same_moments():
    rng = np.random.default_rng(3)
    xy, labels = irregular_cloud(rng)
    for mode, room_of in (("gain", lambda k, q: k * (1 - q)), ("loss", lambda k, q: k * q)):
        extractor = LocalColocalizationExtractor(n_neighbors=10, scale_mode=mode, vocabulary=[0, 1, 2])
        result = extractor.fit_transform([(xy, labels)])[0]
        counts = _counts(xy, labels, 3, 10)
        room = room_of(10.0, counts.q)
        valid = result.valid_mask
        np.testing.assert_allclose(result.Y_raw[valid], ((counts.observed - counts.expected) / room)[valid])
        np.testing.assert_allclose(result.s2[valid], (counts.variance / room**2)[valid])
        assert np.all(np.isnan(result.Y_raw[~valid])) and np.all(np.isinf(result.s2[~valid]))
        assert np.all(np.isfinite(result.Y_shrunk)) and np.all(result.degree == 10)


def test_a_singleton_type_has_no_homotypic_estimate():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 30, size=(40, 2))
    labels = np.r_[0, np.ones(20, dtype=int), np.full(19, 2)]  # type 0 has one cell, cell 0
    extractor = LocalColocalizationExtractor(n_neighbors=8, vocabulary=[0, 1, 2]).fit([(xy, labels)])
    result = extractor.transform([(xy, labels)])[0]
    counts = _counts(xy, labels, 3, 8)
    assert counts.q[0, 0] == 0  # n' = n_A - 1 = 0: the focal cell leaves its own type
    assert not result.valid_mask[0, 0] and np.isnan(result.Y_raw[0, 0])
    assert np.isfinite(result.Y_shrunk[0, 0]) and result.homotypic_excess_shrunk[0] == result.Y_shrunk[0, 0]


def test_tiny_clouds_and_neighbourhoods_of_the_whole_cloud_are_invalid_but_finite():
    extractor = LocalColocalizationExtractor(n_neighbors=5, vocabulary=["a", "b"]).fit([(np.zeros((0, 2)), [])])
    for n in (0, 1, 2, 4, 6):  # 6 cells and k = 5: every other cell is a neighbour, so v = 0
        xy = np.random.default_rng(n).uniform(0, 10, size=(n, 2))
        result = extractor.transform([(xy, (["a", "b"] * 3)[:n])])[0]
        assert result.Y_shrunk.shape == (n, 2) and result.degree.shape == (n,)
        assert not result.valid_mask.any() and np.all(np.isfinite(result.Y_shrunk))
        assert np.all(result.degree == max(min(5, n - 1), 0))


def test_cells_at_one_position_are_neighbours_and_never_their_own():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [9.0, 0.0], [9.0, 1.0]])
    nbrs = neighbour_indices(xy, 1)
    assert nbrs[0, 0] == 1 and nbrs[1, 0] == 0
    assert not np.any(neighbour_indices(xy, 4) == np.arange(5)[:, None])
    counts = _counts(xy, np.array([0, 0, 1, 1, 1]), 2, 1)
    assert counts.observed[0, 0] == 1 and counts.observed[1, 0] == 1  # the twin counts, the cell itself does not


def test_background_cells_take_neighbour_places_and_dilute_q():
    rng = np.random.default_rng(9)
    xy = rng.uniform(0, 40, size=(60, 2))
    labels = np.array(["x"] * 25 + ["y"] * 25 + ["unseen"] * 10, dtype=object)
    labels[[3, 30]] = None
    extractor = LocalColocalizationExtractor(n_neighbors=6, vocabulary=["x", "y"]).fit([(xy, labels)])
    result = extractor.transform([(xy, labels)])[0]
    codes = encode_labels(labels, ["x", "y"])
    assert (codes == -1).sum() == 12
    counts = _counts(xy, codes, 2, 6)
    # q counts every cell in N - 1, background included: 24 x (cell 3 is None) and 24 y (cell 30), so an x cell
    # sees 23 other x and 24 y among 59 others
    np.testing.assert_allclose(counts.q[0], [23 / 59, 24 / 59])
    assert np.all(counts.observed.sum(axis=1) <= 6) and np.all(result.degree == 6)
    assert np.all(np.isnan(result.homotypic_excess_shrunk[codes == -1]))
    assert np.all(np.isfinite(result.Y_shrunk))


def segregated(n_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Two blocks side by side, one of each type: every cell's neighbours are mostly its own type."""
    g = np.arange(n_side, dtype=np.float64)
    block = np.array([(x, y) for x in g for y in g])
    return np.vstack([block, block + [n_side + 5.0, 0.0]]), np.repeat([0, 1], len(block))


def checkerboard(n_side: int) -> tuple[np.ndarray, np.ndarray]:
    """A lattice of alternating types, slightly jittered: every cell's nearest neighbours are the other type."""
    g = np.arange(n_side, dtype=np.float64)
    xy = np.array([(x, y) for x in g for y in g]) + np.random.default_rng(2).normal(0, 0.01, (n_side * n_side, 2))
    return xy, np.array([(int(x) + int(y)) % 2 for x in g for y in g])


def test_fit_freezes_the_scale_and_transform_keeps_it_on_clouds_of_the_opposite_sign():
    train, test = segregated(8), checkerboard(10)
    extractor = LocalColocalizationExtractor(n_neighbors=4, vocabulary=[0, 1]).fit([train])
    assert extractor.pair_excess_[0, 0] > 0 and extractor.gain_mask_[0, 0]  # homotypic clustering: gain
    assert LocalColocalizationExtractor(n_neighbors=4, vocabulary=[0, 1]).fit([test]).pair_excess_[0, 0] < 0
    result = extractor.transform([test])[0]
    counts = _counts(test[0], test[1], 2, 4)
    focal0 = (test[1] == 0) & result.valid_mask[:, 0]
    room = 4 * (1 - counts.q[focal0, 0])  # still the gain scale that training fixed
    np.testing.assert_allclose(result.Y_raw[focal0, 0], (counts.observed - counts.expected)[focal0, 0] / room)
    assert result.Y_raw[focal0, 0].mean() < 0  # avoidance, on the scale clustering fixed (corners see a diagonal)


def test_sparser_estimates_have_larger_s2_and_shrink_closer_to_the_prior():
    rng = np.random.default_rng(4)
    xy, labels = irregular_cloud(rng, 200)
    sparse_fit = LocalColocalizationExtractor(n_neighbors=2, scale_mode="gain", tau2=0.05, vocabulary=[0, 1, 2])
    dense_fit = LocalColocalizationExtractor(n_neighbors=40, scale_mode="gain", tau2=0.05, vocabulary=[0, 1, 2])
    few = sparse_fit.fit_transform([(xy, labels)])[0]
    many = dense_fit.fit_transform([(xy, labels)])[0]
    both = few.valid_mask & many.valid_mask
    assert both.any() and np.all(few.s2[both] > many.s2[both])
    y = np.array([0.6, 0.6])
    s2 = np.array([few.s2[both][0], many.s2[both][0]])
    out = shrink(y, s2, np.array([True, True]), np.array([0.1, 0.1]), np.array([0.05, 0.05]))
    assert abs(out[0] - 0.1) < abs(out[1] - 0.1) < abs(0.6 - 0.1)


def test_dersimonian_laird_follows_its_moments():
    y = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [7.0]])
    s2 = np.ones((6, 1))
    valid = np.ones((6, 1), dtype=bool)
    groups = np.array([0, 0, 0, 0, 0, 1])
    tau2 = dersimonian_laird(y, s2, valid, groups, 2, min_cells=5)
    # group 0: w = 1, mean 2, Q = 10, C = 5 - 1 = 4, tau^2 = (10 - 4) / 4; group 1 has one cell
    np.testing.assert_allclose(tau2[:, 0], [1.5, 0.0])
    assert dersimonian_laird(y, s2, valid, np.zeros(6, dtype=int), 1, min_cells=7)[0, 0] == 0


def test_excess_sums_for_many_k_equal_the_graph_counts():
    rng = np.random.default_rng(12)
    xy = rng.uniform(0, 60, size=(250, 2))
    codes = rng.integers(-1, 3, 250)  # background cells too
    ks = [2, 5, 9, 300]
    pair, target = excess_sums(xy, codes, 3, ks)
    for j, k in enumerate(ks):
        counts = _counts(xy, codes, 3, k)
        excess = counts.observed - counts.expected
        groups = np.where(codes >= 0, codes, 3)
        member = sparse.csr_matrix((np.ones(250), (groups, np.arange(250))), shape=(4, 250))
        by_focal = member @ excess
        np.testing.assert_allclose(pair[j], by_focal[:3], atol=1e-9)
        np.testing.assert_allclose(target[j], by_focal.sum(axis=0), atol=1e-9)
    np.testing.assert_allclose(pair[3], 0, atol=1e-9)  # k beyond the cloud: o = e for every cell


def test_summarize_cloud_pools_each_pair_by_precision():
    extractor = LocalColocalizationExtractor.fitted(["a", "b"], np.zeros((2, 2)), np.zeros(2), n_neighbors=3)
    y = np.array([[1.0, 0.5], [3.0, np.nan], [2.0, -1.0], [9.0, 9.0]])
    s2 = np.array([[1.0, 2.0], [0.5, np.inf], [1.0, 1.0], [1.0, 1.0]])
    valid = np.isfinite(y)
    result = CellColocalizationResult(np.nan_to_num(y), y, s2, valid, np.full(4, 3, np.int32), np.zeros(4))
    summary = extractor.summarize_cloud(result, ["a", "a", "b", None])  # the background cell joins no pair
    np.testing.assert_allclose(summary["pooled_excess"][0], [(1 + 3 * 2) / 3, 0.5])
    np.testing.assert_allclose(summary["pooled_variance"][0], [1 / 3, 2.0])
    np.testing.assert_allclose(summary["pooled_excess"][1], [2.0, -1.0])
    assert summary["quantiles"].shape == (3, 2, 2) and summary["n_cells"].tolist() == [2, 1]


def test_labels_encode_strings_integers_and_nulls():
    assert encode_labels(["a", None, "b", float("nan"), "z"], ["a", "b"]).tolist() == [0, -1, 1, -1, -1]
    assert encode_labels(np.array([2, 0, -1, 5]), [0, 1, 2]).tolist() == [2, 0, -1, -1]
    assert encode_labels(np.array([1.0, np.nan]), [0, 1]).tolist() == [1, -1]
    fitted = LocalColocalizationExtractor(n_neighbors=2).fit([(np.zeros((3, 2)), [2, -1, 0])])
    assert fitted.vocabulary_ == [0, 2]


def test_the_extractor_refuses_bad_settings_and_an_unfitted_transform():
    with pytest.raises(ValueError, match="scale_mode"):
        LocalColocalizationExtractor(scale_mode="both")
    with pytest.raises(ValueError, match="tau2"):
        LocalColocalizationExtractor(tau2=-1.0)
    with pytest.raises(ValueError, match="fit"):
        LocalColocalizationExtractor().transform([(np.zeros((1, 2)), [0])])
