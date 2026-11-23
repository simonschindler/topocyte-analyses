"""Tests of the e-means neighbourhood study on small synthetic inputs."""

import numpy as np
import pytest

import common as c
import features


def test_ball_counts_count_the_closed_ball_with_the_cell_and_threshold_from_one():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [100.0, 0.0]])
    classes = np.array([0, 2, 2, 1])
    rho = np.array([1.0, 5.0, 9.0, 9.0])
    thresholds = np.array([1.0, 4.0, 8.0])  # threshold 0 is rho's minimum: every cell kept
    counts = features.ball_counts(xy, classes, rho, thresholds, radius_px=5.0)
    # columns: epithelial, stromal, immune, other, kept at threshold 1 (rho >= 4), kept at threshold 2 (rho >= 8)
    assert counts.tolist() == [[1, 0, 2, 0, 2, 1], [1, 0, 2, 0, 2, 1], [1, 0, 2, 0, 2, 1], [0, 1, 0, 0, 1, 1]]
    features.check_counts(counts, rho, thresholds, "synthetic")


def test_check_counts_refuses_impossible_balls():
    rho, thresholds = np.array([5.0]), np.array([1.0, 4.0])
    with pytest.raises(RuntimeError, match="own cell"):
        features.check_counts(np.array([[0, 0, 0, 0, 0]]), rho, thresholds, "S")
    with pytest.raises(RuntimeError, match="missing from its own ball"):
        features.check_counts(np.array([[1, 0, 0, 0, 0]]), rho, thresholds, "S")
    with pytest.raises(RuntimeError, match="rises"):
        features.check_counts(np.array([[1, 0, 0, 0, 2]]), rho, thresholds, "S")


def test_count_columns_are_four_classes_and_nine_thresholds():
    assert len(c.COUNT_COLUMNS) == len(c.CLASSES) + c.N_THRESHOLDS - 1 == 13


def test_feature_matrix_z_scores_leaves_out_the_constant_column_and_weights_blocks():
    import cluster

    rng = np.random.default_rng(0)
    classes = rng.integers(1, 30, size=(500, 4))
    n_ball = classes.sum(axis=1)
    kept = np.sort(rng.integers(0, 1, size=(500, 9)) + (n_ball[:, None] * rng.random((500, 9))).astype(int), axis=1)[
        :, ::-1
    ]
    kept = np.maximum(kept, 1)
    counts = np.column_stack([classes, np.minimum(kept, n_ball[:, None])]).astype(np.uint16)
    X, built = cluster.feature_matrix(counts, "composition+density")
    assert built["left_out_constant"] == ["kept fraction, threshold 0"]
    assert X.shape == (500, 13)
    z = X / np.array(built["column_weights"], dtype=np.float32)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-4)
    assert built["column_weights"] == [0.5] * 4 + [1 / 3] * 9
    comp, built_comp = cluster.feature_matrix(counts, "composition")
    assert comp.shape == (500, 4) and built_comp["column_weights"] == [1.0] * 4


def test_gap_rules_follow_the_pilot():
    import cluster

    s = np.full(4, 0.01)
    assert cluster.tibshirani(np.array([0.0, 0.1, 0.105, 0.2]), s) == 2  # Gap(2) = 0.1 >= Gap(3) - s = 0.095
    assert cluster.tibshirani(np.array([0.0, -0.01, -0.02, -0.03]), s) == 1
    assert cluster.tibshirani(np.array([0.0, 0.1, 0.2, 0.3]), s) == 4  # never stops: k_max
    assert cluster.pilot_rule(np.array([0.0, 0.02, 0.01, 0.0]), s) == 1  # stops at 2 but gains only 0.02
    assert cluster.pilot_rule(np.array([0.0, 0.05, 0.04, 0.0]), s) == 2


def test_pack_unpack_round_trip():
    import cluster

    labels = np.array([0, 15, 3, 9, 7], dtype=np.uint8)
    assert cluster.unpack(cluster.pack(labels), 5).tolist() == labels.tolist()
    with pytest.raises(ValueError, match="below 16"):
        cluster.pack(np.array([16], dtype=np.uint8))


def test_verify_recounts_a_ball_as_features_counts_it():
    import verify_results as v

    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 3 * c.RADIUS_PX, size=(400, 2))
    classes = rng.integers(0, 4, size=400)
    rho = rng.uniform(0, 10, size=400)
    thresholds = np.linspace(0, 9, c.N_THRESHOLDS)
    counts = features.ball_counts(xy, classes, rho, thresholds, c.RADIUS_PX)
    for centre in (0, 17, 399):
        at, low, high = v.brute_ball(xy, classes, rho, thresholds, centre)
        assert at.tolist() == counts[centre].tolist() and np.all(low <= at) and np.all(at <= high)


def test_verify_label_check_allows_ties_but_not_a_farther_centroid():
    import verify_results as v

    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [2.0, 0.0]])
    own, differ, beyond = v.label_check(X, centroids, np.array([0, 1, 1]))  # row 1 is a tie
    assert own.tolist() == [0.0, 1.0, 0.0] and (differ, beyond) == (0, 0)
    own, differ, beyond = v.label_check(X, centroids, np.array([1, 0, 1]))  # row 0 is labelled far away
    assert (differ, beyond) == (1, 1)


def test_verify_moments_merge_chunks_as_one_pass():
    import verify_results as v

    x = np.random.default_rng(3).normal(5, 2, size=(1000, 3))
    m = v.Moments()
    for chunk in np.split(x, [100, 350, 999]):  # chunks of 100, 250, 649 and 1 rows
        m.add(chunk)
    assert np.allclose(m.mean, x.mean(axis=0), rtol=1e-12) and np.allclose(m.sd, x.std(axis=0), rtol=1e-12)


def test_check_kmeans_rebuilds_the_runs_rows_and_moments():
    import check_kmeans as ck
    import cluster

    rng = np.random.default_rng(4)
    classes = rng.integers(1, 30, size=(300, 4))
    n_ball = classes.sum(axis=1)
    kept = np.sort((n_ball[:, None] * rng.random((300, 9))).astype(int), axis=1)[:, ::-1]
    counts = np.column_stack([classes, np.maximum(kept, 1)]).astype(np.uint16)
    rows = np.array([0, 7, 150, 299])
    for name in c.SETS:
        X, built = cluster.feature_matrix(counts, name)
        assert np.array_equal(ck.rebuild(counts[rows], name, built), X[rows])
        mean, cov = ck.moments(counts, name, built)
        m, v = cluster.column_moments(X)
        assert np.allclose(mean, m, rtol=0, atol=1e-12) and np.allclose(cov, v, rtol=0, atol=1e-12)


def test_check_kmeans_takes_the_nearest_centroid():
    import check_kmeans as ck

    X = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]], dtype=np.float32)
    assert ck.nearest_sq(X, [[0.0, 0.0], [10.0, 1.0]]).tolist() == [0.0, 25.0, 1.0]


def test_reference_has_the_mean_and_covariance_even_when_singular():
    import cluster

    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])  # rank 2
    R = cluster.reference(mean, cov, 400_000, np.random.default_rng(1))
    m, v = cluster.column_moments(R)
    assert np.allclose(m, mean, atol=0.01) and np.allclose(v, cov, atol=0.02)
