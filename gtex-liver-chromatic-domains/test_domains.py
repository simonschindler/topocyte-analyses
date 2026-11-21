"""Unit tests of the rule for k in ``domains.py`` (design step 10), on synthetic clouds.

The gap tests use the controller's constructions (ruling R2): 4,000 rows and
20 columns, drawn from one ``default_rng(1)`` in the order elongated,
skewed, two clusters, three clusters. They run the binding settings (B = 20
reference draws, ``n_init=20``, minimum gain 0.03), so they take about a
minute. No topocyte and no cohort.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.cluster import KMeans

import domains

N = 4000  # rows of each R2 cloud


@pytest.fixture(scope="module")
def clouds() -> dict[str, np.ndarray]:
    """The R2 clouds, drawn from one ``default_rng(1)`` in this order; the first is the elongated test's cloud."""
    rng = np.random.default_rng(1)
    elongated = rng.normal(size=(N, 20)) * np.sqrt(1 / np.arange(1, 21))
    skewed = np.column_stack([np.exp(rng.normal(0, 0.6, N)), rng.normal(size=(N, 19)) * np.sqrt(1 / np.arange(2, 21))])
    two = rng.normal(size=(N, 20))
    two[:, 0] += np.where(rng.random(N) < 0.5, -2.0, 2.0)  # centres 4 SD apart
    three = rng.normal(size=(N, 20))
    three[:, :2] += np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])[rng.integers(0, 3, N)]
    return {"elongated": elongated, "skewed": skewed, "two": two, "three": three}


def test_gap_cluster_free_elongated():
    X = np.random.default_rng(1).normal(size=(4000, 20)) * np.sqrt(1 / np.arange(1, 21))
    result = domains.gap_choice(X, 6)
    assert result["k"] == 1
    assert result["gap"].shape == result["s"].shape == (6,)
    assert np.all(result["s"] > 0)


def test_gap_skewed_unimodal(clouds):
    """A lognormal first axis is skewed but has one mode: no clusters."""
    assert domains.gap_choice(clouds["skewed"], 6)["k"] == 1


def test_gap_two_and_three_clusters(clouds):
    assert domains.gap_choice(clouds["two"], 6)["k"] == 2
    assert domains.gap_choice(clouds["three"], 6)["k"] == 3


def test_prediction_strength_separated():
    """Three blobs 50 SD apart, split at random into train and test: every test cluster stays together."""
    rng = np.random.default_rng(5)
    centres = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    X = centres[rng.integers(0, 3, 900)] + rng.normal(size=(900, 2))
    train = rng.random(900) < 0.5
    assert domains.prediction_strength(X[train], X[~train], 3) == 1.0


def test_ps_from_labels_hand_example():
    assert domains.ps_from_labels(np.array([0, 0, 0, 1, 1]), np.array([5, 5, 6, 7, 7])) == pytest.approx(1 / 3)


def test_choose_k_min_of_halves_and_ps(monkeypatch):
    """k is the smaller half's choice, kept only if prediction strength reaches 0.8 in both directions."""
    A, B = np.full((10, 2), 3.0), np.full((10, 2), 2.0)  # each half's gap choice is its first entry
    directions = [(A, B), (B, A)]
    gap_calls, ps_calls, ps_values = [], [], []

    def fake_gap(X, k_max, **kwargs):
        gap_calls.append((X, k_max, kwargs))
        return {"k": int(X[0, 0]), "gap": np.zeros(k_max), "s": np.zeros(k_max)}

    def fake_ps(train, test, k, **kwargs):
        ps_calls.append((train, test, k, kwargs))
        return ps_values[len(ps_calls) - 1]

    monkeypatch.setattr(domains, "gap_choice", fake_gap)
    monkeypatch.setattr(domains, "prediction_strength", fake_ps)

    ps_values[:] = [0.95, 0.85]
    result = domains.choose_k(A, B, directions, 6, seed=3)
    assert (result["k"], result["k_A"], result["k_B"], result["ps"]) == (2, 3, 2, [0.95, 0.85])
    assert result["gap_A"]["k"] == 3 and result["gap_B"]["k"] == 2
    assert [(X is half, k_max, kw) for (X, k_max, kw), half in zip(gap_calls, (A, B))] == [(True, 6, {"seed": 3})] * 2
    assert [(train is t, test is u, k, kw) for (train, test, k, kw), (t, u) in zip(ps_calls, directions)] == [
        (True, True, 2, {"seed": 3})
    ] * 2

    ps_calls.clear()
    ps_values[:] = [0.95, 0.79]  # below 0.8 in one direction
    result = domains.choose_k(A, B, directions, 6)
    assert (result["k"], result["k_A"], result["k_B"], result["ps"]) == (1, 3, 2, [0.95, 0.79])

    ps_calls.clear()
    ps_values[:] = [0.8, 0.8]  # the bound itself confirms
    assert domains.choose_k(A, B, directions, 6)["k"] == 2

    ps_calls.clear()
    C = np.full((10, 2), 1.0)  # one half finds no clusters: nothing to confirm
    result = domains.choose_k(C, B, [(C, B), (B, C)], 6)
    assert (result["k"], result["k_A"], result["k_B"], result["ps"]) == (1, 1, 2, [])
    assert ps_calls == []

    with pytest.raises(ValueError, match="both directions"):
        domains.choose_k(A, B, directions[:1], 6)


@pytest.mark.parametrize(
    ("gap", "s", "k"),
    [
        # The four cases of the corrected rule (2026-11-21).
        ([0.0, -0.010, 0.020, 0.045, 0.040, 0.035], [0.004] * 6, 1),  # falls at 2, though later gains reach 0.03
        ([0.0, 0.020, 0.060, 0.050, 0.040, 0.030], [0.004] * 6, 3),  # rises to a peak at 3, gain 0.06
        ([0.0, 0.025, 0.015, 0.010, 0.005, 0.000], [0.004] * 6, 1),  # peaks at 2, gain 0.025 under the floor
        ([0.0, 0.010, 0.030, 0.050, 0.070, 0.090], [0.001] * 6, 6),  # never stops, gain 0.09 at k_max
        # The stop and the floor in detail.
        ([0.0, 0.003, 0.004, 0.005, 0.003, 0.002], [0.004] * 6, 1),  # noise: 0 >= 0.003 - 0.004 stops at 1
        ([0.0, 0.044, 0.040, 0.036, 0.030, 0.025], [0.004] * 6, 2),  # stops at 2 with a gain of 0.044
        ([0.0, 0.03, 0.01, 0.0, 0.0, 0.0], [0.004] * 6, 2),  # a gain of exactly 0.03 counts
        ([0.0, 0.025, 0.027, 0.040, 0.035, 0.030], [0.004] * 6, 1),  # stopped at 2; the later 0.040 does not count
        ([0.0, 0.050, 0.055, 0.050, 0.045, 0.040], [0.006] * 6, 2),  # 0.050 >= 0.055 - 0.006
        ([0.0, 0.050, 0.055, 0.050, 0.045, 0.040], [0.002] * 6, 3),  # 0.050 < 0.055 - 0.002
        ([0.0, 0.050, 0.056, 0.050, 0.045, 0.040], [0.001, 0.001, 0.008, 0.001, 0.001, 0.001], 2),  # s of k + 1, not k
        ([0.0, 0.005, 0.010, 0.015, 0.020, 0.029], [0.001] * 6, 1),  # never stops, but 0.029 at k_max
        ([0.5, 0.520, 0.510, 0.500, 0.490, 0.480], [0.004] * 6, 1),  # gains are against Gap(1), not against 0
        ([-0.3, -0.25, -0.27, -0.28, -0.29, -0.30], [0.004] * 6, 2),  # a negative Gap(1)
        ([0.0, 0.04], [0.004] * 2, 2),  # k_max = 2: no stop at 1, gain 0.04
        ([0.0, 0.025], [0.004] * 2, 1),  # k_max = 2: no stop at 1, gain 0.025
        ([0.0, 0.003], [0.004] * 2, 1),  # k_max = 2: stops at 1
        # The pre-flight's level-1 skewed half A under the first version (k = 5 there): the gap falls at 2.
        (
            [0.0, -0.0004, -0.0023, -0.0013, 0.0419, 0.0432, 0.0407, 0.0433, 0.0501, 0.0477],
            [0.0059, 0.0050, 0.0049, 0.0050, 0.0054, 0.0049, 0.0049, 0.0044, 0.0047, 0.0050],
            1,
        ),
    ],
)
def test_stop_rule(gap, s, k):
    """Tibshirani's stopping rule from k = 1 on given curves, then the 0.03 floor on the gain of the k it stops at."""
    assert domains.stop_rule(np.array(gap), np.array(s), domains.MIN_GAIN) == k


def test_stop_rule_refuses_mismatched_curves():
    with pytest.raises(ValueError):
        domains.stop_rule(np.zeros(6), np.zeros(5), 0.03)
    with pytest.raises(ValueError):
        domains.stop_rule(np.zeros(0), np.zeros(0), 0.03)


def test_binding_settings():
    """The rule's settings (design step 10, the minimum gain as raised on 2026-11-21) are gap_choice's defaults."""
    assert (domains.N_REFERENCES, domains.MIN_GAIN, domains.PS_MIN, domains.N_INIT) == (20, 0.03, 0.8, 20)
    parameters = inspect.signature(domains.gap_choice).parameters.values()
    assert {p.name: p.default for p in parameters if p.default is not p.empty} == {"B": 20, "min_gain": 0.03, "seed": 0}


def test_gap_matches_definition():
    """Gap(k) and s_k from the definition, on a small cloud with B = 4.

    W_1 is the total sum of squares about the column means and W_k the k-means
    inertia; the references come from one default_rng(seed), in order, with
    the cloud's mean and covariance; the SD has ddof 0.
    """
    X = np.random.default_rng(3).normal(size=(60, 3)) * [3.0, 1.0, 0.5]
    B, seed = 4, 7

    def log_w(Y: np.ndarray, k: int) -> float:
        if k == 1:
            return float(np.log(((Y - Y.mean(axis=0)) ** 2).sum()))
        return float(np.log(KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Y).inertia_))

    rng = np.random.default_rng(seed)
    observed = np.array([log_w(X, k) for k in (1, 2, 3)])
    references = []
    for _ in range(B):
        Y = rng.multivariate_normal(X.mean(axis=0), np.cov(X, rowvar=False), size=len(X))
        references.append([log_w(Y, k) for k in (1, 2, 3)])
    references = np.array(references)
    result = domains.gap_choice(X, 3, B=B, seed=seed)
    assert np.allclose(result["gap"], references.mean(axis=0) - observed, rtol=0, atol=1e-12)
    assert np.allclose(result["s"], references.std(axis=0) * np.sqrt(1 + 1 / B), rtol=0, atol=1e-12)


def test_kmeans_fit_settings_and_one_cluster():
    X = np.random.default_rng(2).normal(size=(300, 4)) + 5.0
    one = domains.kmeans_fit(X, 1)
    assert np.allclose(one.cluster_centers_, X.mean(axis=0, keepdims=True), rtol=1e-13, atol=0)
    assert np.array_equal(one.predict(X), np.zeros(300))
    assert one.inertia_ == pytest.approx(((X - X.mean(axis=0)) ** 2).sum(), rel=1e-12)
    three = domains.kmeans_fit(X, 3, seed=4)
    assert (three.n_clusters, three.n_init, three.random_state) == (3, 20, 4)
    assert np.array_equal(three.labels_, domains.kmeans_fit(X, 3, seed=4).labels_)


def test_prediction_strength_uses_training_centroids():
    """The test half's clusters are judged by the training centroids' assignment of the test rows, not the reverse.

    Training blobs at x = -5 and 5 cut the test blob at x = 0 in two; test
    blobs at x = 0 and 20 put both training blobs on one side.
    """
    rng = np.random.default_rng(6)
    train = np.column_stack([np.repeat([-5.0, 5.0], 200), np.zeros(400)]) + rng.normal(size=(400, 2))
    test = np.column_stack([np.repeat([0.0, 20.0], 200), np.zeros(400)]) + rng.normal(size=(400, 2))
    assert domains.prediction_strength(train, test, 2) < 0.6
    assert domains.prediction_strength(test, train, 2) == 1.0


def test_ps_from_labels_singleton_and_pair_count():
    """A test cluster of one row counts as 0; otherwise the share equals a count over all pairs."""
    assert domains.ps_from_labels(np.array([0, 0, 1]), np.array([0, 0, 0])) == 0.0
    rng = np.random.default_rng(4)
    test, assign = rng.integers(0, 4, 300), rng.integers(-2, 3, 300)  # train labels need not be 0..k-1
    shares = []
    for cluster in np.unique(test):
        a = assign[test == cluster]
        together = (a[:, None] == a[None, :]).sum() - a.size  # ordered pairs i != j
        shares.append(together / (a.size * (a.size - 1)))
    assert domains.ps_from_labels(test, assign) == pytest.approx(min(shares), rel=1e-15, abs=0)
    labels = np.random.default_rng(8).integers(0, 10, 22_500)
    assert domains.ps_from_labels(labels, (labels + 3) % 10) == 1.0  # renamed clusters, at the pilot's half size
    with pytest.raises(ValueError):
        domains.ps_from_labels(np.array([0, 1]), np.array([0, 1, 1]))
    with pytest.raises(ValueError):
        domains.ps_from_labels(np.array([], dtype=int), np.array([], dtype=int))
