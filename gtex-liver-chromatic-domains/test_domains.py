"""Unit tests of ``domains.py``, on synthetic data: the rule for k (design step 10), then Task 7's main.

The gap tests use the controller's constructions (ruling R2): 4,000 rows and
20 columns, drawn from one ``default_rng(1)`` in the order elongated,
skewed, two clusters, three clusters. They run the binding settings (B = 20
reference draws, ``n_init=20``, minimum gain 0.03), so they take about a
minute.

Task 7's tests cover the composition strata (level 1), the subdomains (level
2), the control's gate, the pairwise guard with its slide bootstrap, the
descriptives, the verdict and the pre-flight's composition null. None reads
``results/``; no topocyte and no cohort.
"""

from __future__ import annotations

import inspect
import itertools
import json
import re
from fractions import Fraction
from functools import partial

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score

import domains
import run_pilot

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


# --------------------------------------------------------------------------------- Task 7: strata, subdomains, guard


def test_match_labels_hungarian():
    """Permuted labels map back, through noise too; the map is one-to-one where a row-wise argmax is not."""
    rng = np.random.default_rng(10)
    own = rng.integers(0, 5, 500)
    perm = np.array([3, 0, 4, 1, 2])
    expected = {i: int(perm[i]) for i in range(5)}
    assert domains.match_labels(own, perm[own]) == expected
    noisy = perm[own]
    flip = rng.random(500) < 0.1
    noisy[flip] = rng.integers(0, 5, int(flip.sum()))
    assert domains.match_labels(own, noisy) == expected
    # The contingency table [[5, 4], [5, 0]]: a row-wise argmax would send both own labels to 0.
    own, final = np.repeat([0, 0, 1], [5, 4, 5]), np.repeat([0, 1, 0], [5, 4, 5])
    assert domains.match_labels(own, final) == {0: 1, 1: 0}
    with pytest.raises(ValueError):
        domains.match_labels(np.array([0, 1]), np.array([0, -1]))


def test_level1_inputs_zscore():
    """Level 1 (ruling R3): each half z-scored with its own means and SDs (ddof 0) for its gap statistic, and both
    halves with the training half's in each prediction-strength direction."""
    rng = np.random.default_rng(11)
    A = rng.normal([2.1, 0.7, 0.15, 0.07, 0.06], [0.15, 0.2, 0.1, 0.09, 0.09], size=(300, 5))
    B = rng.normal([2.3, 0.6, 0.2, 0.1, 0.1], [0.1, 0.3, 0.1, 0.05, 0.05], size=(200, 5))
    gap_A, gap_B, ((train_A, test_B), (train_B, test_A)) = domains.level1_inputs(A, B)
    for z, X, ref in ((gap_A, A, A), (gap_B, B, B), (train_A, A, A), (train_B, B, B), (test_B, B, A), (test_A, A, B)):
        np.testing.assert_allclose(z, (X - ref.mean(axis=0)) / ref.std(axis=0), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="constant"):
        domains.zscore(A, np.ones((10, 5)))


def test_balanced_accuracy_over_present_classes():
    """The mean recall over the classes present in the true labels, from confusion counts (rows true), batched."""
    rng = np.random.default_rng(12)
    truth = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.6, truth, rng.integers(0, 4, 400))
    conf = np.zeros((4, 4), np.int64)
    np.add.at(conf, (truth, pred), 1)
    assert domains.balanced_accuracy(conf) == pytest.approx(balanced_accuracy_score(truth, pred), rel=1e-12)
    absent = np.array([[1, 0, 1], [0, 2, 0], [0, 0, 0]])  # class 2 is predicted but absent from the truth
    assert domains.balanced_accuracy(absent) == 0.75  # recalls 1/2 and 1
    batch = np.stack([conf[:3, :3], absent])
    np.testing.assert_allclose(domains.balanced_accuracy(batch), [domains.balanced_accuracy(conf[:3, :3]), 0.75])
    with pytest.raises(ValueError):
        domains.balanced_accuracy(np.zeros((2, 2), np.int64))


def test_bootstrap_resamples_slides():
    """Each draw is a multiset of whole test slides: its balanced accuracy is that of their centres pooled."""
    rng = np.random.default_rng(13)
    slide = np.repeat([3, 4, 7, 8, 9], [30, 50, 20, 40, 60])  # a test half's slide indices
    truth = rng.integers(0, 3, slide.size)
    pred = np.where(rng.random(slide.size) < 0.7, truth, rng.integers(0, 3, slide.size))
    assert np.array_equal(domains.bootstrap_draws(5), np.random.default_rng(12345).integers(0, 5, size=(4000, 5)))
    draws = domains.bootstrap_draws(5, n_draws=300)
    slides, conf = domains.confusion_by_slide(truth, pred, slide, 3)
    assert slides.tolist() == [3, 4, 7, 8, 9] and conf.shape == (5, 3, 3) and conf.sum() == slide.size
    resampled = domains.resampled_ba(conf, draws)
    assert resampled.shape == (300,)
    for draw, ba in zip(draws, resampled, strict=True):
        rows = np.concatenate([np.flatnonzero(slide == slides[j]) for j in draw])
        t, p = truth[rows], pred[rows]
        assert ba == pytest.approx(np.mean([np.mean(p[t == label] == label) for label in np.unique(t)]), rel=1e-12)


def _guard_case(pairwise_determined: bool) -> dict:
    """One compartment on 12 slides of 120 centres, halves alternating, with two subdomains.

    The level-2 scores in half B's basis are half A's rotated, so each
    direction must keep its training basis for the test half. The
    domain-block scores are noise.
    """
    rng = np.random.default_rng(14)
    slide = np.repeat(np.arange(12), 120)
    n = slide.size
    P, L1 = rng.normal(size=(n, 10)), rng.normal(size=(n, 5))
    if pairwise_determined:
        final, separation = (P[:, 0] > 0).astype(np.int64), 1.0  # P decides the labels, which overlap in level 2
    else:
        final, separation = rng.integers(0, 2, n), 4.0  # P is noise; level 2 separates the labels by 8 SD
    X = rng.normal(size=(n, 20))
    X[:, 0] += separation * (2 * final - 1)
    rotation = np.linalg.qr(rng.normal(size=(20, 20)))[0]
    D = rng.normal(size=(n, 72))
    return {
        "X": {"A": X, "B": X @ rotation},
        "F": np.column_stack([P, L1]),
        "D": {"A": D, "B": D[:, ::-1]},
        "final": final,
        "half": np.where(slide % 2 == 0, "A", "B"),
        "slide": slide,
        "k": 2,
    }


def test_guard_passes_when_pairwise_uninformative():
    """P is noise and the subdomains are clear in level 2: BA_stab - BA_pw has its interval above 0 both ways."""
    case = _guard_case(pairwise_determined=False)
    result = domains.guard(**case)
    for name in ("A->B", "B->A"):
        entry = result[name]
        assert entry["BA_stab"] > 0.99 and entry["BA_pw"] < 0.6 and entry["BA_dom"] < 0.6
        assert entry["stab_minus_pw"] == pytest.approx(entry["BA_stab"] - entry["BA_pw"], rel=1e-12)
        assert entry["stab_minus_pw_interval"][0] > 0 and entry["passed"]
        assert (entry["n_train"], entry["n_test"], entry["n_test_slides"]) == (720, 720, 6)
    assert result["passed"]
    assert sorted(result["draws"]) == ["stab_minus_dom_AB", "stab_minus_dom_BA", "stab_minus_pw_AB", "stab_minus_pw_BA"]
    assert all(draws.shape == (4000,) for draws in result["draws"].values())
    for t, u in (("A", "B"), ("B", "A")):  # the 95% percentile intervals of the stored draws
        for kind in ("pw", "dom"):
            interval = np.percentile(result["draws"][f"stab_minus_{kind}_{t}{u}"], [2.5, 97.5]).tolist()
            assert result[f"{t}->{u}"][f"stab_minus_{kind}_interval"] == interval
    # The training half's own k-means labels reach the final labels through the Hungarian matching, either way round.
    swapped = domains.guard(**(case | {"final": 1 - case["final"]}))
    for name in ("A->B", "B->A"):
        assert swapped[name]["BA_stab"] == result[name]["BA_stab"]


def test_guard_detects_pairwise_determined():
    """The subdomains are a function of P: the pairwise classifier predicts them better, so no direction passes."""
    result = domains.guard(**_guard_case(pairwise_determined=True))
    for name in ("A->B", "B->A"):
        entry = result[name]
        assert entry["BA_pw"] > 0.97 and entry["stab_minus_pw"] < 0 and not entry["passed"]
    assert not result["passed"]


def test_guard_needs_the_interval_above_zero(monkeypatch):
    """A direction passes only if BA_stab - BA_pw and its bootstrap lower bound are both above 0.

    The predictions are fixed by hand: on every slide two balanced subdomains,
    the pairwise prediction right for 70% of each slide's centres, the
    stability prediction right everywhere but on one slide per half. The
    point difference is then +0.133 in both directions, but draws holding
    that slide twice fall below 0. The draws come from a fresh
    ``default_rng(12345)`` per direction, and the interval is their 2.5 and
    97.5 percentiles.
    """
    slide = np.repeat(np.arange(12), 40)
    n = slide.size
    half = np.where(slide % 2 == 0, "A", "B")
    final = np.tile(np.repeat([0, 1], 20), 12)
    stab = np.where(np.isin(slide, [4, 7]), 1 - final, final)  # wrong on slide 4 (half A) and slide 7 (half B)
    pw = np.where(np.tile(np.arange(20) % 10 < 7, 24), final, 1 - final)  # 70% right on every slide and subdomain
    rows = np.arange(n, dtype=np.float64)[:, None]  # each centre's features are its row, so the fakes look it up

    def fake_stability(train, test, k, train_final):
        return stab[test[:, 0].astype(int)]

    def fake_classifier(train_F, train_y, test_F):
        return pw[test_F[:, 0].astype(int)]

    monkeypatch.setattr(domains, "stability_prediction", fake_stability)
    monkeypatch.setattr(domains, "classifier_prediction", fake_classifier)
    result = domains.guard({"A": rows, "B": rows}, rows, {"A": rows, "B": rows}, final, half, slide, 2)
    for t, u in (("A", "B"), ("B", "A")):
        entry = result[f"{t}->{u}"]
        assert (entry["BA_stab"], entry["BA_pw"]) == (pytest.approx(5 / 6), pytest.approx(0.7))
        assert entry["stab_minus_pw"] > 0 and entry["stab_minus_pw_interval"][0] < 0 and not entry["passed"]
        test = half == u
        slides = np.unique(slide[test])
        draws = np.random.default_rng(12345).integers(0, 6, size=(4000, 6))
        confusions = [domains.confusion_by_slide(final[test], p[test], slide[test], 2)[1] for p in (stab, pw)]
        ba = [domains.resampled_ba(conf, draws) for conf in confusions]
        np.testing.assert_array_equal(result["draws"][f"stab_minus_pw_{t}{u}"], ba[0] - ba[1])
        assert entry["test_slides"] == slides.tolist()
        assert entry["stab_minus_pw_interval"] == np.percentile(ba[0] - ba[1], [2.5, 97.5]).tolist()
    assert not result["passed"]


def test_level1_final_fit_on_all_centres(monkeypatch):
    """With k = 2 from the rule, the strata are ``kmeans_fit`` of ``L1`` z-scored with all centres' means and SDs."""
    rng = np.random.default_rng(19)
    half = np.repeat(["A", "B"], 300)
    L1 = rng.normal(size=(600, 5)) * [0.1, 0.2, 0.1, 0.1, 0.1] + [2.0, 0.7, 0.2, 0.05, 0.05]
    L1[half == "B"] *= [1.0, 3.0, 1.0, 1.0, 1.0]  # the halves differ, so their z-scorings differ from all centres'
    gap = {"k": 2, "gap": np.zeros(10), "s": np.zeros(10)}
    chosen = {"k": 2, "k_A": 2, "k_B": 2, "ps": [1.0, 1.0], "gap_A": gap, "gap_B": gap}
    monkeypatch.setattr(domains, "choose_k", lambda *args, **kwargs: chosen)
    calls, real = [], domains.kmeans_fit

    def spy(X, k, seed=0):
        model = real(X, k, seed)
        calls.append((np.asarray(X), k, model))
        return model

    monkeypatch.setattr(domains, "kmeans_fit", spy)
    compartment, choice = domains.level1(L1, half)
    (final,) = [(X, k, model) for X, k, model in calls if len(X) == 600]  # the one fit on all centres
    np.testing.assert_allclose(final[0], (L1 - L1.mean(axis=0)) / L1.std(axis=0), rtol=0, atol=1e-12)
    assert final[1] == 2 and np.array_equal(compartment, final[2].labels_)
    assert choice["k"] == 2 and set(choice["ari"]) == {"A->B", "B->A"}
    assert choice["zscore"]["all"]["sd"] == L1.std(axis=0).tolist()


def test_cross_half_ari_compares_the_ps_partitions():
    """Per direction, the ARI between the test half's own k-means labels and the training centroids' assignment."""
    rng = np.random.default_rng(20)
    train = np.column_stack([np.repeat([-5.0, 5.0], 200), np.zeros(400)]) + rng.normal(size=(400, 2))
    test = np.column_stack([np.repeat([0.0, 20.0], 200), np.zeros(400)]) + rng.normal(size=(400, 2))
    directions = [(train, test), (test, train)]
    got = domains.cross_half_ari(directions, 2)
    for value, (a, b) in zip(got, directions, strict=True):
        own = KMeans(n_clusters=2, n_init=20, random_state=0).fit(b).labels_
        assignment = KMeans(n_clusters=2, n_init=20, random_state=0).fit(a).predict(b)
        assert value == pytest.approx(adjusted_rand_score(own, assignment), rel=1e-12)
    assert max(got) < 0.9  # the training centroids cut the test blobs differently


def test_task7_binding_settings(monkeypatch):
    """The main's fixed settings (design steps 8-9 and "Checks fixed in advance"; rulings T7-c and T7-g)."""
    assert (domains.K_MAX_STRATA, domains.K_MAX_SUBDOMAINS, domains.MIN_HALF) == (10, 6, 500)
    assert (domains.N_DRAWS, domains.BOOT_SEED, domains.K_NN) == (4000, 12345, 5)
    share = domains.MIN_SHARE
    assert isinstance(share, Fraction) and (share.numerator, share.denominator) == (1, 10)
    assert domains.DOMAIN_BLOCKS == (0, 1, 4, 5, 8, 9, 12, 13)
    parameters = inspect.signature(domains.guard).parameters.values()
    assert {p.name: p.default for p in parameters if p.default is not p.empty} == {"n_draws": 4000, "seed": 12345}
    assert inspect.signature(domains.contiguity).parameters["k_nn"].default == 5
    seen = []

    def factory(**kwargs):
        seen.append(kwargs)
        return HistGradientBoostingClassifier(**kwargs)

    monkeypatch.setattr(domains, "HistGradientBoostingClassifier", factory)
    domains.classifier_prediction(np.arange(20.0)[:, None], np.repeat([0, 1], 10), np.zeros((3, 1)))
    assert seen == [{"random_state": 0}]


def test_small_compartment_gets_no_subdomains(monkeypatch):
    """499 centres of a compartment in one half: k = 1 without the rule, and subdomain -1; 500 in each half run it.

    The final subdomains come from the all-slides basis, whose clusters here
    follow other labels than the halves' bases.
    """
    rng = np.random.default_rng(15)
    sizes = {(0, "A"): 499, (0, "B"): 800, (1, "A"): 500, (1, "B"): 500}
    compartment = np.repeat([comp for comp, _ in sizes], list(sizes.values()))
    half = np.repeat([h for _, h in sizes], list(sizes.values()))
    labels, other = rng.integers(0, 2, half.size), rng.integers(0, 2, half.size)
    X, X_all = rng.normal(size=(half.size, 20)), rng.normal(size=(half.size, 20))
    X[:, 0] += 8.0 * labels  # two clusters 8 SD apart in both compartments
    X_all[:, 0] += 8.0 * other
    calls, real = [], domains.choose_k

    def spy(gap_A, gap_B, directions, k_max, **kwargs):
        calls.append((len(gap_A), len(gap_B), k_max))
        return real(gap_A, gap_B, directions, k_max, **kwargs)

    monkeypatch.setattr(domains, "choose_k", spy)
    choices, subdomain = domains.level2({"A": X, "B": X}, half, compartment, 2, X_all=X_all)
    assert (choices[0]["k"], choices[0]["too_small"], choices[0]["n_A"], choices[0]["n_B"]) == (1, True, 499, 800)
    assert np.all(subdomain[compartment == 0] == -1)
    assert calls == [(500, 500, 6)]
    assert (choices[1]["k"], choices[1]["too_small"]) == (2, False)
    assert adjusted_rand_score(other[compartment == 1], subdomain[compartment == 1]) == 1.0


def test_contiguity_expectation_exhaustive():
    """On six centres, the expectation is the mean observed count over every reassignment of the subdomain labels
    within each (slide, compartment) group, per subdomain and in total.

    k_nn = 2. Centre 2 is compartment 1 (k = 1, subdomain -1), a neighbour of
    compartment 0's centres that counts for nothing; compartment 0's five
    centres have 7 same-group directed neighbour pairs out of 20.
    """
    xy = np.array([[0.0, 0.0], [1.0, 0.3], [2.2, -0.4], [3.1, 0.9], [4.6, 0.2], [5.0, 1.7]])
    slide = np.zeros(6, np.int64)
    compartment = np.array([0, 0, 1, 0, 0, 0])
    subdomain = np.array([0, 1, -1, 1, 0, 2])
    result = domains.contiguity(xy, slide, compartment, subdomain, k_nn=2)
    assert list(result) == [0]
    group = np.flatnonzero(compartment == 0)
    counts = []
    for order in itertools.permutations(range(group.size)):
        relabelled = subdomain.copy()
        relabelled[group] = subdomain[group][list(order)]
        stratum = domains.contiguity(xy, slide, compartment, relabelled, k_nn=2)[0]
        counts.append([s["observed"] for s in stratum["subdomains"]])
    counts = np.array(counts, dtype=np.float64)
    assert len(set(counts.sum(axis=1).tolist())) > 1  # the toy tells reassignments apart
    expected = [s["expected"] for s in result[0]["subdomains"]]
    np.testing.assert_allclose(expected, counts.mean(axis=0), rtol=1e-12, atol=1e-15)
    assert result[0]["expected"] == pytest.approx(counts.sum(axis=1).mean(), rel=1e-12)
    assert result[0]["subdomains"][2]["ratio"] is None  # one centre of subdomain 2: nothing expected


def test_contiguity_counts_same_slide_neighbours():
    """The neighbours are each centre's five nearest centres on its own slide, whatever their compartment: the observed
    pairs match a brute-force count, and a compartment without subdomains is left out."""
    rng = np.random.default_rng(16)
    slide = np.repeat([2, 5, 9], 40)
    xy = rng.uniform(0, 100, size=(120, 2))  # the slides overlap in position
    compartment = rng.integers(0, 2, 120)
    subdomain = np.where(compartment == 0, rng.integers(0, 3, 120), -1)
    result = domains.contiguity(xy, slide, compartment, subdomain)
    assert list(result) == [0]
    observed = np.zeros(3, np.int64)
    for i in np.flatnonzero(compartment == 0):
        others = np.flatnonzero((slide == slide[i]) & (np.arange(120) != i))
        for j in others[np.argsort(np.linalg.norm(xy[others] - xy[i], axis=1))[:5]]:
            if compartment[j] == 0 and subdomain[j] == subdomain[i]:
                observed[subdomain[i]] += 1
    assert [s["observed"] for s in result[0]["subdomains"]] == observed.tolist()
    assert result[0]["observed"] == observed.sum()


def test_verdict():
    """The verdict "adds" needs a stratum holding at least 10% of all centres to pass both ways (ruling T7-b)."""
    verdict = domains.verdict
    failed = verdict([5_000, 40_000], [1, 2])
    assert failed["verdict"] == "not run: control failed" and "composition stratum 1" in failed["reason"]
    assert verdict([30_000, 15_000], [1, 1], [1, 1], [None, None]) == {
        "verdict": "does not add",
        "reason": "no compartment with subdomains",
    }
    assert verdict([4_500, 40_500], [1, 1], [2, 1], [True, None])["verdict"] == "adds"  # exactly 10% counts
    assert verdict([4_499, 40_501], [1, 1], [2, 3], [True, False])["verdict"] == "does not add"
    assert verdict([30_000, 15_000], [1, 1], [2, 2], [False, True])["verdict"] == "adds"
    assert verdict([30_000, 15_000], [1, 1], [2, 2], [False, False])["verdict"] == "does not add"


def _features(control_clusters: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Synthetic features of six slides of 200 centres, halves A, A, B, B, A, B, and their subdomain labels.

    Level 1 is one Gaussian cloud. In each level-2 basis (a rotation of its
    own) the labels sit 8 SD apart; the control is noise with PCA-like
    variances, or carries clusters of its own. P and the domain blocks are
    noise.
    """
    rng = np.random.default_rng(17)
    slide = np.repeat(np.arange(6), 200)
    n = slide.size
    labels = rng.integers(0, 2, n)

    def basis(marks: np.ndarray | None) -> np.ndarray:
        X = rng.normal(size=(n, 20)) * np.sqrt(1 / np.arange(1, 21))
        if marks is not None:
            X[:, 0] += 8.0 * marks
        return X @ np.linalg.qr(rng.normal(size=(20, 20)))[0]

    control_marks = rng.integers(0, 2, n) if control_clusters else None
    features = {
        "half": np.array(["A", "A", "B", "B", "A", "B"])[slide],
        "slide_index": slide,
        "xy": rng.uniform(0, 3000, size=(n, 2)),
        "L1": rng.normal([2.15, 0.72, 0.15, 0.07, 0.06], [0.15, 0.2, 0.1, 0.09, 0.09], size=(n, 5)),
        "P": rng.normal(size=(n, 10)),
        **{f"L2_obs_{f}": basis(labels) for f in ("A", "B", "all")},
        **{f"L2_ctl_{f}": basis(control_marks) for f in ("A", "B")},
        **{f"blk_obs_{f}": rng.normal(size=(n, 16, 9)) for f in ("A", "B")},
    }
    return features, labels


def _mentions_modes(text: str) -> bool:
    return re.search(r"\bmodes?\b", text, flags=re.IGNORECASE) is not None


def test_run_finds_subdomains_and_runs_the_guard():
    """End to end on synthetic features: one composition stratum, a passing control, two subdomains and the guard."""
    features, labels = _features(control_clusters=False)
    lines: list[str] = []
    arrays, summary = domains.run(features, log=lines.append)
    level1 = summary["level1"]
    assert level1["k"] == 1 and level1["ari"] is None and np.all(arrays["compartment"] == 0)
    assert len(level1["gap_A"]["gap"]) == 10 and len(level1["ps_curve"]["A->B"]) == 9
    assert summary["control_passed"] and arrays["control_k"].tolist() == [1]
    (stratum,) = summary["compartments"]
    assert (stratum["size"], stratum["size_A"], stratum["size_B"], stratum["share"]) == (1200, 600, 600, 1.0)
    assert stratum["control"]["k"] == 1 and len(stratum["control"]["ps_curve"]["B->A"]) == 5
    level2 = stratum["level2"]
    assert level2["k"] == 2 and adjusted_rand_score(labels, arrays["subdomain"]) == 1.0
    np.testing.assert_array_equal(arrays["subdomain"], domains.kmeans_fit(features["L2_obs_all"], 2).labels_)
    assert level2["ps_curve"]["A->B"][0] == level2["ps"][0] and level2["ari"] == {"A->B": 1.0, "B->A": 1.0}
    assert stratum["guard"]["passed"] and summary["verdict"]["verdict"] == "adds"
    assert stratum["contiguity"]["ratio"] == pytest.approx(1.0, abs=0.15)  # labels placed at random
    assert sorted(key for key in arrays if key.startswith("boot_")) == [
        "boot_0_stab_minus_dom_AB",
        "boot_0_stab_minus_dom_BA",
        "boot_0_stab_minus_pw_AB",
        "boot_0_stab_minus_pw_BA",
    ]
    assert all(np.all(np.isfinite(value)) for value in arrays.values())
    text = json.dumps(summary, allow_nan=False)
    assert not _mentions_modes(text) and not _mentions_modes("\n".join(lines))
    assert any("composition strat" in line for line in lines)


def test_run_stops_level_two_when_the_control_fails(monkeypatch):
    """The control finds subdomains: no observed level-2 fit, subdomain -1 everywhere, no guard (ruling T7-a)."""
    features, _ = _features(control_clusters=True)
    calls, real = [], domains.choose_k

    def spy(*args, **kwargs):
        calls.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(domains, "choose_k", spy)
    arrays, summary = domains.run(features, log=lambda line: None)
    assert calls == [10, 6]  # level 1 and the control, nothing more
    assert not summary["control_passed"] and arrays["control_k"].tolist() == [2]
    assert np.all(arrays["subdomain"] == -1) and not any(key.startswith("boot_") for key in arrays)
    (stratum,) = summary["compartments"]
    assert stratum["level2"] is None and stratum["guard"] is None and stratum["contiguity"] is None
    assert summary["verdict"]["verdict"] == "not run: control failed"
    json.dumps(summary, allow_nan=False)


def test_composition_null_construction():
    """Poisson(120) + 1 cells, Dirichlet(6, 2, 1, 1) compositions and multinomial counts, drawn in that order; the
    columns are log10 of the cells and the four fractions."""
    X = run_pilot._composition(np.random.default_rng(18), 1000)
    rng = np.random.default_rng(18)
    cells = rng.poisson(120, 1000) + 1
    compositions = rng.dirichlet([6.0, 2.0, 1.0, 1.0], 1000)
    counts = np.array([rng.multinomial(n, p) for n, p in zip(cells, compositions, strict=True)])
    np.testing.assert_array_equal(X, np.column_stack([np.log10(cells), counts / cells[:, None]]))


def test_preflight_composition_null_is_reported_not_gated(monkeypatch, tmp_path):
    """The composition null comes after the gated cases from the same generator, is z-scored as level 1, and sits
    under "descriptive", outside "passed"."""
    tiny = partial(run_pilot._gaussian, np.ones(3))
    monkeypatch.setattr(run_pilot, "PREFLIGHT_CASES", (("tiny Gaussian", 40, 6, 1, tiny),))
    monkeypatch.setattr(run_pilot.c, "RESULTS", tmp_path)
    calls = []

    def fake_choose_k(gap_A, gap_B, directions, k_max, **kwargs):
        calls.append((gap_A, gap_B, directions, k_max))
        k = 1 if len(calls) == 1 else 5  # the null's k must not decide "passed"
        half = {"k": k, "gap": np.zeros(k_max), "s": np.ones(k_max)}
        return {"k": k, "k_A": k, "k_B": k, "ps": [0.9, 0.9] if k > 1 else [], "gap_A": half, "gap_B": half}

    monkeypatch.setattr(domains, "choose_k", fake_choose_k)
    assert run_pilot.cmd_preflight()
    out = json.loads((tmp_path / "preflight.json").read_text())
    assert out["passed"] and [case["case"] for case in out["cases"]] == ["tiny Gaussian"]
    (null,) = out["descriptive"]
    assert (null["k"], null["rows_per_half"], null["columns"], null["k_max"]) == (5, 22_500, 5, 10)
    assert "passed" not in null and "expected_k" not in null
    rng = np.random.default_rng(run_pilot.PREFLIGHT_SEED)
    for _ in range(2):
        tiny(rng, 40)  # the gated case's halves come first
    A = run_pilot._composition(rng, 22_500)
    gap_A, gap_B, directions = domains.level1_inputs(A, run_pilot._composition(rng, 22_500))
    got = calls[1]
    assert got[3] == 10
    for a, b in zip((*got[:2], *got[2][0], *got[2][1]), (gap_A, gap_B, *directions[0], *directions[1]), strict=True):
        np.testing.assert_array_equal(a, b)
