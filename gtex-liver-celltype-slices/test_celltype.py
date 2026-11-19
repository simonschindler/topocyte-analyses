"""Unit tests of the study's pure helpers, on synthetic inputs (no cohort, no topocyte)."""

from __future__ import annotations

import numpy as np
import pytest

import common

COHORT_CATEGORIES = [
    "Apoptotic Body", "Cancer cell", "Endothelial Cell", "Eosinophils", "Epithelial", "Fibroblasts",
    "Lymphocytes", "Macrophages", "Minor Stromal Cell", "Mitotic Figures", "Muscle Cell", "Neutrophils",
    "Plasmocytes", "Red blood cell",
]


def test_group_codes_maps_all_14():
    codes = common.group_codes(COHORT_CATEGORIES)
    names = [common.CELL_GROUPS[c] for c in codes]
    assert dict(zip(COHORT_CATEGORIES, names)) == {
        "Apoptotic Body": "other", "Cancer cell": "epithelial", "Endothelial Cell": "stromal",
        "Eosinophils": "immune", "Epithelial": "epithelial", "Fibroblasts": "stromal",
        "Lymphocytes": "immune", "Macrophages": "immune", "Minor Stromal Cell": "stromal",
        "Mitotic Figures": "other", "Muscle Cell": "stromal", "Neutrophils": "immune",
        "Plasmocytes": "immune", "Red blood cell": "other",
    }
    assert set(codes) == {0, 1, 2, 3}


def test_group_codes_unknown_raises():
    with pytest.raises(ValueError, match="Goblet"):
        common.group_codes(COHORT_CATEGORIES + ["Goblet"])


def _blocks(rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.normal(size=(30, 5)) * scale + shift for scale, shift in ((1e3, 5.0), (1.0, -2.0), (1e-3, 0.0))]


def test_mfa_equals_gram_eigh():
    rng = np.random.default_rng(0)
    blocks = _blocks(rng)
    scores, evr, weights = common.mfa(blocks, n_keep=10)
    assert np.allclose(weights, [1.0 / np.sqrt(b[:, 0].var()) for b in blocks])
    weighted = np.hstack([b * w for b, w in zip(blocks, weights)])
    centred = weighted - weighted.mean(axis=0)
    values, vectors = np.linalg.eigh(centred @ centred.T)
    order = np.argsort(values)[::-1][:10]
    assert np.allclose((scores**2).sum(axis=0), values[order], rtol=1e-9)
    assert np.allclose(evr[:10], values[order] / values.clip(min=0).sum(), rtol=1e-9)
    for j, i in enumerate(order):
        assert abs(np.corrcoef(scores[:, j], vectors[:, i])[0, 1]) > 1 - 1e-9


def test_mfa_signs():
    rng = np.random.default_rng(1)
    scores, _, _ = common.mfa(_blocks(rng), n_keep=8)
    for j in range(scores.shape[1]):
        column = scores[:, j]
        assert column[np.argmax(np.abs(column))] > 0


def test_baseline_features_shapes():
    rng = np.random.default_rng(2)
    kept_by_group = rng.integers(1, 1000, size=(10, 12, 4))
    kept_by_group[1:] = np.minimum(kept_by_group[1:], kept_by_group[:1])  # slices nest
    medians = rng.uniform(500, 3000, size=12)
    D, C, d_names, c_names = common.baseline_features(medians, kept_by_group)
    assert D.shape == (12, 11) and C.shape == (12, 40) and len(d_names) == 11 and len(c_names) == 40
    kept = kept_by_group.sum(axis=2)
    assert np.allclose(D[:, 0], np.log10(medians)) and np.allclose(D[:, 1], np.log10(kept[0]))
    assert np.allclose(D[:, 2:], (kept[1:] / kept[0]).T)
    assert np.allclose(C[:, 4 * 3 + 2], np.log10(1 + kept_by_group[3, :, 2]))
    assert np.all(np.isfinite(D)) and np.all(np.isfinite(C))


def test_leading_caps_at_available():
    scores = np.arange(30.0).reshape(3, 10)
    assert common.leading(scores, 4).shape == (3, 4)
    assert common.leading(scores, 20).shape == (3, 10)


def _labelled(rng: np.random.Generator, per_group: int = 20) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(8), per_group)
    proba = rng.dirichlet(np.ones(8), size=labels.size)
    proba[np.arange(labels.size), labels] += rng.uniform(0, 0.5, size=labels.size)  # some signal
    return proba / proba.sum(axis=1, keepdims=True), labels


def test_macro_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    proba, labels = _labelled(rng)
    proba[:5, 2] = proba[5:10, 2]  # ties
    macro, per_condition = common.macro_auc(proba, labels)
    scored = labels < 7
    expected = [roc_auc_score(labels[scored] == c, proba[scored, c]) for c in range(7)]
    assert np.allclose(per_condition, expected) and np.isclose(macro, np.mean(expected))


def test_vs_healthy_auc_uses_log_ratio():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    proba, labels = _labelled(rng)
    result = common.vs_healthy_auc(proba, labels)
    assert len(result) == 6
    mask = (labels == 5) | (labels == 0)
    score = np.log(proba[mask, 5]) - np.log(proba[mask, 0])
    assert np.isclose(result[4], roc_auc_score(labels[mask] == 5, score))


def test_purity_toy():
    # Two tight, far-apart clusters of 16 slides each, all of one group per cluster, plus unclassified far away.
    labels = np.array([0] * 16 + [1] * 16 + [7] * 4)
    rng = np.random.default_rng(5)
    scores = np.vstack([rng.normal(0, 0.01, (16, 3)), rng.normal(10, 0.01, (16, 3)), rng.normal(-50, 0.01, (4, 3))])
    per_slide = common.purity(scores, labels, n_neighbors=15)
    chance = 15 / 35
    assert np.allclose(per_slide[:32], 1 / chance) and np.all(np.isnan(per_slide[32:]))
    macro, per_condition = common.macro_purity(per_slide, labels)
    assert np.isclose(per_condition[0], 1 / chance) and np.isclose(per_condition[1], 1 / chance)
    assert np.all(np.isnan(per_condition[2:])) and np.isclose(macro, 1 / chance)


def test_purity_excludes_self_and_counts_ties_stably():
    labels = np.array([0, 0, 1, 1, 1])
    scores = np.zeros((5, 2))  # every distance ties
    per_slide = common.purity(scores, labels, n_neighbors=2)
    # stable order: slide 0's neighbours are 1, 2 -> one of two same-group; chance (2-1)/(5-1)
    assert np.isclose(per_slide[0], 0.5 / 0.25)


def test_bootstrap_keeps_every_group():
    labels = np.array([0] * 12 + [3] * 5 + [7] * 3)
    draws = common.stratified_bootstrap(labels, n_draws=50, rng=np.random.default_rng(6))
    assert draws.shape == (50, labels.size)
    for row in draws:
        assert np.array_equal(np.bincount(labels[row], minlength=8), np.bincount(labels, minlength=8))


def test_verdict_rule():
    assert common.verdict((0.01, 0.05), (0.2, 0.9)) == "enhanced"
    assert common.verdict((0.01, 0.05), (-0.2, 0.9)) == "mixed"
    assert common.verdict((-0.01, 0.05), (0.2, 0.9)) == "mixed"
    assert common.verdict((-0.01, 0.05), (-0.2, 0.9)) == "not enhanced"
    assert common.verdict((-0.05, -0.01), (-0.2, 0.9)) == "not enhanced"


def test_make_folds_stratified_and_seeded():
    labels = np.array([0] * 12 + [1] * 18 + [2] * 30 + [7] * 10)
    folds = common.make_folds(labels, n_repeats=3, n_splits=5, seed=0)
    assert folds.shape == (3, labels.size) and set(np.unique(folds)) == {0, 1, 2, 3, 4}
    assert np.array_equal(folds, common.make_folds(labels, n_repeats=3, n_splits=5, seed=0))
    for row in folds:
        for g in (0, 1, 2, 7):
            counts = np.bincount(row[labels == g], minlength=5)
            assert counts.max() - counts.min() <= 1


def test_cv_probabilities_shapes_and_out_of_fold():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(8), 15)
    X = rng.normal(size=(labels.size, 3)) + labels[:, None] * 0.5
    folds = common.make_folds(labels, n_repeats=2, n_splits=5, seed=0)
    mean_proba, repeat_proba, warnings_count = common.cv_probabilities(X, labels, folds)
    assert mean_proba.shape == (labels.size, 8) and repeat_proba.shape == (2, labels.size, 8)
    assert np.allclose(mean_proba.sum(axis=1), 1) and np.allclose(mean_proba, repeat_proba.mean(axis=0))
    # out of fold: changing one slide's features changes only other slides' predictions through training
    X2 = X.copy()
    X2[0] += 100.0
    _, repeat2, _ = common.cv_probabilities(X2, labels, folds)
    test_fold = folds[0, 0]
    same_fold = folds[0] == test_fold
    assert np.allclose(repeat_proba[0, same_fold & (np.arange(labels.size) != 0)], repeat2[0, same_fold & (np.arange(labels.size) != 0)])
    assert warnings_count >= 0
