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
