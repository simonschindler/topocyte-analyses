"""Unit tests of the study's constants and helpers, on synthetic inputs (no cohort, no topocyte)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import common

COHORT_CATEGORIES = [
    "Apoptotic Body", "Cancer cell", "Endothelial Cell", "Eosinophils", "Epithelial", "Fibroblasts",
    "Lymphocytes", "Macrophages", "Minor Stromal Cell", "Mitotic Figures", "Muscle Cell", "Neutrophils",
    "Plasmocytes", "Red blood cell",
]


def test_class_codes_maps_all_14():
    codes = common.class_codes(COHORT_CATEGORIES)
    assert codes.dtype == np.int8
    assert common.CLASSES == ("epithelial", "stromal", "immune", "other")
    assert dict(zip(COHORT_CATEGORIES, (common.CLASSES[c] for c in codes))) == {
        "Apoptotic Body": "other", "Cancer cell": "epithelial", "Endothelial Cell": "stromal",
        "Eosinophils": "immune", "Epithelial": "epithelial", "Fibroblasts": "stromal",
        "Lymphocytes": "immune", "Macrophages": "immune", "Minor Stromal Cell": "stromal",
        "Mitotic Figures": "other", "Muscle Cell": "stromal", "Neutrophils": "immune",
        "Plasmocytes": "immune", "Red blood cell": "other",
    }


def test_class_codes_unknown_raises():
    with pytest.raises(ValueError, match="Hepatocyte"):
        common.class_codes(["Hepatocyte"])


def test_delta_px():
    assert common.DELTA_PX == pytest.approx(303.52084176446783, abs=1e-12)


def test_blocks_and_pairs():
    assert len(common.BLOCKS) == 16 and common.BLOCKS[:2] == ((0, "domain", 0), (0, "domain", 1))
    assert common.BLOCKS[-1] == (3, "image", 1)
    assert len(common.PAIRS) == 10 and common.PAIRS[:2] == ((0, 0), (0, 1))
    assert common.PAIRS[-1] == (3, 3)


def test_check_topocyte_needs_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        common.check_topocyte()


def _write_inputs(tmp_path: Path, rows: list[tuple]) -> tuple[Path, Path]:
    """Fake ``holdout.csv`` and ``pathology.csv`` from rows (slide_id, group, holdout, quality_flag, n_cells)."""
    holdout = pd.DataFrame([row[:3] for row in rows], columns=["slide_id", "group", "holdout"])
    pathology = pd.DataFrame(
        [(slide, group, "other", "curated", group, "notes", flag, n) for slide, group, _, flag, n in rows],
        columns=["slide_id", "group", "label", "label_reason", "categories", "notes", "quality_flag", "n_cells"],
    )
    holdout.to_csv(tmp_path / "holdout.csv", index=False)
    pathology.to_csv(tmp_path / "pathology.csv", index=False)
    return tmp_path / "holdout.csv", tmp_path / "pathology.csv"


def test_eligible_excludes_holdout_and_preservation(tmp_path):
    rows = [
        ("GTEX-G", "cirrhosis", False, None, 7000),
        ("GTEX-A", "healthy", False, None, 1000),
        ("GTEX-B", "cirrhosis", True, None, 2000),  # holdout
        ("GTEX-C", "steatosis", False, "preservation", 3000),
        ("GTEX-D", "necrosis", False, None, 4000),  # not a pilot condition
        ("GTEX-E", "unclassified", False, None, 5000),
        ("GTEX-F", "inflammation/hepatitis", False, None, 6000),
    ]
    eligible = common.eligible_slides(*_write_inputs(tmp_path, rows))
    assert eligible.to_dict("list") == {
        "slide_id": ["GTEX-A", "GTEX-F", "GTEX-G"],
        "condition": ["healthy", "inflammation/hepatitis", "cirrhosis"],
        "n_cells": [1000, 6000, 7000],
    }


def fake_eligible(per_condition: int) -> pd.DataFrame:
    """``per_condition`` slides of every pilot condition, as ``eligible_slides`` returns them."""
    rows = [
        (f"GTEX-{c}{i:02d}", condition, 100_000 + i)
        for c, condition in enumerate(common.CONDITIONS)
        for i in range(per_condition)
    ]
    return pd.DataFrame(rows, columns=["slide_id", "condition", "n_cells"])


def test_draw_pilot_stratified_and_reproducible():
    p = common.draw_pilot(fake_eligible(per_condition=8))
    assert p.equals(common.draw_pilot(fake_eligible(per_condition=8)))
    assert p.groupby("condition").size().to_dict() == {c: 5 for c in common.CONDITIONS}
    assert [((p.condition == c) & (p.half == "A")).sum() for c in common.CONDITIONS] == [3, 2, 3, 2, 3, 2]
    assert list(p.slide_index) == list(range(30))
    # One row per drawn slide, of its own condition; conditions in CONDITIONS order, half A first within each.
    fake = fake_eligible(per_condition=8)
    assert list(p.columns) == ["slide_id", "condition", "half", "slide_index"]
    assert p.slide_id.is_unique
    assert list(p.condition) == list(p.slide_id.map(dict(zip(fake.slide_id, fake.condition))))
    assert list(p.condition.unique()) == list(common.CONDITIONS)
    assert list(p.half) == (["A"] * 3 + ["B"] * 2 + ["A"] * 2 + ["B"] * 3) * 3
    # The draw depends on the candidates and the seed, not on the table's row order.
    assert p.equals(common.draw_pilot(fake.sample(frac=1.0, random_state=0)))
    assert not p.equals(common.draw_pilot(fake, seed=1))


def test_draw_centres():
    a = common.draw_centres(10_000, 3)
    assert a.shape == (1500,) and np.unique(a).size == 1500 and a.max() < 10_000
    assert np.array_equal(a, common.draw_centres(10_000, 3))
    # The slide's generator is SeedSequence([SEED, slide_index, 0, R + 1]); rows stay in draw order.
    slide_rng = np.random.default_rng(np.random.SeedSequence([common.SEED, 3, 0, common.R + 1]))
    assert np.array_equal(a, slide_rng.choice(10_000, size=1500, replace=False))
    assert not np.array_equal(a, common.draw_centres(10_000, 4))


def test_centre_draw_is_not_the_pilot_stream():
    # SeedSequence pads short keys with zeros, so the pilot's default_rng(SEED) is SeedSequence([SEED, 0, 0, 0]);
    # slide 0's centre draw must not reuse it.
    pilot_rng = np.random.default_rng(common.SEED)
    assert not np.array_equal(common.draw_centres(10_000, 0), pilot_rng.choice(10_000, size=1500, replace=False))


def test_draw_centres_reads_n_centres_at_call_time(monkeypatch):
    monkeypatch.setattr(common, "N_CENTRES", 20)
    assert common.draw_centres(300, 0).shape == (20,)


def test_labelling_keeps_counts_and_is_reproducible():
    cls = np.repeat(np.arange(4), [40, 30, 20, 10]).astype(np.int8)
    assert np.array_equal(common.labelling(cls, 4, 17, 0), cls)
    a = common.labelling(cls, 4, 17, 5)
    assert np.array_equal(a, common.labelling(cls, 4, 17, 5))
    assert np.array_equal(np.bincount(a, minlength=4), [40, 30, 20, 10])
    assert not np.array_equal(a, common.labelling(cls, 4, 17, 6))
    # Labelling r of a centre comes from SeedSequence([SEED, slide_index, centre_index, r]).
    assert a.dtype == np.int8
    centre_rng = np.random.default_rng(np.random.SeedSequence([common.SEED, 4, 17, 5]))
    assert np.array_equal(a, centre_rng.permutation(cls))
    assert not np.array_equal(a, common.labelling(cls, 4, 18, 5))
