"""Tests of the class-curve fPCA study.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``progress.md``, Rulings). The input tests
read the sibling studies' saved results, read only.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocyte.ph import betti_inner_product_matrix

import common as c
import curves as cv


def test_load_inputs_matches_the_celltype_split():
    d = c.load_inputs()
    assert len(d["slide_ids"]) == 594 and d["DC"].shape == (594, 51)
    assert d["dev"].sum() == 476 and d["holdout"].sum() == 118 and not (d["dev"] & d["holdout"]).any()
    assert np.bincount(d["labels"][d["dev"]], minlength=8).tolist() == [53, 124, 162, 14, 10, 40, 46, 27]
    assert d["bounds"][0] == 0 and d["bounds"][-1] == 126_219_124


def test_load_inputs_refuses_a_density_file_in_another_slide_order(tmp_path, monkeypatch):
    slide_ids = c.load_inputs()["slide_ids"]
    reordered = tmp_path / "density.npz"
    np.savez(reordered, slide_ids=np.array(slide_ids[::-1]), bounds=np.arange(595), deciles=np.arange(10.0))
    monkeypatch.setattr(c, "SIB_DENSITY", reordered)
    with pytest.raises(RuntimeError, match="density.npz lists the slides differently"):
        c.load_inputs()


def test_check_topocyte_accepts_the_pinned_checkout_and_needs_pythonpath(monkeypatch):
    c.check_topocyte()
    monkeypatch.delenv("PYTHONPATH")
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        c.check_topocyte()


def test_betti_inner_product_of_class_curves_is_the_sum_of_minima():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(1, 5000, 300), rng.uniform(1, 5000, 170)
    G = betti_inner_product_matrix([cv.class_curve(a), cv.class_curve(b)], dim=0)
    assert np.isclose(G[0, 1], cv.sum_of_minima(a, b), rtol=1e-12)
    assert np.isclose(G[0, 0], cv.sum_of_minima(a, a), rtol=1e-12)


def test_decile_counts_keep_a_cell_at_its_own_threshold():
    rho = np.array([1.0, 2.0, 2.0, 3.0])
    classes = np.array([0, 1, 1, 0])
    bounds = np.array([0, 4])
    deciles = np.array([1.0, 2.0, 3.0] + [9.0] * 7)
    counts = cv.decile_counts(rho, classes, bounds, deciles)
    assert counts[:3, 0, :2].tolist() == [[2, 2], [1, 2], [1, 0]]


def test_decile_gate_refuses_counts_that_differ_from_composition(tmp_path, monkeypatch):
    kept = np.arange(10 * 2 * 4).reshape(10, 2, 4)
    (tmp_path / "results").mkdir()
    np.savez(tmp_path / "results/composition.npz", kept_by_group=kept, slide_ids=np.array(["a", "b"]))
    monkeypatch.setattr(c, "CELLTYPE", tmp_path)
    cv.decile_gate(kept.copy(), ["a", "b"])
    off = kept.copy()
    off[3, 1, 2] += 1
    with pytest.raises(RuntimeError, match="kept_by_group"):
        cv.decile_gate(off, ["a", "b"])
    with pytest.raises(RuntimeError, match="composition.npz lists the slides differently"):
        cv.decile_gate(kept, ["b", "a"])
