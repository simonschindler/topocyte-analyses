"""Tests of the class-curve fPCA study.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``progress.md``, Rulings). The input tests
read the sibling studies' saved results, read only; the fold tests use
synthetic curves.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocyte.ph import FPCA, betti_inner_product_matrix

import common as c
import curves as cv
import folds


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


def _rho(rng: np.random.Generator, far: bool = False) -> np.ndarray:
    """One slide's densities of one class: 200 cells at 1,000-2,000 per mm², or, far away, 2,000 at 10,000-20,000."""
    return rng.uniform(10_000, 20_000, 2000) if far else rng.uniform(1000, 2000, 200)


def _grams(rhos: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Each class's exact Gram matrix of the class curves of ``rhos[class]``, one density array per slide."""
    return {name: betti_inner_product_matrix([cv.class_curve(r) for r in rhos[name]], dim=0) for name in c.CLASSES}


def test_block_fpca_matches_topocyte_fpca_transform():
    rng = np.random.default_rng(1)
    curves = [cv.class_curve(rng.uniform(1, 5000, rng.integers(50, 300))) for _ in range(12)]
    G = betti_inner_product_matrix(curves, dim=0)
    for normalization in c.NORMALIZATIONS:
        K = folds.analysis_kernel(G, normalization)  # of all 12 curves, then sliced, as in curve_space
        oracle = FPCA(dim=0, n_components=9, normalization=normalization, eigen_solver="dense").fit(curves[:9])
        fitted, projected = oracle.result_.scores, oracle.transform(curves[9:])
        block = folds.BlockFPCA().fit(K[:9, :9])
        assert block.scores_.shape == fitted.shape == (9, 8)  # centring leaves 8 components of 9 curves
        np.testing.assert_allclose(block.scores_, fitted, rtol=0, atol=1e-8 * np.abs(fitted).max())
        np.testing.assert_allclose(block.transform(K[:9, :9]), fitted, rtol=0, atol=1e-8 * np.abs(fitted).max())
        np.testing.assert_allclose(block.transform(K[9:, :9]), projected, rtol=0, atol=1e-8 * np.abs(projected).max())


def test_mfa_transform_of_the_fit_rows_reproduces_the_fit():
    rng = np.random.default_rng(2)
    # Eight blocks shaped like fPCA scores: nine columns of falling SD, each block on its own scale.
    blocks = [rng.normal(size=(100, 9)) * np.geomspace(1, 0.01, 9) * 10.0 ** rng.uniform(-3, 3) for _ in range(8)]
    mfa = folds.MFA().fit(blocks)
    assert mfa.scores_.shape == (100, 51)
    atol = 1e-10 * np.abs(mfa.scores_).max()
    np.testing.assert_allclose(mfa.transform(blocks), mfa.scores_, rtol=0, atol=atol)
    # Ten of the rows alone: transform uses the fitted weights and means, not those of its input.
    np.testing.assert_allclose(mfa.transform([b[:10] for b in blocks]), mfa.scores_[:10], rtol=0, atol=atol)
    np.testing.assert_allclose(mfa.scores_, c.celltype.mfa(blocks, 51)[0], rtol=0, atol=atol)
    # 20 slides' centred scores have rank 19: the 20th component would be rounding noise, and is not kept.
    assert folds.MFA().fit([b[:20] for b in blocks]).scores_.shape == (20, 19)


def test_block_distances_are_squared_distances_to_the_group_mean():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 5))  # twelve vectors for curves; K is their Gram matrix
    labels = np.array([0, 1] * 6)
    train = np.array([11, 0, 1, 2, 3, 5, 7, 8])
    # The mean over the training slides of the slide's group, the slide included.
    expected = [np.sum((X[f] - X[train[labels[train] == labels[f]]].mean(axis=0)) ** 2) for f in train]
    np.testing.assert_allclose(folds.block_distances(X @ X.T, train, labels), expected, rtol=1e-10)


def test_outlier_filter_drops_a_far_slide_and_keeps_small_and_flat_groups():
    rng = np.random.default_rng(4)
    # Group 0: slides 0-9, slide 3 far away. Group 1: slides 10-11, far apart. Group 2: slides 12-18, identical but
    # for the far slide 18, so that the group's MAD is 0. Slide 19, of group 0 and far away, is not a training slide.
    labels = np.array([0] * 10 + [1] * 2 + [2] * 7 + [0])
    far, flat = {3, 11, 18, 19}, range(12, 18)
    rhos = {}
    for name in c.CLASSES:
        same = _rho(rng)
        rhos[name] = [same if s in flat else _rho(rng, far=s in far) for s in range(20)]
    train = rng.permutation(19)  # in any order: the mask follows train
    keep = folds.outlier_keep(_grams(rhos), train, labels)
    assert keep.shape == train.shape
    assert train[~keep].tolist() == [3]


def test_cosine_refuses_a_zero_curve_with_its_slide():
    rng = np.random.default_rng(5)
    rhos = {name: [_rho(rng) for _ in range(6)] for name in c.CLASSES}
    rhos["immune"][3] = np.array([])  # slide 3 has no immune cells: its immune curve is 0
    grams = _grams(rhos)
    assert np.array_equal(folds.analysis_kernel(grams["immune"], "none"), grams["immune"])
    with pytest.raises(ValueError, match="slide 3"):
        folds.analysis_kernel(grams["immune"], "cosine")
    # Through the eight blocks, the error names the class too, whether the slide is a training slide or another.
    with pytest.raises(ValueError, match="immune.*slide 3"):
        folds.outlier_keep(grams, np.arange(6), np.zeros(6, dtype=int))
    with pytest.raises(ValueError, match="immune.*slide 3"):
        folds.curve_space(grams, np.array([0, 1, 2, 4, 5]), np.array([3]))


def test_curve_space_scores_a_fit_slide_alike_as_another_slide():
    rng = np.random.default_rng(6)
    rhos = {name: [rng.uniform(1, 5000, rng.integers(50, 300)) for _ in range(24)] for name in c.CLASSES}
    fit_idx, other_idx = np.arange(4, 24), np.array([0, 1, 4, 13])
    Z_fit, Z_other = folds.curve_space(_grams(rhos), fit_idx, other_idx)
    assert Z_fit.shape == (20, 19) and Z_other.shape == (4, 19)  # 72 block columns; 20 slides leave rank 19
    np.testing.assert_allclose(Z_other[2:], Z_fit[[0, 9]], rtol=0, atol=1e-8 * np.abs(Z_fit).max())
