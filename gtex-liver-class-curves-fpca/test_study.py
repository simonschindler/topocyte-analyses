"""Tests of the class-curve fPCA study.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``progress.md``, Rulings). The input tests
read the sibling studies' saved results, read only; the fold and arm tests
use synthetic curves, but for one that runs the first development fold on
this study's ``results/gram.npz``; the readout and holdout tests use synthetic
probabilities, scored against scikit-learn's ``roc_auc_score``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from topocyte.ph import FPCA, betti_inner_product_matrix

import common as c
import curves as cv
import folds
import run_study as rs


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


def _group_grams(rng: np.random.Generator, labels: np.ndarray, far: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    """Grams of synthetic class curves whose densities rise with the slide's group; the slides ``far`` lie far away."""
    return _grams(
        {
            name: [
                _rho(rng, far=True) if s in far else rng.uniform(1000 + 200 * g, 2000 + 200 * g, 100)
                for s, g in enumerate(labels)
            ]
            for name in c.CLASSES
        }
    )


def _two_curve_grams(rng: np.random.Generator, patterns: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Grams in which each slide carries one of two class curves per class: ``patterns[class][slide]`` says which.

    Two distinct curves give each of a class's blocks (raw and cosine) centred rank 1, and the two blocks' scores
    are collinear: the MFA has one column per distinct pattern.
    """
    grams = {}
    for name, pattern in zip(c.CLASSES, patterns, strict=True):
        pair = betti_inner_product_matrix([cv.class_curve(_rho(rng)), cv.class_curve(_rho(rng, far=True))], dim=0)
        grams[name] = pair[np.ix_(pattern, pattern)]
    return grams


def _fold_study() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An outer fold of 80 synthetic slides, 10 per group: Grams, D+C, labels, training and test slides.

    All eight groups, because choose_k's macro AUC needs the seven conditions. The test slides are two of each group.
    Group 0 makes a filter that counted the test slides decide by their labels: its training slides are four copies
    of one curve (slides 1, 2, 4, 6), three of another (7-9) and the far slide 3, which the filter drops because the
    group's MAD is positive. Its test slides 0 and 5 are two more copies of the first curve: counted under their own
    label, they would make the MAD 0 and keep slide 3.
    """
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(8), 10)
    rhos = {}
    for name in c.CLASSES:
        first, second = rng.uniform(1000, 2000, 100), rng.uniform(1000, 2000, 100)
        rhos[name] = [rng.uniform(1000 + 200 * g, 2000 + 200 * g, 100) for g in labels]
        rhos[name][:10] = [first, first, first, _rho(rng, far=True), first, first, first, second, second, second]
    DC = rng.normal(size=(80, 51)) + 0.5 * labels[:, None]
    test = np.flatnonzero(np.arange(80) % 5 == 0)
    return _grams(rhos), DC, labels, np.setdiff1d(np.arange(80), test), test


def test_fit_predict_is_the_celltype_logistic_regression():
    rng = np.random.default_rng(10)
    labels = np.repeat(np.arange(8), 6)
    X = rng.normal(size=(48, 5)) * 10.0 ** rng.uniform(-3, 3, 5) + labels[:, None]  # columns on very different scales
    X[:, 2] = 7.0  # a constant column, whose SD of 0 counts as 1
    split = (np.arange(48) % 2)[None, :]
    split[0, labels == 3] = 1  # fold 1's training slides lack group 3: its probabilities go through model.classes_
    _, per_repeat, _ = c.celltype.cv_probabilities(X, labels, split)
    for f in (0, 1):
        test = split[0] == f
        np.testing.assert_array_equal(rs.fit_predict(X[~test], labels[~test], X[test]), per_repeat[0][test])
    assert np.all(per_repeat[0][split[0] == 1, 3] == 0)


def test_test_labels_never_change_test_predictions():
    grams, DC, labels, train, test = _fold_study()
    fold = rs.outer_fold(grams, DC, labels, train, test, seed=1000)
    assert fold.keys() == {"keep", "k", *rs.ARMS}
    assert train[~fold["keep"] & (labels[train] == 0)].tolist() == [3]  # in group 0, the filter drops slide 3
    assert fold["k"] in c.K_GRID
    for arm in rs.ARMS:
        assert fold[arm].shape == (16, 8)
        np.testing.assert_allclose(fold[arm].sum(axis=1), 1.0, rtol=1e-12)
    permuted = labels.copy()
    permuted[test] = (labels[test] + 1) % 8  # a permutation of the test labels (two per group) that changes each one
    assert sorted(permuted[test]) == sorted(labels[test]) and np.all(permuted[test] != labels[test])
    again = rs.outer_fold(grams, DC, permuted, train, test, seed=1000)
    assert again.keys() == fold.keys()
    for key in fold:
        np.testing.assert_array_equal(again[key], fold[key])


def test_hidden_test_labels_leave_the_real_first_fold_unchanged():
    """On the real Grams and D+C, the first development fold's filter, k and arms are the same when its test slides'
    labels are -1, as the holdout stage hides the holdout slides'."""
    inputs = c.load_inputs()
    grams = rs.load_grams(inputs["slide_ids"])
    labels, dev = inputs["labels"], np.flatnonzero(inputs["dev"])
    split = c.celltype.make_folds(labels[dev], rs.N_REPEATS, rs.N_SPLITS, seed=c.DEV_FOLD_SEED)
    train, test = dev[split[0] != 0], dev[split[0] == 0]
    hidden = labels.copy()
    hidden[test] = -1
    fold = rs.outer_fold(grams, inputs["DC"], labels, train, test, seed=1000)
    assert 0 < np.sum(~fold["keep"]) < train.size and fold["k"] in c.K_GRID
    again = rs.outer_fold(grams, inputs["DC"], hidden, train, test, seed=1000)
    assert again.keys() == fold.keys()
    for key in fold:
        np.testing.assert_array_equal(again[key], fold[key])


def test_outer_fold_fits_each_arm_on_its_training_slides(monkeypatch):
    grams, DC, labels, train, test = _fold_study()
    chosen = []

    def choose_six(grams, slides, labels, seed):
        chosen.append((slides, seed))
        return 6

    monkeypatch.setattr(rs, "choose_k", choose_six)
    fold = rs.outer_fold(grams, DC, labels, train, test, seed=1000)
    np.testing.assert_array_equal(fold["keep"], folds.outlier_keep(grams, train, labels))
    kept = train[fold["keep"]]  # T': the training slides that the filter keeps
    assert [(slides.tolist(), seed) for slides, seed in chosen] == [(kept.tolist(), 1000)]  # k is chosen on T'
    assert fold["k"] == 6
    Z_kept, Z_test = folds.curve_space(grams, kept, test)
    for arm, Z_fit, Z_out, y in [
        ("curves", Z_kept[:, : fold["k"]], Z_test[:, : fold["k"]], labels[kept]),
        ("curves4", Z_kept[:, :4], Z_test[:, :4], labels[kept]),
        ("dc", DC[kept], DC[test], labels[kept]),
        ("dc_unfiltered", DC[train], DC[test], labels[train]),
    ]:
        np.testing.assert_array_equal(fold[arm], rs.fit_predict(Z_fit, y, Z_out), err_msg=arm)
    assert not np.array_equal(fold["dc"], fold["dc_unfiltered"])  # the filter's slide makes a difference


def test_choose_k_caps_the_grid_at_the_columns_available(monkeypatch):
    rng = np.random.default_rng(8)
    labels = np.repeat(np.arange(8), 10)
    train = np.arange(80)
    tried = []
    fit_predict = rs.fit_predict

    def spy(Z_train, y_train, Z_test):
        tried.append(Z_train.shape[1])
        return fit_predict(Z_train, y_train, Z_test)

    monkeypatch.setattr(rs, "fit_predict", spy)
    # Every block has centred rank 1; four independent patterns leave the MFA four columns.
    patterns = [rng.permutation(np.repeat([0, 1], 40)) for _ in c.CLASSES]
    grams = _two_curve_grams(rng, patterns)
    columns = folds.curve_space(grams, train, train[:1])[0].shape[1]
    assert columns == 4
    k = rs.choose_k(grams, train, labels, seed=1000)
    assert k in c.K_GRID and k <= columns
    assert tried == [4] * 5  # in each inner fold, only the k the columns allow
    # One pattern for every class leaves a single column, fewer than the grid's smallest k: choose_k returns it.
    tried.clear()
    grams = _two_curve_grams(rng, [patterns[0]] * 4)
    assert folds.curve_space(grams, train, train[:1])[0].shape[1] == 1
    assert rs.choose_k(grams, train, labels, seed=1000) == 1
    assert tried == [1] * 5
    # Random curves give every inner fold 51 columns; one inner fold cut to six caps the grid of all five at six.
    tried.clear()
    curve_space, spaces = folds.curve_space, []

    def first_cut_to_six(grams, fit_idx, other_idx):
        spaces.append(fit_idx)
        Z_fit, Z_other = curve_space(grams, fit_idx, other_idx)
        return (Z_fit[:, :6], Z_other[:, :6]) if len(spaces) == 1 else (Z_fit, Z_other)

    monkeypatch.setattr(folds, "curve_space", first_cut_to_six)
    assert rs.choose_k(_group_grams(rng, labels), train, labels, seed=1000) in (4, 6)
    assert tried == [4, 6] * 5


def test_choose_k_prefers_the_smaller_k_on_ties(monkeypatch):
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(8), 10)
    grams = _group_grams(rng, labels)
    train = np.arange(5, 80)  # group 0 keeps five slides, enough for five inner folds
    tried, spaces = [], []
    curve_space = folds.curve_space

    def uniform(Z_train, y_train, Z_test):
        tried.append(Z_train.shape[1])
        return np.full((len(Z_test), 8), 1 / 8)

    def spy(grams, fit_idx, other_idx):
        spaces.append((fit_idx, other_idx))
        return curve_space(grams, fit_idx, other_idx)

    monkeypatch.setattr(rs, "fit_predict", uniform)
    monkeypatch.setattr(folds, "curve_space", spy)
    monkeypatch.setattr(c.celltype, "macro_auc", lambda proba, labels: (0.5, np.full(7, 0.5)))
    assert rs.choose_k(grams, train, labels, seed=1000) == 4
    assert sorted(set(tried)) == list(c.K_GRID)  # every k was tried, and all tied
    # The inner folds: StratifiedKFold(5, shuffle=True, random_state=seed) on the slides given.
    inner = StratifiedKFold(5, shuffle=True, random_state=1000).split(np.zeros(75), labels[train])
    assert len(spaces) == 5
    for (fit_idx, other_idx), (fit, out) in zip(spaces, inner, strict=True):
        assert fit_idx.tolist() == train[fit].tolist() and other_idx.tolist() == train[out].tolist()


def _development_inputs() -> dict:
    """48 synthetic slides, 6 per group; 7 of them (one each of groups 0-3 and 5-7) are holdout slides."""
    labels = np.tile(np.arange(8), 6)
    holdout = np.arange(48) % 7 == 3
    return {"labels": labels, "dev": ~holdout, "DC": np.zeros((48, 51)), "slide_ids": [f"s{i}" for i in range(48)]}


def _fake_outer_fold(dev_positions: np.ndarray, seeds: list[int]):
    """An outer fold that drops the training slides at positions divisible by 3, returns its seed as k, and gives
    each test slide a row that names its position and the arm."""

    def outer_fold(grams, DC, labels, train, test, seed):
        seeds.append(seed)
        assert not np.intersect1d(train, test).size and np.array_equal(np.union1d(train, test), dev_positions)
        fold = {"keep": train % 3 != 0, "k": seed}
        for i, arm in enumerate(rs.ARMS):
            fold[arm] = np.zeros((len(test), 8))
            fold[arm][:, 0] = test / 1000 + i / 10
        return fold

    return outer_fold


def _write_reference(folder, dc_unfiltered: np.ndarray) -> None:
    """The cell-type study's ``improve_dev.npz`` with ``dc_unfiltered`` in every repeat, averaged as it averaged."""
    (folder / "results").mkdir(exist_ok=True)
    mean = np.repeat(dc_unfiltered[None], 20, axis=0).mean(axis=0)
    np.savez(folder / "results/improve_dev.npz", **{"baseline:D+C-features@51": mean})


def test_run_development_files_each_outer_fold_under_its_slides(tmp_path, monkeypatch):
    inputs = _development_inputs()
    dev = np.flatnonzero(inputs["dev"])
    seeds = []
    monkeypatch.setattr(rs, "outer_fold", _fake_outer_fold(dev, seeds))
    expected = np.zeros((dev.size, 8))
    expected[:, 0] = dev / 1000 + 3 / 10  # dc_unfiltered, the fourth arm, in every repeat
    _write_reference(tmp_path, expected)
    monkeypatch.setattr(c, "CELLTYPE", tmp_path)
    monkeypatch.setattr(c, "RESULTS", tmp_path / "out")
    (tmp_path / "out").mkdir()
    summary = rs.run_development(inputs, {})
    assert seeds == [1000 + 5 * r + f for r in range(20) for f in range(5)]
    split = c.celltype.make_folds(inputs["labels"][dev], 20, 5, seed=1)
    dropped = (split[:, None, :] != np.arange(5)[None, :, None]) & (dev % 3 == 0)
    with np.load(tmp_path / "out/development.npz") as saved:
        for i, arm in enumerate(rs.ARMS):
            assert saved[arm].shape == (20, dev.size, 8)
            np.testing.assert_array_equal(saved[arm][:, :, 0], np.broadcast_to(dev / 1000 + i / 10, (20, dev.size)))
        np.testing.assert_array_equal(saved["k"], 1000 + 5 * np.arange(20)[:, None] + np.arange(5))
        np.testing.assert_array_equal(saved["dropped"], dropped)
        np.testing.assert_array_equal(saved["folds"], split)
        assert saved["slide_ids"].tolist() == [inputs["slide_ids"][i] for i in dev]
    assert summary["dc_gate"]["max_abs_difference"] == 0.0
    assert summary["dropped_per_fold"] == dropped.sum(axis=2).tolist()
    assert summary["chosen_k"] == {str(seed): 1 for seed in seeds}


def test_run_development_refuses_dc_unlike_the_celltype_study(tmp_path, monkeypatch):
    inputs = _development_inputs()
    dev = np.flatnonzero(inputs["dev"])
    monkeypatch.setattr(rs, "outer_fold", _fake_outer_fold(dev, []))
    reference = np.zeros((dev.size, 8))
    reference[:, 0] = dev / 1000 + 3 / 10
    reference[5, 0] += 1e-7  # within np.allclose's default rtol=1e-5 of 0.3: only an absolute tolerance refuses it
    _write_reference(tmp_path, reference)
    monkeypatch.setattr(c, "CELLTYPE", tmp_path)
    monkeypatch.setattr(c, "RESULTS", tmp_path / "out")
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match="D\\+C"):
        rs.run_development(inputs, {})
    assert not (tmp_path / "out/development.npz").exists()


def test_dc_gate_allows_an_absolute_difference_of_1e_12_only():
    reference = np.full((3, 8), 0.5)
    assert rs.dc_gate(reference + 5e-13, reference) < 1e-12
    for gap in (2e-12, 1e-7):  # 1e-7 lies within np.allclose's default rtol=1e-5 of 0.5
        with pytest.raises(RuntimeError, match="D\\+C differs"):
            rs.dc_gate(reference + gap, reference)
    with pytest.raises(RuntimeError, match="shape"):
        rs.dc_gate(reference[:2], reference)


def _macro_auc_oracle(proba: np.ndarray, labels: np.ndarray) -> float:
    """The macro AUC by scikit-learn: each of the seven conditions one-vs-rest among the condition slides."""
    scored = labels < 7
    return float(np.mean([roc_auc_score(labels[scored] == g, proba[scored, g]) for g in range(7)]))


def _interval_oracle(a: np.ndarray, b: np.ndarray, labels: np.ndarray, draws: np.ndarray) -> list[float]:
    """The 2.5 and 97.5 percentiles of the macro AUC difference ``a`` minus ``b`` over the bootstrap ``draws``."""
    boot = [_macro_auc_oracle(a[i], labels[i]) - _macro_auc_oracle(b[i], labels[i]) for i in draws]
    return np.percentile(boot, [2.5, 97.5]).tolist()


@pytest.mark.parametrize(
    ("dev_lo", "dev_hi", "holdout_diff", "expected"),
    [
        (0.01, 0.05, 0.02, "beats D+C"),
        (0.01, 0.05, 0.0, "not confirmed"),
        (0.01, 0.05, -0.03, "not confirmed"),
        (-0.02, 0.03, 0.1, "level with D+C"),  # a positive holdout difference does not lift it
        (0.0, 0.03, 0.1, "level with D+C"),  # the boundary: an interval that ends at 0 contains it
        (-0.03, 0.0, 0.1, "level with D+C"),
        (-0.05, -0.01, 0.1, "below D+C"),
    ],
)
def test_verdict_follows_the_rule_fixed_in_advance(dev_lo, dev_hi, holdout_diff, expected):
    assert rs.verdict(dev_lo, dev_hi, holdout_diff) == expected


def test_verdict_refuses_what_the_rule_cannot_read():
    for dev_lo, dev_hi, holdout_diff in [(0.05, 0.01, 0.02), (np.nan, 0.05, 0.02), (0.01, 0.05, np.nan)]:
        with pytest.raises(ValueError):
            rs.verdict(dev_lo, dev_hi, holdout_diff)


def test_paired_interval_is_zero_for_identical_arms():
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(8), 5)
    proba = rng.dirichlet(np.ones(8), size=40)
    draws = c.celltype.stratified_bootstrap(labels, 30, rng)
    assert rs.paired_interval(proba, proba.copy(), labels, draws) == (0.0, 0.0, 0.0)


def test_paired_interval_is_the_macro_auc_difference_with_its_percentiles():
    rng = np.random.default_rng(12)
    labels = np.repeat(np.arange(8), 5)
    a, b = rng.dirichlet(np.ones(8), size=40), rng.dirichlet(np.ones(8), size=40)
    a[np.arange(40), labels] += 0.3  # a leans towards each slide's own group
    draws = c.celltype.stratified_bootstrap(labels, 30, rng)
    diff, lo, hi = rs.paired_interval(a, b, labels, draws)
    assert diff == pytest.approx(_macro_auc_oracle(a, labels) - _macro_auc_oracle(b, labels), abs=1e-12)
    assert [lo, hi] == pytest.approx(_interval_oracle(a, b, labels, draws), abs=1e-12)


def _readouts_study(tmp_path, monkeypatch) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """A development run of 3 repeats on ``_development_inputs``, saved as the development stage saves it.

    Returns the development labels, each arm's per-repeat probabilities and ``development.json``'s metadata. The
    curves lean towards each slide's own group, D+C less, so that the arms differ.
    """
    inputs = _development_inputs()
    dev = np.flatnonzero(inputs["dev"])
    y = inputs["labels"][dev]
    rng = np.random.default_rng(14)
    proba = {arm: rng.dirichlet(np.ones(8), size=(3, dev.size)) for arm in rs.ARMS}
    proba["curves"][:, np.arange(dev.size), y] += 0.5
    proba["dc"][:, np.arange(dev.size), y] += 0.2
    metadata = {"description": "every outer fold", "n_slides": int(dev.size), "md5": {"run_study.py": "0" * 32}}
    (tmp_path / "development.json").write_text(json.dumps(metadata))
    np.savez(tmp_path / "development.npz", **proba, slide_ids=np.array([inputs["slide_ids"][i] for i in dev]))
    monkeypatch.setattr(c, "load_inputs", lambda: inputs)
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    monkeypatch.setattr(c, "N_BOOT", 40)
    monkeypatch.setattr(rs, "DEV_GATE_AUC", round(_macro_auc_oracle(proba["dc_unfiltered"].mean(axis=0), y), 3))
    return y, proba, metadata


def test_readouts_score_the_repeat_averaged_probabilities_beside_the_development_metadata(tmp_path, monkeypatch):
    y, proba, metadata = _readouts_study(tmp_path, monkeypatch)
    rs.main(["readouts"])
    saved = json.loads((tmp_path / "development.json").read_text())
    assert {key: saved[key] for key in metadata} == metadata  # the development run's metadata, kept
    readouts = saved["readouts"]
    mean = {arm: p.mean(axis=0) for arm, p in proba.items()}
    scored = y < 7
    for arm in rs.ARMS:
        row = readouts["arms"][arm]
        assert row["macro_auc"] == pytest.approx(_macro_auc_oracle(mean[arm], y), abs=1e-12)
        assert row["auc_per_repeat"] == pytest.approx([_macro_auc_oracle(p, y) for p in proba[arm]], abs=1e-12)
        assert list(row["auc_per_condition"]) == list(c.celltype.CONDITIONS)
        per = [roc_auc_score(y[scored] == g, mean[arm][scored, g]) for g in range(7)]
        assert list(row["auc_per_condition"].values()) == pytest.approx(per, abs=1e-12)
    assert readouts["gate"]["macro_auc"] == pytest.approx(_macro_auc_oracle(mean["dc_unfiltered"], y), abs=1e-12)
    # The paired differences against D+C fitted on the cleaned training slides, on the pre-registered draws.
    draws = c.celltype.stratified_bootstrap(y, 40, np.random.default_rng(12345))
    for arm in ("curves", "curves4"):
        comparison = readouts["comparisons"][f"{arm} - dc"]
        expected = _macro_auc_oracle(mean[arm], y) - _macro_auc_oracle(mean["dc"], y)
        assert comparison["difference"] == pytest.approx(expected, abs=1e-12)
        assert comparison["interval"] == pytest.approx(_interval_oracle(mean[arm], mean["dc"], y, draws), abs=1e-12)


@pytest.mark.parametrize(("broken", "match"), [("gate", "macro AUC"), ("slides", "development slides")])
def test_readouts_write_nothing_unless_dc_and_the_slides_agree(tmp_path, monkeypatch, broken, match):
    y, proba, _ = _readouts_study(tmp_path, monkeypatch)
    if broken == "gate":
        monkeypatch.setattr(rs, "DEV_GATE_AUC", rs.DEV_GATE_AUC + 0.001)
    else:
        np.savez(tmp_path / "development.npz", **proba, slide_ids=np.array([f"x{i}" for i in range(y.size)]))
    before = (tmp_path / "development.json").read_text()
    with pytest.raises(RuntimeError, match=match):
        rs.main(["readouts"])
    assert (tmp_path / "development.json").read_text() == before


def test_holdout_stage_refuses_a_second_run(tmp_path, monkeypatch):
    scored = '{"verdict": {"verdict": "level with D+C"}}\n'
    (tmp_path / "holdout.json").write_text(scored)
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    monkeypatch.setattr(c, "load_inputs", lambda: pytest.fail("the holdout stage read its inputs"))
    with pytest.raises(RuntimeError, match="scored once"):
        rs.main(["holdout"])
    assert (tmp_path / "holdout.json").read_text() == scored
    assert [p.name for p in tmp_path.iterdir()] == ["holdout.json"]


def _holdout_inputs() -> dict:
    """64 synthetic slides, 8 per group; slides 8-15 and 40-47 (two per group) are holdout slides."""
    labels = np.tile(np.arange(8), 8)
    holdout = (np.arange(64) // 8) % 4 == 1
    return {
        "labels": labels,
        "dev": ~holdout,
        "holdout": holdout,
        "DC": np.zeros((64, 51)),
        "slide_ids": [f"s{i}" for i in range(64)],
    }


def _holdout_study(tmp_path, monkeypatch) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    """A holdout stage ready to run on ``_holdout_inputs``; returns the inputs, the arms' holdout probabilities and
    the outer folds called.

    - The outer fold returns those probabilities, ``k = 6``, and drops the development slides at positions divisible
      by 5. On the holdout, the curves separate the groups, ``k = 4`` inverts them, and D+C is random.
    - The cell-type study's ``improve_holdout.npz`` holds the unfiltered D+C as its second entry (the third in the
      real file), among others.
    - ``development.json``'s readouts put the curves' development interval above 0, ``k = 4``'s below.
    """
    inputs = _holdout_inputs()
    hold = np.flatnonzero(inputs["holdout"])
    y = inputs["labels"][hold]
    rng = np.random.default_rng(13)
    proba = {arm: rng.dirichlet(np.ones(8), size=hold.size) for arm in rs.ARMS}
    proba["curves"][np.arange(hold.size), y] += 1
    proba["curves4"][np.arange(hold.size), y] = 0
    calls = []

    def outer_fold(grams, DC, labels, train, test, seed):
        calls.append({"labels": labels.copy(), "train": train.copy(), "test": test.copy(), "seed": seed})
        return {"keep": train % 5 != 0, "k": 6, **{arm: p.copy() for arm, p in proba.items()}}

    results, celltype = tmp_path / "results", tmp_path / "celltype"
    results.mkdir()
    (celltype / "results").mkdir(parents=True)
    np.savez(
        celltype / "results/improve_holdout.npz",
        names=np.array(["everything@4", "D+C features", "all-cell@4"]),
        e0__proba=rng.dirichlet(np.ones(8), size=hold.size),
        e1__proba=proba["dc_unfiltered"],
        e2__proba=rng.dirichlet(np.ones(8), size=hold.size),
        holdout_rows=hold,
        y_holdout=y,
    )
    readouts = {
        "comparisons": {
            "curves - dc": {"difference": 0.03, "interval": [0.01, 0.05]},
            "curves4 - dc": {"difference": -0.03, "interval": [-0.05, -0.01]},
        }
    }
    (results / "development.json").write_text(json.dumps({"n_slides": 48, "readouts": readouts}))
    monkeypatch.setattr(c, "load_inputs", lambda: inputs)
    monkeypatch.setattr(c, "RESULTS", results)
    monkeypatch.setattr(c, "CELLTYPE", celltype)
    monkeypatch.setattr(c, "N_BOOT", 40)
    monkeypatch.setattr(rs, "load_grams", lambda slide_ids: {})
    monkeypatch.setattr(rs, "outer_fold", outer_fold)
    monkeypatch.setattr(rs, "HOLDOUT_GATE_AUC", round(_macro_auc_oracle(proba["dc_unfiltered"], y), 3))
    return inputs, proba, calls


def test_holdout_stage_fits_on_the_development_slides_and_scores_the_holdout_once(tmp_path, monkeypatch):
    inputs, proba, calls = _holdout_study(tmp_path, monkeypatch)
    labels = inputs["labels"]
    dev, hold = np.flatnonzero(inputs["dev"]), np.flatnonzero(inputs["holdout"])
    y = labels[hold]
    rs.main(["holdout"])
    # One fit: the development slides train with seed 999, the holdout slides are predicted, their labels hidden.
    assert len(calls) == 1
    call = calls[0]
    assert call["train"].tolist() == dev.tolist() and call["test"].tolist() == hold.tolist() and call["seed"] == 999
    assert np.array_equal(call["labels"][dev], labels[dev]) and np.all(call["labels"][hold] < 0)
    scored = json.loads((tmp_path / "results/holdout.json").read_text())
    for arm in rs.ARMS:
        assert scored["arms"][arm]["macro_auc"] == pytest.approx(_macro_auc_oracle(proba[arm], y), abs=1e-12)
    draws = c.celltype.stratified_bootstrap(y, 40, np.random.default_rng(12345))
    for arm in ("curves", "curves4"):
        comparison = scored["comparisons"][f"{arm} - dc"]
        expected = _macro_auc_oracle(proba[arm], y) - _macro_auc_oracle(proba["dc"], y)
        assert comparison["difference"] == pytest.approx(expected, abs=1e-12)
        assert comparison["interval"] == pytest.approx(_interval_oracle(proba[arm], proba["dc"], y, draws), abs=1e-12)
    # The curves' development interval lies above 0 and they lead on the holdout; k = 4's numbers would not do.
    assert scored["verdict"]["verdict"] == "beats D+C"
    assert scored["verdict"]["development_interval"] == [0.01, 0.05]
    assert scored["verdict"]["holdout_difference"] == scored["comparisons"]["curves - dc"]["difference"]
    assert scored["fit"]["chosen_k"] == 6 and scored["fit"]["dropped"] == np.sum(dev % 5 == 0)
    assert scored["gate"]["max_abs_difference"] == 0.0
    with np.load(tmp_path / "results/holdout.npz") as saved:
        for arm in rs.ARMS:
            np.testing.assert_array_equal(saved[arm], proba[arm])
        assert saved["k"] == 6
        np.testing.assert_array_equal(saved["dropped"], dev % 5 == 0)
        assert saved["slide_ids"].tolist() == [f"s{i}" for i in hold]
        np.testing.assert_array_equal(saved["labels"], y)
    # Once written, the holdout refuses a second run before any fit.
    with pytest.raises(RuntimeError, match="scored once"):
        rs.main(["holdout"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("broken", "match"),
    [
        ("probabilities", "D\\+C differs"),
        ("slides", "holdout slides"),
        ("labels", "holdout slides"),
        ("auc", "macro AUC"),
        ("readouts", "readouts"),
    ],
)
def test_holdout_stage_writes_nothing_unless_every_gate_passes(tmp_path, monkeypatch, broken, match):
    _, _, calls = _holdout_study(tmp_path, monkeypatch)
    path = tmp_path / "celltype/results/improve_holdout.npz"
    with np.load(path) as saved:
        reference = dict(saved)
    if broken == "probabilities":
        reference["e1__proba"][4, 1] += 1e-7  # within np.allclose's default rtol: only the absolute tolerance refuses
    elif broken == "slides":
        reference["holdout_rows"] = reference["holdout_rows"][::-1]
    elif broken == "labels":
        reference["y_holdout"][0] = (reference["y_holdout"][0] + 1) % 8
    elif broken == "auc":
        monkeypatch.setattr(rs, "HOLDOUT_GATE_AUC", rs.HOLDOUT_GATE_AUC + 0.001)
    else:
        (tmp_path / "results/development.json").write_text(json.dumps({"n_slides": 48}))
    np.savez(path, **reference)
    with pytest.raises(RuntimeError, match=match):
        rs.main(["holdout"])
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["development.json"]
    if broken in ("slides", "labels", "readouts"):
        assert calls == []  # refused before the fit
