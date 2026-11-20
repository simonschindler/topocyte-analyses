"""Unit tests of the PLS study's helpers, on synthetic inputs (no cohort)."""

from __future__ import annotations

import numpy as np

import improve
import pls_study


def test_target_weights_by_mode():
    y = np.array([0, 0, 0, 0, 1, 2, 2])
    assert np.allclose(pls_study.target_weights(y, "plain")[:3], [1, 1, 1])
    assert np.allclose(pls_study.target_weights(y, "sqrt")[:3], [1 / 2, 1, 1 / np.sqrt(2)])
    assert np.allclose(pls_study.target_weights(y, "inverse")[:3], [1 / 4, 1, 1 / 2])
    assert pls_study.target_weights(y, "sqrt").shape == (8,)


def test_pls_view_matches_the_improvement_projector():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(8), [10, 10, 8, 8, 6, 6, 6, 6])
    X = rng.normal(size=(y.size, 12)) + y[:, None] * 0.3
    ours = pls_study.PLSView(4).fit(X, y).transform(X)
    theirs = improve.Projector("pls", 4).fit(X, y).transform(X)
    assert np.allclose(ours, theirs)


def test_pls_view_without_whitening_keeps_scale():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(8), 8)
    X = rng.normal(size=(y.size, 6)) * np.array([100, 1, 1, 1, 1, 1])
    view = pls_study.PLSView(2, whiten=False).fit(X, y)
    assert np.allclose(view.sd, 1.0)


def test_axis_block_shares_follow_the_weights():
    vt = np.eye(6)  # six components, each one column of the weighted concatenation
    rotations = np.zeros((6, 2))
    rotations[0, 0] = 1.0
    rotations[4, 1] = 2.0
    shares = pls_study.axis_block_shares(rotations, comp_sd=np.ones(6), vt=vt, block_size=3)
    assert shares.shape == (2, 2)
    assert np.allclose(shares, [[1, 0], [0, 1]])


def test_axis_block_shares_divide_by_component_sd():
    vt = np.eye(4)
    rotations = np.array([[1.0], [0.0], [1.0], [0.0]])
    shares = pls_study.axis_block_shares(rotations, comp_sd=np.array([1.0, 1.0, 2.0, 1.0]), vt=vt, block_size=2)
    assert np.allclose(shares, [[0.8, 0.2]])  # 1^2 : (1/2)^2


def test_feature_inputs_keep_every_column():
    rng = np.random.default_rng(2)
    inputs = {"everything": rng.normal(size=(5, 200)), "groups": rng.normal(size=(5, 50)),
              "D+C": rng.normal(size=(5, 51)), "D+C + everything50": rng.normal(size=(5, 101)), "D": rng.normal(size=(5, 11))}
    configs = {cfg["input"]: cfg for cfg in pls_study.configurations(inputs) if cfg["part"] == "inputs"}
    assert configs["D+C"]["k_in"] == 51 and configs["D+C + everything50"]["k_in"] == 101 and configs["D"]["k_in"] == 11
    assert configs["groups"]["k_in"] == 50


def test_axis_block_contributions_are_variance_shares():
    rng = np.random.default_rng(3)
    Xw = rng.normal(size=(40, 4)) * np.array([1.0, 1.0, 3.0, 3.0])  # two blocks of two columns
    vt = np.eye(4)
    rotations = np.array([[1.0], [0.0], [1.0], [0.0]])  # the axis reads column 0 and column 2 equally
    shares = pls_study.axis_block_contributions(Xw, rotations, comp_sd=np.ones(4), vt=vt, block_size=2)
    v0, v2 = Xw[:, 0].var(), Xw[:, 2].var()
    assert np.allclose(shares, [[v0 / (v0 + v2), v2 / (v0 + v2)]])


def test_pls_view_counts_its_convergence_warnings():
    rng = np.random.default_rng(4)
    y = np.repeat(np.arange(8), 6)
    view = pls_study.PLSView(2).fit(rng.normal(size=(y.size, 5)), y)
    assert view.n_warnings >= 0
