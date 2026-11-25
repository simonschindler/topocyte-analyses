"""Tests of the Betti FPCR benchmark.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``requirements.txt``):

    $PY -m pytest -q -W error test_study.py

The tests marked ``needs_upstream`` read the pinned clone in ``upstream/``,
its pickles only through ``safe_pickle.load_pickle``, and are skipped until
``fetch_upstream.py`` has made it. The others use synthetic data.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import pathlib
import pickle
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest
import ripser
import topocyte as tc

import betti
import common as c
import estimators as est
import fetch_upstream
import inputs as inp
import matrices as mat
import oracles
import readouts as ro
import run_study as rs
import safe_pickle
import wasserstein as ws

needs_upstream = pytest.mark.skipif(not (c.UPSTREAM / ".git").exists(), reason="upstream not fetched")


class _Touch:
    """Pickles as the call ``pathlib.Path.touch(marker)``: an unpickler that resolves it creates the marker."""

    def __init__(self, marker: pathlib.Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return pathlib.Path.touch, (self.marker,)


def test_safe_pickle_refuses_a_global_outside_the_whitelist(tmp_path):
    marker = tmp_path / "ran"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Touch(marker)))
    with pytest.raises(pickle.UnpicklingError, match="pathlib"):
        safe_pickle.load_pickle(path)
    assert not marker.exists()


def test_safe_pickle_reads_numpy_arrays(tmp_path):
    # Protocol 4, as all 2,068 upstream pickles: Python 3.14's default, 5, pickles a contiguous array through
    # numpy._core.numeric._frombuffer, which is not one of the whitelisted globals.
    a = np.arange(6.0).reshape(3, 2)
    path = tmp_path / "a.pkl"
    path.write_bytes(pickle.dumps(a, protocol=4))
    assert np.array_equal(safe_pickle.load_pickle(path), a)


def test_check_topocyte_needs_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        c.check_topocyte()


def test_r2_is_one_minus_sse_over_sst():
    y, p = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])
    assert c.r2(y, p) == pytest.approx(1 - 1.25 / np.sum((y - y.mean()) ** 2))


def _stand_in_upstream(tmp_path: pathlib.Path) -> pathlib.Path:
    """A fresh `git init` repository with one commit, standing in for upstream: a clean tree at another commit."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "README.md").write_text("a stand-in for upstream\n")
    config = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", "README.md"], ["commit", "-q", "--no-verify", "-m", "a"]):
        subprocess.run(["git", "-C", str(repo), *config, *args], check=True, capture_output=True)
    return repo


def test_ensure_upstream_refuses_a_clone_at_another_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_upstream, "UPSTREAM", _stand_in_upstream(tmp_path))
    with pytest.raises(RuntimeError, match="fe7d652"):
        fetch_upstream.ensure_upstream()


def test_check_upstream_refuses_a_repository_at_another_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "UPSTREAM", _stand_in_upstream(tmp_path))
    with pytest.raises(RuntimeError, match="fe7d652"):
        c.check_upstream()
    with pytest.raises(RuntimeError, match="fe7d652"):  # not a clone at all, as before fetch_upstream.py has run
        c.check_upstream(tmp_path / "absent")


def test_check_upstream_accepts_a_clean_pin_and_refuses_a_changed_tree(tmp_path, monkeypatch):
    repo = _stand_in_upstream(tmp_path)
    monkeypatch.setattr(c, "UPSTREAM_SHA", c._git(repo, "rev-parse", "HEAD"))  # the stand-in's commit is the pin
    c.check_upstream(repo)
    (repo / "untracked.txt").write_text("not upstream's\n")
    with pytest.raises(RuntimeError, match="tree changed: True"):
        c.check_upstream(repo)


def test_fluid_sim_ids_follow_upstreams_formula():
    ids = inp.fluid_sim_ids()
    assert len(ids) == len(set(ids)) == 400 and ids[:5] == [0, 22, 45, 67, 89] and ids[-3:] == [7954, 7977, 7999]


def test_fluid_ln_re_is_upstreams_reynolds_number():
    assert inp.fluid_ln_re(np.array([0, 7999])) == pytest.approx(np.array([4 / 3, 4 / 3 * 8.999]) * np.log(2))


def test_fluid_split_zero_is_upstreams_first_split():
    test, pool = inp.fluid_split(0)
    perm = np.random.default_rng(0).permutation(400)
    assert np.array_equal(test, perm[:100]) and np.array_equal(pool, perm[100:])


def test_rips_diagram_equals_ripser_as_pairs():
    X = np.random.default_rng(3).normal(size=(20, 2))
    ref = ripser.ripser(X)["dgms"]
    assert all(inp.same_pairs(mine, theirs) for mine, theirs in zip(inp.rips_diagram(X), ref, strict=True))


def test_same_pairs_sees_a_repairing():
    a = np.array([[0.0, 2.0], [1.0, 3.0]])
    b = np.array([[0.0, 3.0], [1.0, 2.0]])
    assert not inp.same_pairs(a, b) and inp.same_pairs(a, a[::-1])


def test_inputs_stage_records_a_failed_check_and_exits_1(tmp_path, monkeypatch):
    # A missing fluid diagram fails G5; the fluid summary, which reads every diagram, is then not attempted.
    monkeypatch.setattr(c, "check_topocyte", lambda: None)
    monkeypatch.setattr(c, "check_upstream", lambda: None)
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    monkeypatch.setattr(inp, "_g5", lambda: {"passed": False, "fluid_diagrams_missing": ["pd_sim0000_t1.pkl"]})
    monkeypatch.setattr(inp, "_fluid_summary", lambda: pytest.fail("the fluid summary read the diagrams"))
    with pytest.raises(SystemExit) as stop:
        inp.main()
    record = json.loads((tmp_path / "inputs.json").read_text())
    assert stop.value.code == 1 and record["passed"] is False and record["fluid"] is None


def _random_diagram(rng: np.random.Generator, n_finite: int, n_essential: int) -> np.ndarray:
    """``n_finite`` bars with births and lengths U(0, 1), and ``n_essential`` with births U(0, 1), rows shuffled."""
    births = rng.uniform(0, 1, n_finite + n_essential)
    deaths = np.concatenate((births[:n_finite] + rng.uniform(0, 1, n_finite), np.full(n_essential, np.inf)))
    return rng.permutation(np.column_stack((births, deaths)))


def _brute_force_w2(a: np.ndarray, b: np.ndarray) -> float:
    """W2 by enumeration, as an oracle independent of ``wasserstein``.

    The finite cost is the cheapest of all permutations of the augmented matrix in which each bar has a diagonal slot
    of its own (inf elsewhere); the essential cost is the cheapest of all pairings of the essential births.
    """
    fa, fb = a[np.isfinite(a[:, 1])], b[np.isfinite(b[:, 1])]
    n, m = len(fa), len(fb)
    C = np.full((n + m, m + n), np.inf)
    C[:n, :m] = ((fa[:, None, :] - fb[None, :, :]) ** 2).sum(axis=2)
    C[np.arange(n), m + np.arange(n)] = (fa[:, 1] - fa[:, 0]) ** 2 / 2
    C[n + np.arange(m), np.arange(m)] = (fb[:, 1] - fb[:, 0]) ** 2 / 2
    C[n:, m:] = 0
    permutations = np.array(list(itertools.permutations(range(n + m))), dtype=int)  # (1, 0) if n + m = 0
    finite = C[np.arange(n + m), permutations].sum(axis=1).min()
    ea, eb = a[np.isinf(a[:, 1]), 0], b[np.isinf(b[:, 1]), 0]
    pairings = np.array(list(itertools.permutations(range(len(eb)))), dtype=int)
    essential = ((ea - eb[pairings]) ** 2).sum(axis=1).min()
    return float(np.sqrt(finite + essential))


def test_w2_of_a_point_against_nothing_is_its_distance_to_the_diagonal():
    assert ws.w2(np.array([[1.0, 3.0]]), np.empty((0, 2))) == pytest.approx(2 / np.sqrt(2))


def test_w2_prefers_the_diagonal_when_it_is_cheaper():
    a, b = np.array([[0.0, 0.2]]), np.array([[5.0, 5.4]])
    assert ws.w2(a, b) == pytest.approx(np.sqrt(0.2**2 / 2 + 0.4**2 / 2))


def test_w2_matches_essential_births_in_sorted_order():
    a = np.array([[0.0, np.inf], [0.5, np.inf]])
    b = np.array([[0.4, np.inf], [0.1, np.inf]])
    assert ws.w2(a, b) == pytest.approx(np.sqrt(0.1**2 + 0.1**2))


def test_w2_is_infinite_for_unequal_essential_counts():
    assert ws.w2(np.array([[0.0, np.inf]]), np.empty((0, 2))) == np.inf


@pytest.mark.parametrize("bar", [[-np.inf, 1.0], [np.nan, 1.0], [np.inf, np.inf], [0.0, np.nan], [0.0, -np.inf]])
def test_w2_refuses_a_non_finite_birth_and_a_death_neither_finite_nor_inf(bar):
    with pytest.raises(ValueError, match="finite"):
        ws.w2(np.array([[0.0, 2.0], bar]), np.array([[0.5, 1.0]]))


def test_w2_matches_brute_force_on_small_diagrams():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n_essential = rng.integers(0, 3)
        a, b = (_random_diagram(rng, rng.integers(0, 4), n_essential) for _ in range(2))
        np.testing.assert_allclose(ws.w2(a, b), _brute_force_w2(a, b), rtol=1e-12, atol=0)


def test_w2_matrix_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(2)
    diagrams = [_random_diagram(rng, n, 1) for n in (0, 3, 5, 1, 4)]
    D = ws.w2_matrix(diagrams)
    assert D.shape == (5, 5) and np.array_equal(D, D.T) and not D.diagonal().any()
    for p, q in itertools.combinations(range(5), 2):
        assert D[p, q] == pytest.approx(ws.w2(diagrams[p], diagrams[q]), rel=1e-12, abs=0)


def test_w2_matrix_does_not_depend_on_n_jobs():
    rng = np.random.default_rng(3)
    diagrams = [_random_diagram(rng, n, 1) for n in rng.integers(0, 8, 12)]
    assert np.array_equal(ws.w2_matrix(diagrams, n_jobs=2), ws.w2_matrix(diagrams))


def _overlaps(diagrams: list[np.ndarray]) -> np.ndarray:
    """The matrix of ``oracles.overlap`` between every two of ``diagrams``."""
    return np.array([[oracles.overlap(a, b) for b in diagrams] for a in diagrams])


def test_curves_hold_the_finite_bars_shifted_and_refuse_a_nan_death():
    (curve,) = betti.curves([np.array([[1.0, np.inf], [0.0, 2.0], [-1.0, 0.5]])], 1, shift=3.0)
    assert (curve.dim, curve.essential.size) == (1, 0)
    assert np.array_equal(curve.births, [2.0, 3.0]) and np.array_equal(curve.deaths, [3.5, 5.0])
    with pytest.raises(ValueError, match="invalid bar"):
        betti.curves([np.array([[0.0, np.nan]])], 0)


def test_gram_is_the_sum_of_bar_overlaps():
    # Births and lengths U(0, 1) and one essential bar each; the two diagrams without a finite bar give zero rows.
    rng = np.random.default_rng(4)
    diagrams = [_random_diagram(rng, n, 1) for n in (0, 1, 4, 7, 3, 0, 5)]
    np.testing.assert_allclose(betti.gram(betti.curves(diagrams, 1), 1), _overlaps(diagrams), rtol=1e-12, atol=0)


def test_shift_makes_the_gram_full_line():
    # Births U(-2, 1) and lengths U(0, 3): some bars lie below 0, some cross it, some lie above it.
    rng = np.random.default_rng(5)
    diagrams = [3 * _random_diagram(rng, n, 1) - 2 for n in (6, 3, 0, 8, 5)]
    s = math.ceil(-min(d[:, 0].min() for d in diagrams))  # FLUID_SHIFT's rule: every value is then at least 0
    G, G10 = (betti.gram(betti.curves(diagrams, 0, shift), 0) for shift in (s, s + 10))
    np.testing.assert_allclose(G10, G, rtol=1e-12, atol=0)
    np.testing.assert_allclose(G, _overlaps(diagrams), rtol=1e-12, atol=0)
    assert not np.allclose(betti.gram(betti.curves(diagrams, 0), 0), G)  # unshifted, the parts below 0 are lost


def test_unshifted_negative_bars_lose_their_part_below_zero():
    # topocyte's rule: its inner product integrates over [0, inf), so the bar [-2, 1) counts as [0, 1).
    bar = np.array([[-2.0, 1.0]])
    assert tc.ph.betti_norm(tc.ph.BettiCurve(bar, dim=1), dim=1) == pytest.approx(1.0)
    assert math.sqrt(oracles.overlap(bar, bar)) == pytest.approx(math.sqrt(3))  # the full-line norm


def test_l2_matrix_is_the_overlap_distance():
    rng = np.random.default_rng(6)
    diagrams = [_random_diagram(rng, n, 1) for n in (0, 5, 2, 0, 7, 4)]
    curves = betti.curves(diagrams, 1)
    L, full = betti.l2_matrix(curves, 1), _overlaps(diagrams)
    # ‖f − g‖² = ⟨f, f⟩ + ⟨g, g⟩ − 2⟨f, g⟩; the two diagrams without a finite bar are at distance 0
    np.testing.assert_allclose(L, np.sqrt(full.diagonal()[:, None] + full.diagonal() - 2 * full), rtol=1e-9, atol=0)
    assert np.array_equal(L, L.T) and not L.diagonal().any()
    # every entry is topocyte's direct L² distance, not one from the Gram matrix
    assert np.array_equal(L, [[tc.ph.betti_l2_distance(f, g, dim=1) for g in curves] for f in curves])


def test_fpca_scores_reproduce_topocyte_fit_and_transform():
    # Gate G3's oracle: topocyte's own fit on the training curves and transform of the test curves.
    rng = np.random.default_rng(7)
    for h in (0, 1):
        train, test = np.split(rng.permutation(30), [20])
        sizes = rng.integers(1, 8, 30)
        sizes[[train[0], test[0]]] = 0  # a curve without a finite bar in each set
        curves = betti.curves([_random_diagram(rng, n, 1) for n in sizes], h)
        S_train, S_test = betti.fpca_scores(betti.gram(curves, h), train, test)
        model = tc.ph.FPCA(dim=h, eigen_solver="dense").fit([curves[i] for i in train])
        expected = (model.result_.scores, model.transform([curves[i] for i in test]))
        assert model.n_components_ == 19 and S_train.shape == (20, 19) and S_test.shape == (10, 19)
        for mine, theirs in zip((S_train, S_test), expected, strict=True):
            np.testing.assert_allclose(mine, theirs, rtol=0, atol=c.G3_TOL * np.abs(theirs).max())


def test_fpca_scores_of_an_all_empty_dimension_has_no_columns():
    # Review Focus 2: training curves without a finite bar give a rank-0 fPCA, with no division by √0.
    empty = [np.empty((0, 2)), np.array([[0.5, np.inf]]), np.array([[0.0, np.inf], [0.2, np.inf]])]
    diagrams = [*empty, np.array([[0.1, np.inf]]), np.array([[0.0, 1.0], [0.3, np.inf]])]
    G = betti.gram(betti.curves(diagrams, 1), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S_train, S_test = betti.fpca_scores(G, np.arange(3), np.arange(3, 5))
    assert S_train.shape == (3, 0) and S_test.shape == (2, 0)


def test_fpca_scores_returns_the_fitted_scores_of_its_training_curves():
    # 12 random diagrams and a copy of each with one death moved by 1e-5: the 12 smallest centred eigenvalues are
    # below 1e-6 of the largest, and their computed eigenvectors are not orthogonal to 1 to working precision. There
    # the projection's two constant terms decide whether a training curve, projected as a test curve, gets its
    # fitted score back.
    rng = np.random.default_rng(10)
    diagrams = [_random_diagram(rng, n, 1) for n in rng.integers(3, 9, 12)]
    copies = [d.copy() for d in diagrams]
    for d in copies:
        d[np.flatnonzero(np.isfinite(d[:, 1]))[0], 1] += 1e-5
    G = betti.gram(betti.curves(diagrams + copies, 1), 1)
    idx = rng.permutation(24)
    S_train, S_test = betti.fpca_scores(G, idx, idx)
    K = G[np.ix_(idx, idx)]
    r = tc.ph.fpca_from_gram(K, normalization="none")
    rho, tol = r.gram_eigenvalues, 1e-10 * np.abs(r.scores).max()
    assert rho.size == 23 and rho[-1] < 1e-6 * rho[0]
    assert np.array_equal(S_train, r.scores)
    np.testing.assert_allclose(S_test, r.scores, rtol=0, atol=tol)
    # the test sees the constant terms: the row without them (FPCA.transform's) misses by far more
    partial = (K - K.mean(axis=0)) @ (r.eigenvectors / np.sqrt(rho))
    assert np.abs(partial - r.scores).max() > 100 * tol


def _nw_scenario(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random distances on which Nadaraya-Watson reaches both of its fallbacks: ``(D_train, y, D_test)``.

    20 training samples lie at random symmetric distances U(0.5, 1.5) from one another, with targets U(1, 2). A 21st
    lies at U(20, 21) from them, with the target 1,000: at the smaller bandwidths all its LOO weights underflow, so
    its LOO estimate falls back to the mean of the other targets, and its target moves the choice of bandwidth. 9
    test samples lie like the first 20, and a 10th at 1,000 from every training sample, where every weight underflows.
    """
    A = rng.uniform(0.5, 1.5, (21, 21))
    A[:, -1] = rng.uniform(20, 21, 21)
    D_test = rng.uniform(0.5, 1.5, (10, 21))
    D_test[:, -1] = rng.uniform(20, 21, 10)
    D_test[-1] = 1e3
    return np.triu(A, 1) + np.triu(A, 1).T, np.append(rng.uniform(1, 2, 20), 1e3), D_test


def test_nw_matches_a_loop_transcription_of_upstreams_estimator():
    for seed in range(5):
        D_train, y, D_test = _nw_scenario(np.random.default_rng(seed))
        # the outlier's LOO weights all underflow at the smallest bandwidth: e^(-z²/2) = 0 for z = d/h > 38.6
        assert D_train[-1, :-1].min() / (0.1 * np.median(D_train[D_train > 0])) > 40
        pred, h = est.nw_fit_predict(D_train, y, D_test)
        expected, h_upstream = oracles.upstream_nw(D_train, y, D_test)
        assert h == h_upstream
        np.testing.assert_allclose(pred, expected, rtol=1e-12, atol=0)
        assert pred[-1] == y.mean()
    # all targets 0: every bandwidth's LOO error is exactly 0, and the first bandwidth, 0.1 m, wins in both
    D_train, y, D_test = _nw_scenario(np.random.default_rng(0))
    zero, m = np.zeros(len(y)), np.median(D_train[D_train > 0])
    assert est.nw_fit_predict(D_train, zero, D_test)[1] == oracles.upstream_nw(D_train, zero, D_test)[1] == 0.1 * m


def test_nw_falls_back_to_means_when_every_weight_underflows():
    rng = np.random.default_rng(11)
    A = rng.uniform(0.5, 1.5, (12, 12))
    D_train, y = np.triu(A, 1) + np.triu(A, 1).T, rng.normal(size=12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pred, _ = est.nw_fit_predict(D_train, y, np.full((4, 12), 1e6))
    assert np.array_equal(pred, np.full(4, y.mean()))


def test_nw_median_ignores_zero_distances():
    # Review Focus 3: 5 points, each 3 times, so that 30 distances off the diagonal are 0. The bandwidths come from the
    # median of the positive distances only, and the copies weigh one another at the kernel's maximum, as upstream's.
    rng = np.random.default_rng(12)
    X = rng.uniform(0, 1, (5, 2))[np.repeat(np.arange(5), 3)]
    D_train = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))
    D_test = np.sqrt(((rng.uniform(0, 1, (6, 2))[:, None] - X[None]) ** 2).sum(axis=2))
    y = rng.uniform(1, 2, 15)
    pred, h = est.nw_fit_predict(D_train, y, D_test)
    m_positive, m_all = np.median(D_train[D_train > 0]), np.median(D_train)
    assert h in np.linspace(0.1 * m_positive, 2.0 * m_positive, 50)
    assert h not in np.linspace(0.1 * m_all, 2.0 * m_all, 50)
    expected, h_upstream = oracles.upstream_nw(D_train, y, D_test)
    assert h == h_upstream
    np.testing.assert_allclose(pred, expected, rtol=1e-12, atol=0)


def _refit_loo_mse(Z: np.ndarray, y: np.ndarray) -> float:
    """The mean squared error of each ``y[i]``'s prediction by the OLS on the other rows of ``Z``, refitted by lstsq."""
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = np.linalg.lstsq(Z[keep], y[keep], rcond=None)[0]
        errors.append((y[i] - Z[i] @ beta) ** 2)
    return float(np.mean(errors))


def test_loo_select_press_equals_refitting_without_each_sample():
    rng = np.random.default_rng(13)
    S0, S1 = rng.normal(size=(15, 4)), rng.normal(size=(15, 3))
    y = S0[:, 0] - 0.5 * S1[:, 1] + 0.3 * rng.normal(size=15)
    k0, k1, mse = est.loo_select(S0, S1, y)
    Z = {k: np.column_stack((np.ones(15), S0[:, : k[0]], S1[:, : k[1]])) for k in itertools.product(range(5), range(4))}
    refits = {k: _refit_loo_mse(Zk, y) for k, Zk in Z.items()}
    others = [k for k in ((0, 0), (1, 0), (0, 3), (2, 1), (4, 3), (3, 2)) if k != (k0, k1)][:5]
    for k in ((k0, k1), *others):
        assert est._loo_mse(Z[k], y) == pytest.approx(refits[k], rel=1e-10, abs=0)
    assert mse == pytest.approx(refits[k0, k1], rel=1e-10, abs=0)
    # the choice is the first minimum of the refitted errors, in the order (K0 + K1, K0)
    assert (k0, k1) != (0, 0) and min(sorted(refits, key=lambda k: (sum(k), k[0])), key=refits.get) == (k0, k1)


def test_loo_select_picks_the_mean_for_constant_targets():
    rng = np.random.default_rng(14)
    S0, S1 = rng.normal(size=(20, 4)), rng.normal(size=(20, 3))
    for value in (0.0, 1.0, -2.5, np.pi, 1e3):
        assert est.loo_select(S0, S1, np.full(20, value)) == (0, 0, 0.0)


def test_loo_select_breaks_a_tie_by_the_smaller_k0():
    # One score s in both dimensions: (1, 0) and (0, 1) have the same design, so their LOO errors tie exactly, and
    # (1, 1), whose two score columns are equal, is not eligible (σ_min = 0).
    rng = np.random.default_rng(19)
    s = rng.normal(size=(20, 1))
    y = 1 + 2 * s[:, 0] + 0.1 * rng.normal(size=20)
    assert est.loo_select(s, s, y) == (0, 1, est._loo_mse(np.column_stack((np.ones(20), s)), y))


def test_loo_select_respects_n_minus_two_and_k_max():
    rng = np.random.default_rng(15)
    # n = 6: y is linear in the first four H0 scores, which (4, 0) fits exactly; five scores or more are not allowed
    S0, S1 = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    assert est.loo_select(S0, S1, 1 + S0[:, :4] @ np.array([1.0, -2.0, 0.5, 3.0]))[:2] == (4, 0)
    # n = 30: y needs five H0 scores, more than k_max = 2 allows
    S0, S1 = rng.normal(size=(30, 6)), rng.normal(size=(30, 6))
    y = 1 + S0[:, :5].sum(axis=1) + 0.01 * rng.normal(size=30)
    assert est.loo_select(S0, S1, y)[0] >= 5
    k0, k1, _ = est.loo_select(S0, S1, y, k_max=2)
    assert k0 <= 2 and k1 <= 2


def test_loo_select_uses_only_dimensions_with_components():
    # Review Focus 2: a dimension without components offers only K_d = 0
    rng = np.random.default_rng(16)
    S, empty = rng.normal(size=(20, 4)), np.empty((20, 0))
    y = 1 + S[:, 0] + 0.1 * rng.normal(size=20)
    k0, k1, _ = est.loo_select(S, empty, y)
    assert k1 == 0 and k0 >= 1
    k0, k1, _ = est.loo_select(empty, S, y)
    assert k0 == 0 and k1 >= 1
    # end to end: the 10 training samples hold only essential bars in H1, so their H1 fPCA has rank 0
    diagrams0 = [_random_diagram(rng, n, 1) for n in rng.integers(1, 6, 15)]
    diagrams1 = [_random_diagram(rng, n, 2) for n in [0] * 10 + [3] * 5]
    G0, G1 = betti.gram(betti.curves(diagrams0, 0), 0), betti.gram(betti.curves(diagrams1, 1), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pred, k0, k1 = est.fpcr_fit_predict(G0, G1, rng.normal(size=10), np.arange(10), np.arange(10, 15))
    assert k1 == 0 and np.isfinite(pred).all()


def test_fpcr_recovers_a_linear_functional_of_the_scores():
    rng = np.random.default_rng(17)
    G0 = betti.gram(betti.curves([_random_diagram(rng, n, 1) for n in rng.integers(1, 8, 30)], 0), 0)
    G1 = betti.gram(betti.curves([_random_diagram(rng, n, 2) for n in rng.integers(1, 8, 30)], 1), 1)
    train, test = np.split(rng.permutation(30), [20])
    (S0, T0), (S1, T1) = betti.fpca_scores(G0, train, test), betti.fpca_scores(G1, train, test)
    pred, k0, _ = est.fpcr_fit_predict(G0, G1, 2 + 3 * S0[:, 0] - S0[:, 1], train, test)
    expected = 2 + 3 * T0[:, 0] - T0[:, 1]
    assert k0 >= 2
    np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-8 * np.abs(expected).max())
    # with an H1 score in the functional too, the H1 columns must be the H1 scores
    pred, k0, k1 = est.fpcr_fit_predict(G0, G1, 2 + 3 * S0[:, 0] - S1[:, 0], train, test)
    expected = 2 + 3 * T0[:, 0] - T1[:, 0]
    assert k0 >= 1 and k1 >= 1
    np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-8 * np.abs(expected).max())


def test_arms_give_finite_predictions_with_empty_curves():
    # Review Focus 1: 5 of 40 samples hold only essential bars (one in H0, two in H1, as the fluid diagrams), so their
    # curves are empty: zero Gram rows, a Betti L2 distance of 0 between two of them, and zero curves to project.
    rng = np.random.default_rng(18)
    sizes = rng.integers(1, 8, (40, 2))
    sizes[[0, 7, 19, 33, 38]] = 0  # three in training (0-29), two in test (30-39)
    diagrams = [[_random_diagram(rng, n, 1 + h) for n in sizes[:, h]] for h in (0, 1)]
    curves = [betti.curves(diagrams[h], h) for h in (0, 1)]
    D_K = ws.w2_matrix(diagrams[0]) + ws.w2_matrix(diagrams[1])
    D_B = betti.l2_matrix(curves[0], 0) + betti.l2_matrix(curves[1], 1)
    G0, G1 = betti.gram(curves[0], 0), betti.gram(curves[1], 1)
    assert D_B[0, 7] == D_B[7, 19] == 0 and not G0[38].any() and not G1[38].any()
    y, train, test = sizes.sum(axis=1) + rng.normal(size=40), np.arange(30), np.arange(30, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for D in (D_K, D_B):
            pred, _ = est.nw_fit_predict(D[np.ix_(train, train)], y[train], D[np.ix_(test, train)])
            assert np.isfinite(pred).all()
        pred, _, _ = est.fpcr_fit_predict(G0, G1, y[train], train, test)
        assert np.isfinite(pred).all()


def test_mkey_names_every_matrix():
    assert mat.mkey("interp", "w2", 1, 7) == "interp_07_w2_1"
    assert mat.mkey("fluid", "gram", 0, 4) == "fluid_4_gram_0" and mat.mkey("curv", "l2", 0) == "curv_l2_0"


def _large_checks(bars: np.ndarray, checks: list[tuple[int, int, int, int]]) -> int:
    """How many of G2's ``checks`` have more than 1,000 finite bars on both sides."""
    return sum(min(bars[t, h, p], bars[t, h, q]) > 1000 for t, h, p, q in checks)


def test_g2_sample_caps_size_and_has_ten_large_checks():
    # Review Focus 4: with counts U(0, 3000), 2/9 of the checks exceed 4,000 bars and are rejected.
    bars = np.random.default_rng(8).integers(0, 3000, (5, 2, 400))
    checks = mat.g2_sample(bars, np.random.default_rng(2))
    assert len(checks) >= 100 and _large_checks(bars, checks) >= 10
    assert all(bars[t, h, p] + bars[t, h, q] <= 4000 for t, h, p, q in checks) and len(checks) < 200
    pairs = {(t, p, q) for t, _, p, q in checks}
    assert len(pairs) == 100 and all(0 <= t < 5 and 0 <= p < q < 400 for t, p, q in pairs)
    assert mat.g2_sample(bars, np.random.default_rng(2)) == checks


def test_g2_sample_adds_pairs_until_ten_large_checks():
    # Only 40 of the 400 simulations have more than 1,000 finite bars, so 100 uniform pairs hold about 2 large checks.
    bars = np.random.default_rng(9).integers(0, 1000, (5, 2, 400))
    bars[:, :, :40] = 1500
    checks = mat.g2_sample(bars, np.random.default_rng(2))
    pairs = list(dict.fromkeys((t, p, q) for t, _, p, q in checks))  # in the order drawn
    assert len(pairs) > 100 and _large_checks(bars, checks) >= 10
    assert _large_checks(bars, [(t, h, p, q) for t, h, p, q in checks if (t, p, q) != pairs[-1]]) < 10
    # no check fits under the cap: refused once every pair has been drawn, never an endless loop
    with pytest.raises(ValueError, match="G2 needs"):
        mat.g2_sample(np.full((2, 2, 30), 2500), np.random.default_rng(2))


def test_dataset_matrices_of_a_tiny_task():
    rng = np.random.default_rng(20)
    diagrams = [inp.rips_diagram(rng.normal(size=(12, 2))) for _ in range(6)]
    M = mat.dataset_matrices(diagrams)
    assert sorted(M) == ["gram_0", "gram_1", "l2_0", "l2_1", "w2_0", "w2_1"]
    for key, A in M.items():
        assert A.shape == (6, 6) and np.array_equal(A, A.T)
        if key.startswith("gram"):
            assert np.linalg.eigvalsh(A)[0] >= -1e-9 * np.trace(A)
        else:
            assert not A.diagonal().any()
    for h in (0, 1):  # each matrix is that of its own dimension's diagrams
        bars = [d[h] for d in diagrams]
        curves = betti.curves(bars, h)
        assert np.array_equal(M[f"w2_{h}"], ws.w2_matrix(bars))
        assert np.array_equal(M[f"l2_{h}"], betti.l2_matrix(curves, h))
        assert np.array_equal(M[f"gram_{h}"], betti.gram(curves, h))
    # fluid's call: no W2, and the curves shifted
    shifted = mat.dataset_matrices(diagrams, shift=3.0, with_w2=False)
    assert sorted(shifted) == ["gram_0", "gram_1", "l2_0", "l2_1"]
    assert np.array_equal(shifted["gram_1"], betti.gram(betti.curves([d[1] for d in diagrams], 1, 3.0), 1))


def test_g4_is_relative_to_the_largest_entry():
    # Ruling (7): a tiny entry may move by far more than 1e-12 of itself, as the shifted endpoints round, so long as no
    # difference exceeds 1e-12 of the matrix's largest entry.
    G, bars = np.array([[1e4, 1e-6], [1e-6, 2.0]]), np.array([900, 1])
    rounded = G.copy()
    rounded[0, 1] = rounded[1, 0] = 1e-6 + 1e-15  # 1e-9 of the entry, 1e-19 of the largest
    row = mat._g4_measures(G, rounded, bars)
    assert row["passed"] and row["entrywise"]["worst_relative"] > 1e-12
    clipped = G.copy()
    clipped[1, 1] += 2e-12 * 1e4  # 2e-12 of the largest entry
    assert not mat._g4_measures(G, clipped, bars)["passed"]


@pytest.mark.parametrize("g2_passed, g4_passed", [(True, True), (False, None), (True, False)])
def test_matrices_stage_writes_the_npz_only_if_both_gates_pass(tmp_path, monkeypatch, g2_passed, g4_passed):
    rips, fluid = {"curv_w2_0": np.eye(2), "curv_y": np.arange(2.0)}, {"fluid_0_gram_0": np.ones((3, 3))}

    def rips_matrices() -> dict[str, np.ndarray]:
        if not g2_passed:
            pytest.fail("the matrices were computed after G2 failed")
        return rips

    monkeypatch.setattr(c, "check_topocyte", lambda: None)
    monkeypatch.setattr(c, "check_upstream", lambda: None)
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    monkeypatch.setattr(inp, "fluid_diagrams", lambda t: [])
    monkeypatch.setattr(inp, "fluid_cached_w2", lambda t, h: np.zeros((0, 0)))
    monkeypatch.setattr(mat, "_g2", lambda diagrams, cached: {"passed": g2_passed})
    monkeypatch.setattr(mat, "_rips_matrices", rips_matrices)
    monkeypatch.setattr(mat, "_fluid_matrices", lambda diagrams, cached: (fluid, {"passed": g4_passed}))
    if g2_passed and g4_passed:
        mat.main()
        saved = mat.load_matrices()
        assert saved.keys() == rips.keys() | fluid.keys()
        assert all(np.array_equal(saved[key], A) for key, A in {**rips, **fluid}.items())
    else:
        with pytest.raises(SystemExit) as stop:
            mat.main()
        assert stop.value.code == 1 and not (tmp_path / "matrices.npz").exists()
    record = json.loads((tmp_path / "matrices.json").read_text())
    assert record["passed"] is bool(g2_passed and g4_passed) and record["g2"]["passed"] is g2_passed
    assert record["g4"] == (None if g4_passed is None else {"passed": g4_passed})


def test_load_matrices_refuses_an_npz_that_its_json_does_not_vouch_for(tmp_path, monkeypatch):
    # Carried from Task 6's review: a stale npz, left by an earlier run when a later run failed, must not load.
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    np.savez(tmp_path / "matrices.npz", curv_y=np.arange(3.0))
    sha = hashlib.sha256((tmp_path / "matrices.npz").read_bytes()).hexdigest()

    def record(g2: bool, g4: bool | None, sha256: str) -> None:
        gates = {"g2": {"passed": g2}, "g4": None if g4 is None else {"passed": g4}}
        (tmp_path / "matrices.json").write_text(
            json.dumps({"passed": g2 and bool(g4), **gates, "npz": {"sha256": sha256}})
        )

    record(True, True, sha)
    assert np.array_equal(mat.load_matrices()["curv_y"], np.arange(3.0))
    for g2, g4, sha256 in ((False, None, sha), (True, False, sha), (True, True, "0" * 64)):
        record(g2, g4, sha256)
        with pytest.raises(RuntimeError, match="matrices.json"):
            mat.load_matrices()
    record(True, True, sha)
    np.savez(tmp_path / "matrices.npz", curv_y=np.arange(4.0))  # rewritten after its json
    with pytest.raises(RuntimeError, match="sha256"):
        mat.load_matrices()


def test_fluid_training_sets_are_nested_prefixes_of_the_pool():
    cells = list(rs.fluid_cells())
    assert [cell[:3] for cell in cells] == list(itertools.product(range(5), range(100), range(5)))
    test, pool = inp.fluid_split(3)
    split3 = [(train, cell_test) for _, s, _, train, cell_test in cells if s == 3]  # 5 times × 5 sizes
    assert len(split3) == 25
    for train, cell_test in split3:
        assert np.array_equal(cell_test, test) and not np.intersect1d(train, test).size
    trains = [train for train, _ in split3]
    assert [len(train) for train in trains[:5]] == [25, 50, 100, 200, 300]
    for small, large in itertools.pairwise(trains[:5]):
        assert np.array_equal(large[: len(small)], small)
    assert np.array_equal(trains[4], pool) and np.union1d(pool, test).size == 400  # the largest is the whole pool
    assert all(np.array_equal(a, b) for a, b in zip(trains, trains[5:] + trains[:5], strict=True))  # every time alike


def _tiny_world(monkeypatch) -> dict[str, np.ndarray]:
    """The three tasks made tiny, in place, and their matrices and targets as ``results/matrices.npz`` holds them.

    Interpolation: sizes 6 and 8, 2 runs each, 4 test clouds. Curvature: 8 training and 4 test clouds. Fluid: 14
    simulations at 2 times, 3 splits of 4 test simulations, sizes 5 and 9. G3's cells shrink to match. The loaders
    give 15-point clouds in the plane, and fluid diagrams with births in [-3, -2), one essential bar in H0 and two in
    H1; at each time 3 of them have no finite bar (Review Focus 1). The matrices are ``matrices.dataset_matrices``' of
    the same diagrams, the fluid curves shifted by ``FLUID_SHIFT``.
    """
    rng = np.random.default_rng(22)
    shrunk = {"INTERP_SIZES": (6, 8), "INTERP_RUNS": 2, "INTERP_N_TEST": 4}
    shrunk |= {"FLUID_TIMES": (1.0, 2.5), "FLUID_SIZES": (5, 9), "FLUID_SPLITS": 3, "FLUID_N_TEST": 4}
    for name, value in shrunk.items():
        monkeypatch.setattr(c, name, value)
    monkeypatch.setattr(inp, "_N_FLUID", 14)
    monkeypatch.setattr(rs, "CURV_TRAIN", np.arange(8))
    monkeypatch.setattr(rs, "CURV_TEST", np.arange(8, 12))
    monkeypatch.setattr(rs, "G3_INTERP", (0, 3))
    monkeypatch.setattr(rs, "G3_FLUID", ((0, 0, 5), (1, 2, 9)))
    datasets = [
        {"n_train": n, "run": run, "clouds": list(rng.normal(size=(n + 4, 15, 2))), "y": rng.uniform(0, 1, n + 4)}
        for n in (6, 8)
        for run in range(2)
    ]
    curvature = {"clouds": list(rng.normal(size=(12, 15, 2))), "y": rng.uniform(-2, 2, 12)}
    fluid = {}
    for t in c.FLUID_TIMES:
        sizes = rng.integers(1, 8, (14, 2))
        sizes[[0, 5, 11]] = 0
        fluid[t] = [(_random_diagram(rng, n0, 1) - 3, _random_diagram(rng, n1, 2) - 3) for n0, n1 in sizes]
    monkeypatch.setattr(inp, "interpolation_datasets", lambda: datasets)
    monkeypatch.setattr(inp, "curvature_data", lambda: curvature)
    monkeypatch.setattr(inp, "fluid_diagrams", lambda t: fluid[t])
    M = {}
    for idx, ds in enumerate(datasets):
        M |= mat._keyed("interp", mat.dataset_matrices([inp.rips_diagram(X) for X in ds["clouds"]]), idx)
        M[f"interp_{idx:02d}_y"] = ds["y"]
    M |= mat._keyed("curv", mat.dataset_matrices([inp.rips_diagram(X) for X in curvature["clouds"]]))
    M["curv_y"] = curvature["y"]
    for ti, t in enumerate(c.FLUID_TIMES):
        M |= mat._keyed("fluid", mat.dataset_matrices(fluid[t], shift=c.FLUID_SHIFT), ti)
    M["fluid_y"] = rng.uniform(1, 12, 14)
    return M


def test_run_arm_on_a_tiny_fluid_task(monkeypatch):
    M = _tiny_world(monkeypatch)
    out = {arm: rs.run_arm(arm, "fluid", M) for arm in ("K", "F", "B")}
    for arm, choice in (("K", "h"), ("F", "k"), ("B", "h")):
        shapes = {key: a.shape for key, a in out[arm].items()}
        assert shapes == {
            f"fluid_pred_{arm}": (2, 3, 2, 4),
            f"fluid_r2_{arm}": (2, 3, 2),
            f"fluid_{choice}_{arm}": (2, 3, 2, 2) if arm == "F" else (2, 3, 2),
        }
        assert all(np.isfinite(a).all() for a in out[arm].values())  # 3 of the 14 curves are empty in each dimension
    # the cell (time 1, split 2, size 9), by hand: each arm's estimator on that time's own matrices
    y, (test, pool) = M["fluid_y"], inp.fluid_split(2)
    train = pool[:9]
    for arm, kind in (("K", "w2"), ("B", "l2")):
        D = M[f"fluid_1_{kind}_0"] + M[f"fluid_1_{kind}_1"]
        pred, h = est.nw_fit_predict(D[np.ix_(train, train)], y[train], D[np.ix_(test, train)])
        assert np.array_equal(out[arm][f"fluid_pred_{arm}"][1, 2, 1], pred) and out[arm][f"fluid_h_{arm}"][1, 2, 1] == h
        assert out[arm][f"fluid_r2_{arm}"][1, 2, 1] == c.r2(y[test], pred)
    pred, k0, k1 = est.fpcr_fit_predict(M["fluid_1_gram_0"], M["fluid_1_gram_1"], y[train], train, test)
    assert np.array_equal(out["F"]["fluid_pred_F"][1, 2, 1], pred) and tuple(out["F"]["fluid_k_F"][1, 2, 1]) == (k0, k1)
    assert out["F"]["fluid_r2_F"][1, 2, 1] == c.r2(y[test], pred)


def test_run_arm_on_tiny_interpolation_and_curvature_tasks(monkeypatch):
    M = _tiny_world(monkeypatch)
    out = {arm: rs.run_arm(arm, "interp", M) | rs.run_arm(arm, "curv", M) for arm in ("K", "F", "B")}
    for arm, choice in (("K", "h"), ("F", "k"), ("B", "h")):
        assert {key: a.shape for key, a in out[arm].items()} == {
            f"interp_pred_{arm}": (4, 4),
            f"interp_r2_{arm}": (4,),
            f"interp_{choice}_{arm}": (4, 2) if arm == "F" else (4,),
            f"curv_pred_{arm}": (4,),
            f"curv_mse_{arm}": (),
            f"curv_{choice}_{arm}": (2,) if arm == "F" else (),
        }
        assert all(np.isfinite(a).all() for a in out[arm].values())
    # interpolation dataset 3 trains on its first 8 clouds and tests on its last 4, by hand with arm B
    y = M["interp_03_y"]
    D = M["interp_03_l2_0"] + M["interp_03_l2_1"]
    pred, h = est.nw_fit_predict(D[:8, :8], y[:8], D[8:, :8])
    assert np.array_equal(out["B"]["interp_pred_B"][3], pred) and out["B"]["interp_h_B"][3] == h
    assert out["B"]["interp_r2_B"][3] == c.r2(y[8:], pred)
    # curvature trains on CURV_TRAIN and tests on CURV_TEST, by hand with arm F; its metric is the MSE
    y = M["curv_y"]
    pred, k0, k1 = est.fpcr_fit_predict(M["curv_gram_0"], M["curv_gram_1"], y[:8], np.arange(8), np.arange(8, 12))
    assert np.array_equal(out["F"]["curv_pred_F"], pred) and tuple(out["F"]["curv_k_F"]) == (k0, k1)
    assert out["F"]["curv_mse_F"] == np.mean((y[8:] - pred) ** 2)
    with pytest.raises(ValueError, match="arm"):
        rs.run_arm("X", "curv", M)


def _write_upstream_results(root: pathlib.Path, k: dict, moved: str, by: float) -> None:
    """Upstream's three result CSVs under ``root``, with upstream's columns, 5e-10 from arm K's results ``k``.

    Each reported value is K's, up or down by 5e-10 in turn; the one named by ``moved`` is K's plus ``by``:
    ``interp_mean`` (size 30), ``interp_std`` (size 80), ``curv``, or ``fluid`` (t 7.5, split 4, n 200).
    """
    r2 = k["interp_r2_K"].reshape(10, 10)
    turn = 5e-10 * np.where(np.arange(10) % 2, -1.0, 1.0)
    mean, std = r2.mean(axis=1) + turn, r2.std(axis=1, ddof=1) - turn
    if moved == "interp_mean":
        mean[2] = r2[2].mean() + by
    if moved == "interp_std":
        std[7] = r2[7].std(ddof=1) + by
    interp = {"n_train": c.INTERP_SIZES, "PD_kernel_R2_mean": mean, "PD_kernel_R2_std": std}
    interp |= {"PI_kernel_R2_mean": 0.7, "PI_kernel_R2_std": 0.1}
    mse = float(k["curv_mse_K"]) + (by if moved == "curv" else -5e-10)
    curv = {"method": ["ml_on_ph0", "PD_kernel", "nn_deep"], "MSE": [0.35, mse, 2.8]}
    fluid = [
        {"t": t, "split": s, "n_train": n, "n_test": 100, "R2": k["fluid_r2_K"][ti, s, si] + 5e-10 * (-1) ** (s + si)}
        for ti, t in enumerate(c.FLUID_TIMES)
        for s in range(5)
        for si, n in enumerate(c.FLUID_SIZES)
    ]
    if moved == "fluid":
        fluid[(3 * 5 + 4) * 5 + 3]["R2"] = k["fluid_r2_K"][3, 4, 3] + by
    for folder, name, table in (
        ("interpolation", "results_interpolation.csv", interp),
        ("curvature", "results_curvature.csv", curv),
        ("fluid", "results_fluid_splits.csv", fluid),
    ):
        (root / folder).mkdir(exist_ok=True)
        pd.DataFrame(table).to_csv(root / folder / name, index=False)


@pytest.mark.parametrize(
    "moved, by, cell",
    [
        ("interp_mean", 2e-9, {"n_train": 30, "statistic": "mean"}),
        ("interp_std", -2e-9, {"n_train": 80, "statistic": "std"}),
        ("curv", -2e-9, {"method": "PD_kernel"}),
        ("fluid", 2e-9, {"t": 7.5, "split": 4, "n_train": 200}),
    ],
)
def test_g1_passes_within_tolerance_and_fails_beyond(tmp_path, monkeypatch, moved, by, cell):
    # Moves of both signs: a signed difference would pass a reported value 2e-9 above the rerun.
    rng = np.random.default_rng(23)
    k = {"interp_r2_K": rng.uniform(0.6, 1, 100), "curv_mse_K": np.float64(rng.uniform(0.1, 0.5))}
    k["fluid_r2_K"] = rng.uniform(0.3, 0.9, (5, 100, 5))  # splits 5-99 have no reported value
    monkeypatch.setattr(c, "UPSTREAM", tmp_path)
    _write_upstream_results(tmp_path, k, moved, by / 4)  # 5e-10 away
    record = rs.g1(k)
    assert record["passed"] and {task: record[task]["compared"] for task in rs.TASKS} == {
        "interp": 20,
        "curv": 1,
        "fluid": 125,
    }
    _write_upstream_results(tmp_path, k, moved, by)  # 2e-9 away
    record = rs.g1(k)
    task = moved.split("_")[0]
    assert not record["passed"] and [t for t in rs.TASKS if not record[t]["passed"]] == [task]
    (failing,) = record[task]["failing"]
    assert failing.items() >= cell.items() and failing["abs_diff"] == pytest.approx(2e-9, rel=1e-6)
    assert record[task]["worst_abs_diff"] == failing["abs_diff"]


def test_g3_gates_only_the_columns_arm_f_can_use():
    # The ruling after Task 4: FPCA.transform misses in near-null components, which FPCR never uses (K_d <= K_MAX),
    # so G3 compares the first min(K_MAX, rank) columns, at G3_TOL of their largest score; all columns are information.
    rng = np.random.default_rng(24)
    T = rng.normal(size=(10, 30)) * np.geomspace(100, 1e-6, 30)
    largest = np.abs(T[:, :20]).max()
    S = T.copy()
    S[:, 25] += 1e-6 * largest  # beyond K_MAX
    row = rs._g3_measures(S, T)
    assert row["passed"] and (row["rank"], row["compared_columns"]) == (30, 20) and row["worst_relative"] < 1e-15
    assert row["worst_relative_all_columns"] == pytest.approx(1e-6, rel=1e-9)
    S[:, 19] += 2e-9 * largest  # the last column FPCR can use
    assert not rs._g3_measures(S, T)["passed"]
    small = T[:, :5]  # rank below K_MAX: every column is compared
    row = rs._g3_measures(small + 5e-10 * largest, small)
    assert row["passed"] and row["compared_columns"] == 5
    assert not rs._g3_measures(small + 2e-9 * np.abs(small).max(), small)["passed"]
    assert not rs._g3_measures(T[:, :12], T[:, :13])["passed"]  # ranks that differ fail


def test_g3_fails_closed_on_a_score_that_is_not_finite():
    # A NaN must not pass as a difference of 0, in a compared column or beyond K_MAX, in either score matrix.
    T = np.random.default_rng(25).normal(size=(6, 25))
    for where in ((0, 3), (5, 22)):
        for bad in (np.nan, np.inf):
            broken = T.copy()
            broken[where] = bad
            for row in (rs._g3_measures(T, broken), rs._g3_measures(broken, T)):
                assert not row["passed"] and row["worst_relative"] == row["worst_relative_all_columns"] == math.inf
    assert rs._relative_worst(T, np.full_like(T, np.nan)) == math.inf


def test_g1_fails_on_a_nan_and_names_it_the_worst(tmp_path, monkeypatch):
    rng = np.random.default_rng(23)
    k = {"interp_r2_K": rng.uniform(0.6, 1, 100), "curv_mse_K": np.float64(rng.uniform(0.1, 0.5))}
    k["fluid_r2_K"] = rng.uniform(0.3, 0.9, (5, 100, 5))
    monkeypatch.setattr(c, "UPSTREAM", tmp_path)
    _write_upstream_results(tmp_path, k, "none", 0.0)  # every value 5e-10 away
    k["fluid_r2_K"][0, 1, 2] = np.nan  # the 8th fluid value compared: max() over NaN comparisons would skip it
    record = rs.g1(k)
    assert not record["passed"] and [task for task in rs.TASKS if not record[task]["passed"]] == ["fluid"]
    (failing,) = record["fluid"]["failing"]
    assert (failing["t"], failing["split"], failing["n_train"]) == (1.0, 1, 100)
    assert record["fluid"]["worst"] is failing and math.isnan(record["fluid"]["worst_abs_diff"])


def test_g3_on_tiny_tasks_passes_and_sees_a_changed_gram(monkeypatch):
    # The curves rebuilt from the loaders must be the matrices' own: the fluid ones lie below 0 unless shifted.
    M = _tiny_world(monkeypatch)
    record = rs.g3(M)
    cells = [(row["task"], row.get("dataset", row.get("time")), row["h"]) for row in record["cells"]]
    assert cells == [(task, at, h) for task, at in (("interp", 0), ("interp", 3), ("curv", None)) for h in (0, 1)] + [
        ("fluid", t, h) for t in (1.0, 2.5) for h in (0, 1)
    ]
    assert record["passed"] and all(row["rank"] > 0 for row in record["cells"])
    assert set(record["worst_relative"]) == set(record["worst_relative_all_columns"]) == set(rs.TASKS)
    M["fluid_1_gram_1"] = M["fluid_1_gram_1"] * (1 + 1e-6)  # its scores move by about 5e-7 of themselves
    record = rs.g3(M)
    failing = [(row["task"], row.get("time"), row["h"]) for row in record["cells"] if not row["passed"]]
    assert failing == [("fluid", 2.5, 1)]
    assert not record["passed"]


class _Reads(dict):
    """A dict that records every key read through ``[]``: the matrices an arm reads."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.read: set[str] = set()

    def __getitem__(self, key: str):
        self.read.add(key)
        return super().__getitem__(key)

    def kinds(self) -> set[str]:
        """The kinds of matrix read: ``w2`` (arm K), ``gram`` (F, G3) or ``l2`` (B)."""
        return {kind for key in self.read for kind in mat.KINDS if f"_{kind}_" in key}


def _arms_stage(monkeypatch, tmp_path, M: dict, g1_passed: bool) -> tuple[object, list, list]:
    """``run_study.py arms`` on ``M``, G1 monkeypatched: the exit code, ``run_arm``'s calls and what G1 was given.

    ``run_arm`` is the real one, spied on; G3 is the real one unless the test monkeypatches it first.
    """
    calls, given, run_arm = [], [], rs.run_arm

    def spy(arm: str, task: str, M: dict) -> dict:
        calls.append((arm, task))
        return run_arm(arm, task, M)

    def g1(k: dict) -> dict:
        given.append(sorted(k))
        return {"passed": g1_passed}

    monkeypatch.setattr(c, "check_topocyte", lambda: None)
    monkeypatch.setattr(c, "check_upstream", lambda: None)
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    monkeypatch.setattr(mat, "load_matrices", lambda: M)
    monkeypatch.setattr(rs, "run_arm", spy)
    monkeypatch.setattr(rs, "g1", g1)
    try:
        rs.main(["arms"])
    except SystemExit as stop:
        return stop.code, calls, given
    return None, calls, given


K_RESULTS = sorted(f"{task}_{name}_K" for task in rs.TASKS for name in ("pred", "mse" if task == "curv" else "r2", "h"))


def test_arms_stage_stops_before_f_and_b_when_g1_fails(tmp_path, monkeypatch):
    M = _Reads(_tiny_world(monkeypatch))
    monkeypatch.setattr(rs, "g3", lambda M: pytest.fail("G3 ran after G1 failed"))
    monkeypatch.setattr(est, "fpcr_fit_predict", lambda *args: pytest.fail("arm F ran after G1 failed"))
    np.savez(tmp_path / "arms.npz", stale=np.zeros(1))  # an earlier run's, which a failed rerun must not leave
    code, calls, given = _arms_stage(monkeypatch, tmp_path, M, g1_passed=False)
    assert code == 1 and calls == [("K", "interp"), ("K", "curv"), ("K", "fluid")] and given == [K_RESULTS]
    # F reads the Gram matrices and B the Betti L2 distances: neither was read, so neither arm ran by any route
    assert M.kinds() == {"w2"}
    record = json.loads((tmp_path / "arms.json").read_text())
    assert record["passed"] is False and record["g1"] == {"passed": False} and record["g3"] is None
    assert set(record["seconds"]) == {"load", "K", "g1", "total"} and not (tmp_path / "arms.npz").exists()


def test_arms_stage_stops_before_f_and_b_when_g3_fails(tmp_path, monkeypatch):
    M = _Reads(_tiny_world(monkeypatch))
    monkeypatch.setattr(rs, "g3", lambda M: {"passed": False})
    monkeypatch.setattr(est, "fpcr_fit_predict", lambda *args: pytest.fail("arm F ran after G3 failed"))
    np.savez(tmp_path / "arms.npz", stale=np.zeros(1))  # an earlier run's, which a failed rerun must not leave
    code, calls, _ = _arms_stage(monkeypatch, tmp_path, M, g1_passed=True)
    assert code == 1 and calls == [("K", "interp"), ("K", "curv"), ("K", "fluid")] and M.kinds() == {"w2"}
    record = json.loads((tmp_path / "arms.json").read_text())
    assert record["passed"] is False and record["g1"] == {"passed": True} and record["g3"] == {"passed": False}
    assert not (tmp_path / "arms.npz").exists()


def test_arms_stage_writes_every_arm_after_both_gates_pass(tmp_path, monkeypatch):
    M = _tiny_world(monkeypatch)
    code, calls, _ = _arms_stage(monkeypatch, tmp_path, M, g1_passed=True)  # G3 runs, on the tiny tasks
    assert code is None and calls == [(arm, task) for arm in ("K", "F", "B") for task in rs.TASKS]
    with np.load(tmp_path / "arms.npz", allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    expected = {"interp_y_test": (4, 4), "curv_y_test": (4,), "fluid_y": (14,), "fluid_test": (3, 4)}
    for arm in ("K", "F", "B"):
        expected |= {f"interp_pred_{arm}": (4, 4), f"interp_r2_{arm}": (4,), f"curv_pred_{arm}": (4,)}
        expected |= {f"curv_mse_{arm}": (), f"fluid_pred_{arm}": (2, 3, 2, 4), f"fluid_r2_{arm}": (2, 3, 2)}
    expected |= {"interp_k_F": (4, 2), "curv_k_F": (2,), "fluid_k_F": (2, 3, 2, 2)}
    expected |= {f"{task}_h_{arm}": shape for arm in "KB" for task, shape in (("interp", (4,)), ("curv", ()))}
    expected |= {"fluid_h_K": (2, 3, 2), "fluid_h_B": (2, 3, 2)}
    assert {key: a.shape for key, a in arrays.items()} == expected
    assert np.array_equal(arrays["fluid_test"], [inp.fluid_split(s)[0] for s in range(3)])
    assert np.array_equal(arrays["interp_y_test"][3], M["interp_03_y"][8:])
    assert np.array_equal(arrays["curv_y_test"], M["curv_y"][8:]) and np.array_equal(arrays["fluid_y"], M["fluid_y"])
    record = json.loads((tmp_path / "arms.json").read_text())
    assert record["passed"] and record["g1"] == {"passed": True} and record["g3"]["passed"]
    assert record["npz"]["sha256"] == hashlib.sha256((tmp_path / "arms.npz").read_bytes()).hexdigest()
    assert {"K", "F", "B"} <= set(record["seconds"]) and set(record["provenance"]["md5"]) == set(rs._SCRIPTS)


# The readouts: design.md's "Decision rule, fixed in advance", on the arrays of arms.npz.


@pytest.mark.parametrize(
    "lo,hi,expected",
    [(0.01, 0.2, "beats"), (-0.2, -0.01, "below"), (-0.1, 0.1, "level"), (0.0, 0.1, "level"), (-0.1, 0.0, "level")],
)
def test_verdict_follows_the_rule_fixed_in_advance(lo, hi, expected):
    assert ro.verdict(lo, hi) == expected


@pytest.mark.parametrize("lo, hi", [(np.nan, 0.1), (0.01, np.nan), (np.nan, np.nan), (0.01, np.inf), (-np.inf, -0.01)])
def test_verdict_refuses_a_bound_that_is_not_finite(lo, hi):
    # By the rule's comparisons alone, a NaN interval would read "level"; it is no readout at all.
    with pytest.raises(ValueError, match="finite"):
        ro.verdict(lo, hi)


def test_intervals_of_identical_arms_are_zero():
    assert ro.interval_interp(np.zeros((10, 10))) == (0.0, 0.0, 0.0)


def test_interval_interp_resamples_runs_within_sizes():
    delta = np.repeat(np.linspace(-1, 1, 10)[:, None], 10, axis=1)  # no variation within a size
    mean, lo, hi = ro.interval_interp(delta)
    assert lo == pytest.approx(mean) and hi == pytest.approx(mean)


def test_interval_interp_gives_the_mean_over_the_datasets_and_is_reproducible():
    delta = np.random.default_rng(39).normal(size=(10, 10))
    mean, lo, hi = ro.interval_interp(delta)
    assert mean == delta.mean() and lo < mean < hi and ro.interval_interp(delta) == (mean, lo, hi)


def test_interval_units_is_reproducible():
    # Every interval draws from a fresh default_rng(BOOT_SEED): another interval in between changes nothing.
    values = np.random.default_rng(40).normal(size=50)
    first = ro.interval_units(values)
    ro.interval_interp(np.random.default_rng(41).normal(size=(10, 10)))
    assert ro.interval_units(values) == first
    mean, lo, hi = first
    assert mean == values.mean() and lo < mean < hi
    assert ro.interval_units(np.full(8, 0.25)) == (0.25, 0.25, 0.25)  # units alike: every resample is alike


@pytest.mark.parametrize("values", [np.zeros((100, 5)), np.zeros((5, 100, 5)), np.float64(0.5)])
def test_interval_units_takes_one_value_per_unit(values):
    # More dimensions would resample the first axis and pool the percentiles of the other axes' means, silently.
    with pytest.raises(ValueError, match="1-D"):
        ro.interval_units(values)


def _synthetic_arms(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The arrays of ``arms.npz`` that the readouts read, at their real shapes, with random values."""
    target = rng.uniform(-2, 2, 50)
    arms = {"curv_y_test": target, "interp_k_F": rng.integers(0, 4, (100, 2)), "curv_k_F": np.array([3, 1])}
    arms["fluid_k_F"] = rng.integers(0, 4, (5, 100, 5, 2))
    for arm in rs.ARMS:
        arms[f"interp_r2_{arm}"] = rng.uniform(0.5, 1.0, 100)
        arms[f"curv_pred_{arm}"] = target + rng.normal(0, 0.5, 50)
        arms[f"curv_mse_{arm}"] = np.mean((target - arms[f"curv_pred_{arm}"]) ** 2)
        arms[f"fluid_r2_{arm}"] = rng.uniform(0.1, 0.9, (5, 100, 5))
    return arms


_TURKES = (
    "ml_on_ph0",
    "ml_on_ph1",
    "ml_on_ph0_longest",
    "ml_on_ph1_longest",
    "ph0_ml",
    "ph1_ml",
    "ml_on_dis_mat",
    "nn_shallow",
    "nn_deep",
    "point_net",
)


@pytest.fixture
def reported_tables(tmp_path, monkeypatch) -> dict[str, pd.DataFrame]:
    """Upstream's three reported tables, with upstream's columns and random values, in place of upstream's.

    ``c.UPSTREAM`` is ``tmp_path``; the tables are returned by their paths under it. PD_kernel's MSE is upstream's,
    0.22575596585327898, which pandas' default float parser reads as 0.2257559658532789.
    """
    rng = np.random.default_rng(42)
    interp = {"n_train": c.INTERP_SIZES}
    interp |= {
        f"{method}_R2_{stat}": rng.uniform(0, 1, 10)
        for method in ("PD_kernel", "PI_kernel")
        for stat in ("mean", "std")
    }
    fluid = [
        {"t": t, "n_train": n, "R2_mean": rng.uniform(0, 1), "R2_std": rng.uniform(0, 0.1)}
        for t in c.FLUID_TIMES
        for n in c.FLUID_SIZES
    ]
    tables = {
        "interpolation/results_interpolation.csv": pd.DataFrame(interp),
        "curvature/results_curvature.csv": pd.DataFrame(
            {"method": [*_TURKES, "PD_kernel"], "MSE": [*rng.uniform(0.2, 3, 10), 0.22575596585327898]}
        ),
        "fluid/results_fluid.csv": pd.DataFrame(fluid),
    }
    for name, table in tables.items():
        (tmp_path / name).parent.mkdir()
        table.to_csv(tmp_path / name, index=False)
    monkeypatch.setattr(c, "UPSTREAM", tmp_path)
    return tables


def _point(readout: dict, value: float) -> bool:
    """Whether a readout's mean and both bounds are ``value``, to 1e-12: an interval whose resamples are all alike."""
    return [readout["mean"], readout["lo"], readout["hi"]] == pytest.approx([value] * 3, abs=1e-12)


def test_curvature_delta_is_signed_so_positive_favours_f(reported_tables):
    arms = _synthetic_arms(np.random.default_rng(43))
    target, err = arms["curv_y_test"], np.random.default_rng(44).uniform(0.1, 0.5, 50) * np.resize([1.0, -1.0], 50)
    arms["curv_pred_F"], arms["curv_pred_K"] = target + err, target - 2 * err  # se_K = 4 se_F > se_F everywhere
    curv = ro.readouts(arms)["primary"]["curv"]
    assert curv["verdict"] == "beats" and curv["lo"] > 0 and curv["mean"] == pytest.approx(np.mean(3 * err**2))
    assert [curv["mean"], curv["lo"], curv["hi"]] == pytest.approx(ro.interval_units(3 * err**2))  # units: the clouds
    arms["curv_pred_F"], arms["curv_pred_K"] = arms["curv_pred_K"], arms["curv_pred_F"]
    assert ro.readouts(arms)["primary"]["curv"]["verdict"] == "below"


def test_every_pair_is_signed_so_positive_favours_its_first_arm(reported_tables):
    # F above B above K on every R², and the reverse on every squared error: every pair's Δ lies above 0.
    arms = _synthetic_arms(np.random.default_rng(45))
    for task in ("interp", "fluid"):
        arms[f"{task}_r2_B"], arms[f"{task}_r2_F"] = arms[f"{task}_r2_K"] + 0.05, arms[f"{task}_r2_K"] + 0.125
    err = np.random.default_rng(46).uniform(0.1, 0.5, 50)
    for arm, scale in (("F", 1.0), ("B", 2.0), ("K", 3.0)):
        arms[f"curv_pred_{arm}"] = arms["curv_y_test"] + scale * err
    out = ro.readouts(arms)
    primary, exploratory, se = out["primary"], out["exploratory"], np.mean(err**2)
    assert primary["pair"] == ["F", "K"] and [primary[task]["verdict"] for task in rs.TASKS] == ["beats"] * 3
    assert [primary[task]["mean"] for task in rs.TASKS] == pytest.approx([0.125, 9 * se - se, 0.125])
    assert {name: pair["pair"] for name, pair in exploratory.items()} == {"B_vs_K": ["B", "K"], "F_vs_B": ["F", "B"]}
    assert [exploratory["B_vs_K"][task]["mean"] for task in rs.TASKS] == pytest.approx([0.05, 9 * se - 4 * se, 0.05])
    assert [exploratory["F_vs_B"][task]["mean"] for task in rs.TASKS] == pytest.approx([0.075, 4 * se - se, 0.075])
    for pair in exploratory.values():  # exploratory: the same readouts, no verdict
        assert all(pair[task]["lo"] > 0 and "verdict" not in pair[task] for task in rs.TASKS)
    descriptive = primary["interp"]["by_size"] + primary["fluid"]["by_time"] + primary["fluid"]["by_size"]
    assert all("verdict" not in readout for readout in [*descriptive, primary["fluid"]["paper_splits"]])


def test_readouts_resample_the_units_of_the_rule(reported_tables):
    rng = np.random.default_rng(47)
    arms = _synthetic_arms(rng)
    # interpolation: Δ alike within each size (the datasets come in size order, then run order), so the intervals
    # that resample runs within a size are points; a resample across sizes would not be
    sizes = np.linspace(-0.2, 0.3, 10)
    arms["interp_r2_F"] = arms["interp_r2_K"] + np.repeat(sizes, 10)
    # fluid: Δ alike at every split, so the intervals over splits are points; one over cells would not be
    e = rng.normal(0, 0.1, (5, 1, 5))
    arms["fluid_r2_F"] = arms["fluid_r2_K"] + e
    primary = ro.readouts(arms)["primary"]
    assert _point(primary["interp"], sizes.mean())
    assert [readout["n_train"] for readout in primary["interp"]["by_size"]] == list(c.INTERP_SIZES)
    assert all(_point(readout, v) for readout, v in zip(primary["interp"]["by_size"], sizes, strict=True))
    assert _point(primary["fluid"], e.mean()) and _point(primary["fluid"]["paper_splits"], e.mean())
    assert [readout["t"] for readout in primary["fluid"]["by_time"]] == list(c.FLUID_TIMES)
    assert all(_point(readout, e[ti].mean()) for ti, readout in enumerate(primary["fluid"]["by_time"]))
    # the paper's splits alone: splits 0-4 one up
    arms["fluid_r2_F"] = arms["fluid_r2_K"] + e + np.where(np.arange(100) < 5, 1.0, 0.0)[:, None]
    fluid = ro.readouts(arms)["primary"]["fluid"]
    assert fluid["paper_splits"]["splits"] == [0, 1, 2, 3, 4] and _point(fluid["paper_splits"], e.mean() + 1)
    assert fluid["mean"] == pytest.approx(e.mean() + 0.05)
    # Δ that varies across splits only: each split is one unit, at every time and size together
    b = rng.normal(0, 0.1, 100)
    arms["fluid_r2_F"] = arms["fluid_r2_K"] + b[:, None]
    fluid = ro.readouts(arms)["primary"]["fluid"]
    assert [fluid["mean"], fluid["lo"], fluid["hi"]] == pytest.approx(ro.interval_units(b), abs=1e-12)


def test_fluid_by_size_resamples_the_splits_by_their_mean_over_the_times(reported_tables):
    # Δ = s[size] + t[time], with t summing to 0: at each size, every split's mean over the times is s[size], so a
    # resample of the splits is the point s[size]; one of the times or of the cells would not be a point, and one that
    # mixed sizes would not sit at s[size]. Dyadic values keep every sum exact.
    arms = _synthetic_arms(np.random.default_rng(53))
    size_effect, time_effect = np.array([0.5, 0.25, -0.125, 1.0, 0.0]), np.array([0.5, -0.5, 0.25, -0.25, 0.0])
    arms["fluid_r2_K"] = np.zeros((5, 100, 5))
    arms["fluid_r2_F"] = np.broadcast_to(time_effect[:, None, None] + size_effect, (5, 100, 5)).copy()
    assert ro.readouts(arms)["primary"]["fluid"]["by_size"] == [
        {"n_train": 25, "mean": 0.5, "lo": 0.5, "hi": 0.5},
        {"n_train": 50, "mean": 0.25, "lo": 0.25, "hi": 0.25},
        {"n_train": 100, "mean": -0.125, "lo": -0.125, "hi": -0.125},
        {"n_train": 200, "mean": 1.0, "lo": 1.0, "hi": 1.0},
        {"n_train": 300, "mean": 0.0, "lo": 0.0, "hi": 0.0},
    ]
    # random Δ: every pair has the readout, each size's mean Δ inside a proper interval, and no verdict
    arms = _synthetic_arms(np.random.default_rng(54))
    out = ro.readouts(arms)
    for x, y in (("F", "K"), ("B", "K"), ("F", "B")):
        pair = out["primary"] if (x, y) == ("F", "K") else out["exploratory"][f"{x}_vs_{y}"]
        delta = arms[f"fluid_r2_{x}"] - arms[f"fluid_r2_{y}"]
        assert [readout["n_train"] for readout in pair["fluid"]["by_size"]] == list(c.FLUID_SIZES)
        for si, readout in enumerate(pair["fluid"]["by_size"]):
            assert readout["mean"] == pytest.approx(delta[:, :, si].mean()) and "verdict" not in readout
            assert readout["lo"] < readout["mean"] < readout["hi"]


def test_readouts_read_the_reported_values_exactly(tmp_path, reported_tables):
    # pandas' default float parser misreads 52 of upstream's 101 reported values in their last bits (30 with 16
    # significant digits, 22 with 17): every reported value must be the float of its CSV text, as Python's float()
    # reads it.
    tables = ro.readouts(_synthetic_arms(np.random.default_rng(55)))["tables"]
    rows = {}
    for name in reported_tables:
        with open(tmp_path / name, newline="") as file:
            rows[name] = list(csv.DictReader(file))
    methods, stats = ("PD_kernel", "PI_kernel"), ("mean", "std")
    assert [row["reported"] for row in tables["interp"]] == [
        {method: {stat: float(text[f"{method}_R2_{stat}"]) for stat in stats} for method in methods}
        for text in rows["interpolation/results_interpolation.csv"]
    ]
    assert tables["curv"]["reported"] == {
        text["method"]: float(text["MSE"]) for text in rows["curvature/results_curvature.csv"]
    }
    assert [row["reported"] for row in tables["fluid"]] == [
        {"mean": float(text["R2_mean"]), "std": float(text["R2_std"])} for text in rows["fluid/results_fluid.csv"]
    ]


def test_readouts_set_the_arms_beside_the_reported_tables(reported_tables):
    arms = _synthetic_arms(np.random.default_rng(48))
    tables = ro.readouts(arms)["tables"]
    interp, curv, fluid = (
        reported_tables[f"{name}/results_{name}.csv"] for name in ("interpolation", "curvature", "fluid")
    )
    # interpolation, size 30 (datasets 20-29), beside the reported PD_kernel and PI_kernel of that size
    assert [row["n_train"] for row in tables["interp"]] == list(c.INTERP_SIZES)
    row = tables["interp"][2]
    for arm in rs.ARMS:
        runs = arms[f"interp_r2_{arm}"][20:30]
        assert row[arm] == pytest.approx({"mean": runs.mean(), "std": runs.std(ddof=1)})
    for method in ("PD_kernel", "PI_kernel"):
        reported = {"mean": interp[f"{method}_R2_mean"][2], "std": interp[f"{method}_R2_std"][2]}
        assert row["reported"][method] == pytest.approx(reported)
    # curvature: every reported method, in the CSV's order, beside each arm's MSE
    assert list(tables["curv"]["reported"]) == [*_TURKES, "PD_kernel"]
    assert tables["curv"]["reported"] == pytest.approx(dict(zip(curv["method"], curv["MSE"], strict=True)))
    assert {arm: tables["curv"][arm] for arm in rs.ARMS} == {arm: float(arms[f"curv_mse_{arm}"]) for arm in rs.ARMS}
    # fluid, t 7.5 and n 200 (time 3, size 3): the paper's splits 0-4 and all 100, beside the reported row
    assert [(row["t"], row["n_train"]) for row in tables["fluid"]] == list(
        itertools.product(c.FLUID_TIMES, c.FLUID_SIZES)
    )
    row = tables["fluid"][18]
    assert row["reported"] == pytest.approx({"mean": fluid["R2_mean"][18], "std": fluid["R2_std"][18]})
    for arm in rs.ARMS:
        r2 = arms[f"fluid_r2_{arm}"][3, :, 3]
        assert row["paper_splits"][arm] == pytest.approx({"mean": r2[:5].mean(), "std": r2[:5].std(ddof=1)})
        assert row["all_splits"][arm] == pytest.approx({"mean": r2.mean(), "std": r2.std(ddof=1)})


@pytest.mark.parametrize(
    "name, broken",
    [
        ("interpolation/results_interpolation.csv", lambda table: table[table["n_train"] != 50]),
        ("curvature/results_curvature.csv", lambda table: pd.concat([table, table.tail(1)])),  # PD_kernel twice
        ("curvature/results_curvature.csv", lambda table: table[table["method"] != "PD_kernel"]),
        ("fluid/results_fluid.csv", lambda table: table.head(24)),  # without t 9.0, n 300
    ],
)
def test_readouts_refuse_a_reported_table_without_its_cells(tmp_path, reported_tables, name, broken):
    broken(reported_tables[name]).to_csv(tmp_path / name, index=False)
    with pytest.raises(ValueError, match=name.split("/")[1]):
        ro.readouts(_synthetic_arms(np.random.default_rng(49)))


def test_readouts_count_the_chosen_k_per_task_and_fluid_time(reported_tables):
    arms = _synthetic_arms(np.random.default_rng(50))
    arms["interp_k_F"] = np.array([[0, 1]] * 60 + [[2, 3]] * 40)
    arms["curv_k_F"] = np.array([4, 0])
    k = np.empty((5, 100, 5, 2), dtype=int)
    k[0], k[1:, :50], k[1:, 50:] = [1, 0], [0, 2], [3, 1]
    arms["fluid_k_F"] = k
    chosen = ro.readouts(arms)["chosen_k"]
    assert chosen["interp"] == [{"k0": 0, "k1": 1, "count": 60}, {"k0": 2, "k1": 3, "count": 40}]
    assert chosen["curv"] == [{"k0": 4, "k1": 0, "count": 1}]
    assert chosen["fluid"] == [
        {"k0": 0, "k1": 2, "count": 1000},
        {"k0": 1, "k1": 0, "count": 500},
        {"k0": 3, "k1": 1, "count": 1000},
    ]
    later = [{"k0": 0, "k1": 2, "count": 250}, {"k0": 3, "k1": 1, "count": 250}]
    assert chosen["fluid_by_time"] == [{"t": 1.0, "counts": [{"k0": 1, "k1": 0, "count": 500}]}] + [
        {"t": t, "counts": later} for t in (2.5, 5.0, 7.5, 9.0)
    ]


def test_load_arms_refuses_an_npz_that_arms_json_does_not_vouch_for(tmp_path, monkeypatch):
    # Carried from Task 7's review: an npz that a failed or interrupted arms run left must not be read.
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    np.savez(tmp_path / "arms.npz", curv_y_test=np.arange(3.0))
    sha = hashlib.sha256((tmp_path / "arms.npz").read_bytes()).hexdigest()

    def record(g1: bool, g3: bool | None, sha256: str | None) -> None:
        gates = {"g1": {"passed": g1}, "g3": None if g3 is None else {"passed": g3}}
        npz = None if sha256 is None else {"sha256": sha256}
        (tmp_path / "arms.json").write_text(json.dumps({"passed": g1 and bool(g3), **gates, "npz": npz}))

    record(True, True, sha)
    arrays, loaded = rs.load_arms()
    assert np.array_equal(arrays["curv_y_test"], np.arange(3.0)) and loaded == sha  # the sha256 of the loaded bytes
    for g1, g3, sha256 in ((False, None, None), (False, None, sha), (True, False, sha), (True, True, "0" * 64)):
        record(g1, g3, sha256)
        with pytest.raises(RuntimeError, match="arms.json"):
            rs.load_arms()
    record(True, True, sha)
    np.savez(tmp_path / "arms.npz", curv_y_test=np.arange(4.0))  # rewritten after its json
    with pytest.raises(RuntimeError, match="sha256"):
        rs.load_arms()


def _readouts_stage_on(tmp_path, monkeypatch, arms: dict, g3_passed: bool = True) -> tuple[list[str], str]:
    """``run_study.py readouts`` made ready on ``arms``: the calls it makes to the checks and ``load_arms``, as a list
    that fills as it runs, and the npz's sha256.

    ``results/`` is ``tmp_path``, with ``arms.npz`` holding ``arms`` and an ``arms.json`` that records G1 passed, G3
    as ``g3_passed``, and the npz's sha256. Once ``load_arms`` has loaded, the spy overwrites ``arms.json`` as a later
    run would, so that only the sha256 of the bytes loaded can reach ``readouts.json``.
    """
    calls, load_arms = [], rs.load_arms

    def spy() -> tuple[dict[str, np.ndarray], str]:
        calls.append("load_arms")
        loaded = load_arms()
        (tmp_path / "arms.json").write_text(json.dumps({"npz": {"sha256": "f" * 64}, "date": "later"}))
        return loaded

    monkeypatch.setattr(c, "check_topocyte", lambda: calls.append("check_topocyte"))
    monkeypatch.setattr(c, "check_upstream", lambda: calls.append("check_upstream"))
    monkeypatch.setattr(c, "RESULTS", tmp_path)
    monkeypatch.setattr(rs, "load_arms", spy)
    np.savez(tmp_path / "arms.npz", **arms)
    sha = hashlib.sha256((tmp_path / "arms.npz").read_bytes()).hexdigest()
    record = {"passed": g3_passed, "g1": {"passed": True}, "g3": {"passed": g3_passed}, "npz": {"sha256": sha}}
    (tmp_path / "arms.json").write_text(json.dumps(record | {"date": "2026-11-25 11:41:06"}))
    return calls, sha


def test_readouts_stage_checks_first_then_writes_readouts_json(tmp_path, monkeypatch, reported_tables):
    arms = _synthetic_arms(np.random.default_rng(51))
    calls, _ = _readouts_stage_on(tmp_path, monkeypatch, arms, g3_passed=False)
    with pytest.raises(RuntimeError, match="arms.json"):
        rs.main(["readouts"])
    assert calls == ["check_topocyte", "check_upstream", "load_arms"] and not (tmp_path / "readouts.json").exists()
    calls, sha = _readouts_stage_on(tmp_path, monkeypatch, arms)
    rs.main(["readouts"])
    assert calls == ["check_topocyte", "check_upstream", "load_arms"]
    record = json.loads((tmp_path / "readouts.json").read_text())
    expected = ro.readouts(arms)  # the stage writes readouts.readouts of the arms it loaded
    assert {key: record[key] for key in expected} == expected
    assert record["arms_npz_sha256"] == sha and "arms" not in record
    assert set(record["provenance"]["md5"]) == {"run_study.py", "common.py", "readouts.py"}


def test_readouts_stage_writes_no_json_with_a_value_that_is_not_finite(tmp_path, monkeypatch, reported_tables):
    # A NaN in arm B reaches only the exploratory pairs and the tables, where no verdict refuses it.
    arms = _synthetic_arms(np.random.default_rng(52))
    arms["interp_r2_B"][7] = np.nan
    _readouts_stage_on(tmp_path, monkeypatch, arms)
    with pytest.raises(ValueError, match="JSON"):
        rs.main(["readouts"])
    assert not (tmp_path / "readouts.json").exists()


@needs_upstream
def test_check_upstream_accepts_the_pinned_clone():
    c.check_upstream()


@needs_upstream
def test_interpolation_regenerates_from_seed_zero():
    stored, regenerated = inp.interpolation_datasets(), inp.regenerate_interpolation()
    order = [(n, run) for n in c.INTERP_SIZES for run in range(c.INTERP_RUNS)]
    assert [(d["n_train"], d["run"]) for d in stored] == [(d["n_train"], d["run"]) for d in regenerated] == order
    for a, b in zip(stored, regenerated, strict=True):
        assert len(a["clouds"]) == len(b["clouds"]) == len(a["y"]) == a["n_train"] + c.INTERP_N_TEST
        assert np.array_equal(a["y"], b["y"])
        assert all(np.array_equal(p, q) for p, q in zip(a["clouds"], b["clouds"], strict=True))


@needs_upstream
def test_curvature_trains_on_the_first_fifty_clouds():
    data = inp.curvature_data()
    assert len(data["clouds"]) == 100 and data["y"].shape == (100,)
    assert np.array_equal(data["train"], np.arange(50)) and np.array_equal(data["test"], np.arange(50, 100))
    assert np.array_equal(data["y"][data["train"]], np.linspace(-2, 2, 50))


@needs_upstream
def test_fluid_diagrams_have_one_h0_and_two_h1_essential_bars():
    diagrams = inp.fluid_diagrams(5.0)
    assert len(diagrams) == 400
    for h0, h1 in diagrams[:20]:
        assert h0.dtype == h1.dtype == np.float64 and h0.shape[1:] == h1.shape[1:] == (2,)
        assert np.isinf(h0[:, 1]).sum() == 1 and np.isinf(h1[:, 1]).sum() == 2


@needs_upstream
def test_fluid_cached_w2_is_symmetric_and_complete():
    D = inp.fluid_cached_w2(1.0, 0)
    assert D.shape == (400, 400) and np.isfinite(D).all()
    assert np.array_equal(D, D.T) and not D.diagonal().any()


@needs_upstream
def test_w2_reproduces_upstreams_cached_distance_of_small_fluid_diagrams():
    # The two diagrams with the fewest bars hold only their three essential bars, so a second pair, the two with the
    # fewest bars among those with finite bars in both dimensions, checks the finite matching too.
    diagrams = inp.fluid_diagrams(1.0)
    bars = np.array([len(h0) + len(h1) for h0, h1 in diagrams])
    with_finite = np.array([all(np.isfinite(h[:, 1]).any() for h in d) for d in diagrams])
    fewest = np.argsort(bars, kind="stable")
    for h in (0, 1):
        cached = inp.fluid_cached_w2(1.0, h)
        for p, q in (fewest[:2], fewest[with_finite[fewest]][:2]):
            assert ws.w2(diagrams[p][h], diagrams[q][h]) == pytest.approx(cached[p, q], rel=1e-12, abs=0)
