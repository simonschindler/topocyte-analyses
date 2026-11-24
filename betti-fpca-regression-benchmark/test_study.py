"""Tests of the Betti FPCR benchmark.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``requirements.txt``):

    $PY -m pytest -q -W error test_study.py

The tests marked ``needs_upstream`` read the pinned clone in ``upstream/``,
its pickles only through ``safe_pickle.load_pickle``, and are skipped until
``fetch_upstream.py`` has made it. The others use synthetic data.
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import pickle
import subprocess
import warnings

import numpy as np
import pytest
import ripser
import topocyte as tc

import betti
import common as c
import estimators as est
import fetch_upstream
import inputs as inp
import oracles
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


def test_ensure_upstream_refuses_a_clone_at_another_commit(tmp_path, monkeypatch):
    # a fresh `git init` repository with one commit stands in for upstream
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "README.md").write_text("a stand-in for upstream\n")
    config = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", "README.md"], ["commit", "-q", "--no-verify", "-m", "a"]):
        subprocess.run(["git", "-C", str(repo), *config, *args], check=True, capture_output=True)
    monkeypatch.setattr(fetch_upstream, "UPSTREAM", repo)
    with pytest.raises(RuntimeError, match="fe7d652"):
        fetch_upstream.ensure_upstream()


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
