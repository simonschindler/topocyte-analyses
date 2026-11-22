"""Unit tests of ``features.py`` on synthetic bars (no cohort).

topocyte is the oracle of the fPCA (``fpca_from_gram``) and of the unit-weight
Gram matrix (``betti_inner_product_matrix``), so the tests need the pinned
topocyte on ``PYTHONPATH``. The run tests write four small synthetic slides in
the layout of ``results/bars/{slide_id}.npz``, with ``common.N_CENTRES`` set to
24 and ``common.FIT_PER_SLIDE`` to 12, and run ``features.run`` on them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import chromatic
import common
import curves
import features

R = common.R
N_BLOCKS = len(common.BLOCKS)


def diagram(rng: np.random.Generator, n_bars: int, h0: bool) -> np.ndarray:
    """n finite bars in squared radii (px²), rounded to 0.5 so that endpoints tie across curves; H0 bars start at 0."""
    births = np.zeros(n_bars) if h0 else np.round(rng.uniform(0, 1e3, n_bars) * 2) / 2
    deaths = births + 0.5 + np.round(rng.exponential(300, n_bars) * 2) / 2
    return np.column_stack([births, deaths])


def diagrams(rng: np.random.Generator, n: int, hi: int = 8) -> list[np.ndarray]:
    """n random diagrams of 0..hi bars each, H0-like or not at random."""
    return [diagram(rng, int(rng.integers(0, hi + 1)), bool(rng.integers(2))) for _ in range(n)]


def fake_pieces(rng: np.random.Generator, n_centres: int, absent: int = 3) -> dict[tuple[int, int, int], np.ndarray]:
    """Random finite bars for every (centre, labelling, block); the blocks of class ``absent`` hold no bars at all."""
    pieces = {}
    for centre in range(n_centres):
        for lab in range(R + 1):
            for block, (t, _, p) in enumerate(common.BLOCKS):
                n_bars = 0 if t == absent else int(rng.integers(0, 7))
                pieces[centre, lab, block] = diagram(rng, n_bars, h0=p == 0)
    return pieces


def pack(pieces: dict[tuple[int, int, int], np.ndarray], n_centres: int) -> dict[str, np.ndarray]:
    """The bars of ``pieces`` as results/bars/*.npz stores them: sorted by owner, birth and death, with offsets."""
    keys = sorted(pieces, key=lambda key: common.encode_owner(*key))
    owner = np.concatenate([np.full(len(pieces[key]), common.encode_owner(*key), np.int32) for key in keys])
    bars = np.concatenate([pieces[key] for key in keys])
    order = np.lexsort((bars[:, 1], bars[:, 0], owner))
    n_owners = n_centres * (R + 1) * N_BLOCKS
    offsets = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=n_owners))]).astype(np.int64)
    return {"bars": bars[order], "owner": owner[order], "offsets": offsets}


def write_slide(path: Path, rng: np.random.Generator, pieces: dict, n_centres: int) -> None:
    """A slide file with ``pieces``' bars and random per-centre arrays (every ball holds all four classes)."""
    class_counts = rng.integers(1, 40, size=(n_centres, 4))
    np.savez(
        path,
        centre_rows=rng.choice(10_000, n_centres, replace=False),
        centre_xy=rng.uniform(0, 5e4, size=(n_centres, 2)),
        n_cells=class_counts.sum(axis=1),
        class_counts=class_counts,
        pair_counts=rng.integers(0, 60, size=(n_centres, len(common.PAIRS))),
        n_edges=rng.integers(100, 300, size=n_centres),
        seconds=np.ones(n_centres),
        **pack(pieces, n_centres),
        codomain=np.empty((0, 2)),
        codomain_owner=np.empty(0, np.int64),
    )


@pytest.fixture
def pilot_run(tmp_path, monkeypatch):
    """Four synthetic slides (two per half, slide indices 0..3) with 24 centres each and fit samples of 12.

    Slide 0, block 0: centre 0's references all agree while its labelling 0
    differs (zero spread, non-zero residual), and centre 1's 20 labellings all
    agree (zero spread, zero residual). Class 3 is absent everywhere, so its
    four blocks have no spread and are dropped in every fit.
    """
    monkeypatch.setattr(common, "N_CENTRES", 24)
    monkeypatch.setattr(common, "FIT_PER_SLIDE", 12)
    rng = np.random.default_rng(11)
    pilot = pd.DataFrame(
        {
            "slide_id": [f"S{k}" for k in range(4)],
            "condition": "healthy",
            "half": ["A", "B", "A", "B"],
            "slide_index": range(4),
        }
    )
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    slides = []
    for k in range(4):
        pieces = fake_pieces(rng, 24)
        if k == 0:
            same = diagram(rng, 4, h0=True)
            for lab in range(1, R + 1):
                pieces[0, lab, 0] = same
                pieces[1, lab, 0] = same
            pieces[0, 0, 0] = diagram(rng, 5, h0=True)
            pieces[1, 0, 0] = same
        write_slide(bars_dir / f"S{k}.npz", np.random.default_rng(100 + k), pieces, 24)
        slides.append(pieces)
    return pilot, bars_dir, slides


# ------------------------------------------------------------ per-centre features


def test_level1_features():
    L1 = features.level1(np.array([[50, 30, 15, 5]]))
    assert L1.shape == (1, 5)
    assert np.allclose(L1, [[2.0, 0.5, 0.3, 0.15, 0.05]], rtol=0, atol=1e-15)


def test_pairwise_features_subtract_the_expectation_row_by_row():
    class_counts = np.array([[3, 1, 0, 0], [2, 2, 1, 5]])
    n_edges = np.array([5, 21])
    pair_counts = np.arange(20).reshape(2, 10)
    P = features.pairwise(pair_counts, class_counts, n_edges)
    rows = zip(class_counts, n_edges, strict=True)
    expected = pair_counts - np.array([chromatic.pair_expectation(cc, e) for cc, e in rows])
    assert P.shape == (2, 10) and np.array_equal(P, expected)
    assert P[0, 0] == 0 - 5 * 3 / 6  # 5 edges, C(3, 2) epithelial pairs among C(4, 2)


def test_fit_sample_key():
    """400 sorted, distinct centre indices from the 4-entry key [SEED, slide_index, 0, 0xF17], one sample per slide."""
    sample = features.fit_sample(7)
    rng = np.random.default_rng(np.random.SeedSequence([common.SEED, 7, 0, 0xF17]))
    expected = np.sort(rng.choice(common.N_CENTRES, size=common.FIT_PER_SLIDE, replace=False))
    assert np.array_equal(sample, expected)
    assert sample.size == common.FIT_PER_SLIDE == np.unique(sample).size and sample.max() < common.N_CENTRES
    assert not np.array_equal(sample, features.fit_sample(8))
    # The 3-entry key pads to [SEED, 7, 0xF17, 0], which the study's no-trailing-zero rule forbids.
    padded = np.random.default_rng(np.random.SeedSequence([common.SEED, 7, 0xF17]))
    padded_sample = np.sort(padded.choice(common.N_CENTRES, size=common.FIT_PER_SLIDE, replace=False))
    assert not np.array_equal(sample, padded_sample)


# ------------------------------------------------------------ curve sets


def test_take_and_concat_copy_whole_curves():
    rng = np.random.default_rng(1)
    cs = curves.curve_set(diagrams(rng, 7))
    cs = curves.scale(cs, rng.uniform(0.5, 2.0, 7))
    sub = features.take(cs, np.array([5, 0, 5]))
    both = features.concat([sub, cs])
    assert len(sub) == 3 and len(both) == 10
    for j, i in enumerate([5, 0, 5, *range(7)]):
        src = slice(cs.offsets[i], cs.offsets[i + 1])
        out = slice(both.offsets[j], both.offsets[j + 1])
        assert np.array_equal(both.b[out], cs.b[src]) and np.array_equal(both.d[out], cs.d[src])
        assert np.array_equal(both.w[out], cs.w[src])
    assert len(features.take(cs, np.array([], dtype=np.int64))) == 0


def test_labellings_follow_the_owner_code():
    rng = np.random.default_rng(2)
    stored = pack(fake_pieces(rng, 5), 5)
    for block in (0, 5, 12):
        labs = features.labellings(stored["bars"], stored["offsets"], block)
        assert len(labs) == R + 1 and all(len(lab) == 5 for lab in labs)
        for r in range(R + 1):
            for centre in range(5):
                o = common.encode_owner(centre, r, block)
                expected = stored["bars"][stored["offsets"][o] : stored["offsets"][o + 1]]
                got = slice(labs[r].offsets[centre], labs[r].offsets[centre + 1])
                assert np.array_equal(np.column_stack([labs[r].b[got], labs[r].d[got]]), expected)
                assert np.all(labs[r].w[got] == 1.0)


def test_control_uses_labelling_1_against_2_to_19():
    """By inner products, the control is lab1 - mean(lab2..19), the observed lab0 - mean(lab1..19); spreads likewise."""
    rng = np.random.default_rng(3)
    stored = pack(fake_pieces(rng, 6), 6)
    labs = features.labellings(stored["bars"], stored["offsets"], 1)
    probe = curves.curve_set(diagrams(rng, 9))
    inner = [curves.inner_matrix(lab, probe) for lab in labs]
    for first in (0, 1):
        expected = inner[first] - sum(inner[first + 1 :]) / (R - first)
        got = curves.inner_matrix(features.residual(labs, first), probe)
        scale = np.max(np.abs(inner[first])) + np.max(np.abs(sum(inner[first + 1 :]))) / (R - first)
        assert np.max(np.abs(got - expected)) <= 1e-12 * scale
        refs = labs[first + 1 :]
        mean = curves.combine(refs, [1.0 / len(refs)] * len(refs))
        deviations = [np.diag(curves.inner_matrix(curves.combine([ref, mean], [1.0, -1.0]))) for ref in refs]
        brute = np.mean(deviations, axis=0)
        assert np.allclose(features.spread(labs, first), brute, rtol=1e-10, atol=0)
    # Labelling 0 plays no part in the control.
    swapped = [curves.curve_set(diagrams(rng, 6)), *labs[1:]]
    ctl = curves.inner_matrix(features.residual(labs, 1), probe)
    assert np.array_equal(curves.inner_matrix(features.residual(swapped, 1), probe), ctl)
    assert np.array_equal(features.spread(swapped, 1), features.spread(labs, 1))


# ------------------------------------------------------------ scaling


def test_s0_median_positive_only():
    assert features.s0_squared(np.array([0.0, 0.0, 0.0, 1.0, 4.0, 9.0])) == 4.0  # the median of all six is 0.5
    assert features.s0_squared(np.array([0.0, 2.0, 4.0])) == 3.0
    assert features.s0_squared(np.zeros(5)) == 0.0  # no positive spread: the block is dropped


def test_zero_spread_nonzero_residual_is_finite():
    """Review Focus 1: references that agree (s = 0) and an observed curve that differs keep a finite residual (via s0).

    Centre 0: the 19 references agree, labelling 0 differs. Centre 1: all 20
    labellings agree, so its residual is zero. Centres 2..5 have random
    references.
    """
    rng = np.random.default_rng(4)
    ref = np.array([[0.0, 3.0]])
    lab0 = [np.array([[0.0, 5.0], [0.0, 2.0]]), ref, *diagrams(rng, 4)]
    labs = [curves.curve_set(lab0)] + [curves.curve_set([ref, ref, *diagrams(rng, 4, hi=6)]) for _ in range(R)]
    s2 = features.spread(labs, 0)
    assert s2[0] == 0.0 and s2[1] == 0.0 and np.all(s2[2:] > 0)
    assert features.nonzero(labs).tolist() == [True, False, True, True, True, True]
    s0sq = features.s0_squared(s2)
    assert s0sq == np.median(s2[2:]) > 0
    scaled = features.scaled(features.residual(labs, 0), s2, s0sq)
    norms = np.diag(curves.inner_matrix(scaled))
    assert np.all(np.isfinite(scaled.w)) and np.all(np.isfinite(norms))
    # (1[0,5) + 1[0,2) - 1[0,3))² is 1 on [0, 2) and [3, 5), so ||residual||² = 4, divided by s0².
    assert norms[0] == pytest.approx(4.0 / s0sq, rel=1e-12)
    assert norms[1] <= 1e-28 * norms[0]  # zero up to the rounding of the weights -1/19


def test_scaled_divides_each_curve_by_its_moderated_spread():
    rng = np.random.default_rng(5)
    cs = curves.curve_set(diagrams(rng, 4, hi=5))
    s2 = np.array([0.0, 1.0, 3.0, 8.0])
    out = features.scaled(cs, s2, 1.0)
    assert np.array_equal(out.w, cs.w / np.repeat(np.sqrt(s2 + 1.0), np.diff(cs.offsets)))


# ------------------------------------------------------------ fPCA per block


def residual_set(rng: np.random.Generator, n: int) -> tuple[curves.CurveSet, np.ndarray]:
    """n residual curves of the study's form and their spreads."""
    labs = [curves.curve_set(diagrams(rng, n)) for _ in range(R + 1)]
    return features.residual(labs, 0), features.spread(labs, 0)


def test_fit_block_drops_blocks_without_spread_or_residual():
    rng = np.random.default_rng(6)
    residuals, s2 = residual_set(rng, 30)
    no_spread = features.fit_block(residuals, np.zeros(30), np.ones(30, bool))
    assert no_spread.dropped and no_spread.s0sq == 0.0 and no_spread.comp is None and no_spread.kept == 0
    all_zero = features.fit_block(residuals, s2, np.zeros(30, bool))
    assert all_zero.dropped and all_zero.s0sq == features.s0_squared(s2) and all_zero.comp is None
    for fit in (no_spread, all_zero):
        assert np.array_equal(fit.evr, np.zeros(features.N_COMPONENTS))
        assert np.array_equal(features.block_scores(residuals, s2, fit), np.zeros((30, features.N_COMPONENTS)))


def test_fit_block_scores_match_fpca_from_gram():
    """The fit sample's scores from the components equal fpca_from_gram's (gate 2), and block_scores gives them too."""
    import topocyte as tc

    rng = np.random.default_rng(7)
    residuals, s2 = residual_set(rng, 60)
    fit = features.fit_block(residuals, s2, np.ones(60, bool))
    assert not fit.dropped and fit.kept == features.N_COMPONENTS and fit.gate <= 1e-9
    curves_fit = features.scaled(residuals, s2, fit.s0sq)
    res = tc.ph.fpca_from_gram(curves.inner_matrix(curves_fit))
    S = features.block_scores(residuals, s2, fit)
    assert np.array_equal(S, fit.fit_scores)
    assert np.max(np.abs(S - res.scores[:, :9])) <= 1e-9 * np.max(np.abs(res.scores[:, :9]))
    assert np.allclose(fit.evr, res.explained_variance_ratio[:9], rtol=1e-12, atol=0)


def test_fit_block_keeps_only_the_components_of_a_rank_deficient_block():
    """Curves in a two-dimensional span keep two components; the other seven columns and fractions are 0."""
    rng = np.random.default_rng(8)
    n = 40
    u, v = (curves.curve_set([diagram(rng, 6, h0=False)] * n) for _ in range(2))
    span = curves.combine([curves.scale(u, rng.normal(size=n)), curves.scale(v, rng.normal(size=n))], [1.0, 1.0])
    s2 = rng.uniform(0.5, 2.0, n)
    fit = features.fit_block(span, s2, np.ones(n, bool))
    assert not fit.dropped and fit.kept == 2
    assert fit.evr[:2].sum() == pytest.approx(1.0, abs=1e-12) and np.all(fit.evr[2:] == 0)
    S = features.block_scores(span, s2, fit)
    assert S.shape == (n, features.N_COMPONENTS) and np.all(S[:, 2:] == 0) and np.all(np.abs(S[:, :2]).max(axis=0) > 0)


def test_gram_gate_compares_unit_curves_with_topocyte():
    rng = np.random.default_rng(9)
    for p in (0, 1):
        unit = curves.curve_set([diagram(rng, int(rng.integers(0, 9)), h0=p == 0) for _ in range(25)])
        assert features.gram_error(unit, p) <= 1e-12
        doubled = curves.scale(unit, np.full(25, 2.0))
        assert features.gram_error(doubled, p) == pytest.approx(3.0)  # 4 times topocyte's matrix


# ------------------------------------------------------------ MFA


def test_mfa_projection_matches_fit():
    rng = np.random.default_rng(10)
    blocks = [rng.normal(size=(300, 9)) * rng.uniform(0.5, 2.0, 9) * rng.uniform(0.1, 10.0) for _ in range(5)]
    scores, evr, weights, mean, loadings = features.mfa(blocks, 20)
    assert scores.shape == (300, 20) and loadings.shape == (20, 45) and mean.shape == (45,)
    assert np.max(np.abs(features.mfa_project(blocks, weights, mean, loadings) - scores)) <= 1e-12
    first_columns = [block[:, 0] * w for block, w in zip(blocks, weights, strict=True)]
    assert np.allclose([np.var(column) for column in first_columns], 1.0, rtol=1e-12, atol=0)
    assert np.all(scores[np.argmax(np.abs(scores), axis=0), np.arange(20)] > 0)  # the sibling's sign rule
    assert evr.size == 45 and evr.sum() == pytest.approx(1.0) and np.all(np.diff(evr) <= 1e-15)
    # New rows are centred with the fit's mean and projected on its loadings.
    new = [rng.normal(size=(7, 9)) for _ in range(5)]
    X = np.hstack([block * w for block, w in zip(new, weights, strict=True)])
    projected = features.mfa_project(new, weights, mean, loadings)
    assert np.allclose(projected, (X - mean) @ loadings.T, rtol=1e-14, atol=1e-14)


def test_mfa_refuses_fewer_components_than_kept():
    """Level 2 has 20 components: two blocks of 9 columns, or 15 rows, cannot give them."""
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError, match="20"):
        features.mfa([rng.normal(size=(100, 9)) for _ in range(2)], 20)
    with pytest.raises(ValueError, match="20"):
        features.mfa([rng.normal(size=(15, 9)) for _ in range(5)], 20)


def test_level2_projects_the_control_in_the_observed_basis():
    """The control is projected with the observed MFA's weights, centring and loadings, never fitted on its own.

    With control scores twice the observed ones (centred on the fit sample),
    the control's level-2 features are twice the observed features; an MFA of
    its own would undo the factor. A dropped block plays no part.
    """
    rng = np.random.default_rng(14)
    n = 200
    blk = rng.normal(size=(n, N_BLOCKS, 9)) * rng.uniform(0.2, 5.0, size=(N_BLOCKS, 9))
    rows = rng.random(n) < 0.6
    blk -= blk[rows].mean(axis=0)
    dropped = np.zeros(N_BLOCKS, bool)
    dropped[[3, 15]] = True
    obs, ctl, evr = features.level2(blk, 2 * blk, rows, dropped)
    assert obs.shape == ctl.shape == (n, 20) and evr.shape == (20,)
    assert np.allclose(ctl, 2 * obs, rtol=0, atol=1e-12 * np.abs(obs).max())
    kept = np.flatnonzero(~dropped)
    scores, expected_evr, *_ = features.mfa([blk[rows, b] for b in kept], 20)
    assert np.max(np.abs(obs[rows] - scores)) <= 1e-12 and np.array_equal(evr, expected_evr[:20])
    garbage = blk.copy()
    garbage[:, dropped] = rng.normal(size=(n, 2, 9)) * 1e3
    assert np.array_equal(features.level2(garbage, 2 * garbage, rows, dropped)[0], obs)


# ------------------------------------------------------------ the run

N_C = 24  # centres per synthetic slide
N = 4 * N_C


def fixture_masks(pilot: pd.DataFrame) -> dict[str, np.ndarray]:
    """The fit-sample rows of each fitting set of the synthetic pilot."""
    fit = np.zeros(N, bool)
    for k in range(4):
        fit[N_C * k + features.fit_sample(k)] = True
    half = np.repeat(pilot["half"].to_numpy(dtype=str), N_C)
    return {"A": fit & (half == "A"), "B": fit & (half == "B"), "all": fit}


def test_block_features_scale_each_residual_with_its_own_spread_and_the_fit_s0(pilot_run):
    """Observed residuals are scaled with s² of labellings 1..19 and control residuals with s² of 2..19, both with
    the fit's s0², which is the median positive observed s² of that fit's fit sample."""
    pilot, bars_dir, _ = pilot_run
    paths = [bars_dir / f"S{k}.npz" for k in range(4)]
    masks = fixture_masks(pilot)
    block = 1
    result = features.block_features(paths, masks, block)
    labs = []
    for path in paths:
        with np.load(path) as z:
            labs.append(features.labellings(z["bars"], z["offsets"], block))
    s2_obs = np.concatenate([features.spread(lab, 0) for lab in labs])
    for f in features.FITS:
        fit = result["fits"][f]
        assert fit.s0sq == features.s0_squared(s2_obs[masks[f]])
        for k, lab in enumerate(labs):
            rows = slice(N_C * k, N_C * (k + 1))
            observed, control = features.residual(lab, 0), features.residual(lab, 1)
            assert np.array_equal(result["obs"][f][rows], features.block_scores(observed, features.spread(lab, 0), fit))
            assert np.array_equal(result["ctl"][f][rows], features.block_scores(control, features.spread(lab, 1), fit))
            assert not np.allclose(result["ctl"][f][rows], features.block_scores(control, features.spread(lab, 0), fit))


def test_run_writes_every_feature(pilot_run, tmp_path):
    pilot, bars_dir, _ = pilot_run
    out = tmp_path / "features.npz"
    arrays = features.run(pilot, bars_dir, out)
    with np.load(out) as z:
        stored = {key: z[key] for key in z.files}
    assert set(stored) == set(arrays)
    for key, value in arrays.items():
        assert np.array_equal(stored[key], value), key
        assert value.dtype.kind in "biuU" or np.all(np.isfinite(value)), key
    assert np.array_equal(arrays["slide_index"], np.repeat(np.arange(4), N_C))
    assert arrays["half"].tolist() == (["A"] * N_C + ["B"] * N_C) * 2
    fit = np.zeros(N, bool)
    for k in range(4):
        fit[N_C * k + features.fit_sample(k)] = True
    assert np.array_equal(arrays["fit"], fit)
    for k in range(4):
        with np.load(bars_dir / f"S{k}.npz") as z:
            rows = slice(N_C * k, N_C * (k + 1))
            assert np.array_equal(arrays["centre_row"][rows], z["centre_rows"])
            assert np.array_equal(arrays["xy"][rows], z["centre_xy"])
            assert np.array_equal(arrays["L1"][rows], features.level1(z["class_counts"]))
            P = features.pairwise(z["pair_counts"], z["class_counts"], z["n_edges"])
            assert np.array_equal(arrays["P"][rows], P)
    assert arrays["L1"].shape == (N, 5) and arrays["P"].shape == (N, 10)
    expected_zsn = np.zeros(N_BLOCKS, np.int64)
    expected_zsn[0] = 1  # slide 0, centre 0
    assert np.array_equal(arrays["zero_spread_nonzero"], expected_zsn)
    assert arrays["gate_gram"].shape == (N_BLOCKS,) and np.all(arrays["gate_gram"] <= 1e-10)
    class3 = np.array([t == 3 for t, _, _ in common.BLOCKS])
    for f in features.FITS:
        assert arrays[f"L2_obs_{f}"].shape == arrays[f"L2_ctl_{f}"].shape == (N, 20)
        assert arrays[f"blk_obs_{f}"].shape == (N, N_BLOCKS, 9)
        assert arrays[f"evr_{f}"].shape == (N_BLOCKS, 9) and arrays[f"mfa_evr_{f}"].shape == (20,)
        assert np.array_equal(arrays[f"dropped_{f}"], class3)
        assert np.all(arrays[f"s0_{f}"][class3] == 0) and np.all(arrays[f"s0_{f}"][~class3] > 0)
        assert np.all(arrays[f"blk_obs_{f}"][:, class3] == 0) and np.all(arrays[f"evr_{f}"][class3] == 0)
        assert np.array_equal(arrays[f"kept_{f}"], np.where(class3, 0, 9))
        assert np.all(arrays[f"gate_scores_{f}"] <= 1e-9)
        assert np.all(np.abs(arrays[f"L2_obs_{f}"]).max(axis=0) > 0)
        assert np.all(np.abs(arrays[f"L2_ctl_{f}"]).max(axis=0) > 0)
    # Each fit sample's fPCA scores and MFA scores are centred.
    for f, rows in (("all", fit), ("A", fit & (arrays["half"] == "A")), ("B", fit & (arrays["half"] == "B"))):
        assert np.allclose(arrays[f"blk_obs_{f}"][rows].mean(axis=0), 0, atol=1e-10)
        assert np.allclose(arrays[f"L2_obs_{f}"][rows].mean(axis=0), 0, atol=1e-10)


def test_run_fits_on_the_fit_sample_and_scores_the_control_in_its_bases(pilot_run, tmp_path):
    """Changing a centre outside the fit samples moves only that centre's features, never s0, the bases or the MFA.

    Its labelling 0 changes only its observed features; a reference changes
    its control features too.
    """
    pilot, bars_dir, slides = pilot_run
    before = features.run(pilot, bars_dir, tmp_path / "before.npz")
    j = int(np.setdiff1d(np.arange(N_C), features.fit_sample(0))[0])
    rng = np.random.default_rng(12)
    slides[0][j, 0, 2] = diagram(rng, 6, h0=True)
    write_slide(bars_dir / "S0.npz", np.random.default_rng(100), slides[0], N_C)
    after_lab0 = features.run(pilot, bars_dir, tmp_path / "after_lab0.npz")
    slides[0][j, 5, 2] = diagram(rng, 6, h0=True)
    write_slide(bars_dir / "S0.npz", np.random.default_rng(100), slides[0], N_C)
    after_ref = features.run(pilot, bars_dir, tmp_path / "after_ref.npz")
    others = np.arange(N) != j
    changed = np.zeros((N, N_BLOCKS), bool)
    changed[j, 2] = True
    for f in features.FITS:
        for key in ("s0", "evr", "kept", "dropped", "mfa_evr", "gate_scores"):
            assert np.array_equal(after_lab0[f"{key}_{f}"], before[f"{key}_{f}"])
            assert np.array_equal(after_ref[f"{key}_{f}"], before[f"{key}_{f}"])
        obs, ctl, blk = f"L2_obs_{f}", f"L2_ctl_{f}", f"blk_obs_{f}"
        assert np.array_equal(after_lab0[obs][others], before[obs][others])
        assert not np.array_equal(after_lab0[obs][j], before[obs][j])
        assert np.array_equal(after_lab0[blk][~changed], before[blk][~changed])
        assert not np.array_equal(after_lab0[blk][j, 2], before[blk][j, 2])
        assert np.array_equal(after_lab0[ctl], before[ctl])
        assert np.array_equal(after_ref[ctl][others], before[ctl][others])
        assert not np.array_equal(after_ref[ctl][j], before[ctl][j])
