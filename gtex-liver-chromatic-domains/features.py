#!/usr/bin/env python
"""Features of the chromatic-domains pilot (design steps 5–7 and 11): ``results/features.npz``.

Run from this folder with the study venv and
``PYTHONPATH=<topocyte checkout at 18e3aa5>``; it refuses any other topocyte
(``common.check_topocyte``). It reads ``results/pilot.csv`` and
``results/bars/{slide_id}.npz`` and writes ``results/features.npz``, one row
per centre: slides in slide-index order, then centre index 0..1499.

Per centre:
- ``slide_index``, ``half``, ``centre_row`` (the centre cell's row in its
  slide), ``xy``, and ``fit``: whether the centre is in its slide's fit sample;
- ``L1`` (N, 5): log10 of the ball's cells, then its four class fractions
  (level 1, design step 8);
- ``P`` (N, 10): the ball's Delaunay edges per class pair (``common.PAIRS``)
  minus their exact expectation when the classes are shuffled among its cells
  (step 11).

Per fitting set f in ``FITS``, half A, half B and all 30 slides (step 6):
- ``blk_obs_{f}`` (N, 16, 9): each block's fPCA scores of the observed residuals;
- ``L2_obs_{f}``, ``L2_ctl_{f}`` (N, 20): the MFA of the blocks (step 7), observed and control;
- ``s0_{f}`` (16,): each block's s0; ``kept_{f}`` (16,): its components kept;
  ``dropped_{f}`` (16,): whether it was dropped; ``gate_scores_{f}`` (16,): gate 2's error;
- ``evr_{f}`` (16, 9): the components' fractions of variance; ``mfa_evr_{f}`` (20,): the MFA's.

Diagnostics: ``zero_spread_nonzero`` (16,), the centres whose references all
agree (s = 0) while their observed residual is not zero; ``gate_gram`` (16,),
gate 1's error.

Residual curves (step 5). Block b = (class t, module, degree p) of
``common.BLOCKS``. Each labelling's curve is the reduced Betti curve of its
stored finite bars, with unit weights. The observed residual of a centre is
labelling 0 minus the mean of labellings 1..19, ``combine`` with weights 1 and
-1/19, every bar kept. Its spread s² is that of labellings 1..19
(``curves.spread2``). The residual is divided by sqrt(s² + s0²), where s0² is
the median of the positive s² over the fitting set's fit sample.

Fit samples and fits (step 6). Each slide has one fit sample: 400 of its
1,500 centres, drawn without replacement with the 4-entry key
``[SEED, slide_index, 0, FIT_STREAM]`` and sorted. Fitting set f fits on the
fit samples of its slides: 6,000 centres for a half, 12,000 for all. For each
block, the scaled residuals of that sample give an exact Gram matrix
(``curves.inner_matrix``) and ``topocyte.ph.fpca_from_gram`` (normalization
"none", divisor n). Every centre is then scored in that basis (``curves.score``),
with its own s² and the fit's s0, so s0, the components and the MFA weights all
come from one sample and are applied unchanged to every centre.

Dropped blocks and rank (the controller's rulings for Task 6):
- a block is dropped in a fit when no fit centre has s > 0, or when every fit
  residual is zero, decided exactly (``nonzero``). Its ``blk`` columns are 0,
  its s0 is 0 when it has no positive s², and it is left out of that fit's MFA;
- a block keeps at most 9 components with an eigenvalue above 1e-12 times its
  largest; with fewer, its remaining columns and fractions are 0.
  ``fpca_from_gram`` has by then dropped every eigenvalue up to 1e-10 of the
  Gram matrix's trace (its default ``rtol``), and the trace is at least the
  largest centred eigenvalue. So every component it returns passes the
  ruling's threshold, and a block keeps the first 9 it returns, or all of
  them when there are fewer.

Control (design, "Checks fixed in advance", 1). Labelling 1 minus the mean of
labellings 2..19, with its own spread (labellings 2..19) and the observed fit's
s0, scored in the observed fit's components and projected with its MFA weights,
centring and loadings. Nothing is fitted on the control.

MFA (step 7), as in the sibling studies (``pca`` and ``mfa`` copied from
``gtex-liver-celltype-slices/common.py``): on the fit sample, each kept block
divided by the square root of the variance of its first score, the columns
concatenated, and a PCA of the centred concatenation with the sibling sign
rule. Every centre is projected with that centring and those loadings; 20
components are kept.

Gates; the run stops if either fails:
1. the unit-weight Gram matrix of 200 observed labelling-0 curves of each block
   (every 60th centre of the all-slides fit sample) equals topocyte's
   ``betti_inner_product_matrix`` to 1e-10 relative, entry by entry;
2. each fit sample's scores from ``curves.components`` equal
   ``fpca_from_gram``'s to 1e-9 of the largest score.

Memory: one block at a time, in two passes over the slides. A slide's bars are
read, the block's 20 labellings extracted, and the bars released. The first
pass keeps the spreads, the zero tests and the fit sample's residuals (about a
quarter of the block's bars); the second scores every centre. No copy of the
bars is ever written.
"""

from __future__ import annotations

import dataclasses
import math
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import topocyte as tc

import common as c
import curves
from chromatic import pair_expectation

FITS = ("A", "B", "all")  # the fitting sets: the two halves and all slides
FIT_STREAM = 0xF17  # the stream of the fit-sample key [SEED, slide_index, 0, FIT_STREAM]
N_COMPONENTS = 9  # fPCA components per block
N_MFA = 20  # MFA components kept: the level-2 feature space
MFA_RTOL = 1e-12  # an MFA component counts when its variance exceeds this times the first's
GATE_CURVES = 200  # observed labelling-0 curves per block in gate 1
GATE_GRAM_RTOL = 1e-10
GATE_SCORES_RTOL = 1e-9


def fit_sample(slide_index: int) -> np.ndarray:
    """The slide's fit sample: ``FIT_PER_SLIDE`` of its ``N_CENTRES`` centre indices, without replacement, sorted.

    The generator's key is ``[SEED, slide_index, 0, FIT_STREAM]``. Its last
    entry is non-zero, so SeedSequence's zero padding cannot make it collide
    with the pilot draw or another key (``common``'s docstring).
    """
    rng = np.random.default_rng(np.random.SeedSequence([c.SEED, int(slide_index), 0, FIT_STREAM]))
    return np.sort(rng.choice(c.N_CENTRES, size=c.FIT_PER_SLIDE, replace=False))


# ------------------------------------------------------------ per-centre features


def level1(class_counts: np.ndarray) -> np.ndarray:
    """(n, 5): log10 of each ball's cells, then its four class fractions."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(c.CLASSES):
        raise ValueError(f"class counts must be (n, {len(c.CLASSES)}); got shape {counts.shape}")
    cells = counts.sum(axis=1)
    if np.any(cells < 1):
        raise ValueError("every ball holds at least its centre cell")
    return np.column_stack([np.log10(cells), counts / cells[:, None]])


def pairwise(pair_counts: np.ndarray, class_counts: np.ndarray, n_edges: np.ndarray) -> np.ndarray:
    """(n, 10): each ball's edges per class pair minus their expectation under random labelling.

    The expectation is ``chromatic.pair_expectation``, called on one ball's
    (4,) class counts at a time.
    """
    pair_counts, class_counts, n_edges = (np.asarray(a) for a in (pair_counts, class_counts, n_edges))
    n = len(pair_counts)
    if pair_counts.shape != (n, len(c.PAIRS)) or class_counts.shape != (n, len(c.CLASSES)) or n_edges.shape != (n,):
        raise ValueError("need (n, 10) pair counts, (n, 4) class counts and (n,) edge counts")
    expected = np.array([pair_expectation(row, int(edges)) for row, edges in zip(class_counts, n_edges, strict=True)])
    return pair_counts - expected.reshape(n, len(c.PAIRS))


# ------------------------------------------------------------ curve sets


def take(cs: curves.CurveSet, index: np.ndarray) -> curves.CurveSet:
    """Curves ``index`` of ``cs``, in that order, with their bars and weights copied."""
    index = np.asarray(index, dtype=np.int64)
    lengths = np.diff(cs.offsets)[index]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    rows = np.repeat(cs.offsets[index] - offsets[:-1], lengths) + np.arange(offsets[-1])
    return curves.CurveSet(cs.b[rows], cs.d[rows], cs.w[rows], offsets)


def concat(sets: Sequence[curves.CurveSet]) -> curves.CurveSet:
    """The curves of ``sets``, one set after another."""
    sets = list(sets)
    if not sets:
        raise ValueError("concat needs at least one curve set")
    lengths = np.concatenate([np.diff(s.offsets) for s in sets])
    return curves.CurveSet(
        np.concatenate([s.b for s in sets]),
        np.concatenate([s.d for s in sets]),
        np.concatenate([s.w for s in sets]),
        np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
    )


def labellings(bars: np.ndarray, offsets: np.ndarray, block: int) -> list[curves.CurveSet]:
    """Labellings 0..R of one block of a slide, as unit-weight curve sets with one curve per centre.

    ``bars`` and ``offsets`` are a ``results/bars`` file's: the bars of owner
    ``common.encode_owner(centre, labelling, block)`` are
    ``bars[offsets[o]:offsets[o + 1]]``, finite and with death > birth. A
    curve is their reduced Betti curve. The number of centres follows from
    the offsets.
    """
    per_centre = (c.R + 1) * len(c.BLOCKS)
    if (len(offsets) - 1) % per_centre:
        raise ValueError(f"{len(offsets)} offsets do not cover whole centres of {per_centre} owners")
    centres = np.arange((len(offsets) - 1) // per_centre)
    sets = []
    for r in range(c.R + 1):
        owners = c.encode_owner(centres, r, block).astype(np.int64)
        starts = offsets[owners]
        lengths = offsets[owners + 1] - starts
        cs_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        rows = np.repeat(starts - cs_offsets[:-1], lengths) + np.arange(cs_offsets[-1])
        sets.append(curves.CurveSet(bars[rows, 0], bars[rows, 1], np.ones(rows.size), cs_offsets))
    return sets


def residual(labs: Sequence[curves.CurveSet], first: int) -> curves.CurveSet:
    """Each centre's curve of labelling ``first`` minus the mean of its curves of labellings first + 1..R.

    ``first`` = 0 is the observed residual: labelling 0 with weight 1,
    labellings 1..19 with -1/19. ``first`` = 1 is the control: labelling 1
    against 2..19, weights 1 and -1/18. Every bar is kept (``curves.combine``).
    """
    n_refs = len(labs) - 1 - first
    if first < 0 or n_refs < 1:
        raise ValueError(f"labelling {first} has no later labellings among {len(labs)}")
    return curves.combine(labs[first:], [1.0] + [-1.0 / n_refs] * n_refs)


def spread(labs: Sequence[curves.CurveSet], first: int) -> np.ndarray:
    """Each centre's s² for the residual of labelling ``first``: the spread of labellings first + 1..R."""
    return curves.spread2(labs[first + 1 :])


def nonzero(labs: Sequence[curves.CurveSet]) -> np.ndarray:
    """Whether each centre's observed residual is a non-zero function, decided exactly.

    The residual is zero if and only if R·lab0 − Σ_r lab_r is, a curve whose
    step values are exact integers. Its squared norm is four times
    ``curves.spread2`` of the two sets R·lab0 and Σ_r lab_r, a sweep that is 0
    exactly when they agree. The residual itself cannot tell: its weights
    -1/R are rounded, so a zero residual keeps a residue of about 1e-17.
    """
    n_refs = len(labs) - 1
    observed = curves.scale(labs[0], np.full(len(labs[0]), float(n_refs)))
    return curves.spread2([observed, curves.combine(labs[1:], [1.0] * n_refs)]) > 0


# ------------------------------------------------------------ scaling and fPCA


def s0_squared(s2: np.ndarray) -> float:
    """A block's s0²: the median of the positive s² of a fit sample, or 0 when none is positive."""
    s2 = np.asarray(s2, dtype=np.float64)
    positive = s2[s2 > 0]
    return float(np.median(positive)) if positive.size else 0.0


def scaled(cs: curves.CurveSet, s2: np.ndarray, s0sq: float) -> curves.CurveSet:
    """Each curve divided by sqrt(s² + s0²): its own spread, moderated by the block's (design step 5)."""
    return curves.scale(cs, 1.0 / np.sqrt(np.asarray(s2, dtype=np.float64) + s0sq))


@dataclass
class BlockFit:
    """One block's fPCA on one fitting set's fit sample.

    ``s0sq`` is the block's s0² (0 without a positive s²). ``comp`` holds its
    first ``kept`` component functions, or None for a dropped block, with
    ``reason`` saying why. ``evr`` holds the components' fractions of
    variance, 0 beyond ``kept``. ``fit_scores`` are the fit sample's scores
    (n_fit, N_COMPONENTS), and ``gate`` their largest deviation from
    ``fpca_from_gram``'s, relative to the largest score (gate 2). ``seconds``
    times the steps.
    """

    s0sq: float
    comp: curves.Components | None = None
    kept: int = 0
    evr: np.ndarray = field(default_factory=lambda: np.zeros(N_COMPONENTS))
    fit_scores: np.ndarray | None = None
    gate: float = 0.0
    reason: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def dropped(self) -> bool:
        return self.comp is None


def fit_block(residuals: curves.CurveSet, s2: np.ndarray, nonzero: np.ndarray) -> BlockFit:
    """The fPCA of one block on a fit sample: its unscaled observed residuals, their s² and their zero tests.

    The block is dropped when no centre has s > 0, when every residual is
    zero, or when the centred Gram matrix has no eigenvalue above
    ``fpca_from_gram``'s threshold. Otherwise the residuals are scaled with
    this sample's s0², and their exact Gram matrix goes through
    ``fpca_from_gram`` (normalization "none", divisor n, default ``rtol``).
    The first ``N_COMPONENTS`` components it returns are kept, or all of them
    when there are fewer (module docstring: they all pass the rank ruling's
    1e-12 of the largest). Raises ``RuntimeError`` when gate 2 fails.
    """
    s0sq = s0_squared(s2)
    if s0sq == 0.0:
        return BlockFit(0.0, reason="no fit centre with s > 0")
    if not np.any(nonzero):
        return BlockFit(s0sq, reason="every fit residual is zero")
    seconds = {}
    t = time.perf_counter()
    fit = scaled(residuals, s2, s0sq)
    held = [curves.inner_matrix(fit)]
    seconds["gram"] = time.perf_counter() - t
    t = time.perf_counter()
    # Handed over without a name, so that fpca_from_gram's symmetrized copy replaces it (1.2 GB at 12,000 curves).
    res = tc.ph.fpca_from_gram(held.pop())
    kept = min(N_COMPONENTS, res.gram_eigenvalues.size)
    if kept == 0:
        return BlockFit(s0sq, reason="no component above fpca_from_gram's threshold")
    # Keep what the components and the gate read; the result's n x n matrices go now.
    res = dataclasses.replace(
        res,
        scores=res.scores[:, :kept].copy(),
        eigenvectors=res.eigenvectors[:, :kept].copy(),
        raw_gram=None,
        analysis_gram=None,
        centered_gram=None,
    )
    seconds["fpca"] = time.perf_counter() - t
    t = time.perf_counter()
    comp = curves.components(fit, res, kept)
    S = curves.score(fit, comp)
    seconds["components"] = time.perf_counter() - t
    gate = float(np.max(np.abs(S - res.scores)) / np.max(np.abs(res.scores)))
    if not gate <= GATE_SCORES_RTOL:
        raise RuntimeError(f"gate 2 failed: the fit sample's scores deviate from fpca_from_gram's by {gate:.3e}")
    evr = np.zeros(N_COMPONENTS)
    evr[:kept] = res.explained_variance_ratio[:kept]
    fit_scores = np.zeros((len(fit), N_COMPONENTS))
    fit_scores[:, :kept] = S
    return BlockFit(s0sq, comp, kept, evr, fit_scores, gate, seconds=seconds)


def block_scores(residuals: curves.CurveSet, s2: np.ndarray, fit: BlockFit) -> np.ndarray:
    """(n, N_COMPONENTS): residual curves scored in a block's fitted basis.

    Each curve is scaled with its own s² and the fit's s0². Columns beyond the
    kept components, and every column of a dropped block, are 0.
    """
    out = np.zeros((len(residuals), N_COMPONENTS))
    if fit.comp is not None:
        out[:, : fit.kept] = curves.score(scaled(residuals, s2, fit.s0sq), fit.comp)
    return out


def gram_error(unit: curves.CurveSet, p: int) -> float:
    """Gate 1: the largest relative deviation, entry by entry, of ``inner_matrix(unit)`` from topocyte's.

    topocyte's matrix is ``betti_inner_product_matrix`` of the Betti curves of
    the same bars in degree ``p``, so ``unit`` must have unit weights for the
    two to agree. An entry that is 0 in topocyte's matrix must be 0 here,
    or the error is inf.
    """
    G = curves.inner_matrix(unit)
    bounds = zip(unit.offsets[:-1], unit.offsets[1:], strict=True)
    diagrams = [np.column_stack([unit.b[lo:hi], unit.d[lo:hi]]) for lo, hi in bounds]
    T = tc.ph.betti_inner_product_matrix([tc.ph.BettiCurve(diagram, p) for diagram in diagrams], dim=p)
    deviation = np.abs(G - T)
    zero = T == 0
    if np.any(deviation[zero] > 0):
        return math.inf
    return float(np.max(deviation[~zero] / np.abs(T[~zero]))) if np.any(~zero) else 0.0


# ------------------------------------------------------------ MFA (the sibling studies')


def pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, fractions of variance and components of the centred matrix, signs fixed (sibling rule)."""
    centred = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    # Sign: the largest-magnitude entry of each left singular vector is positive.
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    signs[signs == 0] = 1.0
    u, vt = u * signs, vt * signs[:, None]
    variance = s**2
    return u * s, variance / variance.sum(), vt


def mfa(blocks: Sequence[np.ndarray], n_keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multiple factor analysis of score blocks, with the basis that ``mfa_project`` applies to other rows.

    As the sibling's ``common.mfa``: each block is divided by the square root
    of the variance of its first column, and the centred concatenation goes
    through ``pca``. Returns the first ``n_keep`` scores, every fraction of
    variance, the block weights, the concatenation's column means and the
    first ``n_keep`` loadings (rows of vt). ``ValueError`` when fewer than
    ``n_keep`` components have a variance above ``MFA_RTOL`` times the first:
    level 2 has ``N_MFA`` components, never rounding noise.
    """
    weights = np.array([1.0 / np.sqrt(block[:, 0].var()) for block in blocks])
    X = np.hstack([block * w for block, w in zip(blocks, weights, strict=True)])
    scores, evr, vt = pca(X)
    real = int(np.count_nonzero(evr > MFA_RTOL * evr[0]))
    if real < n_keep:
        raise ValueError(f"the MFA has {real} components with variance, fewer than the {n_keep} kept")
    return scores[:, :n_keep], evr, weights, X.mean(axis=0), vt[:n_keep]


def mfa_project(
    blocks: Sequence[np.ndarray], weights: np.ndarray, mean: np.ndarray, loadings: np.ndarray
) -> np.ndarray:
    """Rows in a fitted MFA basis: the blocks times their weights, concatenated, centred with the fit's means."""
    X = np.hstack([block * w for block, w in zip(blocks, weights, strict=True)])
    return (X - mean) @ loadings.T


def level2(
    blk_obs: np.ndarray, blk_ctl: np.ndarray, rows: np.ndarray, dropped: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One fitting set's level-2 features: L2_obs and L2_ctl (N, N_MFA) and the MFA's first N_MFA fractions of variance.

    The MFA is fitted on the observed block scores (N, 16, 9) of the fit
    sample (``rows``), leaving out the ``dropped`` blocks. Every centre's
    observed and control scores are projected with its weights, centring and
    loadings.
    """
    blocks = np.flatnonzero(~np.asarray(dropped))
    if blocks.size == 0:
        raise RuntimeError("every block was dropped")
    _, evr, weights, mean, loadings = mfa([blk_obs[rows, b] for b in blocks], N_MFA)
    obs = mfa_project([blk_obs[:, b] for b in blocks], weights, mean, loadings)
    ctl = mfa_project([blk_ctl[:, b] for b in blocks], weights, mean, loadings)
    return obs, ctl, evr[:N_MFA]


# ------------------------------------------------------------ the run


def _peak_rss_gb() -> float:
    """This process's peak resident set size so far, in GB (``ru_maxrss`` counts bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (peak if sys.platform == "darwin" else 1024 * peak) / 1e9


def _labellings_of(path: Path, block: int) -> list[curves.CurveSet]:
    """One block's labellings from a slide's bars file, whose bars are released on return."""
    with np.load(path, allow_pickle=False) as z:
        return labellings(z["bars"], z["offsets"], block)


def per_centre(pilot: pd.DataFrame, paths: Sequence[Path]) -> dict[str, np.ndarray]:
    """The per-centre arrays: slide index, half, centre row, position, fit-sample membership, ``L1`` and ``P``."""
    n_c = c.N_CENTRES
    n_offsets = n_c * (c.R + 1) * len(c.BLOCKS) + 1
    keys = ("centre_rows", "centre_xy", "n_cells", "class_counts", "pair_counts", "n_edges")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    fit = []
    for path, slide_index in zip(paths, pilot["slide_index"], strict=True):
        with np.load(path, allow_pickle=False) as z:
            if z["offsets"].shape != (n_offsets,):
                raise ValueError(f"{path.name}: offsets of shape {z['offsets'].shape}, not ({n_offsets},)")
            for key in keys:
                if len(z[key]) != n_c:
                    raise ValueError(f"{path.name}: {key} has {len(z[key])} rows, not {n_c}")
                parts[key].append(z[key])
        member = np.zeros(n_c, bool)
        member[fit_sample(slide_index)] = True
        fit.append(member)
    joined = {key: np.concatenate(value) for key, value in parts.items()}
    if not np.array_equal(joined["class_counts"].sum(axis=1), joined["n_cells"]):
        raise ValueError("class counts do not add up to the balls' cells")
    return {
        "slide_index": np.repeat(pilot["slide_index"].to_numpy(dtype=np.int64), n_c),
        "half": np.repeat(pilot["half"].to_numpy(dtype=str), n_c),
        "centre_row": joined["centre_rows"].astype(np.int64),
        "xy": joined["centre_xy"].astype(np.float64),
        "fit": np.concatenate(fit),
        "L1": level1(joined["class_counts"]),
        "P": pairwise(joined["pair_counts"], joined["class_counts"], joined["n_edges"]),
    }


def block_features(paths: Sequence[Path], masks: dict[str, np.ndarray], block: int) -> dict:
    """One block over all slides: the spreads and zero tests, gate 1, the three fits, and every centre's scores.

    ``masks[f]`` marks the rows of fitting set f's fit sample. Returns the
    fits, the observed and control scores per fit (N, N_COMPONENTS), gate 1's
    error, the count of zero-spread non-zero residuals and the passes' times.
    """
    _, _, p = c.BLOCKS[block]
    n_c = c.N_CENTRES
    N = len(paths) * n_c
    t = time.perf_counter()
    # Pass 1: per slide, the spreads and zero tests of every centre, and the residuals of its fit sample.
    s2_obs, s2_ctl, nz = np.empty(N), np.empty(N), np.empty(N, bool)
    fit_rows = np.flatnonzero(masks["all"])
    gate_rows = fit_rows[:: max(1, fit_rows.size // GATE_CURVES)][:GATE_CURVES]
    fit_pieces, gate_pieces = [], []
    for k, path in enumerate(paths):
        rows = slice(k * n_c, (k + 1) * n_c)
        labs = _labellings_of(path, block)
        s2_obs[rows], s2_ctl[rows], nz[rows] = spread(labs, 0), spread(labs, 1), nonzero(labs)
        fit_pieces.append(take(residual(labs, 0), np.flatnonzero(masks["all"][rows])))
        gate_pieces.append(take(labs[0], gate_rows[(gate_rows >= rows.start) & (gate_rows < rows.stop)] - rows.start))
    gate_gram = gram_error(concat(gate_pieces), p)
    if not gate_gram <= GATE_GRAM_RTOL:
        raise RuntimeError(f"gate 1 failed in block {block}: the unit-weight Gram matrix deviates by {gate_gram:.3e}")
    pass1 = time.perf_counter() - t
    # The fits, the largest first; the fit set's curves are the fit rows of all slides in row order.
    fit_set = concat(fit_pieces)
    del fit_pieces
    fits = {}
    for f in ("all", "A", "B"):
        own = np.flatnonzero(masks[f][fit_rows])
        fits[f] = fit_block(fit_set if f == "all" else take(fit_set, own), s2_obs[masks[f]], nz[masks[f]])
    del fit_set
    # Pass 2: every centre's observed and control residuals, scored in each fit's basis.
    t = time.perf_counter()
    obs = {f: np.zeros((N, N_COMPONENTS)) for f in FITS}
    ctl = {f: np.zeros((N, N_COMPONENTS)) for f in FITS}
    for k, path in enumerate(paths):
        rows = slice(k * n_c, (k + 1) * n_c)
        labs = _labellings_of(path, block)
        observed, control = residual(labs, 0), residual(labs, 1)
        for f in FITS:
            obs[f][rows] = block_scores(observed, s2_obs[rows], fits[f])
            ctl[f][rows] = block_scores(control, s2_ctl[rows], fits[f])
    for f in FITS:
        if not fits[f].dropped and not np.array_equal(obs[f][masks[f]], fits[f].fit_scores):
            raise RuntimeError(f"block {block}, fit {f}: the fit sample's scores differ between the fit and pass 2")
    return {
        "fits": fits,
        "obs": obs,
        "ctl": ctl,
        "gate_gram": gate_gram,
        "zero_spread_nonzero": int(np.count_nonzero((s2_obs == 0) & nz)),
        "seconds": {"pass1": pass1, "pass2": time.perf_counter() - t},
    }


def _block_line(block: int, result: dict) -> str:
    t, module, p = c.BLOCKS[block]
    parts = []
    for f in FITS:
        fit = result["fits"][f]
        if fit.dropped:
            parts.append(f"{f}: dropped ({fit.reason})")
        else:
            steps = ", ".join(f"{name} {seconds:.0f} s" for name, seconds in fit.seconds.items())
            parts.append(
                f"{f}: s0 {math.sqrt(fit.s0sq):.4g}, {fit.kept} components, FVE {fit.evr.sum():.3f}, "
                f"gate 2 {fit.gate:.1e} ({steps})"
            )
    seconds = result["seconds"]
    return (
        f"block {block} ({c.CLASSES[t]} {module} H{p}): gate 1 {result['gate_gram']:.1e}; "
        f"zero spread, non-zero residual: {result['zero_spread_nonzero']}; " + "; ".join(parts) + "; "
        f"pass 1 {seconds['pass1']:.0f} s, pass 2 {seconds['pass2']:.0f} s; peak RSS {_peak_rss_gb():.2f} GB"
    )


def run(pilot: pd.DataFrame, bars_dir: Path, out: Path, log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """Compute every feature of the pilot's centres, write them to ``out`` (an npz, atomically) and return them.

    ``pilot`` holds ``slide_id``, ``half`` and ``slide_index`` (0..n-1); each
    slide's bars are ``bars_dir/{slide_id}.npz``.
    """
    start = time.perf_counter()
    pilot = pilot.sort_values("slide_index", ignore_index=True)
    if not np.array_equal(pilot["slide_index"].to_numpy(), np.arange(len(pilot))):
        raise ValueError("slide indices must run 0..n-1")
    if not pilot["half"].isin(["A", "B"]).all():
        raise ValueError("every slide belongs to half A or half B")
    paths = [Path(bars_dir) / f"{slide_id}.npz" for slide_id in pilot["slide_id"]]
    arrays = per_centre(pilot, paths)
    N, n_blocks = len(arrays["fit"]), len(c.BLOCKS)
    masks = {half: arrays["fit"] & (arrays["half"] == half) for half in ("A", "B")} | {"all": arrays["fit"]}
    log(f"{len(paths)} slides, {N} centres; fit samples: " + ", ".join(f"{f} {int(masks[f].sum())}" for f in FITS))
    blk_obs = {f: np.zeros((N, n_blocks, N_COMPONENTS)) for f in FITS}
    blk_ctl = {f: np.zeros((N, n_blocks, N_COMPONENTS)) for f in FITS}
    s0 = {f: np.zeros(n_blocks) for f in FITS}
    kept = {f: np.zeros(n_blocks, np.int64) for f in FITS}
    dropped = {f: np.zeros(n_blocks, bool) for f in FITS}
    gate_scores = {f: np.zeros(n_blocks) for f in FITS}
    evr = {f: np.zeros((n_blocks, N_COMPONENTS)) for f in FITS}
    gate_gram = np.zeros(n_blocks)
    zero_spread_nonzero = np.zeros(n_blocks, np.int64)
    for block in range(n_blocks):
        result = block_features(paths, masks, block)
        gate_gram[block] = result["gate_gram"]
        zero_spread_nonzero[block] = result["zero_spread_nonzero"]
        for f in FITS:
            fit = result["fits"][f]
            blk_obs[f][:, block], blk_ctl[f][:, block] = result["obs"][f], result["ctl"][f]
            s0[f][block], kept[f][block], dropped[f][block] = math.sqrt(fit.s0sq), fit.kept, fit.dropped
            gate_scores[f][block], evr[f][block] = fit.gate, fit.evr
        log(_block_line(block, result) + f"; {time.perf_counter() - start:.0f} s so far")
        del result
    for f in FITS:
        L2 = level2(blk_obs[f], blk_ctl[f], masks[f], dropped[f])
        arrays[f"L2_obs_{f}"], arrays[f"L2_ctl_{f}"], arrays[f"mfa_evr_{f}"] = L2
        arrays[f"blk_obs_{f}"] = blk_obs[f]
        arrays[f"s0_{f}"], arrays[f"kept_{f}"], arrays[f"dropped_{f}"] = s0[f], kept[f], dropped[f]
        arrays[f"gate_scores_{f}"], arrays[f"evr_{f}"] = gate_scores[f], evr[f]
        log(
            f"MFA {f}: {int(np.count_nonzero(~dropped[f]))} blocks; the first {N_MFA} components explain "
            f"{arrays[f'mfa_evr_{f}'].sum():.3f} of the variance"
        )
    arrays["zero_spread_nonzero"], arrays["gate_gram"] = zero_spread_nonzero, gate_gram
    out = Path(out)
    tmp = out.with_name(out.stem + ".tmp.npz")  # matched by *.npz in .gitignore
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    seconds = time.perf_counter() - start
    log(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB) in {seconds:.0f} s; peak RSS {_peak_rss_gb():.2f} GB")
    return arrays


def main() -> None:
    c.check_topocyte()
    pilot = pd.read_csv(c.RESULTS / "pilot.csv")
    run(pilot, c.RESULTS / "bars", c.RESULTS / "features.npz", log=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
