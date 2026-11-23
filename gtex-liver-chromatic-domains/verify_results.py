#!/usr/bin/env python
"""Independent checks of every saved result of the chromatic-domains pilot; prints ``ALL CHECKS PASSED``.

Run from this folder with the study venv and ``PYTHONPATH=<topocyte checkout at 18e3aa5>``
(``common.check_topocyte``). It reads the cohort read only (h5py mode "r") and ``results/``,
writes ``results/verify.json`` (each check with its sample, tolerance and measured value, and
the run's time and peak RSS) and exits 1 if any check fails. Nothing else is written.

Each check recomputes a saved result by another route than the one that made it:

* **arrays and provenance:** every array of ``features.npz``, ``domains.npz`` and the 30 bars
  files finite; the bars files' owners, offsets and sort order; the centre draws and fit samples
  redrawn from their keys; ``L1`` from the class counts; ``P`` from exact rational expectations;
* **six-packs** on 200 sampled real balls (cells found by brute force): the saved labelling-0
  domain and image bars and the codomain against chalc 15.0.1, the port (faces to dimension 3 and
  to dimension 2, every module) against chalc, the three Betti identities at every endpoint, and
  the codomain against topocyte's alpha persistent homology. Bars shorter than 1e-12 of the
  ball's scale (its largest finite value) are dropped before counts are compared, since exact
  ties can pair near-zero bars differently in chalc's exact arithmetic (progress.md, Task 2);
  the bottleneck distance must stay within 1e-9 of the scale;
* **bars bit for bit:** one ball per slide recomputed with ``chromatic.ball_sixpacks``, all 20
  labellings and 16 blocks, the codomain and the pair counts;
* **pair counts:** on the first 20 sampled balls, the Delaunay edges by scipy (Qhull) against the
  saved counts, and the expectation against 10,000 brute-force shuffles (within 5 standard errors);
* **inner products:** per block, ``curves.inner_matrix`` against topocyte's
  ``betti_inner_product_matrix`` on the 200 unit-weight curves of the 10 sampled centres with the
  most bars (Gram and cross matrices), and for their scaled residual curves, composed from those
  unit-weight inner products;
* **spreads:** every centre's s² (labellings 1..19), control s² (2..19) and zero test, by an
  integer-count sweep over the union of the references' endpoints, giving s0 per block and fit and
  the zero-spread counts; on the first 20 sampled balls, s² by brute force from topocyte's 19 × 19
  Gram matrix against the study's ``curves.spread2`` and the sweep;
* **decompositions:** each half's fit of each block refitted from the bars, with the Gram matrix of
  ``curves.inner_matrix`` and topocyte's ``fpca_from_gram``, against the saved scores, fractions of
  variance and kept counts; the all-slides fit only for ``ALL_FIT_BLOCKS`` (a refit of all 16
  takes about 40 min more). The 200 sampled centres are scored by the kernel route
  (c - mean Gram row)·v/√ρ, observed and control. The MFA of each fit by a numpy
  eigendecomposition with the sibling sign rule, against every saved level-2 row, and the control's
  level-2 rows of the sampled centres in the halves' bases (the only control rows the rule reads);
* **clustering and verdict:** level 1, the control and level 2 recomputed from ``features.npz``
  with scikit-learn's ``KMeans`` and the saved seeds, by this file's own gap statistic, stopping
  rule and prediction strength (k exactly; gap, s and prediction strength within 1e-12, since
  k-means inertia moves by an ulp between calls); the verdict from the recomputed k's. The guard
  did not run: its balanced accuracies reduce to asserting it is absent.

Samples. Every draw uses ``np.random.SeedSequence`` with a 4-entry key whose last entry is
non-zero (``common``'s rule). The verifier's own keys carry a mark in the second entry, above
every slide index, so none meets a study key ``[SEED, slide_index, centre_index, stream]``.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from fractions import Fraction
from math import comb

import gudhi
import numba as nb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

import chromatic
import common as c
import curves

N_SLIDES = 30
N_ROWS = N_SLIDES * c.N_CENTRES
N_LAB = c.R + 1
N_BLOCKS = len(c.BLOCKS)
N_COMPONENTS = 9
N_MFA = 20
FITS = ("A", "B", "all")
FIT_STREAM = 0xF17  # the study's fit-sample stream (features.py)
ALL_FIT_BLOCKS = (1, 11)  # the most distinct endpoints, and the largest gate-2 error, among the all-slides fits
MARK_BALLS, MARK_ONE_PER_SLIDE, MARK_SHUFFLES = 0xBA11, 0xB17, 0x5AFF  # [SEED, mark, index, 1]
N_BALLS, N_FIRST, N_INNER = 200, 20, 10
N_SHUFFLES = 10_000
DROP_RTOL, BOTTLENECK_RTOL = 1e-12, 1e-9
PAIR_INDEX = np.full((4, 4), -1, np.int64)
for _k, (_a, _b) in enumerate(c.PAIRS):
    PAIR_INDEX[_a, _b] = PAIR_INDEX[_b, _a] = _k
assert (
    len({MARK_BALLS, MARK_ONE_PER_SLIDE, MARK_SHUFFLES}) == 3
    and min(MARK_BALLS, MARK_ONE_PER_SLIDE, MARK_SHUFFLES) >= N_SLIDES
)

RECORDS: list[dict] = []


def check(name: str, passed: bool, measured: float, tolerance: str, sample: str) -> None:
    """Print and record one check."""
    passed = bool(passed)
    print(f"{'ok  ' if passed else 'FAIL'} {name}: {measured:.3g} (tolerance {tolerance}; {sample})", flush=True)
    RECORDS.append(
        {"check": name, "passed": passed, "measured": float(measured), "tolerance": tolerance, "sample": sample}
    )


def rng_of(key: list[int]) -> np.random.Generator:
    """A generator of a 4-entry key whose last entry is non-zero."""
    key = [int(k) for k in key]
    if len(key) != 4 or key[-1] == 0:
        raise ValueError(f"a draw needs a 4-entry key with a non-zero last entry, got {key}")
    return np.random.default_rng(np.random.SeedSequence(key))


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (peak if sys.platform == "darwin" else 1024 * peak) / 1e9


def load_bars(slide_id: str, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(c.RESULTS / "bars" / f"{slide_id}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in (keys or z.files)}


def owner_bars(z: dict, centre: int, labelling: int, block: int) -> np.ndarray:
    o = (centre * N_LAB + labelling) * N_BLOCKS + block
    return z["bars"][z["offsets"][o] : z["offsets"][o + 1]]


def finite_sorted(diagram: np.ndarray) -> np.ndarray:
    d = diagram[np.isfinite(diagram[:, 1])]
    return d[np.lexsort((d[:, 1], d[:, 0]))]


def fit_rows(F: dict, f: str) -> np.ndarray:
    """The rows of fitting set f's fit sample, ascending."""
    return np.flatnonzero(F["fit"] & ((F["half"] == f) if f != "all" else True))


# ---------------------------------------------------------------- arrays and provenance


def check_arrays_and_provenance(pilot: pd.DataFrame, F: dict, D: dict) -> None:
    for name, arrays in (("features.npz", F), ("domains.npz", D)):
        bad = [key for key, a in arrays.items() if a.dtype.kind in "fc" and not np.all(np.isfinite(a))]
        check(f"{name}: every array finite", not bad, len(bad), "0 arrays", f"{len(arrays)} arrays")
    by_condition = pilot.groupby("condition")["half"].agg(lambda h: len(h) == 5 and h.nunique() == 2)
    check(
        "pilot: 30 slides, 15 per half, 5 per condition in both halves",
        len(pilot) == N_SLIDES
        and (pilot["half"] == "A").sum() == 15
        and by_condition.all()
        and len(by_condition) == 6
        and np.array_equal(pilot["slide_index"], np.arange(N_SLIDES)),
        0,
        "exact",
        "results/pilot.csv",
    )
    bad_files, finite_bad, n_bars, n_cod = [], 0, 0, 0
    parts: dict[str, list] = {
        key: [] for key in ("centre_rows", "centre_xy", "fit", "class_counts", "pair_counts", "n_edges")
    }
    n_owners = c.N_CENTRES * N_LAB * N_BLOCKS
    for s, slide in pilot.iterrows():
        meta = json.loads((c.RESULTS / "bars" / f"{slide['slide_id']}.json").read_text())
        z = load_bars(slide["slide_id"])
        finite_bad += sum(not np.all(np.isfinite(a)) for a in z.values() if a.dtype.kind in "fc")
        bars, owner, offsets = z["bars"], z["owner"].astype(np.int64), z["offsets"]
        cod, cod_owner = z["codomain"], z["codomain_owner"]

        def lexicographic(x: np.ndarray, o: np.ndarray) -> bool:
            later = (x[1:, 0] > x[:-1, 0]) | ((x[1:, 0] == x[:-1, 0]) & (x[1:, 1] >= x[:-1, 1]))
            return bool(np.all(np.diff(o) >= 0) and np.all(later | (o[1:] != o[:-1])))

        drawn = rng_of([c.SEED, s, 0, c.R + 1]).choice(meta["slide_cells"], size=c.N_CENTRES, replace=False)
        member = np.zeros(c.N_CENTRES, bool)
        member[rng_of([c.SEED, s, 0, FIT_STREAM]).choice(c.N_CENTRES, size=c.FIT_PER_SLIDE, replace=False)] = True
        ok = (
            offsets.shape == (n_owners + 1,)
            and offsets[0] == 0
            and offsets[-1] == len(bars) == len(owner)
            and np.array_equal(np.bincount(owner, minlength=n_owners), np.diff(offsets))
            and lexicographic(bars, owner)
            and np.all(bars[:, 0] >= 0)
            and np.all(bars[:, 1] > bars[:, 0])
            and lexicographic(cod, cod_owner)
            and cod_owner.min() >= 0
            and cod_owner.max() < 2 * c.N_CENTRES
            and np.all(cod[:, 1] > cod[:, 0])
            and np.array_equal(z["class_counts"].sum(axis=1), z["n_cells"])
            and np.array_equal(z["pair_counts"].sum(axis=1), z["n_edges"])
            and np.array_equal(z["centre_rows"], drawn)
            and meta["topocyte_commit"] == c.TOPOCYTE_COMMIT
            and meta["slide_index"] == s
            and meta["half"] == slide["half"]
            and meta["condition"] == slide["condition"]
            and meta["bars"] == len(bars)
            and meta["codomain_bars"] == len(cod)
            and meta["delta_px"] == c.DELTA_PX
        )
        bad_files += [] if ok else [slide["slide_id"]]
        n_bars, n_cod = n_bars + len(bars), n_cod + len(cod)
        for key in parts:
            parts[key].append(member if key == "fit" else z[key])
        print(f"  arrays: {slide['slide_id']}", flush=True)
    check("bars files: every array finite", finite_bad == 0, finite_bad, "0 arrays", "30 slides, every array")
    check(
        "bars files: owners, offsets, sort order, counts, centre draw and json",
        not bad_files,
        len(bad_files),
        "0 slides",
        f"30 slides, {n_bars:,} bars and {n_cod:,} codomain bars",
    )
    joined = {key: np.concatenate(value) for key, value in parts.items()}
    counts = joined["class_counts"].astype(np.float64)
    cells = counts.sum(axis=1)
    L1 = np.column_stack([np.log10(cells), counts / cells[:, None]])
    cc, ne = joined["class_counts"], joined["n_edges"]
    expected = np.empty((N_ROWS, len(c.PAIRS)))
    for i in range(N_ROWS):
        n_pairs = comb(int(cc[i].sum()), 2)
        for k, (a, b) in enumerate(c.PAIRS):
            same = comb(int(cc[i, a]), 2) if a == b else int(cc[i, a]) * int(cc[i, b])
            expected[i, k] = float(Fraction(int(ne[i]) * same, n_pairs)) if n_pairs else 0.0
    P = joined["pair_counts"] - expected
    rows_ok = (
        np.array_equal(F["slide_index"], np.repeat(np.arange(N_SLIDES), c.N_CENTRES))
        and np.array_equal(F["half"], np.repeat(pilot["half"].to_numpy(dtype=str), c.N_CENTRES))
        and np.array_equal(F["centre_row"], joined["centre_rows"])
        and np.array_equal(F["xy"], joined["centre_xy"])
        and np.array_equal(F["fit"], joined["fit"])
    )
    check(
        "features.npz: slides, halves, centre rows, positions and fit samples redrawn",
        rows_ok,
        int(not rows_ok),
        "exact",
        "45,000 rows; fit keys [SEED, slide, 0, 0xF17]",
    )
    check(
        "features.npz: L1 from the class counts",
        np.array_equal(F["L1"], L1),
        float(np.max(np.abs(F["L1"] - L1))),
        "exact",
        "45,000 rows",
    )
    check(
        "features.npz: P = pair counts - exact rational expectation",
        np.array_equal(F["P"], P),
        float(np.max(np.abs(F["P"] - P))),
        "exact",
        "45,000 rows × 10 pairs",
    )


# ---------------------------------------------------------------- balls: six-packs, bit for bit, pair counts


def sampled_rows() -> np.ndarray:
    """The 200 sampled rows of features.npz, in draw order."""
    return rng_of([c.SEED, MARK_BALLS, 0, 1]).choice(N_ROWS, size=N_BALLS, replace=False)


def chalc_sixpacks(xy: np.ndarray, cls: np.ndarray) -> dict:
    """chalc 15.0.1's six-pack of (class t -> all cells) for each class present, radii squared, keyed like the port."""
    import chalc
    from chalc.morphisms import SubChromaticInclusion

    present = np.unique(cls)
    f = chalc.chromatic.delaunay_cech(np.ascontiguousarray(xy.T), np.searchsorted(present, cls).tolist(), 0)
    out = {}
    for rank, t in enumerate(present.tolist()):
        pack = SubChromaticInclusion(f, [rank]).sixpack(1)
        for name in ("kernel", "domain", "image", "codomain", "cokernel", "relative"):
            for p in (0, 1):
                bars = np.asarray(getattr(pack, name).intervals(p), dtype=np.float64).reshape(-1, 2) ** 2
                out[name, -1 if name == "codomain" else t, p] = bars[bars[:, 1] > bars[:, 0]]
    return out


def distance(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Bottleneck distance over the scale after dropping bars shorter than DROP_RTOL·scale; inf if counts differ."""
    a, b = (x[(x[:, 1] - x[:, 0]) >= DROP_RTOL * scale] for x in (a.reshape(-1, 2), b.reshape(-1, 2)))
    if len(a) != len(b) or np.isinf(a[:, 1]).sum() != np.isinf(b[:, 1]).sum():
        return np.inf
    return 0.0 if len(a) == 0 else gudhi.bottleneck_distance(a, b) / scale


def betti(bars: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(bars[:, 0]), ts, side="right") - np.searchsorted(
        np.sort(bars[:, 1]), ts, side="right"
    )


def identity_violation(packs: dict, t: int) -> int:
    """The largest violation of dom = ker + im, cod = im + cok and rel_p = cok_p + ker_(p-1) at any endpoint or 0."""
    worst = 0
    for p in (0, 1):
        m = {name: packs[name, t, p] for name in ("kernel", "domain", "image", "cokernel", "relative")}
        m["codomain"] = packs["codomain", -1, p]
        ker_prev = packs["kernel", t, 0] if p == 1 else np.empty((0, 2))
        ends = np.concatenate([a.ravel() for a in m.values()] + [ker_prev.ravel(), [0.0]])
        ts = np.unique(ends[np.isfinite(ends)])
        B = {name: betti(a, ts) for name, a in m.items()}
        worst = max(
            worst,
            int(np.abs(B["domain"] - B["kernel"] - B["image"]).max()),
            int(np.abs(B["codomain"] - B["image"] - B["cokernel"]).max()),
            int(np.abs(B["relative"] - B["cokernel"] - betti(ker_prev, ts)).max()),
        )
    return worst


def pair_check(bxy: np.ndarray, bcls: np.ndarray, z: dict, centre: int, j: int) -> tuple[bool, float]:
    """Qhull's Delaunay edges and class-pair counts against the saved; the exact expectation against
    ``pair_expectation`` and against the mean of 10,000 shuffles. Returns (counts agree, largest |z|)."""
    tri = Delaunay(bxy).simplices
    edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]]), axis=1), axis=0)
    observed = np.bincount(PAIR_INDEX[bcls[edges[:, 0]], bcls[edges[:, 1]]], minlength=len(c.PAIRS))
    n_class, n_pairs = np.bincount(bcls, minlength=4), comb(len(bcls), 2)
    exact = np.array(
        [
            float(
                Fraction(len(edges) * (comb(int(n_class[a]), 2) if a == b else int(n_class[a] * n_class[b])), n_pairs)
            )
            for a, b in c.PAIRS
        ]
    )
    perms = rng_of([c.SEED, MARK_SHUFFLES, j, 1]).permuted(np.tile(bcls, (N_SHUFFLES, 1)), axis=1)
    index = PAIR_INDEX[perms[:, edges[:, 0]], perms[:, edges[:, 1]]] + len(c.PAIRS) * np.arange(N_SHUFFLES)[:, None]
    shuffled = np.bincount(index.ravel(), minlength=N_SHUFFLES * len(c.PAIRS)).reshape(N_SHUFFLES, -1)
    mean, se = shuffled.mean(axis=0), shuffled.std(axis=0, ddof=1) / np.sqrt(N_SHUFFLES)
    zs = np.where(se > 0, np.abs(mean - exact) / np.where(se > 0, se, 1.0), np.where(mean == exact, 0.0, np.inf))
    agree = (
        len(np.unique(bxy, axis=0)) == len(bxy)
        and len(edges) == z["n_edges"][centre]
        and np.array_equal(observed, z["pair_counts"][centre])
        and np.array_equal(chromatic.pair_expectation(n_class, len(edges)), exact)
    )
    return agree, float(zs.max())


def check_balls(pilot: pd.DataFrame, sample: np.ndarray) -> None:
    import topocyte as tc

    worst = {"saved": 0.0, "dim3": 0.0, "dim2": 0.0, "alpha": 0.0}
    n_cmp = dict.fromkeys(worst, 0)
    n_dropped = identity = ball_bad = bit_bad = pair_bad = n_bit_bars = 0
    z_worst, first = 0.0, [int(r) for r in sample[:N_FIRST]]
    chromatic.ball_sixpacks(np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 4.0]]), np.array([0, 1, 2]))  # load numba
    for s, slide in pilot.iterrows():
        rows = sorted(int(r) for r in sample if r // c.N_CENTRES == s)
        one = int(rng_of([c.SEED, MARK_ONE_PER_SLIDE, s, 1]).integers(c.N_CENTRES))
        xy, cls = c.read_slide(slide["slide_id"])
        z = load_bars(slide["slide_id"])
        balls = {}
        for centre in [r - s * c.N_CENTRES for r in rows] + [one]:
            ball = np.flatnonzero(np.sum((xy - xy[z["centre_rows"][centre]]) ** 2, axis=1) <= c.DELTA_PX**2)
            balls[centre] = (xy[ball], cls[ball])
            ball_bad += not (
                len(ball) == z["n_cells"][centre]
                and np.array_equal(np.bincount(cls[ball], minlength=4), z["class_counts"][centre])
            )
        for row in rows:
            centre = row - s * c.N_CENTRES
            bxy, bcls = balls[centre][0], balls[centre][1].astype(np.int64)
            theirs = chalc_sixpacks(bxy, bcls)
            scale = max(1.0, float(np.concatenate([b[np.isfinite(b)] for b in theirs.values()]).max()))
            dim2 = chromatic.ball_sixpacks(bxy, bcls, full=True)
            dim3 = chromatic.ball_sixpacks(bxy, bcls, full=True, _max_face_dim=3)
            present = set(np.unique(bcls).tolist())
            alpha = tc.ph.compute_persistent_homology(bxy, filtration="alpha", max_dim=1).diagrams
            comparisons = []
            for block, (t, module, p) in enumerate(c.BLOCKS):
                ref = theirs[module, t, p] if t in present else np.empty((0, 2))
                comparisons.append(("saved", owner_bars(z, centre, 0, block), ref[np.isfinite(ref[:, 1])]))
            for p in (0, 1):
                saved = z["codomain"][z["codomain_owner"] == 2 * centre + p]
                ref = theirs["codomain", -1, p]
                comparisons += [
                    ("saved", saved, ref[np.isfinite(ref[:, 1])]),
                    ("alpha", saved, alpha[p][np.isfinite(alpha[p][:, 1])]),
                ]
            for key, ref in theirs.items():
                comparisons += [("dim2", dim2[key], ref), ("dim3", dim3[key], ref)]
            for kind, ours, ref in comparisons:
                worst[kind] = max(worst[kind], distance(ours, ref, scale))
                n_cmp[kind] += 1
                n_dropped += len(ours) != len(ref)
            identity = max([identity] + [identity_violation(pack, t) for pack in (dim2, dim3) for t in present])
            if row in first:
                agree, z_max = pair_check(bxy, bcls, z, centre, first.index(row))
                pair_bad += not agree
                z_worst = max(z_worst, z_max)
        bxy, bcls = balls[one]  # one ball per slide, all labellings, bit for bit
        same = True
        for r in range(N_LAB):
            packs = chromatic.ball_sixpacks(bxy, bcls if r == 0 else rng_of([c.SEED, s, one, r]).permutation(bcls))
            for block, (t, module, p) in enumerate(c.BLOCKS):
                saved = owner_bars(z, one, r, block)
                same &= np.array_equal(finite_sorted(packs[module, t, p]), saved)
                n_bit_bars += len(saved)
            for p in (0, 1) if r == 0 else ():
                same &= np.array_equal(
                    finite_sorted(packs["codomain", -1, p]), z["codomain"][z["codomain_owner"] == 2 * one + p]
                )
        counts, n_edges = chromatic.pair_counts(bxy, bcls)
        bit_bad += not (same and np.array_equal(counts, z["pair_counts"][one]) and n_edges == z["n_edges"][one])
        print(f"  balls: {slide['slide_id']}: {len(rows)} sampled, centre {one} bit for bit", flush=True)
    check(
        "balls: cells by brute force equal the saved counts",
        ball_bad == 0,
        ball_bad,
        "0 balls",
        f"{len(sample)} sampled balls and one per slide",
    )
    for kind, label in (
        ("saved", "saved labelling-0 domain, image and codomain bars against chalc 15.0.1"),
        ("dim3", "the port, faces to dimension 3, every module, against chalc"),
        ("dim2", "the port, faces to dimension 2 (the study's), every module, against chalc"),
        ("alpha", "saved codomain against topocyte's alpha persistent homology"),
    ):
        check(
            f"six-packs: {label}",
            worst[kind] <= BOTTLENECK_RTOL,
            worst[kind],
            f"equal counts, bottleneck <= {BOTTLENECK_RTOL:g} of the ball's scale",
            f"{len(sample)} balls, {n_cmp[kind]:,} diagrams",
        )
    check(
        "six-packs: diagrams whose raw bar counts differed (before bars under 1e-12 of the scale were dropped)",
        True,
        n_dropped,
        "descriptive",
        f"{sum(n_cmp.values()):,} diagrams",
    )
    check(
        "six-packs: Betti identities (dom = ker + im, cod = im + cok, rel = cok + ker) at every endpoint",
        identity == 0,
        identity,
        "0",
        f"{len(sample)} balls, faces to dimension 2 and 3, every class present",
    )
    check(
        "bars bit for bit: one ball per slide recomputed with ball_sixpacks",
        bit_bad == 0,
        bit_bad,
        "0 balls",
        f"{len(pilot)} balls × 20 labellings × 16 blocks ({n_bit_bars:,} bars), codomain, pair counts",
    )
    check(
        "pair counts: Qhull Delaunay edges and class-pair counts equal the saved; expectation exact",
        pair_bad == 0,
        pair_bad,
        "0 balls",
        f"first {N_FIRST} sampled balls",
    )
    check(
        "pair counts: expectation against 10,000 shuffles, largest |z|",
        z_worst <= 5,
        z_worst,
        "5 SE",
        f"first {N_FIRST} sampled balls × 10 pairs",
    )


# ---------------------------------------------------------------- per block: spreads, inner products, decompositions


@nb.njit(cache=True)
def _sweep(centre, lab, pos, step, n_centres, R):
    """Events sorted by (centre, position): per centre, R²·s² of labellings 1..R, (R-1)²·s² of 2..R, and whether
    R·lab0 - Σ lab_r is non-zero. Each labelling's count of live bars is an integer on every interval between
    consecutive endpoints, so R·Σβ_r² - (Σβ_r)² is exact there, and 0 exactly where the references agree."""
    s2 = np.zeros(n_centres)
    s2c = np.zeros(n_centres)
    nz = np.zeros(n_centres, np.bool_)
    beta = np.zeros(R + 1, np.int64)
    n, e = pos.size, 0
    while e < n:
        i = centre[e]
        beta[:] = 0
        while e < n and centre[e] == i:
            x = pos[e]
            while e < n and centre[e] == i and pos[e] == x:
                beta[lab[e]] += step[e]
                e += 1
            if e < n and centre[e] == i:
                width = pos[e] - x
                s1, sq = 0, 0
                for r in range(1, R + 1):
                    s1 += beta[r]
                    sq += beta[r] * beta[r]
                s1c, sqc = s1 - beta[1], sq - beta[1] * beta[1]
                s2[i] += width * (R * sq - s1 * s1)
                s2c[i] += width * ((R - 1) * sqc - s1c * s1c)
                if R * beta[0] != s1:
                    nz[i] = True
    return s2, s2c, nz


def spreads(centre: np.ndarray, lab: np.ndarray, b: np.ndarray, d: np.ndarray, n_centres: int):
    """s² (1..R), control s² (2..R) and the zero test of every centre of one block of one slide (``_sweep``)."""
    pos = np.concatenate([b, d])
    cen, lb = np.concatenate([centre, centre]), np.concatenate([lab, lab])
    step = np.concatenate([np.ones(b.size, np.int64), -np.ones(b.size, np.int64)])
    order = np.lexsort((pos, cen))
    s2, s2c, nz = _sweep(cen[order], lb[order], pos[order], step[order], n_centres, c.R)
    return s2 / c.R**2, s2c / (c.R - 1) ** 2, nz


def subset(part: tuple, rows: np.ndarray) -> tuple:
    """The bars of ``part`` (row, labelling, birth, death; grouped by ascending row) whose row is in ``rows``."""
    m = np.isin(part[0], rows)
    return tuple(x[m] for x in part)


def residuals(part: tuple, rows: np.ndarray, a: np.ndarray, first: int) -> curves.CurveSet:
    """Residual curves of the global ``rows`` (ascending): labelling ``first`` with weight 1/a, each later labelling
    with -1/((R - first)·a)."""
    row, lab, b, d = subset(part, rows)
    take = lab >= first
    row, lab, b, d = row[take], lab[take], b[take], d[take]
    where = np.searchsorted(rows, row)
    w = np.where(lab == first, 1.0, -1.0 / (c.R - first)) / a[where]
    return curves.CurveSet(b, d, w, np.concatenate([[0], np.cumsum(np.bincount(where, minlength=len(rows)))]))


def unit_curves(part: tuple, rows: np.ndarray) -> tuple[curves.CurveSet, list[np.ndarray]]:
    """The unit-weight curves of every labelling of ``rows``, labelling-major (r·len(rows) + i), and their diagrams."""
    row, lab, b, d = subset(part, rows)
    diagrams = [
        np.column_stack([b[(row == i) & (lab == r)], d[(row == i) & (lab == r)]]) for r in range(N_LAB) for i in rows
    ]
    bars = np.concatenate(diagrams)
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in diagrams])])
    return curves.CurveSet(bars[:, 0], bars[:, 1], np.ones(len(bars)), offsets), diagrams


def curve_range(cs: curves.CurveSet, lo: int, hi: int) -> curves.CurveSet:
    a, b = cs.offsets[lo], cs.offsets[hi]
    return curves.CurveSet(cs.b[a:b], cs.d[a:b], cs.w[a:b], cs.offsets[lo : hi + 1] - a)


def relative(x: np.ndarray, ref: np.ndarray) -> float:
    """The largest |x - ref| / |ref| over the entries where ref is not 0."""
    nonzero = ref != 0
    return float(np.max(np.abs(x - ref)[nonzero] / np.abs(ref[nonzero]), initial=0.0))


def check_block(block: int, pilot: pd.DataFrame, F: dict, sample: np.ndarray, out: dict) -> None:
    """One block: a pass over the slides, then its spreads, inner products and decompositions (module docstring)."""
    import topocyte as tc

    t0 = time.perf_counter()
    t, module, p = c.BLOCKS[block]
    s2, s2c, nz = np.empty(N_ROWS), np.empty(N_ROWS), np.empty(N_ROWS, bool)
    keep_rows = np.union1d(np.flatnonzero(F["fit"]), sample)
    pieces = []
    for s, slide in pilot.iterrows():
        z = load_bars(slide["slide_id"], ("bars", "owner"))
        owner = z["owner"].astype(np.int64)
        sel = np.flatnonzero(owner % N_BLOCKS == block)
        rest = owner[sel] // N_BLOCKS
        centre, lab, b, d = rest // N_LAB, rest % N_LAB, z["bars"][sel, 0], z["bars"][sel, 1]
        del z, owner
        rows = slice(s * c.N_CENTRES, (s + 1) * c.N_CENTRES)
        s2[rows], s2c[rows], nz[rows] = spreads(centre, lab, b, d, c.N_CENTRES)
        m = np.isin(centre + s * c.N_CENTRES, keep_rows)
        pieces.append((centre[m] + s * c.N_CENTRES, lab[m], b[m], d[m]))
    part = tuple(np.concatenate(x) for x in zip(*pieces, strict=True))
    del pieces
    t_pass = time.perf_counter() - t0
    sample_sorted = np.sort(sample)
    # s0 and the zero-spread count.
    s0sq = {}
    for f in FITS:
        rows_f = fit_rows(F, f)
        s0sq[f] = float(np.median(s2[rows_f][s2[rows_f] > 0]))
        out["s0"][f][block] = abs(np.sqrt(s0sq[f]) - F[f"s0_{f}"][block]) / F[f"s0_{f}"][block]
    out["zero"][block] = int(np.count_nonzero((s2 == 0) & nz)) - int(F["zero_spread_nonzero"][block])
    # Spreads by brute force on the first sampled balls: topocyte's Gram of the references.
    first = np.sort(sample[:N_FIRST])
    unit, diagrams = unit_curves(part, first)
    T = tc.ph.betti_inner_product_matrix([tc.ph.BettiCurve(x, p) for x in diagrams], dim=p)
    n = len(first)
    for start, mine in ((1, s2[first]), (2, s2c[first])):
        study = curves.spread2([curve_range(unit, r * n, (r + 1) * n) for r in range(start, N_LAB)])
        for i in range(n):
            idx = [r * n + i for r in range(start, N_LAB)]
            G = T[np.ix_(idx, idx)]
            brute = np.trace(G) / len(idx) - G.sum() / len(idx) ** 2
            norm = max(np.trace(G) / len(idx), np.finfo(float).tiny)
            out["spread"] = max(out["spread"], abs(study[i] - brute) / norm, abs(mine[i] - brute) / norm)
            agree = all(np.array_equal(finite_sorted(diagrams[k]), finite_sorted(diagrams[idx[0]])) for k in idx)
            out["spread_zero_bad"] += not ((study[i] == 0) == (mine[i] == 0) == agree)
    # Inner products on the sampled centres with the most bars in this block.
    counts = np.bincount(np.searchsorted(sample_sorted, subset(part, sample_sorted)[0]), minlength=len(sample))
    top = np.sort(sample_sorted[np.argsort(-counts, kind="stable")[:N_INNER]])
    unit, diagrams = unit_curves(part, top)
    T = tc.ph.betti_inner_product_matrix([tc.ph.BettiCurve(x, p) for x in diagrams], dim=p)
    G = curves.inner_matrix(unit)
    h = len(unit) // 2
    cross = curves.inner_matrix(curve_range(unit, 0, h), curve_range(unit, h, len(unit)))
    out["inner_unit"] = max(out["inner_unit"], relative(G, T))
    out["inner_cross"] = max(out["inner_cross"], relative(cross, T[:h, h:]))
    out["inner_zero_bad"] += int(np.count_nonzero(G[T == 0])) + int(np.count_nonzero(cross[T[:h, h:] == 0]))
    a = np.sqrt(s2[top] + s0sq["all"])
    W = np.zeros((len(unit), N_INNER))
    for r in range(N_LAB):
        W[r * N_INNER + np.arange(N_INNER), np.arange(N_INNER)] = (1.0 if r == 0 else -1.0 / c.R) / a
    composed, bound = W.T @ T @ W, np.abs(W).T @ np.abs(T) @ np.abs(W)
    ours = curves.inner_matrix(residuals(part, top, a, 0))
    out["inner_composed"] = max(
        out["inner_composed"], float(np.max(np.abs(ours - composed) / np.where(bound > 0, bound, 1.0)))
    )
    out["inner_zero_bad"] += int(np.count_nonzero(ours[bound == 0]))
    # Decompositions: the halves' fits, and the all-slides fit of ALL_FIT_BLOCKS.
    for f in FITS if block in ALL_FIT_BLOCKS else ("A", "B"):
        rows_f = fit_rows(F, f)
        fit_set = residuals(part, rows_f, np.sqrt(s2[rows_f] + s0sq[f]), 0)
        held = [curves.inner_matrix(fit_set)]
        kbar = held[0].mean(axis=0)
        res = tc.ph.fpca_from_gram(held.pop())  # handed over without a name: its symmetrized copy replaces it
        kept = min(N_COMPONENTS, res.gram_eigenvalues.size)
        V, rho = res.eigenvectors[:, :N_COMPONENTS].copy(), res.gram_eigenvalues[:N_COMPONENTS].copy()
        scores, evr = res.scores[:, :N_COMPONENTS].copy(), res.explained_variance_ratio[:N_COMPONENTS].copy()
        del res
        saved = F[f"blk_obs_{f}"][rows_f, block]
        top_score = np.max(np.abs(saved))
        out["fit"][f][block] = float(np.max(np.abs(scores - saved)) / top_score)
        out["evr"][f][block] = float(np.max(np.abs(evr - F[f"evr_{f}"][block])))
        out["kept"][f][block] = float(kept == N_COMPONENTS == F[f"kept_{f}"][block] and not F[f"dropped_{f}"][block])
        for first_lab, spread_used in ((0, s2), (1, s2c)):
            a = np.sqrt(spread_used[sample_sorted] + s0sq[f])
            kernel = (curves.inner_matrix(residuals(part, sample_sorted, a, first_lab), fit_set) - kbar) @ (
                V / np.sqrt(rho)
            )
            if first_lab == 0:
                out["kernel"][f][block] = float(
                    np.max(np.abs(kernel - F[f"blk_obs_{f}"][sample_sorted, block])) / top_score
                )
            else:
                out["ctl_scores"][f][:, block] = kernel
        del fit_set, kbar
    print(
        f"  block {block} ({c.CLASSES[t]} {module} H{p}): pass {t_pass:.0f} s, all {time.perf_counter() - t0:.0f} s; "
        f"s0 {max(out['s0'][f][block] for f in FITS):.1e}, fit {max(out['fit'][f][block] for f in FITS):.1e}, "
        f"kernel {max(out['kernel'][f][block] for f in FITS):.1e}; peak RSS {peak_rss_gb():.2f} GB",
        flush=True,
    )


def mfa_basis(blk: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The MFA of the blocks' scores on ``rows`` by eigendecomposition: weights, column means, the first N_MFA
    loadings (signs by the sibling rule: each component's largest-magnitude score positive) and every fraction."""
    weights = 1.0 / np.sqrt(blk[rows, :, 0].var(axis=0))
    Z = (blk[rows] * weights[None, :, None]).reshape(len(rows), -1)
    mean = Z.mean(axis=0)
    Zc = Z - mean
    lam, V = np.linalg.eigh(Zc.T @ Zc)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order[:N_MFA]]
    U = Zc @ V
    return weights, mean, V * np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(N_MFA)]), lam / lam.sum()


def check_blocks(pilot: pd.DataFrame, F: dict, sample: np.ndarray, blocks: tuple[int, ...] | None = None) -> None:
    blocks = tuple(range(N_BLOCKS)) if blocks is None else blocks
    out = {key: {f: np.zeros(N_BLOCKS) for f in FITS} for key in ("s0", "fit", "evr", "kernel")}
    out |= {
        "kept": {f: np.ones(N_BLOCKS) for f in FITS},
        "zero": np.zeros(N_BLOCKS, np.int64),
        "spread": 0.0,
        "spread_zero_bad": 0,
        "inner_unit": 0.0,
        "inner_cross": 0.0,
        "inner_composed": 0.0,
        "inner_zero_bad": 0,
        "ctl_scores": {f: np.zeros((len(sample), N_BLOCKS, N_COMPONENTS)) for f in FITS},
    }
    for block in blocks:
        check_block(block, pilot, F, sample, out)
    nb_ = len(blocks)
    worst_s0 = max(out["s0"][f].max() for f in FITS)
    check(
        "spreads: s0 per block and fit from every fit centre's s² (integer-count sweep)",
        worst_s0 <= 1e-12,
        worst_s0,
        "1e-12 relative",
        f"{nb_} blocks × 3 fits, 12,000 fit centres",
    )
    check(
        "spreads: zero-spread non-zero residuals counted over every centre",
        not np.any(out["zero"]),
        int(np.abs(out["zero"]).sum()),
        "exact",
        f"{nb_} blocks × 45,000 centres",
    )
    check(
        "spreads: s² by brute force (topocyte's 19 × 19 Gram) against curves.spread2 and the sweep",
        out["spread"] <= 1e-10,
        out["spread"],
        "1e-10 of the mean squared reference norm",
        f"first {N_FIRST} sampled balls × {nb_} blocks, observed and control",
    )
    check(
        "spreads: s² = 0 exactly when all references are the same diagram",
        out["spread_zero_bad"] == 0,
        out["spread_zero_bad"],
        "0",
        f"first {N_FIRST} sampled balls × {nb_} blocks, observed and control",
    )
    check(
        "inner products: unit weights against betti_inner_product_matrix",
        out["inner_unit"] <= 1e-10,
        out["inner_unit"],
        "1e-10 relative, entry by entry",
        f"{N_INNER} sampled centres × 20 labellings × {nb_} blocks",
    )
    check(
        "inner products: cross matrices against betti_inner_product_matrix",
        out["inner_cross"] <= 1e-10,
        out["inner_cross"],
        "1e-10 relative, entry by entry",
        f"the same curves in two sets, {nb_} blocks",
    )
    check(
        "inner products: scaled residual curves against topocyte's unit-weight inner products, composed",
        out["inner_composed"] <= 1e-10,
        out["inner_composed"],
        "1e-10 of the composition's absolute terms",
        f"{N_INNER} sampled centres × {nb_} blocks",
    )
    check(
        "inner products: exact zeros where topocyte has them",
        out["inner_zero_bad"] == 0,
        out["inner_zero_bad"],
        "0 entries",
        f"{nb_} blocks",
    )
    for f in FITS:
        refit = [b for b in blocks if f != "all" or b in ALL_FIT_BLOCKS]
        if not refit:
            continue
        text = f"{len(refit)} blocks" + (
            f" ({', '.join(map(str, refit))}; the others not refitted)" if f == "all" else ""
        )
        check(
            f"fPCA {f}: fit-sample scores against fpca_from_gram",
            max(out["fit"][f][refit]) <= 1e-9,
            max(out["fit"][f][refit]),
            "1e-9 of the largest score",
            text,
        )
        check(
            f"fPCA {f}: fractions of variance",
            max(out["evr"][f][refit]) <= 1e-9,
            max(out["evr"][f][refit]),
            "1e-9",
            text,
        )
        check(
            f"fPCA {f}: 9 components kept, no block dropped",
            all(out["kept"][f][refit] == 1),
            int(np.sum(out["kept"][f][refit] != 1)),
            "0 blocks",
            text,
        )
        check(
            f"fPCA {f}: sampled centres' scores by the kernel route",
            max(out["kernel"][f][refit]) <= 1e-9,
            max(out["kernel"][f][refit]),
            "1e-9 of the largest score",
            f"{len(sample)} sampled centres, {text}",
        )
    sample_sorted = np.sort(sample)
    for f in FITS:
        weights, mean, V, evr = mfa_basis(F[f"blk_obs_{f}"], fit_rows(F, f))
        L2 = ((F[f"blk_obs_{f}"] * weights[None, :, None]).reshape(N_ROWS, -1) - mean) @ V
        scale = np.max(np.abs(F[f"L2_obs_{f}"]))
        err = float(np.max(np.abs(L2 - F[f"L2_obs_{f}"])) / scale)
        check(
            f"MFA {f}: every level-2 row by eigendecomposition",
            err <= 1e-9,
            err,
            "1e-9 of the largest score",
            "45,000 rows × 20 components",
        )
        err = float(np.max(np.abs(evr[:N_MFA] - F[f"mfa_evr_{f}"])))
        check(f"MFA {f}: fractions of variance", err <= 1e-9, err, "1e-9", "20 components")
        if f != "all" and nb_ == N_BLOCKS:
            ctl = ((out["ctl_scores"][f] * weights[None, :, None]).reshape(len(sample), -1) - mean) @ V
            err = float(np.max(np.abs(ctl - F[f"L2_ctl_{f}"][sample_sorted])) / scale)
            check(
                f"MFA {f}: control level-2 rows of the sampled centres (kernel route, then the MFA)",
                err <= 1e-9,
                err,
                "1e-9 of the largest observed score",
                f"{len(sample)} sampled centres × 16 blocks",
            )


# ---------------------------------------------------------------- clustering and verdict


def zscore(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (X - ref.mean(axis=0)) / ref.std(axis=0)


def kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)


def gap_statistic(X: np.ndarray, k_max: int, B: int = 20) -> tuple[np.ndarray, np.ndarray, dict]:
    """Gap(k) and s_k for k = 1..k_max against B Gaussian draws with X's mean and covariance (one default_rng(0),
    in order), with W_1 the total sum of squares; and the k-means fits of X for k >= 2."""
    rng = np.random.default_rng(0)
    mean, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    fits = {k: kmeans(X, k) for k in range(2, k_max + 1)}

    def log_w(Y: np.ndarray, models: dict | None = None) -> np.ndarray:
        centred = Y - Y.mean(axis=0)
        inertias = [(models[k] if models else kmeans(Y, k)).inertia_ for k in range(2, k_max + 1)]
        return np.log([np.sum(centred * centred), *inertias])

    observed = log_w(X, fits)
    refs = np.array([log_w(rng.multivariate_normal(mean, cov, size=len(X))) for _ in range(B)])
    return refs.mean(axis=0) - observed, refs.std(axis=0) * np.sqrt(1 + 1 / B), fits


def stop(gap: np.ndarray, s: np.ndarray) -> int:
    """Where the gap stops rising (design step 10): the smallest k >= 1 with Gap(k) >= Gap(k+1) - s_(k+1), else
    k_max; a k >= 2 must gain at least 0.03 over Gap(1), else 1."""
    k_hat = next((k for k in range(1, len(gap)) if gap[k - 1] >= gap[k] - s[k]), len(gap))
    return k_hat if k_hat == 1 or gap[k_hat - 1] - gap[0] >= 0.03 else 1


def strength(train_model: KMeans, test: np.ndarray, k: int) -> float:
    """Prediction strength: the smallest share, over the test half's own k-means clusters, of a cluster's pairs that
    the training centroids put together (a cluster of one counts 0)."""
    own, assigned = kmeans(test, k).labels_, train_model.predict(test)
    shares = []
    for label in range(k):
        n_ij = np.bincount(assigned[own == label], minlength=k)
        n = int(n_ij.sum())
        shares.append(float(np.sum(n_ij * (n_ij - 1)) / (n * (n - 1))) if n > 1 else 0.0)
    return min(shares)


def recompute_rule(halves: dict[str, np.ndarray], tests: dict[str, np.ndarray], k_max: int) -> dict:
    """The rule for k: ``halves[h]`` is half h's rows for its gap statistic and as the training half of direction
    h -> other; ``tests[h]`` the other half's rows in half h's coordinates."""
    gaps = {h: gap_statistic(X, k_max) for h, X in halves.items()}
    k_A, k_B = stop(*gaps["A"][:2]), stop(*gaps["B"][:2])
    curve = {
        f"{h}->{o}": [strength(gaps[h][2][k], tests[h], k) for k in range(2, k_max + 1)]
        for h, o in (("A", "B"), ("B", "A"))
    }
    k = min(k_A, k_B)
    ps = [] if k == 1 else [curve["A->B"][k - 2], curve["B->A"][k - 2]]
    return {
        "k": 1 if any(x < 0.8 for x in ps) else k,
        "k_A": k_A,
        "k_B": k_B,
        "ps": ps,
        "curves": curve,
        "gap": {h: gaps[h][:2] for h in "AB"},
    }


def compare_rule(label: str, mine: dict, saved: dict) -> None:
    exact = mine["k"] == saved["k"] and mine["k_A"] == saved["k_A"] and mine["k_B"] == saved["k_B"]
    check(
        f"{label}: k, k_A and k_B recomputed",
        exact,
        abs(mine["k"] - saved["k"]),
        "exact",
        f"k {mine['k']}, k_A {mine['k_A']}, k_B {mine['k_B']}",
    )
    err = max(
        float(np.max(np.abs(mine["gap"][h][j] - np.array(saved[f"gap_{h}"][key]))))
        for h in "AB"
        for j, key in enumerate(("gap", "s"))
    )
    check(f"{label}: both halves' gap and s curves", err <= 1e-12, err, "1e-12", "k = 1..k_max")
    errors = [float(np.max(np.abs(np.array(mine["curves"][d]) - saved["ps_curve"][d]))) for d in ("A->B", "B->A")]
    errors += [abs(x - y) for x, y in zip(mine["ps"], saved["ps"], strict=False)]
    err = max(errors) if len(mine["ps"]) == len(saved["ps"]) else np.inf
    check(
        f"{label}: prediction-strength curves and the rule's values",
        err <= 1e-12,
        err,
        "1e-12",
        "k = 2..k_max, both directions",
    )


def check_clustering(F: dict, D: dict, summary: dict) -> None:
    half, L1 = F["half"], F["L1"]
    A, B = half == "A", half == "B"
    level1 = recompute_rule(
        {"A": zscore(L1[A], L1[A]), "B": zscore(L1[B], L1[B])},
        {"A": zscore(L1[B], L1[A]), "B": zscore(L1[A], L1[B])},
        10,
    )
    compare_rule("level 1", level1, summary["level1"])
    saved_z = summary["level1"]["zscore"]
    err = max(
        float(np.max(np.abs(X.mean(axis=0) - saved_z[f]["mean"])) + np.max(np.abs(X.std(axis=0) - saved_z[f]["sd"])))
        for f, X in (("A", L1[A]), ("B", L1[B]), ("all", L1))
    )
    check("level 1: z-scoring of each fitting set", err <= 1e-12, err, "1e-12", "means and SDs, ddof 0")
    strata = level1["k"]
    compartment = np.zeros(N_ROWS, np.int64) if strata == 1 else kmeans(zscore(L1, L1), strata).labels_
    check(
        "level 1: composition strata labels",
        np.array_equal(compartment, D["compartment"]),
        int(np.count_nonzero(compartment != D["compartment"])),
        "exact",
        f"45,000 rows, {strata} stratum or strata",
    )
    control_k, level2_k = [], []
    for comp in range(strata):
        Ac, Bc = (compartment == comp) & A, (compartment == comp) & B
        record = summary["compartments"][comp]
        if min(Ac.sum(), Bc.sum()) < 500:
            control_k.append(1)
            level2_k.append(1)
            continue
        for key, label, ks in (("ctl", "control", control_k), ("obs", "level2", level2_k)):
            if key == "obs" and control_k[-1] > 1:
                break
            X = {h: F[f"L2_{key}_{h}"] for h in "AB"}
            mine = recompute_rule({"A": X["A"][Ac], "B": X["B"][Bc]}, {"A": X["A"][Bc], "B": X["B"][Ac]}, 6)
            compare_rule(f"stratum {comp}, {'control' if key == 'ctl' else 'level 2'}", mine, record[label])
            ks.append(mine["k"])
    control_passed = all(k == 1 for k in control_k)
    if not control_passed:
        verdict = ("not run: control failed", None)
    elif all(k == 1 for k in level2_k):
        verdict = ("does not add", "no compartment with subdomains")
    else:
        verdict = ("the guard ran", None)  # not this run's outcome: this verifier does not recompute the guard
    same = control_k == summary["control_k"] == D["control_k"].tolist() and control_passed == summary["control_passed"]
    check("control: k per stratum and passed", same, int(not same), "exact", f"control k {control_k}")
    saved = summary["verdict"]
    same = verdict == (saved["verdict"], saved["reason"])
    check(
        "verdict recomputed from the recomputed k's",
        same,
        int(not same),
        "exact",
        f"{saved['verdict']} ({saved['reason']})",
    )
    absent = (
        np.all(D["subdomain"] == -1)
        and not any(key.startswith("boot_") for key in D)
        and all(k == 1 for k in level2_k)
        and all(
            r["guard"] is None and r["contiguity"] is None and r["level2"]["ari"] is None
            for r in summary["compartments"]
        )
    )
    check(
        "guard: absent, as the control passed and every stratum has k = 1 (no balanced accuracies to recompute)",
        absent,
        int(not absent),
        "exact",
        "domains.npz and domains.json",
    )


# ---------------------------------------------------------------- main


def main() -> None:
    started = time.perf_counter()
    c.check_topocyte()
    pilot = pd.read_csv(c.RESULTS / "pilot.csv")
    with np.load(c.RESULTS / "features.npz", allow_pickle=False) as z:
        F = {key: z[key] for key in z.files}
    with np.load(c.RESULTS / "domains.npz", allow_pickle=False) as z:
        D = {key: z[key] for key in z.files}
    summary = json.loads((c.RESULTS / "domains.json").read_text())
    sample = sampled_rows()
    seconds = {}
    for name, run in (
        ("arrays", lambda: check_arrays_and_provenance(pilot, F, D)),
        ("clustering", lambda: check_clustering(F, D, summary)),
        ("balls", lambda: check_balls(pilot, sample)),
        ("blocks", lambda: check_blocks(pilot, F, sample)),
    ):
        t = time.perf_counter()
        run()
        seconds[name] = round(time.perf_counter() - t, 1)
        print(f"-- {name}: {seconds[name]:.0f} s", flush=True)
    failed = [r["check"] for r in RECORDS if not r["passed"]]
    seconds["total"] = round(time.perf_counter() - started, 1)
    report = {
        "passed": not failed,
        "n_checks": len(RECORDS),
        "checks": RECORDS,
        "seconds": seconds,
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "topocyte_commit": c.TOPOCYTE_COMMIT,
        "date": time.strftime("%Y-%m-%d"),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "gudhi": gudhi.__version__,
        },
        "samples": {"balls": [int(r) for r in sample], "all_fit_blocks": list(ALL_FIT_BLOCKS)},
    }
    out = c.RESULTS / "verify.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, out)
    print(f"{seconds['total']:.0f} s, peak RSS {peak_rss_gb():.2f} GB; wrote {out}")
    if failed:
        raise SystemExit(f"{len(failed)} CHECKS FAILED: " + "; ".join(failed))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
