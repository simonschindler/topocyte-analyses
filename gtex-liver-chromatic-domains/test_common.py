"""Unit tests of the study's constants and helpers, on synthetic inputs (no cohort, no topocyte).

The ``bars`` tests run ``run_pilot.cmd_bars`` on two 300-cell synthetic
slides with ``common.read_slide`` patched and ``common.N_CENTRES`` set to 20,
through a real spawn pool of two processes and chunks of 6 centres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

import chromatic
import common
import run_pilot

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


@pytest.mark.parametrize("r", [-1, common.R + 1, 100])
def test_labelling_refuses_other_streams(r):
    # Stream R + 1 is the slide's centre draw; a labelling outside 0..R would reuse or invent a stream.
    cls = np.repeat(np.arange(4), 5).astype(np.int8)
    with pytest.raises(ValueError, match="labelling"):
        common.labelling(cls, 2, 7, r)
    assert np.array_equal(np.bincount(common.labelling(cls, 2, 7, common.R), minlength=4), [5, 5, 5, 5])


def test_owner_roundtrip():
    n_labellings, n_blocks = common.R + 1, len(common.BLOCKS)
    assert (n_labellings, n_blocks) == (20, 16)
    last = common.N_CENTRES - 1
    for triple in [(0, 0, 0), (0, 0, 15), (0, 19, 0), (last, 0, 0), (last, 19, 15), (1, 0, 0), (0, 1, 0)]:
        code = common.encode_owner(*triple)
        assert tuple(int(v) for v in common.decode_owner(code)) == triple
    assert common.encode_owner(last, 19, 15) == common.N_CENTRES * n_labellings * n_blocks - 1 == 479_999
    # Every (centre, labelling, block) has its own code, and the codes run through 0..N - 1 in that order.
    grid = np.meshgrid(np.arange(common.N_CENTRES), np.arange(n_labellings), np.arange(n_blocks), indexing="ij")
    codes = common.encode_owner(*(axis.ravel() for axis in grid))
    assert codes.dtype == np.int32
    assert np.array_equal(codes, np.arange(common.N_CENTRES * n_labellings * n_blocks))
    assert all(np.array_equal(x, y.ravel()) for x, y in zip(common.decode_owner(codes), grid, strict=True))
    for bad in [(-1, 0, 0), (0, 20, 0), (0, -1, 0), (0, 0, 16), (0, 0, -1)]:
        with pytest.raises(ValueError, match="owner"):
            common.encode_owner(*bad)
    with pytest.raises(ValueError, match="owner"):
        common.decode_owner(-1)


FAKE_PILOT = pd.DataFrame(
    {
        "slide_id": ["GTEX-FAKE-0", "GTEX-FAKE-1"],
        "condition": ["healthy", "cirrhosis"],
        "half": ["A", "B"],
        "slide_index": [0, 1],
    }
)


def fake_slide(slide_id: str) -> tuple[np.ndarray, np.ndarray]:
    """300 cells uniform in a 1,500 px square (balls of about 17-44 cells), four classes, one draw per fake slide."""
    rng = np.random.default_rng({"GTEX-FAKE-0": 10, "GTEX-FAKE-1": 11}[slide_id])
    return rng.uniform(0, 1500, size=(300, 2)), rng.integers(0, 4, 300).astype(np.int8)


@pytest.fixture
def fake_run(monkeypatch):
    """``read_slide`` patched to the fake slides and 20 centres per slide; ``bars`` runs on 2 processes, chunks of 6."""
    monkeypatch.setattr(common, "read_slide", fake_slide)
    monkeypatch.setattr(common, "N_CENTRES", 20)
    return lambda out, slides=None: run_pilot.cmd_bars(slides, workers=2, pilot=FAKE_PILOT, out=out, chunk=6)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def test_bars_complete_needs_every_key_and_whole_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "N_CENTRES", 2)
    n_owners = 2 * (common.R + 1) * len(common.BLOCKS)
    good = {key: np.zeros(1) for key in run_pilot.BARS_KEYS}
    good.update(bars=np.ones((3, 2)), owner=np.array([0, 0, 5], np.int32), offsets=np.r_[0, np.full(n_owners, 3)])
    good["offsets"][1:6] = 2

    def write(name: str, arrays: dict) -> Path:
        path = tmp_path / name
        with open(path, "wb") as f:
            np.savez(f, **arrays)
        return path

    assert run_pilot.bars_complete(write("good.npz", good))
    assert not run_pilot.bars_complete(tmp_path / "missing.npz")
    assert not run_pilot.bars_complete(write("no_key.npz", {k: v for k, v in good.items() if k != "codomain"}))
    short = good | {"bars": np.ones((2, 2)), "owner": np.zeros(2, np.int32)}
    assert not run_pilot.bars_complete(write("short_bars.npz", short))
    assert not run_pilot.bars_complete(write("short_offsets.npz", good | {"offsets": good["offsets"][:-1]}))
    assert not run_pilot.bars_complete(write("owner_length.npz", good | {"owner": np.zeros(2, np.int32)}))
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes((tmp_path / "good.npz").read_bytes()[:-40])
    assert not run_pilot.bars_complete(truncated)
    (tmp_path / "junk.npz").write_bytes(b"not a zip file")
    assert not run_pilot.bars_complete(tmp_path / "junk.npz")


def test_resume_skips_complete_and_redoes_partial(tmp_path, fake_run):
    out = tmp_path / "bars"
    assert fake_run(out) == {"GTEX-FAKE-0": "computed", "GTEX-FAKE-1": "computed"}
    complete, partial = out / "GTEX-FAKE-0.npz", out / "GTEX-FAKE-1.npz"
    assert run_pilot.bars_complete(complete) and run_pilot.bars_complete(partial)
    first = _arrays(partial)
    mtime = complete.stat().st_mtime_ns
    data = partial.read_bytes()
    partial.write_bytes(data[: len(data) // 2])  # a copy cut short
    (out / "GTEX-FAKE-1.npz.tmp").write_bytes(data[:100])  # and an interrupted write beside it
    assert not run_pilot.bars_complete(partial)

    assert fake_run(out) == {"GTEX-FAKE-0": "skipped", "GTEX-FAKE-1": "computed"}
    assert complete.stat().st_mtime_ns == mtime
    assert run_pilot.bars_complete(partial)
    again = _arrays(partial)
    assert set(again) == set(first) == set(run_pilot.BARS_KEYS)
    for key in first:
        if key != "seconds":  # timings differ between runs; everything else is recomputed bit for bit
            assert np.array_equal(again[key], first[key]), key
    assert sorted(p.name for p in out.iterdir()) == [
        "GTEX-FAKE-0.json", "GTEX-FAKE-0.npz", "GTEX-FAKE-1.json", "GTEX-FAKE-1.npz",
    ]


def test_bars_match_direct_computation(tmp_path, fake_run):
    assert fake_run(tmp_path, ["GTEX-FAKE-1"]) == {"GTEX-FAKE-1": "computed"}
    z = _arrays(tmp_path / "GTEX-FAKE-1.npz")
    xy, cls = fake_slide("GTEX-FAKE-1")
    centres = common.draw_centres(len(xy), 1)
    assert np.array_equal(z["centre_rows"], centres) and np.array_equal(z["centre_xy"], xy[centres])
    n_owners = 20 * (common.R + 1) * len(common.BLOCKS)
    bars, owner, offsets = z["bars"], z["owner"], z["offsets"]
    assert bars.dtype == np.float64 and owner.dtype == np.int32 and offsets.shape == (n_owners + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(bars) == len(owner) and np.all(np.diff(owner) >= 0)
    assert np.all(np.isfinite(bars)) and np.all(bars[:, 1] > bars[:, 0])
    assert np.all(np.isfinite(z["codomain"])) and np.all(np.diff(z["codomain_owner"]) >= 0)
    assert z["seconds"].shape == (20,) and np.all(z["seconds"] > 0)

    def finite_sorted(diagram: np.ndarray) -> np.ndarray:
        finite = diagram[np.isfinite(diagram[:, 1])]
        return finite[np.lexsort((finite[:, 1], finite[:, 0]))]

    tree = cKDTree(xy)
    for i, row in enumerate(centres):
        ball = np.sort(np.asarray(tree.query_ball_point(xy[row], r=common.DELTA_PX), dtype=np.int64))
        assert np.array_equal(ball, np.flatnonzero(np.hypot(*(xy - xy[row]).T) <= common.DELTA_PX))  # closed ball
        bxy, bcls = xy[ball], cls[ball]
        assert z["n_cells"][i] == len(ball) == z["class_counts"][i].sum()
        assert np.array_equal(z["class_counts"][i], np.bincount(bcls, minlength=4))
        counts, n_edges = chromatic.pair_counts(bxy, bcls)
        assert np.array_equal(z["pair_counts"][i], counts) and z["n_edges"][i] == n_edges
        for label in range(common.R + 1):
            packs = chromatic.ball_sixpacks(bxy, common.labelling(bcls, 1, i, label))
            for block, (t, module, p) in enumerate(common.BLOCKS):
                code = common.encode_owner(i, label, block)
                assert np.array_equal(bars[offsets[code] : offsets[code + 1]], finite_sorted(packs[module, t, p]))
                assert np.all(owner[offsets[code] : offsets[code + 1]] == code)
            if label == 0:
                for p in (0, 1):
                    mine = z["codomain"][z["codomain_owner"] == 2 * i + p]
                    assert np.array_equal(mine, finite_sorted(packs["codomain", -1, p]))
    # Labelling 0 is the observed classes; the shuffles keep each ball's count per class.
    assert len(z["codomain"]) > 0 and np.unique(z["codomain_owner"]).size == 2 * 20
