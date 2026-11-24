"""Tests of the Betti FPCR benchmark.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``requirements.txt``):

    $PY -m pytest -q -W error test_study.py

The tests marked ``needs_upstream`` read the pinned clone in ``upstream/``,
its pickles only through ``safe_pickle.load_pickle``, and are skipped until
``fetch_upstream.py`` has made it. The others use synthetic data.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import subprocess

import numpy as np
import pytest
import ripser

import common as c
import fetch_upstream
import inputs as inp
import safe_pickle

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
