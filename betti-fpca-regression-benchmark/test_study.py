"""Tests of the Betti FPCR benchmark.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``requirements.txt``):

    $PY -m pytest -q -W error test_study.py

The tests marked ``needs_upstream`` read the pinned clone in ``upstream/``,
its pickles only through ``safe_pickle.load_pickle``, and are skipped until
``fetch_upstream.py`` has made it. The others use synthetic data.
"""

from __future__ import annotations

import pathlib
import pickle
import subprocess

import numpy as np
import pytest

import common as c
import fetch_upstream
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
