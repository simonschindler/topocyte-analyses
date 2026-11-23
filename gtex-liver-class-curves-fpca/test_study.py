"""Tests of the class-curve fPCA study.

Run them from this folder in topocyte's main-checkout venv, with the pinned
checkout first on ``PYTHONPATH`` (``progress.md``, Rulings). The input tests
read the sibling studies' saved results, read only.
"""

from __future__ import annotations

import numpy as np
import pytest

import common as c


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
