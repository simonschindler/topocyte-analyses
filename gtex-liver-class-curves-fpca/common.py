"""Constants, inputs and the topocyte check of the class-curve fPCA study.

The cell-type study's ``common.py`` is loaded as the module ``celltype``,
because this folder has a ``common.py`` too; its groups, readouts, folds,
bootstrap and logistic regression are reused, not copied. ``load_inputs``
reads the sibling studies' saved results read only and writes nothing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
# The pathology study's density.npz: each cell's 10-NN density rho, the slides' cohort rows and the deciles.
SIB_DENSITY = HERE.parent / "gtex-liver-pathology-slices/results/density.npz"
# The cell-type study: its helpers (common.py) and results (spaces.npz, holdout.csv, ...).
CELLTYPE = HERE.parent / "gtex-liver-celltype-slices"
TOPOCYTE_COMMIT = "18e3aa5"


def _load_celltype() -> ModuleType:
    """The cell-type study's ``common.py`` as the module ``celltype``, its bytecode not written into that folder."""
    spec = importlib.util.spec_from_file_location("celltype", CELLTYPE / "common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["celltype"] = module
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


celltype = _load_celltype()

CLASSES = celltype.CELL_GROUPS  # ("epithelial", "stromal", "immune", "other"); a class's code is its index
NORMALIZATIONS = ("none", "cosine")  # the two fPCA models of each class: 4 classes x 2 = the 8 blocks
N_COMPONENTS = 9  # fPCA components per block
K_GRID = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 51)  # MFA components the inner cross-validation chooses from
MAD_FACTOR = 3 * 1.4826  # outlier: summed distance above its group's median + MAD_FACTOR x MAD (training slides)
DEV_FOLD_SEED = 1  # make_folds(dev labels, 20, 5, seed=1): the cell-type improvement phase's development folds
BOOT_SEED = 12345  # stratified_bootstrap(labels, N_BOOT, np.random.default_rng(BOOT_SEED))
N_BOOT = 4000


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, check=True).stdout.strip()


def check_topocyte() -> None:
    """Refuse a topocyte other than a clean checkout at ``TOPOCYTE_COMMIT`` named first on ``PYTHONPATH``.

    The sibling studies' check: topocyte must be imported from that checkout,
    its ``HEAD`` must start with the commit, and its ``topocyte/`` must not
    differ from the commit.
    """
    first = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if not first:
        raise RuntimeError("run with PYTHONPATH=<topocyte checkout>; see README.md")
    checkout = Path(first).expanduser().resolve()
    import topocyte

    if not Path(topocyte.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError(f"imported topocyte from {topocyte.__file__}, not from the checkout {checkout}")
    head = _git(checkout, "rev-parse", "HEAD")
    changed = _git(checkout, "status", "--porcelain", "--", "topocyte")
    if not head.startswith(TOPOCYTE_COMMIT) or changed:
        raise RuntimeError(f"topocyte checkout is at {head[:7]} (package changed: {bool(changed)}), not a clean {TOPOCYTE_COMMIT}")


def load_inputs() -> dict:
    """The slides, their groups and split, ``D+C``, and the cohort rows and deciles of ``density.npz``.

    Keys:

    - ``slide_ids``: the cell-type study's ``spaces.npz`` order, which
      ``holdout.csv`` repeats;
    - ``labels``: indices into ``celltype.GROUP_ORDER``;
    - ``dev``, ``holdout``: ``holdout.csv``'s split (476 and 118 slides);
    - ``DC``: ``D_features`` (11 columns) beside ``C_features`` (40);
    - ``bounds`` (595), ``deciles`` (10): from ``density.npz``, whose
      ``rho`` (1 GB) is not read.

    ``RuntimeError`` when ``spaces.npz``'s group order is not ``GROUP_ORDER``,
    or ``holdout.csv`` or ``density.npz`` lists the slides differently.
    """
    with np.load(CELLTYPE / "results/spaces.npz", allow_pickle=False) as spaces:
        slide_ids = [str(s) for s in spaces["slide_ids"]]
        group_order = tuple(str(g) for g in spaces["group_order"])
        labels = spaces["labels"]
        DC = np.hstack([spaces["D_features"], spaces["C_features"]])
    if group_order != celltype.GROUP_ORDER:
        raise RuntimeError(f"spaces.npz labels index {group_order}, not the cell-type study's GROUP_ORDER")
    split = pd.read_csv(CELLTYPE / "results/holdout.csv")
    if [str(s) for s in split["slide_id"]] != slide_ids:
        raise RuntimeError("holdout.csv lists the slides differently from spaces.npz")
    holdout = split["holdout"].to_numpy(dtype=bool)
    with np.load(SIB_DENSITY, allow_pickle=False) as density:
        if [str(s) for s in density["slide_ids"]] != slide_ids:
            raise RuntimeError("density.npz lists the slides differently from spaces.npz")
        bounds, deciles = density["bounds"], density["deciles"]
    return {
        "slide_ids": slide_ids,
        "labels": labels,
        "dev": ~holdout,
        "holdout": holdout,
        "DC": DC,
        "bounds": bounds,
        "deciles": deciles,
    }
