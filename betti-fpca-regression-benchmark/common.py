"""Paths, constants, the topocyte check and the shared helpers of the Betti FPCR benchmark.

The constants are the numbers ``design.md`` fixed before any result: the
paper's three tasks, the estimators' grids and caps, the bootstrap and the
gates' tolerances. ``provenance`` is the record that every stage's json
embeds.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
# Upstream: the code and data of Aggarwal, Nichols and Turner (arXiv:2611.25295), cloned at UPSTREAM_SHA by
# fetch_upstream.py. Data only: never imported, executed or put on sys.path; its pickles go through safe_pickle.
UPSTREAM = HERE / "upstream"
UPSTREAM_URL = "https://github.com/saksham-aggarwal-2002/kernel-regression-persistence-diagrams.git"
UPSTREAM_SHA = "fe7d652eff478ffe6081c7e6765458651ad3cd74"
TOPOCYTE_COMMIT = "18e3aa5"

# Interpolation (the paper's Table 1): 10 runs of each training size, each with its own 50 test clouds.
INTERP_SIZES = tuple(range(10, 101, 10))
INTERP_RUNS = 10
INTERP_N_TEST = 50
# Fluid (Table 3): the snapshot times; split s tests on 100 simulations and trains on the first n of the rest.
FLUID_TIMES = (1.0, 2.5, 5.0, 7.5, 9.0)
FLUID_SIZES = (25, 50, 100, 200, 300)
FLUID_N_TEST = 100
FLUID_SPLITS = 100  # splits 0-99, by the paper's recipe
PAPER_SPLITS = 5  # the paper's splits, 0-4
FLUID_SHIFT = 26.0  # added to every fluid bar endpoint: the ceiling of minus the lowest birth (-25.09)

K_MAX = 20  # FPCR: at most 20 components per dimension
LEVERAGE_CAP = 1 - 1e-10  # FPCR: a (K0, K1) with a leverage at or above it is not eligible
N_BANDWIDTHS = 50  # Nadaraya-Watson: linspace(0.1 m, 2 m, 50), m the median positive training distance
N_BOOT = 10_000
BOOT_SEED = 0  # np.random.default_rng(BOOT_SEED), a fresh one for every interval
G1_TOL = 1e-9  # absolute: arm K against upstream's results
G2_TOL = 1e-12  # relative: the scipy W2 against upstream's cached distances
G3_TOL = 1e-9  # relative to the largest score: projected test scores against FPCA.transform
G4_TOL = 1e-12  # relative: the fluid Gram matrices at FLUID_SHIFT against FLUID_SHIFT + 10


# _git and check_topocyte are copied from ../gtex-liver-class-curves-fpca/common.py, not imported, so that the
# study stays self-contained and pinned to its own topocyte.
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


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """The paper's R² of predictions ``p`` of ``y``: ``1 - SSE/SST``, with SST about ``y.mean()``."""
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def provenance(*scripts: Path) -> dict:
    """The md5 of each script (by file name), the versions of numpy, scipy, pandas and ripser, and the pins."""
    return {
        "md5": {Path(script).name: hashlib.md5(Path(script).read_bytes()).hexdigest() for script in scripts},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "ripser")},
        "topocyte_commit": TOPOCYTE_COMMIT,
        "upstream_sha": UPSTREAM_SHA,
    }
