"""Constants, readers, checks, seeded draws and the bars' owner code of the chromatic-domains pilot.

The readers open the cohort read only (h5py mode "r") and write nothing. The
draws need numpy and pandas only and are unit-tested in ``test_common.py`` on
synthetic inputs.

Seeds: the pilot draw runs one ``default_rng(SEED)`` through the conditions.
Every other generator comes from a 4-entry key
``SeedSequence([SEED, slide_index, centre_index, stream])`` whose last entry
is non-zero. SeedSequence pads short keys with zeros, so ``default_rng(SEED)``
is ``SeedSequence([SEED, 0, 0, 0])``, and the non-zero stream keeps every
other key apart from it. Any slide or centre can be recomputed alone:

- the slide's centre draw is centre index 0, stream R + 1;
- labelling r (1..R) of a centre is stream r. Labelling 0 is the observed
  classes and draws nothing.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
H5AD = Path("/Users/simon/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad")
# The sibling studies' results: the pathology-slice study's density.npz (each slide's cohort rows) and
# pathology.csv (each slide's condition), and the cell-type study's holdout.csv.
SIB = HERE.parent / "gtex-liver-pathology-slices" / "results"
CT = HERE.parent / "gtex-liver-celltype-slices" / "results"

# The cell-type study's four groups of HistoPLUS cell types (its common.GROUP_MEMBERS); a class's code is its index.
CLASSES = ("epithelial", "stromal", "immune", "other")
GROUP_MEMBERS = {
    "epithelial": ("Epithelial", "Cancer cell"),
    "stromal": ("Fibroblasts", "Muscle Cell", "Endothelial Cell", "Minor Stromal Cell"),
    "immune": ("Lymphocytes", "Neutrophils", "Macrophages", "Plasmocytes", "Eosinophils"),
    "other": ("Apoptotic Body", "Red blood cell", "Mitotic Figures"),
}
# The pilot's conditions (pathology.csv's group), healthy first; necrosis and unclassified are left out.
CONDITIONS = ("healthy", "congestion", "steatosis", "inflammation/hepatitis", "fibrosis", "cirrhosis")

PX_UM = 0.4942  # µm per level-0 pixel: assumed, since the cohort has no mpp and every local SVS has 0.4942
DELTA_UM = 150.0  # radius of the closed ball around each centre
DELTA_PX = DELTA_UM / PX_UM

R = 19  # random labellings per centre
N_CENTRES = 1500  # centres per slide
PER_CONDITION = 5  # pilot slides per condition
FIT_PER_SLIDE = 400  # centres per slide in a fit sample
SEED = 20261120
TOPOCYTE_COMMIT = "18e3aa5"

MODULES = ("domain", "image")
# The 16 blocks (class t, module, degree p), indexed by b.
BLOCKS = tuple((t, m, p) for t in range(4) for m in MODULES for p in (0, 1))
# The 10 unordered class pairs (a, b), a <= b.
PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))


def _strings(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


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


def class_codes(categories: Sequence[str]) -> np.ndarray:
    """The class code (index into ``CLASSES``) of every HistoPLUS category, int8; ``ValueError`` names any unmapped one."""
    lookup = {member: code for code, name in enumerate(CLASSES) for member in GROUP_MEMBERS[name]}
    unknown = [c for c in categories if c not in lookup]
    if unknown:
        raise ValueError(f"cell types without a class: {unknown}")
    return np.array([lookup[c] for c in categories], dtype=np.int8)


def read_slide(slide_id: str) -> tuple[np.ndarray, np.ndarray]:
    """One slide's cells, read only: coordinates (n, 2) in level-0 px as float64, and class codes (n,) as int8.

    The slide's rows come from the sibling study's ``density.npz``; the
    cohort file confirms that its first and last rows belong to the slide.
    """
    import h5py

    with np.load(SIB / "density.npz", allow_pickle=False) as density:
        slides = [str(s) for s in density["slide_ids"]]
        bounds = density["bounds"]
    index = slides.index(slide_id)
    start, end = int(bounds[index]), int(bounds[index + 1])
    with h5py.File(H5AD, "r") as f:
        names = _strings(f["obs/slide_id/categories"][:])
        owners = f["obs/slide_id/codes"]
        first, last = names[int(owners[start])], names[int(owners[end - 1])]
        xy = np.asarray(f["obsm/spatial"][start:end], dtype=np.float64)
        codes = np.asarray(f["obs/cell_type/codes"][start:end])
        categories = _strings(f["obs/cell_type/categories"][:])
    if first != slide_id or last != slide_id:
        raise RuntimeError(f"cohort rows {start}:{end} run from {first} to {last}, not {slide_id}")
    if xy.shape != (end - start, 2) or not np.all(np.isfinite(xy)):
        raise RuntimeError(f"{slide_id}: obsm/spatial has shape {xy.shape} or non-finite values")
    if codes.min() < 0:
        raise RuntimeError(f"{slide_id}: a cell has no cell type")
    return xy, class_codes(categories)[codes]


def eligible_slides(holdout_csv: Path, pathology_csv: Path) -> pd.DataFrame:
    """The slides the pilot draws from, sorted by slide: ``slide_id``, ``condition``, ``n_cells``.

    The cell-type study's development slides (``holdout`` False) without a
    ``quality_flag``, whose condition (``pathology.csv``'s ``group``) is one
    of ``CONDITIONS``.
    """
    holdout = pd.read_csv(holdout_csv)
    pathology = pd.read_csv(pathology_csv)
    table = pathology.merge(holdout[["slide_id", "holdout"]], on="slide_id", how="inner", validate="one_to_one")
    keep = table["holdout"].eq(False) & table["quality_flag"].isna() & table["group"].isin(CONDITIONS)
    eligible = table.loc[keep, ["slide_id", "group", "n_cells"]].rename(columns={"group": "condition"})
    return eligible.sort_values("slide_id", ignore_index=True)


def draw_pilot(eligible: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """The pilot slides and their halves (design step 1): ``slide_id``, ``condition``, ``half``, ``slide_index``.

    One generator, ``default_rng(seed)``, runs through ``CONDITIONS`` in
    order. For each condition it draws ``PER_CONDITION`` of the candidates
    sorted by slide ID, without replacement, then permutes them. The first 3
    of the permutation (even condition index) or the first 2 (odd) go to
    half A and the rest to half B, so each half holds 15 slides and every
    condition appears in both. Rows follow the permutation, and
    ``slide_index`` numbers them 0..29 in table order.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index, condition in enumerate(CONDITIONS):
        candidates = np.sort(eligible.loc[eligible["condition"] == condition, "slide_id"].to_numpy(dtype=str))
        drawn = rng.permutation(rng.choice(candidates, size=PER_CONDITION, replace=False))
        in_a = (PER_CONDITION + 1) // 2 if index % 2 == 0 else PER_CONDITION // 2
        rows += [(str(slide), condition, "A" if j < in_a else "B") for j, slide in enumerate(drawn)]
    table = pd.DataFrame(rows, columns=["slide_id", "condition", "half"])
    table["slide_index"] = np.arange(len(table))
    return table


def _generator(slide_index: int, centre_index: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, slide_index, centre_index, stream]))


def draw_centres(n_cells: int, slide_index: int) -> np.ndarray:
    """``N_CENTRES`` distinct rows of a slide's cells, uniform without replacement, in draw order.

    A row's position in the result is its centre index. The generator is the
    slide's: centre index 0, stream R + 1, after the labellings' streams 1..R.
    """
    return _generator(slide_index, 0, R + 1).choice(n_cells, size=N_CENTRES, replace=False)


def labelling(cls: np.ndarray, slide_index: int, centre_index: int, r: int) -> np.ndarray:
    """Labelling ``r`` of a centre's ball: the observed classes ``cls`` themselves for r = 0, else a seeded permutation.

    A permutation keeps the positions and the count per class. Labelling r of
    a centre comes from its stream r, 1..R; any other r is refused, since
    stream R + 1 is the slide's centre draw.
    """
    if r == 0:
        return cls
    if not 1 <= r <= R:
        raise ValueError(f"labelling r must be 0 (observed) or 1..{R} (shuffles), got {r}")
    return _generator(slide_index, centre_index, r).permutation(cls)


def encode_owner(centre: int | np.ndarray, labelling: int | np.ndarray, block: int | np.ndarray) -> np.ndarray:
    """The owner code of a block's bars in ``results/bars/*.npz``: ``(centre·(R + 1) + labelling)·len(BLOCKS) + block``.

    ``labelling`` is 0..R and ``block`` the index into ``BLOCKS``. Codes run
    through 0..N_CENTRES·(R + 1)·len(BLOCKS) − 1 in (centre, labelling, block)
    order, as int32; ``ValueError`` for an argument out of range.
    """
    centre, labelling, block = (np.asarray(v, dtype=np.int64) for v in (centre, labelling, block))
    if np.any(centre < 0) or np.any((labelling < 0) | (labelling > R)) or np.any((block < 0) | (block >= len(BLOCKS))):
        raise ValueError(f"owner needs centre >= 0, labelling in 0..{R} and block in 0..{len(BLOCKS) - 1}")
    code = (centre * (R + 1) + labelling) * len(BLOCKS) + block
    if np.any(code > np.iinfo(np.int32).max):
        raise ValueError("owner code beyond int32")
    return code.astype(np.int32)


def decode_owner(code: int | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centre, labelling, block)`` of owner codes from ``encode_owner``, as int64."""
    code = np.asarray(code, dtype=np.int64)
    if np.any(code < 0):
        raise ValueError("owner codes are >= 0")
    rest, block = np.divmod(code, len(BLOCKS))
    centre, labelling = np.divmod(rest, R + 1)
    return centre, labelling, block
