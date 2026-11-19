"""Constants, cohort readers and pure helpers of the cell-group density-slice study.

The readers open the cohort read only (h5py mode "r"). The helpers need numpy
only and are unit-tested in ``test_celltype.py`` on synthetic inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
SOURCE = HERE.parent / "gtex-liver-pathology-slices" / "results"
COHORT = Path("/Users/simon/data/gtex_liver_cells/gtex_liver_cells")
H5AD = COHORT / "gtex_liver_cells.h5ad"
N_CELLS = 126_219_124
N_SLIDES = 594
N_SLICES = 10
N_COMPONENTS = 9

CELL_GROUPS = ("epithelial", "stromal", "immune", "other")
GROUP_MEMBERS = {
    "epithelial": ("Epithelial", "Cancer cell"),
    "stromal": ("Fibroblasts", "Muscle Cell", "Endothelial Cell", "Minor Stromal Cell"),
    "immune": ("Lymphocytes", "Neutrophils", "Macrophages", "Plasmocytes", "Eosinophils"),
    "other": ("Apoptotic Body", "Red blood cell", "Mitotic Figures"),
}
POINT_SETS = ("all",) + CELL_GROUPS
MODELS = (
    ("h0_none", 0, "none"),
    ("h1_none", 1, "none"),
    ("h0_cosine", 0, "cosine"),
    ("h1_cosine", 1, "cosine"),
)
MODEL_NAMES = tuple(name for name, _, _ in MODELS)
# The sibling study's groups, healthy first, most severe last, unclassified apart.
GROUP_ORDER = (
    "healthy",
    "congestion",
    "steatosis",
    "necrosis",
    "inflammation/hepatitis",
    "fibrosis",
    "cirrhosis",
    "unclassified",
)
CONDITIONS = GROUP_ORDER[:-1]


def _strings(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def group_codes(categories: Sequence[str]) -> np.ndarray:
    """Index into ``CELL_GROUPS`` of every HistoPLUS category; ``ValueError`` names any unmapped one."""
    lookup = {member: index for index, group in enumerate(CELL_GROUPS) for member in GROUP_MEMBERS[group]}
    unknown = [c for c in categories if c not in lookup]
    if unknown:
        raise ValueError(f"cell types without a group: {unknown}")
    return np.array([lookup[c] for c in categories], dtype=np.int8)


def read_slides() -> tuple[list[str], np.ndarray]:
    """The slide names and each slide's first row in the cohort (plus the end)."""
    import h5py

    with h5py.File(H5AD, "r") as f:
        codes = np.asarray(f["obs/slide_id/codes"][:], dtype=np.int16)
        slides = _strings(f["obs/slide_id/categories"][:])
    if codes.size != N_CELLS or len(slides) != N_SLIDES:
        raise RuntimeError(f"expected {N_CELLS:,} cells on {N_SLIDES} slides, got {codes.size:,} on {len(slides)}")
    if np.any(np.diff(codes) < 0):
        raise RuntimeError("cells are not grouped by slide code")
    bounds = np.searchsorted(codes, np.arange(len(slides) + 1), side="left")
    if np.any(np.diff(bounds) <= 0):
        raise RuntimeError("a slide has no cells")
    return slides, bounds


def read_coords() -> np.ndarray:
    import h5py

    with h5py.File(H5AD, "r") as f:
        coords = np.asarray(f["obsm/spatial"][:], dtype=np.float32)
    if coords.shape != (N_CELLS, 2) or not np.all(np.isfinite(coords)):
        raise RuntimeError(f"obsm/spatial has shape {coords.shape} or non-finite values")
    return coords


def read_cell_groups() -> tuple[np.ndarray, list[str]]:
    """Every cell's index into ``CELL_GROUPS`` (int8), and the cohort's cell-type categories."""
    import h5py

    with h5py.File(H5AD, "r") as f:
        codes = np.asarray(f["obs/cell_type/codes"][:], dtype=np.int8)
        categories = _strings(f["obs/cell_type/categories"][:])
    if codes.size != N_CELLS or codes.min() < 0:
        raise RuntimeError("obs/cell_type has the wrong length or a missing value")
    return group_codes(categories)[codes], categories


def slice_mask(rho: np.ndarray, deciles: np.ndarray, index: int) -> np.ndarray:
    """The cells of slice ``index``: every cell at slice 0, else ``rho`` at or above the decile (sibling rule)."""
    return np.ones(rho.size, dtype=bool) if index == 0 else rho >= deciles[index]


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


def mfa(blocks: Sequence[np.ndarray], n_keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multiple factor analysis of score blocks: the first ``n_keep`` scores, every fraction of variance, the weights.

    Each block is divided by the square root of the variance of its first
    column (its leading component then has variance 1), and the centred
    concatenation goes through ``pca``.
    """
    weights = np.array([1.0 / np.sqrt(block[:, 0].var()) for block in blocks])
    scores, evr, _ = pca(np.hstack([block * w for block, w in zip(blocks, weights)]))
    return scores[:, :n_keep], evr, weights
