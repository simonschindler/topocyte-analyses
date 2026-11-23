"""Paths, constants and read-only cohort access shared by the e-means neighbourhood study."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
H5AD = Path("/Users/simon/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad")
# The pathology-slice study's per-cell 10-NN density (density.npz: rho, deciles, bounds) and pathology.csv.
SIB = HERE.parent / "gtex-liver-pathology-slices" / "results"

# The cell-type study's four classes of HistoPLUS cell types (as gtex-liver-chromatic-domains/common.py).
CLASSES = ("epithelial", "stromal", "immune", "other")
GROUP_MEMBERS = {
    "epithelial": ("Epithelial", "Cancer cell"),
    "stromal": ("Fibroblasts", "Muscle Cell", "Endothelial Cell", "Minor Stromal Cell"),
    "immune": ("Lymphocytes", "Neutrophils", "Macrophages", "Plasmocytes", "Eosinophils"),
    "other": ("Apoptotic Body", "Red blood cell", "Mitotic Figures"),
}

PX_UM = 0.4942  # µm per level-0 pixel, assumed: the cohort records none and every local SVS has 0.4942
RADIUS_UM = 150.0  # the closed ball around every cell, the chromatic pilot's
RADIUS_PX = RADIUS_UM / PX_UM
N_THRESHOLDS = 10  # the pathology study's density thresholds: its ten deciles of rho

# counts.npy columns (uint16): the ball's cells of each class, then the cells kept at thresholds 1..9
# (rho >= decile j); threshold 0 is the minimum of rho, which keeps every cell, so its count is the ball's.
COUNT_COLUMNS = (*(f"n {c}" for c in CLASSES), *(f"kept at threshold {j}" for j in range(1, N_THRESHOLDS)))

SEED = 20261123
K_MAX = 10
N_EPOCHS = 5  # e-means' default number of passes
N_REFERENCES = 20  # gap references per k, as the chromatic pilot
N_REFERENCE_POINTS = 20_000_000  # points per reference (memory); W_k is a mean, so n cancels in the gap
MIN_GAIN = 0.03  # the chromatic pilot's floor on Gap(k) - Gap(1)
SETS = ("composition", "density", "composition+density")


def strings(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def class_lookup(categories: list[str]) -> np.ndarray:
    """The class code of every HistoPLUS category (int8), in category order; ``ValueError`` names an unmapped one."""
    lookup = {member: code for code, name in enumerate(CLASSES) for member in GROUP_MEMBERS[name]}
    unknown = [c for c in categories if c not in lookup]
    if unknown:
        raise ValueError(f"cell types without a class: {unknown}")
    return np.array([lookup[c] for c in categories], dtype=np.int8)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()
