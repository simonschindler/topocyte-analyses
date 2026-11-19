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


def baseline_features(medians: np.ndarray, kept_by_group: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """The density summaries ``D`` (11 columns) and the composition ``C`` (40), with column names.

    ``D``: log10 median density, log10 cell count, and the share of the
    slide's cells kept at each slice 1..9. ``C``: log10(1 + cells of each
    group kept at each slice), slice-major. ``kept_by_group`` is
    (slices, slides, groups).
    """
    kept = kept_by_group.sum(axis=2)
    n_slices = kept_by_group.shape[0]
    D = np.column_stack([np.log10(medians), np.log10(kept[0])] + [kept[s] / kept[0] for s in range(1, n_slices)])
    d_names = ["log10 median density", "log10 cells"] + [f"share kept at slice {s}" for s in range(1, n_slices)]
    C = np.column_stack([np.log10(1.0 + kept_by_group[s, :, g]) for s in range(n_slices) for g in range(len(CELL_GROUPS))])
    c_names = [f"log10(1 + {g} cells) at slice {s}" for s in range(n_slices) for g in CELL_GROUPS]
    return D, C, d_names, c_names


def leading(scores: np.ndarray, k: int) -> np.ndarray:
    """The first ``k`` columns, or all of them when there are fewer."""
    return scores[:, : min(k, scores.shape[1])]


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive > negative) + P(tie)/2, by the Mann-Whitney rank formula."""
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(proba: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """One-vs-rest AUC of each condition among the condition slides (unclassified unscored), and their mean."""
    scored = labels < len(CONDITIONS)
    per = np.array([auc(proba[scored & (labels == c), c], proba[scored & (labels != c), c]) for c in range(len(CONDITIONS))])
    return float(per.mean()), per


def vs_healthy_auc(proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """For each condition after healthy: AUC of log p_c - log p_healthy between its slides and healthy's."""
    logp = np.log(np.clip(proba, 1e-300, None))
    healthy = labels == 0
    out = []
    for c in range(1, len(CONDITIONS)):
        score = logp[:, c] - logp[:, 0]
        out.append(auc(score[labels == c], score[healthy]))
    return np.array(out)


def purity(scores: np.ndarray, labels: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Share of each condition slide's nearest neighbours in its own group, over the chance share; NaN for unclassified.

    Euclidean distances among all slides, the slide itself excluded, ties
    broken by slide order; chance is ``(n_g - 1) / (N - 1)``.
    """
    from scipy.spatial.distance import cdist

    distances = cdist(scores, scores)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbors]
    same = (labels[nearest] == labels[:, None]).mean(axis=1)
    counts = np.bincount(labels, minlength=len(GROUP_ORDER))
    chance = (counts[labels] - 1) / (labels.size - 1)
    out = same / chance
    out[labels >= len(CONDITIONS)] = np.nan
    return out


def macro_purity(per_slide: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean purity of each condition's slides (NaN for a condition without slides), and their mean."""
    per = np.array([per_slide[labels == c].mean() if np.any(labels == c) else np.nan for c in range(len(CONDITIONS))])
    return float(np.nanmean(per)), per


def stratified_bootstrap(labels: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Index arrays (n_draws, n): every group's slides resampled with replacement within the group."""
    draws = np.empty((n_draws, labels.size), dtype=np.int64)
    for g in np.unique(labels):
        members = np.flatnonzero(labels == g)
        draws[:, members] = rng.choice(members, size=(n_draws, members.size), replace=True)
    return draws


def verdict(auc_interval: Sequence[float], purity_interval: Sequence[float]) -> str:
    """The primary comparison's rule, fixed in advance: which 95% intervals of the difference lie above 0."""
    above = [auc_interval[0] > 0, purity_interval[0] > 0]
    return "enhanced" if all(above) else "mixed" if any(above) else "not enhanced"


def make_folds(labels: np.ndarray, n_repeats: int = 20, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index of every slide in every repeat of stratified k-fold (scikit-learn's RepeatedStratifiedKFold)."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    folds = np.empty((n_repeats, labels.size), dtype=np.int64)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (_, test) in enumerate(splitter.split(np.zeros(labels.size), labels)):
        folds[i // n_splits, test] = i % n_splits
    return folds


def cv_probabilities(X: np.ndarray, labels: np.ndarray, folds: np.ndarray, C: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Out-of-fold class probabilities of the readout's logistic regression in every repeat, their mean, and the
    number of convergence warnings.

    Multinomial L2 logistic regression (lbfgs, ``class_weight="balanced"``,
    ``max_iter=5000``) on features standardized with the training fold's
    mean and SD; columns follow ``GROUP_ORDER``.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    n_repeats, n = folds.shape
    proba = np.zeros((n_repeats, n, len(GROUP_ORDER)))
    n_warnings = 0
    for r in range(n_repeats):
        for f in np.unique(folds[r]):
            test = folds[r] == f
            train = ~test
            mean, sd = X[train].mean(axis=0), X[train].std(axis=0)
            sd[sd == 0] = 1.0
            model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit((X[train] - mean) / sd, labels[train])
            n_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            proba[r][np.ix_(np.flatnonzero(test), model.classes_)] = model.predict_proba((X[test] - mean) / sd)
    return proba.mean(axis=0), proba, n_warnings
