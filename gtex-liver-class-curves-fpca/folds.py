"""The class-curve study's algebra inside a fold: the outlier filter, each block's fPCA and the MFA.

Everything here is linear algebra on the exact Gram matrices of ``curves.py``; nothing reads a file. ``grams`` maps each
class of ``CLASSES`` to its 594 × 594 Gram matrix in the study's slide order, and indices are positions in that order.
A block is a class under one of ``NORMALIZATIONS``, eight blocks in all. Its analysis kernel is the Gram matrix
(``"none"``) or the cosine similarities (``"cosine"``), always computed over all the slides: a cosine similarity needs
both slides' norms, which only the whole Gram's diagonal holds.

Inside a fold, everything is fitted on the training slides, and only ``outlier_keep`` reads labels, theirs alone:

- ``outlier_keep``: each training slide's squared distance to its group's mean curve in every block, divided by the
  block's median and summed over the blocks; a group drops the slides above its median + ``MAD_FACTOR`` × MAD;
- ``BlockFPCA``: topocyte's ``fpca_from_gram`` on a block's kernel among the fitted slides, and the exact projection of
  any other slide from its kernel values against them;
- ``MFA``: the eight blocks' scores, each block divided by the SD of its first column, concatenated, centred and
  decomposed by SVD;
- ``curve_space``: the eight fPCAs and the MFA fitted on some slides, scoring those and others.
"""

from __future__ import annotations

import numpy as np
from topocyte.ph import fpca_from_gram

import common as c

MFA_COMPONENTS = max(c.K_GRID)  # the MFA keeps at most 51 components, the largest k the inner cross-validation tries,
# and only those whose share of the variance exceeds MFA_RTOL (fpca_from_gram's default rtol): centring leaves n slides
# at most n - 1 components, so the SVD's n-th is rounding noise, as is any beyond the blocks' joint rank.
MFA_RTOL = 1e-10


def analysis_kernel(G: np.ndarray, normalization: str) -> np.ndarray:
    """A block's analysis kernel: the Gram matrix ``G`` itself, or the cosine similarities ``D^-1/2 G D^-1/2``.

    ``D`` is the diagonal of ``G``, the curves' squared norms. Under ``"cosine"``, ``ValueError`` names each slide (an
    index into ``G``) whose curve is 0, which has no direction: a slide without the class's cells.
    """
    if normalization == "none":
        return G
    if normalization != "cosine":
        raise ValueError(f"normalization must be one of {c.NORMALIZATIONS}, not {normalization!r}")
    squared = np.diag(G)
    zero = np.flatnonzero(~(squared > 0))
    if zero.size:
        raise ValueError(f"a zero curve has no cosine similarity: {', '.join(f'slide {i}' for i in zero)}")
    norms = np.sqrt(squared)
    return G / np.outer(norms, norms)


def _kernels(grams: dict[str, np.ndarray]) -> list[np.ndarray]:
    """The eight blocks' analysis kernels over all the slides, class by class; a zero curve's error names its class."""
    kernels = []
    for name in c.CLASSES:
        for normalization in c.NORMALIZATIONS:
            try:
                kernels.append(analysis_kernel(grams[name], normalization))
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
    return kernels


def block_distances(K: np.ndarray, train: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Each training slide's squared distance to its group's mean curve over ``train``, from the analysis kernel ``K``.

    For slide ``f`` of group ``g``: ``K_ff - 2 mean_j K_fj + mean_jk K_jk`` over the training slides ``j``, ``k`` of
    ``g``, ``f`` among them. ``K`` and ``labels`` cover all the slides; only ``labels[train]`` is read.
    """
    y = labels[train]
    distances = np.empty(len(train))
    for group in np.unique(y):
        rows = y == group
        members = train[rows]
        block = K[np.ix_(members, members)]
        distances[rows] = np.diag(block) - 2 * block.mean(axis=1) + block.mean()
    return distances


def outlier_keep(grams: dict[str, np.ndarray], train: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mask over ``train``: False for the training slides the outlier rule drops (design.md, Pipeline 4.1).

    Each block's ``block_distances`` are divided by their median over ``train``, and the eight blocks' are summed. A
    slide is dropped when its sum exceeds its group's median + ``MAD_FACTOR`` × MAD. A group of fewer than 3 training
    slides, or with MAD 0, drops nobody, and no group is emptied: at least half of its slides lie at or below its
    median.
    """
    summed = np.zeros(len(train))
    for K in _kernels(grams):
        distances = block_distances(K, train, labels)
        summed += distances / np.median(distances)
    y = labels[train]
    keep = np.ones(len(train), dtype=bool)
    for group in np.unique(y):
        rows = y == group
        if rows.sum() < 3:
            continue
        median = np.median(summed[rows])
        mad = np.median(np.abs(summed[rows] - median))
        if mad > 0:
            keep[rows] = summed[rows] <= median + c.MAD_FACTOR * mad
    return keep


class BlockFPCA:
    """One block's fPCA: topocyte's fit on the kernel among the fitted slides, and the exact projection of others.

    ``fit(K_tt)`` runs ``fpca_from_gram`` with ``normalization="none"``, because ``K_tt`` is already the analysis kernel
    (a cosine block's cosine similarities), and keeps the first ``N_COMPONENTS`` components, fewer when the rank is
    lower. ``transform(K_xt)`` scores slides from their kernel values against the fitted slides, as topocyte's
    ``FPCA.transform`` does from the curves: a row ``k`` centred with the fitted kernel's means,
    ``k_c = k - mean(k) - colmean(K_tt) + mean(K_tt)``, holds the inner products of the slide's centred curve with the
    fitted slides' centred curves, and ``score_k = k_c · v_k / sqrt(rho_k)``. A fitted slide gets its fitted scores
    back.
    """

    def fit(self, K_tt: np.ndarray) -> BlockFPCA:
        result = fpca_from_gram(K_tt, normalization="none")
        k = min(c.N_COMPONENTS, result.gram_eigenvalues.size)
        self.scores_ = result.scores[:, :k]  # sqrt(rho_k) v_k
        self.explained_variance_ratio_ = result.explained_variance_ratio[:k]  # of the total centred variation
        self.eigenvectors_ = result.eigenvectors[:, :k]
        self.gram_eigenvalues_ = result.gram_eigenvalues[:k]
        self.column_means_ = result.analysis_gram.mean(axis=0)
        self.mean_ = result.analysis_gram.mean()
        return self

    def transform(self, K_xt: np.ndarray) -> np.ndarray:
        centred = K_xt - K_xt.mean(axis=1, keepdims=True) - self.column_means_ + self.mean_
        return centred @ self.eigenvectors_ / np.sqrt(self.gram_eigenvalues_)


class MFA:
    """Multiple factor analysis of the blocks' scores, fitted on some slides and applied to any.

    ``fit``: each block is divided by the SD of its first column, as in ``celltype.mfa``, and the concatenation is
    centred and goes through ``celltype.pca`` (SVD; the largest-magnitude entry of each left singular vector positive).
    Up to ``MFA_COMPONENTS`` components are kept, those whose share of the variance exceeds ``MFA_RTOL``. ``transform``
    weights and centres rows with the fitted weights and means and projects them on the fitted components; a fitted row
    gets its fitted scores back.
    """

    def fit(self, blocks: list[np.ndarray]) -> MFA:
        self.weights_ = np.array([1.0 / np.sqrt(block[:, 0].var()) for block in blocks])
        X = self._weighted(blocks)
        self.mean_ = X.mean(axis=0)
        scores, evr, vt = c.celltype.pca(X)
        k = min(MFA_COMPONENTS, int(np.count_nonzero(evr > MFA_RTOL)))
        self.scores_, self.components_ = scores[:, :k], vt[:k]
        return self

    def transform(self, blocks: list[np.ndarray]) -> np.ndarray:
        return (self._weighted(blocks) - self.mean_) @ self.components_.T

    def _weighted(self, blocks: list[np.ndarray]) -> np.ndarray:
        return np.hstack([block * w for block, w in zip(blocks, self.weights_, strict=True)])


def curve_space(
    grams: dict[str, np.ndarray], fit_idx: np.ndarray, other_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The MFA scores of the slides ``fit_idx`` and of ``other_idx``, the eight fPCAs and the MFA fitted on ``fit_idx``.

    Each block's analysis kernel is computed over all the slides; its ``fit_idx`` × ``fit_idx`` part fits the block's
    ``BlockFPCA`` and its ``other_idx`` × ``fit_idx`` part is projected. No label is read.
    """
    fit_blocks, other_blocks = [], []
    for K in _kernels(grams):
        block = BlockFPCA().fit(K[np.ix_(fit_idx, fit_idx)])
        fit_blocks.append(block.scores_)
        other_blocks.append(block.transform(K[np.ix_(other_idx, fit_idx)]))
    mfa = MFA().fit(fit_blocks)
    return mfa.scores_, mfa.transform(other_blocks)
