#!/usr/bin/env python
"""The class-curve study's arms: the curves' fPCA against ``D+C`` on the development slides (design.md, Pipeline 3-5).

In each outer fold, ``outer_fold`` fits on the training slides ``T`` and predicts the test slides:

1. the outlier filter on ``T`` (``folds.outlier_keep``) leaves ``T'``;
2. ``choose_k`` picks the number of MFA components of the curves by an inner stratified 5-fold on ``T'``;
3. each arm is the cell-type study's logistic regression (``fit_predict``) fitted on its training slides:
   - ``curves``: the first ``k`` columns of the curve space fitted on ``T'`` (``folds.curve_space``);
   - ``curves4``: its first four columns;
   - ``dc``: the 51 ``D+C`` features on ``T'``;
   - ``dc_unfiltered``: ``D+C`` on all of ``T``.

Only the training slides' labels are read: the filter, the curve spaces, the inner folds and the fits never see a test
slide's label.

Stages
------
``development``
    Every outer fold of the 476 development slides: ``make_folds(labels, 20, 5, seed=DEV_FOLD_SEED)``, the cell-type
    improvement phase's development folds, with the inner seed ``1000 + 5 × repeat + fold``. Gate: ``dc_unfiltered``,
    averaged over the repeats, equals that phase's ``improve_dev.npz`` ``baseline:D+C-features@51`` (``np.allclose``,
    ``atol=1e-12``), else ``RuntimeError`` and nothing written. Writes ``results/development.npz`` (every arm's
    out-of-fold probabilities per repeat, the chosen ``k`` and the dropped slides per fold, the folds, the slides) and
    ``results/development.json`` (the run's metadata).

Run from this folder: ``PYTHONPATH=<topocyte checkout at 18e3aa5> <python> run_study.py development``. It reads
``results/gram.npz`` and the cell-type study's saved results, read only, and writes only to ``results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
import warnings
from collections.abc import Sequence

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common as c
import folds

ARMS = ("curves", "curves4", "dc", "dc_unfiltered")  # each arm's test probabilities, n_test × 8
N_REPEATS, N_SPLITS = 20, 5  # the development folds: make_folds(dev labels, N_REPEATS, N_SPLITS, seed=DEV_FOLD_SEED)
INNER_SPLITS = 5  # choose_k's StratifiedKFold
DC_GATE_KEY = "baseline:D+C-features@51"  # the improvement phase's unfiltered D+C probabilities in improve_dev.npz
N_GROUPS = len(c.celltype.GROUP_ORDER)
SOURCES = ("run_study.py", "folds.py", "common.py")  # development.json names their md5s


def fit_predict(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray) -> np.ndarray:
    """The cell-type study's logistic regression fitted on ``Z_train``: ``Z_test``'s probabilities, n_test × 8.

    As the improvement phase's ``fit_logistic`` and its prediction (``improve.py``): features standardized with the
    training mean and SD (an SD of 0 counts as 1), ``LogisticRegression(C=1.0, class_weight="balanced",
    max_iter=5000)`` (multinomial L2, lbfgs), and the probabilities written at ``model.classes_`` into the columns of
    ``GROUP_ORDER``.
    """
    mean, sd = Z_train.mean(axis=0), Z_train.std(axis=0)
    sd[sd == 0] = 1.0
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    model.fit((Z_train - mean) / sd, y_train)
    proba = np.zeros((len(Z_test), N_GROUPS))
    proba[:, model.classes_] = model.predict_proba((Z_test - mean) / sd)
    return proba


def choose_k(grams: dict[str, np.ndarray], train: np.ndarray, labels: np.ndarray, seed: int) -> int:
    """The number of MFA components of the curves, chosen by inner stratified 5-fold cross-validation on ``train``.

    Each inner fold fits the curve space on its training slides and ``fit_predict`` on its first ``k`` columns, for
    every ``k`` of ``K_GRID`` that all the inner folds' columns allow; with fewer than 4 columns, their number is the
    only ``k``. The ``k`` whose out-of-fold probabilities, pooled over the inner folds, have the highest macro AUC
    wins, and ties go to the smaller ``k``. Of the labels, only ``labels[train]`` is read.
    """
    y = labels[train]
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed).split(np.zeros(len(train)), y)
    spaces = [(fit, out, *folds.curve_space(grams, train[fit], train[out])) for fit, out in inner]
    columns = min(Z_fit.shape[1] for _, _, Z_fit, _ in spaces)
    ks = [k for k in c.K_GRID if k <= columns] or [columns]
    pooled = np.zeros((len(ks), len(train), N_GROUPS))
    for fit, out, Z_fit, Z_out in spaces:
        for i, k in enumerate(ks):
            pooled[i, out] = fit_predict(Z_fit[:, :k], y[fit], Z_out[:, :k])
    auc = [c.celltype.macro_auc(proba, y)[0] for proba in pooled]
    return ks[int(np.argmax(auc))]  # the first of equal maxima: the smaller k


def outer_fold(
    grams: dict[str, np.ndarray], DC: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int
) -> dict:
    """One outer fold: the filter on ``train``, ``k`` by ``choose_k`` on the slides kept, and the arms' predictions.

    Keys: ``keep`` (mask over ``train``, False for the slides the filter drops), ``k``, and each of ``ARMS`` (the test
    slides' probabilities, n_test × 8). Of the labels, only ``labels[train]`` is read; ``seed`` seeds the inner folds.
    """
    keep = folds.outlier_keep(grams, train, labels)
    kept = train[keep]
    k = choose_k(grams, kept, labels, seed)
    Z_kept, Z_test = folds.curve_space(grams, kept, test)
    return {
        "keep": keep,
        "k": k,
        "curves": fit_predict(Z_kept[:, :k], labels[kept], Z_test[:, :k]),
        "curves4": fit_predict(Z_kept[:, :4], labels[kept], Z_test[:, :4]),
        "dc": fit_predict(DC[kept], labels[kept], DC[test]),
        "dc_unfiltered": fit_predict(DC[train], labels[train], DC[test]),
    }


def dc_gate(dc_unfiltered: np.ndarray, reference: np.ndarray) -> float:
    """The largest absolute difference of the unfiltered ``D+C`` probabilities from the cell-type study's.

    ``RuntimeError`` unless the shapes agree and ``np.allclose(dc_unfiltered, reference, atol=1e-12)``.
    """
    if dc_unfiltered.shape != reference.shape:
        raise RuntimeError(f"unfiltered D+C has shape {dc_unfiltered.shape}, the cell-type study's {reference.shape}")
    gap = float(np.max(np.abs(dc_unfiltered - reference)))
    if not np.allclose(dc_unfiltered, reference, atol=1e-12):
        raise RuntimeError(f"unfiltered D+C differs from the cell-type study's by up to {gap:.3g}")
    return gap


def _convergence_warnings(caught: list[warnings.WarningMessage]) -> int:
    """How many of the warnings ``caught`` are convergence warnings; the others are shown as usual."""
    n = 0
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            n += 1
        else:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)
    return n


def run_development(inputs: dict, grams: dict[str, np.ndarray]) -> dict:
    """Every outer fold of the development slides, the ``D+C`` gate and ``results/development.npz``; the summary.

    ``development.npz`` holds, the development slides in the study's order:

    - each of ``ARMS``: the out-of-fold probabilities of every repeat (20 × 476 × 8);
    - ``k``: the chosen ``k`` (20 × 5);
    - ``dropped``: dropped by the filter from the training slides of repeat ``r``, fold ``f`` (20 × 5 × 476 bool);
    - ``folds``: each slide's fold in each repeat (20 × 476), and ``slide_ids`` (476).
    """
    labels, dev = inputs["labels"], np.flatnonzero(inputs["dev"])
    split = c.celltype.make_folds(labels[dev], N_REPEATS, N_SPLITS, seed=c.DEV_FOLD_SEED)
    proba = {arm: np.zeros((N_REPEATS, dev.size, N_GROUPS)) for arm in ARMS}
    chosen = np.zeros((N_REPEATS, N_SPLITS), dtype=np.int64)
    dropped = np.zeros((N_REPEATS, N_SPLITS, dev.size), dtype=bool)
    n_convergence = 0
    for r in range(N_REPEATS):
        for f in range(N_SPLITS):
            started = time.perf_counter()
            in_test = split[r] == f
            rows = np.flatnonzero(~in_test)  # the training slides, as rows of the development slides
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fold = outer_fold(grams, inputs["DC"], labels, dev[rows], dev[in_test], seed=1000 + 5 * r + f)
            n_convergence += _convergence_warnings(caught)
            for arm in ARMS:
                proba[arm][r, in_test] = fold[arm]
            chosen[r, f] = fold["k"]
            dropped[r, f, rows[~fold["keep"]]] = True
            print(
                f"repeat {r} fold {f}: {dropped[r, f].sum()} dropped, k = {fold['k']}, "
                f"{time.perf_counter() - started:.1f} s",
                flush=True,
            )
    with np.load(c.CELLTYPE / "results/improve_dev.npz", allow_pickle=False) as improve:
        gap = dc_gate(proba["dc_unfiltered"].mean(axis=0), improve[DC_GATE_KEY])
    print(f"D+C gate passed: largest difference {gap:.3g}", flush=True)
    slide_ids = np.array([inputs["slide_ids"][i] for i in dev])
    np.savez(c.RESULTS / "development.npz", **proba, k=chosen, dropped=dropped, folds=split, slide_ids=slide_ids)
    values, counts = np.unique(chosen, return_counts=True)
    return {
        "description": "every outer fold of the development slides: each arm's probabilities in development.npz",
        "n_slides": int(dev.size),
        "folds": {
            "repeats": N_REPEATS,
            "splits": N_SPLITS,
            "seed": c.DEV_FOLD_SEED,
            "inner_seed": "1000 + 5 * repeat + fold",
        },
        "dc_gate": {
            "reference": f"gtex-liver-celltype-slices/results/improve_dev.npz {DC_GATE_KEY}",
            "atol": 1e-12,
            "max_abs_difference": gap,
        },
        "chosen_k": {str(k): int(n) for k, n in zip(values, counts, strict=True)},
        "dropped_per_fold": dropped.sum(axis=2).tolist(),
        "convergence_warnings": n_convergence,
    }


def load_grams(slide_ids: list[str]) -> dict[str, np.ndarray]:
    """Each class's Gram matrix from ``results/gram.npz``; ``RuntimeError`` if it lists the slides differently."""
    with np.load(c.RESULTS / "gram.npz", allow_pickle=False) as saved:
        if [str(s) for s in saved["slide_ids"]] != slide_ids:
            raise RuntimeError("gram.npz lists the slides differently from spaces.npz")
        return {name: saved[f"gram_{name}"] for name in c.CLASSES}


def main(argv: Sequence[str] | None = None) -> None:
    c.check_topocyte()
    # At the start: the code that runs.
    md5 = {name: hashlib.md5((c.HERE / name).read_bytes()).hexdigest() for name in SOURCES}
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("stage", choices=("development",))
    parser.parse_args(argv)
    started = time.perf_counter()
    inputs = c.load_inputs()
    summary = run_development(inputs, load_grams(inputs["slide_ids"]))
    summary.update(
        {
            "seconds": round(time.perf_counter() - started, 1),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),  # bytes on macOS
            "md5": md5,
            "topocyte_commit": c.TOPOCYTE_COMMIT,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__,
            },
            "platform": platform.platform(),
        }
    )
    (c.RESULTS / "development.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote results/development.npz and .json in {summary['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
