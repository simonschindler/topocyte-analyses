#!/usr/bin/env python
"""Check the class-curve study's saved results by other routes than the code that made them.

Run from this folder after the stages (README.md, Run), with nothing else heavy open:
``PYTHONPATH=<topocyte checkout at 18e3aa5> <python> verify_results.py``. It reads, read only, this study's
``results/``, the pathology study's ``density.npz``, the cohort's cell types (h5py mode "r") and the cell-type study's
``spaces.npz``, ``holdout.csv``, ``composition.npz``, ``improve_dev.npz`` and ``.json`` and ``improve_holdout.npz``.
It runs no stage and fits no classifier: every AUC it computes comes from probabilities the stages saved, so the
holdout is not scored again. It writes ``results/verify.json``: every check with its measured value and tolerance,
the md5 of this script and of the cell-type study's ``common.py`` (imported as ``celltype``), and the sha256 of every
data file it read. Exits with status 1 if a check fails. About seven minutes, most of it the fPCA of one fold.

Checks:

- ``sum_of_minima``, the oracle of the Gram check, against the brute-force double sum ``np.minimum.outer(a, b).sum()``:
  ties, an empty side, random values;
- ``gram.npz``: four finite, exactly symmetric 594 × 594 matrices in the study's slide order; 30 entries per class
  (25 off the diagonal, 5 on it, drawn with ``VERIFY_SEED``) recomputed by ``sum_of_minima`` from the slides' densities;
- the cells of each class at or above each decile, counted by one flat ``np.bincount`` per decile, and by
  ``curves.decile_counts`` (the gram stage's gate), against the cell-type study's ``composition.npz``;
- ``development.npz`` and ``holdout.npz`` against the jsons and each other: slides, labels, the chosen ``k``, the
  dropped slides (never a test slide), probabilities that sum to 1;
- the development folds against ``RepeatedStratifiedKFold`` built here, seeded with the improvement phase's
  ``fold_seed`` (its ``improve_dev.json``), and stratified;
- on one outer fold (repeat 0, fold 0; ``T'`` the training slides the run kept): each block's ``BlockFPCA`` scores of
  ``T'`` and projection of the test slides against topocyte's ``FPCA(eigen_solver="dense")`` fitted on the ``T'``
  curves themselves. Its fit computes its own Gram matrix of those curves, so it rechecks the saved Gram's ``T'``
  block, and its ``transform`` integrates component functions over the bars rather than using the Gram. Then
  ``curve_space``'s MFA scores against an eigendecomposition of the sum of the blocks' weighted, centred Gram matrices;
- the unfiltered ``D+C`` against the improvement phase's probabilities (development averaged over the repeats; holdout);
- ``celltype.auc`` against a count of pairs (Mann–Whitney U) on every arm's probabilities; then every saved AUC (macro,
  per condition, per repeat, the gates) recomputed with it from the saved probabilities;
- the paired differences and their 95% intervals on the pre-registered bootstrap draws (redrawn), and the verdict by
  the rule fixed in advance (design.md, Pipeline 7), written out here again;
- the holdout scored once: ``logs/holdout.log``, ``holdout.npz`` and ``holdout.json`` created in that order within
  one run, each file written once (last modified within a second of its creation) and before the one commit that
  holds ``holdout.json``, which nothing has changed since; ``scored_at`` within that run; ``development.json``, whose
  interval the verdict read, last written before ``scored_at``;
- every script md5 the results name against the script at the commit that holds that version and, where the script
  has not changed since, the working tree; the cell-type study's ``common.py`` unchanged since before this study's
  design.

Not redone: the outlier filter's drops and ``choose_k``'s choices (checked for consistency only), the logistic
regressions (the D+C gates show they reproduce the cell-type study's exactly), the other 99 outer folds and the
holdout fit, and the runs' timings and versions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold
from topocyte.ph import FPCA

import common as c
import curves as cv
import folds

VERIFY_SEED = 20261123  # draws the Gram entries and the oracle's random cases
N_OFF, N_DIAG = 25, 5  # Gram entries checked per class, off and on the diagonal
FOLD = (0, 0)  # the outer fold of the fPCA and MFA checks: repeat 0, fold 0 of the development folds
N_REPEATS, N_SPLITS = 20, 5
N_GROUPS = len(c.celltype.GROUP_ORDER)
ARMS = ("curves", "curves4", "dc", "dc_unfiltered")
COMPARISONS = (("curves", "dc"), ("curves4", "dc"))
PRIMARY = "curves - dc"  # the comparison the verdict reads
DC_GATE_KEY = "baseline:D+C-features@51"  # improve_dev.npz's unfiltered D+C
HOLDOUT_GATE_NAME = "D+C features"  # improve_holdout.npz's unfiltered D+C, e<index>__proba
DESIGN_COMMIT = "e07c262"  # design.md, committed before any result
TOL_GRAM = 1e-12  # relative: sum_of_minima and topocyte's Gram sum the same minima in other orders
TOL_FPCA = 1e-7  # relative to the block's largest score: component integrals against Gram projections
TOL_MFA = 1e-8  # relative to each component's largest fitted score: eigh of the summed Gram against an SVD
TOL_EXACT = 1e-12  # AUCs, intervals and gates recomputed from the same saved numbers
WRITE_WINDOW = 10.0  # seconds: the holdout stage serialized holdout.json (scored_at) and wrote it within its run

# Every md5 the results name: (result file, its keys down to the md5, script, the commit holding that version, whether
# the working tree's script is still that version). The development run used run_study.py as committed in 727667e;
# the readouts and the holdout ran with a58c812's, the current file.
PROVENANCE = (
    ("spike.json", ("script_md5",), "curves.py", "64919fe", True),
    ("gram.json", ("script_md5",), "curves.py", "64919fe", True),
    ("development.json", ("md5", "run_study.py"), "run_study.py", "727667e", False),
    ("development.json", ("md5", "folds.py"), "folds.py", "c050e34", True),
    ("development.json", ("md5", "common.py"), "common.py", "ba2dcca", True),
    ("development.json", ("readouts", "md5", "run_study.py"), "run_study.py", "a58c812", True),
    ("development.json", ("readouts", "md5", "folds.py"), "folds.py", "c050e34", True),
    ("development.json", ("readouts", "md5", "common.py"), "common.py", "ba2dcca", True),
    ("holdout.json", ("md5", "run_study.py"), "run_study.py", "a58c812", True),
    ("holdout.json", ("md5", "folds.py"), "folds.py", "c050e34", True),
    ("holdout.json", ("md5", "common.py"), "common.py", "ba2dcca", True),
)

checks: list[dict] = []
read_files: list[Path] = []  # the data files read, for their sha256


def record(name: str, ok: bool, measured, tolerance=None) -> bool:
    checks.append({"check": name, "ok": bool(ok), "measured": measured, "tolerance": tolerance})
    tail = "" if tolerance is None else f" (tolerance {tolerance})"
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {json.dumps(measured, ensure_ascii=False)}{tail}", flush=True)
    return bool(ok)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    read_files.append(path)
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path) -> str:
    """A path relative to the repository, or to the home folder (the cohort)."""
    if path.is_relative_to(c.HERE.parent):
        return str(path.relative_to(c.HERE.parent))
    return f"~/{path.relative_to(Path.home())}"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(c.HERE), *args], capture_output=True, check=False)


def worst(values) -> float:
    return float(max(values, default=0.0))


# ---------------------------------------------------------------- the Gram matrices and the decile counts


def check_oracle() -> None:
    """``sum_of_minima`` against ``np.minimum.outer(a, b).sum()``, exact where the values are integers."""
    rng = np.random.default_rng([VERIFY_SEED, 0])
    cases = {
        "ties": (np.array([1.0, 2.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0, 0.5, 3.0])),
        "a empty": (np.array([]), np.array([1.0, 4.0])),
        "b empty": (np.array([3.0, 1.0]), np.array([])),
        "both empty": (np.array([]), np.array([])),
        "integer values, many ties": (rng.integers(0, 40, 300).astype(float), rng.integers(0, 40, 170).astype(float)),
        "uniform": (rng.uniform(1, 5000, 300), rng.uniform(1, 5000, 170)),
    }
    measured, ok = {}, True
    for name, (a, b) in cases.items():
        brute, fast = float(np.minimum.outer(a, b).sum()), cv.sum_of_minima(a, b)
        if name == "uniform":
            measured[name] = abs(fast / brute - 1)
            ok &= measured[name] <= TOL_GRAM
        else:  # integer-valued sums are exact in float64
            measured[name] = [fast, brute]
            ok &= fast == brute
    record("sum_of_minima, the Gram check's oracle, against np.minimum.outer(a, b).sum()", ok, measured, TOL_GRAM)


def check_gram(grams: dict[str, np.ndarray], gram_ids: list[str], inputs: dict, rho, classes) -> None:
    n = len(inputs["slide_ids"])
    form = all(G.shape == (n, n) and np.isfinite(G).all() and np.array_equal(G, G.T) for G in grams.values())
    record(
        f"gram.npz: {len(grams)} finite, exactly symmetric {n} × {n} matrices, its slides in the study's order",
        form and gram_ids == inputs["slide_ids"] and list(grams) == list(c.CLASSES),
        {name: list(G.shape) for name, G in grams.items()},
    )
    rng = np.random.default_rng([VERIFY_SEED, 1])
    bounds = inputs["bounds"]
    errors = {}
    for k, name in enumerate(c.CLASSES):
        pairs = [tuple(rng.choice(n, 2, replace=False)) for _ in range(N_OFF)]
        pairs += [(i, i) for i in rng.choice(n, N_DIAG, replace=False)]
        densities = {}
        for s in {s for pair in pairs for s in pair}:
            start, end = bounds[s], bounds[s + 1]
            densities[s] = rho[start:end][classes[start:end] == k]
        errors[name] = worst(
            abs(grams[name][i, j] / cv.sum_of_minima(densities[i], densities[j]) - 1) for i, j in pairs
        )
    record(
        f"gram.npz: {N_OFF + N_DIAG} entries per class ({N_OFF} off the diagonal, {N_DIAG} on it) against "
        "sum_of_minima of the two slides' densities (largest relative difference)",
        worst(errors.values()) <= TOL_GRAM,
        errors,
        TOL_GRAM,
    )


def check_deciles(inputs: dict, rho, classes) -> None:
    bounds, deciles = inputs["bounds"], inputs["deciles"]
    n, n_classes = len(bounds) - 1, len(c.CLASSES)
    composition = load_npz(c.CELLTYPE / "results/composition.npz")
    kept = composition["kept_by_group"]
    code = np.repeat(np.arange(n, dtype=np.int32) * n_classes, np.diff(bounds)) + classes  # slide × class, per cell
    flat = np.stack([np.bincount(code[rho >= d], minlength=n * n_classes).reshape(n, n_classes) for d in deciles])
    del code
    gate = cv.decile_counts(rho, classes, bounds, deciles)
    record(
        "cells of each class at or above each decile: one flat bincount per decile, and curves.decile_counts (the "
        "gram stage's gate), against the cell-type study's composition.npz kept_by_group",
        [str(s) for s in composition["slide_ids"]] == inputs["slide_ids"]
        and np.array_equal(flat, kept)
        and np.array_equal(gate, kept),
        {
            "cells at decile 0": int(flat[0].sum()),
            "entries differing, bincount": int(np.count_nonzero(flat != kept)),
            "entries differing, decile_counts": int(np.count_nonzero(gate != kept)),
        },
    )


# ---------------------------------------------------------------- the saved arrays and the folds


def check_saved_arrays(inputs: dict, saved_dev: dict, saved_hold: dict, dev_json: dict, hold_json: dict) -> None:
    ids, labels = inputs["slide_ids"], inputs["labels"]
    dev, hold = np.flatnonzero(inputs["dev"]), np.flatnonzero(inputs["holdout"])
    split, dropped, k = saved_dev["folds"], saved_dev["dropped"], saved_dev["k"]
    in_test = split[:, None, :] == np.arange(N_SPLITS)[None, :, None]  # (repeat, fold, slide): a test slide there
    values, counts = np.unique(k, return_counts=True)
    row_sums = [np.abs(saved[arm].sum(axis=-1) - 1).max() for saved in (saved_dev, saved_hold) for arm in ARMS]
    dev_ok = (
        [str(s) for s in saved_dev["slide_ids"]] == [ids[i] for i in dev]
        and all(saved_dev[arm].shape == (N_REPEATS, dev.size, N_GROUPS) for arm in ARMS)
        and k.shape == (N_REPEATS, N_SPLITS)
        and set(values.tolist()) <= set(c.K_GRID)
        and {str(v): int(n) for v, n in zip(values, counts, strict=True)} == dev_json["chosen_k"]
        and dropped.shape == (N_REPEATS, N_SPLITS, dev.size)
        and dropped.sum(axis=2).tolist() == dev_json["dropped_per_fold"]
        and not (dropped & in_test).any()
    )
    y = labels[hold]
    per_group = {
        g: int(np.sum(saved_hold["dropped"] & (labels[dev] == i))) for i, g in enumerate(c.celltype.GROUP_ORDER)
    }
    hold_ok = (
        [str(s) for s in saved_hold["slide_ids"]] == [ids[i] for i in hold]
        and [str(s) for s in saved_hold["dev_slide_ids"]] == [ids[i] for i in dev]
        and np.array_equal(saved_hold["labels"], y)
        and all(saved_hold[arm].shape == (hold.size, N_GROUPS) for arm in ARMS)
        and int(saved_hold["k"]) == hold_json["fit"]["chosen_k"]
        and saved_hold["dropped"].shape == (dev.size,)
        and int(saved_hold["dropped"].sum()) == hold_json["fit"]["dropped"]
        and per_group == hold_json["fit"]["dropped_per_group"]
        and dict(zip(c.celltype.GROUP_ORDER, np.bincount(y, minlength=N_GROUPS).tolist(), strict=True))
        == hold_json["holdout_per_group"]
    )
    record(
        "development.npz and holdout.npz agree with the jsons: slides, labels, the chosen k, the drops (never a test "
        "slide of their fold), probabilities summing to 1",
        dev_ok and hold_ok and worst(row_sums) <= TOL_EXACT,
        {
            "development": bool(dev_ok),
            "holdout": bool(hold_ok),
            "drops of test slides": int((dropped & in_test).sum()),
            "largest |row sum - 1|": worst(row_sums),
        },
        TOL_EXACT,
    )


def check_folds(inputs: dict, saved_dev: dict) -> None:
    dev = np.flatnonzero(inputs["dev"])
    y = inputs["labels"][dev]
    path = c.CELLTYPE / "results/improve_dev.json"
    read_files.append(path)
    improve = json.loads(path.read_text())
    expected = np.empty((N_REPEATS, dev.size), dtype=np.int64)
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=improve["fold_seed"])
    for i, (_, test) in enumerate(splitter.split(np.zeros(dev.size), y)):
        expected[i // N_SPLITS, test] = i % N_SPLITS
    saved = saved_dev["folds"]
    spread = max(
        int(np.ptp(np.bincount(saved[r, y == g], minlength=N_SPLITS))) for r in range(N_REPEATS) for g in np.unique(y)
    )
    record(
        f"development folds: RepeatedStratifiedKFold ({N_SPLITS} splits, {N_REPEATS} repeats) built here with the "
        "improvement phase's fold_seed from its improve_dev.json, each repeat a partition that spreads every group "
        "evenly",
        np.array_equal(saved, expected)
        and improve["fold_seed"] == c.DEV_FOLD_SEED
        and improve["dev_slides"] == dev.size
        and spread <= 1,
        {
            "fold_seed": improve["fold_seed"],
            "development slides": improve["dev_slides"],
            "entries differing": int(np.count_nonzero(saved != expected)),
            "largest spread of a group over the folds": spread,
        },
    )


# ---------------------------------------------------------------- one outer fold: the fPCA and the MFA


def fold_slides(inputs: dict, saved_dev: dict) -> tuple[np.ndarray, np.ndarray]:
    """``T'`` and the test slides of ``FOLD``, positions in the study's slide order, from the saved folds and drops."""
    dev = np.flatnonzero(inputs["dev"])
    r, f = FOLD
    in_test = saved_dev["folds"][r] == f
    train = np.flatnonzero(~in_test)
    return dev[train[~saved_dev["dropped"][r, f, train]]], dev[in_test]


def check_block_fpca(grams: dict, kept: np.ndarray, test: np.ndarray, inputs: dict, rho, classes) -> None:
    bounds = inputs["bounds"]
    errors, shapes = {}, True
    for k, name in enumerate(c.CLASSES):
        t0 = time.perf_counter()
        curves = {}
        for s in np.concatenate([kept, test]):
            start, end = bounds[s], bounds[s + 1]
            curves[s] = cv.class_curve(rho[start:end][classes[start:end] == k])
        for normalization in c.NORMALIZATIONS:
            oracle = FPCA(
                dim=0, n_components=c.N_COMPONENTS, normalization=normalization, eigen_solver="dense", n_jobs=-1
            ).fit([curves[s] for s in kept])
            fitted, projected = oracle.result_.scores, oracle.transform([curves[s] for s in test])
            K = folds.analysis_kernel(grams[name], normalization)
            block = folds.BlockFPCA().fit(K[np.ix_(kept, kept)])
            mine_fit, mine_test = block.scores_, block.transform(K[np.ix_(test, kept)])
            shapes &= mine_fit.shape == fitted.shape == (kept.size, c.N_COMPONENTS)
            shapes &= mine_test.shape == projected.shape == (test.size, c.N_COMPONENTS)
            scale = np.abs(fitted).max()
            errors[f"{name} {normalization}"] = [
                float(np.abs(mine_fit - fitted).max() / scale),
                float(np.abs(mine_test - projected).max() / scale),
            ]
        print(f"  {name}: both fPCAs of the fold checked in {time.perf_counter() - t0:.0f} s", flush=True)
        del curves
    record(
        f"repeat {FOLD[0]}, fold {FOLD[1]} ({kept.size} slides kept, {test.size} test slides): each block's "
        "BlockFPCA scores of the kept slides and projection of the test slides against topocyte's "
        "FPCA(eigen_solver='dense') fitted on the kept slides' curves and its transform ([fit, projection], largest "
        "difference over the block's largest score)",
        shapes and worst(e for pair in errors.values() for e in pair) <= TOL_FPCA,
        errors,
        TOL_FPCA,
    )


def check_mfa(grams: dict, kept: np.ndarray, test: np.ndarray) -> None:
    Z_fit, Z_test = folds.curve_space(grams, kept, test)
    fit_blocks, test_blocks = [], []
    for name in c.CLASSES:
        for normalization in c.NORMALIZATIONS:
            K = folds.analysis_kernel(grams[name], normalization)
            block = folds.BlockFPCA().fit(K[np.ix_(kept, kept)])
            fit_blocks.append(block.scores_)
            test_blocks.append(block.transform(K[np.ix_(test, kept)]))
    weights = [1 / np.sqrt(b[:, 0].var()) for b in fit_blocks]
    means = [w * b.mean(axis=0) for w, b in zip(weights, fit_blocks, strict=True)]
    centred = [w * b - m for w, b, m in zip(weights, fit_blocks, means, strict=True)]
    gram = sum(x @ x.T for x in centred)  # the blocks' weighted, centred Gram matrices, summed
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    n_keep = min(folds.MFA_COMPONENTS, int(np.count_nonzero(eigenvalues / eigenvalues.sum() > folds.MFA_RTOL)))
    eigenvalues, vectors = eigenvalues[:n_keep], vectors[:, :n_keep]
    vectors = vectors * np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(n_keep)])
    scores = vectors * np.sqrt(eigenvalues)
    cross = sum((w * b - m) @ x.T for w, b, m, x in zip(weights, test_blocks, means, centred, strict=True))
    projected = cross @ vectors / np.sqrt(eigenvalues)
    same_shape = scores.shape == Z_fit.shape and projected.shape == Z_test.shape
    scale = np.abs(Z_fit).max(axis=0)
    fit_error = float((np.abs(scores - Z_fit).max(axis=0) / scale).max()) if same_shape else float("inf")
    test_error = float((np.abs(projected - Z_test).max(axis=0) / scale).max()) if same_shape else float("inf")
    record(
        f"repeat {FOLD[0]}, fold {FOLD[1]}: curve_space's MFA scores of the kept and the test slides against an "
        f"eigendecomposition of the sum of the {len(fit_blocks)} blocks' weighted, centred Gram matrices (largest "
        "difference over the component's largest fitted score)",
        same_shape and max(fit_error, test_error) <= TOL_MFA,
        {"components": [int(Z_fit.shape[1]), n_keep], "kept slides": fit_error, "test slides": test_error},
        TOL_MFA,
    )


# ---------------------------------------------------------------- the gates, the AUCs, the intervals and the verdict


def pair_count_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive > negative) + P(tie) / 2, counted over every pair (the Mann–Whitney U over the pairs)."""
    greater = np.count_nonzero(positive[:, None] > negative[None, :])
    ties = np.count_nonzero(positive[:, None] == negative[None, :])
    return (greater + 0.5 * ties) / (positive.size * negative.size)


def macro(proba: np.ndarray, y: np.ndarray, auc=None) -> tuple[float, list[float]]:
    """Each condition's one-vs-rest AUC among the condition slides (unclassified unscored), and their mean."""
    auc = auc or c.celltype.auc
    scored = y < len(c.celltype.CONDITIONS)
    per = [auc(proba[scored & (y == g), g], proba[scored & (y != g), g]) for g in range(len(c.celltype.CONDITIONS))]
    return float(np.mean(per)), per


def check_gates(inputs: dict, saved_dev: dict, saved_hold: dict, dev_json: dict, hold_json: dict) -> None:
    hold = np.flatnonzero(inputs["holdout"])
    improve_dev = load_npz(c.CELLTYPE / "results/improve_dev.npz")
    gap_dev = float(np.abs(saved_dev["dc_unfiltered"].mean(axis=0) - improve_dev[DC_GATE_KEY]).max())
    improve = load_npz(c.CELLTYPE / "results/improve_holdout.npz")
    reference = improve[f"e{[str(n) for n in improve['names']].index(HOLDOUT_GATE_NAME)}__proba"]
    gap_hold = float(np.abs(saved_hold["dc_unfiltered"] - reference).max())
    y = inputs["labels"][hold]
    same = np.array_equal(improve["holdout_rows"], hold) and np.array_equal(improve["y_holdout"], y)
    record(
        "the D+C gates: the unfiltered D+C, averaged over the repeats on the development slides, equals the "
        "improvement phase's improve_dev.npz; on the holdout, its improve_holdout.npz (same slides and groups)",
        same
        and max(gap_dev, gap_hold) <= TOL_EXACT
        and gap_dev == dev_json["dc_gate"]["max_abs_difference"]
        and gap_hold == hold_json["gate"]["max_abs_difference"],
        {"development": gap_dev, "holdout": gap_hold},
        TOL_EXACT,
    )


def check_auc_oracle(saved_dev: dict, saved_hold: dict, y_dev: np.ndarray, y_hold: np.ndarray) -> None:
    errors = []
    for arm in ARMS:
        for proba, y in ((saved_dev[arm].mean(axis=0), y_dev), (saved_hold[arm], y_hold)):
            mine, pairs = macro(proba, y)[1], macro(proba, y, pair_count_auc)[1]
            errors += [abs(a - b) for a, b in zip(mine, pairs, strict=True)]
    record(
        "celltype.auc (rank formula) against a count of every positive-negative pair, each condition of every arm "
        "(development averaged over the repeats, and holdout)",
        worst(errors) <= TOL_EXACT,
        {"AUCs": len(errors), "largest difference": worst(errors)},
        TOL_EXACT,
    )


def check_development_aucs(saved_dev: dict, y: np.ndarray, readouts: dict) -> None:
    errors = []
    for arm in ARMS:
        row = readouts["arms"][arm]
        average, per = macro(saved_dev[arm].mean(axis=0), y)
        repeats = [macro(p, y)[0] for p in saved_dev[arm]]
        errors.append(abs(average - row["macro_auc"]))
        errors += [abs(a - b) for a, b in zip(per, row["auc_per_condition"].values(), strict=True)]
        errors += [abs(a - b) for a, b in zip(repeats, row["auc_per_repeat"], strict=True)]
        errors += [abs(np.mean(repeats) - row["auc_repeat_mean"]), abs(np.std(repeats, ddof=1) - row["auc_repeat_sd"])]
        if arm == "dc_unfiltered":
            errors.append(abs(average - readouts["gate"]["macro_auc"]))
    keys = all(list(readouts["arms"][arm]["auc_per_condition"]) == list(c.celltype.CONDITIONS) for arm in ARMS)
    record(
        "development AUCs from development.npz by celltype.auc: each arm's macro and per-condition AUCs of the "
        "probabilities averaged over the repeats, each repeat's macro AUC, their mean and SD, and the gate",
        keys and list(readouts["arms"]) == list(ARMS) and worst(errors) <= TOL_EXACT,
        {"AUCs": len(errors), "largest difference": float(worst(errors))},
        TOL_EXACT,
    )


def check_holdout_aucs(saved_hold: dict, y: np.ndarray, hold_json: dict) -> None:
    errors = []
    for arm in ARMS:
        row = hold_json["arms"][arm]
        average, per = macro(saved_hold[arm], y)
        errors.append(abs(average - row["macro_auc"]))
        errors += [abs(a - b) for a, b in zip(per, row["auc_per_condition"].values(), strict=True)]
        if arm == "dc_unfiltered":
            errors.append(abs(average - hold_json["gate"]["macro_auc"]))
    record(
        "holdout AUCs from holdout.npz by celltype.auc: each arm's macro and per-condition AUCs, and the gate",
        list(hold_json["arms"]) == list(ARMS) and worst(errors) <= TOL_EXACT,
        {"AUCs": len(errors), "largest difference": float(worst(errors))},
        TOL_EXACT,
    )


def intervals(proba: dict[str, np.ndarray], y: np.ndarray, bootstrap: dict) -> dict[str, list[float]]:
    """Each comparison's macro AUC difference and its 2.5 and 97.5 percentiles over the redrawn bootstrap."""
    draws = c.celltype.stratified_bootstrap(y, bootstrap["draws"], np.random.default_rng(bootstrap["seed"]))
    out = {}
    for arm, reference in COMPARISONS:
        diff = macro(proba[arm], y)[0] - macro(proba[reference], y)[0]
        boot = [macro(proba[arm][rows], y[rows])[0] - macro(proba[reference][rows], y[rows])[0] for rows in draws]
        out[f"{arm} - {reference}"] = [diff, *np.percentile(boot, [2.5, 97.5]).tolist()]
    return out


def rule(dev_lo: float, dev_hi: float, holdout_diff: float) -> str:
    """The decision rule of design.md (Pipeline 7), on curves at the chosen k minus D+C."""
    if dev_lo > 0:
        return "beats D+C" if holdout_diff > 0 else "not confirmed"
    return "below D+C" if dev_hi < 0 else "level with D+C"


def check_intervals_and_verdict(saved_dev: dict, saved_hold: dict, y_dev, y_hold, dev_json: dict, hold_json) -> None:
    readouts = dev_json["readouts"]
    settings = [readouts["bootstrap"], hold_json["bootstrap"]]
    pre_registered = all(b == {"draws": c.N_BOOT, "seed": c.BOOT_SEED, "stratified_by": "group"} for b in settings)
    mine = {
        "development": intervals({arm: saved_dev[arm].mean(axis=0) for arm in ARMS}, y_dev, readouts["bootstrap"]),
        "holdout": intervals({arm: saved_hold[arm] for arm in ARMS}, y_hold, hold_json["bootstrap"]),
    }
    saved = {"development": readouts["comparisons"], "holdout": hold_json["comparisons"]}
    errors = [
        abs(a - b)
        for where in mine
        for name, values in mine[where].items()
        for a, b in zip(values, [saved[where][name]["difference"], *saved[where][name]["interval"]], strict=True)
    ]
    record(
        "paired differences and 95% intervals from the saved probabilities on the pre-registered bootstrap draws "
        f"({c.N_BOOT}, stratified by group, seed {c.BOOT_SEED}), redrawn ([difference, 2.5%, 97.5%])",
        pre_registered and worst(errors) <= TOL_EXACT,
        {where: {name: [round(v, 6) for v in values] for name, values in rows.items()} for where, rows in mine.items()}
        | {"largest difference": worst(errors)},
        TOL_EXACT,
    )
    _, lo, hi = mine["development"][PRIMARY]
    holdout_diff = mine["holdout"][PRIMARY][0]
    v = hold_json["verdict"]
    record(
        "verdict: the rule fixed in advance on the recomputed development interval and holdout difference of curves "
        "at the chosen k minus D+C, against holdout.json's, which read development.json's interval",
        rule(lo, hi, holdout_diff) == v["verdict"]
        and v["development_interval"] == readouts["comparisons"][PRIMARY]["interval"]
        and v["holdout_difference"] == hold_json["comparisons"][PRIMARY]["difference"],
        {"development interval": [lo, hi], "holdout difference": holdout_diff, "verdict": rule(lo, hi, holdout_diff)},
    )


# ---------------------------------------------------------------- the holdout once, and the code


def check_holdout_once(hold_json: dict) -> None:
    """The files of one holdout run, committed as written: the log, the npz and the json created in that order within
    one run, each file written once (last modified within a second of its creation), before the commit that added
    the json, which nothing has changed since; development.json, whose interval the verdict read, last written
    before the holdout was scored."""

    def stamp(seconds: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))

    commits = git("log", "--format=%h %ct", "--", "results/holdout.json").stdout.decode().splitlines()
    committed_at = float(commits[-1].split()[1]) if commits else 0.0
    unchanged = git("diff", "--quiet", "HEAD", "--", "results/holdout.json").returncode == 0
    log, npz, out = (
        path.stat() for path in (c.LOGS / "holdout.log", c.RESULTS / "holdout.npz", c.RESULTS / "holdout.json")
    )
    development = (c.RESULTS / "development.json").stat()
    scored = time.mktime(time.strptime(hold_json["scored_at"], "%Y-%m-%d %H:%M:%S"))
    record(
        "the holdout scored once: logs/holdout.log, holdout.npz and holdout.json created in that order within one "
        "run, each file last modified within a second of its creation, scored_at within that run, before the one "
        "commit that holds holdout.json, unchanged since; development.json last written before scored_at",
        len(commits) == 1
        and unchanged
        and all(s.st_mtime - s.st_birthtime < 1.0 for s in (npz, out))
        and log.st_birthtime <= npz.st_birthtime <= out.st_birthtime <= log.st_birthtime + WRITE_WINDOW
        and 0 <= out.st_birthtime - scored <= WRITE_WINDOW
        and out.st_birthtime <= committed_at
        and development.st_mtime < scored,
        {
            "commits": [line.split()[0] for line in commits],
            "committed at": stamp(committed_at),
            "unchanged since": unchanged,
            "holdout.log created": stamp(log.st_birthtime),
            "holdout.npz created, modified": [stamp(npz.st_birthtime), stamp(npz.st_mtime)],
            "holdout.json created, modified": [stamp(out.st_birthtime), stamp(out.st_mtime)],
            "scored_at": hold_json["scored_at"],
            "development.json modified": stamp(development.st_mtime),
        },
        f"{WRITE_WINDOW:.0f} s",
    )


def check_md5s(results: dict[str, dict]) -> None:
    rows, ok = [], True
    for result, keys, script, commit, current in PROVENANCE:
        named = results[result]
        for key in keys:
            named = named[key]
        shown = git("show", f"{commit}:./{script}")
        at_commit = md5(shown.stdout) if shown.returncode == 0 else None
        now = md5((c.HERE / script).read_bytes())
        ok &= at_commit == named and (now == named) == current
        rows.append(
            [f"{result}{' readouts' if keys[0] == 'readouts' else ''}", script, named[:8], commit, now == named]
        )
    record(
        "the script md5s the results name: each equals the script at the commit that holds that version, and the "
        "working tree's where the script has not changed since ([result, script, md5, commit, working tree])",
        ok,
        rows,
    )


def check_celltype_common() -> str:
    path = c.CELLTYPE / "common.py"
    last = git("log", "-1", "--format=%h", "--", str(path)).stdout.decode().strip()
    before = git("merge-base", "--is-ancestor", last, DESIGN_COMMIT).returncode == 0
    unchanged = git("diff", "--quiet", "HEAD", "--", str(path)).returncode == 0
    digest = md5(path.read_bytes())
    record(
        "the cell-type study's common.py (celltype): last committed before this study's design and unchanged since",
        bool(last) and before and unchanged,
        {"md5": digest, "last commit": last, "design commit": DESIGN_COMMIT},
    )
    return digest


def main() -> None:
    c.check_topocyte()
    script_md5 = md5(Path(__file__).read_bytes())
    started = time.perf_counter()
    inputs = c.load_inputs()
    read_files.extend([c.CELLTYPE / "results/spaces.npz", c.CELLTYPE / "results/holdout.csv", c.SIB_DENSITY])
    check_oracle()

    saved_gram = load_npz(c.RESULTS / "gram.npz")
    grams = {name: saved_gram[f"gram_{name}"] for name in c.CLASSES}
    with np.load(c.SIB_DENSITY, allow_pickle=False) as density:
        rho = density["rho"]
    classes, _ = c.celltype.read_cell_groups()
    read_files.append(c.celltype.H5AD)
    check_gram(grams, [str(s) for s in saved_gram["slide_ids"]], inputs, rho, classes)
    check_deciles(inputs, rho, classes)

    saved_dev = load_npz(c.RESULTS / "development.npz")
    saved_hold = load_npz(c.RESULTS / "holdout.npz")
    names = ("spike.json", "gram.json", "development.json", "holdout.json")
    results = {name: json.loads((c.RESULTS / name).read_text()) for name in names}
    dev_json, hold_json = results["development.json"], results["holdout.json"]
    check_saved_arrays(inputs, saved_dev, saved_hold, dev_json, hold_json)
    check_folds(inputs, saved_dev)

    kept, test = fold_slides(inputs, saved_dev)
    check_block_fpca(grams, kept, test, inputs, rho, classes)
    del rho, classes
    check_mfa(grams, kept, test)

    y_dev = inputs["labels"][inputs["dev"]]
    y_hold = inputs["labels"][inputs["holdout"]]
    check_gates(inputs, saved_dev, saved_hold, dev_json, hold_json)
    check_auc_oracle(saved_dev, saved_hold, y_dev, y_hold)
    check_development_aucs(saved_dev, y_dev, dev_json["readouts"])
    check_holdout_aucs(saved_hold, y_hold, hold_json)
    check_intervals_and_verdict(saved_dev, saved_hold, y_dev, y_hold, dev_json, hold_json)
    check_holdout_once(hold_json)
    check_md5s(results)
    celltype_md5 = check_celltype_common()

    sha = {display(path): sha256(path) for path in dict.fromkeys(read_files)}  # in the order read, each once
    out = {
        "all_ok": all(check["ok"] for check in checks),
        "checks": checks,
        "script_md5": script_md5,
        "celltype_common_md5": celltype_md5,
        "sha256": sha,
        "seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),  # bytes on macOS
        "topocyte_commit": c.TOPOCYTE_COMMIT,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "h5py": h5py.__version__,
        },
        "platform": platform.platform(),
    }
    (c.RESULTS / "verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{sum(ch['ok'] for ch in checks)}/{len(checks)} checks pass; {out['seconds']:.0f} s", flush=True)
    sys.exit(0 if out["all_ok"] else 1)


if __name__ == "__main__":
    main()
