#!/usr/bin/env python
"""The class-curve study's arms: the curves' fPCA against ``D+C``, its readouts and verdict (design.md, Pipeline 3-7).

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
    ``rtol=0``, ``atol=1e-12``), else ``RuntimeError`` and nothing written. Writes ``results/development.npz``
    (every arm's out-of-fold probabilities per repeat, the chosen ``k`` and the dropped slides per fold, the folds,
    the slides) and ``results/development.json`` (the run's metadata).
``readouts``
    From ``results/development.npz``, each arm's macro AUC over the seven conditions (unclassified unscored) and
    per-condition AUCs, from its probabilities averaged over the repeats, and each repeat's macro AUC (the spread);
    the paired differences ``curves`` − ``dc`` and ``curves4`` − ``dc`` with their 95% intervals over ``N_BOOT``
    stratified bootstrap draws of the development slides (``BOOT_SEED``). Gate: ``dc_unfiltered`` reaches the
    improvement phase's development macro AUC, ``DEV_GATE_AUC``. Adds them to ``results/development.json`` as
    ``readouts``, its metadata kept.
``holdout``
    Scored once: refuses if ``results/holdout.json`` exists. ``outer_fold`` with the 476 development slides as
    training slides and the 118 holdout slides as test slides (filter, ``choose_k`` with ``HOLDOUT_SEED``, curve
    space and the four arms, the holdout labels hidden), then the gates: the unfiltered ``D+C`` equals the
    improvement phase's ``improve_holdout.npz`` ``D+C features`` (``dc_gate``) and reaches its macro AUC,
    ``HOLDOUT_GATE_AUC``. Writes ``results/holdout.npz`` (the probabilities, the chosen ``k``, the development slides
    dropped) and then ``results/holdout.json``: the AUCs, the paired differences with intervals over the holdout
    slides, and ``verdict`` from the development interval of ``curves`` − ``dc`` and the holdout difference.

Run from this folder, the stages in this order:
``PYTHONPATH=<topocyte checkout at 18e3aa5> <python> run_study.py development|readouts|holdout``. They read
``results/gram.npz`` and the cell-type study's saved results, read only, and write only to ``results/``.
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
HOLDOUT_GATE_NAME = "D+C features"  # its entry in improve_holdout.npz's names: e<index>__proba
DEV_GATE_AUC, HOLDOUT_GATE_AUC = 0.764, 0.774  # the improvement phase's unfiltered D+C macro AUCs, to 3 decimals
HOLDOUT_SEED = 999  # choose_k's inner folds when the development slides are fitted for the holdout
COMPARISONS = (("curves", "dc"), ("curves4", "dc"))  # paired differences against D+C fitted on the cleaned slides
PRIMARY = "curves - dc"  # the comparison the verdict reads; curves4 - dc is reported beside it, with no verdict
N_GROUPS = len(c.celltype.GROUP_ORDER)
SOURCES = ("run_study.py", "folds.py", "common.py")  # each stage's json names their md5s


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

    ``RuntimeError`` unless the shapes agree and ``np.allclose(dc_unfiltered, reference, rtol=0, atol=1e-12)``: an
    absolute tolerance alone, since ``np.allclose``'s default ``rtol=1e-5`` would pass a difference of 5e-6 at 0.5.
    """
    if dc_unfiltered.shape != reference.shape:
        raise RuntimeError(f"unfiltered D+C has shape {dc_unfiltered.shape}, the cell-type study's {reference.shape}")
    gap = float(np.max(np.abs(dc_unfiltered - reference)))
    if not np.allclose(dc_unfiltered, reference, rtol=0, atol=1e-12):
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
            "rtol": 0,
            "atol": 1e-12,
            "max_abs_difference": gap,
        },
        "chosen_k": {str(k): int(n) for k, n in zip(values, counts, strict=True)},
        "dropped_per_fold": dropped.sum(axis=2).tolist(),
        "convergence_warnings": n_convergence,
    }


def paired_interval(
    p_a: np.ndarray, p_b: np.ndarray, labels: np.ndarray, draws: np.ndarray
) -> tuple[float, float, float]:
    """The macro AUC of ``p_a`` minus that of ``p_b``, and the 2.5 and 97.5 percentiles of the difference over ``draws``.

    ``p_a`` and ``p_b`` are two arms' probabilities of the same slides (n × 8; on the development slides, averaged over
    the repeats), ``labels`` the slides' groups, and ``draws`` the bootstrap's index arrays (``stratified_bootstrap``):
    each draw scores both arms on the same resampled slides.
    """
    macro_auc = c.celltype.macro_auc
    diff = macro_auc(p_a, labels)[0] - macro_auc(p_b, labels)[0]
    boot = [macro_auc(p_a[rows], labels[rows])[0] - macro_auc(p_b[rows], labels[rows])[0] for rows in draws]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff), float(lo), float(hi)


def verdict(dev_lo: float, dev_hi: float, holdout_diff: float) -> str:
    """The decision rule fixed before any result (design.md, Pipeline 7), on the curves at the chosen ``k`` minus ``D+C``.

    - ``"beats D+C"``: the development interval ``[dev_lo, dev_hi]`` lies above 0 and the holdout difference is
      positive;
    - ``"not confirmed"``: the development interval lies above 0, and the holdout difference is not positive;
    - ``"level with D+C"``: the development interval contains 0, an end at 0 included;
    - ``"below D+C"``: the development interval lies below 0.

    ``ValueError`` for what the rule cannot read: an interval whose ends are out of order or NaN, or a NaN difference.
    """
    if not dev_lo <= dev_hi or np.isnan(holdout_diff):
        raise ValueError(f"the rule cannot read the interval [{dev_lo}, {dev_hi}] with the difference {holdout_diff}")
    if dev_lo > 0:
        return "beats D+C" if holdout_diff > 0 else "not confirmed"
    return "below D+C" if dev_hi < 0 else "level with D+C"


def auc_readout(proba: np.ndarray, labels: np.ndarray) -> dict:
    """The macro AUC over the seven conditions (unclassified unscored) and each condition's one-vs-rest AUC."""
    macro, per = c.celltype.macro_auc(proba, labels)
    return {"macro_auc": macro, "auc_per_condition": dict(zip(c.celltype.CONDITIONS, per.tolist(), strict=True))}


def compare(proba: dict[str, np.ndarray], labels: np.ndarray, draws: np.ndarray) -> dict:
    """Each of ``COMPARISONS`` as ``paired_interval`` on ``draws``, keyed ``"<arm> - <reference>"``."""
    out = {}
    for arm, reference in COMPARISONS:
        diff, lo, hi = paired_interval(proba[arm], proba[reference], labels, draws)
        out[f"{arm} - {reference}"] = {"difference": diff, "interval": [lo, hi]}
    return out


def auc_gate(proba: np.ndarray, labels: np.ndarray, expected: float, slides: str) -> float:
    """The unfiltered ``D+C``'s macro AUC; ``RuntimeError`` unless it rounds to the improvement phase's ``expected``."""
    macro = c.celltype.macro_auc(proba, labels)[0]
    if round(macro, 3) != expected:
        raise RuntimeError(f"unfiltered D+C has {slides} macro AUC {macro:.4f}, not the improvement phase's {expected}")
    return macro


def run_readouts(inputs: dict) -> dict:
    """The development readouts from ``results/development.npz``, as ``development.json``'s section ``readouts``.

    ``RuntimeError``, before any AUC, unless ``development.npz`` lists the development slides in order, and unless the
    unfiltered ``D+C``, averaged over the repeats, passes ``auc_gate`` at ``DEV_GATE_AUC``.
    """
    dev = np.flatnonzero(inputs["dev"])
    labels = inputs["labels"][dev]
    with np.load(c.RESULTS / "development.npz", allow_pickle=False) as saved:
        if [str(s) for s in saved["slide_ids"]] != [inputs["slide_ids"][i] for i in dev]:
            raise RuntimeError("development.npz lists the development slides differently")
        repeats = {arm: saved[arm] for arm in ARMS}
    proba = {arm: p.mean(axis=0) for arm, p in repeats.items()}  # averaged over the repeats, as the cell-type study's
    gate = auc_gate(proba["dc_unfiltered"], labels, DEV_GATE_AUC, "development")
    arms = {}
    for arm in ARMS:
        per_repeat = [c.celltype.macro_auc(p, labels)[0] for p in repeats[arm]]
        arms[arm] = auc_readout(proba[arm], labels) | {
            "auc_per_repeat": per_repeat,
            "auc_repeat_mean": float(np.mean(per_repeat)),
            "auc_repeat_sd": float(np.std(per_repeat, ddof=1)),
        }
    draws = c.celltype.stratified_bootstrap(labels, c.N_BOOT, np.random.default_rng(c.BOOT_SEED))
    return {
        "description": "each arm's AUCs from its out-of-fold probabilities averaged over the repeats; auc_per_repeat "
        "holds each repeat's macro AUC",
        "gate": {
            "arm": "dc_unfiltered",
            "macro_auc": gate,
            "expected": DEV_GATE_AUC,
            "reference": "gtex-liver-celltype-slices/results/improve_dev.json baseline:D+C-features at k 51",
        },
        "arms": arms,
        "comparisons": compare(proba, labels, draws),
        "bootstrap": {"draws": c.N_BOOT, "seed": c.BOOT_SEED, "stratified_by": "group"},
    }


def run_holdout(inputs: dict, grams: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    """The holdout, scored once (the ``holdout`` stage): ``holdout.json``'s content and ``holdout.npz``'s arrays.

    Before the fit, ``RuntimeError`` unless ``development.json`` has its readouts, whose development interval of
    ``PRIMARY`` the verdict reads, and unless ``improve_holdout.npz`` lists the holdout slides and their groups as
    ``inputs`` do. After it, ``dc_gate`` and ``auc_gate`` on the unfiltered ``D+C``. Writes nothing.
    """
    readouts = json.loads((c.RESULTS / "development.json").read_text()).get("readouts")
    if readouts is None:
        raise RuntimeError("development.json has no readouts: run the readouts stage before the holdout")
    dev_lo, dev_hi = readouts["comparisons"][PRIMARY]["interval"]
    labels = inputs["labels"]
    dev, hold = np.flatnonzero(inputs["dev"]), np.flatnonzero(inputs["holdout"])
    y = labels[hold]
    with np.load(c.CELLTYPE / "results/improve_holdout.npz", allow_pickle=False) as improve:
        if not (np.array_equal(improve["holdout_rows"], hold) and np.array_equal(improve["y_holdout"], y)):
            raise RuntimeError("improve_holdout.npz lists the holdout slides or their groups differently")
        key = f"e{[str(n) for n in improve['names']].index(HOLDOUT_GATE_NAME)}__proba"
        reference = improve[key]
    hidden = np.where(inputs["dev"], labels, -1)  # the holdout slides' labels never reach the fit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = outer_fold(grams, inputs["DC"], hidden, dev, hold, seed=HOLDOUT_SEED)
    n_convergence = _convergence_warnings(caught)
    gap = dc_gate(fit["dc_unfiltered"], reference)
    gate = auc_gate(fit["dc_unfiltered"], y, HOLDOUT_GATE_AUC, "holdout")
    proba = {arm: fit[arm] for arm in ARMS}
    draws = c.celltype.stratified_bootstrap(y, c.N_BOOT, np.random.default_rng(c.BOOT_SEED))
    comparisons = compare(proba, y, draws)
    holdout_diff = comparisons[PRIMARY]["difference"]
    dropped = ~fit["keep"]  # over the development slides, in order
    groups = c.celltype.GROUP_ORDER
    per_group = {g: int(n) for g, n in zip(groups, np.bincount(y, minlength=N_GROUPS), strict=True)}
    few = ", ".join(f"{g} {n}" for g, n in per_group.items() if n < 5)
    payload = {
        "description": "the holdout, scored once: each arm fitted on the development slides predicts the holdout slides",
        "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_development": int(dev.size),
        "n_holdout": int(hold.size),
        "holdout_per_group": per_group,
        "fit": {
            "inner_seed": HOLDOUT_SEED,
            "chosen_k": int(fit["k"]),
            "dropped": int(dropped.sum()),
            "dropped_per_group": {g: int(np.sum(dropped & (labels[dev] == i))) for i, g in enumerate(groups)},
        },
        "gate": {
            "reference": f"gtex-liver-celltype-slices/results/improve_holdout.npz {HOLDOUT_GATE_NAME} ({key})",
            "rtol": 0,
            "atol": 1e-12,
            "max_abs_difference": gap,
            "macro_auc": gate,
            "expected": HOLDOUT_GATE_AUC,
        },
        "arms": {arm: auc_readout(proba[arm], y) for arm in ARMS},
        "comparisons": comparisons,
        "bootstrap": {"draws": c.N_BOOT, "seed": c.BOOT_SEED, "stratified_by": "group"},
        "note": "The intervals are too narrow: the bootstrap resamples the holdout slides within their groups, and "
        f"groups of few slides ({few}) vary little across the draws.",
        "verdict": {
            "rule": "beats D+C: the development interval of curves - dc lies above 0 and its holdout difference is "
            "positive; not confirmed: the interval lies above 0 and the holdout difference is not positive; level "
            "with D+C: the interval contains 0; below D+C: it lies below 0. curves4 - dc is reported beside, with no "
            "verdict.",
            "development_interval": [dev_lo, dev_hi],
            "holdout_difference": holdout_diff,
            "verdict": verdict(dev_lo, dev_hi, holdout_diff),
        },
        "convergence_warnings": n_convergence,
    }
    arrays = {
        **proba,
        "k": np.array(fit["k"]),
        "dropped": dropped,
        "slide_ids": np.array([inputs["slide_ids"][i] for i in hold]),
        "labels": y,
        "dev_slide_ids": np.array([inputs["slide_ids"][i] for i in dev]),
    }
    return payload, arrays


def load_grams(slide_ids: list[str]) -> dict[str, np.ndarray]:
    """Each class's Gram matrix from ``results/gram.npz``; ``RuntimeError`` if it lists the slides differently."""
    with np.load(c.RESULTS / "gram.npz", allow_pickle=False) as saved:
        if [str(s) for s in saved["slide_ids"]] != slide_ids:
            raise RuntimeError("gram.npz lists the slides differently from spaces.npz")
        return {name: saved[f"gram_{name}"] for name in c.CLASSES}


def _run_record(started: float, md5: dict[str, str]) -> dict:
    """A stage's wall time and peak memory, its code (the md5s taken as it started), topocyte's commit and versions."""
    return {
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


def _print_scores(section: dict) -> None:
    """Each arm's macro AUC and each comparison's difference with its interval, for the log."""
    for arm, row in section["arms"].items():
        print(f"{arm}: macro AUC {row['macro_auc']:.4f}", flush=True)
    for name, row in section["comparisons"].items():
        lo, hi = row["interval"]
        print(f"{name}: {row['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    c.check_topocyte()
    # At the start: the code that runs.
    md5 = {name: hashlib.md5((c.HERE / name).read_bytes()).hexdigest() for name in SOURCES}
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("stage", choices=("development", "readouts", "holdout"))
    stage = parser.parse_args(argv).stage
    scored = c.RESULTS / "holdout.json"
    if stage == "holdout" and scored.exists():
        raise RuntimeError(f"{scored} exists: the holdout is scored once")
    started = time.perf_counter()
    inputs = c.load_inputs()
    if stage == "readouts":
        path = c.RESULTS / "development.json"
        summary = json.loads(path.read_text())
        summary["readouts"] = run_readouts(inputs) | _run_record(started, md5)
        path.write_text(json.dumps(summary, indent=1) + "\n")
        _print_scores(summary["readouts"])
        print(f"added the readouts to results/development.json in {summary['readouts']['seconds']} s", flush=True)
        return
    grams = load_grams(inputs["slide_ids"])
    if stage == "development":
        summary = run_development(inputs, grams) | _run_record(started, md5)
        (c.RESULTS / "development.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"wrote results/development.npz and .json in {summary['seconds']} s", flush=True)
        return
    payload, arrays = run_holdout(inputs, grams)
    payload |= _run_record(started, md5)
    text = json.dumps(payload, indent=1) + "\n"  # serialized before anything is written
    np.savez(c.RESULTS / "holdout.npz", **arrays)
    with scored.open("x") as out:  # holdout.json last, and never over an existing one
        out.write(text)
    _print_scores(payload)
    print(f"verdict: {payload['verdict']['verdict']}; wrote results/holdout.npz and .json", flush=True)


if __name__ == "__main__":
    main()
