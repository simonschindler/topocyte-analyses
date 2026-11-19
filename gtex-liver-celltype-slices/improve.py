#!/usr/bin/env python
"""Improvement phase on the development slides only (design.md: Pipeline 10 and the rulings after the builder check).

Each idea builds features for all 594 slides from the saved results. Its
unsupervised steps may see every slide, but no labels. A supervised step
(a projection that reads labels) is fitted inside each training fold.
Every idea is scored on the 476 development slides by 20 repeats of
stratified 5-fold (seed 1) with the readout's logistic regression
(``common.cv_probabilities``, or the same model after a supervised
projection):

* macro AUC from the out-of-fold probabilities averaged over the repeats;
* its paired difference to ``everything`` on the first four components,
  with a 95% interval from 4000 bootstrap draws of development slides
  stratified by group (seed 54321);
* for unsupervised spaces, purity with neighbours among the development
  slides, and its paired difference to ``everything``.

The holdout slides' labels are never read here.

Usage: ``improve.py`` runs every idea not yet in ``results/improve_dev.json``
and appends it, so ideas can be added one at a time; ``improve.py --redo
NAME`` reruns one.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as c  # noqa: E402

DEV_FOLD_SEED = 1
DEV_BOOT_SEED = 54321
N_DRAWS = 4000
OUT_JSON = c.RESULTS / "improve_dev.json"
OUT_NPZ = c.RESULTS / "improve_dev.npz"


# ---------------------------------------------------------------- data


def load() -> dict:
    with np.load(c.RESULTS / "spaces.npz", allow_pickle=False) as data:
        names = [str(n) for n in data["names"]]
        scores = {name: data[f"s{i}__scores"] for i, name in enumerate(names)}
        blocks_of = {name: [str(b) for b in data[f"s{i}__blocks"]] for i, name in enumerate(names) if name.startswith("mfa:")}
        labels, D, C = data["labels"], data["D_features"], data["C_features"]
        slides = [str(s) for s in data["slide_ids"]]
    holdout_table = pd.read_csv(c.RESULTS / "holdout.csv")
    if [str(s) for s in holdout_table["slide_id"]] != slides:
        raise RuntimeError("holdout.csv lists the slides differently from spaces.npz")
    aliases = json.loads((c.RESULTS / "spaces.json").read_text())["aliases"]
    holdout = holdout_table["holdout"].to_numpy(dtype=bool)
    blocks = {name.split(":", 1)[1]: s for name, s in scores.items() if name.startswith("block:")}
    percell_path = c.RESULTS / "percell_blocks.npz"
    if percell_path.exists():  # written by percell_grams.py fpca: keys "<set>:h<d>_percell:<slice>"
        with np.load(percell_path, allow_pickle=False) as data:
            blocks.update({key: data[key] for key in data.files if not key.endswith("__evr")})
    return {"scores": scores, "blocks_of": blocks_of, "labels": labels, "D": D, "C": C, "holdout": holdout,
            "dev": ~holdout, "aliases": aliases, "blocks": blocks}


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def mfa_of(blocks: list[np.ndarray], n_keep: int = 50) -> np.ndarray:
    return c.mfa(blocks, n_keep=n_keep)[0]


def weighted_columns(blocks: list[np.ndarray]) -> np.ndarray:
    """The MFA-weighted concatenation of the blocks (each divided by the SD of its first score)."""
    return np.hstack([block / block[:, 0].std() for block in blocks])


def residualize(block: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Least-squares residual of every column on the covariates and an intercept (no labels)."""
    design = np.column_stack([np.ones(len(covariates)), covariates])
    coef, *_ = np.linalg.lstsq(design, block, rcond=None)
    return block - design @ coef


# ---------------------------------------------------------------- supervised projections (fitted inside folds)


class Projector:
    """Standardize, then a supervised projection to ``n`` dimensions, fitted on the training fold only."""

    def __init__(self, method: str, n: int) -> None:
        self.method, self.n = method, n

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Projector":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.mean, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mean) / self.sd
        if self.method == "lda":
            self.model = LinearDiscriminantAnalysis(n_components=self.n, solver="eigen", shrinkage="auto").fit(Z, y)
        elif self.method == "pls":
            onehot = np.eye(len(c.GROUP_ORDER))[y]
            # Class-balanced targets: each class's indicator scaled by 1/sqrt(its size).
            onehot = onehot / np.sqrt(np.maximum(onehot.sum(axis=0), 1.0))
            self.model = PLSRegression(n_components=self.n, scale=False).fit(Z, onehot)
        else:
            raise ValueError(self.method)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.sd
        return self.model.transform(Z)


def cv_projected(X: np.ndarray, labels: np.ndarray, folds: np.ndarray, method: str, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Out-of-fold probabilities of the readout's logistic regression on a projection fitted in each training fold."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    n_repeats, n_slides = folds.shape
    proba = np.zeros((n_repeats, n_slides, len(c.GROUP_ORDER)))
    n_warnings = 0
    for r in range(n_repeats):
        for f in np.unique(folds[r]):
            test = folds[r] == f
            train = ~test
            projector = Projector(method, n).fit(X[train], labels[train])
            P_train, P_test = projector.transform(X[train]), projector.transform(X[test])
            mean, sd = P_train.mean(axis=0), P_train.std(axis=0)
            sd[sd == 0] = 1.0
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit((P_train - mean) / sd, labels[train])
            n_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            proba[r][np.ix_(np.flatnonzero(test), model.classes_)] = model.predict_proba((P_test - mean) / sd)
    return proba.mean(axis=0), proba, n_warnings


def kruskal_block_scores(blocks: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Each block's sum, over its first four components, of the Kruskal-Wallis statistic across the groups in ``y``."""
    from scipy.stats import kruskal

    present = np.unique(y)
    return np.array([sum(kruskal(*[b[y == g, j] for g in present]).statistic for j in range(4)) for b in blocks])


def selected_space(blocks: list[np.ndarray], rows: np.ndarray, y: np.ndarray, n_blocks: int, k: int) -> np.ndarray:
    """MFA (on every slide) of the ``n_blocks`` blocks ranked highest on the labelled ``rows``; first ``k`` scores."""
    ranking = np.argsort(-kruskal_block_scores([b[rows] for b in blocks], y), kind="stable")[:n_blocks]
    return c.mfa([blocks[i] for i in ranking], n_keep=k)[0]


def cv_selected(blocks: list[np.ndarray], dev_rows: np.ndarray, labels: np.ndarray, folds: np.ndarray,
                n_blocks: int, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Out-of-fold probabilities when each training fold chooses its blocks (labels of that fold only)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    n_repeats, n_slides = folds.shape
    proba = np.zeros((n_repeats, n_slides, len(c.GROUP_ORDER)))
    n_warnings = 0
    for r in range(n_repeats):
        for f in np.unique(folds[r]):
            test = folds[r] == f
            train = ~test
            X = selected_space(blocks, dev_rows[train], labels[train], n_blocks, k)[dev_rows]
            mean, sd = X[train].mean(axis=0), X[train].std(axis=0)
            sd[sd == 0] = 1.0
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit((X[train] - mean) / sd, labels[train])
            n_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            proba[r][np.ix_(np.flatnonzero(test), model.classes_)] = model.predict_proba((X[test] - mean) / sd)
    return proba.mean(axis=0), proba, n_warnings


# ---------------------------------------------------------------- ideas


def blocks_list(d: dict, sets: tuple[str, ...], models: tuple[str, ...] = c.MODEL_NAMES) -> list[np.ndarray]:
    return [d["blocks"][f"{p}:{m}:{k}"] for p in sets for m in models for k in range(c.N_SLICES) if f"{p}:{m}:{k}" in d["blocks"]]


def idea_definitions(d: dict) -> list[dict]:
    """Every idea, in the order tried: name, mechanism, what it is, kind, and how to build it."""
    everything_sets = c.POINT_SETS
    DC = np.hstack([d["D"], d["C"]])
    ideas = [
        {"name": "baseline:everything", "mechanism": "baseline", "kind": "space",
         "what": "the main experiment's everything space (MFA of all 200 blocks), for reference on the development slides",
         "build": lambda: d["scores"][d["aliases"]["everything"]]},
        {"name": "baseline:all-cell", "mechanism": "baseline", "kind": "space",
         "what": "the main experiment's all-cell space (MFA of the 40 all-cell blocks)",
         "build": lambda: d["scores"][d["aliases"]["all-cell"]]},
        {"name": "baseline:D+C-features", "mechanism": "baseline", "kind": "features",
         "what": "density summaries and cell counts per group and slice (51 features), no topology",
         "build": lambda: DC},
        {"name": "zscore-columns", "mechanism": "block weighting", "kind": "space",
         "what": "PCA of all 1800 block columns, each z-scored (every fPCA component weighs the same, not each block's first)",
         "build": lambda: c.pca(zscore(np.hstack(blocks_list(d, everything_sets))))[0][:, :50]},
        {"name": "residual-density", "mechanism": "density removal", "kind": "space",
         "what": "MFA of all 200 blocks after regressing every score on the 11 density summaries (topology beyond density)",
         "build": lambda: mfa_of([residualize(b, d["D"]) for b in blocks_list(d, everything_sets)])},
        {"name": "residual-density-composition", "mechanism": "composition removal", "kind": "space",
         "what": "MFA of all 200 blocks after regressing every score on density summaries and composition (51 features)",
         "build": lambda: mfa_of([residualize(b, DC) for b in blocks_list(d, everything_sets)])},
        {"name": "cosine-everything", "mechanism": "shape only", "kind": "space",
         "what": "MFA of the cosine blocks of all five point sets (curve shape, no magnitude, so no raw cell counts)",
         "build": lambda: mfa_of(blocks_list(d, everything_sets, ("h0_cosine", "h1_cosine")))},
        {"name": "select20-kruskal", "mechanism": "block selection", "kind": "selection", "n_blocks": 20,
         "what": "in each training fold, the 20 of 200 blocks whose first four scores differ most across groups (Kruskal-Wallis), then MFA of those",
         "build": lambda: blocks_list(d, everything_sets)},
        {"name": "select40-kruskal", "mechanism": "block selection (40)", "kind": "selection", "n_blocks": 40,
         "what": "the same with 40 blocks",
         "build": lambda: blocks_list(d, everything_sets)},
        {"name": "lda4-everything20", "mechanism": "supervised projection", "kind": "supervised", "method": "lda", "source_k": 20,
         "what": "shrinkage LDA to four discriminants of the everything space's first 20 components, fitted in each training fold",
         "build": lambda: d["scores"][d["aliases"]["everything"]]},
        {"name": "pls4-everything50", "mechanism": "supervised projection", "kind": "supervised", "method": "pls", "source_k": 50,
         "what": "class-balanced PLS-DA to four components of the first 50 components of the everything MFA, fitted in each training fold",
         "build": lambda: mfa_of(blocks_list(d, everything_sets))},
        {"name": "lda4-allcell20", "mechanism": "supervised projection (all-cell)", "kind": "supervised", "method": "lda", "source_k": 20,
         "what": "the same shrinkage LDA on the all-cell space's first 20 components: what supervision alone buys",
         "build": lambda: d["scores"][d["aliases"]["all-cell"]]},
        {"name": "lda4-DC", "mechanism": "supervised projection (no topology)", "kind": "supervised", "method": "lda", "source_k": 51,
         "what": "the same shrinkage LDA on the 51 density and composition features: the yardstick for the supervised ideas",
         "build": lambda: DC},
        {"name": "lda4-DC+everything20", "mechanism": "supervised projection (with composition)", "kind": "supervised", "method": "lda",
         "source_k": 71, "what": "shrinkage LDA on density, composition and the everything space's first 20 components",
         "build": lambda: np.hstack([DC, d["scores"][d["aliases"]["everything"]][:, :20]])},
        # Round 2, chosen after round 1 and the main readouts (post hoc; the holdout is the check).
        {"name": "h0cosine-everything", "mechanism": "shape only, H0", "kind": "space",
         "what": "MFA of the H0 cosine blocks of all five point sets (the main heatmaps' strongest model)",
         "build": lambda: mfa_of(blocks_list(d, everything_sets, ("h0_cosine",)))},
        {"name": "pls4-allcolumns", "mechanism": "supervised projection", "kind": "supervised", "method": "pls", "source_k": 1800,
         "what": "class-balanced PLS-DA to four components of all 1800 MFA-weighted block columns, no truncation",
         "build": lambda: weighted_columns(blocks_list(d, everything_sets))},
        {"name": "pls4-DC", "mechanism": "supervised projection (no topology)", "kind": "supervised", "method": "pls", "source_k": 51,
         "what": "class-balanced PLS-DA to four components of the 51 density and composition features",
         "build": lambda: DC},
        {"name": "pls4-DC+everything50", "mechanism": "supervised projection (with composition)", "kind": "supervised",
         "method": "pls", "source_k": 101,
         "what": "class-balanced PLS-DA to four components of density, composition and the everything MFA's first 50 components",
         "build": lambda: np.hstack([DC, mfa_of(blocks_list(d, everything_sets))])},
        {"name": "pls4-residual-density-composition", "mechanism": "supervised projection beyond composition",
         "kind": "supervised", "method": "pls", "source_k": 50,
         "what": "class-balanced PLS-DA to four components of the first 50 components of the MFA of all blocks with density and composition regressed out: is any signal left?",
         "build": lambda: mfa_of([residualize(b, DC) for b in blocks_list(d, everything_sets)])},
    ]
    if "all:h0_percell:0" in d["blocks"]:
        percell = ("h0_percell", "h1_percell")
        ideas += [
            {"name": "percell-everything", "mechanism": "per-cell curves", "kind": "space",
             "what": "MFA over all five point sets of the per-cell raw blocks (Betti curve / number of points) and the cosine blocks: raw replaced by per-cell",
             "build": lambda: mfa_of(blocks_list(d, everything_sets, percell + ("h0_cosine", "h1_cosine")))},
            {"name": "percell-only", "mechanism": "per-cell curves only", "kind": "space",
             "what": "MFA over all five point sets of the per-cell H0 and H1 blocks alone",
             "build": lambda: mfa_of(blocks_list(d, everything_sets, percell))},
        ]
    return ideas


# ---------------------------------------------------------------- scoring

_STATE: dict = {}


def _init(labels: np.ndarray, folds: np.ndarray) -> None:
    _STATE["labels"], _STATE["folds"] = labels, folds


def _run(task: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    kind, X, method, n = task
    if kind == "supervised":
        return cv_projected(X, _STATE["labels"], _STATE["folds"], method, n)
    if kind == "selection":
        blocks, dev_rows = X
        return cv_selected(blocks, dev_rows, _STATE["labels"], _STATE["folds"], method, n)
    return c.cv_probabilities(X, _STATE["labels"], _STATE["folds"])


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="append", default=[])
    args = parser.parse_args()
    started = time.time()
    d = load()
    dev_labels = d["labels"][d["dev"]]
    folds = c.make_folds(dev_labels, n_repeats=20, n_splits=5, seed=DEV_FOLD_SEED)
    draws = c.stratified_bootstrap(dev_labels, N_DRAWS, np.random.default_rng(DEV_BOOT_SEED))
    done = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"ideas": []}
    done["ideas"] = [entry for entry in done["ideas"] if entry["name"] not in args.redo]
    names_done = {entry["name"] for entry in done["ideas"]}
    stored = dict(np.load(OUT_NPZ)) if OUT_NPZ.exists() else {}
    ideas = [idea for idea in idea_definitions(d) if idea["name"] not in names_done]

    tasks, keys = [], []
    for idea in ideas:
        X = idea["build"]()
        if idea["kind"] == "supervised":
            tasks.append(("supervised", c.leading(X, idea["source_k"])[d["dev"]], idea["method"], 4))
            keys.append((idea["name"], 4))
        elif idea["kind"] == "selection":
            tasks.append(("selection", (X, np.flatnonzero(d["dev"])), idea["n_blocks"], 4))
            keys.append((idea["name"], 4))
        elif idea["kind"] == "features":
            tasks.append(("plain", X[d["dev"]], None, None))
            keys.append((idea["name"], X.shape[1]))
        else:
            for k in (4, 20):
                tasks.append(("plain", c.leading(X, k)[d["dev"]], None, None))
                keys.append((idea["name"], k))
            idea["_X"] = X[d["dev"]]
    if tasks:
        with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(dev_labels, folds)) as pool:
            results = list(pool.map(_run, tasks))
        for (name, k), (mean_proba, repeat_proba, n_warn) in zip(keys, results):
            stored[f"{name}@{k}"] = mean_proba
            stored[f"{name}@{k}__warnings"] = np.array(n_warn)

    reference = stored["baseline:everything@4"]
    ref_draws = np.array([c.macro_auc(reference[idx], dev_labels[idx])[0] for idx in draws])
    ref_space = d["scores"][d["aliases"]["everything"]][d["dev"]]
    ref_purity = c.purity(c.leading(ref_space, 4), dev_labels)

    def purity_macro_draws(values: np.ndarray) -> np.ndarray:
        return np.mean([values[draws[:, dev_labels == g]].mean(axis=1) for g in range(len(c.CONDITIONS))], axis=0)

    for idea in ideas:
        entry = {key: idea[key] for key in ("name", "mechanism", "kind", "what")}
        entry["readouts"] = []
        for (name, k) in [key for key in keys if key[0] == idea["name"]]:
            proba = stored[f"{name}@{k}"]
            macro, per = c.macro_auc(proba, dev_labels)
            boot = np.array([c.macro_auc(proba[idx], dev_labels[idx])[0] for idx in draws])
            row = {"k": k, "macro_auc": macro, "auc_per_condition": per.tolist(), "auc_interval": interval(boot),
                   "n_convergence_warnings": int(stored[f"{name}@{k}__warnings"])}
            if k == 4 and idea["kind"] in ("space", "supervised", "selection"):
                row["vs_everything4"] = {"difference": macro - c.macro_auc(reference, dev_labels)[0],
                                         "interval": interval(boot - ref_draws)}
            if idea["kind"] == "space":
                values = c.purity(c.leading(idea["_X"], k), dev_labels)
                row["purity_macro"] = c.macro_purity(values, dev_labels)[0]
                if k == 4:
                    row["purity_vs_everything4"] = {
                        "difference": row["purity_macro"] - c.macro_purity(ref_purity, dev_labels)[0],
                        "interval": interval(purity_macro_draws(values) - purity_macro_draws(ref_purity))}
            entry["readouts"].append(row)
        entry["tried_at"] = time.strftime("%Y-%m-%d %H:%M")
        done["ideas"].append(entry)
        k4 = next((r for r in entry["readouts"] if r["k"] == 4), entry["readouts"][0])
        extra = f", vs everything {k4['vs_everything4']['difference']:+.3f} {np.round(k4['vs_everything4']['interval'], 3)}" if "vs_everything4" in k4 else ""
        print(f"{idea['name']}: macro AUC (k={k4['k']}) {k4['macro_auc']:.3f} {np.round(k4['auc_interval'], 3)}{extra}", flush=True)
    done["dev_slides"] = int(d["dev"].sum())
    done["fold_seed"], done["bootstrap_seed"], done["n_draws"] = DEV_FOLD_SEED, DEV_BOOT_SEED, N_DRAWS
    OUT_JSON.write_text(json.dumps(done, indent=1) + "\n")
    np.savez(OUT_NPZ, **stored)
    print(f"{len(ideas)} ideas in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
