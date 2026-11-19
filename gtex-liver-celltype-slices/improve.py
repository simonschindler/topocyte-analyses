#!/usr/bin/env python
"""Improvement phase on the development slides only (design.md: Pipeline 10 and the rulings after the builder check).

Each idea builds features for all 594 slides from the saved results; its
unsupervised steps may see every slide, but no labels. A supervised step (a
projection or a block selection that reads labels) is fitted inside each
training fold. Every idea is scored on the 476 development slides by 20
repeats of stratified 5-fold (seed 1) with the readout's logistic regression
(multinomial L2, C = 1, balanced class weights, standardized on the
training fold) on the idea's four-dimensional view:

* an unsupervised space: its first four components (and, descriptive, 20);
* a supervised projection: its four components, fitted on the training fold;
* a block selection: the first four MFA components of the blocks the
  training fold chose.

Readouts:

* macro AUC from the out-of-fold probabilities averaged over the repeats,
  with its 95% interval and its paired difference to ``everything`` on four
  components (4000 bootstrap draws of development slides stratified by
  group, seed 54321);
* out-of-fold purity: each development slide of a test fold against its 15
  nearest training-fold slides in that fold's view, over the chance share
  n_g / N of the training fold, averaged over the repeats, then over the
  conditions; with its paired difference to ``everything``'s;
* for unsupervised spaces also the in-sample purity among all development
  slides, as in the main readouts (descriptive).

Every idea has a ``role``:

* ``idea``: a candidate for the holdout;
* ``baseline``: the main experiment's spaces, for reference;
* ``control``: the same method without topology, or on all-cell topology alone, a yardstick that is never chosen;
* ``diagnostic``: built to remove signal, never chosen.

Its ``mechanism`` names the modelling change; ideas that share one compete
for a single place on the holdout. Its ``round`` says when it was
conceived: 1 before any readout existed, 2 after the main readouts and
round 1's results (post hoc, which makes the holdout a weaker check for
it). The holdout slides' labels are never read here; the main experiment's
readouts did read them, as Simon ruled.

Usage: ``improve.py`` runs every idea not yet in ``results/improve_dev.json``
and appends it; ``improve.py --redo NAME`` reruns one. The per-cell ideas
appear once ``percell_fpca.py collect`` has written
``results/percell_blocks.npz``.
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
N_NEIGHBORS = 15
OUT_JSON = c.RESULTS / "improve_dev.json"
OUT_NPZ = c.RESULTS / "improve_dev.npz"
VIEW_KINDS = ("space", "supervised", "selection")


# ---------------------------------------------------------------- data


def load() -> dict:
    with np.load(c.RESULTS / "spaces.npz", allow_pickle=False) as data:
        names = [str(n) for n in data["names"]]
        scores = {name: data[f"s{i}__scores"] for i, name in enumerate(names)}
        labels, D, C = data["labels"], data["D_features"], data["C_features"]
        slides = [str(s) for s in data["slide_ids"]]
    holdout_table = pd.read_csv(c.RESULTS / "holdout.csv")
    if [str(s) for s in holdout_table["slide_id"]] != slides:
        raise RuntimeError("holdout.csv lists the slides differently from spaces.npz")
    aliases = json.loads((c.RESULTS / "spaces.json").read_text())["aliases"]
    holdout = holdout_table["holdout"].to_numpy(dtype=bool)
    blocks = {name.split(":", 1)[1]: s for name, s in scores.items() if name.startswith("block:")}
    percell_path = c.RESULTS / "percell_blocks.npz"
    if percell_path.exists():  # written by percell_fpca.py collect: keys "<set>:h<d>_percell:<slice>"
        with np.load(percell_path, allow_pickle=False) as data:
            blocks.update({key: data[key] for key in data.files if not key.endswith("__evr")})
    return {"scores": scores, "labels": labels, "D": D, "C": C, "holdout": holdout, "dev": ~holdout,
            "aliases": aliases, "blocks": blocks}


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def mfa_of(blocks: list[np.ndarray], n_keep: int = 50) -> np.ndarray:
    return c.mfa(blocks, n_keep=n_keep)[0]


def residualize(block: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Least-squares residual of every column on the covariates and an intercept (no labels)."""
    design = np.column_stack([np.ones(len(covariates)), covariates])
    coef, *_ = np.linalg.lstsq(design, block, rcond=None)
    return block - design @ coef


def neighbour_purity(queries: np.ndarray, query_labels: np.ndarray, pool: np.ndarray, pool_labels: np.ndarray,
                     n_neighbors: int = N_NEIGHBORS) -> np.ndarray:
    """Share of each query's nearest pool slides in its own group, over the pool's chance share; NaN for unclassified."""
    from scipy.spatial.distance import cdist

    nearest = np.argsort(cdist(queries, pool), axis=1, kind="stable")[:, :n_neighbors]
    counts = np.bincount(pool_labels, minlength=len(c.GROUP_ORDER))
    out = np.full(len(queries), np.nan)
    scored = query_labels < len(c.CONDITIONS)
    same = (pool_labels[nearest[scored]] == query_labels[scored, None]).mean(axis=1)
    out[scored] = same / (counts[query_labels[scored]] / pool_labels.size)
    return out


# ---------------------------------------------------------------- views fitted inside the folds


class Projector:
    """Z-score on the training fold, then a supervised projection to ``n`` dimensions fitted there."""

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
        return self.model.transform((X - self.mean) / self.sd)


def kruskal_block_scores(blocks: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Each block's sum, over its first four components, of the Kruskal-Wallis statistic across the groups in ``y``."""
    from scipy.stats import kruskal

    present = np.unique(y)
    return np.array([sum(kruskal(*[b[y == g, j] for g in present]).statistic for j in range(4)) for b in blocks])


def selected_space(blocks: list[np.ndarray], rows: np.ndarray, y: np.ndarray, n_blocks: int, k: int) -> np.ndarray:
    """MFA (on every slide) of the ``n_blocks`` blocks ranked highest on the labelled ``rows``; first ``k`` scores."""
    ranking = np.argsort(-kruskal_block_scores([b[rows] for b in blocks], y), kind="stable")[:n_blocks]
    return c.mfa([blocks[i] for i in ranking], n_keep=k)[0]


def fit_logistic(Z_train: np.ndarray, y_train: np.ndarray) -> tuple[object, np.ndarray, np.ndarray, int]:
    """The readout's logistic regression on features standardized with the training fold; its convergence warnings."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    mean, sd = Z_train.mean(axis=0), Z_train.std(axis=0)
    sd[sd == 0] = 1.0
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit((Z_train - mean) / sd, y_train)
    return model, mean, sd, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def cv_view(make_view, labels: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Out-of-fold probabilities (mean, per repeat), convergence warnings, and out-of-fold purity per slide."""
    n_repeats, n = folds.shape
    proba = np.zeros((n_repeats, n, len(c.GROUP_ORDER)))
    purity = np.full((n_repeats, n), np.nan)
    n_warnings = 0
    for r in range(n_repeats):
        for f in np.unique(folds[r]):
            test = folds[r] == f
            train = ~test
            Z_train, Z_test = make_view(train, test)
            model, mean, sd, n_warn = fit_logistic(Z_train, labels[train])
            n_warnings += n_warn
            proba[r][np.ix_(np.flatnonzero(test), model.classes_)] = model.predict_proba((Z_test - mean) / sd)
            purity[r, test] = neighbour_purity(Z_test, labels[test], Z_train, labels[train])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # unclassified slides: NaN in every repeat
        per_slide = np.nanmean(purity, axis=0)
    return proba.mean(axis=0), proba, n_warnings, per_slide


_STATE: dict = {}


def _init(labels: np.ndarray, folds: np.ndarray) -> None:
    _STATE["labels"], _STATE["folds"] = labels, folds


def _run(task: tuple) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    kind, payload, params = task
    labels, folds = _STATE["labels"], _STATE["folds"]
    if kind == "supervised":
        X, (method, n) = payload, params

        def view(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            projector = Projector(method, n).fit(X[train], labels[train])
            return projector.transform(X[train]), projector.transform(X[test])
    elif kind == "selection":
        (blocks, dev_rows), (n_blocks, k) = payload, params

        def view(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Z = selected_space(blocks, dev_rows[train], labels[train], n_blocks, k)[dev_rows]
            return Z[train], Z[test]
    else:
        X = payload

        def view(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return X[train], X[test]
    return cv_view(view, labels, folds)


# ---------------------------------------------------------------- ideas


def blocks_list(d: dict, sets: tuple[str, ...], models: tuple[str, ...] = c.MODEL_NAMES) -> list[np.ndarray]:
    return [d["blocks"][f"{p}:{m}:{k}"] for p in sets for m in models for k in range(c.N_SLICES) if f"{p}:{m}:{k}" in d["blocks"]]


def weighted_columns(blocks: list[np.ndarray]) -> np.ndarray:
    """The MFA-weighted concatenation of the blocks (each divided by the SD of its first score)."""
    return np.hstack([block / block[:, 0].std() for block in blocks])


def idea_definitions(d: dict) -> list[dict]:
    """Every entry, in the order tried: name, role, mechanism, round, inputs, what it is, kind, and how to build it."""
    sets = c.POINT_SETS
    DC = np.hstack([d["D"], d["C"]])
    everything = d["scores"][d["aliases"]["everything"]]
    sup_topology = "supervised projection (topology)"
    sup_both = "supervised projection (composition + topology)"
    ideas = [
        {"name": "baseline:everything", "role": "baseline", "mechanism": "baseline", "round": 1, "kind": "space",
         "inputs": "MFA of all 200 blocks", "what": "the main experiment's everything space, for reference",
         "build": lambda: everything},
        {"name": "baseline:all-cell", "role": "baseline", "mechanism": "baseline", "round": 1, "kind": "space",
         "inputs": "MFA of the 40 all-cell blocks", "what": "the main experiment's all-cell space",
         "build": lambda: d["scores"][d["aliases"]["all-cell"]]},
        {"name": "baseline:D+C-features", "role": "baseline", "mechanism": "baseline", "round": 1, "kind": "features",
         "inputs": "51 features", "what": "density summaries and cells per group and slice, no topology",
         "build": lambda: DC},
        {"name": "zscore-columns", "role": "idea", "mechanism": "block weighting", "round": 1, "kind": "space",
         "inputs": "PCA of 1800 z-scored block columns",
         "what": "every fPCA component weighs the same, instead of each block's first",
         "build": lambda: c.pca(zscore(np.hstack(blocks_list(d, sets))))[0][:, :50]},
        {"name": "residual-density", "role": "diagnostic", "mechanism": "density removed", "round": 1, "kind": "space",
         "inputs": "MFA of 200 residual blocks", "what": "every score regressed on the 11 density summaries first",
         "build": lambda: mfa_of([residualize(b, d["D"]) for b in blocks_list(d, sets)])},
        {"name": "residual-density-composition", "role": "diagnostic", "mechanism": "density and composition removed",
         "round": 1, "kind": "space", "inputs": "MFA of 200 residual blocks",
         "what": "every score regressed on the 51 density and composition features first",
         "build": lambda: mfa_of([residualize(b, DC) for b in blocks_list(d, sets)])},
        {"name": "cosine-everything", "role": "idea", "mechanism": "shape only", "round": 1, "kind": "space",
         "inputs": "MFA of the 100 cosine blocks", "what": "curve shape only, no magnitude, so no raw cell counts",
         "build": lambda: mfa_of(blocks_list(d, sets, ("h0_cosine", "h1_cosine")))},
        {"name": "select20-kruskal", "role": "idea", "mechanism": "block selection", "round": 1, "kind": "selection",
         "n_blocks": 20, "inputs": "MFA of 20 of 200 blocks",
         "what": "each training fold keeps the 20 blocks whose first four scores differ most across groups (Kruskal-Wallis)",
         "build": lambda: blocks_list(d, sets)},
        {"name": "select40-kruskal", "role": "idea", "mechanism": "block selection", "round": 1, "kind": "selection",
         "n_blocks": 40, "inputs": "MFA of 40 of 200 blocks", "what": "the same with 40 blocks",
         "build": lambda: blocks_list(d, sets)},
        {"name": "lda4-everything20", "role": "idea", "mechanism": sup_topology, "round": 1, "kind": "supervised",
         "method": "lda", "source_k": 20, "inputs": "20 MFA components, z-scored",
         "what": "shrinkage LDA to four discriminants, fitted in each training fold",
         "build": lambda: everything},
        {"name": "pls4-everything50", "role": "idea", "mechanism": sup_topology, "round": 1, "kind": "supervised",
         "method": "pls", "source_k": 50, "inputs": "50 MFA components of all 200 blocks, z-scored (whitened)",
         "what": "class-balanced PLS-DA to four components, fitted in each training fold",
         "build": lambda: mfa_of(blocks_list(d, sets))},
        {"name": "lda4-allcell20", "role": "control", "mechanism": "supervised projection (all-cell topology)", "round": 1,
         "kind": "supervised", "method": "lda", "source_k": 20, "inputs": "20 all-cell MFA components, z-scored",
         "what": "the same LDA on all-cell topology: what supervision alone buys",
         "build": lambda: d["scores"][d["aliases"]["all-cell"]]},
        {"name": "lda4-DC", "role": "control", "mechanism": "supervised projection (composition)", "round": 1,
         "kind": "supervised", "method": "lda", "source_k": 51, "inputs": "51 density and composition features, z-scored",
         "what": "the same LDA without topology: the yardstick for the supervised ideas",
         "build": lambda: DC},
        {"name": "lda4-DC+everything20", "role": "idea", "mechanism": sup_both, "round": 1, "kind": "supervised",
         "method": "lda", "source_k": 71, "inputs": "51 density and composition features + 20 MFA components, z-scored",
         "what": "shrinkage LDA to four discriminants, fitted in each training fold",
         "build": lambda: np.hstack([DC, everything[:, :20]])},
        # Round 2: conceived after the main readouts and round 1 (post hoc).
        {"name": "h0cosine-everything", "role": "idea", "mechanism": "shape only", "round": 2, "kind": "space",
         "inputs": "MFA of the 50 H0 cosine blocks", "what": "the H0 cosine blocks alone (the main heatmaps' strongest model)",
         "build": lambda: mfa_of(blocks_list(d, sets, ("h0_cosine",)))},
        {"name": "pls4-allcolumns", "role": "idea", "mechanism": sup_topology, "round": 2, "kind": "supervised",
         "method": "pls", "source_k": 1800, "inputs": "1800 block columns, z-scored (MFA weights do not survive)",
         "what": "class-balanced PLS-DA to four components, fitted in each training fold",
         "build": lambda: weighted_columns(blocks_list(d, sets))},
        {"name": "pls4-DC", "role": "control", "mechanism": "supervised projection (composition)", "round": 2,
         "kind": "supervised", "method": "pls", "source_k": 51, "inputs": "51 density and composition features, z-scored",
         "what": "the same PLS-DA without topology", "build": lambda: DC},
        {"name": "pls4-DC+everything50", "role": "idea", "mechanism": sup_both, "round": 2, "kind": "supervised",
         "method": "pls", "source_k": 101, "inputs": "51 features + 50 MFA components, z-scored",
         "what": "class-balanced PLS-DA to four components, fitted in each training fold",
         "build": lambda: np.hstack([DC, mfa_of(blocks_list(d, sets))])},
        {"name": "pls4-residual-density-composition", "role": "diagnostic",
         "mechanism": "supervised projection (topology beyond composition)", "round": 2, "kind": "supervised",
         "method": "pls", "source_k": 50, "inputs": "50 MFA components of 200 residual blocks, z-scored",
         "what": "PLS-DA after regressing every score on density and composition: is any signal left?",
         "build": lambda: mfa_of([residualize(b, DC) for b in blocks_list(d, sets)])},
    ]
    if "all:h0_percell:0" in d["blocks"]:
        percell = ("h0_percell", "h1_percell")
        ideas += [  # conceived before any readout (round 1), run once the per-cell fits were done
            {"name": "percell-everything", "role": "idea", "mechanism": "per-cell curves", "round": 1, "kind": "space",
             "inputs": "MFA of 100 per-cell and 100 cosine blocks",
             "what": "raw blocks replaced by per-cell ones (Betti curve / number of points)",
             "build": lambda: mfa_of(blocks_list(d, sets, percell + ("h0_cosine", "h1_cosine")))},
            {"name": "percell-only", "role": "idea", "mechanism": "per-cell curves", "round": 1, "kind": "space",
             "inputs": "MFA of the 100 per-cell blocks", "what": "the per-cell H0 and H1 blocks alone",
             "build": lambda: mfa_of(blocks_list(d, sets, percell))},
        ]
    return ideas


# ---------------------------------------------------------------- scoring


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
            tasks.append(("supervised", c.leading(X, idea["source_k"])[d["dev"]], (idea["method"], 4)))
            keys.append((idea["name"], 4))
        elif idea["kind"] == "selection":
            tasks.append(("selection", (X, np.flatnonzero(d["dev"])), (idea["n_blocks"], 4)))
            keys.append((idea["name"], 4))
        elif idea["kind"] == "features":
            tasks.append(("plain", X[d["dev"]], None))
            keys.append((idea["name"], X.shape[1]))
        else:
            for k in (4, 20):
                tasks.append(("plain", c.leading(X, k)[d["dev"]], None))
                keys.append((idea["name"], k))
            idea["_X"] = X[d["dev"]]
    if tasks:
        with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(dev_labels, folds)) as pool:
            results = list(pool.map(_run, tasks))
        for (name, k), (mean_proba, _, n_warn, oof_purity) in zip(keys, results):
            stored[f"{name}@{k}"] = mean_proba
            stored[f"{name}@{k}__warnings"] = np.array(n_warn)
            stored[f"{name}@{k}__purity_oof"] = oof_purity

    def macro_draws(proba: np.ndarray) -> np.ndarray:
        return np.array([c.macro_auc(proba[idx], dev_labels[idx])[0] for idx in draws])

    def purity_draws(values: np.ndarray) -> np.ndarray:
        return np.mean([values[draws[:, dev_labels == g]].mean(axis=1) for g in range(len(c.CONDITIONS))], axis=0)

    reference = stored["baseline:everything@4"]
    reference_auc = c.macro_auc(reference, dev_labels)[0]
    reference_draws = macro_draws(reference)
    reference_purity = stored["baseline:everything@4__purity_oof"]
    reference_purity_macro = c.macro_purity(reference_purity, dev_labels)[0]
    reference_purity_draws = purity_draws(reference_purity)

    for idea in ideas:
        entry = {key: idea[key] for key in ("name", "role", "mechanism", "round", "kind", "inputs", "what")}
        entry["readouts"] = []
        for (name, k) in [key for key in keys if key[0] == idea["name"]]:
            proba = stored[f"{name}@{k}"]
            macro, per = c.macro_auc(proba, dev_labels)
            boot = macro_draws(proba)
            oof = stored[f"{name}@{k}__purity_oof"]
            oof_macro, oof_per = c.macro_purity(oof, dev_labels)
            row = {"k": k, "macro_auc": macro, "auc_per_condition": per.tolist(), "auc_interval": interval(boot),
                   "purity_oof_macro": oof_macro, "purity_oof_per_condition": oof_per.tolist(),
                   "n_convergence_warnings": int(stored[f"{name}@{k}__warnings"])}
            if k == 4 and idea["kind"] in VIEW_KINDS:
                row["vs_everything4"] = {"difference": macro - reference_auc, "interval": interval(boot - reference_draws)}
                row["purity_oof_vs_everything4"] = {"difference": oof_macro - reference_purity_macro,
                                                    "interval": interval(purity_draws(oof) - reference_purity_draws)}
            if idea["kind"] == "space":
                row["purity_in_sample_macro"] = c.macro_purity(c.purity(c.leading(idea["_X"], k), dev_labels), dev_labels)[0]
            entry["readouts"].append(row)
        entry["tried_at"] = time.strftime("%Y-%m-%d %H:%M")
        done["ideas"].append(entry)
        first = entry["readouts"][0]
        extra = (f", vs everything {first['vs_everything4']['difference']:+.3f} {np.round(first['vs_everything4']['interval'], 3)}"
                 f", purity {first['purity_oof_vs_everything4']['difference']:+.2f}"
                 f" {np.round(first['purity_oof_vs_everything4']['interval'], 2)}") if "vs_everything4" in first else ""
        print(f"{idea['name']} [{idea['role']}]: macro AUC (k={first['k']}) {first['macro_auc']:.3f} "
              f"{np.round(first['auc_interval'], 3)}{extra}", flush=True)
    done["dev_slides"] = int(d["dev"].sum())
    done["fold_seed"], done["bootstrap_seed"], done["n_draws"] = DEV_FOLD_SEED, DEV_BOOT_SEED, N_DRAWS
    OUT_JSON.write_text(json.dumps(done, indent=1) + "\n")
    np.savez(OUT_NPZ, **stored)
    print(f"{len(ideas)} ideas in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
