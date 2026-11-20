#!/usr/bin/env python
"""PLS-DA in depth, on the development slides only (exploratory, after the improvement phase).

The improvement phase sent ``pls4-everything50`` to the holdout: class-balanced PLS-DA to four
components of the first 50 components of the ``everything`` MFA, z-scored and fitted inside each
training fold. This script asks how that result depends on its settings and its input. Everything
runs on the 476 development slides with the improvement phase's folds (20 x stratified 5-fold,
seed 1) and bootstrap draws (4000, seed 54321). The holdout, scored once, is not used.

Parts (every readout is the main experiment's logistic regression on the view, out of fold):

* ``grid``: k_in in {4, 10, 20, 50, 100, 200} input components of the ``everything`` MFA x n_out
  in {1, 2, 3, 4, 6, 8} PLS components; plus the logistic regression on the k_in components
  directly, without PLS (the same input, no supervised projection);
* ``preprocessing`` at (50, 4): z-scoring the components on or off x the class targets as plain
  indicators, divided by sqrt(class size) (the default), or by the class size;
* ``inputs`` at n_out = 4:
  - the ``everything``, ``groups`` and ``all-cell`` MFAs, each cell group's and each model's
    (over all point sets), first 50 components;
  - D, C, D+C, and D+C with ``everything``'s first 50 components;
  - the ``everything`` blocks with D, or with D+C, regressed out;
* ``nested``: k_in in {10, 20, 50, 100, 200} and n_out in {2, 4, 6} chosen in each training fold by
  an inner stratified 5-fold (macro AUC), then refitted on the training fold: the estimate of a
  tuned PLS-DA;
* ``axes``: PLS-DA (50, 4) fitted on all development slides:
  - each axis's share of loading mass by point set, model and slice;
  - the conditions' mean scores on each axis;
  - each axis's cross-validated R^2 on D, C and D+C;
  - over the 100 training-fold fits, the spread of the point-set shares and how well each fold's
    axis scores agree with the full fit's.

Gates: the default setting must reproduce ``improve.py``'s ``pls4-everything50`` and the residual
input its ``pls4-residual-density-composition`` (out-of-fold probabilities equal).

Writes ``results/pls_study.json`` and ``.npz``. Needs no topocyte import.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import common as c  # noqa: E402
import improve as imp  # noqa: E402

OUT_JSON = c.RESULTS / "pls_study.json"
OUT_NPZ = c.RESULTS / "pls_study.npz"
K_IN = (4, 10, 20, 50, 100, 200)
N_OUT = (1, 2, 3, 4, 6, 8)
NESTED_K_IN = (10, 20, 50, 100, 200)
NESTED_N_OUT = (2, 4, 6)
REFERENCE = "everything k50 n4 z sqrt"
N_CLASSES = len(c.GROUP_ORDER)


def target_weights(y: np.ndarray, mode: str, n_classes: int = N_CLASSES) -> np.ndarray:
    """Per-class factor on the 0/1 class targets: 1 (plain), 1/sqrt(size) (sqrt) or 1/size (inverse)."""
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(np.float64), 1.0)
    if mode == "plain":
        return np.ones(n_classes)
    if mode == "sqrt":
        return 1.0 / np.sqrt(counts)
    if mode == "inverse":
        return 1.0 / counts
    raise ValueError(mode)


class PLSView:
    """PLS-DA to ``n_out`` components, fitted on the training slides; z-scoring (``whiten``) and target weights as set."""

    def __init__(self, n_out: int, whiten: bool = True, targets: str = "sqrt") -> None:
        self.n_out, self.whiten, self.targets = n_out, whiten, targets

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSView":
        from sklearn.cross_decomposition import PLSRegression

        self.mean = X.mean(axis=0)
        self.sd = X.std(axis=0) if self.whiten else np.ones(X.shape[1])
        self.sd[self.sd == 0] = 1.0
        Y = np.eye(N_CLASSES)[y] * target_weights(y, self.targets)
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model = PLSRegression(n_components=self.n_out, scale=False).fit((X - self.mean) / self.sd, Y)
        self.n_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform((X - self.mean) / self.sd)


def axis_block_shares(rotations: np.ndarray, comp_sd: np.ndarray, vt: np.ndarray, block_size: int) -> np.ndarray:
    """Each axis's share of its squared loading mass on each block of the MFA-weighted concatenation.

    ``rotations`` (k_in, n_axes) map the z-scored components to the axes; ``comp_sd`` are the SDs
    they were z-scored with; the rows of ``vt`` are the components' directions in the weighted
    concatenation, whose columns come in consecutive blocks of ``block_size``.
    """
    directions = vt[: rotations.shape[0]].T @ (rotations / comp_sd[:, None])
    per_block = (directions**2).reshape(-1, block_size, rotations.shape[1]).sum(axis=1)
    return (per_block / per_block.sum(axis=0)).T


def axis_block_contributions(Xw: np.ndarray, rotations: np.ndarray, comp_sd: np.ndarray, vt: np.ndarray,
                             block_size: int) -> np.ndarray:
    """Each axis's share of variance from each block: var(X_b u_b) over its sum across blocks.

    The axis score is the sum over blocks of the block's (MFA-weighted, centred) columns times the
    axis's direction on them; each block's part has a variance, and the shares ignore the
    covariances between the parts. ``Xw`` is the weighted concatenation of the slides analysed.
    """
    directions = vt[: rotations.shape[0]].T @ (rotations / comp_sd[:, None])
    centred = Xw - Xw.mean(axis=0)
    n_blocks = Xw.shape[1] // block_size
    parts = np.array([(centred[:, b * block_size:(b + 1) * block_size] @ directions[b * block_size:(b + 1) * block_size]).var(axis=0)
                      for b in range(n_blocks)])
    return (parts / parts.sum(axis=0)).T


# ---------------------------------------------------------------- inputs


def build_inputs(d: dict) -> dict:
    """Every input matrix (all 594 slides; unsupervised steps only) and the everything MFA's directions."""
    sets = c.POINT_SETS
    blocks = imp.blocks_list(d, sets)
    keys = [(p, m, k) for p in sets for m in c.MODEL_NAMES for k in range(c.N_SLICES)]
    if len(blocks) != len(keys):
        raise RuntimeError("a block is missing")
    weights = np.array([1.0 / np.sqrt(b[:, 0].var()) for b in blocks])
    weighted = np.hstack([b * w for b, w in zip(blocks, weights)])
    scores, _, vt = c.pca(weighted)
    DC = np.hstack([d["D"], d["C"]])
    inputs = {
        "everything": scores[:, :200],
        "groups": imp.mfa_of(imp.blocks_list(d, c.CELL_GROUPS)),
        "all-cell": imp.mfa_of(imp.blocks_list(d, ("all",))),
        **{f"{g} (all models)": imp.mfa_of(imp.blocks_list(d, (g,))) for g in c.CELL_GROUPS},
        **{f"{m} (all point sets)": imp.mfa_of(imp.blocks_list(d, sets, (m,))) for m in c.MODEL_NAMES},
        "D": d["D"],
        "C": d["C"],
        "D+C": DC,
        "D+C + everything50": np.hstack([DC, scores[:, :50]]),
        "everything, density removed": imp.mfa_of([imp.residualize(b, d["D"]) for b in blocks]),
        "everything, density and composition removed": imp.mfa_of([imp.residualize(b, DC) for b in blocks]),
    }
    return {"inputs": inputs, "vt": vt, "keys": keys, "weighted": weighted}


def configurations(inputs: dict) -> list[dict]:
    """Every (input, k_in, n_out, whiten, targets) of the grid, preprocessing and input parts; and the direct fits."""
    configs = []

    def add(part: str, source: str, k_in: int, n_out: int | None, whiten: bool = True, targets: str = "sqrt") -> None:
        if n_out is None:
            name = f"{source} k{k_in} logistic"
        else:
            name = f"{source} k{k_in} n{n_out} {'z' if whiten else 'raw'} {targets}"
        if all(cfg["name"] != name for cfg in configs):
            configs.append({"name": name, "part": part, "input": source, "k_in": k_in, "n_out": n_out,
                            "whiten": whiten, "targets": targets})

    for k_in in K_IN:
        add("grid", "everything", k_in, None)
        for n_out in N_OUT:
            if n_out <= k_in:
                add("grid", "everything", k_in, n_out)
    for whiten in (True, False):
        for targets in ("plain", "sqrt", "inverse"):
            add("preprocessing", "everything", 50, 4, whiten, targets)
    for source, X in inputs.items():
        if source != "everything":
            # MFA inputs give 50 components; feature inputs (D, C, D+C, D+C + everything50) keep every column.
            add("inputs", source, X.shape[1] if source.startswith(("D", "C")) else min(50, X.shape[1]), 4)
    return configs


# ---------------------------------------------------------------- workers

_STATE: dict = {}


def _init(labels: np.ndarray, folds: np.ndarray) -> None:
    _STATE["labels"], _STATE["folds"] = labels, folds


def _run(task: tuple) -> tuple[np.ndarray, int, np.ndarray, int]:
    X, n_out, whiten, targets = task
    labels, folds = _STATE["labels"], _STATE["folds"]
    pls_warnings = [0]
    if n_out is None:
        def view(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return X[train], X[test]
    else:
        def view(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            fitted = PLSView(n_out, whiten, targets).fit(X[train], labels[train])
            pls_warnings[0] += fitted.n_warnings
            return fitted.transform(X[train]), fitted.transform(X[test])
    mean_proba, _, n_warnings, purity = imp.cv_view(view, labels, folds)
    return mean_proba, n_warnings, purity, pls_warnings[0]


def _predict(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray) -> np.ndarray:
    model, mean, sd, _ = imp.fit_logistic(Z_train, y_train)
    proba = np.zeros((len(Z_test), N_CLASSES))
    proba[:, model.classes_] = model.predict_proba((Z_test - mean) / sd)
    return proba


def _nested(task: tuple) -> tuple[int, int, int, int, np.ndarray, int]:
    """One outer fold: choose (k_in, n_out) by an inner stratified 5-fold, refit, predict the test fold."""
    from sklearn.model_selection import StratifiedKFold

    r, f, X = task
    labels, folds = _STATE["labels"], _STATE["folds"]
    test = folds[r] == f
    X_train, y_train = X[~test], labels[~test]
    inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + 10 * r + f).split(X_train, y_train))
    best: tuple[float, int, int] | None = None
    pls_warnings = 0
    for k_in in NESTED_K_IN:
        for n_out in NESTED_N_OUT:
            proba = np.zeros((len(y_train), N_CLASSES))
            for inner_train, inner_test in inner:
                fitted = PLSView(n_out).fit(X_train[inner_train, :k_in], y_train[inner_train])
                pls_warnings += fitted.n_warnings
                proba[inner_test] = _predict(fitted.transform(X_train[inner_train, :k_in]), y_train[inner_train],
                                             fitted.transform(X_train[inner_test, :k_in]))
            score = c.macro_auc(proba, y_train)[0]
            if best is None or score > best[0]:
                best = (score, k_in, n_out)
    _, k_in, n_out = best
    fitted = PLSView(n_out).fit(X_train[:, :k_in], y_train)
    proba = _predict(fitted.transform(X_train[:, :k_in]), y_train, fitted.transform(X[test, :k_in]))
    return r, f, k_in, n_out, proba, pls_warnings + fitted.n_warnings


# ---------------------------------------------------------------- axes


def cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Cross-validated R^2 (5-fold) of a linear regression of y on the z-scored X."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    prediction = cross_val_predict(LinearRegression(), imp.zscore(X), y, cv=KFold(5, shuffle=True, random_state=0))
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def axes_analysis(X: np.ndarray, vt: np.ndarray, keys: list, labels: np.ndarray, folds: np.ndarray,
                  D: np.ndarray, C: np.ndarray, weighted: np.ndarray) -> dict:
    full = PLSView(4).fit(X, labels)
    scores = full.transform(X)
    mass = axis_block_shares(full.model.x_rotations_, full.sd, vt, c.N_COMPONENTS)
    shares = axis_block_contributions(weighted, full.model.x_rotations_, full.sd, vt, c.N_COMPONENTS)

    def by(index: int, values: tuple, table: np.ndarray | None = None) -> np.ndarray:
        table = shares if table is None else table
        return np.array([[table[a, [i for i, key in enumerate(keys) if key[index] == v]].sum() for v in values]
                         for a in range(table.shape[0])])

    standardized = scores / scores.std(axis=0)
    condition_means = np.array([standardized[labels == g].mean(axis=0) for g in range(len(c.CONDITIONS))])
    r2 = {name: [cv_r2(F, scores[:, a]) for a in range(scores.shape[1])]
          for name, F in (("D", D), ("C", C), ("D+C", np.hstack([D, C])))}
    fold_sets, agreement = [], []
    for r in range(folds.shape[0]):
        for f in np.unique(folds[r]):
            train = folds[r] != f
            fitted = PLSView(4).fit(X[train], labels[train])
            fold_shares = axis_block_contributions(weighted, fitted.model.x_rotations_, fitted.sd, vt, c.N_COMPONENTS)
            fold_sets.append([[fold_shares[a, [i for i, key in enumerate(keys) if key[0] == p]].sum() for p in c.POINT_SETS]
                              for a in range(4)])
            fold_scores = fitted.transform(X)
            agreement.append([abs(float(np.corrcoef(fold_scores[:, a], scores[:, a])[0, 1])) for a in range(4)])
    fold_sets, agreement = np.array(fold_sets), np.array(agreement)
    return {
        "measure": "share of each axis's variance from each block (variance of the block's part of the axis score)",
        "by_point_set": by(0, c.POINT_SETS).tolist(),
        "mass_by_point_set": by(0, c.POINT_SETS, mass).tolist(),
        "mass_by_model": by(1, c.MODEL_NAMES, mass).tolist(),
        "by_model": by(1, c.MODEL_NAMES).tolist(),
        "by_slice": by(2, tuple(range(c.N_SLICES))).tolist(),
        "condition_means": condition_means.tolist(),
        "r2": r2,
        "fold_point_set_mean": fold_sets.mean(axis=0).tolist(),
        "fold_point_set_sd": fold_sets.std(axis=0).tolist(),
        "fold_agreement_median": np.median(agreement, axis=0).tolist(),
        "fold_agreement_p10": np.percentile(agreement, 10, axis=0).tolist(),
    }


# ---------------------------------------------------------------- main


def main() -> None:
    started = time.time()
    d = imp.load()
    built = build_inputs(d)
    inputs, dev = built["inputs"], d["dev"]
    labels = d["labels"][dev]
    folds = c.make_folds(labels, n_repeats=20, n_splits=5, seed=imp.DEV_FOLD_SEED)
    draws = c.stratified_bootstrap(labels, imp.N_DRAWS, np.random.default_rng(imp.DEV_BOOT_SEED))
    configs = configurations(inputs)
    tasks = [(c.leading(inputs[cfg["input"]], cfg["k_in"])[dev], cfg["n_out"], cfg["whiten"], cfg["targets"])
             for cfg in configs]
    print(f"{len(configs)} configurations x 100 fits", flush=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(labels, folds)) as pool:
        fitted = list(pool.map(_run, tasks))
        nested_tasks = [(r, f, inputs["everything"][dev]) for r in range(folds.shape[0]) for f in np.unique(folds[r])]
        nested = list(pool.map(_nested, nested_tasks))
    print(f"fits done in {time.time() - started:.1f} s", flush=True)

    # Gates: the default setting and the residual input reproduce improve.py's development probabilities.
    with np.load(imp.OUT_NPZ) as stored:
        reference_everything4 = stored["baseline:everything@4"]
        gates = {
            REFERENCE: float(np.max(np.abs(fitted[[cfg["name"] for cfg in configs].index(REFERENCE)][0]
                                           - stored["pls4-everything50@4"]))),
            "everything, density and composition removed k50 n4 z sqrt": float(np.max(np.abs(
                fitted[[cfg["name"] for cfg in configs].index("everything, density and composition removed k50 n4 z sqrt")][0]
                - stored["pls4-residual-density-composition@4"]))),
        }
    if max(gates.values()) > 1e-9:
        raise RuntimeError(f"PLS study does not reproduce the improvement phase: {gates}")

    def macro_draws(proba: np.ndarray) -> np.ndarray:
        return np.array([c.macro_auc(proba[idx], labels[idx])[0] for idx in draws])

    def purity_draws(values: np.ndarray) -> np.ndarray:
        return np.mean([values[draws[:, labels == g]].mean(axis=1) for g in range(len(c.CONDITIONS))], axis=0)

    reference_index = [cfg["name"] for cfg in configs].index(REFERENCE)
    ref_proba, _, ref_purity, _ = fitted[reference_index]
    ref_draws, e4_draws = macro_draws(ref_proba), macro_draws(reference_everything4)
    ref_auc, e4_auc = c.macro_auc(ref_proba, labels)[0], c.macro_auc(reference_everything4, labels)[0]
    ref_purity_draws = purity_draws(ref_purity)
    rows, arrays = [], {}
    for i, (cfg, (proba, n_warn, purity, pls_warn)) in enumerate(zip(configs, fitted)):
        macro, per = c.macro_auc(proba, labels)
        boot = macro_draws(proba)
        purity_macro = c.macro_purity(purity, labels)[0]
        rows.append({**cfg, "macro_auc": macro, "auc_interval": imp.interval(boot), "auc_per_condition": per.tolist(),
                     "vs_healthy": c.vs_healthy_auc(proba, labels).tolist(),
                     "vs_reference": {"difference": macro - ref_auc, "interval": imp.interval(boot - ref_draws)},
                     "vs_everything4": {"difference": macro - e4_auc, "interval": imp.interval(boot - e4_draws)},
                     "purity_oof_macro": purity_macro,
                     "purity_vs_reference": {"difference": purity_macro - c.macro_purity(ref_purity, labels)[0],
                                             "interval": imp.interval(purity_draws(purity) - ref_purity_draws)},
                     "n_convergence_warnings": int(n_warn), "n_pls_convergence_warnings": int(pls_warn)})
        arrays[f"c{i}__proba"] = proba
        arrays[f"c{i}__purity"] = purity

    # Nested: one prediction per slide and repeat, averaged over repeats.
    nested_proba = np.zeros((folds.shape[0], len(labels), N_CLASSES))
    chosen: dict[str, int] = {}
    nested_pls_warnings = 0
    for r, f, k_in, n_out, proba, pls_warn in nested:
        nested_pls_warnings += pls_warn
        nested_proba[r][folds[r] == f] = proba
        chosen[f"k{k_in} n{n_out}"] = chosen.get(f"k{k_in} n{n_out}", 0) + 1
    nested_mean = nested_proba.mean(axis=0)
    nested_boot = macro_draws(nested_mean)
    nested_auc = c.macro_auc(nested_mean, labels)[0]
    arrays["nested__proba"] = nested_mean
    nested_result = {
        "macro_auc": nested_auc, "auc_interval": imp.interval(nested_boot),
        "vs_reference": {"difference": nested_auc - ref_auc, "interval": imp.interval(nested_boot - ref_draws)},
        "vs_everything4": {"difference": nested_auc - e4_auc, "interval": imp.interval(nested_boot - e4_draws)},
        "chosen": dict(sorted(chosen.items(), key=lambda item: -item[1])),
        "grid": {"k_in": list(NESTED_K_IN), "n_out": list(NESTED_N_OUT)},
        "n_pls_convergence_warnings": nested_pls_warnings,
    }
    axes = axes_analysis(c.leading(inputs["everything"], 50)[dev], built["vt"], built["keys"], labels, folds,
                         d["D"][dev], d["C"][dev], built["weighted"][dev])
    payload = {
        "dev_slides": int(dev.sum()), "fold_seed": imp.DEV_FOLD_SEED, "bootstrap_seed": imp.DEV_BOOT_SEED,
        "n_draws": imp.N_DRAWS, "reference": REFERENCE, "reference_auc": ref_auc, "everything4_auc": e4_auc,
        "gates_max_abs_proba_diff": gates, "rows": rows, "nested": nested_result, "axes": axes,
        "runtime_seconds": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
    np.savez(OUT_NPZ, names=np.array([cfg["name"] for cfg in configs]), labels=labels, **arrays)
    print(f"reference {ref_auc:.3f}; everything@4 {e4_auc:.3f}; nested {nested_auc:.3f} "
          f"{np.round(nested_result['auc_interval'], 3)}; chosen {nested_result['chosen']}; "
          f"{time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
