#!/usr/bin/env python
"""Both separability readouts for every space, and the comparisons fixed in advance (design.md, Pipeline 6-7).

Jobs:

* every space of ``results/spaces.npz`` on its leading 4 components, and on
  20 for MFA spaces, 9 for single blocks, every component for the baselines;
* the ladder's feature sets, logistic regression only: the ``D``, ``C`` and
  ``D+C`` features, and the ``D+C`` features with the first 20 components of
  ``all-cell``, ``groups`` or ``everything``.

Logistic regression: ``common.cv_probabilities`` on the folds of
``common.make_folds`` (20 repeats of stratified 5-fold, seed 0), the same
folds for every job; macro AUC, each condition's one-vs-rest AUC and each
condition against healthy from the out-of-fold probabilities averaged over
the repeats, and the per-repeat macro AUCs. Purity: ``common.purity`` on the
leading components as fitted.

Comparisons (``results/comparisons.json``): 4000 bootstrap draws of slides
stratified by group (``common.stratified_bootstrap``, ``default_rng(12345)``),
the same draws for every comparison, 95% percentile intervals:

* primary, fixed in advance: ``everything`` against ``all-cell`` at k = 4,
  difference in macro AUC and in macro purity, and ``common.verdict``; the
  same at k = 20, descriptive;
* secondary, fixed in advance: ``D+C`` features plus the first 20
  components of ``groups`` against the ``D+C`` features alone, macro AUC;
* the ladder: every rung's macro AUC with its interval, and paired
  differences between neighbouring rungs;
* descriptive: the composition's PCA (``C``) against ``everything`` on four
  components, both readouts;
* each condition against healthy for the key spaces.

Writes ``results/folds.npz``, ``results/readouts.npz``,
``results/readouts.json`` and ``results/comparisons.json``. Needs no topocyte
import.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process; the pool supplies the parallelism.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import common as c  # noqa: E402

N_REPEATS = 20
N_SPLITS = 5
FOLD_SEED = 0
N_DRAWS = 4000
BOOT_SEED = 12345
N_NEIGHBORS = 15
SMALL_K = 4
LARGE_K = 20
LADDER = (
    ("features:D", "density summaries D"),
    ("features:C", "composition C"),
    ("features:D+C", "D + C"),
    ("all-cell@20", "all-cell topology (20)"),
    ("groups@20", "per-group topology (20)"),
    ("everything@20", "everything (20)"),
    ("features:D+C+all-cell20", "D + C + all-cell (20)"),
    ("features:D+C+groups20", "D + C + per-group (20)"),
    ("features:D+C+everything20", "D + C + everything (20)"),
)
LADDER_DIFFERENCES = (
    ("features:D+C+groups20", "features:D+C"),
    ("features:D+C+all-cell20", "features:D+C"),
    ("features:D+C+everything20", "features:D+C+all-cell20"),
    ("everything@20", "all-cell@20"),
    ("groups@20", "all-cell@20"),
)

_LABELS: np.ndarray | None = None
_FOLDS: np.ndarray | None = None


def _init(labels: np.ndarray, folds: np.ndarray) -> None:
    global _LABELS, _FOLDS
    _LABELS, _FOLDS = labels, folds


def _fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mean_proba, repeat_proba, n_warnings = c.cv_probabilities(X, _LABELS, _FOLDS)
    repeat_auc = np.array([c.macro_auc(p, _LABELS)[0] for p in repeat_proba])
    return mean_proba, repeat_auc, n_warnings


def build_jobs(data: np.lib.npyio.NpzFile, aliases: dict[str, str]) -> list[tuple[str, str, int, str, np.ndarray]]:
    """(job, space or "", k, kind, X) for every job, in a fixed order."""
    names = [str(n) for n in data["names"]]
    jobs = []
    for i, name in enumerate(names):
        scores = data[f"s{i}__scores"]
        kind = name.split(":", 1)[0] if ":" in name else "baseline"
        large = LARGE_K if kind == "mfa" else scores.shape[1]
        for k in (SMALL_K, large):
            jobs.append((f"{name}@{k}", name, k, kind, c.leading(scores, k)))
    D, C = data["D_features"], data["C_features"]
    DC = np.hstack([D, C])
    jobs += [("features:D", "", D.shape[1], "features", D), ("features:C", "", C.shape[1], "features", C),
             ("features:D+C", "", DC.shape[1], "features", DC)]
    for alias in ("all-cell", "groups", "everything"):
        scores = data[f"s{names.index(aliases[alias])}__scores"]
        jobs.append((f"features:D+C+{alias}20", "", DC.shape[1] + LARGE_K, "features", np.hstack([DC, c.leading(scores, LARGE_K)])))
    return jobs


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    started = time.time()
    aliases = json.loads((c.RESULTS / "spaces.json").read_text())["aliases"]
    with np.load(c.RESULTS / "spaces.npz", allow_pickle=False) as data:
        labels = data["labels"]
        jobs = build_jobs(data, aliases)
    folds = c.make_folds(labels, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=FOLD_SEED)
    np.savez(c.RESULTS / "folds.npz", folds=folds, labels=labels)
    print(f"{len(jobs)} jobs x {N_REPEATS * N_SPLITS} fits", flush=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(labels, folds)) as pool:
        fitted = list(pool.map(_fit, [job[4] for job in jobs], chunksize=4))
    print(f"logistic regression done in {time.time() - started:.1f} s", flush=True)

    n = len(jobs)
    proba = np.stack([f[0] for f in fitted])
    repeat_auc = np.stack([f[1] for f in fitted])
    warnings_count = np.array([f[2] for f in fitted])
    purity = np.full((n, labels.size), np.nan)
    rows = []
    for j, (job, space, k, kind, X) in enumerate(jobs):
        macro, per = c.macro_auc(proba[j], labels)
        row = {
            "job": job, "space": space, "k": int(X.shape[1]), "kind": kind, "macro_auc": macro,
            "auc_per_condition": per.tolist(), "auc_repeat_mean": float(repeat_auc[j].mean()),
            "auc_repeat_sd": float(repeat_auc[j].std(ddof=1)), "vs_healthy": c.vs_healthy_auc(proba[j], labels).tolist(),
            "n_convergence_warnings": int(warnings_count[j]),
        }
        if kind != "features":
            purity[j] = c.purity(X, labels, n_neighbors=N_NEIGHBORS)
            row["purity_macro"], per_purity = c.macro_purity(purity[j], labels)
            row["purity_per_condition"] = per_purity.tolist()
        rows.append(row)
    np.savez(
        c.RESULTS / "readouts.npz",
        jobs=np.array([job[0] for job in jobs]),
        proba=proba,
        repeat_auc=repeat_auc,
        purity=purity,
        n_convergence_warnings=warnings_count,
        labels=labels,
    )
    (c.RESULTS / "readouts.json").write_text(json.dumps({"conditions": list(c.CONDITIONS), "rows": rows}, indent=1) + "\n")

    # Comparisons, on one set of bootstrap draws.
    index = {job[0]: j for j, job in enumerate(jobs)}
    for alias, full in aliases.items():
        for k in (SMALL_K, LARGE_K):
            index[f"{alias}@{k}"] = index[f"{full}@{k}"]
    draws = c.stratified_bootstrap(labels, N_DRAWS, np.random.default_rng(BOOT_SEED))
    boot_auc = {}

    def macro_draws(job: str) -> np.ndarray:
        if job not in boot_auc:
            p = proba[index[job]]
            boot_auc[job] = np.array([c.macro_auc(p[idx], labels[idx])[0] for idx in draws])
        return boot_auc[job]

    def purity_draws(job: str) -> np.ndarray:
        values = purity[index[job]]
        per = [values[draws[:, labels == g]].mean(axis=1) for g in range(len(c.CONDITIONS))]
        # draws[:, labels == g] are the resampled members of g: stratification keeps them in g's positions
        return np.mean(per, axis=0)

    def auc_of(job: str) -> float:
        return c.macro_auc(proba[index[job]], labels)[0]

    def purity_of(job: str) -> float:
        return c.macro_purity(purity[index[job]], labels)[0]

    def compare(a: str, b: str, with_purity: bool) -> dict:
        out = {"a": a, "b": b, "auc": {"a": auc_of(a), "b": auc_of(b), "difference": auc_of(a) - auc_of(b),
                                       "interval": interval(macro_draws(a) - macro_draws(b))}}
        if with_purity:
            out["purity"] = {"a": purity_of(a), "b": purity_of(b), "difference": purity_of(a) - purity_of(b),
                             "interval": interval(purity_draws(a) - purity_draws(b))}
        return out

    comparisons: dict[str, object] = {"n_draws": N_DRAWS, "seed": BOOT_SEED, "aliases": aliases}
    for k in (SMALL_K, LARGE_K):
        result = compare(f"everything@{k}", f"all-cell@{k}", with_purity=True)
        result["verdict"] = c.verdict(result["auc"]["interval"], result["purity"]["interval"])
        result["below_zero"] = {"auc": result["auc"]["interval"][1] < 0, "purity": result["purity"]["interval"][1] < 0}
        result["fixed_in_advance"] = k == SMALL_K
        comparisons[f"primary_k{k}"] = result
    # Descriptive: the composition's own PCA on the same four components, against everything.
    comparisons["composition_pca_k4"] = compare("C@4", "everything@4", with_purity=True)
    secondary = compare("features:D+C+groups20", "features:D+C", with_purity=False)
    secondary["adds"] = secondary["auc"]["interval"][0] > 0
    comparisons["secondary"] = secondary
    comparisons["ladder"] = [{"job": job, "label": label, "auc": auc_of(job), "interval": interval(macro_draws(job)),
                              "purity": purity_of(job) if not job.startswith("features:") else None}
                             for job, label in LADDER]
    comparisons["ladder_differences"] = [compare(a, b, with_purity=False) for a, b in LADDER_DIFFERENCES]
    healthy = labels == 0
    vs_healthy = {}
    for job in ("all-cell@4", "everything@4", "all-cell@20", "groups@20", "everything@20", "features:D+C",
                "features:D+C+groups20"):
        p = np.log(np.clip(proba[index[job]], 1e-300, None))
        entries = []
        for cond in range(1, len(c.CONDITIONS)):
            score = p[:, cond] - p[:, 0]
            point = c.auc(score[labels == cond], score[healthy])
            boot = [c.auc(score[idx][labels[idx] == cond], score[idx][labels[idx] == 0]) for idx in draws]
            entries.append({"condition": c.CONDITIONS[cond], "auc": point, "interval": interval(np.array(boot))})
        vs_healthy[job] = entries
    comparisons["vs_healthy"] = vs_healthy
    comparisons["runtime_seconds"] = round(time.time() - started, 1)
    (c.RESULTS / "comparisons.json").write_text(json.dumps(comparisons, indent=1) + "\n")

    p4 = comparisons["primary_k4"]
    print(f"primary (k=4): AUC everything {p4['auc']['a']:.3f} vs all-cell {p4['auc']['b']:.3f}, "
          f"diff {p4['auc']['difference']:+.3f} {np.round(p4['auc']['interval'], 3)}; purity diff "
          f"{p4['purity']['difference']:+.3f} {np.round(p4['purity']['interval'], 3)} -> {p4['verdict']}")
    print(f"secondary: AUC D+C+groups20 {secondary['auc']['a']:.3f} vs D+C {secondary['auc']['b']:.3f}, diff "
          f"{secondary['auc']['difference']:+.3f} {np.round(secondary['auc']['interval'], 3)} -> adds {secondary['adds']}")
    print(f"convergence warnings: {int(warnings_count.sum())} in {int(np.count_nonzero(warnings_count))} jobs; "
          f"{time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
