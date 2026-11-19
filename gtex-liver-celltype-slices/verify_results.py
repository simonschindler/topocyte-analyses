#!/usr/bin/env python
"""Independent checks of every result the report uses.

Each check recomputes a saved result by a route other than the one that made
it:

* composition: the cohort's cell types counted per slide, group and slice
  by one flat ``np.bincount`` per slice, against ``composition.npz`` and the
  sibling's kept cells;
* gates: both all-cell gates passed;
* slice 9: each group's fPCA through the dense route (``tc.ph``'s exact
  pairwise ``betti_inner_product_matrix`` and ``fpca_from_gram``) against the
  saved scores of the matrix-free fit;
* spaces: every named space and 20 others recomputed by eigendecomposition
  of the sum of the blocks' weighted, centred Gram matrices; the baselines'
  features recomputed from their definitions;
* readouts: folds redrawn; the logistic regression of the named spaces at
  k = 4 refitted through a scikit-learn pipeline; every saved macro AUC
  recounted pair by pair (Mann-Whitney) from the saved probabilities; purity
  of the named spaces by explicit loops;
* comparisons: the primary and secondary differences, intervals and
  verdicts recomputed from the saved arrays with the bootstrap redrawn by
  hand.

Run with ``PYTHONPATH=<topocyte checkout>`` (for the slice-9 check); prints
``ALL CHECKS PASSED`` or stops at the first failure.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import common as c

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("ok   " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILURES.append(message)


def corr_ok(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    return min(abs(float(np.corrcoef(a[:, j], b[:, j])[0, 1])) for j in range(a.shape[1]))


def check_composition() -> None:
    slides, bounds = c.read_slides()
    groups, _ = c.read_cell_groups()
    slide_of_cell = np.repeat(np.arange(c.N_SLIDES), np.diff(bounds))
    with np.load(c.SOURCE / "density.npz", allow_pickle=False) as density:
        rho, deciles, sibling = density["rho"], density["deciles"], density["kept_per_slide"]
    with np.load(c.RESULTS / "composition.npz", allow_pickle=False) as composition:
        saved = composition["kept_by_group"]
        check([str(s) for s in composition["slide_ids"]] == slides, "composition lists the cohort's slides in order")
    recount = np.empty_like(saved)
    for k in range(c.N_SLICES):
        keep = np.ones(rho.size, dtype=bool) if k == 0 else rho >= deciles[k]
        flat = slide_of_cell[keep].astype(np.int64) * len(c.CELL_GROUPS) + groups[keep]
        recount[k] = np.bincount(flat, minlength=c.N_SLIDES * len(c.CELL_GROUPS)).reshape(c.N_SLIDES, -1)
    check(np.array_equal(recount, saved), "cells per slide, group and slice recounted")
    check(np.array_equal(saved.sum(axis=2), sibling), "group counts add up to the sibling's kept cells")


def check_gates() -> None:
    for k in (9, 8):
        gate = json.loads((c.RESULTS / f"gate_{k}.json").read_text())
        check(gate["passed"] and gate["n_bars_equal"], f"all-cell gate at slice {k} passed")


def check_slice9_dense() -> None:
    import topocyte as tc

    with np.load(c.SOURCE / "density.npz", allow_pickle=False) as density:
        keep = density["rho"] >= density["deciles"][9]
        bounds = density["bounds"]
    groups, _ = c.read_cell_groups()
    coords = c.read_coords()
    with np.load(c.RESULTS / "slice_9.npz", allow_pickle=False) as saved:
        saved = {key: saved[key] for key in saved.files}
    from concurrent.futures import ThreadPoolExecutor

    for j, group in enumerate(c.CELL_GROUPS):
        mask = keep & (groups == j)

        def compute(code: int) -> object:
            start, end = bounds[code], bounds[code + 1]
            return tc.ph.compute_persistent_homology(coords[start:end][mask[start:end]].astype(np.float64),
                                                     filtration="alpha", max_dim=1)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            results = list(pool.map(compute, range(c.N_SLIDES)))
        for dim in (0, 1):
            gram = tc.ph.betti_inner_product_matrix(results, dim=dim, n_jobs=-1)
            for normalization in ("none", "cosine"):
                name = f"h{dim}_{normalization}"
                result = tc.ph.fpca_from_gram(gram, normalization=normalization)
                dense = np.asarray(result.scores)[:, : c.N_COMPONENTS]
                arpack = saved[f"{group}_{name}_scores"]
                r = corr_ok(dense, arpack, 0)
                fve = float(np.max(np.abs(np.asarray(result.explained_variance_ratio)[: c.N_COMPONENTS] - saved[f"{group}_{name}_evr"])))
                check(r >= 0.99999 and fve <= 1e-6, f"slice 9 {group} {name}: dense fit vs saved, min |r| {r:.8f}, dFVE {fve:.1e}")


def check_spaces() -> dict:
    aliases = json.loads((c.RESULTS / "spaces.json").read_text())["aliases"]
    with np.load(c.RESULTS / "spaces.npz", allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    names = [str(n) for n in saved["names"]]
    rng = np.random.default_rng(0)
    mfa_names = [n for n in names if n.startswith("mfa:")]
    picks = sorted(set(aliases.values()) | set(rng.choice(mfa_names, size=20, replace=False).tolist()))
    sources: dict[int, tuple] = {}
    for name in picks:
        i = names.index(name)
        keys = [str(k) for k in saved[f"s{i}__blocks"]]
        gram = np.zeros((c.N_SLIDES, c.N_SLIDES))
        for key in keys:
            p, m, k = key.split(":")
            k = int(k)
            if k not in sources:
                sources[k] = (np.load(c.SOURCE / f"slice_{k}.npz"), np.load(c.RESULTS / f"slice_{k}.npz"))
            block = sources[k][0][f"{m}_scores"] if p == "all" else sources[k][1][f"{p}_{m}_scores"]
            centred = (block - block.mean(axis=0)) / block[:, 0].std()
            gram += centred @ centred.T
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1][:20]
        expected = vectors[:, order] * np.sqrt(values[order])
        scores = saved[f"s{i}__scores"]
        r = corr_ok(scores, expected, 0)
        evr_ok = np.allclose(saved[f"s{i}__evr"][:20], values[order] / values[values > 0].sum(), rtol=1e-7, atol=1e-12)
        check(r >= 1 - 1e-9 and evr_ok, f"space {name}: eigh of summed Gram, min |r| {r:.12f}")
    with np.load(c.SOURCE / "density.npz", allow_pickle=False) as density, \
            np.load(c.RESULTS / "composition.npz", allow_pickle=False) as composition:
        kept = composition["kept_by_group"]
        total = kept.sum(axis=2)
        D = np.column_stack([np.log10(density["slide_medians"]), np.log10(total[0])] + [total[s] / total[0] for s in range(1, 10)])
        C = np.log10(1.0 + kept.transpose(1, 0, 2).reshape(c.N_SLIDES, -1))
    check(np.allclose(saved["D_features"], D) and np.allclose(saved["C_features"], C), "baseline features from their definitions")
    return {"names": names, "aliases": aliases, "saved": saved}


def check_readouts(spaces: dict) -> None:
    names, aliases, saved = spaces["names"], spaces["aliases"], spaces["saved"]
    labels = saved["labels"]
    with np.load(c.RESULTS / "folds.npz", allow_pickle=False) as data:
        folds = data["folds"]
    check(np.array_equal(folds, c.make_folds(labels, 20, 5, 0)), "folds redrawn identically")
    for row in folds:
        for g in range(len(c.GROUP_ORDER)):
            counts = np.bincount(row[labels == g], minlength=5)
            if counts.max() - counts.min() > 1:
                check(False, f"fold sizes of group {g} differ by more than one")
    with np.load(c.RESULTS / "readouts.npz", allow_pickle=False) as data:
        readouts = {key: data[key] for key in data.files}
    jobs = [str(j) for j in readouts["jobs"]]
    table = {row["job"]: row for row in json.loads((c.RESULTS / "readouts.json").read_text())["rows"]}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for alias in ("all-cell", "everything"):
        full = aliases[alias]
        X = saved[f"s{names.index(full)}__scores"][:, :4]
        proba = np.zeros((20, c.N_SLIDES, len(c.GROUP_ORDER)))
        for r in range(20):
            for f in range(5):
                test = folds[r] == f
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
                model.fit(X[~test], labels[~test])
                proba[r][np.ix_(np.flatnonzero(test), model.classes_)] = model.predict_proba(X[test])
        diff = float(np.max(np.abs(proba.mean(axis=0) - readouts["proba"][jobs.index(f"{full}@4")])))
        check(diff < 1e-6, f"{alias}@4: logistic regression refitted through a pipeline, max |dp| {diff:.1e}")

    def pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
        return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size))

    worst = 0.0
    for j, job in enumerate(jobs):
        p = readouts["proba"][j]
        scored = labels < 7
        per = [pairwise_auc(p[scored & (labels == k), k], p[scored & (labels != k), k]) for k in range(7)]
        worst = max(worst, abs(np.mean(per) - table[job]["macro_auc"]))
    check(worst < 1e-12, f"every macro AUC recounted pair by pair ({len(jobs)} jobs), max diff {worst:.1e}")

    for alias in ("all-cell", "everything", "groups"):
        for k in (4, 20):
            job = f"{aliases[alias]}@{k}"
            X = saved[f"s{names.index(aliases[alias])}__scores"][:, :k]
            counts = np.bincount(labels, minlength=8)
            loop = np.full(c.N_SLIDES, np.nan)
            for i in range(c.N_SLIDES):
                if labels[i] >= 7:
                    continue
                d = [(float(np.sqrt(((X[i] - X[o]) ** 2).sum())), o) for o in range(c.N_SLIDES) if o != i]
                d.sort()
                nearest = [o for _, o in d[:15]]
                loop[i] = np.mean(labels[nearest] == labels[i]) / ((counts[labels[i]] - 1) / (c.N_SLIDES - 1))
            saved_purity = readouts["purity"][jobs.index(job)]
            ok = np.allclose(loop[labels < 7], saved_purity[labels < 7], rtol=1e-12) and np.all(np.isnan(saved_purity[labels >= 7]))
            check(ok, f"{alias}@{k}: purity by explicit loops")
    check(all(np.isfinite(readouts["proba"]).ravel()), "every saved probability finite")


def check_comparisons() -> None:
    comparisons = json.loads((c.RESULTS / "comparisons.json").read_text())
    aliases = comparisons["aliases"]
    with np.load(c.RESULTS / "readouts.npz", allow_pickle=False) as data:
        jobs = [str(j) for j in data["jobs"]]
        proba, purity, labels = data["proba"], data["purity"], data["labels"]
    rng = np.random.default_rng(12345)
    draws = np.empty((4000, labels.size), dtype=np.int64)
    for g in np.unique(labels):
        members = np.flatnonzero(labels == g)
        draws[:, members] = rng.choice(members, size=(4000, members.size), replace=True)

    def job_index(job: str) -> int:
        name, _, k = job.rpartition("@")
        return jobs.index(f"{aliases.get(name, name)}@{k}") if "@" in job else jobs.index(job)

    def macro(p: np.ndarray, lab: np.ndarray) -> float:
        return c.macro_auc(p, lab)[0]

    for key, a, b, with_purity in (("primary_k4", "everything@4", "all-cell@4", True),
                                   ("secondary", "features:D+C+groups20", "features:D+C", False)):
        pa, pb = proba[job_index(a)], proba[job_index(b)]
        diffs = np.array([macro(pa[idx], labels[idx]) - macro(pb[idx], labels[idx]) for idx in draws])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        saved = comparisons[key]["auc"]
        ok = np.isclose(saved["difference"], macro(pa, labels) - macro(pb, labels)) and np.allclose(saved["interval"], [lo, hi])
        check(ok, f"{key}: AUC difference and interval recomputed")
        if with_purity:
            ua, ub = purity[job_index(a)], purity[job_index(b)]
            per = [(ua[draws[:, labels == g]] - ub[draws[:, labels == g]]).mean(axis=1) for g in range(7)]
            plo, phi = np.percentile(np.mean(per, axis=0), [2.5, 97.5])
            check(np.allclose(comparisons[key]["purity"]["interval"], [plo, phi]), f"{key}: purity interval recomputed")
            verdict = "enhanced" if lo > 0 and plo > 0 else "mixed" if (lo > 0) != (plo > 0) else "not enhanced"
            check(comparisons[key]["verdict"] == verdict, f"{key}: verdict {verdict} by the rule")
    adds = bool(comparisons["secondary"]["auc"]["interval"][0] > 0)
    check(comparisons["secondary"]["adds"] == adds, "secondary: adds-flag by the rule")


def main() -> None:
    started = time.time()
    check_composition()
    check_gates()
    spaces = check_spaces()
    check_readouts(spaces)
    check_comparisons()
    check_slice9_dense()
    print(f"{time.time() - started:.0f} s")
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} CHECKS FAILED")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
