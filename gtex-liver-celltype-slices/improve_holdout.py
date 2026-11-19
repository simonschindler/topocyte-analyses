#!/usr/bin/env python
"""Score the chosen ideas once on the holdout (design.md: the rulings after the builder check).

Selection, fixed before any development result existed:

* eligible: ideas with a four-dimensional view (an unsupervised space on its
  first four components, or a supervised projection to four components);
* ranked by development macro AUC at k = 4; at most three, each with a
  different mechanism;
* an unsupervised idea must keep purity: the interval of its purity
  difference to ``everything`` must not lie entirely below 0;
* if no idea's AUC interval against ``everything`` lies above 0, the best one
  still goes, as a check.

Holdout entries: the chosen ideas, ``all-cell`` and ``everything`` on four
components, the ``D+C`` features, and ``D+C`` with the first 20 components of
``groups``. Each is fitted on the 476 development slides (supervised
projection, standardization and logistic regression; unsupervised steps as
fitted on all slides) and predicts the 118 holdout slides once.

Readouts:
* macro AUC, and its paired difference to ``everything@4``, with 95%
  intervals from 4000 bootstrap draws of holdout slides stratified by group
  (seed 777);
* per-condition AUCs;
* purity of the holdout slides, with their 15 nearest neighbours among the
  development slides, over the chance share n_g / N_dev.

Writes ``results/improve_holdout.json`` once, and refuses to run again.
"""

from __future__ import annotations

import json
import time

import numpy as np

import common as c
import improve as imp

OUT = c.RESULTS / "improve_holdout.json"
HOLD_BOOT_SEED = 777


def choose(dev: dict) -> list[dict]:
    eligible = []
    for entry in dev["ideas"]:
        if entry["mechanism"] == "baseline" or entry["kind"] not in ("space", "supervised", "selection"):
            continue
        k4 = next(r for r in entry["readouts"] if r["k"] == 4)
        if entry["kind"] == "space" and k4["purity_vs_everything4"]["interval"][1] < 0:
            continue
        eligible.append((k4["macro_auc"], entry))
    eligible.sort(key=lambda pair: -pair[0])
    chosen, mechanisms = [], set()
    for _, entry in eligible:
        if entry["mechanism"] in mechanisms:
            continue
        chosen.append(entry)
        mechanisms.add(entry["mechanism"])
        if len(chosen) == 3:
            break
    return chosen


def fit_predict(X_dev: np.ndarray, y_dev: np.ndarray, X_hold: np.ndarray, supervised: tuple[str, int] | None) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    if supervised is not None:
        projector = imp.Projector(*supervised).fit(X_dev, y_dev)
        X_dev, X_hold = projector.transform(X_dev), projector.transform(X_hold)
    mean, sd = X_dev.mean(axis=0), X_dev.std(axis=0)
    sd[sd == 0] = 1.0
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit((X_dev - mean) / sd, y_dev)
    proba = np.zeros((len(X_hold), len(c.GROUP_ORDER)))
    proba[:, model.classes_] = model.predict_proba((X_hold - mean) / sd)
    return proba


def holdout_purity(X_dev: np.ndarray, y_dev: np.ndarray, X_hold: np.ndarray, y_hold: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    nearest = np.argsort(cdist(X_hold, X_dev), axis=1, kind="stable")[:, :15]
    counts = np.bincount(y_dev, minlength=len(c.GROUP_ORDER))
    out = (y_dev[nearest] == y_hold[:, None]).mean(axis=1) / (counts[y_hold] / y_dev.size)
    out[y_hold >= len(c.CONDITIONS)] = np.nan
    return out


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the holdout is scored once")
    started = time.time()
    dev = json.loads(imp.OUT_JSON.read_text())
    chosen = choose(dev)
    d = imp.load()
    ideas = {idea["name"]: idea for idea in imp.idea_definitions(d)}
    labels = d["labels"]
    y_dev, y_hold = labels[d["dev"]], labels[d["holdout"]]
    DC = np.hstack([d["D"], d["C"]])
    entries = [
        ("everything@4", c.leading(d["scores"][d["aliases"]["everything"]], 4), None, True),
        ("all-cell@4", c.leading(d["scores"][d["aliases"]["all-cell"]], 4), None, True),
        ("D+C features", DC, None, False),
        ("D+C + groups (20)", np.hstack([DC, c.leading(d["scores"][d["aliases"]["groups"]], 20)]), None, False),
    ]
    for entry in chosen:
        idea = ideas[entry["name"]]
        X = idea["build"]()
        if idea["kind"] == "supervised":
            entries.append((f"{entry['name']}@4", c.leading(X, idea["source_k"]), (idea["method"], 4), False))
        elif idea["kind"] == "selection":
            # The blocks are chosen on the labels of every development slide, then MFA on all slides.
            space = imp.selected_space(X, np.flatnonzero(d["dev"]), y_dev, idea["n_blocks"], 4)
            entries.append((f"{entry['name']}@4", space, None, False))
        else:
            entries.append((f"{entry['name']}@4", c.leading(X, 4), None, True))
    draws = c.stratified_bootstrap(y_hold, 4000, np.random.default_rng(HOLD_BOOT_SEED))
    results, boot = {}, {}
    for name, X, supervised, is_space in entries:
        proba = fit_predict(X[d["dev"]], y_dev, X[d["holdout"]], supervised)
        macro, per = c.macro_auc(proba, y_hold)
        boot[name] = np.array([c.macro_auc(proba[idx], y_hold[idx])[0] for idx in draws])
        row = {"macro_auc": macro, "auc_interval": imp.interval(boot[name]), "auc_per_condition": per.tolist()}
        if is_space:
            values = holdout_purity(X[d["dev"]], y_dev, X[d["holdout"]], y_hold)
            row["purity_macro"] = c.macro_purity(values, y_hold)[0]
        results[name] = row
    for name in results:
        results[name]["vs_everything4"] = {
            "difference": results[name]["macro_auc"] - results["everything@4"]["macro_auc"],
            "interval": imp.interval(boot[name] - boot["everything@4"]),
        }
    payload = {
        "chosen": [entry["name"] for entry in chosen],
        "holdout_slides": int(d["holdout"].sum()),
        "holdout_per_group": {g: int(np.sum(y_hold == i)) for i, g in enumerate(c.GROUP_ORDER)},
        "bootstrap_seed": HOLD_BOOT_SEED,
        "results": results,
        "scored_at": time.strftime("%Y-%m-%d %H:%M"),
        "runtime_seconds": round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    for name, row in results.items():
        print(f"{name}: holdout macro AUC {row['macro_auc']:.3f} {np.round(row['auc_interval'], 3)}; vs everything "
              f"{row['vs_everything4']['difference']:+.3f} {np.round(row['vs_everything4']['interval'], 3)}")


if __name__ == "__main__":
    main()
