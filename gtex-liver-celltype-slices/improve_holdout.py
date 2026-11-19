#!/usr/bin/env python
"""Score the chosen ideas once on the holdout (design.md: the rulings after the builder check; progress.md rulings).

Selection (``choose``), reading only ``results/improve_dev.json``:

* candidates: entries whose role is ``idea`` and that have a
  four-dimensional view (unsupervised space, supervised projection, block
  selection), and whose out-of-fold purity does not fall below
  ``everything``'s (the upper end of the paired interval not below 0);
* ranked by development macro AUC at four components; taken while their
  paired AUC interval against ``everything`` lies above 0, at most one per
  mechanism and three in all;
* if none lies above 0, the best candidate still goes, as a check.

Holdout entries:

* ``everything`` and ``all-cell`` on four components;
* the ``D+C`` features, and ``D+C`` with the first 20 components of
  ``groups`` (the secondary comparison, replicated);
* the best supervised control without topology;
* the chosen ideas.

Each is fitted on the 476 development slides (supervised projection, block
selection, standardization and logistic regression; unsupervised steps as
fitted on all slides) and predicts the 118 holdout slides once.

Readouts:

* macro AUC with its 95% interval;
* paired differences to ``everything@4`` and to the ``D+C`` features;
* per-condition AUCs;
* for every four-dimensional view, purity of the holdout slides against
  their 15 nearest development slides in the view fitted on the development
  slides, with its paired difference to ``everything@4``'s.

The intervals come from 4000 bootstrap draws of holdout slides stratified
by group (seed 777). With 2 inflammation/hepatitis and 4 necrosis slides,
draws within such groups understate the uncertainty.

Writes ``results/improve_holdout.json`` and ``.npz`` once, and refuses to
run again.
"""

from __future__ import annotations

import json
import time

import numpy as np

import common as c
import improve as imp

OUT = c.RESULTS / "improve_holdout.json"
OUT_NPZ = c.RESULTS / "improve_holdout.npz"
HOLD_BOOT_SEED = 777
CONTROL_MECHANISM = "supervised projection (composition)"


def _k4(entry: dict) -> dict:
    return next(r for r in entry["readouts"] if r["k"] == 4)


def choose(dev: dict) -> list[dict]:
    """The ideas that reach the holdout, by the rule fixed before the run (see the module docstring)."""
    candidates = []
    for entry in dev["ideas"]:
        if entry.get("role") != "idea" or entry["kind"] not in imp.VIEW_KINDS:
            continue
        if _k4(entry)["purity_oof_vs_everything4"]["interval"][1] < 0:
            continue
        candidates.append(entry)
    candidates.sort(key=lambda entry: -_k4(entry)["macro_auc"])
    chosen, mechanisms = [], set()
    for entry in candidates:
        if _k4(entry)["vs_everything4"]["interval"][0] <= 0 or entry["mechanism"] in mechanisms:
            continue
        chosen.append(entry)
        mechanisms.add(entry["mechanism"])
        if len(chosen) == 3:
            break
    if not chosen and candidates:
        chosen = [candidates[0]]
    return chosen


def comparisons(points: dict[str, float], boots: dict[str, np.ndarray]) -> dict[str, dict]:
    """Every entry's paired macro-AUC differences to ``everything@4`` and to the ``D+C`` features."""
    references = (("vs_everything4", "everything@4"), ("vs_DC", "D+C features"))
    return {name: {key: {"difference": points[name] - points[ref], "interval": imp.interval(boots[name] - boots[ref])}
                   for key, ref in references}
            for name in boots}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the holdout is scored once")
    started = time.time()
    dev = json.loads(imp.OUT_JSON.read_text())
    chosen = choose(dev)
    controls = [e for e in dev["ideas"] if e.get("role") == "control" and e["mechanism"] == CONTROL_MECHANISM]
    control = max(controls, key=lambda e: _k4(e)["macro_auc"]) if controls else None
    d = imp.load()
    ideas = {idea["name"]: idea for idea in imp.idea_definitions(d)}
    labels, dev_mask, hold_mask = d["labels"], d["dev"], d["holdout"]
    y_dev, y_hold = labels[dev_mask], labels[hold_mask]
    DC = np.hstack([d["D"], d["C"]])
    entries = [
        ("everything@4", "space", c.leading(d["scores"][d["aliases"]["everything"]], 4), None),
        ("all-cell@4", "space", c.leading(d["scores"][d["aliases"]["all-cell"]], 4), None),
        ("D+C features", "features", DC, None),
        ("D+C + groups (20)", "features", np.hstack([DC, c.leading(d["scores"][d["aliases"]["groups"]], 20)]), None),
    ]
    for entry, tag in ([(control, " (control)")] if control else []) + [(e, "") for e in chosen]:
        idea = ideas[entry["name"]]
        X = idea["build"]()
        if idea["kind"] == "supervised":
            entries.append((f"{entry['name']}@4{tag}", "supervised", c.leading(X, idea["source_k"]), (idea["method"], 4)))
        elif idea["kind"] == "selection":
            entries.append((f"{entry['name']}@4{tag}", "selection", X, (idea["n_blocks"], 4)))
        else:
            entries.append((f"{entry['name']}@4{tag}", "space", c.leading(X, 4), None))

    dev_rows = np.flatnonzero(dev_mask)
    draws = c.stratified_bootstrap(y_hold, 4000, np.random.default_rng(HOLD_BOOT_SEED))
    points, boots, results, arrays = {}, {}, {}, {}
    purity_points, purity_boots = {}, {}
    for i, (name, kind, X, params) in enumerate(entries):
        if kind == "supervised":
            projector = imp.Projector(*params).fit(X[dev_mask], y_dev)
            Z_dev, Z_hold = projector.transform(X[dev_mask]), projector.transform(X[hold_mask])
        elif kind == "selection":
            # The blocks are chosen on the labels of the development slides only, then MFA on every slide.
            Z = imp.selected_space(X, dev_rows, y_dev, *params)
            Z_dev, Z_hold = Z[dev_mask], Z[hold_mask]
        else:
            Z_dev, Z_hold = X[dev_mask], X[hold_mask]
        model, mean, sd, n_warn = imp.fit_logistic(Z_dev, y_dev)
        proba = np.zeros((len(y_hold), len(c.GROUP_ORDER)))
        proba[:, model.classes_] = model.predict_proba((Z_hold - mean) / sd)
        macro, per = c.macro_auc(proba, y_hold)
        points[name] = macro
        boots[name] = np.array([c.macro_auc(proba[idx], y_hold[idx])[0] for idx in draws])
        row = {"kind": kind, "macro_auc": macro, "auc_interval": imp.interval(boots[name]),
               "auc_per_condition": per.tolist(), "n_convergence_warnings": int(n_warn)}
        arrays[f"e{i}__proba"] = proba
        if kind != "features":
            values = imp.neighbour_purity(Z_hold, y_hold, Z_dev, y_dev)
            arrays[f"e{i}__purity"] = values
            purity_points[name] = c.macro_purity(values, y_hold)[0]
            purity_boots[name] = np.mean([values[draws[:, y_hold == g]].mean(axis=1) for g in range(len(c.CONDITIONS))],
                                         axis=0)
            row["purity_macro"] = purity_points[name]
        results[name] = row
    for name, table in comparisons(points, boots).items():
        results[name].update(table)
        if name in purity_points:
            results[name]["purity_vs_everything4"] = {
                "difference": purity_points[name] - purity_points["everything@4"],
                "interval": imp.interval(purity_boots[name] - purity_boots["everything@4"]),
            }
    payload = {
        "chosen": [entry["name"] for entry in chosen],
        "control": control["name"] if control else None,
        "holdout_slides": int(hold_mask.sum()),
        "holdout_per_group": {g: int(np.sum(y_hold == i)) for i, g in enumerate(c.GROUP_ORDER)},
        "bootstrap_seed": HOLD_BOOT_SEED,
        "results": results,
        "secondary_on_holdout": results["D+C + groups (20)"]["vs_DC"],
        "scored_at": time.strftime("%Y-%m-%d %H:%M"),
        "runtime_seconds": round(time.time() - started, 1),
    }
    np.savez(OUT_NPZ, names=np.array([e[0] for e in entries]), draws=draws, y_holdout=y_hold,
             holdout_rows=np.flatnonzero(hold_mask), **arrays)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    for name, row in results.items():
        print(f"{name}: holdout macro AUC {row['macro_auc']:.3f} {np.round(row['auc_interval'], 3)}; vs everything "
              f"{row['vs_everything4']['difference']:+.3f} {np.round(row['vs_everything4']['interval'], 3)}; vs D+C "
              f"{row['vs_DC']['difference']:+.3f} {np.round(row['vs_DC']['interval'], 3)}")


if __name__ == "__main__":
    main()
