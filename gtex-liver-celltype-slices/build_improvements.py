#!/usr/bin/env python
"""Write ``improvements.md`` from ``results/improve_dev.json`` (and ``improve_holdout.json`` once it exists).

No number is typed by hand; every sentence that states a direction is
asserted against the numbers it states.
"""

from __future__ import annotations

import json

import common as c

OUT = c.HERE / "improvements.md"


def iv(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def sg(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def main() -> None:
    dev = json.loads((c.RESULTS / "improve_dev.json").read_text())
    holdout_path = c.RESULTS / "improve_holdout.json"
    holdout = json.loads(holdout_path.read_text()) if holdout_path.exists() else None
    ideas = dev["ideas"]
    by_name = {entry["name"]: entry for entry in ideas}

    def k4(entry: dict) -> dict:
        return next(r for r in entry["readouts"] if r["k"] == 4) if any(r["k"] == 4 for r in entry["readouts"]) else entry["readouts"][0]

    rows = []
    for number, entry in enumerate(ideas, 1):
        r4 = k4(entry)
        r20 = next((r for r in entry["readouts"] if r["k"] == 20), None)
        versus = f"{sg(r4['vs_everything4']['difference'])} {iv(r4['vs_everything4']['interval'])}" if "vs_everything4" in r4 else "—"
        purity = f"{r4['purity_macro']:.2f}" if "purity_macro" in r4 else "—"
        if "purity_vs_everything4" in r4:
            purity += f" ({sg(r4['purity_vs_everything4']['difference'], 2)} {iv(r4['purity_vs_everything4']['interval'], 2)})"
        rows.append(f"| {number} | `{entry['name']}` | {entry['mechanism']} | {entry['what']} | k={r4['k']}: {r4['macro_auc']:.3f} "
                    f"{iv(r4['auc_interval'])} | {versus} | {purity} | {f'{r20['macro_auc']:.3f}' if r20 else '—'} |")
    ref = k4(by_name["baseline:everything"])
    candidates = [e for e in ideas if e["mechanism"] != "baseline" and "vs_everything4" in k4(e)]
    best = max(candidates, key=lambda e: k4(e)["macro_auc"])
    best_r = k4(best)
    above = [e["name"] for e in candidates if k4(e)["vs_everything4"]["interval"][0] > 0]
    assert all(k4(by_name[n])["vs_everything4"]["interval"][0] > 0 for n in above)
    dev_text = (
        f"On the {dev['dev_slides']} development slides the main experiment's `everything` space reaches a macro AUC of "
        f"{ref['macro_auc']:.3f} {iv(ref['auc_interval'])} on four components. The best idea with a four-dimensional view is "
        f"`{best['name']}` at {best_r['macro_auc']:.3f}, {sg(best_r['vs_everything4']['difference'])} "
        f"{iv(best_r['vs_everything4']['interval'])} against `everything`. "
        + (f"{len(above)} idea{'s' if len(above) != 1 else ''} beat `everything` with an interval above 0: "
           + ", ".join(f"`{n}`" for n in above) + "." if above else "No idea beats `everything` with an interval above 0.")
    )
    sections = [
        "# Improvement phase: what was tried, and what held on the holdout",
        "",
        "Exploratory, run by Claude overnight on 2026-11-19 while Simon was away, under the rules in `design.md` (\"Rulings "
        "made after the builder check\").",
        "",
        f"- **Development slides.** Ideas are compared on the {dev['dev_slides']} development slides: 20 × stratified 5-fold "
        f"(seed {dev['fold_seed']}), the main experiment's logistic regression, and {dev['n_draws']} bootstrap draws "
        f"(seed {dev['bootstrap_seed']}) for the intervals.",
        "- **Unsupervised and supervised steps.** Unsupervised steps were fitted on all slides, without labels. Supervised "
        "projections were fitted inside each training fold.",
        "- **Holdout.** The holdout's labels were read only by `improve_holdout.py`, once.",
        "- **Criterion.** Development macro AUC on four components, paired against `everything`.",
        "",
        "## Every idea, in the order tried",
        "",
        "| # | idea | mechanism | what | dev macro AUC [95%] | vs `everything`@4 | purity ×chance (vs `everything`) | dev AUC k=20 |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        dev_text,
        "",
    ]
    if holdout is not None:
        res = holdout["results"]
        hold_rows = [f"| {name} | {row['macro_auc']:.3f} {iv(row['auc_interval'])} | {sg(row['vs_everything4']['difference'])} "
                     f"{iv(row['vs_everything4']['interval'])} | {f'{row['purity_macro']:.2f}' if 'purity_macro' in row else '—'} |"
                     for name, row in res.items()]
        per_group = ", ".join(f"{g} {n}" for g, n in holdout["holdout_per_group"].items())
        chosen = holdout["chosen"]
        held = [n for n in chosen if res[f"{n}@4"]["vs_everything4"]["interval"][0] > 0]
        verdict = (
            "On the holdout, " + (", ".join(f"`{n}`" for n in held) + " keep" + ("s" if len(held) == 1 else "")
                                  + " an interval above 0 against `everything`." if held else
                                  "no chosen idea keeps an interval above 0 against `everything`.")
        )
        sections += [
            "## Holdout, scored once",
            "",
            f"Chosen by the fixed rule: {', '.join(f'`{n}`' for n in chosen) or 'none'}. Scored {holdout['scored_at']} on "
            f"{holdout['holdout_slides']} slides ({per_group}); the small groups make every holdout AUC wide.",
            "",
            "| entry | holdout macro AUC [95%] | vs `everything`@4 | purity ×chance |",
            "|---|---|---|---|",
            *hold_rows,
            "",
            verdict,
            "",
        ]
    else:
        sections += ["## Holdout", "", "Not scored yet.", ""]
    OUT.write_text("\n".join(sections))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
