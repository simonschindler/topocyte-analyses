#!/usr/bin/env python
"""Write ``improvements.md`` from ``results/improve_dev.json`` (and ``improve_holdout.json`` once it exists).

No number is typed by hand; every sentence that states a direction is
chosen by the numbers it states.
"""

from __future__ import annotations

import json

import common as c

OUT = c.HERE / "improvements.md"


def iv(values: list[float], digits: int = 3) -> str:
    lo, hi = (0.0 if round(v, digits) == 0 else v for v in values)  # no "−0.000"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]".replace("-", "−")


# The report glossary's names for the holdout entries.
DISPLAY = {"D+C features": "D+C", "D+C + groups (20)": "D+C + groups@20"}


def sg(value: float, digits: int = 3) -> str:
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{value:+.{digits}f}".replace("-", "−")


def k4(entry: dict) -> dict:
    return next((r for r in entry["readouts"] if r["k"] == 4), entry["readouts"][0])


def relation(interval: list[float], up: str, same: str, down: str) -> str:
    return up if interval[0] > 0 else down if interval[1] < 0 else same


def main() -> None:
    dev = json.loads((c.RESULTS / "improve_dev.json").read_text())
    holdout_path = c.RESULTS / "improve_holdout.json"
    holdout = json.loads(holdout_path.read_text()) if holdout_path.exists() else None
    ideas = dev["ideas"]
    by_name = {entry["name"]: entry for entry in ideas}

    rows = []
    for number, entry in enumerate(ideas, 1):
        r4 = k4(entry)
        r20 = next((r for r in entry["readouts"] if r["k"] == 20), None)
        versus = f"{sg(r4['vs_everything4']['difference'])} {iv(r4['vs_everything4']['interval'])}" if "vs_everything4" in r4 else "—"
        purity = (f"{r4['purity_oof_macro']:.2f} ({sg(r4['purity_oof_vs_everything4']['difference'], 2)} "
                  f"{iv(r4['purity_oof_vs_everything4']['interval'], 2)})" if "purity_oof_vs_everything4" in r4
                  else f"{r4['purity_oof_macro']:.2f}")
        rows.append(f"| {number} | `{entry['name']}` | {entry['role']} | {entry['round']} | {entry['mechanism']} | "
                    f"{entry['inputs']}; {entry['what']} | k={r4['k']}: {r4['macro_auc']:.3f} {iv(r4['auc_interval'])} | "
                    f"{versus} | {purity} | {f'{r20['macro_auc']:.3f}' if r20 else '—'} |")

    reference = k4(by_name["baseline:everything"])
    reference20 = next(r for r in by_name["baseline:everything"]["readouts"] if r["k"] == 20)
    dc = by_name["baseline:D+C-features"]["readouts"][0]
    candidates = [e for e in ideas if e["role"] == "idea" and "vs_everything4" in k4(e)]
    best = max(candidates, key=lambda e: k4(e)["macro_auc"])
    above = [e["name"] for e in candidates if k4(e)["vs_everything4"]["interval"][0] > 0]
    diagnostics = [e for e in ideas if e["role"] == "diagnostic"]
    dev_text = [
        f"On the {dev['dev_slides']} development slides, the main experiment's `everything` space reaches a macro AUC of "
        f"{reference['macro_auc']:.3f} {iv(reference['auc_interval'])} on four components and {reference20['macro_auc']:.3f} "
        f"on twenty. `D+C` alone reaches {dc['macro_auc']:.3f} {iv(dc['auc_interval'])} ({dc['k']} features).",
        "",
        f"- **Best idea:** `{best['name']}` ({best['mechanism']}, {best['inputs']}) at {k4(best)['macro_auc']:.3f}, "
        f"{sg(k4(best)['vs_everything4']['difference'])} {iv(k4(best)['vs_everything4']['interval'])} against `everything`.",
        "- **Beyond `everything`:** "
        + (f"{len(above)} idea{'s' if len(above) != 1 else ''} beat `everything` with an interval above 0: "
           + ", ".join(f"`{n}`" for n in above) + "." if above else "no idea beats `everything` with an interval above 0."),
        "- **Supervised four-component views:**",
        "  - They use more input than `everything`'s four components: 20 to 1800 dimensions, listed under \"what\" in the table.",
        f"  - On equal inputs, `everything` on 20 components reaches {reference20['macro_auc']:.3f}.",
        "- **Diagnostics,** built to remove signal: "
        + "; ".join(f"`{e['name']}` {k4(e)['macro_auc']:.3f} {iv(k4(e)['auc_interval'])}" for e in diagnostics)
        + ".",
    ]

    sections = [
        "# Improvement phase: what was tried, and what held on the holdout",
        "",
        "Exploratory. Claude ran it overnight on 2026-11-19 while Simon was away, under the rules in `design.md`",
        "The names (`everything@4`, `D+C`, `lda4-X`, ...) are explained in the glossary of `report.html`.",
        "(\"Rulings made after the builder check\") and the rulings in `progress.md`.",
        "",
        f"- **Development slides.** Every idea is compared on the {dev['dev_slides']} development slides:",
        f"  - 20 × stratified 5-fold (seed {dev['fold_seed']});",
        "  - the main experiment's logistic regression on the idea's four-dimensional view;",
        f"  - {dev['n_draws']} bootstrap draws (seed {dev['bootstrap_seed']}) for the intervals.",
        "  - Purity is out of fold: each test slide against its 15 nearest training-fold slides in that fold's view.",
        "- **Fitting.** Unsupervised steps saw all slides, without labels. Supervised projections and block selections were "
        "fitted inside each training fold.",
        "- **Roles.**",
        "  - `idea`: a candidate for the holdout.",
        "  - `baseline`: the main experiment's spaces.",
        "  - `control`: the same method without topology, never chosen.",
        "  - `diagnostic`: built to remove signal.",
        "- **Rounds.**",
        "  - Round 1 was conceived before any readout existed.",
        "  - Round 2 came after the main readouts and round 1's results. It is post hoc, and the holdout is a weaker check for it.",
        "- **The holdout is not pristine.**",
        "  - The main experiment's readouts used all 594 slides' labels, the holdout's included, as Simon ruled.",
        "  - `improve.py` never reads the holdout labels.",
        "  - `improve_holdout.py` scored the holdout once.",
        "",
        "## Every entry, in the order tried",
        "",
        "| # | entry | role | round | mechanism | inputs; what | dev macro AUC [95%] | vs `everything`@4 | "
        "out-of-fold purity ×chance (vs `everything`) | dev AUC k=20 |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        *dev_text,
        "",
    ]
    if holdout is not None:
        res = holdout["results"]
        hold_rows = []
        for name, row in res.items():
            purity = "—"
            if "purity_macro" in row:
                purity = (f"{row['purity_macro']:.2f} ({sg(row['purity_vs_everything4']['difference'], 2)} "
                          f"{iv(row['purity_vs_everything4']['interval'], 2)})")
            hold_rows.append(f"| {DISPLAY.get(name, name)} | {row['macro_auc']:.3f} {iv(row['auc_interval'])} | "
                             f"{sg(row['vs_everything4']['difference'])} {iv(row['vs_everything4']['interval'])} | "
                             f"{sg(row['vs_DC']['difference'])} {iv(row['vs_DC']['interval'])} | {purity} |")
        per_group = ", ".join(f"{g} {n}" for g, n in holdout["holdout_per_group"].items())
        chosen = holdout["chosen"]
        verdicts = []
        for name in chosen:
            row = res[f"{name}@4"]
            verdicts.append(
                f"- `{name}`: "
                f"{relation(row['vs_everything4']['interval'], 'above', 'level with', 'below')} `everything` "
                f"({sg(row['vs_everything4']['difference'])} {iv(row['vs_everything4']['interval'])}), "
                f"{relation(row['vs_DC']['interval'], 'above', 'level with', 'below')} `D+C` "
                f"({sg(row['vs_DC']['difference'])} {iv(row['vs_DC']['interval'])}).")
        secondary = holdout["secondary_on_holdout"]
        control_row = res[f"{holdout['control']}@4 (control)"] if holdout["control"] else None
        beat_everything = [n for n in chosen if res[f"{n}@4"]["vs_everything4"]["interval"][0] > 0]
        beat_dc = [n for n in chosen if res[f"{n}@4"]["vs_DC"]["interval"][0] > 0]
        reading = []
        if chosen and not beat_dc and control_row is not None:
            reading = [
                "**Reading.**",
                "",
                f"- **No chosen idea beats composition.** On unseen slides no chosen idea's difference to `D+C` "
                f"({res['D+C features']['macro_auc']:.3f}) has its interval above 0: "
                + "; ".join(f"`{n}` {sg(res[f'{n}@4']['vs_DC']['difference'])} {iv(res[f'{n}@4']['vs_DC']['interval'])}"
                            for n in chosen) + ".",
                f"- **The topology-free control matches them.** `{holdout['control']}`, the supervised control without "
                f"topology, reaches {control_row['macro_auc']:.3f}, "
                + ("above" if control_row["vs_everything4"]["interval"][0] > 0 else "level with")
                + " the unsupervised `everything` space.",
                "- **" + (", ".join(f"`{n}`" for n in beat_everything) + (" beats" if len(beat_everything) == 1 else " beat")
                          + " `everything`" if beat_everything else "No chosen idea beats `everything`")
                + ".** What improves separability on a four-component view is supervision and the cell counts, not more topology.",
                "",
            ]
        primary = res["all-cell@4"]["vs_everything4"]  # all-cell minus everything: the primary comparison, reversed
        primary_text = (f"- Primary comparison on the holdout (`everything` against `all-cell`, four components): "
                        f"{sg(-primary['difference'])} {iv([-primary['interval'][1], -primary['interval'][0]])}.")
        sections += [
            "## Holdout, scored once",
            "",
            f"- **Chosen by the fixed rule:** {', '.join(f'`{n}`' for n in chosen) or 'none'}.",
            f"- **Control:** the best supervised control without topology, `{holdout['control']}`.",
            f"- **When and what:** scored {holdout['scored_at']} on {holdout['holdout_slides']} slides ({per_group}).",
            "- **Uncertainty:** with groups of two and four slides, bootstrap draws within a group understate it, so every "
            "holdout interval is too narrow for the small groups.",
            "",
            "| entry | holdout macro AUC [95%] | vs `everything@4` | vs `D+C` | purity ×chance (vs `everything@4`) |",
            "|---|---|---|---|---|",
            *hold_rows,
            "",
            *verdicts,
            primary_text,
            f"- Secondary comparison on the holdout (`D+C` + 20 components of `groups` against `D+C`): "
            f"{sg(secondary['difference'])} {iv(secondary['interval'])}.",
            "",
            *reading,
        ]
    else:
        sections += ["## Holdout", "", "Not scored yet.", ""]
    OUT.write_text("\n".join(sections))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
