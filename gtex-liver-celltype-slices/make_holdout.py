#!/usr/bin/env python
"""The improvement phase's holdout, drawn before any result of this study existed.

A seeded draw of a fifth of the slides of every pathology group in the
sibling study's ``pathology.csv`` (rounded to the nearest slide). The main
experiment uses every slide; the improvement phase compares its ideas on the
other slides only and scores the chosen ones once on these at the end.
Writes ``results/holdout.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "gtex-liver-pathology-slices" / "results"
SEED = 20261119
SHARE = 0.2


def main() -> None:
    table = pd.read_csv(SOURCE / "pathology.csv")
    rng = np.random.default_rng(SEED)
    holdout = np.zeros(len(table), dtype=bool)
    groups = table["group"].to_numpy()
    for group in sorted(set(groups)):
        members = np.flatnonzero(groups == group)
        holdout[rng.choice(members, size=int(round(SHARE * members.size)), replace=False)] = True
    out = pd.DataFrame({"slide_id": table["slide_id"], "group": groups, "holdout": holdout})
    (HERE / "results").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "holdout.csv", index=False)
    print(out.groupby("group")["holdout"].agg(["size", "sum"]).to_string())
    print(f"holdout {int(holdout.sum())} of {holdout.size} slides")


if __name__ == "__main__":
    main()
