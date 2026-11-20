#!/usr/bin/env python
"""Run script of the chromatic-domains pilot.

Subcommands
-----------
``pilot``
    Draw the 30 pilot slides from the eligible development slides (design
    step 1) and write ``results/pilot.csv``: slide, condition, half and slide
    index. It refuses inputs whose eligible counts per condition differ from
    the design's.

Run from this folder with the study venv and
``PYTHONPATH=<topocyte checkout at 18e3aa5>``. Every subcommand first
refuses a topocyte from anywhere else, or a checkout whose package differs
from the commit (``common.check_topocyte``). The cohort is opened read only,
and nothing is written outside this folder.
"""

from __future__ import annotations

import argparse

import pandas as pd

import common as c

# Design step 1: the eligible slides per condition.
EXPECTED_ELIGIBLE = {
    "healthy": 53,
    "congestion": 118,
    "steatosis": 158,
    "inflammation/hepatitis": 9,
    "fibrosis": 40,
    "cirrhosis": 44,
}


def cmd_pilot() -> None:
    eligible = c.eligible_slides(c.CT / "holdout.csv", c.SIB / "pathology.csv")
    counts = {condition: int((eligible["condition"] == condition).sum()) for condition in c.CONDITIONS}
    print("eligible slides: " + ", ".join(f"{condition} {n}" for condition, n in counts.items()), flush=True)
    if counts != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"eligible slides per condition are {counts}, not the design's {EXPECTED_ELIGIBLE}")
    pilot = c.draw_pilot(eligible)
    pilot.to_csv(c.RESULTS / "pilot.csv", index=False)
    print(pd.crosstab(pilot["condition"], pilot["half"]).reindex(list(c.CONDITIONS)).to_string())
    halves = pilot["half"].value_counts()
    print(f"pilot: {len(pilot)} slides, half A {halves['A']}, half B {halves['B']}; wrote {c.RESULTS / 'pilot.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pilot", help="draw the 30 pilot slides: results/pilot.csv")
    args = parser.parse_args()
    c.check_topocyte()
    c.RESULTS.mkdir(exist_ok=True)
    if args.command == "pilot":
        cmd_pilot()


if __name__ == "__main__":
    main()
