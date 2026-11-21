#!/usr/bin/env python
"""Run script of the chromatic-domains pilot.

Subcommands
-----------
``pilot``
    Draw the 30 pilot slides from the eligible development slides (design
    step 1) and write ``results/pilot.csv``: slide, condition, half and slide
    index. It refuses inputs whose eligible counts per condition differ from
    the design's.
``preflight``
    Check the rule for k (design step 10, ``domains.choose_k``) before the
    run, on synthetic halves of the pilot's sizes (design, "Checks fixed in
    advance", 1). Matched Gaussians with PCA-like variances must give k = 1:
    22,500 × 5 with k_max 10 (level 1), and 5,000 and 22,500 × 20 with k_max
    6 (level 2). So must skewed cluster-free clouds of the same sizes, whose
    first column is lognormal (σ = 0.6), and a 5-d one whose lognormal column
    carries the most variance (3). Two- and three-cluster mixtures of 5,000 ×
    20 must give k = 2 and 3. Writes ``results/preflight.json`` and exits 1 if
    any case fails.

Run from this folder with the study venv and
``PYTHONPATH=<topocyte checkout at 18e3aa5>``. Every subcommand first
refuses a topocyte from anywhere else, or a checkout whose package differs
from the commit (``common.check_topocyte``). The cohort is opened read only,
and nothing is written outside this folder.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from functools import partial

import numpy as np
import pandas as pd
import sklearn

import common as c
import domains

# Design step 1: the eligible slides per condition.
EXPECTED_ELIGIBLE = {
    "healthy": 53,
    "congestion": 118,
    "steatosis": 158,
    "inflammation/hepatitis": 9,
    "fibrosis": 40,
    "cirrhosis": 44,
}

# The pre-flight's synthetic halves (the controller's constructions, ruling R2, and the skewed cases of Simon's ruling
# of 2026-11-21). One default_rng(PREFLIGHT_SEED) draws the cases in PREFLIGHT_CASES order, half A then half B.
PREFLIGHT_SEED = 1
LEVEL1_VARIANCES = np.array([3.0, 1.5, 0.8, 0.4, 0.2])  # PCA-like variances of the five level-1 features
LEVEL2_VARIANCES = 1 / np.arange(1, 21)  # PCA-like variances of the 20 level-2 components
LOGNORMAL_VARIANCE = (np.exp(0.36) - 1) * np.exp(0.36)  # the variance of exp(N(0, 0.6)), about 0.621


def _gaussian(variances: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """n rows of a centred Gaussian with independent columns of the given variances."""
    return rng.normal(size=(n, variances.size)) * np.sqrt(variances)


def _skewed(variances: np.ndarray, rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    """n rows of a skewed cluster-free cloud: column 0 scale * exp(N(0, 0.6)), the rest Gaussian of these variances."""
    return np.column_stack([scale * np.exp(rng.normal(0, 0.6, n)), _gaussian(variances, rng, n)])


def _two_clusters(rng: np.random.Generator, n: int) -> np.ndarray:
    """N(0, I) in 20 dimensions with column 0 shifted by -2 or +2 at random: centres 4 SD apart."""
    X = rng.normal(size=(n, 20))
    X[:, 0] += np.where(rng.random(n) < 0.5, -2.0, 2.0)
    return X


def _three_clusters(rng: np.random.Generator, n: int) -> np.ndarray:
    """N(0, I) in 20 dimensions with centres (-3, 0), (0, 3) and (3, 0) on columns 0 and 1, drawn at random."""
    X = rng.normal(size=(n, 20))
    X[:, :2] += np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])[rng.integers(0, 3, n)]
    return X


# (case, rows per half, k_max, the k the rule must give, construction)
PREFLIGHT_CASES = (
    ("level-1 Gaussian, 22,500 x 5", 22_500, 10, 1, partial(_gaussian, LEVEL1_VARIANCES)),
    ("level-2 Gaussian, 5,000 x 20", 5_000, 6, 1, partial(_gaussian, LEVEL2_VARIANCES)),
    ("level-2 Gaussian, 22,500 x 20", 22_500, 6, 1, partial(_gaussian, LEVEL2_VARIANCES)),
    ("two clusters, 5,000 x 20", 5_000, 6, 2, _two_clusters),
    ("three clusters, 5,000 x 20", 5_000, 6, 3, _three_clusters),
    # Appended, so that the cases above keep their draws.
    ("level-2 skewed, 5,000 x 20", 5_000, 6, 1, partial(_skewed, 1 / np.arange(2, 21))),
    ("level-2 skewed, 22,500 x 20", 22_500, 6, 1, partial(_skewed, 1 / np.arange(2, 21))),
    ("level-1 skewed, 22,500 x 5", 22_500, 10, 1, partial(_skewed, np.array([1.5, 0.8, 0.4, 0.2]))),
    # The lognormal column scaled to variance 3, the largest (the corrected rule, 2026-11-21).
    (
        "level-1 skew-dominant, 22,500 x 5",
        22_500,
        10,
        1,
        partial(_skewed, np.array([1.5, 0.8, 0.4, 0.2]), scale=np.sqrt(3 / LOGNORMAL_VARIANCE)),
    ),
)


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


def _half(result: dict) -> dict:
    """One half's gap curves for the JSON, with its largest gain Gap(k) - Gap(1) over k >= 2."""
    gap, s = result["gap"], result["s"]
    return {"k": result["k"], "gap": gap.tolist(), "s": s.tolist(), "largest_gain": float(np.max(gap[1:] - gap[0]))}


def cmd_preflight() -> bool:
    """Run the full rule for k on each pre-flight case, write ``results/preflight.json``; True if every case passes."""
    rng = np.random.default_rng(PREFLIGHT_SEED)
    cases, start = [], time.perf_counter()
    for name, n, k_max, expected, draw in PREFLIGHT_CASES:
        A, B = draw(rng, n), draw(rng, n)
        t = time.perf_counter()
        result = domains.choose_k(A, B, [(A, B), (B, A)], k_max)
        case = {
            "case": name,
            "rows_per_half": n,
            "columns": A.shape[1],
            "k_max": k_max,
            "expected_k": expected,
            "k": result["k"],
            "k_A": result["k_A"],
            "k_B": result["k_B"],
            "ps": result["ps"],
            "gap_A": _half(result["gap_A"]),
            "gap_B": _half(result["gap_B"]),
            "seconds": round(time.perf_counter() - t, 1),
            "passed": result["k"] == expected,
        }
        cases.append(case)
        print(
            f"{name}: k {case['k']} (expected {expected}), k_A {case['k_A']}, k_B {case['k_B']}, "
            f"largest gain {case['gap_A']['largest_gain']:.4f} / {case['gap_B']['largest_gain']:.4f}, "
            f"PS {[round(p, 3) for p in case['ps']]}, {'pass' if case['passed'] else 'FAIL'}, {case['seconds']} s",
            flush=True,
        )
    passed = all(case["passed"] for case in cases)
    out = {
        "passed": passed,
        "rule": {
            "stop": "Tibshirani's stopping rule from k = 1, then the gain floor on the k it stops at",
            "B": domains.N_REFERENCES,
            "min_gain": domains.MIN_GAIN,
            "ps_min": domains.PS_MIN,
            "n_init": domains.N_INIT,
            "seed": 0,
        },
        "data": f"one default_rng({PREFLIGHT_SEED}) draws the cases in order, half A then half B",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "seconds": round(time.perf_counter() - start, 1),
        "cases": cases,
    }
    (c.RESULTS / "preflight.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"pre-flight {'passed' if passed else 'FAILED'} in {out['seconds']} s; wrote {c.RESULTS / 'preflight.json'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pilot", help="draw the 30 pilot slides: results/pilot.csv")
    sub.add_parser("preflight", help="check the rule for k on synthetic halves: results/preflight.json")
    args = parser.parse_args()
    c.check_topocyte()
    c.RESULTS.mkdir(exist_ok=True)
    if args.command == "pilot":
        cmd_pilot()
    elif args.command == "preflight" and not cmd_preflight():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
