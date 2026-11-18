#!/usr/bin/env python
"""Does PC2's cirrhosis gap survive removing the densest-decile share?

For each combined space of ``combined_spaces.py``, PC2 is oriented so that
cirrhosis's median lies above healthy's, then residualised by least squares
over all 594 slides on

* ``share``: the densest-decile share (the share of a slide's cells kept at
  slice 9) and its square - the primary readout;
* ``strict``: the share and its square, PC1 and PC1^2 (the arch), log10
  median density and log10 cell count.

For PC2 as fitted, both residuals and, for scale, the share alone, it saves
the AUC of cirrhosis against healthy (P(cirrhosis > healthy) over all pairs)
with a 95% bootstrap interval (4000 draws, each group resampled on its own,
``default_rng(12345)``) and the gap of the group medians in standard
deviations.  The ledger (Task 4) fixed all of this before the run.

Reads ``results/combined.npz``, ``results/density.npz`` and
``results/pathology.csv``; writes ``results/pc2_check.npz`` and ``.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
N_BOOT = 4000
SEED = 12345
VARIANTS = ("raw", "share", "strict")


def auc(cirrhosis: np.ndarray, healthy: np.ndarray) -> float:
    """P(cirrhosis > healthy) over all pairs, ties counting a half."""
    greater = (cirrhosis[:, None] > healthy[None, :]).mean()
    ties = (cirrhosis[:, None] == healthy[None, :]).mean()
    return float(greater + 0.5 * ties)


def residual(y: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    """Least-squares residual of y on an intercept and the given columns."""
    design = np.column_stack([np.ones(y.size)] + columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def main() -> None:
    started = time.time()
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as data:
        combined = {key: data[key] for key in data.files}
    with np.load(RESULTS / "density.npz", allow_pickle=False) as data:
        slide_ids = data["slide_ids"]
        kept = data["kept_per_slide"]
        medians = data["slide_medians"]
        cells = np.diff(data["bounds"])
    if not np.array_equal(combined["slide_ids"], slide_ids):
        raise RuntimeError("combined.npz and density.npz list the slides differently")
    table = pd.read_csv(RESULTS / "pathology.csv", keep_default_na=False)
    if list(table["slide_id"]) != [str(s) for s in slide_ids]:
        raise RuntimeError("pathology.csv lists the slides differently")
    group = table["group"].to_numpy()
    cirrhosis, healthy = group == "cirrhosis", group == "healthy"
    share = kept[9] / kept[0]
    log_median = np.log10(medians)
    log_cells = np.log10(cells)

    rng = np.random.default_rng(SEED)
    draws_c = rng.integers(0, cirrhosis.sum(), (N_BOOT, cirrhosis.sum()))
    draws_h = rng.integers(0, healthy.sum(), (N_BOOT, healthy.sum()))

    def readout(values: np.ndarray) -> np.ndarray:
        """AUC, its 95% bootstrap interval, and the median gap in SD."""
        c, h = values[cirrhosis], values[healthy]
        boot = np.array([auc(c[i], h[j]) for i, j in zip(draws_c, draws_h)])
        gap = (np.median(c) - np.median(h)) / values.std()
        return np.array([auc(c, h), np.quantile(boot, 0.025), np.quantile(boot, 0.975), gap])

    payload: dict[str, np.ndarray] = {
        "slide_ids": slide_ids,
        "share": share,
        "share_auc": readout(share),
        "log_median_auc": readout(log_median),
        "log_cells_auc": readout(log_cells),
    }
    spaces = [str(s) for s in combined["spaces"]]
    for space in spaces:
        scores = combined[f"{space}_scores"]
        pc1, pc2 = scores[:, 0], scores[:, 1]
        sign = 1.0 if np.median(pc2[cirrhosis]) >= np.median(pc2[healthy]) else -1.0
        pc2 = sign * pc2
        values = {
            "raw": pc2,
            "share": residual(pc2, [share, share**2]),
            "strict": residual(pc2, [share, share**2, pc1, pc1**2, log_median, log_cells]),
        }
        payload[f"{space}_sign"] = np.float64(sign)
        for variant in VARIANTS:
            payload[f"{space}_{variant}"] = readout(values[variant])
            payload[f"{space}_{variant}_values"] = values[variant]
        print(
            f"{space}: " + "; ".join(
                f"{v} AUC {payload[f'{space}_{v}'][0]:.3f} [{payload[f'{space}_{v}'][1]:.3f}, "
                f"{payload[f'{space}_{v}'][2]:.3f}] gap {payload[f'{space}_{v}'][3]:+.2f}"
                for v in VARIANTS
            ),
            flush=True,
        )
    print(f"share alone: AUC {payload['share_auc'][0]:.3f} [{payload['share_auc'][1]:.3f}, {payload['share_auc'][2]:.3f}]")
    payload["spaces"] = np.array(spaces)
    np.savez(RESULTS / "pc2_check.npz", **payload)
    meta = {
        "status": "decided before the run (ledger Task 4); descriptive, nothing tuned",
        "readout": "AUC P(cirrhosis > healthy), 95% bootstrap interval (groups resampled separately), median gap in SD",
        "variants": {
            "raw": "PC2 as fitted, oriented so that cirrhosis's median lies above healthy's",
            "share": "PC2 residualised on the densest-decile share and its square",
            "strict": "PC2 residualised on the share, its square, PC1, PC1^2, log10 median density and log10 cell count",
        },
        "bootstrap": {"draws": N_BOOT, "seed": SEED},
        "runtime_seconds": round(time.time() - started, 1),
    }
    (RESULTS / "pc2_check.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote results/pc2_check.npz in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
