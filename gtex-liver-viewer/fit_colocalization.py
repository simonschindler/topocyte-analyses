"""Fit the colocalization excess's gain/loss scale over every cohort slide: results/colocalization_scale.json.

For every k from 2 to 100 nearest neighbours and both label sets (the 14 HistoPLUS types and the four classes),
it sums each slide's (o - e) by focal type and neighbour type (``local_colocalization.excess_sums``), over all 594
slides: what ``LocalColocalizationExtractor.fit`` sums with the cohort as its training clouds, whose signs give
the pair scale (gain where the cohort's excess of a pair is >= 0, loss below). One query of 100 neighbours per
slide serves every k and both label sets. The viewer reads the file to colour cells by their excess; it reads
the cohort read only, as the viewer does, and writes only the JSON.

Run from this folder with the chromatic-domains study's venv::

    ../gtex-liver-chromatic-domains/.venv/bin/python fit_colocalization.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

import serve
from local_colocalization import excess_sums, neighbour_indices

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "colocalization_scale.json"
KS = list(range(serve.NEIGHBOURS_MIN, serve.NEIGHBOURS_MAX + 1))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    start = time.time()
    slides = serve.load_slide_table()
    types = serve.cell_types()
    type_class = np.array([int(k) for k in serve.c.class_codes(types)], dtype=np.int64)
    label_sets = {"types": list(types), "classes": list(serve.c.CLASSES)}
    sums = {by: (np.zeros((len(KS), len(names), len(names))), np.zeros((len(KS), len(names))))
            for by, names in label_sets.items()}
    n_cells = 0
    for i, row in enumerate(slides.itertuples(index=False)):
        xy, codes = serve.read_cells(row.slide_id, int(row.start), int(row.end), len(types))
        nbrs = neighbour_indices(xy, max(KS))
        labels = {"types": codes.astype(np.int64), "classes": type_class[codes]}
        for by, names in label_sets.items():
            pair, target = excess_sums(xy, labels[by], len(names), KS, nbrs)
            sums[by][0][...] += pair
            sums[by][1][...] += target
        n_cells += len(codes)
        if (i + 1) % 25 == 0 or i + 1 == len(slides):
            print(f"{i + 1} of {len(slides)} slides, {n_cells:,} cells, {time.time() - start:.0f} s", flush=True)
    out = {
        "description": "Summed hypergeometric count excess (o - e) of every cohort cell's k nearest other cells, by "
        "focal type (rows) and neighbour type (columns), and over every cell (target), for k = 2..100: "
        "LocalColocalizationExtractor.fit's sums with the whole cohort as its training clouds.",
        "k": KS,
        "label_sets": {
            by: {
                "labels": names,
                "pair_excess": np.round(sums[by][0], 3).tolist(),
                "target_excess": np.round(sums[by][1], 3).tolist(),
            }
            for by, names in label_sets.items()
        },
        "n_slides": len(slides),
        "n_cells": n_cells,
        "seconds": round(time.time() - start, 1),
        "md5": {name: md5(HERE / name) for name in ("fit_colocalization.py", "local_colocalization.py")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB) in {out['seconds']} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
