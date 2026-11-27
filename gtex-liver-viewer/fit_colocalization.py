"""Fit the colocalization excess's gain/loss scale over every slide of a cohort: the liver cohort's
results/colocalization_scale.json, or with ``--tissue`` one GTEx tissue folder's ``<out-dir>/<slug>.json``.

For every k from 2 to 100 nearest neighbours and both label sets (the cell types and the four classes), it sums
each slide's (o - e) by focal type and neighbour type (``local_colocalization.excess_sums``), over all the cohort's
slides: what ``LocalColocalizationExtractor.fit`` sums with the cohort as its training clouds, whose signs give the
pair scale (gain where the cohort's excess of a pair is >= 0, loss below). One query of 100 neighbours per slide
serves every k and both label sets. The viewer reads the file to colour cells by their excess; it reads the cohort
read only, as the viewer does, and writes only the JSON.

A tissue's scale is fitted over that tissue's own slides, with its own cell types and the viewer's class map for it
(``serve.class_codes_or_other``): the liver cohort's pair scale says nothing of another tissue's.

Run from this folder with the chromatic-domains study's venv::

    ../gtex-liver-chromatic-domains/.venv/bin/python fit_colocalization.py
    ../gtex-liver-chromatic-domains/.venv/bin/python fit_colocalization.py --tissue <root>/<slug> --out-dir <dir>

and serve a tissue folder's fits with ``serve.py --tissues <root> --colocalization-dir <dir>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import scipy

import serve
from local_colocalization import excess_sums, neighbour_indices

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "colocalization_scale.json"
KS = list(range(serve.NEIGHBOURS_MIN, serve.NEIGHBOURS_MAX + 1))
DESCRIPTION = (
    "Summed hypergeometric count excess (o - e) of every cohort cell's k nearest other cells, by focal type (rows) "
    "and neighbour type (columns), and over every cell (target), for k = 2..100: "
    "LocalColocalizationExtractor.fit's sums with the whole cohort as its training clouds."
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fit(
    slides: Iterable[tuple[str, int, int]],
    n_slides: int,
    types: list[str],
    type_class: np.ndarray,
    read: Callable[[str, int, int], tuple[np.ndarray, np.ndarray]],
    ks: list[int] = KS,
) -> dict:
    """The summed excess of every slide in ``slides`` ((slide ID, start, end) rows, ``n_slides`` of them), whose
    cells ``read`` gives (level-0 px positions and type codes into ``types``): the JSON's ``k``, ``label_sets``,
    ``n_slides``, ``n_cells`` and ``seconds``."""
    start = time.time()
    label_sets = {"types": list(types), "classes": list(serve.c.CLASSES)}
    sums = {by: (np.zeros((len(ks), len(names), len(names))), np.zeros((len(ks), len(names))))
            for by, names in label_sets.items()}
    n_cells = done = 0
    for slide_id, first, end in slides:
        xy, codes = read(slide_id, int(first), int(end))
        nbrs = neighbour_indices(xy, max(ks))
        labels = {"types": codes.astype(np.int64), "classes": type_class[codes]}
        for by, names in label_sets.items():
            pair, target = excess_sums(xy, labels[by], len(names), ks, nbrs)
            sums[by][0][...] += pair
            sums[by][1][...] += target
        del nbrs  # (N, 100) int64: 800 B a cell, before the next slide's
        n_cells += len(codes)
        done += 1
        if done % 25 == 0 or done == n_slides:
            print(f"{done} of {n_slides} slides, {n_cells:,} cells, {time.time() - start:.0f} s", flush=True)
    if done != n_slides:
        raise RuntimeError(f"read {done} slides, not the {n_slides} expected")
    return {
        "description": DESCRIPTION,
        "k": list(ks),
        "label_sets": {
            by: {
                "labels": names,
                "pair_excess": np.round(sums[by][0], 3).tolist(),
                "target_excess": np.round(sums[by][1], 3).tolist(),
            }
            for by, names in label_sets.items()
        },
        "n_slides": n_slides,
        "n_cells": n_cells,
        "seconds": round(time.time() - start, 1),
    }


def fit_liver() -> dict:
    """The liver cohort's fit, from the cohort's object and slide table."""
    slides = serve.load_slide_table()
    types = serve.cell_types()
    type_class = np.array([int(k) for k in serve.c.class_codes(types)], dtype=np.int64)

    def read(slide_id: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        return serve.read_cells(slide_id, start, end, len(types))

    rows = ((r.slide_id, r.start, r.end) for r in slides.itertuples(index=False))
    return fit(rows, len(slides), types, type_class, read)


def fit_tissue(folder: Path) -> dict:
    """One GTEx tissue folder's fit, from the slides, cell types and class map the viewer gives it (``serve.Cohort``),
    with the tissue it is of and the object's cells, which the viewer checks the fit against."""
    cohort = serve.Cohort(None, folder)

    def read(slide_id: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        return serve.read_cells(slide_id, start, end, len(cohort.types), cohort.h5ad)

    rows = ((r.slide_id, r.start, r.end) for r in cohort.slides.itertuples(index=False))
    out = fit(rows, len(cohort.slides), cohort.types, cohort.type_class, read)
    out["tissue"] = {"slug": cohort.slug, "name": cohort.name, "h5ad": str(cohort.h5ad), "unmapped": cohort.unmapped}
    return out


def write(out: dict, path: Path) -> None:
    """``out`` with the scripts' MD5s and the versions, as JSON at ``path``, written whole or not at all."""
    out = {
        **out,
        "md5": {name: md5(HERE / name) for name in ("fit_colocalization.py", "local_colocalization.py")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(path)
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB) in {out['seconds']} s", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tissue", type=Path, help="a GTEx tissue folder (<slug>/ holding gtex_<slug>_cells.h5ad and "
                        "slides.csv) to fit instead of the liver cohort")
    parser.add_argument("--out-dir", type=Path, help="with --tissue: the folder to write <slug>.json to")
    args = parser.parse_args(argv)
    if (args.tissue is None) != (args.out_dir is None):
        parser.error("--tissue and --out-dir go together")
    if args.tissue is not None:
        args.tissue = args.tissue.expanduser().resolve()  # a folder's own name is its slug
        if not serve.find_tissues(args.tissue)[0] == [args.tissue]:
            parser.error(f"--tissue {args.tissue} is not one tissue folder (<slug>/ holding gtex_<slug>_cells.h5ad and "
                         "slides.csv)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.tissue is None:
        write(fit_liver(), OUT)
    else:
        write(fit_tissue(args.tissue), args.out_dir.expanduser() / f"{args.tissue.name}.json")


if __name__ == "__main__":
    main()
