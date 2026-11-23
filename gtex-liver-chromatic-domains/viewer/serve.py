#!/usr/bin/env python
"""Browser viewer of the GTEx liver cohort's cells: every slide's point cloud, coloured by HistoPLUS cell type
or by the study's four classes, with the slides filtered by pathology group.

Adapted from ``tissuegeometry/web/cosine_viewer`` (its DeepZoom tile serving and WebGL point overlay) without
its local-homology endpoints, its FastAPI backend or its tissuegraphs and Rust imports.

Run from the study folder with the study venv::

    .venv/bin/python viewer/serve.py                       # point clouds only
    .venv/bin/python viewer/serve.py --wsi-dir DIR         # plus the H&E of every slide with a file in DIR

then open the printed address. A slide is shown on its H&E when DIR holds a file named ``<slide_id><suffix>``
with a suffix OpenSlide reads (``.svs`` for GTEx); the cells' coordinates are level-0 pixels of that slide.

Read only: the cohort ``.h5ad`` (h5py mode "r"), the pathology study's ``density.npz`` (each slide's cohort
rows) and ``pathology.csv`` (each slide's group), this study's ``results/pilot.csv`` and the slide files.
Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as c  # noqa: E402

STATIC = HERE / "static"
# pathology.csv's groups: the pilot's six conditions first, then the two the pilot left out.
GROUPS = (*c.CONDITIONS, "necrosis", "unclassified")
# File suffixes OpenSlide reads; a slide's file is <slide_id><suffix>.
WSI_SUFFIXES = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif", ".vms", ".vmu", ".svslide")
TILE_SIZE = 256  # DeepZoom tiles without overlap, as in cosine_viewer
MARGIN_PX = 256  # blank border around a point cloud shown without its slide
CACHE = 4  # slides whose cells, and slide files whose tile generators, stay in memory
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css"}


def slide_table(pathology: pd.DataFrame, slide_ids: list[str], bounds: np.ndarray, pilot: set[str]) -> pd.DataFrame:
    """Every cohort slide with its group, cohort rows (start, end), cell count and pilot flag.

    Ordered by group (``GROUPS``) and then slide ID. ``ValueError`` if a group is not in ``GROUPS``, if
    ``bounds`` does not have one entry more than ``slide_ids``, or if pathology.csv and density.npz do not
    hold the same slides.
    """
    unknown = sorted(set(pathology["group"]) - set(GROUPS))
    if unknown:
        raise ValueError(f"pathology groups not in GROUPS: {unknown}")
    if len(bounds) != len(slide_ids) + 1:
        raise ValueError(f"{len(bounds)} bounds for {len(slide_ids)} slides")
    rows = pd.DataFrame({"slide_id": slide_ids, "start": bounds[:-1], "end": bounds[1:]})
    if set(rows["slide_id"]) != set(pathology["slide_id"]) or len(rows) != len(pathology):
        raise ValueError("pathology.csv and density.npz do not hold the same slides")
    table = pathology[["slide_id", "group"]].merge(rows, on="slide_id", validate="one_to_one")
    table["n_cells"] = table["end"] - table["start"]
    table["pilot"] = table["slide_id"].isin(pilot)
    table["rank"] = table["group"].map(GROUPS.index)
    return table.sort_values(["rank", "slide_id"]).drop(columns="rank").reset_index(drop=True)


def load_slide_table() -> pd.DataFrame:
    """``slide_table`` from the sibling study's pathology.csv and density.npz and this study's pilot.csv."""
    pathology = pd.read_csv(c.SIB / "pathology.csv", usecols=["slide_id", "group"])
    with np.load(c.SIB / "density.npz", allow_pickle=False) as density:
        slide_ids = [str(s) for s in density["slide_ids"]]
        bounds = np.asarray(density["bounds"], dtype=np.int64)
    pilot = set(pd.read_csv(c.RESULTS / "pilot.csv")["slide_id"])
    return slide_table(pathology, slide_ids, bounds, pilot)


def cell_types() -> list[str]:
    """The cohort's HistoPLUS cell types; a cell's type code is its index here."""
    import h5py

    with h5py.File(c.H5AD, "r") as f:
        return c._strings(f["obs/cell_type/categories"][:])


def read_cells(slide_id: str, start: int, end: int, n_types: int) -> tuple[np.ndarray, np.ndarray]:
    """One slide's cells, read only: level-0 px coordinates (n, 2) as float32 and HistoPLUS type codes as uint8.

    The cohort file must name the slide at the first and the last of its rows, and every cell must have a
    finite position and a type.
    """
    import h5py

    with h5py.File(c.H5AD, "r") as f:
        names = c._strings(f["obs/slide_id/categories"][:])
        owners = f["obs/slide_id/codes"]
        first, last = names[int(owners[start])], names[int(owners[end - 1])]
        xy = np.asarray(f["obsm/spatial"][start:end], dtype=np.float64)
        codes = np.asarray(f["obs/cell_type/codes"][start:end])
    if first != slide_id or last != slide_id:
        raise RuntimeError(f"cohort rows {start}:{end} run from {first} to {last}, not {slide_id}")
    if xy.shape != (end - start, 2) or not np.all(np.isfinite(xy)):
        raise RuntimeError(f"{slide_id}: obsm/spatial has shape {xy.shape} or non-finite values")
    if codes.min() < 0 or codes.max() >= n_types:
        raise RuntimeError(f"{slide_id}: a cell has no cell type")
    return xy.astype(np.float32), codes.astype(np.uint8)


def cells_blob(xy: np.ndarray, types: np.ndarray) -> bytes:
    """``[n u32][n × (x, y) f32][n u8 type]``, little-endian; coordinates in level-0 px."""
    n = len(types)
    if xy.shape != (n, 2):
        raise ValueError(f"{xy.shape} coordinates for {n} types")
    head = struct.pack("<I", n)
    return (
        head + np.ascontiguousarray(xy, dtype="<f4").tobytes() + np.ascontiguousarray(types, dtype=np.uint8).tobytes()
    )


def cloud_extent(xy: np.ndarray) -> tuple[int, int]:
    """Width and height of a blank canvas that holds every cell with ``MARGIN_PX`` to spare, in level-0 px."""
    if len(xy) == 0:
        return 2 * MARGIN_PX, 2 * MARGIN_PX
    if xy.min() < 0:
        raise ValueError("negative cell coordinates")
    width, height = (math.ceil(float(v)) + MARGIN_PX for v in xy.max(axis=0))
    return width, height


def find_wsis(folder: Path, slide_ids: set[str]) -> dict[str, Path]:
    """The cohort slides with a file ``<slide_id><suffix>`` directly in ``folder``; ``ValueError`` on two files."""
    found: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in WSI_SUFFIXES or path.stem not in slide_ids:
            continue
        if path.stem in found:
            raise ValueError(f"two slide files for {path.stem}: {found[path.stem].name} and {path.name}")
        found[path.stem] = path
    return found


class Cohort:
    """The slide table, the cell types and two small caches: cells blobs and DeepZoom generators."""

    def __init__(self, wsi_dir: Path | None) -> None:
        self.slides = load_slide_table()
        self.by_id = {row.slide_id: row for row in self.slides.itertuples(index=False)}
        self.types = cell_types()
        self.type_class = [int(k) for k in c.class_codes(self.types)]
        self.wsis = find_wsis(wsi_dir, set(self.by_id)) if wsi_dir else {}
        self._lock = threading.Lock()
        self._cells: OrderedDict[str, tuple[bytes, tuple[int, int]]] = OrderedDict()
        self._zooms: OrderedDict[str, tuple] = OrderedDict()  # slide ID -> (OpenSlide, DeepZoomGenerator)

    def meta(self) -> dict:
        counts = self.slides["group"].value_counts()
        return {
            "groups": [{"name": g, "n_slides": int(counts.get(g, 0))} for g in GROUPS],
            "types": self.types,
            "type_class": self.type_class,
            "classes": list(c.CLASSES),
            "um_per_px": c.PX_UM,
            "slides": [
                {
                    "slide_id": r.slide_id,
                    "group": r.group,
                    "n_cells": int(r.n_cells),
                    "pilot": bool(r.pilot),
                    "wsi": r.slide_id in self.wsis,
                }
                for r in self.slides.itertuples(index=False)
            ],
        }

    def cells(self, slide_id: str) -> tuple[bytes, tuple[int, int]]:
        with self._lock:
            if slide_id in self._cells:
                self._cells.move_to_end(slide_id)
                return self._cells[slide_id]
        row = self.by_id[slide_id]
        xy, types = read_cells(slide_id, int(row.start), int(row.end), len(self.types))
        entry = (cells_blob(xy, types), cloud_extent(xy))
        with self._lock:
            self._cells[slide_id] = entry
            while len(self._cells) > CACHE:
                self._cells.popitem(last=False)
        return entry

    def zoom(self, slide_id: str):
        """The slide's DeepZoom generator over the whole slide (limit_bounds off, so level-0 px match the cells)."""
        from openslide import OpenSlide
        from openslide.deepzoom import DeepZoomGenerator

        with self._lock:
            if slide_id not in self._zooms:
                slide = OpenSlide(str(self.wsis[slide_id]))
                generator = DeepZoomGenerator(slide, tile_size=TILE_SIZE, overlap=0, limit_bounds=False)
                self._zooms[slide_id] = (slide, generator)
                while len(self._zooms) > CACHE:
                    self._zooms.popitem(last=False)[1][0].close()
            self._zooms.move_to_end(slide_id)
            return self._zooms[slide_id][1]

    def info(self, slide_id: str) -> dict:
        row = self.by_id[slide_id]
        out = {
            "slide_id": slide_id,
            "group": row.group,
            "n_cells": int(row.n_cells),
            "pilot": bool(row.pilot),
            "tile_size": TILE_SIZE,
            "wsi": slide_id in self.wsis,
        }
        if out["wsi"]:
            zoom = self.zoom(slide_id)
            out["width"], out["height"] = zoom.level_dimensions[-1]
            out["max_level"] = zoom.level_count - 1
        else:
            out["width"], out["height"] = self.cells(slide_id)[1]
        return out

    def tile(self, slide_id: str, level: int, col: int, row: int) -> bytes:
        zoom = self.zoom(slide_id)
        if not 0 <= level < zoom.level_count:
            raise KeyError(f"level {level}")
        cols, rows = zoom.level_tiles[level]
        if not (0 <= col < cols and 0 <= row < rows):
            raise KeyError(f"tile {col}, {row} at level {level}")
        buf = BytesIO()
        zoom.get_tile(level, (col, row)).convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def make_handler(cohort: Cohort) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # tiles would flood the log
            if "/api/tile/" not in self.path:
                super().log_message(fmt, *args)

        def send(self, status: HTTPStatus, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def fail(self, status: HTTPStatus, message: str) -> None:
            self.send(status, json.dumps({"error": message}).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
            parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
            try:
                self.route(parts)
            except KeyError as err:
                self.fail(HTTPStatus.NOT_FOUND, f"not found: {err}")
            except Exception as err:  # a slide that cannot be read must not stop the server
                self.log_error("%s: %r", self.path, err)
                self.fail(HTTPStatus.INTERNAL_SERVER_ERROR, str(err))

        def route(self, parts: list[str]) -> None:
            if parts in ([], ["index.html"]):
                self.send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), STATIC_TYPES[".html"])
            elif len(parts) == 2 and parts[0] == "static":
                path = (STATIC / parts[1]).resolve()
                if path.parent != STATIC.resolve() or path.suffix not in STATIC_TYPES or not path.is_file():
                    raise KeyError(parts[1])
                self.send(HTTPStatus.OK, path.read_bytes(), STATIC_TYPES[path.suffix])
            elif parts == ["api", "meta"]:
                self.send(HTTPStatus.OK, json.dumps(cohort.meta()).encode(), "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "slide"] and parts[2] in cohort.by_id:
                self.send(HTTPStatus.OK, json.dumps(cohort.info(parts[2])).encode(), "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "cells"] and parts[2] in cohort.by_id:
                self.send(HTTPStatus.OK, cohort.cells(parts[2])[0], "application/octet-stream")
            elif len(parts) == 5 and parts[:2] == ["api", "tile"] and parts[2] in cohort.wsis:
                col, row = parts[4].removesuffix(".jpeg").split("_")
                tile = cohort.tile(parts[2], int(parts[3]), int(col), int(row))
                self.send(HTTPStatus.OK, tile, "image/jpeg", cache=True)
            else:
                raise KeyError("/".join(parts))

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--wsi-dir", type=Path, help="folder with slide files named <slide_id>.svs (or another OpenSlide format)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.wsi_dir is not None:
        args.wsi_dir = args.wsi_dir.expanduser()
        if not args.wsi_dir.is_dir():
            parser.error(f"--wsi-dir {args.wsi_dir} is not a folder")
        try:
            import openslide  # noqa: F401
        except ImportError:
            parser.error("--wsi-dir needs openslide-python and openslide-bin (requirements.txt)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cohort = Cohort(args.wsi_dir)
    print(f"{len(cohort.slides)} slides, {len(cohort.types)} cell types", flush=True)
    if args.wsi_dir is not None:
        print(f"{len(cohort.wsis)} of them with a slide file in {args.wsi_dir}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cohort))
    print(f"open http://{args.host}:{args.port}/  (Ctrl-C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
