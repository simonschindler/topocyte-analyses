#!/usr/bin/env python
"""Browser viewer of the GTEx liver cohort's cells: every slide's point cloud, coloured by HistoPLUS cell type
or by the study's four classes, with the slides filtered by pathology group. On one slide at a time it also
clusters every cell by the composition of its 150 µm ball (the pilot's level-1 features) with a chosen k.

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
import base64
import json
import math
import struct
import sys
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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
# Composition clusters: the pilot's level 1 (features.level1, domains.kmeans) on every cell of one slide.
K_MIN, K_MAX = 2, 10
N_INIT, KMEANS_SEED = 20, 0
FEATURES = ("log10 cells", *(f"{name} fraction" for name in c.CLASSES))
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
    """One slide's cells, read only: level-0 px coordinates (n, 2) as float64 and HistoPLUS type codes as uint8.

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
    return xy, codes.astype(np.uint8)


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


def slide_um_per_px(properties: dict[str, str]) -> tuple[float, str]:
    """µm per level-0 px from a slide file's OpenSlide properties, else the cohort's assumed ``PX_UM``.

    Returns the value and where it comes from: "slide file" or "assumed" (the cohort records no pixel size).
    """
    try:
        value = float(properties.get("openslide.mpp-x", "nan"))
    except ValueError:
        value = math.nan
    if math.isfinite(value) and value > 0:
        return value, "slide file"
    return c.PX_UM, "assumed"


def ball_class_counts(xy: np.ndarray, classes: np.ndarray, radius_px: float) -> np.ndarray:
    """(n, 4): the cells of each class in every cell's closed ball of ``radius_px``, the cell itself included.

    The balls of run_pilot.py: ``cKDTree.query_ball_point``, whose ball is closed.
    """
    from scipy.spatial import cKDTree

    counts = np.zeros((len(xy), len(c.CLASSES)), np.int64)
    for k in range(len(c.CLASSES)):
        members = xy[classes == k]
        if len(members):
            counts[:, k] = cKDTree(members).query_ball_point(xy, r=radius_px, return_length=True, workers=-1)
    return counts


def level1(counts: np.ndarray) -> np.ndarray:
    """(n, 5): log10 of each ball's cells, then its four class fractions (features.level1)."""
    cells = counts.sum(axis=1)
    if np.any(cells < 1):
        raise ValueError("every ball holds at least its centre cell")
    return np.column_stack([np.log10(cells), counts / cells[:, None]])


def composition_clusters(features: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
    """k-means of the z-scored columns (ddof 0) of ``features``: ``KMeans(k, n_init=20, random_state=0)``.

    Columns that are constant on the slide carry nothing and are left out. The labels are renumbered by cluster
    size, largest first. Returns the labels (uint8) and the indices of the columns used.
    """
    from sklearn.cluster import KMeans

    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k must be from {K_MIN} to {K_MAX}")
    sd = features.std(axis=0)
    used = [j for j in range(features.shape[1]) if sd[j] > 0]
    if not used:
        raise ValueError("every feature is constant on this slide")
    z = (features[:, used] - features[:, used].mean(axis=0)) / sd[used]
    labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=KMEANS_SEED).fit_predict(z)
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    return rank[labels].astype(np.uint8), used


def cluster_profiles(features: np.ndarray, labels: np.ndarray, k: int) -> list[dict]:
    """Each cluster's cells, mean cells per ball and mean class fractions."""
    profiles = []
    for j in range(k):
        members = features[labels == j]
        profiles.append(
            {
                "n": len(members),
                "cells_per_ball": float(np.mean(10 ** members[:, 0])) if len(members) else 0.0,
                "fractions": [float(v) for v in members[:, 1:].mean(axis=0)] if len(members) else [0.0] * 4,
            }
        )
    return profiles


def remember(cache: OrderedDict, key, value, size: int):
    """Put ``value`` in the LRU ``cache`` under ``key``, dropping the oldest entries beyond ``size``."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > size:
        cache.popitem(last=False)
    return value


class Cohort:
    """The slide table, the cell types and small caches: cells, slide files, ball features and clusterings."""

    def __init__(self, wsi_dir: Path | None) -> None:
        self.slides = load_slide_table()
        self.by_id = {row.slide_id: row for row in self.slides.itertuples(index=False)}
        self.types = cell_types()
        self.type_class = np.array([int(k) for k in c.class_codes(self.types)], dtype=np.int64)
        self.wsis = find_wsis(wsi_dir, set(self.by_id)) if wsi_dir else {}
        self._lock = threading.Lock()
        self._cells: OrderedDict[str, dict] = OrderedDict()  # slide ID -> xy, types, blob, extent
        self._zooms: OrderedDict[str, tuple] = OrderedDict()  # slide ID -> (OpenSlide, DeepZoomGenerator)
        self._features: OrderedDict[tuple, np.ndarray] = OrderedDict()  # (slide ID, µm/px) -> level-1 features
        self._clusters: OrderedDict[tuple, dict] = OrderedDict()  # (slide ID, k, density) -> clustering

    def meta(self) -> dict:
        counts = self.slides["group"].value_counts()
        return {
            "groups": [{"name": g, "n_slides": int(counts.get(g, 0))} for g in GROUPS],
            "types": self.types,
            "type_class": self.type_class.tolist(),
            "classes": list(c.CLASSES),
            "um_per_px": c.PX_UM,
            "cluster": {"k_min": K_MIN, "k_max": K_MAX, "radius_um": c.DELTA_UM, "n_init": N_INIT},
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

    def cells(self, slide_id: str) -> dict:
        """The slide's cells: ``xy`` (level-0 px), HistoPLUS ``types``, the ``blob`` for the page and the ``extent``."""
        with self._lock:
            if slide_id in self._cells:
                self._cells.move_to_end(slide_id)
                return self._cells[slide_id]
        row = self.by_id[slide_id]
        xy, types = read_cells(slide_id, int(row.start), int(row.end), len(self.types))
        entry = {"xy": xy, "types": types, "blob": cells_blob(xy, types), "extent": cloud_extent(xy)}
        with self._lock:
            return remember(self._cells, slide_id, entry, CACHE)

    def um_per_px(self, slide_id: str) -> tuple[float, str]:
        """µm per level-0 px of the slide, and where the value comes from."""
        if slide_id in self.wsis:
            return slide_um_per_px(dict(self.opened(slide_id)[0].properties))
        return slide_um_per_px({})

    def features(self, slide_id: str, um_per_px: float) -> np.ndarray:
        """The level-1 features of every cell's ``DELTA_UM`` ball (log10 cells, four class fractions)."""
        key = (slide_id, um_per_px)
        with self._lock:
            if key in self._features:
                self._features.move_to_end(key)
                return self._features[key]
        cells = self.cells(slide_id)
        counts = ball_class_counts(cells["xy"], self.type_class[cells["types"]], c.DELTA_UM / um_per_px)
        with self._lock:
            return remember(self._features, key, level1(counts), 2)

    def cluster(self, slide_id: str, k: int, density: bool) -> dict:
        """Composition clusters of every cell of the slide, with each cluster's profile, as JSON-ready data."""
        key = (slide_id, k, density)
        with self._lock:
            if key in self._clusters:
                self._clusters.move_to_end(key)
                return self._clusters[key]
        started = time.perf_counter()
        um_per_px, source = self.um_per_px(slide_id)
        features = self.features(slide_id, um_per_px)
        columns = [0, 1, 2, 3, 4] if density else [1, 2, 3, 4]
        labels, used = composition_clusters(features[:, columns], k)
        result = {
            "slide_id": slide_id,
            "k": k,
            "density": density,
            "radius_um": c.DELTA_UM,
            "um_per_px": um_per_px,
            "um_source": source,
            "features": [FEATURES[columns[j]] for j in used],
            "n_init": N_INIT,
            "seed": KMEANS_SEED,
            "clusters": cluster_profiles(features, labels, k),
            "labels": base64.b64encode(labels.tobytes()).decode("ascii"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        with self._lock:
            return remember(self._clusters, key, result, 8)

    def opened(self, slide_id: str) -> tuple:
        """The slide file opened with OpenSlide, and its DeepZoom generator over the whole slide (limit_bounds
        off, so level-0 px match the cells)."""
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
            return self._zooms[slide_id]

    def zoom(self, slide_id: str):
        return self.opened(slide_id)[1]

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
            out["width"], out["height"] = self.cells(slide_id)["extent"]
        out["um_per_px"], out["um_source"] = self.um_per_px(slide_id)
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
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            try:
                self.route(parts, parse_qs(url.query))
            except KeyError as err:
                self.fail(HTTPStatus.NOT_FOUND, f"not found: {err}")
            except ValueError as err:
                self.fail(HTTPStatus.BAD_REQUEST, str(err))
            except Exception as err:  # a slide that cannot be read must not stop the server
                self.log_error("%s: %r", self.path, err)
                self.fail(HTTPStatus.INTERNAL_SERVER_ERROR, str(err))

        def route(self, parts: list[str], query: dict[str, list[str]]) -> None:
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
                self.send(HTTPStatus.OK, cohort.cells(parts[2])["blob"], "application/octet-stream")
            elif len(parts) == 3 and parts[:2] == ["api", "cluster"] and parts[2] in cohort.by_id:
                k = int(query.get("k", [""])[0])
                density = query.get("density", ["1"])[0] == "1"
                result = cohort.cluster(parts[2], k, density)
                self.send(HTTPStatus.OK, json.dumps(result).encode(), "application/json")
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
