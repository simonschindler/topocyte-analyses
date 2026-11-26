#!/usr/bin/env python
"""Browser viewer of the GTEx liver cohort's cells: every slide's point cloud, coloured by HistoPLUS cell type
or by the cell-type study's four classes, with the slides filtered by pathology group. On one slide at a time it
also clusters every cell by the composition of its ball (the chromatic-domains pilot's level-1 features) with a
chosen k and radius, or, on that pilot's 30 slides, the 1,500 sampled centres by their saved chromatic TDA
features (level 2), or colours every cell by its saved cohort-wide e-means cluster (gtex-liver-emeans-neighbourhoods).
Every cell can also be coloured by its 10-nearest-neighbour density, binned at the cohort's deciles, or by its
Voronoi density (1 / the area of its Voronoi cell) in the same bins, or by its homotypic fraction or local
Shannon entropy among its k nearest other cells, or by its hypergeometric colocalization excess there
(local_colocalization.py, on the scale fit_colocalization.py fitted over the cohort), and the cells below a
threshold of the colouring's value can be hidden. The shown slide's GTEx pathology notes and its donor's sex,
age bracket and Hardy scale are shown.

Adapted from ``tissuegeometry/web/cosine_viewer`` (its DeepZoom tile serving and WebGL point overlay) without
its local-homology endpoints, its FastAPI backend or its tissuegraphs and Rust imports.

Run from this folder with the chromatic-domains study's venv (README.md)::

    ../gtex-liver-chromatic-domains/.venv/bin/python serve.py                  # point clouds only
    ../gtex-liver-chromatic-domains/.venv/bin/python serve.py --wsi-dir DIR    # plus the H&E of the slides in DIR
    ../gtex-liver-chromatic-domains/.venv/bin/python serve.py --tissues ROOT   # GTEx tissue folders instead

then open the printed address. A slide is shown on its H&E when DIR holds a file named ``<slide_id><suffix>``
with a suffix OpenSlide reads (``.svs`` for GTEx); the cells' coordinates are level-0 pixels of that slide.

Read only: the cohort ``.h5ad`` (h5py mode "r"), the pathology study's ``density.npz`` (each slide's cohort
rows, and every cell's density, memory-mapped) and ``pathology.csv`` (each slide's group, GTEx pathology
categories and notes), the cohort table ``liver_cohort.csv`` beside the ``.h5ad`` (each donor's sex, age
bracket and Hardy scale), the chromatic-domains study's ``results/pilot.csv`` and ``results/features.npz``,
the e-means study's ``results/labels_<set>.npz``, ``clusters.json``, ``gap.json`` and ``kmeans_check.json``
(the e-means option is off when they are not there), this folder's ``results/colocalization_scale.json``
(the colocalization option is off without it), and the slide files.

With ``--tissues``, the server shows GTEx tissue folders instead of the liver cohort: one folder ``<slug>/`` of the
per-tissue layout, or a folder of them, among which the page chooses (``?tissue=<slug>``). It reads each folder's
``gtex_<slug>_cells.h5ad`` (h5py mode "r") and ``slides.csv`` (each slide's rows, GTEx Portal donor fields and
pathology categories and notes), and nothing else there. The layers that need the liver cohort's precomputed files
(the 10-NN density, the pathology groups, the chromatic TDA, the e-means clusters, the colocalization scale) are
off; a cell type outside the four classes counts as "other".
Nothing is written.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import math
import os
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

# BLAS's thread pools, set before numpy loads them: one thread per core of the node, each with its own buffers,
# does not fit a capped address space, and the viewer's linear algebra is small. A value already set is kept.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from local_colocalization import WORKERS, LocalColocalizationExtractor  # noqa: E402

HERE = Path(__file__).resolve().parent
# The chromatic-domains study's common module: the cohort's paths, the four classes and the pilot's constants.
sys.path.insert(0, str(HERE.parent / "gtex-liver-chromatic-domains"))
import common as c  # noqa: E402

STATIC = HERE / "static"
# pathology.csv's groups: the pilot's six conditions first, then the two the pilot left out.
GROUPS = (*c.CONDITIONS, "necrosis", "unclassified")
# File suffixes OpenSlide reads; a slide's file is <slide_id><suffix>.
WSI_SUFFIXES = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif", ".vms", ".vmu", ".svslide")
TILE_SIZE = 256  # DeepZoom tiles without overlap, as in cosine_viewer
MARGIN_PX = 256  # blank border around a point cloud shown without its slide
CACHE = 4  # slides whose cells, and slide files whose tile generators, stay in memory
# Composition clusters: the chromatic-domains pilot's level 1 (its features.level1, domains.kmeans) on every cell
# of one slide.
K_MIN, K_MAX = 2, 10
RADIUS_MIN_UM, RADIUS_MAX_UM = 10.0, 500.0  # the ball's radius; the pilot's is c.DELTA_UM (150 µm)
N_INIT, KMEANS_SEED = 20, 0
FEATURES = ("log10 cells", *(f"{name} fraction" for name in c.CLASSES))
# Chromatic TDA clusters: domains.level2's features and k-means on one pilot slide's saved centres.
TDA_KEY = "L2_obs_all"  # 20 MFA scores of the 16 blocks' fPCA scores of the residual Betti curves, all-slides basis
BACKGROUND = 15  # the code of the cells that are not centres, in a TDA clustering
# Cohort e-means clusters: the saved labels of every cohort cell, in the cohort's row order, k = 2..10 per set.
EMEANS = HERE.parent / "gtex-liver-emeans-neighbourhoods" / "results"
EMEANS_SETS = ("composition", "density", "composition+density")
WORSE = 0.002  # log ratio of W_k above which a fit is named worse than k-means++'s (the e-means study's report)
PACKED = 3  # (set, k) whose packed labels of every cohort cell (63 MB each) stay in memory
DENSITY_BINS = 10  # the density layer's bins: the cohort's deciles of the 10-NN density, the density slices
CURVE_STEPS = 1000  # the threshold slider's steps: the cohort's density quantiles at levels 0, 0.001, ..., 1
VORONOI_NONE = DENSITY_BINS  # the code of a cell without a Voronoi density, after the ten density bins
# Homotypic fraction and local entropy: each cell's k nearest other cells, by HistoPLUS type or by class.
NEIGHBOURS = 15  # the default k
NEIGHBOURS_MIN, NEIGHBOURS_MAX = 2, 100
LABEL_SETS = ("types", "classes")
UNIT_BINS = 10  # both values lie in [0, 1], coloured in bins of 0.1
# The same against the slide's composition: the homotypic fraction's excess over the slide's share of the cell's own
# type, in 11 bins of 0.2 around chance (0), and the Kullback-Leibler divergence in nats, in steps of KL_STEP.
EXCESS_EDGES = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
KL_STEP = 0.25
API = 4  # what /api/meta and the endpoints give; app.js refuses a server with another, as one started before an update
# Colocalization excess: fit_colocalization.py's cohort sums, which fix each pair's gain or loss scale per k.
COLOCALIZATION = HERE / "results" / "colocalization_scale.json"
COLOCALIZATION_NONE = len(EXCESS_EDGES) + 1  # the code of a cell without an estimate, after the 11 excess bins
COLOCALIZATION_CACHE = 2  # (slide, k, label set) whose cell-by-type excess (up to 80 MB each) stays in memory
COLOCALIZATION_SCORES = ("shrunk", "raw")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css"}
# GTEx tissue folders (--tissues): slides.csv's GTEx Portal columns, as the viewer names them, and its row columns.
TISSUE_FIELDS = {
    "Tissue Sample ID": "slide_id",
    "Tissue": "tissue",
    "Subject ID": "subject_id",
    "Sex": "sex",
    "Age Bracket": "age_bracket",
    "Hardy Scale": "hardy_scale",
    "Pathology Categories": "categories",
    "Pathology Notes": "notes",
    "Hidden": "hidden",
}
ROW_FIELDS = ("n_cells", "row_start", "row_end")  # row_end exclusive, in the object's row order
MAX_TYPES = 16  # cell types the page can colour: its colour table has one entry per code (overlay.js's MAX_TYPES)
WARM_TISSUES = 2  # tissue folders whose slides stay in memory (up to the caches' sizes above), the last ones used


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
    """``slide_table`` from the pathology study's pathology.csv and density.npz and the chromatic pilot's pilot.csv."""
    pathology = pd.read_csv(c.SIB / "pathology.csv", usecols=["slide_id", "group"])
    with np.load(c.SIB / "density.npz", allow_pickle=False) as density:
        slide_ids = [str(s) for s in density["slide_ids"]]
        bounds = np.asarray(density["bounds"], dtype=np.int64)
    pilot = set(pd.read_csv(c.RESULTS / "pilot.csv")["slide_id"])
    return slide_table(pathology, slide_ids, bounds, pilot)


def npz_memmap(path: Path, name: str) -> np.memmap:
    """A member of an uncompressed ``.npz`` file, memory-mapped read only.

    ``np.load`` reads a whole member into memory, and the density of every cohort cell is 1 GB of which a slide
    needs a few MB. ``ValueError`` if the member is compressed or its headers are not where the zip says.
    """
    import zipfile

    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{name}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path.name}: {name} is compressed and cannot be memory-mapped")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError(f"{path.name}: no zip header where {name} should start")
        name_length, extra_length = struct.unpack("<HH", local[26:30])
        f.seek(info.header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"{path.name}: {name} has .npy format {version}")
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def density_codes(rho: np.ndarray, deciles: np.ndarray) -> np.ndarray:
    """Each cell's density bin (uint8): j where d_j <= rho < d_(j+1), and the last bin from the top decile on.

    That is the last density slice that keeps the cell: the pathology study keeps a cell at threshold j when
    rho >= d_j, and d_0 is the cohort's minimum.
    """
    return np.searchsorted(deciles[1:], rho, side="right").astype(np.uint8)


def density_blob(rho: np.ndarray, bins: np.ndarray) -> bytes:
    """``[n u32][n × f32 density][n u8 bin]``, little-endian: a slide's densities (cells per mm²) and bins."""
    n = len(bins)
    if rho.shape != (n,):
        raise ValueError(f"{rho.shape} densities for {n} bins")
    return (
        struct.pack("<I", n)
        + np.ascontiguousarray(rho, dtype="<f4").tobytes()
        + np.ascontiguousarray(bins, dtype=np.uint8).tobytes()
    )


def voronoi_areas(xy: np.ndarray) -> np.ndarray:
    """The area of each point's Voronoi cell among the points ``xy`` (n × 2), in squared units of ``xy``.

    NaN where a point has no cell of its own: on the convex hull its cell is unbounded, and points at one
    position share theirs. Exact on the Delaunay triangulation (``scipy.spatial.Delaunay``): a bounded cell is
    the polygon of the circumcentres of the triangles around its point, of area (1/8) Σ (cot α + cot β) |ab|²
    over the point's edges ab, where α and β are the angles opposite ab; the sum is taken triangle by triangle.
    Fewer than three points, or points on one line, give NaN everywhere.
    """
    from scipy.spatial import Delaunay, QhullError

    q = np.asarray(xy, dtype=np.float64)
    n = len(q)
    area = np.full(n, np.nan)
    if n < 3:
        return area
    q = q - q.mean(axis=0)
    try:
        tri = Delaunay(q)
    except QhullError:  # no three points off one line
        return area
    corners = q[tri.simplices]  # triangles × 3 × 2
    e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    twice = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])  # twice each triangle's area
    flat = twice == 0
    total = np.zeros(n)
    for a in range(3):
        b, d = (a + 1) % 3, (a + 2) % 3
        u, v = corners[:, b] - corners[:, a], corners[:, d] - corners[:, a]
        cot = np.where(flat, 0.0, (u * v).sum(axis=1) / np.where(flat, 1.0, twice))  # the angle at corner a
        share = cot * ((corners[:, d] - corners[:, b]) ** 2).sum(axis=1) / 8  # to both ends of the edge opposite a
        total += np.bincount(tri.simplices[:, b], share, n) + np.bincount(tri.simplices[:, d], share, n)
    none = np.zeros(n, dtype=bool)
    none[tri.convex_hull.ravel()] = True
    none[tri.coplanar[:, 0]] = True  # left out of the triangulation: at (or within rounding of) a vertex's position
    none[tri.coplanar[:, 2]] = True  # that vertex
    none[tri.simplices[flat].ravel()] = True
    area[~none] = total[~none]
    return area


def voronoi_density(xy: np.ndarray, um_per_px: float) -> np.ndarray:
    """Each cell's Voronoi density, 1 / the area of its Voronoi cell, in cells per mm²; NaN where it has none."""
    return 1e6 / (voronoi_areas(xy) * um_per_px**2)


def voronoi_codes(rho: np.ndarray, deciles: np.ndarray) -> np.ndarray:
    """Each cell's bin by ``density_codes``' rule on the 10-NN deciles, and ``VORONOI_NONE`` where ``rho`` is NaN."""
    codes = np.full(len(rho), VORONOI_NONE, dtype=np.uint8)
    some = np.isfinite(rho)
    codes[some] = density_codes(rho[some], deciles)
    return codes


def nearest_others(xy: np.ndarray, k: int) -> np.ndarray:
    """(n, k): each cell's k nearest other cells, nearest first (``cKDTree``); the cell itself is left out, even
    where another cell shares its position. ``ValueError`` unless there are more than k cells."""
    from scipy.spatial import cKDTree

    n = len(xy)
    if n <= k:
        raise ValueError(f"{n} cells, too few for {k} neighbours each")
    idx = cKDTree(xy).query(xy, k=k + 1, workers=WORKERS)[1]
    others = idx != np.arange(n)[:, None]
    others[others.all(axis=1), -1] = False  # k + 1 others at one distance: the cell itself fell out, drop the last
    return idx[others].reshape(n, k)


def homotypic_counts(labels: np.ndarray, nbrs: np.ndarray) -> np.ndarray:
    """Each cell's neighbours (rows of ``nbrs``) with its own label: the homotypic fraction times k."""
    return (labels[nbrs] == labels[:, None]).sum(axis=1)


def entropy_nats(labels: np.ndarray, nbrs: np.ndarray, n_labels: int) -> np.ndarray:
    """The Shannon entropy of each cell's neighbours' labels, in nats.

    With counts c of the k neighbours, H = log k - (1/k) Σ c log c, where 0 log 0 = 0.
    """
    n, k = nbrs.shape
    neighbour_labels = labels[nbrs]
    c_log_c = np.zeros(k + 1)  # c log c for the counts c = 0, 1, ..., k
    c_log_c[1:] = np.arange(1, k + 1) * np.log(np.arange(1, k + 1))
    total = np.zeros(n)
    for label in range(n_labels):
        total += c_log_c[(neighbour_labels == label).sum(axis=1)]
    return np.log(k) - total / k


def local_entropy(labels: np.ndarray, nbrs: np.ndarray, n_labels: int) -> np.ndarray:
    """``entropy_nats`` divided by log(n_labels): Pielou's evenness, in [0, 1]."""
    return np.clip(entropy_nats(labels, nbrs, n_labels) / np.log(n_labels), 0.0, 1.0)


def slide_shares(labels: np.ndarray, n_labels: int) -> np.ndarray:
    """The share of the slide's cells with each label: the composition the two corrections measure against."""
    return np.bincount(labels, minlength=n_labels) / len(labels)


def homotypic_excess(counts: np.ndarray, k: int, labels: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """The homotypic fraction counts / k less the slide's share of the cell's own label: 0 is what chance gives."""
    return counts / k - shares[labels]


def kl_divergence(labels: np.ndarray, nbrs: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """The Kullback-Leibler divergence D(p_i || P) of each cell's neighbours' labels p_i from the slide's shares P,
    in nats: the cross-entropy -(1/k) Σ_j log P(label of neighbour j) less ``entropy_nats``. Every neighbour is a
    cell of the slide, so P is positive wherever p_i is."""
    log_shares = np.log(np.where(shares > 0, shares, 1.0))  # a label absent from the slide is no one's neighbour
    cross = -log_shares[labels[nbrs]].mean(axis=1)
    return np.maximum(cross - entropy_nats(labels, nbrs, len(shares)), 0.0)


def excess_codes(values: np.ndarray) -> np.ndarray:
    """The bin of each excess among the 11 that ``EXCESS_EDGES`` bound, each closed below: 5 is chance, ±0.1."""
    return np.searchsorted(EXCESS_EDGES, values, side="right").astype(np.uint8)


def kl_codes(values: np.ndarray) -> np.ndarray:
    """The bin of each divergence: j for [j, j + 1) × ``KL_STEP`` nats, the last one open above."""
    return np.minimum(np.floor(values / KL_STEP + 1e-9), UNIT_BINS - 1).astype(np.uint8)


def unit_codes(values: np.ndarray) -> np.ndarray:
    """The bin of each value in [0, 1]: j for [j/10, (j+1)/10), the last bin with 1 itself. A value that is a tenth
    up to rounding (an even split, say) goes up into the bin that it starts."""
    return np.minimum(np.floor(values * UNIT_BINS + 1e-9), UNIT_BINS - 1).astype(np.uint8)


def homotypic_codes(counts: np.ndarray, k: int) -> np.ndarray:
    """``unit_codes`` of the fractions counts / k, in integers, so no fraction is binned by its rounding."""
    return np.minimum(UNIT_BINS * counts // k, UNIT_BINS - 1).astype(np.uint8)


def neighbourhood_request(query: dict[str, list[str]]) -> tuple[int, str]:
    """k and the label set (``LABEL_SETS``) of a neighbourhood request's query; ``ValueError`` if invalid."""
    try:
        k = int(query.get("k", [str(NEIGHBOURS)])[0])
    except ValueError:
        raise ValueError("k must be a whole number") from None
    if not NEIGHBOURS_MIN <= k <= NEIGHBOURS_MAX:
        raise ValueError(f"k must be from {NEIGHBOURS_MIN} to {NEIGHBOURS_MAX}")
    by = query.get("by", ["types"])[0]
    if by not in LABEL_SETS:
        raise ValueError(f"by must be one of {', '.join(LABEL_SETS)}")
    return k, by


def neighbourhood_blob(values: list[np.ndarray], codes: list[np.ndarray]) -> bytes:
    """``[n u32]``, then each value per cell as ``n × f32``, then each bin as ``n × u8``, little-endian, in the
    order given (``Cohort.neighbourhood``: homotypic fraction, local entropy, homotypic excess, KL divergence)."""
    n = len(values[0])
    if len(values) != len(codes) or any(a.shape != (n,) for a in (*values, *codes)):
        raise ValueError("the values and bins are not one per cell")
    return (
        struct.pack("<I", n)
        + b"".join(np.ascontiguousarray(v, dtype="<f4").tobytes() for v in values)
        + b"".join(np.ascontiguousarray(c, dtype=np.uint8).tobytes() for c in codes)
    )


def load_colocalization(path: Path) -> dict | None:
    """fit_colocalization.py's summed excess as arrays, per label set: ``labels``, ``pair`` (k, K, K) and
    ``target`` (k, K), with the ``k`` they are for and the slides they sum; None without the file."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    out: dict = {
        "k": [int(k) for k in data["k"]],
        "n_slides": int(data["n_slides"]),
        "n_cells": None if data.get("n_cells") is None else int(data["n_cells"]),  # None in a fit that does not say
    }
    for by, fit in data["label_sets"].items():
        pair = np.asarray(fit["pair_excess"], dtype=np.float64)
        target = np.asarray(fit["target_excess"], dtype=np.float64)
        n = len(fit["labels"])
        if pair.shape != (len(out["k"]), n, n) or target.shape != (len(out["k"]), n):
            raise ValueError(f"{path.name}: {by}'s sums are {pair.shape} and {target.shape} for {n} labels")
        out[by] = {"labels": list(fit["labels"]), "pair": pair, "target": target}
    return out


def tissue_colocalization(
    path: Path, types: list[str], n_slides: int, n_cells: int
) -> tuple[dict | None, str | None]:
    """A tissue's fit (``load_colocalization``) at ``path``, ``fit_colocalization.py --tissue``'s, if it is of this
    tissue's object: its cell types and the four classes as labels, in order, and its number of slides and of
    cells. Else None, with why (None too when there is no file: that tissue's excess is then off, as without a fit).
    A fit of another object would put another tissue's pair scale, or an older build's, on these cells."""
    try:
        fit = load_colocalization(path)
    except (OSError, ValueError, KeyError, TypeError) as err:  # a truncated or foreign JSON among others
        return None, f"{path.name} cannot be read ({type(err).__name__}: {err})"
    if fit is None:
        return None, None
    labels = {by: fit[by]["labels"] if by in fit else None for by in LABEL_SETS}
    if labels != {"types": list(types), "classes": list(c.CLASSES)}:
        return None, f"{path.name} is of other cell types than the object's"
    if (fit["n_slides"], fit["n_cells"]) != (n_slides, n_cells):
        cells = "an unstated number of" if fit["n_cells"] is None else f"{fit['n_cells']:,}"
        return None, (f"{path.name} sums {fit['n_slides']} slides and {cells} cells, the object has "
                      f"{n_slides} and {n_cells:,}: fit it again")
    return fit, None


def colocalization_request(query: dict[str, list[str]], n_labels: dict[str, int]) -> tuple[int, str, int | None, str]:
    """k, the label set, the neighbour type (None: the cell's own) and the score (``COLOCALIZATION_SCORES``) of a
    colocalization request's query; ``ValueError`` if invalid."""
    k, by = neighbourhood_request(query)
    of = query.get("of", ["own"])[0]
    target = None
    if of != "own":
        try:
            target = int(of)
        except ValueError:
            raise ValueError("of must be own or the index of a type") from None
        if not 0 <= target < n_labels[by]:
            raise ValueError(f"of must be own or from 0 to {n_labels[by] - 1}")
    score = query.get("score", ["shrunk"])[0]
    if score not in COLOCALIZATION_SCORES:
        raise ValueError(f"score must be one of {', '.join(COLOCALIZATION_SCORES)}")
    return k, by, target, score


def colocalization_codes(values: np.ndarray) -> np.ndarray:
    """``excess_codes`` of each value, and ``COLOCALIZATION_NONE`` where it is NaN (no estimate)."""
    codes = np.full(len(values), COLOCALIZATION_NONE, dtype=np.uint8)
    some = np.isfinite(values)
    codes[some] = excess_codes(values[some])
    return codes


def threshold_curve(curve: np.ndarray, levels: np.ndarray, deciles: np.ndarray) -> np.ndarray:
    """The cohort's density quantiles at levels 0, 0.001, ..., 1 (the pathology study's curve), with every hundredth
    step set to that study's decile, so that the threshold slider's steps 0, 100, ..., 900 are the density slices'
    thresholds. ``ValueError`` if the levels are not those steps or the result does not rise."""
    if len(curve) != CURVE_STEPS + 1 or not np.allclose(levels, np.linspace(0, 1, CURVE_STEPS + 1), rtol=0, atol=1e-12):
        raise ValueError(f"the density curve has {len(curve)} levels, not 0 to 1 in steps of 1/{CURVE_STEPS}")
    out = np.array(curve, dtype=np.float64)
    out[0 : CURVE_STEPS : CURVE_STEPS // DENSITY_BINS] = deciles
    if np.any(np.diff(out) < 0):
        raise ValueError("the density curve with the deciles in place does not rise")
    return out


def load_notes(path: Path) -> dict[str, dict[str, str]]:
    """Each slide's GTEx pathology categories and notes and its quality flag, from the pathology study's table."""
    columns = ["slide_id", "categories", "notes", "quality_flag"]
    table = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    return {row.slide_id: {key: getattr(row, key) for key in columns[1:]} for row in table.itertuples(index=False)}


def load_donors(path: Path) -> dict[str, dict[str, str]]:
    """Each slide's donor sex, age bracket and Hardy scale (the circumstances of death), from the cohort table.

    ``ValueError`` if a slide appears twice.
    """
    columns = {"Tissue Sample ID": "slide_id", "Sex": "sex", "Age Bracket": "age_bracket", "Hardy Scale": "hardy_scale"}
    table = pd.read_csv(path, usecols=list(columns), dtype=str, keep_default_na=False).rename(columns=columns)
    if table["slide_id"].duplicated().any():
        raise ValueError(f"{path.name} lists a slide twice")
    return {row.slide_id: {key: getattr(row, key) for key in list(columns.values())[1:]} for row in table.itertuples()}


def tissue_h5ad(folder: Path) -> Path:
    """A tissue folder's object: ``<slug>/gtex_<slug>_cells.h5ad``, the slug being the folder's name."""
    return folder / f"gtex_{folder.name}_cells.h5ad"


def find_tissues(path: Path) -> tuple[list[Path], dict[str, str]]:
    """The GTEx tissue folders at ``path``: ``path`` itself when it is one, else its subfolders that are, by name;
    and the subfolders that hold one of the two files a tissue folder needs but not the other, with the one they
    lack (a folder still being copied, say). A tissue folder holds ``gtex_<its name>_cells.h5ad`` and ``slides.csv``;
    nothing else in it (provenance.json, centroids/) or beside it (tissues.csv) is read."""

    def lacks(folder: Path) -> list[str]:
        return [name for name in (tissue_h5ad(folder).name, "slides.csv") if not (folder / name).is_file()]

    if not lacks(path):
        return [path], {}
    found, skipped = [], {}
    for folder in sorted(p for p in path.iterdir() if p.is_dir()):
        missing = lacks(folder)
        if not missing:
            found.append(folder)
        elif len(missing) == 1:
            skipped[folder.name] = f"no {missing[0]}"
    return found, skipped


def h5ad_slides(path: Path) -> tuple[list[str], int]:
    """An object's slide IDs (``obs/slide_id``'s categories) and its number of cells (rows)."""
    import h5py

    with h5py.File(path, "r") as f:
        return c._strings(f["obs/slide_id/categories"][:]), int(f["obs/slide_id/codes"].shape[0])


def load_tissue_slides(path: Path, names: list[str], n_rows: int) -> pd.DataFrame:
    """A tissue folder's slides.csv: per slide its rows (``start``, ``end`` exclusive, ``n_cells``) and the GTEx
    Portal fields, as text (an empty field stays empty, "NA" stays "NA"), named as ``TISSUE_FIELDS`` names them;
    ``hidden`` is True where the Portal's Hidden is "true". In the file's order.

    ``ValueError`` if a column is missing, the rows are not whole numbers, a slide is listed twice or has no cells,
    ``n_cells`` is not ``row_end - row_start``, the rows do not run without gaps from 0 to the object's ``n_rows``
    in the file's order, or the object (whose slide IDs are ``names``) does not name a slide.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in (*TISSUE_FIELDS, *ROW_FIELDS) if column not in table.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    if table.empty:
        raise ValueError(f"{path} lists no slides")
    try:
        n, start, end = (table[column].astype(np.int64).to_numpy() for column in ROW_FIELDS)
    except ValueError:
        raise ValueError(f"{path}: n_cells, row_start and row_end must be whole numbers") from None
    table = table.rename(columns=TISSUE_FIELDS)
    if table["slide_id"].duplicated().any():
        raise ValueError(f"{path} lists a slide twice: {table.loc[table['slide_id'].duplicated(), 'slide_id'].iloc[0]}")
    if np.any(end - start != n) or np.any(n < 1):
        raise ValueError(f"{path}: n_cells is not row_end - row_start, or a slide has no cells")
    if start[0] != 0 or np.any(start[1:] != end[:-1]) or end[-1] != n_rows:
        raise ValueError(f"{path}: the slides' rows do not run without gaps from 0 to the object's {n_rows:,} cells")
    unknown = sorted(set(table["slide_id"]) - set(names))
    if unknown:
        raise ValueError(f"{path} lists {len(unknown)} slides that the object does not name, {unknown[0]} first")
    out = table[[name for name in TISSUE_FIELDS.values() if name != "hidden"]].copy()
    out["hidden"] = table["hidden"].str.strip().str.lower() == "true"
    out["start"], out["end"], out["n_cells"] = start, end, n
    return out


def class_codes_or_other(types: list[str]) -> tuple[np.ndarray, list[str]]:
    """The class code (index into ``CLASSES``) of every cell type, as ``common.class_codes`` gives it, but with the
    types outside the cell-type study's four classes in "other" rather than refused; and those types, in order.
    Every liver type has its class, so on liver's types this is ``common.class_codes``."""
    lookup = {member: code for code, name in enumerate(c.CLASSES) for member in c.GROUP_MEMBERS[name]}
    other = c.CLASSES.index("other")
    return np.array([lookup.get(t, other) for t in types], dtype=np.int8), [t for t in types if t not in lookup]


def slide_deciles(rho: np.ndarray) -> np.ndarray:
    """The deciles of a slide's own finite values (numpy's linear quantiles at 0, 0.1, ..., 0.9): the Voronoi
    density's bins in a tissue folder, where there are no cohort deciles; NaN where no value is finite."""
    finite = rho[np.isfinite(rho)]
    if not finite.size:
        return np.full(DENSITY_BINS, np.nan)
    return np.quantile(finite, np.arange(DENSITY_BINS) / DENSITY_BINS)


def cell_types(path: Path | None = None) -> list[str]:
    """The HistoPLUS cell types of the cohort (or of the object at ``path``); a cell's type code is its index here."""
    import h5py

    with h5py.File(c.H5AD if path is None else path, "r") as f:
        return c._strings(f["obs/cell_type/categories"][:])


def read_cells(
    slide_id: str, start: int, end: int, n_types: int, path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One slide's cells, read only from the cohort (or the object at ``path``): level-0 px coordinates (n, 2) as
    float64 and HistoPLUS type codes as uint8.

    The file must name the slide at the first and the last of its rows and not at the rows just before and after
    them (so that they are all of its cells, as a slide's cells are one block of rows), and every cell must have a
    finite position and a type.
    """
    import h5py

    with h5py.File(c.H5AD if path is None else path, "r") as f:
        names = c._strings(f["obs/slide_id/categories"][:])
        owners = f["obs/slide_id/codes"]
        first, last = names[int(owners[start])], names[int(owners[end - 1])]
        beside = [int(owners[row]) for row in (start - 1, end) if 0 <= row < owners.shape[0]]
        xy = np.asarray(f["obsm/spatial"][start:end], dtype=np.float64)
        codes = np.asarray(f["obs/cell_type/codes"][start:end])
    if first != slide_id or last != slide_id:
        raise RuntimeError(f"cohort rows {start}:{end} run from {first} to {last}, not {slide_id}")
    if any(code >= 0 and names[code] == slide_id for code in beside):
        raise RuntimeError(f"cohort rows {start}:{end} are only part of {slide_id}'s cells, which run on beside them")
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


def list_wsis(folder: Path) -> dict[str, list[Path]]:
    """The files directly in ``folder`` with a suffix OpenSlide reads, by name without it (the slide ID): one
    listing, which may be of every GTEx slide (25,713 on the cluster), checking whether an entry is a file only
    when its name has such a suffix."""
    listing: dict[str, list[Path]] = {}
    with os.scandir(folder) as entries:
        for entry in entries:
            path = Path(entry.path)
            if path.suffix.lower() in WSI_SUFFIXES and entry.is_file():
                listing.setdefault(path.stem, []).append(path)
    return listing


def find_wsis(folder: Path, slide_ids: set[str], listing: dict[str, list[Path]] | None = None) -> dict[str, Path]:
    """The cohort slides with a file ``<slide_id><suffix>`` directly in ``folder``; ``ValueError`` on two files.
    ``listing`` is the folder's ``list_wsis`` when already made, so that one listing serves every tissue."""
    listing = list_wsis(folder) if listing is None else listing
    found: dict[str, Path] = {}
    for slide_id in sorted(slide_ids & listing.keys()):
        paths = sorted(listing[slide_id])
        if len(paths) > 1:
            raise ValueError(f"two slide files for {slide_id}: {paths[0].name} and {paths[1].name}")
        found[slide_id] = paths[0]
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
            counts[:, k] = cKDTree(members).query_ball_point(xy, r=radius_px, return_length=True, workers=WORKERS)
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
    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k must be from {K_MIN} to {K_MAX}")
    sd = features.std(axis=0)
    used = [j for j in range(features.shape[1]) if sd[j] > 0]
    if not used:
        raise ValueError("every feature is constant on this slide")
    z = (features[:, used] - features[:, used].mean(axis=0)) / sd[used]
    return kmeans_by_size(z, k), used


def kmeans_by_size(X: np.ndarray, k: int) -> np.ndarray:
    """``KMeans(k, n_init=20, random_state=0)`` labels of the rows of ``X`` (uint8), numbered by size, largest first."""
    from sklearn.cluster import KMeans

    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k must be from {K_MIN} to {K_MAX}")
    if len(X) < k:
        raise ValueError(f"{len(X)} rows cannot form {k} clusters")
    labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=KMEANS_SEED).fit_predict(X)
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    return rank[labels].astype(np.uint8)


def cluster_request(query: dict[str, list[str]]) -> tuple[int, bool, float]:
    """k, density on or off, and the ball's radius in µm from a cluster request's query; ``ValueError`` if invalid."""
    try:
        k = int(query.get("k", [""])[0])
        radius = float(query.get("radius", [str(c.DELTA_UM)])[0])
    except ValueError:
        raise ValueError("k must be a whole number and radius a number of µm") from None
    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"k must be from {K_MIN} to {K_MAX}")
    if not (math.isfinite(radius) and RADIUS_MIN_UM <= radius <= RADIUS_MAX_UM):
        raise ValueError(f"the radius must be from {RADIUS_MIN_UM:g} to {RADIUS_MAX_UM:g} µm")
    return k, query.get("density", ["1"])[0] == "1", radius


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


def load_pilot_centres() -> dict:
    """The chromatic pilot's sampled centres from its ``results/features.npz`` and ``results/pilot.csv``: per slide
    ID, the centres' slide rows, positions, level-1 features and level-2 scores (``TDA_KEY``)."""
    pilot = pd.read_csv(c.RESULTS / "pilot.csv")
    with np.load(c.RESULTS / "features.npz", allow_pickle=False) as f:
        index, rows, xy, l1, l2 = (f[key] for key in ("slide_index", "centre_row", "xy", "L1", TDA_KEY))
    out = {}
    for slide in pilot.itertuples(index=False):
        mine = index == slide.slide_index
        out[slide.slide_id] = {"rows": rows[mine], "xy": xy[mine], "L1": l1[mine], "L2": l2[mine]}
    if sum(len(v["rows"]) for v in out.values()) != len(index):
        raise ValueError("features.npz holds centres of slides that pilot.csv does not list")
    return out


def emeans_file(name: str) -> str:
    """The e-means study's label file of one feature set."""
    return f"labels_{name.replace('+', '_')}.npz"


def load_emeans(folder: Path) -> dict | None:
    """Per e-means set: its columns, Tibshirani's k̂, its cells, the cohort profile of every cluster at every k
    (clusters.json) and the k whose fit is worse than k-means++'s, as a log ratio of W_k (kmeans_check.json).
    None when the e-means study's results are not in ``folder``."""
    names = ("gap.json", "clusters.json", "kmeans_check.json", *(emeans_file(name) for name in EMEANS_SETS))
    if not all((folder / name).is_file() for name in names):
        return None
    gap_all = json.loads((folder / "gap.json").read_text())
    gap = gap_all["sets"]
    clusters = json.loads((folder / "clusters.json").read_text())
    check = json.loads((folder / "kmeans_check.json").read_text())["sets"]
    out = {}
    for name in EMEANS_SETS:
        ratios = dict(zip(check[name]["k"], check[name]["log_ratio_run_to_kmeans"], strict=True))
        out[name] = {
            "columns": gap[name]["features"]["columns"],
            "k_tibshirani": int(gap[name]["k_tibshirani"]),
            "n_cells": int(gap[name]["n_cells"]),
            "radius_um": float(gap_all["settings"]["radius_um"]),
            "profiles": clusters[name]["clusters"],
            "worse": {k: ratio for k, ratio in ratios.items() if ratio > WORSE},
        }
    return out


def unpack_rows(packed: np.ndarray, start: int, end: int) -> np.ndarray:
    """The labels of rows ``start`` to ``end - 1`` of two labels per byte, the even rows in the high half (the
    e-means study's ``cluster.pack``)."""
    if not 0 <= start <= end <= 2 * len(packed):
        raise ValueError(f"rows {start}:{end} of {2 * len(packed)} packed labels")
    chunk = packed[start // 2 : (end + 1) // 2]
    labels = np.empty(2 * len(chunk), np.uint8)
    labels[0::2], labels[1::2] = chunk >> 4, chunk & 15
    return labels[start % 2 : start % 2 + end - start]


def emeans_request(query: dict[str, list[str]]) -> tuple[str, int]:
    """The feature set and k of an e-means request's query; ``ValueError`` if invalid."""
    name = query.get("set", [""])[0]
    if name not in EMEANS_SETS:
        raise ValueError(f"set must be one of {', '.join(EMEANS_SETS)}")
    return name, cluster_request({"k": query.get("k", [""])})[0]


def emeans_clusters(profiles: list[dict], labels: np.ndarray, k: int) -> tuple[np.ndarray, list[dict]]:
    """The saved labels recoded by cohort cluster size, largest first (the viewer's other clusters' order, and
    the palette's), and per code the cluster's saved label, its cells on this slide and its cohort profile."""
    if len(profiles) != k:
        raise ValueError(f"{len(profiles)} cluster profiles for k = {k}")
    if labels.size and int(labels.max()) >= k:
        raise RuntimeError(f"a saved label is {int(labels.max())}, not below k = {k}")
    order = sorted(range(k), key=lambda j: -profiles[j]["n"])
    rank = np.empty(k, np.uint8)
    rank[order] = np.arange(k)
    on_slide = np.bincount(labels, minlength=k)
    entries = []
    for j in order:
        p = profiles[j]
        if list(p["class_fractions"]) != list(c.CLASSES):
            raise ValueError(f"cluster {j}'s classes are {list(p['class_fractions'])}, not {list(c.CLASSES)}")
        entries.append(
            {
                "label": j,
                "n": int(on_slide[j]),
                "cohort_n": int(p["n"]),
                "cohort_share": float(p["share"]),
                "cells_per_ball": float(p["cells_per_ball"]),
                "fractions": [float(p["class_fractions"][name]) for name in c.CLASSES],
                "kept_fractions": [float(v) for v in p["kept_fractions"]],
            }
        )
    return rank[labels], entries


def remember(cache: OrderedDict, key, value, size: int):
    """Put ``value`` in the LRU ``cache`` under ``key``, dropping the oldest entries beyond ``size``."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > size:
        cache.popitem(last=False)
    return value


class Cohort:
    """The slide table, the cell types and small caches: cells, slide files, ball features and clusterings.

    Of the liver cohort, or with ``tissue`` of one GTEx tissue folder (``find_tissues``), which has none of the
    liver cohort's precomputed layers: no 10-NN density, pathology groups, chromatic TDA or e-means clusters. Its
    colocalization scale is its own fit, ``<colocalization_dir>/<slug>.json`` (``fit_colocalization.py --tissue``),
    if there is one of this object (``tissue_colocalization``); else that layer is off, and
    ``colocalization_note`` says why when a file is there. Its Voronoi densities are binned at the slide's own
    deciles, and its cell types outside the four classes count as "other".
    """

    def __init__(
        self,
        wsi_dir: Path | None,
        tissue: Path | None = None,
        wsi_listing: dict[str, list[Path]] | None = None,
        colocalization_dir: Path | None = None,
    ) -> None:
        self.liver = tissue is None
        self.h5ad = c.H5AD if tissue is None else tissue_h5ad(tissue)
        self.slug = None if tissue is None else tissue.name
        self._lock = threading.Lock()
        self._cells: OrderedDict[str, dict] = OrderedDict()  # slide ID -> xy, types, blob, extent
        self._zooms: OrderedDict[str, tuple] = OrderedDict()  # slide ID -> (OpenSlide, DeepZoomGenerator)
        self._features: OrderedDict[tuple, np.ndarray] = OrderedDict()  # (slide ID, µm/px, radius) -> features
        self._clusters: OrderedDict[tuple, dict] = OrderedDict()  # (slide ID, k, density, radius) -> clustering
        self._pilot: dict | None = None  # slide ID -> the pilot's saved centres, loaded on the first TDA request
        self._packed: OrderedDict[tuple, np.ndarray] = OrderedDict()  # (set, k) -> packed labels of every cell
        self._density: OrderedDict[str, bytes] = OrderedDict()  # slide ID -> its cells' densities and bins
        self._voronoi: OrderedDict[str, bytes] = OrderedDict()  # slide ID -> its cells' Voronoi densities and codes
        self._neighbourhood: OrderedDict[tuple, bytes] = OrderedDict()  # (slide ID, k, label set) -> both values
        self._colocalization: OrderedDict[tuple, dict] = OrderedDict()  # (slide ID, k, label set) -> excess
        self.colocalization_note: str | None = None  # why a tissue's fit file is not used
        if tissue is None:
            self._load_liver(wsi_dir)
        else:
            self._load_tissue(tissue, wsi_dir, wsi_listing)
            if colocalization_dir is not None:
                self.colocalization_fit, self.colocalization_note = tissue_colocalization(
                    colocalization_dir / f"{self.slug}.json", self.types, len(self.slides), int(self.slides["end"].max())
                )
        self.n_labels = {"types": len(self.types), "classes": len(c.CLASSES)}

    def _load_tissue(self, folder: Path, wsi_dir: Path | None, wsi_listing: dict[str, list[Path]] | None) -> None:
        """The slides, their rows and Portal fields from the folder's slides.csv, checked against its object."""
        names, n_rows = h5ad_slides(self.h5ad)
        table = load_tissue_slides(folder / "slides.csv", names, n_rows).sort_values("slide_id", ignore_index=True)
        tissues = sorted(set(table["tissue"]))
        if len(tissues) != 1:
            raise ValueError(f"{folder / 'slides.csv'} names {len(tissues)} tissues, not one: {tissues[:3]}")
        self.name = tissues[0]
        self.slides = table.assign(group="", pilot=False)[["slide_id", "group", "start", "end", "n_cells", "pilot"]]
        self.by_id = {row.slide_id: row for row in self.slides.itertuples(index=False)}
        self.types = cell_types(self.h5ad)
        if len(self.types) > MAX_TYPES:
            raise ValueError(f"{self.h5ad.name} has {len(self.types)} cell types; the page colours {MAX_TYPES} at most")
        classes, self.unmapped = class_codes_or_other(self.types)
        self.type_class = classes.astype(np.int64)
        self.wsis = find_wsis(wsi_dir, set(self.by_id), wsi_listing) if wsi_dir else {}
        rows = list(table.itertuples(index=False))
        # As the liver cohort's pathology notes and donors, with the Portal's Hidden flag and the donor's ID; the
        # quality flag is the pathology study's, so empty.
        self.notes = {
            r.slide_id: {"categories": r.categories, "notes": r.notes, "quality_flag": "", "hidden": bool(r.hidden)}
            for r in rows
        }
        self.donors = {
            r.slide_id: {"sex": r.sex, "age_bracket": r.age_bracket, "hardy_scale": r.hardy_scale, "subject_id": r.subject_id}
            for r in rows
        }
        self.emeans_sets = None  # the liver cohort's
        self.colocalization_fit = None  # the tissue's own, if any, set after loading (the liver cohort's says nothing of it)
        self.rho = self.deciles = self.curve = self.k_nn = None  # the pathology study's 10-NN density is the liver's
        self.density_um_per_px = c.PX_UM  # the Voronoi density's scale: GTEx's assumed µm per px, as for liver

    def _load_liver(self, wsi_dir: Path | None) -> None:
        """The liver cohort: its slide table, the studies' precomputed layers and its donors."""
        self.name = "Liver"
        self.unmapped: list[str] = []  # every liver type has its class (common.class_codes refuses any other)
        self.slides = load_slide_table()
        self.by_id = {row.slide_id: row for row in self.slides.itertuples(index=False)}
        self.types = cell_types()
        self.type_class = np.array([int(k) for k in c.class_codes(self.types)], dtype=np.int64)
        self.wsis = find_wsis(wsi_dir, set(self.by_id)) if wsi_dir else {}
        self.emeans_sets = load_emeans(EMEANS)  # None without the e-means study's results
        self.colocalization_fit = load_colocalization(COLOCALIZATION)  # None without fit_colocalization.py's
        n_cohort = int(self.slides["end"].max())
        self.rho = npz_memmap(c.SIB / "density.npz", "rho")  # every cohort cell's 10-NN density, cells per mm²
        with np.load(c.SIB / "density.npz", allow_pickle=False) as density:
            self.deciles = np.asarray(density["deciles"], dtype=np.float64)
            self.k_nn, self.density_um_per_px = int(density["k_nn"]), float(density["pixel_um"])
            self.curve = threshold_curve(density["curve"], density["curve_levels"], self.deciles)
        self.notes = load_notes(c.SIB / "pathology.csv")
        if set(self.notes) != set(self.by_id):
            raise ValueError("pathology.csv's notes and the slide table hold different slides")
        donors = load_donors(c.H5AD.parent / "liver_cohort.csv")
        missing = sorted(set(self.by_id) - set(donors))
        if missing:
            raise ValueError(f"liver_cohort.csv has no donor for {len(missing)} slides, {missing[0]} first")
        self.donors = {slide_id: donors[slide_id] for slide_id in self.by_id}
        if self.rho.shape != (n_cohort,) or self.rho.dtype != np.float64:
            raise ValueError(f"density.npz's rho is {self.rho.dtype} {self.rho.shape}, not {n_cohort:,} densities")
        if len(self.deciles) != DENSITY_BINS or not np.all(np.diff(self.deciles) > 0):
            raise ValueError(f"density.npz's deciles are not {DENSITY_BINS} rising thresholds: {self.deciles}")
        for name, s in (self.emeans_sets or {}).items():
            if s["n_cells"] != n_cohort:
                raise ValueError(f"the e-means set {name} has {s['n_cells']:,} cells, the cohort {n_cohort:,}")

    def forget(self) -> None:
        """Let go of the slides held in memory: their cells and every value and clustering computed from them (the
        slide files stay open). They are read and computed again when asked for."""
        with self._lock:
            for cache in (self._cells, self._features, self._clusters, self._packed, self._density, self._voronoi,
                          self._neighbourhood, self._colocalization):
                cache.clear()

    def meta(self) -> dict:
        """What the page needs to start. A tissue folder has no pathology groups, its ``density`` holds only what the
        Voronoi density needs (no 10-NN deciles, curve or k), and its ``tissue`` gives its slug, its Portal name and
        its cell types outside the four classes (``unmapped``, counted as "other"); the liver cohort's is None."""
        counts = self.slides["group"].value_counts()
        density = (
            {
                "deciles": self.deciles.tolist(),
                "k_nn": self.k_nn,
                "um_per_px": self.density_um_per_px,
                "curve": self.curve.tolist(),
                "voronoi_none": VORONOI_NONE,
            }
            if self.liver
            else {"um_per_px": self.density_um_per_px, "voronoi_none": VORONOI_NONE}
        )
        return {
            "groups": [{"name": g, "n_slides": int(counts.get(g, 0))} for g in GROUPS] if self.liver else [],
            "types": self.types,
            "type_class": self.type_class.tolist(),
            "classes": list(c.CLASSES),
            "um_per_px": c.PX_UM,
            "cluster": {
                "k_min": K_MIN,
                "k_max": K_MAX,
                "radius_um": c.DELTA_UM,
                "radius_min_um": RADIUS_MIN_UM,
                "radius_max_um": RADIUS_MAX_UM,
                "n_init": N_INIT,
                "tda_radius_um": c.DELTA_UM,
                "tda_labellings": c.R,
                "tda_centres": c.N_CENTRES,
            },
            "density": density,
            "neighbourhood": {
                "k": NEIGHBOURS,
                "k_min": NEIGHBOURS_MIN,
                "k_max": NEIGHBOURS_MAX,
                "n_labels": {"types": len(self.types), "classes": len(c.CLASSES)},
                "excess_edges": EXCESS_EDGES.tolist(),
                "kl_step": KL_STEP,
            },
            "api": API,
            "colocalization": None
            if self.colocalization_fit is None
            else {
                "k_min": min(self.colocalization_fit["k"]),
                "k_max": max(self.colocalization_fit["k"]),
                "n_slides": self.colocalization_fit["n_slides"],
                "none": COLOCALIZATION_NONE,
                "labels": {by: self.colocalization_fit[by]["labels"] for by in LABEL_SETS},
            },
            "emeans": None
            if self.emeans_sets is None
            else [
                {"name": name, "columns": len(s["columns"]), "k_tibshirani": s["k_tibshirani"]}
                for name, s in self.emeans_sets.items()
            ],
            "tissue": None
            if self.liver
            else {"slug": self.slug, "name": self.name, "unmapped": self.unmapped},
            "slides": [
                {
                    "slide_id": r.slide_id,
                    "group": r.group,
                    "n_cells": int(r.n_cells),
                    "pilot": bool(r.pilot),
                    "wsi": r.slide_id in self.wsis,
                    **self.notes[r.slide_id],
                    **self.donors[r.slide_id],
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
        xy, types = read_cells(slide_id, int(row.start), int(row.end), len(self.types), self.h5ad)
        entry = {"xy": xy, "types": types, "blob": cells_blob(xy, types), "extent": cloud_extent(xy)}
        with self._lock:
            return remember(self._cells, slide_id, entry, CACHE)

    def um_per_px(self, slide_id: str) -> tuple[float, str]:
        """µm per level-0 px of the slide, and where the value comes from."""
        if slide_id in self.wsis:
            return slide_um_per_px(dict(self.opened(slide_id)[0].properties))
        return slide_um_per_px({})

    def features(self, slide_id: str, um_per_px: float, radius_um: float) -> np.ndarray:
        """The level-1 features of every cell's ball of ``radius_um`` (log10 cells, four class fractions)."""
        key = (slide_id, um_per_px, radius_um)
        with self._lock:
            if key in self._features:
                self._features.move_to_end(key)
                return self._features[key]
        cells = self.cells(slide_id)
        counts = ball_class_counts(cells["xy"], self.type_class[cells["types"]], radius_um / um_per_px)
        with self._lock:
            return remember(self._features, key, level1(counts), 2)

    def cluster(self, slide_id: str, k: int, density: bool, radius_um: float = c.DELTA_UM) -> dict:
        """Composition clusters of every cell of the slide, with each cluster's profile, as JSON-ready data."""
        key = (slide_id, k, density, radius_um)
        with self._lock:
            if key in self._clusters:
                self._clusters.move_to_end(key)
                return self._clusters[key]
        started = time.perf_counter()
        um_per_px, source = self.um_per_px(slide_id)
        features = self.features(slide_id, um_per_px, radius_um)
        columns = [0, 1, 2, 3, 4] if density else [1, 2, 3, 4]
        labels, used = composition_clusters(features[:, columns], k)
        result = {
            "slide_id": slide_id,
            "k": k,
            "density": density,
            "radius_um": radius_um,
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

    def tda(self, slide_id: str, k: int) -> dict:
        """k-means of a pilot slide's saved centres on their level-2 scores (domains.level2's features and k-means),
        with each cluster's mean ball composition, as JSON-ready data."""
        if not self.liver:
            raise KeyError("no chromatic TDA outside the liver cohort: the pilot's centres are liver slides")
        key = ("tda", slide_id, k)
        with self._lock:
            if key in self._clusters:
                self._clusters.move_to_end(key)
                return self._clusters[key]
            if self._pilot is None:
                self._pilot = load_pilot_centres()
            centres = self._pilot.get(slide_id)
        if centres is None:
            raise KeyError(f"{slide_id} is not a pilot slide, so it has no saved chromatic features")
        started = time.perf_counter()
        xy = self.cells(slide_id)["xy"]
        rows = centres["rows"]
        if rows.min() < 0 or rows.max() >= len(xy) or not np.array_equal(xy[rows], centres["xy"]):
            raise RuntimeError(f"{slide_id}: the saved centres are not at the positions of their slide rows")
        labels = kmeans_by_size(centres["L2"], k)
        result = {
            "slide_id": slide_id,
            "k": k,
            "tda": True,
            "n_centres": len(rows),
            "radius_um": c.DELTA_UM,
            "n_labellings": c.R,
            "features": f"{centres['L2'].shape[1]} MFA scores ({TDA_KEY})",
            "n_init": N_INIT,
            "seed": KMEANS_SEED,
            "clusters": cluster_profiles(centres["L1"], labels, k),
            "centre_rows": base64.b64encode(rows.astype("<u4").tobytes()).decode("ascii"),
            "labels": base64.b64encode(labels.tobytes()).decode("ascii"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        with self._lock:
            return remember(self._clusters, key, result, 8)

    def density(self, slide_id: str) -> bytes:
        """Every cell's density and density bin (``density_codes``) on the slide, as ``density_blob``."""
        if self.rho is None:
            raise KeyError("no 10-NN density outside the liver cohort: the pathology study computed it for liver")
        with self._lock:
            if slide_id in self._density:
                self._density.move_to_end(slide_id)
                return self._density[slide_id]
        row = self.by_id[slide_id]
        rho = np.asarray(self.rho[int(row.start) : int(row.end)])
        if not np.all(np.isfinite(rho) & (rho > 0)):
            raise RuntimeError(f"{slide_id}: a cell has no positive, finite density")
        blob = density_blob(rho, density_codes(rho, self.deciles))
        with self._lock:
            return remember(self._density, slide_id, blob, CACHE)

    def voronoi(self, slide_id: str) -> bytes:
        """Every cell's Voronoi density (``voronoi_density``, at the 10-NN density's µm per px; NaN without one)
        and code (``voronoi_codes``: at the cohort's 10-NN deciles, or in a tissue folder at the slide's own deciles)
        on the slide, as ``density_blob``; computed on first use."""
        with self._lock:
            if slide_id in self._voronoi:
                self._voronoi.move_to_end(slide_id)
                return self._voronoi[slide_id]
        rho = voronoi_density(self.cells(slide_id)["xy"], self.density_um_per_px)
        blob = density_blob(rho, voronoi_codes(rho, self.deciles if self.liver else slide_deciles(rho)))
        with self._lock:
            return remember(self._voronoi, slide_id, blob, CACHE)

    def colocalization(self, slide_id: str, k: int, by: str) -> dict:
        """The slide's colocalization excess among every cell's k nearest other cells, by HistoPLUS type or by
        class: ``LocalColocalizationExtractor`` on the cohort's pair scale for this k (fit_colocalization.py; a
        tissue's own fit over its slides), shrunk toward the slide's pooled excess. Keys: ``codes`` (each cell's
        label), ``shrunk`` and ``raw`` ((N, K) float32), ``summary`` (``summarize_cloud``) and ``gain`` ((K, K) bool,
        the pair scale)."""
        fit = self.colocalization_fit
        if fit is None and self.liver:
            raise KeyError("colocalization: results/colocalization_scale.json is missing (fit_colocalization.py)")
        if fit is None:
            raise KeyError(
                f"no colocalization excess for {self.slug}: "
                + (self.colocalization_note or "no fit of its own (fit_colocalization.py --tissue, served with "
                   "--colocalization-dir); the liver cohort's pair scale says nothing of another tissue")
            )
        if k not in fit["k"]:
            raise ValueError(f"the cohort fit has k from {min(fit['k'])} to {max(fit['k'])}")
        key = (slide_id, k, by)
        with self._lock:
            if key in self._colocalization:
                self._colocalization.move_to_end(key)
                return self._colocalization[key]
        cells = self.cells(slide_id)
        codes = cells["types"].astype(np.int64)
        if by == "classes":
            codes = self.type_class[codes]
        n_labels = self.n_labels[by]
        j = fit["k"].index(k)
        extractor = LocalColocalizationExtractor.fitted(
            list(range(n_labels)), fit[by]["pair"][j], fit[by]["target"][j], n_neighbors=k, dtype=np.float32
        )
        result = extractor.transform([(cells["xy"], codes)])[0]
        entry = {
            "codes": codes,
            "shrunk": result.Y_shrunk,
            "raw": result.Y_raw,
            "summary": extractor.summarize_cloud(result, codes),
            "gain": extractor.gain_mask_[:n_labels],
        }
        with self._lock:
            return remember(self._colocalization, key, entry, COLOCALIZATION_CACHE)

    def colocalization_values(self, slide_id: str, k: int, by: str, target: int | None, score: str) -> bytes:
        """Each cell's excess of one neighbour type (``target``; None: the cell's own) as ``density_blob``, with
        ``colocalization_codes``."""
        entry = self.colocalization(slide_id, k, by)
        matrix = entry[score]
        codes = entry["codes"]
        values = matrix[np.arange(len(codes)), codes] if target is None else matrix[:, target]
        return density_blob(values, colocalization_codes(values))

    def colocalization_summary(self, slide_id: str, k: int, by: str) -> dict:
        """The slide's pooled excess by (focal, neighbour) type for the page's matrix, NaN as null."""
        entry = self.colocalization(slide_id, k, by)
        summary = entry["summary"]

        def rows(a: np.ndarray) -> list:
            return [[round(float(x), 4) if np.isfinite(x) else None for x in row] for row in a]

        return {
            "k": k,
            "by": by,
            "labels": self.colocalization_fit[by]["labels"],
            "pooled_excess": rows(summary["pooled_excess"]),
            "pooled_variance": rows(summary["pooled_variance"]),
            "median": rows(summary["quantiles"][1]),
            "n_cells": summary["n_cells"].tolist(),
            "gain": entry["gain"].tolist(),
        }

    def neighbourhood(self, slide_id: str, k: int, by: str) -> bytes:
        """Every cell's homotypic fraction and local entropy among its k nearest other cells, by HistoPLUS type or
        by class (``by``), with their bins, as ``neighbourhood_blob``; computed on first use."""
        key = (slide_id, k, by)
        with self._lock:
            if key in self._neighbourhood:
                self._neighbourhood.move_to_end(key)
                return self._neighbourhood[key]
        cells = self.cells(slide_id)
        labels = cells["types"].astype(np.int64)
        n_labels = len(self.types)
        if by == "classes":
            labels, n_labels = self.type_class[labels], len(c.CLASSES)
        nbrs = nearest_others(cells["xy"], k)
        same = homotypic_counts(labels, nbrs)
        entropy = local_entropy(labels, nbrs, n_labels)
        shares = slide_shares(labels, n_labels)
        excess = homotypic_excess(same, k, labels, shares)
        kl = kl_divergence(labels, nbrs, shares)
        blob = neighbourhood_blob(
            [same / k, entropy, excess, kl],
            [homotypic_codes(same, k), unit_codes(entropy), excess_codes(excess), kl_codes(kl)],
        )
        with self._lock:
            return remember(self._neighbourhood, key, blob, CACHE)

    def packed(self, name: str, k: int) -> np.ndarray:
        """The e-means study's packed labels of every cohort cell for one set and k."""
        key = (name, k)
        with self._lock:
            if key in self._packed:
                self._packed.move_to_end(key)
                return self._packed[key]
        with np.load(EMEANS / emeans_file(name), allow_pickle=False) as z:
            n, packed = int(z["n_cells"]), np.asarray(z[f"k{k}"])
        if n != self.emeans_sets[name]["n_cells"] or len(packed) != (n + 1) // 2:
            raise RuntimeError(f"{emeans_file(name)}: {len(packed):,} bytes of labels for {n:,} cells")
        with self._lock:
            return remember(self._packed, key, packed, PACKED)

    def emeans(self, slide_id: str, name: str, k: int) -> dict:
        """Every cell of the slide coloured by its saved e-means cluster, with each cluster's cohort profile, as
        JSON-ready data."""
        if not self.liver:
            raise KeyError("no e-means clusters outside the liver cohort: they were fitted to the liver cohort's cells")
        if self.emeans_sets is None:
            raise KeyError(f"the e-means study's results are not in {EMEANS}")
        key = ("emeans", slide_id, name, k)
        with self._lock:
            if key in self._clusters:
                self._clusters.move_to_end(key)
                return self._clusters[key]
        started = time.perf_counter()
        s = self.emeans_sets[name]
        row = self.by_id[slide_id]
        labels = unpack_rows(self.packed(name, k), int(row.start), int(row.end))
        codes, clusters = emeans_clusters(s["profiles"][str(k)], labels, k)
        result = {
            "slide_id": slide_id,
            "k": k,
            "emeans": True,
            "set": name,
            "features": s["columns"],
            "file": emeans_file(name),
            "radius_um": s["radius_um"],
            "k_tibshirani": s["k_tibshirani"],
            "worse_than_kmeans": s["worse"].get(k),
            "clusters": clusters,
            "labels": base64.b64encode(codes.tobytes()).decode("ascii"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        with self._lock:
            return remember(self._clusters, key, result, 8)

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


class Library:
    """What the server shows: the liver cohort alone, or with ``tissues`` every GTEx tissue folder ``find_tissues``
    finds there, by slug, each opened at the start. A folder that cannot be opened is listed with its error, so
    that the others still open; the first by slug that opens is the default. Only the ``WARM_TISSUES`` tissues
    used last keep their slides in memory, so that the server needs no more for many tissues than for a few.
    ``colocalization_dir``: where each tissue's own colocalization fit is, ``<slug>.json`` (``Cohort``)."""

    def __init__(
        self, wsi_dir: Path | None, tissues: Path | None = None, colocalization_dir: Path | None = None
    ) -> None:
        self.cohorts: dict[str | None, Cohort] = {}
        self.failed: dict[str, str] = {}  # slug -> why its folder cannot be opened
        self.skipped: dict[str, str] = {}  # folder name -> the file it lacks
        self._warm: OrderedDict[str | None, None] = OrderedDict()  # the tissues used last, the latest at the end
        self._lock = threading.Lock()
        if tissues is None:
            self.cohorts[None] = Cohort(wsi_dir)
            self.default = None
            return
        found, self.skipped = find_tissues(tissues)
        listing = list_wsis(wsi_dir) if wsi_dir else None  # once for every tissue: the folder may hold every slide
        for folder in found:
            try:
                self.cohorts[folder.name] = Cohort(wsi_dir, folder, listing, colocalization_dir)
            except Exception as err:  # an unreadable object or a slides.csv that does not fit it, among others
                self.failed[folder.name] = f"{type(err).__name__}: {err}"
        if not self.cohorts:
            raise ValueError(f"none of the tissue folders in {tissues} can be opened: {self.failed}")
        self.default = min(self.cohorts)

    def get(self, slug: str | None, warm: bool = True) -> Cohort:
        """The cohort of a tissue slug, or the default one for None; ``KeyError`` for a slug not served. ``warm``:
        the request uses the tissue's slides (not only a tile of a slide file), so they stay in memory, and the
        slides of the tissue used longest ago beyond ``WARM_TISSUES`` are let go (requests for it still running
        may put theirs back)."""
        if slug in self.failed:
            raise RuntimeError(f"tissue {slug} could not be opened: {self.failed[slug]}")
        if slug is not None and slug not in self.cohorts:
            if self.default is None:
                raise KeyError(f"tissue {slug}: this server shows the liver cohort (no --tissues)")
            raise KeyError(f"tissue {slug}: the tissue folders here are {', '.join(row['slug'] for row in self.listing())}")
        key = self.default if slug is None else slug
        if warm:
            with self._lock:  # one tissue at a time, so that none is let go while it is used again
                self._warm[key] = None
                self._warm.move_to_end(key)
                while len(self._warm) > WARM_TISSUES:
                    self.cohorts[self._warm.popitem(last=False)[0]].forget()
        return self.cohorts[key]

    def listing(self) -> list[dict]:
        """Every tissue folder found, by slug, with its Portal name, its slides and, if it cannot be opened, why."""
        rows = [
            {"slug": slug, "name": cohort.name, "n_slides": len(cohort.slides), "error": None}
            for slug, cohort in self.cohorts.items()
            if slug is not None
        ]
        rows += [{"slug": slug, "name": slug, "n_slides": None, "error": error} for slug, error in self.failed.items()]
        return sorted(rows, key=lambda row: row["slug"])


def make_handler(library: Library) -> type[BaseHTTPRequestHandler]:
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
                return
            if len(parts) == 2 and parts[0] == "static":
                path = (STATIC / parts[1]).resolve()
                if path.parent != STATIC.resolve() or path.suffix not in STATIC_TYPES or not path.is_file():
                    raise KeyError(parts[1])
                self.send(HTTPStatus.OK, path.read_bytes(), STATIC_TYPES[path.suffix])
                return
            # The liver cohort, or a tissue folder by slug; a tile of a slide file uses none of its slides' values.
            cohort = library.get(query.get("tissue", [None])[0], warm=parts[:2] != ["api", "tile"])
            if parts == ["api", "meta"]:
                meta = {**cohort.meta(), "tissues": library.listing()}
                self.send(HTTPStatus.OK, json.dumps(meta).encode(), "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "slide"] and parts[2] in cohort.by_id:
                self.send(HTTPStatus.OK, json.dumps(cohort.info(parts[2])).encode(), "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "cells"] and parts[2] in cohort.by_id:
                self.send(HTTPStatus.OK, cohort.cells(parts[2])["blob"], "application/octet-stream")
            elif len(parts) == 3 and parts[:2] == ["api", "cluster"] and parts[2] in cohort.by_id:
                k, density, radius = cluster_request(query)
                result = cohort.cluster(parts[2], k, density, radius)
                self.send(HTTPStatus.OK, json.dumps(result).encode(), "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "tda"] and parts[2] in cohort.by_id:
                k = cluster_request({"k": query.get("k", [""])})[0]
                self.send(HTTPStatus.OK, json.dumps(cohort.tda(parts[2], k)).encode(), "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "density"] and parts[2] in cohort.by_id:
                self.send(HTTPStatus.OK, cohort.density(parts[2]), "application/octet-stream")
            elif len(parts) == 3 and parts[:2] == ["api", "voronoi"] and parts[2] in cohort.by_id:
                self.send(HTTPStatus.OK, cohort.voronoi(parts[2]), "application/octet-stream")
            elif len(parts) == 3 and parts[:2] == ["api", "colocalization"] and parts[2] in cohort.by_id:
                k, by, target, score = colocalization_request(query, cohort.n_labels)
                blob = cohort.colocalization_values(parts[2], k, by, target, score)
                self.send(HTTPStatus.OK, blob, "application/octet-stream")
            elif len(parts) == 4 and parts[:2] == ["api", "colocalization"] and parts[3] == "summary" and (
                parts[2] in cohort.by_id
            ):
                k, by = neighbourhood_request(query)
                summary = json.dumps(cohort.colocalization_summary(parts[2], k, by)).encode()
                self.send(HTTPStatus.OK, summary, "application/json")
            elif len(parts) == 3 and parts[:2] == ["api", "neighbourhood"] and parts[2] in cohort.by_id:
                k, by = neighbourhood_request(query)
                self.send(HTTPStatus.OK, cohort.neighbourhood(parts[2], k, by), "application/octet-stream")
            elif len(parts) == 3 and parts[:2] == ["api", "emeans"] and parts[2] in cohort.by_id:
                name, k = emeans_request(query)
                self.send(HTTPStatus.OK, json.dumps(cohort.emeans(parts[2], name, k)).encode(), "application/json")
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
    parser.add_argument(
        "--tissues",
        type=Path,
        help="a GTEx tissue folder (<slug>/ holding gtex_<slug>_cells.h5ad and slides.csv), or a folder of them, the\n"
        "page choosing among them: shown instead of the liver cohort, without its precomputed layers",
    )
    parser.add_argument(
        "--colocalization-dir",
        type=Path,
        help="with --tissues: a folder of each tissue's own colocalization fit, <slug>.json (fit_colocalization.py\n"
        "--tissue); a tissue without one, or with one of another object, has that colouring off",
    )
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.tissues is not None:
        args.tissues = Path(os.path.abspath(args.tissues.expanduser()))  # a folder's own name is its slug, so no "."
        if not args.tissues.is_dir():
            parser.error(f"--tissues {args.tissues} is not a folder")
        if not find_tissues(args.tissues)[0]:
            parser.error(f"--tissues {args.tissues} is not a tissue folder (<slug>/ holding gtex_<slug>_cells.h5ad and "
                         "slides.csv) and holds none")
    if args.colocalization_dir is not None:
        if args.tissues is None:
            parser.error("--colocalization-dir goes with --tissues (the liver cohort's fit is results/colocalization_scale.json)")
        args.colocalization_dir = args.colocalization_dir.expanduser()
        if not args.colocalization_dir.is_dir():
            parser.error(f"--colocalization-dir {args.colocalization_dir} is not a folder")
    if args.wsi_dir is not None:
        args.wsi_dir = args.wsi_dir.expanduser()
        if not args.wsi_dir.is_dir():
            parser.error(f"--wsi-dir {args.wsi_dir} is not a folder")
        try:
            import openslide  # noqa: F401
        except ImportError:
            parser.error("--wsi-dir needs openslide-python and openslide-bin (see README.md)")
    return args


def print_tissues(library: Library, wsi_dir: Path | None, colocalization_dir: Path | None = None) -> None:
    """One line per tissue folder: its slides and cell types, those outside the four classes, its slide files, with
    ``colocalization_dir`` whether it has its colocalization fit; or why it cannot be opened. Then the folders that
    lack one of the two files."""
    for row in library.listing():
        if row["error"] is not None:
            print(f"{row['slug']}: cannot be opened: {row['error']}", flush=True)
            continue
        cohort = library.get(row["slug"], warm=False)
        line = f"{row['slug']}: {cohort.name}, {len(cohort.slides)} slides, {len(cohort.types)} cell types"
        if cohort.unmapped:
            line += f" ({', '.join(cohort.unmapped)} outside the four classes: counted as other)"
        if wsi_dir is not None:
            line += f", {len(cohort.wsis)} with a slide file"
        if colocalization_dir is not None:
            if cohort.colocalization_fit is not None:
                line += ", colocalization fitted over its slides"
            else:
                why = cohort.colocalization_note or f"no {row['slug']}.json"
                line += f", no colocalization ({why})"
        print(line, flush=True)
    for name, reason in library.skipped.items():
        print(f"{name}: skipped, {reason}", flush=True)


def limit_malloc_arenas(n: int = 2) -> bool:
    """Let glibc's malloc keep at most ``n`` arenas (``mallopt(M_ARENA_MAX)``), as MALLOC_ARENA_MAX does. By default
    each of the server's threads may get its own, reserving up to 64 MB of address space each, which exhausts a
    capped address space (ulimit -v; 8 GiB on CeMM's login nodes) long before the memory is used: requests then fail
    with MemoryError or "can't start new thread". False where there is no glibc."""
    try:
        return bool(ctypes.CDLL("libc.so.6").mallopt(-8, int(n)))  # -8: M_ARENA_MAX
    except (OSError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if "MALLOC_ARENA_MAX" not in os.environ:  # a value set by hand wins
        limit_malloc_arenas()
    try:
        library = Library(args.wsi_dir, args.tissues, args.colocalization_dir)
    except ValueError as err:
        if args.tissues is None:
            raise
        raise SystemExit(str(err)) from None
    if args.tissues is None:
        cohort = library.get(None)
        print(f"{len(cohort.slides)} slides, {len(cohort.types)} cell types", flush=True)
        if cohort.colocalization_fit is None:
            print(f"no colocalization: {COLOCALIZATION} is missing (fit_colocalization.py writes it)", flush=True)
        if cohort.emeans_sets is None:
            print(f"no e-means clusters: the e-means study's results are not in {EMEANS}", flush=True)
        if args.wsi_dir is not None:
            print(f"{len(cohort.wsis)} of them with a slide file in {args.wsi_dir}", flush=True)
    else:
        print_tissues(library, args.wsi_dir, args.colocalization_dir)
        print(f"opens {library.default} first; /?tissue=<slug> opens another, as the page's tissue menu does", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(library))
    print(f"open http://{args.host}:{args.port}/  (Ctrl-C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
