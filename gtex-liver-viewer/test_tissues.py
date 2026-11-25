"""Tests of the viewer's GTEx tissue folders (``serve.py --tissues``): finding the folders, reading slides.csv, the
class map's fallback for cell types outside the four classes, the Voronoi density's bins at the slide's own deciles,
a tissue's layers (the liver-only ones off) and the server's routing by tissue. The tissue objects are written here
with h5py in the encoding anndata gives the liver object (this venv has no anndata)."""

import csv
import json
import struct
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
import pytest

import serve

c = serve.c

ARRAY = {"encoding-type": "array", "encoding-version": "0.2.0"}
STRINGS = {"encoding-type": "string-array", "encoding-version": "0.2.0"}
PORTAL = (
    "Tissue Sample ID", "Tissue", "Subject ID", "Sex", "Age Bracket", "Hardy Scale", "Pathology Categories",
    "Pathology Notes", "Hidden",
)
# A tissue's cell types: the sorted union of its class names, one of them outside the four classes.
CLASS_NAMES = ["Adipocyte", "Epithelial", "Fibroblasts", "Lymphocytes"]
# Slide ID, cells, and the Portal fields after the tissue: subject, sex, age, Hardy scale, categories, notes, hidden.
SLIDES = [
    ("GTEX-AAAA-0126", 60, ("GTEX-AAAA", "female", "60-69", "Slow death", "", "", "false")),
    ("GTEX-BBBB-0226", 45, ("GTEX-BBBB", "male", "20-29", "Ventilator Case", "clean_specimens, atrophy", "2 pieces, NA", "true")),
    ("GTEX-CCCC-0326", 80, ("GTEX-CCCC", "male", "50-59", "", "", "None", "false")),
]


def slug(name: str) -> str:
    """The layout's folder name of a Portal tissue: lower case, each run of other characters than letters and digits
    one underscore, none at either end."""
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def strings(group: h5py.Group, name: str, values) -> None:
    group.create_dataset(name, data=np.array(values, dtype=object), dtype=h5py.string_dtype(), compression="gzip")
    group[name].attrs.update(STRINGS)


def categorical(group: h5py.Group, name: str, values: list[str], categories: list[str] | None = None) -> None:
    """A column as anndata writes a categorical: a group of integer ``codes`` and string ``categories``. The codes
    are as wide as pandas makes them: int8 below 127 categories, int16 below 32,767, else int32 (so a tissue with
    up to 126 slides has int8 ``obs/slide_id/codes``, liver's 594 int16)."""
    categories = sorted(set(values)) if categories is None else categories
    lookup = {v: j for j, v in enumerate(categories)}
    column = group.create_group(name)
    column.attrs.update({"encoding-type": "categorical", "encoding-version": "0.2.0", "ordered": False})
    width = np.int8 if len(categories) < 127 else np.int16 if len(categories) < 32767 else np.int32
    codes = np.array([lookup[v] for v in values], dtype=width)
    column.create_dataset("codes", data=codes, compression="gzip").attrs.update(ARRAY)
    strings(column, "categories", categories)


def write_tissue(root: Path, tissue: str, slides=SLIDES, class_names=CLASS_NAMES, seed: int = 0) -> dict:
    """A tissue folder of the layout under ``root``: gtex_<slug>_cells.h5ad, slides.csv, provenance.json and a
    centroids/ cache (neither of which the viewer reads). Returns the folder and each slide's cells as written."""
    rng = np.random.default_rng(seed)
    folder = root / slug(tissue)
    (folder / "centroids").mkdir(parents=True)
    cells = {}
    for slide_id, n, _ in slides:
        xy = rng.uniform(1000, 3000, size=(n, 2)).astype(np.float32)
        cells[slide_id] = {"xy": xy, "types": [class_names[j] for j in rng.integers(0, len(class_names), n)]}
        np.savez(folder / "centroids" / f"{slide_id}.npz", xy=xy)
    n_cells = sum(n for _, n, _ in slides)
    per_cell = {column: [] for column in PORTAL[1:]}
    for slide_id, n, fields in slides:
        for column, value in zip(PORTAL[1:], (tissue, *fields), strict=True):
            per_cell[column] += [value] * n
    type_names = [name for slide_id, _, _ in slides for name in cells[slide_id]["types"]]
    with h5py.File(folder / f"gtex_{slug(tissue)}_cells.h5ad", "w") as f:
        f.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        obs = f.create_group("obs")
        columns = ["slide_id", "cell_type", "prob", *PORTAL[1:]]
        obs.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.2.0"})
        obs.attrs["column-order"] = np.array(columns, dtype=h5py.string_dtype())
        strings(obs, "_index", [f"cell-{i}" for i in range(n_cells)])
        categorical(obs, "slide_id", [s for s, n, _ in slides for _ in range(n)], sorted(s for s, _, _ in slides))
        categorical(obs, "cell_type", type_names, class_names)
        obs.create_dataset("prob", data=rng.uniform(0.5, 1, n_cells).astype(np.float32)).attrs.update(ARRAY)
        for column, values in per_cell.items():
            categorical(obs, column, values)
        obsm = f.create_group("obsm")
        obsm.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
        xy = np.vstack([cells[s]["xy"] for s, _, _ in slides])
        obsm.create_dataset("spatial", data=xy, compression="gzip").attrs.update(ARRAY)
        x = f.create_group("X")
        x.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": (n_cells, len(class_names))})
        x.create_dataset("data", data=np.ones(n_cells, np.float32))
        x.create_dataset("indices", data=np.array([class_names.index(t) for t in type_names], np.int32))
        x.create_dataset("indptr", data=np.arange(n_cells + 1, dtype=np.int32))
        var = f.create_group("var")
        var.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.2.0"})
        var.attrs["column-order"] = np.array([], dtype=h5py.string_dtype())
        strings(var, "_index", class_names)
        uns = f.create_group("uns")
        uns.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
        strings(uns, "class_names", class_names)
        strings(uns, "slide_ids", sorted(s for s, _, _ in slides))
        strings(uns, "metadata_columns", list(PORTAL[1:]))
        uns.create_dataset("n_slides", data=len(slides))
        uns.create_dataset("n_cells", data=n_cells)
        uns.create_dataset("created", data="2026-11-25T12:00:00", dtype=h5py.string_dtype())
    with open(folder / "slides.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*PORTAL, "n_cells", "row_start", "row_end"])
        start = 0
        for slide_id, n, fields in slides:
            writer.writerow([slide_id, tissue, *fields, n, start, start + n])
            start += n
    (folder / "provenance.json").write_text(json.dumps({"tissue": tissue}))
    return {"folder": folder, "cells": cells}


def type_codes(names: list[str], class_names=CLASS_NAMES) -> list[int]:
    return [class_names.index(name) for name in names]


def test_the_slug_rule_of_the_layout():
    assert slug("Liver") == "liver"
    assert slug("Skin - Sun Exposed (Lower leg)") == "skin_sun_exposed_lower_leg"
    assert slug("Adipose - Visceral (Omentum)") == "adipose_visceral_omentum"


def test_find_tissues_takes_one_tissue_folder_or_a_root_of_them(tmp_path):
    skin = write_tissue(tmp_path, "Skin - Sun Exposed (Lower leg)")["folder"]
    liver = write_tissue(tmp_path, "Liver")["folder"]
    (tmp_path / "tissues.csv").write_text("index,slug,tissue,n_slides,n_with_source\n0,liver,Liver,3,3\n")
    (tmp_path / "lung").mkdir()
    (tmp_path / "lung" / "slides.csv").write_text("")  # still being copied: no object yet
    (tmp_path / "notes").mkdir()  # not a tissue folder at all
    assert serve.find_tissues(tmp_path) == ([liver, skin], {"lung": "no gtex_lung_cells.h5ad"})
    assert serve.find_tissues(skin) == ([skin], {})
    assert serve.find_tissues(tmp_path / "notes") == ([], {})


def test_tissues_must_name_a_tissue_folder_or_a_folder_of_them(tmp_path):
    with pytest.raises(SystemExit):
        serve.parse_args(["--tissues", str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        serve.parse_args(["--tissues", str(tmp_path)])  # no tissue folder in it
    folder = write_tissue(tmp_path, "Liver")["folder"]
    assert serve.parse_args(["--tissues", str(tmp_path)]).tissues == tmp_path
    assert serve.parse_args(["--tissues", str(folder / ".." / "liver")]).tissues == folder
    assert serve.parse_args([]).tissues is None


def write_slides(path: Path, rows: list[list]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*PORTAL, "n_cells", "row_start", "row_end"])
        writer.writerows(rows)
    return path


def portal_row(slide_id: str, *rows: int) -> list:
    return [slide_id, "Lung", "GTEX-X", "female", "60-69", "NA", "", "None", "false", *rows]


def test_load_tissue_slides_keeps_the_portal_fields_as_text_and_reads_the_rows(tmp_path):
    path = write_slides(tmp_path / "slides.csv", [
        ["GTEX-A-1", "Lung", "GTEX-A", "female", "60-69", "NA", "", "None", "false", 3, 0, 3],
        ["GTEX-B-1", "Lung", "GTEX-B", "", "", "Slow death", "atrophy, emphysema", "2 pieces, NA", "TRUE", 2, 3, 5],
    ])
    table = serve.load_tissue_slides(path, ["GTEX-A-1", "GTEX-B-1", "GTEX-C-1"], 5)
    assert table["slide_id"].tolist() == ["GTEX-A-1", "GTEX-B-1"]
    assert table["start"].tolist() == [0, 3] and table["end"].tolist() == [3, 5] and table["n_cells"].tolist() == [3, 2]
    assert table["hardy_scale"].tolist() == ["NA", "Slow death"] and table["sex"].tolist() == ["female", ""]
    assert table["categories"].tolist() == ["", "atrophy, emphysema"]
    assert table["notes"].tolist() == ["None", "2 pieces, NA"]
    assert table["subject_id"].tolist() == ["GTEX-A", "GTEX-B"] and table["tissue"].tolist() == ["Lung", "Lung"]
    assert table["hidden"].tolist() == [False, True]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([portal_row("GTEX-A-1", 3, 1, 4), portal_row("GTEX-B-1", 2, 4, 6)], "without gaps"),  # starts at 1
        ([portal_row("GTEX-A-1", 3, 0, 3), portal_row("GTEX-B-1", 2, 4, 6)], "without gaps"),  # a gap
        ([portal_row("GTEX-A-1", 4, 0, 4), portal_row("GTEX-B-1", 3, 3, 6)], "without gaps"),  # an overlap
        ([portal_row("GTEX-A-1", 3, 0, 3), portal_row("GTEX-B-1", 2, 3, 5)], "without gaps"),  # short of the cells
        ([portal_row("GTEX-A-1", 3, 0, 3), portal_row("GTEX-B-1", 2, 3, 6)], "n_cells"),
        ([portal_row("GTEX-A-1", 0, 0, 0), portal_row("GTEX-B-1", 6, 0, 6)], "n_cells"),  # a slide without cells
        ([portal_row("GTEX-A-1", 3, 0, 3), portal_row("GTEX-A-1", 3, 3, 6)], "twice"),
        ([portal_row("GTEX-A-1", 3, 0, 3), portal_row("GTEX-Z-1", 3, 3, 6)], "does not name"),
        ([portal_row("GTEX-A-1", "3", "0", "three")], "whole numbers"),
        ([], "no slides"),
    ],
)
def test_load_tissue_slides_refuses_rows_that_do_not_tile_the_object(tmp_path, rows, message):
    path = write_slides(tmp_path / "slides.csv", rows)
    with pytest.raises(ValueError, match=message):
        serve.load_tissue_slides(path, ["GTEX-A-1", "GTEX-B-1"], 6)


def test_load_tissue_slides_names_the_columns_it_lacks(tmp_path):
    path = tmp_path / "slides.csv"
    path.write_text("Tissue Sample ID,Tissue,n_cells,row_start,row_end\nGTEX-A-1,Lung,3,0,3\n")
    with pytest.raises(ValueError, match="Subject ID"):
        serve.load_tissue_slides(path, ["GTEX-A-1"], 3)


def test_types_outside_the_four_classes_count_as_other_and_livers_map_is_unchanged():
    liver = ["Apoptotic Body", "Cancer cell", "Endothelial Cell", "Eosinophils", "Epithelial", "Fibroblasts",
             "Lymphocytes", "Macrophages", "Minor Stromal Cell", "Mitotic Figures", "Muscle Cell", "Neutrophils",
             "Plasmocytes", "Red blood cell"]
    codes, unmapped = serve.class_codes_or_other(liver)
    assert codes.tolist() == [3, 0, 1, 2, 0, 1, 2, 2, 1, 3, 1, 2, 2, 3] and unmapped == []
    assert codes.tolist() == c.class_codes(liver).tolist()
    codes, unmapped = serve.class_codes_or_other(["Adipocyte", "Epithelial", "Lymphocytes", "Schwann cell"])
    assert codes.tolist() == [3, 0, 2, 3] and unmapped == ["Adipocyte", "Schwann cell"]


def test_slide_deciles_are_the_slides_own_and_bin_a_tenth_of_its_cells_each():
    rho = np.concatenate([np.arange(1.0, 102.0), [np.nan, np.nan]])  # 1 to 101, and two cells without a value
    deciles = serve.slide_deciles(rho)
    np.testing.assert_allclose(deciles, [1, 11, 21, 31, 41, 51, 61, 71, 81, 91], rtol=0, atol=1e-12)
    codes = serve.voronoi_codes(rho, deciles)
    assert np.bincount(codes, minlength=serve.VORONOI_NONE + 1).tolist() == [10] * 9 + [11, 2]
    assert np.all(np.isnan(serve.slide_deciles(np.array([np.nan, np.nan]))))
    assert np.all(serve.voronoi_codes(np.array([np.nan]), serve.slide_deciles(np.array([np.nan]))) == serve.VORONOI_NONE)


def test_a_tissue_folder_gives_its_slides_with_their_portal_fields(tmp_path):
    written = write_tissue(tmp_path, "Skin - Sun Exposed (Lower leg)")
    cohort = serve.Cohort(None, written["folder"])
    meta = cohort.meta()
    assert meta["tissue"] == {
        "slug": "skin_sun_exposed_lower_leg", "name": "Skin - Sun Exposed (Lower leg)", "unmapped": ["Adipocyte"],
    }
    assert meta["groups"] == [] and meta["colocalization"] is None and meta["emeans"] is None
    assert meta["density"] == {"um_per_px": c.PX_UM, "voronoi_none": serve.VORONOI_NONE}
    assert meta["types"] == CLASS_NAMES and meta["type_class"] == [3, 0, 1, 2]
    assert meta["slides"][1] == {
        "slide_id": "GTEX-BBBB-0226", "group": "", "n_cells": 45, "pilot": False, "wsi": False,
        "categories": "clean_specimens, atrophy", "notes": "2 pieces, NA", "quality_flag": "", "hidden": True,
        "sex": "male", "age_bracket": "20-29", "hardy_scale": "Ventilator Case", "subject_id": "GTEX-BBBB",
    }
    assert [s["slide_id"] for s in meta["slides"]] == ["GTEX-AAAA-0126", "GTEX-BBBB-0226", "GTEX-CCCC-0326"]
    assert [s["hidden"] for s in meta["slides"]] == [False, True, False]
    info = cohort.info("GTEX-CCCC-0326")
    assert info["n_cells"] == 80 and info["group"] == "" and not info["wsi"] and info["um_source"] == "assumed"


def test_a_tissue_folder_gives_each_slides_own_cells(tmp_path):
    written = write_tissue(tmp_path, "Liver")
    cohort = serve.Cohort(None, written["folder"])
    for slide_id in ("GTEX-AAAA-0126", "GTEX-BBBB-0226", "GTEX-CCCC-0326"):
        cells = cohort.cells(slide_id)
        np.testing.assert_array_equal(cells["xy"], written["cells"][slide_id]["xy"])
        assert cells["types"].tolist() == type_codes(written["cells"][slide_id]["types"])


def test_a_tissue_folders_slides_are_found_on_disk_by_id(tmp_path):
    written = write_tissue(tmp_path / "tissues", "Liver")
    wsi = tmp_path / "wsi"
    wsi.mkdir()
    (wsi / "GTEX-BBBB-0226.svs").write_bytes(b"")
    cohort = serve.Cohort(wsi, written["folder"])
    assert [s["wsi"] for s in cohort.meta()["slides"]] == [False, True, False]


def test_read_cells_refuses_rows_of_another_slide_in_a_tissue_object(tmp_path):
    written = write_tissue(tmp_path, "Liver")
    h5ad = serve.tissue_h5ad(written["folder"])
    xy, types = serve.read_cells("GTEX-BBBB-0226", 60, 105, 4, h5ad)
    np.testing.assert_array_equal(xy, written["cells"]["GTEX-BBBB-0226"]["xy"])
    with pytest.raises(RuntimeError, match="not GTEX-BBBB-0226"):
        serve.read_cells("GTEX-BBBB-0226", 0, 45, 4, h5ad)


def test_a_tissue_has_none_of_the_liver_cohorts_precomputed_layers(tmp_path):
    cohort = serve.Cohort(None, write_tissue(tmp_path, "Lung")["folder"])
    for layer in (
        lambda: cohort.density("GTEX-AAAA-0126"),
        lambda: cohort.tda("GTEX-AAAA-0126", 4),
        lambda: cohort.emeans("GTEX-AAAA-0126", "composition", 4),
        lambda: cohort.colocalization("GTEX-AAAA-0126", 15, "types"),
    ):
        with pytest.raises(KeyError, match="liver"):
            layer()


def test_a_tissue_computes_the_layers_that_need_only_its_cells(tmp_path):
    cohort = serve.Cohort(None, write_tissue(tmp_path, "Lung")["folder"])
    blob = cohort.voronoi("GTEX-CCCC-0326")
    n = struct.unpack_from("<I", blob)[0]
    rho = np.frombuffer(blob, "<f4", n, 4)
    codes = np.frombuffer(blob, np.uint8, n, 4 + 4 * n)
    with_value = int(np.isfinite(rho).sum())
    assert n == 80 and 40 < with_value < 80 and np.all(codes[~np.isfinite(rho)] == serve.VORONOI_NONE)
    counts = np.bincount(codes[np.isfinite(rho)], minlength=serve.DENSITY_BINS)
    assert counts.sum() == with_value and np.all(np.abs(counts - with_value / 10) <= 1)  # the slide's own deciles
    blob = cohort.neighbourhood("GTEX-CCCC-0326", 5, "classes")
    assert struct.unpack_from("<I", blob)[0] == 80 and len(blob) == 4 + 4 * 5 * 80
    clusters = cohort.cluster("GTEX-CCCC-0326", 2, True, 150.0)
    assert sum(cluster["n"] for cluster in clusters["clusters"]) == 80


def test_a_tissue_with_more_cell_types_than_the_page_can_colour_is_refused(tmp_path):
    names = sorted(f"Type {j:02d}" for j in range(serve.MAX_TYPES + 1))
    folder = write_tissue(tmp_path, "Lung", class_names=names)["folder"]
    with pytest.raises(ValueError, match="cell types"):
        serve.Cohort(None, folder)


@pytest.fixture
def tissue_server(tmp_path):
    """A server of three tissue folders, one of which cannot be opened, and the cells written to the other two."""
    skin = write_tissue(tmp_path, "Skin - Sun Exposed (Lower leg)", seed=1)
    adipose = write_tissue(
        tmp_path,
        "Adipose - Visceral (Omentum)",
        slides=[("GTEX-DDDD-0426", 30, ("GTEX-DDDD", "female", "40-49", "Slow death", "", "", "false"))],
        seed=2,
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "gtex_broken_cells.h5ad").write_bytes(b"not an HDF5 file")
    (broken / "slides.csv").write_text(",".join([*PORTAL, "n_cells", "row_start", "row_end"]) + "\n")
    library = serve.Library(None, tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(library))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, {**skin["cells"], **adipose["cells"]}
    server.shutdown()
    server.server_close()
    thread.join()


def get(server: ThreadingHTTPServer, path: str) -> tuple[int, bytes]:
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        with err:
            return err.code, err.read()


def test_the_server_lists_the_tissues_and_opens_the_first_by_default(tissue_server):
    server, _ = tissue_server
    status, body = get(server, "/api/meta")
    meta = json.loads(body)
    assert status == 200 and meta["tissue"]["slug"] == "adipose_visceral_omentum"
    assert [s["slide_id"] for s in meta["slides"]] == ["GTEX-DDDD-0426"]
    listing = {t["slug"]: t for t in meta["tissues"]}
    assert list(listing) == ["adipose_visceral_omentum", "broken", "skin_sun_exposed_lower_leg"]
    assert listing["skin_sun_exposed_lower_leg"] == {
        "slug": "skin_sun_exposed_lower_leg", "name": "Skin - Sun Exposed (Lower leg)", "n_slides": 3, "error": None,
    }
    assert listing["broken"]["n_slides"] is None and listing["broken"]["error"]


def test_the_server_serves_each_tissues_own_slides_by_its_slug(tissue_server):
    server, cells = tissue_server
    tissue = "?tissue=skin_sun_exposed_lower_leg"
    status, body = get(server, f"/api/meta{tissue}")
    assert status == 200 and [s["slide_id"] for s in json.loads(body)["slides"]][0] == "GTEX-AAAA-0126"
    status, blob = get(server, f"/api/cells/GTEX-BBBB-0226{tissue}")
    n = struct.unpack_from("<I", blob)[0]
    assert status == 200 and n == 45
    np.testing.assert_array_equal(np.frombuffer(blob, "<f4", 2 * n, 4).reshape(n, 2), cells["GTEX-BBBB-0226"]["xy"])
    assert np.frombuffer(blob, np.uint8, n, 4 + 8 * n).tolist() == type_codes(cells["GTEX-BBBB-0226"]["types"])
    status, body = get(server, f"/api/slide/GTEX-BBBB-0226{tissue}")
    assert status == 200 and json.loads(body)["n_cells"] == 45
    assert get(server, f"/api/voronoi/GTEX-BBBB-0226{tissue}")[0] == 200
    assert get(server, f"/api/neighbourhood/GTEX-BBBB-0226{tissue}&k=5&by=types")[0] == 200


def test_the_server_refuses_the_liver_only_layers_other_tissues_and_unknown_ones(tissue_server):
    server, _ = tissue_server
    tissue = "?tissue=skin_sun_exposed_lower_leg"
    for path in (
        f"/api/density/GTEX-BBBB-0226{tissue}",
        f"/api/tda/GTEX-BBBB-0226{tissue}&k=4",
        f"/api/emeans/GTEX-BBBB-0226{tissue}&set=composition&k=4",
        f"/api/colocalization/GTEX-BBBB-0226{tissue}&k=15",
        f"/api/cells/GTEX-DDDD-0426{tissue}",  # a slide of another tissue
        "/api/cells/GTEX-DDDD-0426?tissue=lung",  # no such tissue
    ):
        status, body = get(server, path)
        assert status == 404, path
    status, body = get(server, "/api/meta?tissue=broken")
    assert status == 500 and "could not be opened" in json.loads(body)["error"]


def test_only_the_two_tissues_used_last_keep_their_slides_in_memory(tmp_path):
    for j, tissue in enumerate(("Liver", "Lung", "Spleen")):
        write_tissue(tmp_path, tissue, slides=[(f"GTEX-{tissue.upper()}-0126", 30 + j, SLIDES[0][2])], seed=j)
    library = serve.Library(None, tmp_path)
    liver = library.get("liver")
    liver.cells("GTEX-LIVER-0126")
    liver.voronoi("GTEX-LIVER-0126")
    library.get("lung").cells("GTEX-LUNG-0126")
    assert liver._cells and liver._voronoi  # two tissues in use: both keep theirs
    library.get("spleen").cells("GTEX-SPLEEN-0126")
    assert not liver._cells and not liver._voronoi  # a third: the one used longest ago lets go of its slides
    assert library.get("lung")._cells and library.get("spleen")._cells
    np.testing.assert_array_equal(liver.cells("GTEX-LIVER-0126")["xy"].shape, (30, 2))  # and reads them again


def test_the_fixtures_codes_take_the_width_pandas_gives_them(tmp_path):
    """anndata writes a categorical's codes as pandas holds them: int8 below 127 categories, int16 from there."""
    import pandas as pd

    with h5py.File(tmp_path / "codes.h5", "w") as f:
        for n in (126, 127):
            names = [f"S{j:03d}" for j in range(n)]
            categorical(f, f"c{n}", names, names)
            assert f[f"c{n}/codes"].dtype == pd.Categorical(names).codes.dtype
        assert f["c126/codes"].dtype == np.int8 and f["c127/codes"].dtype == np.int16


@pytest.mark.parametrize(("n_slides", "width"), [(3, np.int8), (130, np.int16)])
def test_a_tissue_reads_its_slides_whatever_the_width_of_their_codes(tmp_path, n_slides, width):
    slides = [(f"GTEX-S{j:03d}-0126", 3 + j % 4, SLIDES[0][2]) for j in range(n_slides)]
    written = write_tissue(tmp_path, "Lung", slides=slides)
    with h5py.File(serve.tissue_h5ad(written["folder"]), "r") as f:
        assert f["obs/slide_id/codes"].dtype == width and f["obs/cell_type/codes"].dtype == np.int8
    cohort = serve.Cohort(None, written["folder"])
    assert [s["slide_id"] for s in cohort.meta()["slides"]] == [s for s, _, _ in slides]
    for slide_id in (slides[0][0], slides[n_slides // 2][0], slides[-1][0]):
        cells = cohort.cells(slide_id)
        np.testing.assert_array_equal(cells["xy"], written["cells"][slide_id]["xy"])
        assert cells["types"].tolist() == type_codes(written["cells"][slide_id]["types"])
