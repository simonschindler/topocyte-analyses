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
    assert "skin_sun_exposed_lower_leg" in json.loads(body)["error"]  # an unknown tissue: those served, by name
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


def test_one_listing_of_the_slide_folder_serves_every_tissue(tmp_path, monkeypatch):
    for j, tissue in enumerate(("Liver", "Lung")):
        name = tissue.upper()
        slides = [(f"GTEX-{name}-0126", 30, SLIDES[0][2]), (f"GTEX-{name}-0226", 20, SLIDES[1][2])]
        write_tissue(tmp_path / "tissues", tissue, slides=slides, seed=j)
    wsi = tmp_path / "svs"  # one folder of every GTEx slide file, as on the cluster
    wsi.mkdir()
    for name in ("GTEX-LIVER-0226.svs", "GTEX-LUNG-0126.svs", "GTEX-OTHER-0126.svs", "GTEX-LUNG-0226.txt"):
        (wsi / name).write_bytes(b"")
    listings = []
    listed = serve.list_wsis
    monkeypatch.setattr(serve, "list_wsis", lambda folder: listings.append(folder) or listed(folder))
    library = serve.Library(wsi, tmp_path / "tissues")
    assert listings == [wsi]  # listed once for both tissues, not once each
    assert {s["slide_id"]: s["wsi"] for s in library.get("liver").meta()["slides"]} == {
        "GTEX-LIVER-0126": False, "GTEX-LIVER-0226": True,
    }
    assert {s["slide_id"]: s["wsi"] for s in library.get("lung").meta()["slides"]} == {
        "GTEX-LUNG-0126": True, "GTEX-LUNG-0226": False,
    }


def test_a_slides_rows_must_be_all_of_its_cells_not_part_of_them(tmp_path):
    written = write_tissue(tmp_path, "Lung")  # in the object, A is rows 0-60, B 60-105, C 105-185
    path = written["folder"] / "slides.csv"
    lines = path.read_text().splitlines()
    rows = [line.rsplit(",", 3) for line in lines[1:]]
    rows[0][1:] = ["30", "0", "30"]  # A: only its first 30 cells
    rows[1][1:] = ["75", "30", "105"]  # B: from A's 31st cell on
    path.write_text("\n".join([lines[0], *(",".join(row) for row in rows)]) + "\n")
    cohort = serve.Cohort(None, written["folder"])  # the rows still run from 0 to the last cell without gaps
    with pytest.raises(RuntimeError, match="GTEX-AAAA-0126"):
        cohort.cells("GTEX-AAAA-0126")
    with pytest.raises(RuntimeError, match="GTEX-BBBB-0226"):
        cohort.cells("GTEX-BBBB-0226")
    np.testing.assert_array_equal(cohort.cells("GTEX-CCCC-0326")["xy"], written["cells"]["GTEX-CCCC-0326"]["xy"])


def test_a_tissues_slide_file_tiles_do_not_count_as_using_it(tmp_path):
    for j, tissue in enumerate(("Liver", "Lung", "Spleen")):
        write_tissue(tmp_path, tissue, slides=[(f"GTEX-{tissue.upper()}-0126", 30 + j, SLIDES[0][2])], seed=j)
    library = serve.Library(None, tmp_path)
    library.get("liver").cells("GTEX-LIVER-0126")
    library.get("lung").cells("GTEX-LUNG-0126")
    library.get("spleen", warm=False)  # a tile of a spleen slide: the H&E only, none of the slides' values
    assert library.get("liver", warm=False)._cells and library.get("lung", warm=False)._cells


# -- a tissue's own colocalization fit (fit_colocalization.py --tissue, served with --colocalization-dir) ----------


def fit_tissue(folder: Path, out_dir: Path) -> dict:
    import fit_colocalization

    fit_colocalization.main(["--tissue", str(folder), "--out-dir", str(out_dir)])
    return json.loads((out_dir / f"{folder.name}.json").read_text())


def test_a_tissues_fit_sums_its_own_slides_by_its_own_types_and_class_map(tmp_path):
    from local_colocalization import excess_sums

    written = write_tissue(tmp_path / "root", "Lung")
    fit = fit_tissue(written["folder"], tmp_path / "fits")
    assert fit["tissue"]["slug"] == "lung" and fit["tissue"]["unmapped"] == ["Adipocyte"]
    assert (fit["n_slides"], fit["n_cells"]) == (3, 185) and fit["k"] == list(range(2, 101))
    assert fit["label_sets"]["types"]["labels"] == CLASS_NAMES
    assert fit["label_sets"]["classes"]["labels"] == list(c.CLASSES)
    classes, _ = serve.class_codes_or_other(CLASS_NAMES)  # Adipocyte, outside the four classes, counts as other
    for by, n_labels in (("types", len(CLASS_NAMES)), ("classes", len(c.CLASSES))):
        pair = np.zeros((len(fit["k"]), n_labels, n_labels))
        target = np.zeros((len(fit["k"]), n_labels))
        for slide_id, _, _ in SLIDES:  # each slide alone, as the fit's training clouds
            codes = np.array(type_codes(written["cells"][slide_id]["types"]), dtype=np.int64)
            labels = codes if by == "types" else classes[codes].astype(np.int64)
            xy = written["cells"][slide_id]["xy"].astype(np.float64)
            p, t = excess_sums(xy, labels, n_labels, fit["k"])
            pair += p
            target += t
        np.testing.assert_allclose(fit["label_sets"][by]["pair_excess"], np.round(pair, 3), atol=1e-9)
        np.testing.assert_allclose(fit["label_sets"][by]["target_excess"], np.round(target, 3), atol=1e-9)


def test_a_tissue_with_its_fit_colours_its_cells_by_their_colocalization_excess(tmp_path):
    written = write_tissue(tmp_path / "root", "Lung")
    fit_tissue(written["folder"], tmp_path / "fits")
    cohort = serve.Cohort(None, written["folder"], colocalization_dir=tmp_path / "fits")
    meta = cohort.meta()["colocalization"]
    assert meta["n_slides"] == 3 and (meta["k_min"], meta["k_max"]) == (2, 100)
    assert meta["labels"] == {"types": CLASS_NAMES, "classes": list(c.CLASSES)}
    for by, n_labels in (("types", len(CLASS_NAMES)), ("classes", len(c.CLASSES))):
        entry = cohort.colocalization("GTEX-CCCC-0326", 15, by)
        assert entry["raw"].shape == entry["shrunk"].shape == (80, n_labels)
        assert entry["gain"].shape == (n_labels, n_labels) and np.isfinite(entry["shrunk"]).any()
    assert cohort.colocalization_note is None


def test_a_tissue_without_its_fit_has_no_colocalization_excess(tmp_path):
    written = write_tissue(tmp_path / "root", "Lung")
    (tmp_path / "fits").mkdir()
    cohort = serve.Cohort(None, written["folder"], colocalization_dir=tmp_path / "fits")
    assert cohort.meta()["colocalization"] is None and cohort.colocalization_note is None
    with pytest.raises(KeyError, match="no fit of its own"):
        cohort.colocalization("GTEX-AAAA-0126", 15, "types")


@pytest.mark.parametrize(
    "spoil, reason",
    [
        (lambda fit: fit.update(n_cells=fit["n_cells"] + 1), "fit it again"),  # an older build's object
        (lambda fit: fit.update(n_slides=2), "fit it again"),
        (lambda fit: fit.pop("n_cells"), "unstated number"),
        (lambda fit: fit["label_sets"]["types"].update(labels=["A", "B", "C", "D"]), "other cell types"),
        (lambda fit: fit["label_sets"].pop("classes"), "other cell types"),
        (lambda fit: fit.update(k="not a list"), "cannot be read"),
    ],
)
def test_a_tissue_ignores_a_fit_of_another_object_and_says_why(tmp_path, spoil, reason):
    written = write_tissue(tmp_path / "root", "Lung")
    fit = fit_tissue(written["folder"], tmp_path / "fits")
    spoil(fit)
    (tmp_path / "fits" / "lung.json").write_text(json.dumps(fit))
    cohort = serve.Cohort(None, written["folder"], colocalization_dir=tmp_path / "fits")
    assert cohort.colocalization_fit is None and reason in cohort.colocalization_note
    assert cohort.meta()["colocalization"] is None
    with pytest.raises(KeyError, match=reason):
        cohort.colocalization("GTEX-AAAA-0126", 15, "types")


def test_a_truncated_fit_is_ignored_not_fatal(tmp_path):
    written = write_tissue(tmp_path / "root", "Lung")
    (tmp_path / "fits").mkdir()
    (tmp_path / "fits" / "lung.json").write_text('{"k": [2, 3')
    cohort = serve.Cohort(None, written["folder"], colocalization_dir=tmp_path / "fits")
    assert cohort.colocalization_fit is None and "cannot be read" in cohort.colocalization_note


def test_the_server_serves_the_excess_of_the_tissues_with_a_fit_only(tmp_path):
    lung = write_tissue(tmp_path / "root", "Lung", seed=1)
    write_tissue(tmp_path / "root", "Spleen", slides=[("GTEX-DDDD-0426", 30, SLIDES[0][2])], seed=2)
    fit_tissue(lung["folder"], tmp_path / "fits")
    library = serve.Library(None, tmp_path / "root", tmp_path / "fits")
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(library))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = get(server, "/api/meta?tissue=lung")
        assert status == 200 and json.loads(body)["colocalization"]["n_slides"] == 3
        status, body = get(server, "/api/meta?tissue=spleen")
        assert status == 200 and json.loads(body)["colocalization"] is None
        status, blob = get(server, "/api/colocalization/GTEX-CCCC-0326?tissue=lung&k=15&by=types")
        assert status == 200 and struct.unpack_from("<I", blob)[0] == 80
        assert get(server, "/api/colocalization/GTEX-CCCC-0326/summary?tissue=lung&k=15&by=classes")[0] == 200
        status, body = get(server, "/api/colocalization/GTEX-DDDD-0426?tissue=spleen&k=15")
        assert status == 404 and "no fit of its own" in json.loads(body)["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_the_colocalization_folder_goes_with_tissues_and_a_tissue_fit_with_its_folder(tmp_path, capsys):
    import fit_colocalization

    written = write_tissue(tmp_path / "root", "Lung")
    (tmp_path / "fits").mkdir()
    with pytest.raises(SystemExit):
        serve.parse_args(["--colocalization-dir", str(tmp_path / "fits")])  # the liver cohort: no tissues
    with pytest.raises(SystemExit):
        serve.parse_args(["--tissues", str(tmp_path / "root"), "--colocalization-dir", str(tmp_path / "none")])
    args = serve.parse_args(["--tissues", str(tmp_path / "root"), "--colocalization-dir", str(tmp_path / "fits")])
    assert args.colocalization_dir == tmp_path / "fits"
    with pytest.raises(SystemExit):
        fit_colocalization.parse_args(["--tissue", str(written["folder"])])  # without --out-dir
    with pytest.raises(SystemExit):
        fit_colocalization.parse_args(["--tissue", str(tmp_path / "root"), "--out-dir", str(tmp_path / "fits")])
    assert "one tissue folder" in capsys.readouterr().err


def test_k_d_tree_queries_use_a_bounded_number_of_threads():
    """Not one thread per core of the node (workers=-1): on a 48-core node their malloc arenas exhausted the
    viewer's capped address space, and colocalization requests failed."""
    import local_colocalization as lc

    assert 1 <= lc.WORKERS <= 8
    assert serve.WORKERS == lc.WORKERS
    for module in (serve, lc):
        assert "workers=-1" not in Path(module.__file__).read_text()


def test_blas_threads_default_to_a_few():
    import os

    assert os.environ["OPENBLAS_NUM_THREADS"].isdigit()


def test_malloc_arenas_are_limited_on_glibc():
    import platform

    if platform.libc_ver()[0] != "glibc":
        pytest.skip("mallopt(M_ARENA_MAX) is glibc's")
    assert serve.limit_malloc_arenas(2) is True


class Concurrency:
    """Wraps a function to count its calls and the most of them running at once."""

    def __init__(self, fn):
        self.fn, self.calls, self.running, self.most = fn, 0, 0, 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        import time

        with self.lock:
            self.calls += 1
            self.running += 1
            self.most = max(self.most, self.running)
        try:
            time.sleep(0.2)  # long enough for the other requests to arrive meanwhile
            return self.fn(*args, **kwargs)
        finally:
            with self.lock:
                self.running -= 1

    def __get__(self, instance, owner=None):  # patched onto a class, it wraps a method
        import functools

        return self if instance is None else functools.partial(self, instance)


def all_at_once(*calls) -> list:
    results, errors = [None] * len(calls), []

    def run(i, call):
        try:
            results[i] = call()
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=run, args=(i, call)) for i, call in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return results


def test_the_pages_two_requests_for_one_slides_excess_compute_it_once(tmp_path, monkeypatch):
    """The page asks for a slide's excess and its summary at once: computing it twice at once held two of the
    largest slide's neighbour tables (4.4 GB each at k = 100) and failed in an 8 GiB address space."""
    from local_colocalization import LocalColocalizationExtractor

    written = write_tissue(tmp_path / "root", "Lung")
    fit_tissue(written["folder"], tmp_path / "fits")
    cohort = serve.Cohort(None, written["folder"], colocalization_dir=tmp_path / "fits")
    transform = Concurrency(LocalColocalizationExtractor.transform)
    monkeypatch.setattr(LocalColocalizationExtractor, "transform", transform)
    values, summary, again = all_at_once(
        lambda: cohort.colocalization_values("GTEX-CCCC-0326", 50, "types", None, "shrunk"),
        lambda: cohort.colocalization_summary("GTEX-CCCC-0326", 50, "types"),
        lambda: cohort.colocalization_summary("GTEX-CCCC-0326", 50, "types"),
    )
    assert transform.calls == 1 and summary == again and values


def test_neighbour_tables_are_built_one_at_a_time_across_slides_layers_and_tissues(tmp_path, monkeypatch):
    from local_colocalization import LocalColocalizationExtractor

    lung = write_tissue(tmp_path / "root", "Lung")
    spleen = write_tissue(tmp_path / "root", "Spleen")
    for written in (lung, spleen):
        fit_tissue(written["folder"], tmp_path / "fits")
    lungs, spleens = (serve.Cohort(None, w["folder"], colocalization_dir=tmp_path / "fits") for w in (lung, spleen))
    transform = Concurrency(LocalColocalizationExtractor.transform)
    monkeypatch.setattr(LocalColocalizationExtractor, "transform", transform)
    nearest = Concurrency(serve.nearest_others)
    monkeypatch.setattr(serve, "nearest_others", nearest)
    all_at_once(
        lambda: lungs.colocalization("GTEX-CCCC-0326", 15, "types"),
        lambda: lungs.colocalization("GTEX-CCCC-0326", 20, "types"),
        lambda: lungs.colocalization("GTEX-AAAA-0126", 15, "classes"),
        lambda: spleens.colocalization("GTEX-CCCC-0326", 15, "types"),
        lambda: lungs.neighbourhood("GTEX-CCCC-0326", 15, "types"),
        lambda: spleens.neighbourhood("GTEX-AAAA-0126", 30, "classes"),
    )
    assert (transform.calls, nearest.calls) == (4, 2)
    assert transform.most == nearest.most == 1
