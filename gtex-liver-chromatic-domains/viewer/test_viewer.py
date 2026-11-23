"""Tests of the viewer's server: the slide table, the cells blob, the blank canvas and the slide-file lookup."""

import struct

import numpy as np
import pandas as pd
import pytest
import serve

import common as c


def test_groups_are_the_pilot_conditions_then_the_two_left_out():
    assert (*c.CONDITIONS, "necrosis", "unclassified") == serve.GROUPS


def test_slide_table_orders_by_group_then_id_and_keeps_each_slides_rows():
    pathology = pd.DataFrame(
        {
            "slide_id": ["S3", "S1", "S2", "S4"],
            "group": ["cirrhosis", "unclassified", "healthy", "healthy"],
        }
    )
    table = serve.slide_table(pathology, ["S1", "S2", "S3", "S4"], np.array([0, 10, 25, 40, 41]), {"S2"})
    assert table["slide_id"].tolist() == ["S2", "S4", "S3", "S1"]
    assert table["start"].tolist() == [10, 40, 25, 0]
    assert table["n_cells"].tolist() == [15, 1, 15, 10]
    assert table["pilot"].tolist() == [True, False, False, False]


def test_slide_table_refuses_an_unknown_group_or_different_slides():
    bounds = np.array([0, 5, 9])
    with pytest.raises(ValueError, match="not in GROUPS"):
        serve.slide_table(
            pd.DataFrame({"slide_id": ["S1", "S2"], "group": ["healthy", "tumour"]}), ["S1", "S2"], bounds, set()
        )
    with pytest.raises(ValueError, match="same slides"):
        serve.slide_table(
            pd.DataFrame({"slide_id": ["S1", "S3"], "group": ["healthy", "healthy"]}), ["S1", "S2"], bounds, set()
        )
    with pytest.raises(ValueError, match="bounds"):
        serve.slide_table(pd.DataFrame({"slide_id": ["S1"], "group": ["healthy"]}), ["S1"], bounds, set())


def test_cells_blob_layout():
    xy = np.array([[1.5, 2.0], [3.0, 4.25], [65735.5, 37220.0]], dtype=np.float32)
    types = np.array([0, 13, 5], dtype=np.uint8)
    blob = serve.cells_blob(xy, types)
    n = struct.unpack_from("<I", blob)[0]
    assert n == 3 and len(blob) == 4 + 9 * n
    assert np.array_equal(np.frombuffer(blob, "<f4", 2 * n, 4).reshape(n, 2), xy)
    assert np.array_equal(np.frombuffer(blob, np.uint8, n, 4 + 8 * n), types)
    with pytest.raises(ValueError, match="coordinates"):
        serve.cells_blob(xy[:2], types)


def test_cloud_extent_holds_every_cell_with_the_margin():
    assert serve.cloud_extent(np.array([[0.0, 0.0], [10.2, 3.0]])) == (11 + serve.MARGIN_PX, 3 + serve.MARGIN_PX)
    assert serve.cloud_extent(np.zeros((0, 2))) == (2 * serve.MARGIN_PX, 2 * serve.MARGIN_PX)
    with pytest.raises(ValueError, match="negative"):
        serve.cloud_extent(np.array([[-1.0, 2.0]]))


def test_find_wsis_matches_cohort_slides_by_file_name(tmp_path):
    for name in ("GTEX-A.svs", "GTEX-B.TIFF", "GTEX-C.txt", "GTEX-Z.svs", "notes.svs"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "GTEX-D.svs").mkdir()
    found = serve.find_wsis(tmp_path, {"GTEX-A", "GTEX-B", "GTEX-C", "GTEX-D"})
    assert found == {"GTEX-A": tmp_path / "GTEX-A.svs", "GTEX-B": tmp_path / "GTEX-B.TIFF"}
    (tmp_path / "GTEX-A.ndpi").write_bytes(b"")
    with pytest.raises(ValueError, match="two slide files for GTEX-A"):
        serve.find_wsis(tmp_path, {"GTEX-A"})


def test_wsi_dir_must_be_a_folder(tmp_path):
    with pytest.raises(SystemExit):
        serve.parse_args(["--wsi-dir", str(tmp_path / "missing")])
    assert serve.parse_args(["--wsi-dir", str(tmp_path)]).wsi_dir == tmp_path
    assert serve.parse_args([]).wsi_dir is None
