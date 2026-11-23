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


def test_um_per_px_comes_from_the_slide_file_else_the_cohorts_assumption():
    assert serve.slide_um_per_px({"openslide.mpp-x": "0.2527"}) == (0.2527, "slide file")
    for properties in ({}, {"openslide.mpp-x": "0"}, {"openslide.mpp-x": "n/a"}, {"openslide.mpp-x": "inf"}):
        assert serve.slide_um_per_px(properties) == (c.PX_UM, "assumed")


def test_ball_class_counts_use_closed_balls_with_the_cell_itself():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [100.0, 0.0]])
    classes = np.array([0, 2, 2, 1])
    counts = serve.ball_class_counts(xy, classes, radius_px=5.0)
    assert counts.tolist() == [[1, 0, 2, 0], [1, 0, 2, 0], [1, 0, 2, 0], [0, 1, 0, 0]]
    features = serve.level1(counts)
    assert np.allclose(features[0], [np.log10(3), 1 / 3, 0, 2 / 3, 0], rtol=0, atol=1e-15)
    assert np.allclose(features[3], [0, 0, 1, 0, 0], rtol=0, atol=1e-15)


def test_composition_clusters_number_clusters_by_size_and_skip_constant_columns():
    rng = np.random.default_rng(0)
    big = rng.normal(0.0, 0.1, size=(300, 2))
    small = rng.normal(5.0, 0.1, size=(100, 2))
    features = np.column_stack([np.vstack([small, big]), np.full(400, 0.25)])  # the third column is constant
    labels, used = serve.composition_clusters(features, 2)
    assert used == [0, 1]
    assert labels.dtype == np.uint8 and labels[:100].tolist() == [1] * 100 and labels[100:].tolist() == [0] * 300
    profiles = serve.cluster_profiles(np.column_stack([np.zeros(400), features[:, :2], np.zeros((400, 2))]), labels, 2)
    assert [p["n"] for p in profiles] == [300, 100]
    with pytest.raises(ValueError, match="k must be"):
        serve.composition_clusters(features, 11)
    with pytest.raises(ValueError, match="constant"):
        serve.composition_clusters(np.ones((10, 3)), 2)
