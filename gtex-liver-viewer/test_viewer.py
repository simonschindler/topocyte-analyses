"""Tests of the viewer's server: the slide table, the cells blob, the blank canvas, the slide-file lookup, the
clusterings and the saved e-means labels."""

import json
import struct

import numpy as np
import pandas as pd
import pytest

import serve

c = serve.c  # the chromatic-domains study's common module, which serve puts on the path


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


def test_cluster_request_reads_k_density_and_radius():
    assert serve.cluster_request({"k": ["4"]}) == (4, True, c.DELTA_UM)
    assert serve.cluster_request({"k": ["10"], "density": ["0"], "radius": ["75.5"]}) == (10, False, 75.5)
    for query, message in (
        ({"k": ["1"]}, "k must be from"),
        ({"k": ["4.5"]}, "whole number"),
        ({}, "whole number"),
        ({"k": ["4"], "radius": ["5"]}, "radius must be from"),
        ({"k": ["4"], "radius": ["nan"]}, "radius must be from"),
        ({"k": ["4"], "radius": ["600"]}, "radius must be from"),
    ):
        with pytest.raises(ValueError, match=message):
            serve.cluster_request(query)


def test_kmeans_by_size_refuses_too_few_rows_and_numbers_by_size():
    X = np.array([[0.0], [0.1], [0.2], [9.0]])
    assert serve.kmeans_by_size(X, 2).tolist() == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="cannot form"):
        serve.kmeans_by_size(X[:1], 2)


def pack(labels: np.ndarray) -> np.ndarray:
    """The e-means study's ``cluster.pack``: two labels per byte, the even rows in the high half."""
    padded = np.append(labels, np.uint8(0)) if len(labels) % 2 else labels
    return (padded[0::2] << 4) | padded[1::2]


def test_unpack_rows_reads_any_row_range_of_two_labels_per_byte():
    labels = np.array([3, 15, 0, 9, 7, 1, 12], dtype=np.uint8)
    packed = pack(labels)
    for start, end in ((0, 7), (1, 6), (3, 4), (2, 2), (6, 7), (1, 2)):
        assert serve.unpack_rows(packed, start, end).tolist() == labels[start:end].tolist()
    with pytest.raises(ValueError, match="packed labels"):
        serve.unpack_rows(packed, 5, 9)


def test_emeans_request_reads_the_set_and_k():
    assert serve.emeans_request({"set": ["composition+density"], "k": ["3"]}) == ("composition+density", 3)
    with pytest.raises(ValueError, match="set must be one of"):
        serve.emeans_request({"set": ["shape"], "k": ["3"]})
    with pytest.raises(ValueError, match="k must be from"):
        serve.emeans_request({"set": ["density"], "k": ["11"]})


def test_emeans_clusters_code_by_cohort_size_and_keep_the_saved_labels():
    def profile(n: int) -> dict:
        fractions = dict(zip(c.CLASSES, (0.7, 0.1, 0.1, 0.1), strict=True))
        return {
            "n": n,
            "share": n / 60,
            "cells_per_ball": 100.0,
            "class_fractions": fractions,
            "kept_fractions": [1.0] * 10,
        }

    profiles = [profile(10), profile(30), profile(20)]
    codes, entries = serve.emeans_clusters(profiles, np.array([0, 1, 2, 1], dtype=np.uint8), 3)
    assert codes.tolist() == [2, 0, 1, 0]
    assert [e["label"] for e in entries] == [1, 2, 0] and [e["n"] for e in entries] == [2, 1, 1]
    assert [e["cohort_n"] for e in entries] == [30, 20, 10]
    with pytest.raises(RuntimeError, match="not below k"):
        serve.emeans_clusters(profiles, np.array([3], dtype=np.uint8), 3)
    with pytest.raises(ValueError, match="profiles for k"):
        serve.emeans_clusters(profiles[:2], np.array([0], dtype=np.uint8), 3)


def test_load_emeans_reads_each_sets_summary_and_its_worse_fits(tmp_path):
    assert serve.load_emeans(tmp_path) is None
    gap, clusters, check = {"settings": {"radius_um": 150.0}, "sets": {}}, {}, {"sets": {}}
    for name in serve.EMEANS_SETS:
        gap["sets"][name] = {"features": {"columns": ["a", "b"]}, "k_tibshirani": 3, "n_cells": 5}
        clusters[name] = {"clusters": {"2": []}}
        check["sets"][name] = {"k": [1, 2, 3], "log_ratio_run_to_kmeans": [0.0, 0.0431, 0.0019]}
        np.savez(tmp_path / serve.emeans_file(name), n_cells=np.int64(5))
    for file, data in (("gap.json", gap), ("clusters.json", clusters), ("kmeans_check.json", check)):
        (tmp_path / file).write_text(json.dumps(data))
    out = serve.load_emeans(tmp_path)
    assert list(out) == list(serve.EMEANS_SETS)
    assert out["density"]["worse"] == {2: 0.0431} and out["density"]["k_tibshirani"] == 3


def test_npz_memmap_maps_a_stored_member_and_refuses_a_compressed_one(tmp_path):
    rho = np.linspace(0.5, 9.5, 1001)
    np.savez(tmp_path / "stored.npz", other=np.arange(3), rho=rho)
    mapped = serve.npz_memmap(tmp_path / "stored.npz", "rho")
    assert isinstance(mapped, np.memmap) and mapped.dtype == rho.dtype and np.array_equal(mapped, rho)
    np.savez_compressed(tmp_path / "compressed.npz", rho=rho)
    with pytest.raises(ValueError, match="compressed"):
        serve.npz_memmap(tmp_path / "compressed.npz", "rho")


def test_density_codes_are_the_last_density_slice_that_keeps_the_cell():
    deciles = np.array([0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    rho = np.array([0.0, 9.99, 10.0, 55.0, 89.9, 90.0, 1e6])
    assert serve.density_codes(rho, deciles).tolist() == [0, 0, 1, 5, 8, 9, 9]


def test_density_blob_layout():
    rho = np.array([0.5, 1200.25, 4100.0])
    bins = np.array([0, 1, 9], dtype=np.uint8)
    blob = serve.density_blob(rho, bins)
    n = struct.unpack_from("<I", blob)[0]
    assert n == 3 and len(blob) == 4 + 5 * n
    assert np.array_equal(np.frombuffer(blob, "<f4", n, 4), rho.astype(np.float32))
    assert np.array_equal(np.frombuffer(blob, np.uint8, n, 4 + 4 * n), bins)
    with pytest.raises(ValueError, match="densities"):
        serve.density_blob(rho[:2], bins)


def test_voronoi_areas_of_a_square_lattice_are_its_squares_and_the_hull_has_none():
    # 7 × 7 points 3 px apart, far from the origin: every four neighbours are cocircular, the triangulation's
    # worst case, and the 24 points on the outer square (corners and edges alike) have unbounded cells.
    grid = np.arange(7) * 3.0 + 40_000.5
    xy = np.array([(x, y) for x in grid for y in grid])
    area = serve.voronoi_areas(xy)
    border = np.isin(xy[:, 0], grid[[0, -1]]) | np.isin(xy[:, 1], grid[[0, -1]])
    assert border.sum() == 24 and np.all(np.isnan(area[border]))
    np.testing.assert_allclose(area[~border], 9.0, rtol=1e-9)


def test_voronoi_areas_of_a_hexagonal_lattice_are_its_hexagons():
    s = 2.0
    xy = np.array([(i * s + (j % 2) * s / 2, j * s * np.sqrt(3) / 2) for j in range(9) for i in range(9)])
    area = serve.voronoi_areas(xy)
    inner = [9 * j + i for j in range(2, 7) for i in range(2, 7)]
    np.testing.assert_allclose(area[inner], s * s * np.sqrt(3) / 2, rtol=1e-9)


def test_voronoi_areas_are_scipys_voronoi_polygons():
    from scipy.spatial import ConvexHull, Voronoi

    xy = np.random.default_rng(3).uniform(0, 500, size=(400, 2))
    vor = Voronoi(xy)
    expected = np.full(len(xy), np.nan)
    for i, r in enumerate(vor.point_region):
        region = vor.regions[r]
        if region and -1 not in region:
            expected[i] = ConvexHull(vor.vertices[region]).volume
    area = serve.voronoi_areas(xy)
    assert np.array_equal(np.isnan(area), np.isnan(expected)) and np.isfinite(area).sum() > 350
    np.testing.assert_allclose(area, expected, rtol=1e-9)


def test_voronoi_areas_leave_out_shared_positions_and_need_three_points_off_a_line():
    xy = np.random.default_rng(4).uniform(0, 100, size=(60, 2))
    twin = np.vstack([xy, xy[[30]]])  # a 61st cell at the 31st cell's position: neither has a cell of its own
    area = serve.voronoi_areas(twin)
    assert np.isnan(area[30]) and np.isnan(area[60])
    np.testing.assert_allclose(np.delete(area[:60], 30), np.delete(serve.voronoi_areas(xy), 30), rtol=1e-12)
    assert np.all(np.isnan(serve.voronoi_areas(xy[:2])))
    assert np.all(np.isnan(serve.voronoi_areas(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))))
    assert serve.voronoi_areas(np.zeros((0, 2))).shape == (0,)


def test_voronoi_density_is_cells_per_mm2_and_its_codes_mark_cells_without_one():
    grid = np.arange(5) * 3.0
    xy = np.array([(x, y) for x in grid for y in grid])
    rho = serve.voronoi_density(xy, um_per_px=0.5)  # a 3 px square is 1.5 µm on a side
    np.testing.assert_allclose(rho[12], 1e6 / 1.5**2, rtol=1e-9)
    assert np.isnan(rho[0])
    deciles = np.array([5.0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    codes = serve.voronoi_codes(np.array([0.5, np.nan, 15.0, 1e9]), deciles)
    assert codes.dtype == np.uint8 and codes.tolist() == [0, serve.VORONOI_NONE, 1, 9]
    assert serve.VORONOI_NONE == serve.DENSITY_BINS


def test_threshold_curve_puts_the_deciles_at_every_hundredth_step():
    levels = np.linspace(0, 1, serve.CURVE_STEPS + 1)
    curve = 1000 * levels**2
    deciles = curve[::100][:10] + 1e-9  # the study's deciles differ from its curve in the last digits
    out = serve.threshold_curve(curve, levels, deciles)
    assert np.array_equal(out[0:1000:100], deciles) and out[1] == curve[1] and out[-1] == curve[-1]
    with pytest.raises(ValueError, match="levels"):
        serve.threshold_curve(curve[:-1], levels[:-1], deciles)
    with pytest.raises(ValueError, match="does not rise"):
        serve.threshold_curve(curve, levels, deciles[::-1])


def test_load_notes_keeps_empty_fields_as_empty_text(tmp_path):
    path = tmp_path / "pathology.csv"
    path.write_text(
        "slide_id,group,label,label_reason,categories,notes,quality_flag,n_cells\n"
        'S1,cirrhosis,cirrhosis,curated,"cirrhosis, fibrosis","2 pieces; cirrhosis",,10\n'
        "S2,unclassified,other,none,,,preservation,5\n"
    )
    assert serve.load_notes(path) == {
        "S1": {"categories": "cirrhosis, fibrosis", "notes": "2 pieces; cirrhosis", "quality_flag": ""},
        "S2": {"categories": "", "notes": "", "quality_flag": "preservation"},
    }


def test_load_donors_reads_sex_age_and_hardy_scale_and_refuses_a_slide_twice(tmp_path):
    path = tmp_path / "liver_cohort.csv"
    header = "Tissue Sample ID,Tissue,Subject ID,Sex,Age Bracket,Hardy Scale,Pathology Categories,Pathology Notes\n"
    path.write_text(header + 'S1,Liver,D1,female,60-69,Slow death,cirrhosis,"2 pieces, cirrhosis"\n')
    assert serve.load_donors(path) == {"S1": {"sex": "female", "age_bracket": "60-69", "hardy_scale": "Slow death"}}
    path.write_text(header + "S1,Liver,D1,female,60-69,Slow death,,\nS1,Liver,D1,male,20-29,Ventilator case,,\n")
    with pytest.raises(ValueError, match="twice"):
        serve.load_donors(path)
