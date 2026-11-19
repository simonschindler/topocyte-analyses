"""Unit tests of the improvement phase's selection and holdout scoring, on synthetic inputs (no cohort)."""

from __future__ import annotations

import numpy as np

import improve
import improve_holdout


def _entry(name, mechanism, role, kind, auc, auc_interval, purity_interval):
    readout = {"k": 4, "macro_auc": auc, "vs_everything4": {"difference": auc - 0.70, "interval": auc_interval},
               "purity_oof_vs_everything4": {"difference": float(np.mean(purity_interval)), "interval": purity_interval}}
    return {"name": name, "mechanism": mechanism, "role": role, "kind": kind, "readouts": [readout]}


def test_choose_takes_ideas_above_zero_one_per_mechanism():
    dev = {"ideas": [
        _entry("baseline:everything", "baseline", "baseline", "space", 0.70, [0.0, 0.0], [0.0, 0.0]),
        _entry("control", "supervised projection (composition)", "control", "supervised", 0.80, [0.05, 0.10], [0.1, 0.2]),
        _entry("diagnostic", "supervised projection (residual)", "diagnostic", "supervised", 0.79, [0.05, 0.10], [0.1, 0.2]),
        _entry("a", "supervised projection (topology)", "idea", "supervised", 0.77, [0.03, 0.10], [-0.1, 0.3]),
        _entry("b", "supervised projection (topology)", "idea", "supervised", 0.76, [0.04, 0.10], [-0.1, 0.3]),
        _entry("c", "shape only", "idea", "space", 0.72, [-0.01, 0.05], [0.0, 0.3]),
        _entry("d", "block weighting", "idea", "space", 0.75, [0.01, 0.08], [-0.5, -0.1]),  # loses purity
        _entry("e", "supervised projection (composition + topology)", "idea", "supervised", 0.78, [0.04, 0.11], [0.0, 0.2]),
    ]}
    assert [entry["name"] for entry in improve_holdout.choose(dev)] == ["e", "a"]


def test_choose_falls_back_to_the_best_idea():
    dev = {"ideas": [
        _entry("c", "shape only", "idea", "space", 0.72, [-0.01, 0.05], [0.0, 0.3]),
        _entry("f", "block selection", "idea", "selection", 0.73, [-0.02, 0.05], [0.0, 0.3]),
        _entry("control", "supervised projection (composition)", "control", "supervised", 0.90, [0.05, 0.10], [0.1, 0.2]),
    ]}
    assert [entry["name"] for entry in improve_holdout.choose(dev)] == ["f"]


def test_neighbour_purity_against_a_pool():
    rng = np.random.default_rng(0)
    pool = np.vstack([rng.normal(0, 0.01, (20, 2)), rng.normal(5, 0.01, (10, 2))])
    pool_labels = np.array([0] * 20 + [1] * 10)
    queries = np.vstack([rng.normal(0, 0.01, (3, 2)), rng.normal(5, 0.01, (2, 2)), rng.normal(5, 0.01, (1, 2))])
    query_labels = np.array([0, 0, 0, 1, 1, 7])
    values = improve.neighbour_purity(queries, query_labels, pool, pool_labels, n_neighbors=5)
    assert np.allclose(values[:3], 1 / (20 / 30)) and np.allclose(values[3:5], 1 / (10 / 30)) and np.isnan(values[5])


def test_holdout_comparisons_cover_both_references():
    boots = {"everything@4": np.array([0.70, 0.71, 0.69]), "D+C features": np.array([0.76, 0.75, 0.77]),
             "D+C + groups (20)": np.array([0.77, 0.75, 0.78]), "idea@4": np.array([0.78, 0.79, 0.77])}
    points = {name: float(values.mean()) for name, values in boots.items()}
    table = improve_holdout.comparisons(points, boots)
    for name in boots:
        assert set(table[name]) == {"vs_everything4", "vs_DC"}
    assert np.isclose(table["idea@4"]["vs_DC"]["difference"], points["idea@4"] - points["D+C features"])
    assert np.isclose(table["D+C + groups (20)"]["vs_DC"]["difference"], 0.0 + points["D+C + groups (20)"] - points["D+C features"])
