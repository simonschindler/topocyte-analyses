"""The inputs of the Betti FPCR benchmark, and the stage that checks them: gate G5.

The loaders give every later stage the data of ``design.md`` ("The three
tasks"). Upstream's pickles are read only through ``safe_pickle.load_pickle``;
its ``.py`` files are read as text, never run, and where they hold a recipe or
a formula (the interpolation clouds, the 400 fluid simulations, their Reynolds
numbers and splits, the merged distance store) it is transcribed here.

The stage, from this folder (about 6 s)::

    $PY inputs.py > logs/inputs.log 2>&1

checks gate G5: the interpolation datasets equal their seed-0 regeneration;
every cloud's Rips diagrams, H0 and H1, equal ``ripser.ripser``'s as
birth-death pairs (10,500 interpolation and 100 curvature clouds); all 2,000
fluid diagrams are present. It then counts the fluid diagrams' bars and finds
their lowest birth, whose ceiling must be ``FLUID_SHIFT``, and their essential
bars, which must be one in H0 and two in H1 in every diagram. It writes
``results/inputs.json`` and exits with status 1 if a check fails.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import ripser
import topocyte as tc

import common as c
from safe_pickle import load_pickle

Diagram = tuple[np.ndarray, np.ndarray]  # (H0, H1), each (n, 2) float (birth, death); death inf for an essential bar

_INTERPOLATION = c.UPSTREAM / "interpolation" / "data" / "interpolation_dataset.pkl"
_CURVATURE = c.UPSTREAM / "curvature" / "data"
_FLUID = c.UPSTREAM / "fluid" / "data"

# Upstream's interpolation recipe, interpolation/data/interpolation_dataset_creation.py.
_INTERP_SEED = 0
_INTERP_POINTS = 20  # points per cloud, at theta = linspace(0, 2 pi, 20)
_INTERP_NOISE = 0.2  # each coordinate's noise is U(-0.2, 0.2)
# Upstream's fluid ensemble, fluid/fluid_data.py and fluid/distances.py.
_N_SIMULATIONS = 8000  # the ensemble's simulations, ids 0-7999
_N_FLUID = 400  # the simulations of upstream's SIM_IDS


def interpolation_datasets() -> list[dict]:
    """The 100 interpolation datasets of upstream's ``interpolation_dataset.pkl``, in its order.

    The order is training size ascending, then runs 0-9. Each dataset is a dict: ``n_train``, ``run``, ``clouds``
    (the ``n_train`` training clouds, then the 50 test clouds, each (20, 2)) and ``y`` (their targets t, in the same
    order).
    """
    return [
        {
            "n_train": ds["n_train"],
            "run": ds["run"],
            "clouds": [np.asarray(X, dtype=float) for X in [*ds["X_train"], *ds["X_test"]]],
            "y": np.concatenate((ds["y_train"], ds["y_test"])),
        }
        for ds in load_pickle(_INTERPOLATION)["datasets"]
    ]


def regenerate_interpolation() -> list[dict]:
    """The interpolation datasets made again by upstream's recipe, in the form of ``interpolation_datasets``.

    The recipe is ``interpolation_dataset_creation.py`` (read, not run). numpy's legacy global generator, seeded 0,
    draws for each training size and run in turn: the targets t ~ U(0, 1), training first, then for each t its
    cloud's 40 noise values U(-0.2, 0.2), the first 20 for x and the last 20 for y. The cloud's points are
    (cos θ + noise, (1 - t) cos θ + t sin θ + noise) at θ = linspace(0, 2π, 20). Reseeds the global generator.
    """
    np.random.seed(_INTERP_SEED)
    theta = np.linspace(0, 2 * np.pi, _INTERP_POINTS)
    datasets = []
    for n_train in c.INTERP_SIZES:
        for run in range(c.INTERP_RUNS):
            T = np.random.uniform(0.0, 1.0, n_train + c.INTERP_N_TEST)
            clouds = []
            for t in T:
                noise = np.random.uniform(-_INTERP_NOISE, _INTERP_NOISE, 2 * _INTERP_POINTS)
                x = np.cos(theta) + noise[:_INTERP_POINTS]
                y = (1 - t) * np.cos(theta) + t * np.sin(theta) + noise[_INTERP_POINTS:]
                clouds.append(np.array([x, y]).T)
            datasets.append({"n_train": n_train, "run": run, "clouds": clouds, "y": T})
    return datasets


def curvature_data() -> dict:
    """The curvature task: ``clouds`` (100), their curvatures K as ``y`` (100,), and upstream's ``train``/``test``.

    A cloud is (50, 3) on the sphere for K > 0 and (50, 2) in the Poincaré disk for K < 0 (``design.md``).
    ``train`` and ``test`` are upstream's index pickles; ``RuntimeError`` unless they are ``arange(50)`` and
    ``arange(50, 100)`` and there are 100 clouds and 100 targets.
    """
    clouds = [np.asarray(X, dtype=float) for X in load_pickle(_CURVATURE / "point_clouds.pkl")]
    y = np.asarray(load_pickle(_CURVATURE / "labels.pkl"), dtype=float)
    train = np.asarray(load_pickle(_CURVATURE / "train_indices.pkl"))
    test = np.asarray(load_pickle(_CURVATURE / "test_indices.pkl"))
    if not (
        len(clouds) == y.size == 100
        and np.array_equal(train, np.arange(50))
        and np.array_equal(test, np.arange(50, 100))
    ):
        raise RuntimeError(
            f"upstream's curvature data are not 100 clouds and targets split 0-49 / 50-99 (design.md): {len(clouds)} "
            f"clouds, {y.size} targets, train {train.tolist()}, test {test.tolist()}"
        )
    return {"clouds": clouds, "y": y, "train": train, "test": test}


def rips_diagram(X: np.ndarray) -> Diagram:
    """The H0 and H1 diagrams of the Rips filtration of the cloud ``X``, by topocyte, which runs ripser's defaults."""
    diagrams = tc.ph.compute_persistent_homology(X, filtration="rips", max_dim=1).diagrams
    return diagrams[0], diagrams[1]


def same_pairs(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two diagrams hold the same (birth, death) pairs: equal as rows sorted by birth, then death.

    Sorting whole rows keeps each pair together, so a re-pairing of the same births and deaths is seen; sorting
    births and deaths separately would not see it.
    """
    return bool(np.array_equal(a[np.lexsort((a[:, 1], a[:, 0]))], b[np.lexsort((b[:, 1], b[:, 0]))]))


def fluid_sim_ids() -> list[int]:
    """Upstream's ``SIM_IDS``: 400 of the 8,000 simulations, spread over the whole Reynolds-number range.

    Transcribed from ``fluid/distances.py``: the sorted union of 360 and of 50 evenly spaced ids, rounded, and from
    it 400 evenly spaced picks, rounded.
    """
    ids = sorted(
        {int(i) for i in np.linspace(0, _N_SIMULATIONS - 1, 360).round()}
        | {int(i) for i in np.linspace(0, _N_SIMULATIONS - 1, 50).round()}
    )
    return [ids[k] for k in np.linspace(0, len(ids) - 1, _N_FLUID).round().astype(int)]


def fluid_ln_re(ids: np.ndarray) -> np.ndarray:
    """ln Re of the simulations ``ids``, the fluid target: ``(4/3)(1 + i/1000) ln 2``.

    Upstream's ``fluid_data.reynolds_number`` gives Re = 2^(4 b / 3) with b = 1 + i/1000; this is its logarithm,
    the exponent computed as there.
    """
    b_max = 1.0 + np.asarray(ids, dtype=float) / 1000.0
    return 4.0 * b_max / 3.0 * np.log(2.0)


def fluid_split(s: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``s`` of the 400 simulations, as positions in ``fluid_sim_ids()``: ``(test, pool)``.

    Upstream's recipe (``fluid/run_fluid.py``): ``perm = np.random.default_rng(s).permutation(400)``, the test set
    ``perm[:100]`` and the pool ``perm[100:]``, whose first n positions are the training set of size n.
    """
    perm = np.random.default_rng(s).permutation(_N_FLUID)
    return perm[: c.FLUID_N_TEST], perm[c.FLUID_N_TEST :]


def _diagram_path(i: int, t: float) -> Path:
    """Upstream's cached diagram of simulation ``i`` at time ``t`` (``fluid/distances.py``, ``diagram_path``)."""
    return _FLUID / "diagrams" / f"pd_sim{i:04d}_t{t:g}.pkl"


def fluid_diagrams(t: float) -> list[Diagram]:
    """Upstream's diagrams of the 400 simulations at time ``t``, in ``fluid_sim_ids()`` order.

    Each is the sublevel diagram of the vorticity on the periodic cubical complex, essential bars kept, read from the
    DataFrame's Birth, Death and HomologyG columns. ``ValueError`` if a HomologyG is neither 0 nor 1.
    """
    diagrams = []
    for i in fluid_sim_ids():
        path = _diagram_path(i, t)
        frame = load_pickle(path)
        bars = frame[["Birth", "Death"]].to_numpy(dtype=float)
        h = frame["HomologyG"].to_numpy()
        if not np.isin(h, (0, 1)).all():
            raise ValueError(f"{path.name}: HomologyG holds {sorted(set(h.tolist()) - {0, 1})}, not only 0 and 1")
        diagrams.append((bars[h == 0], bars[h == 1]))
    return diagrams


def fluid_cached_w2(t: float, h: int) -> np.ndarray:
    """Upstream's cached W2 distances in dimension ``h`` between the 400 simulations at time ``t``.

    400 × 400 in ``fluid_sim_ids()`` order, symmetric, with a zero diagonal. The store is upstream's ``load_store``
    (``fluid/distances.py``): ``pairs_h{h}.pkl``, then its shards sorted by name, merged in that order; its keys are
    ``(a, b)``, the samples ``a < b`` as ``(sim_id, t)``. ``KeyError`` on a pair that the store lacks.
    """
    folder = _FLUID / "distances"
    store: dict = {}
    for path in [folder / f"pairs_h{h}.pkl", *sorted(folder.glob(f"pairs_h{h}_shard*.pkl"))]:
        store.update(load_pickle(path))
    samples = [(i, float(t)) for i in fluid_sim_ids()]
    D = np.zeros((len(samples), len(samples)))
    for p, a in enumerate(samples):
        for q in range(p + 1, len(samples)):
            b = samples[q]
            D[p, q] = D[q, p] = store[(a, b) if a < b else (b, a)]
    return D


def _same_dataset(a: dict, b: dict) -> bool:
    """Whether two interpolation datasets are equal: training size, run, and every target and cloud, exactly."""
    return (
        (a["n_train"], a["run"]) == (b["n_train"], b["run"])
        and np.array_equal(a["y"], b["y"])
        and len(a["clouds"]) == len(b["clouds"])
        and all(np.array_equal(p, q) for p, q in zip(a["clouds"], b["clouds"], strict=True))
    )


def _rips_as_ripser(X: np.ndarray) -> bool:
    """Whether ``rips_diagram(X)`` holds the pairs of upstream's call, ``ripser.ripser(X)["dgms"]``, in H0 and H1."""
    return all(same_pairs(a, b) for a, b in zip(rips_diagram(X), ripser.ripser(X)["dgms"], strict=True))


def _g5() -> dict:
    """Gate G5 (``design.md``): the interpolation regeneration, the Rips diagrams and the fluid diagrams' presence."""
    datasets, regenerated = interpolation_datasets(), regenerate_interpolation()
    regenerates = len(datasets) == len(regenerated) and all(map(_same_dataset, datasets, regenerated))
    print(f"G5: the {len(datasets)} interpolation datasets equal their seed-0 regeneration: {regenerates}", flush=True)
    started = time.perf_counter()
    clouds = {"interpolation": [X for ds in datasets for X in ds["clouds"]], "curvature": curvature_data()["clouds"]}
    differing = {
        task: [k for k, X in enumerate(task_clouds) if not _rips_as_ripser(X)] for task, task_clouds in clouds.items()
    }
    print(
        f"G5: Rips diagrams against ripser's as pairs, {' and '.join(f'{len(v):,} {k}' for k, v in clouds.items())} "
        f"clouds: {sum(map(len, differing.values()))} differ ({time.perf_counter() - started:.1f} s)",
        flush=True,
    )
    missing = [path.name for t in c.FLUID_TIMES for i in fluid_sim_ids() if not (path := _diagram_path(i, t)).exists()]
    expected = len(c.FLUID_TIMES) * _N_FLUID
    print(f"G5: fluid diagrams present: {expected - len(missing):,} of {expected:,}", flush=True)
    return {
        "passed": regenerates and not any(differing.values()) and not missing,
        "interpolation_datasets": len(datasets),
        "interpolation_equals_its_seed_0_regeneration": regenerates,
        "rips_clouds_compared": {task: len(task_clouds) for task, task_clouds in clouds.items()},
        "rips_clouds_differing_from_ripser": differing,
        "fluid_diagrams_expected": expected,
        "fluid_diagrams_missing": missing,
    }


def _fluid_summary() -> dict:
    """The fluid diagrams' bars, lowest birth and essential bars, with the checks of the shift and of the pattern."""
    bars, essential, lowest, empty = [], set(), math.inf, {}
    for t in c.FLUID_TIMES:
        diagrams = fluid_diagrams(t)
        empty[f"{t:g}"] = {f"h{h}": sum(not np.isfinite(d[h][:, 1]).any() for d in diagrams) for h in (0, 1)}
        for h0, h1 in diagrams:
            bars.append(len(h0) + len(h1))
            essential.add((int(np.isinf(h0[:, 1]).sum()), int(np.isinf(h1[:, 1]).sum())))
            lowest = min(lowest, float(np.min(np.concatenate((h0[:, 0], h1[:, 0])), initial=math.inf)))
        print(f"fluid t = {t:g}: {len(diagrams)} diagrams; without finite bars: {empty[f'{t:g}']}", flush=True)
    ceiling = math.ceil(-lowest)
    shift_ok, pattern_ok = ceiling == c.FLUID_SHIFT, essential == {(1, 2)}
    counts = {"min": min(bars), "median": float(np.median(bars)), "max": max(bars)}
    print(
        f"fluid: bars per diagram {counts}; lowest birth {lowest!r}, ceiling of minus it {ceiling} "
        f"(FLUID_SHIFT {c.FLUID_SHIFT}); essential (H0, H1) per diagram {sorted(essential)}",
        flush=True,
    )
    return {
        "passed": shift_ok and pattern_ok,
        "diagrams": len(bars),
        "bars_per_diagram": counts,
        "empty_finite_per_time": empty,
        "lowest_birth": lowest,
        "ceil_minus_lowest_birth": ceiling,
        "shift": c.FLUID_SHIFT,
        "shift_is_ceil_of_minus_lowest_birth": shift_ok,
        "essential_per_diagram": sorted(essential),
        "essential_is_one_h0_two_h1": pattern_ok,
    }


def main() -> None:
    c.check_topocyte()
    c.check_upstream()
    # at the start: the code that runs
    provenance = c.provenance(c.HERE / "inputs.py", c.HERE / "common.py", c.HERE / "safe_pickle.py")
    started = time.perf_counter()
    g5 = _g5()
    fluid = None if g5["fluid_diagrams_missing"] else _fluid_summary()  # the summary reads every diagram
    passed = g5["passed"] and fluid is not None and fluid["passed"]
    record = {
        "description": "the inputs of the three tasks and gate G5 (design.md): the interpolation datasets against "
        "their seed-0 regeneration, every cloud's Rips diagrams against ripser's as pairs, the 2,000 fluid diagrams' "
        "presence; then the fluid diagrams' bars, lowest birth (the shift) and essential bars",
        "passed": passed,
        "g5": g5,
        "fluid": fluid,
        "seconds": round(time.perf_counter() - started, 1),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance,
    }
    c.RESULTS.mkdir(exist_ok=True)
    (c.RESULTS / "inputs.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote results/inputs.json in {record['seconds']} s: {'passed' if passed else 'a check FAILED'}", flush=True)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
