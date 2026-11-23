#!/usr/bin/env python
"""Build the self-contained HTML report of the class-curve study from its saved results.

Run from this folder after ``verify_results.py``: ``PYTHONPATH=<topocyte checkout at 18e3aa5> <python>
build_report.py``. Reads ``results/{spike,gram,development,holdout,verify}.json`` (it checks that ``verify.json`` comes
from this folder's ``verify_results.py`` and that every check passed), ``results/{development,holdout,gram}.npz``, the
logs in ``logs/`` for the runs' time and memory (``/usr/bin/time -l``), the inputs as the stages read them
(``common.load_inputs``), the pathology study's ``pathology.csv`` for the slides' notes, and, for the mean curves, its
``density.npz`` and the cohort's cell classes (h5py mode "r"). It runs no stage and fits no classifier.

Two descriptive results no stage saved are computed here, and used by no verdict:

- each group's mean class curves on the development slides, counted exactly at displayed thresholds 25 cells per mm²
  apart: the one place a grid of thresholds is used, to draw the curves (the counts at the ten deciles are asserted
  equal to the cell-type study's ``composition.npz``);
- each block's fractions of variance, from ``folds.BlockFPCA`` fitted on all development slides.

Writes ``report.html``, with its figures embedded as 128-colour PNGs (the sibling reports' palette and look, the
classes in the liver cell viewer's colours), and nothing else. Under a minute.

No number in the prose is typed by hand: each is formatted from those files or from ``common``, and every sentence that
states a direction or an outcome is asserted against the numbers it states.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import html
import io
import json
import math
import re
import subprocess

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

import common as c  # noqa: E402
import folds  # noqa: E402

OUT = c.HERE / "report.html"
INK, INK2, MUTED, RULE, SURFACE, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # the dataviz palette's first two slots, validated as a pair: development, holdout
# The liver cell viewer's class colours on a light background (gtex-liver-chromatic-domains/viewer/static/palette.js).
CLASS_COLOR = {"epithelial": "#2a78d6", "stromal": "#1baf7a", "immune": "#eb6834", "other": "#52514e"}
STEP = 25.0  # cells per mm² between the displayed thresholds of the mean curves
TOP_LEVEL = 0.999  # the curves are counted up to the cohort's 99.9th density percentile, rounded up to a thousand
LINEAR_LEVEL = 0.99  # the linear plot ends at the 99th percentile, rounded up to a thousand
GROUPS = c.celltype.GROUP_ORDER
CONDITIONS = c.celltype.CONDITIONS
ARMS = ("curves", "curves4", "dc", "dc_unfiltered")
ARM_LABEL = {
    "curves": "curves, chosen k",
    "curves4": "curves, k = 4",
    "dc": "D+C, cleaned training slides",
    "dc_unfiltered": "D+C, all training slides",
}
COMPARISON_LABEL = {"curves - dc": "curves at the chosen k − D+C", "curves4 - dc": "curves at k = 4 − D+C"}
PRIMARY = "curves - dc"
STAGES = (  # (log, command, md5 of the script as the stage's json names it)
    ("spike.log", "curves.py spike", ("spike.json", ("script_md5",))),
    ("gram.log", "curves.py gram", ("gram.json", ("script_md5",))),
    ("development.log", "run_study.py development", ("development.json", ("md5", "run_study.py"))),
    ("readouts.log", "run_study.py readouts", ("development.json", ("readouts", "md5", "run_study.py"))),
    ("holdout.log", "run_study.py holdout (once)", ("holdout.json", ("md5", "run_study.py"))),
    ("verify.log", "verify_results.py", ("verify.json", ("script_md5",))),
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": CONTEXT,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)

STYLE = """
  :root { --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --rule: #e1e0d9; --surface: #fcfcfb;
          --page: #f9f9f7; --accent: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --ink: #f0efec; --ink2: #c3c2b7; --muted: #a3a29b; --rule: #383835;
                                      --surface: #1f1f1e; --page: #1a1a19; --accent: #3987e5; }
  }
  :root[data-theme="dark"] { --ink: #f0efec; --ink2: #c3c2b7; --muted: #a3a29b; --rule: #383835;
                             --surface: #1f1f1e; --page: #1a1a19; --accent: #3987e5; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink); line-height: 1.55; font-size: 16px;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1180px; margin: 0 auto; padding: 36px 16px 90px; }
  h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 1.3rem; margin: 44px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }
  h3 { font-size: 1.05rem; margin: 30px 0 8px; }
  p, li { max-width: 80ch; }
  .subtitle { color: var(--ink2); font-size: 1.05rem; margin: 0 0 8px; }
  .meta { color: var(--muted); font-size: 0.85rem; margin: 0 0 28px; }
  .summary { background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--accent);
             padding: 16px 20px; margin: 24px 0; border-radius: 4px; }
  .summary p { margin: 0 0 8px; } .summary p:last-child { margin: 0; }
  figure { margin: 22px 0; }
  figure img { width: 100%; height: auto; display: block; border-radius: 3px; }
  figcaption { color: var(--ink2); font-size: 0.85rem; margin-top: 8px; max-width: 110ch; }
  .table-wrap { overflow-x: auto; margin: 16px 0 6px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.82rem; font-variant-numeric: tabular-nums;
          background: var(--surface); }
  th, td { border-bottom: 1px solid var(--rule); padding: 7px 8px; text-align: right; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .notes th, .notes td { text-align: left; }
  .notes td:last-child { white-space: normal; min-width: 22rem; }
  .wrap1 th, .wrap1 td { white-space: normal; text-align: left; vertical-align: top; }
  .wrap1 td:first-child { min-width: 18rem; }
  .wrap1 td:last-child { font-size: 0.76rem; overflow-wrap: anywhere; }
  th { color: var(--ink2); font-weight: 600; vertical-align: bottom; white-space: normal; }
  .unit { color: var(--muted); font-weight: 400; font-size: 0.72rem; }
  .note { color: var(--ink2); font-size: 0.85rem; }
  code { font-size: 0.86em; background: var(--rule); padding: 1px 4px; border-radius: 3px; }
  details { margin: 14px 0; } summary { cursor: pointer; color: var(--ink2); }
  ul { margin: 8px 0 8px 22px; padding: 0; } li { margin: 6px 0; }
"""


# ---------------------------------------------------------------- formatting


def png_bytes(fig: plt.Figure, dpi: int = 110) -> bytes:
    """The figure as a 128-colour palette PNG (sibling reports)."""
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    palette = image.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    palette.save(out, format="PNG", optimize=True)
    return out.getvalue()


def figure(data: bytes, alt: str, caption: str) -> str:
    src = base64.b64encode(data).decode("ascii")
    image = f'<img src="data:image/png;base64,{src}" alt="{html.escape(alt)}">'
    return f"<figure>{image}<figcaption>{caption}</figcaption></figure>"


def table(header: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    inner = f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
    return f"<div class='table-wrap'><table class='{cls}'>{inner}</table></div>"


def num(value: float, digits: int = 3) -> str:
    """A number with a typographic minus and no "−0.000"."""
    return f"{0:.{digits}f}" if round(value, digits) == 0 else f"{value:.{digits}f}".replace("-", "−")


def signed(value: float, digits: int = 4) -> str:
    return f"{0:.{digits}f}" if round(value, digits) == 0 else f"{value:+.{digits}f}".replace("-", "−")


def interval(values: list[float], digits: int = 4) -> str:
    return f"[{signed(values[0], digits)}, {signed(values[1], digits)}]"


def pct(part: float, whole: float, digits: int = 1) -> str:
    return f"{100 * part / whole:.{digits}f}%"


def listing(items) -> str:
    """``a``, ``a or b``, ``a, b or c``."""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


def thousands(value: float, _position=None) -> str:
    return f"{value:,.0f}"


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(c.HERE), *args], capture_output=True, text=True, check=True).stdout.strip()


def resources(log: str) -> dict:
    """Wall time, peak RSS and finishing time of the run that ``/usr/bin/time -l`` closed the log with."""
    path = c.LOGS / log
    text = path.read_text()
    real = float(re.findall(r"([\d.]+) real", text)[-1])
    rss = int(re.findall(r"(\d+)\s+maximum resident set size", text)[-1])
    finished = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {"seconds": real, "rss_gb": rss / 1e9, "finished": finished}


def duration(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def lookup(record: dict, keys: tuple[str, ...]):
    for key in keys:
        record = record[key]
    return record


# ---------------------------------------------------------------- the descriptive computations


def mean_curves(inputs: dict) -> dict:
    """Each group's mean class curves over its development slides, counted exactly at the displayed thresholds.

    A slide's class curve at threshold ``t`` counts the class's cells with ``rho >= t`` (the viewer's count; the
    fPCA's curve counts ``rho > t``, which differs only at the cells' own densities). The thresholds run from 0 in
    steps of ``STEP`` to the cohort's ``TOP_LEVEL`` density quantile (``density.npz``'s ``curve``), rounded up to a
    thousand. Keys: ``thresholds``; ``top`` and ``top_linear``, the ``TOP_LEVEL`` and ``LINEAR_LEVEL`` quantiles;
    ``k_nn`` (the density's neighbours); ``deciles``; ``n`` (slides per group); ``counts`` and ``fractions`` (group ×
    class × threshold: the mean count, and the mean count over the slide's cells); and ``at_deciles`` (group × class ×
    decile, the mean count), whose per-slide counts equal ``kept_by_group``.
    """
    with np.load(c.SIB_DENSITY, allow_pickle=False) as density:
        rho, quantiles, levels, k_nn = density["rho"], density["curve"], density["curve_levels"], int(density["k_nn"])
    classes, _ = c.celltype.read_cell_groups()
    top, top_linear = (float(quantiles[np.flatnonzero(np.isclose(levels, q))[0]]) for q in (TOP_LEVEL, LINEAR_LEVEL))
    thresholds = np.arange(0.0, math.ceil(top / 1000) * 1000 + STEP / 2, STEP)
    bounds, deciles = inputs["bounds"], inputs["deciles"]
    dev = np.flatnonzero(inputs["dev"])
    counts = np.zeros((dev.size, len(c.CLASSES), thresholds.size))
    at_deciles = np.zeros((deciles.size, dev.size, len(c.CLASSES)), dtype=np.int64)
    for i, s in enumerate(dev):
        slide_rho, slide_classes = rho[bounds[s] : bounds[s + 1]], classes[bounds[s] : bounds[s + 1]]
        for k in range(len(c.CLASSES)):
            ordered = np.sort(slide_rho[slide_classes == k])
            counts[i, k] = ordered.size - np.searchsorted(ordered, thresholds, side="left")
            at_deciles[:, i, k] = ordered.size - np.searchsorted(ordered, deciles, side="left")
    with np.load(c.CELLTYPE / "results/composition.npz", allow_pickle=False) as composition:
        assert np.array_equal(at_deciles, composition["kept_by_group"][:, dev]), "the curves at the deciles are not C's"
    cells = np.diff(bounds)[dev].astype(float)
    y = inputs["labels"][dev]
    return {
        "thresholds": thresholds,
        "top": top,
        "top_linear": top_linear,
        "k_nn": k_nn,
        "deciles": deciles,
        "n": np.bincount(y, minlength=len(GROUPS)),
        "counts": np.stack([counts[y == g].mean(axis=0) for g in range(len(GROUPS))]),
        "fractions": np.stack([(counts[y == g] / cells[y == g, None, None]).mean(axis=0) for g in range(len(GROUPS))]),
        "at_deciles": np.stack([at_deciles[:, y == g].mean(axis=1).T for g in range(len(GROUPS))]),
    }


def variance_fractions(grams: dict[str, np.ndarray], dev: np.ndarray) -> dict[str, np.ndarray]:
    """Each block's fractions of variance, its first ``N_COMPONENTS``, from ``BlockFPCA`` on all development slides."""
    out = {}
    for name in c.CLASSES:
        for normalization in c.NORMALIZATIONS:
            K = folds.analysis_kernel(grams[name], normalization)
            out[f"{name} {normalization}"] = folds.BlockFPCA().fit(K[np.ix_(dev, dev)]).explained_variance_ratio_
    return out


# ---------------------------------------------------------------- figures


def fig_mean_curves(curves: dict, key: str, ylabel: str, log: bool, x_max: float) -> bytes:
    """Small multiples, one per group: its mean class curves over the density threshold up to ``x_max``, the deciles
    faint."""
    t = curves["thresholds"]
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), sharex=True, sharey=True, constrained_layout=True)
    for g, ax in enumerate(axes.flat):
        for d in curves["deciles"][1:]:
            ax.axvline(d, color=RULE, lw=0.8, zorder=0)
        keep = t <= x_max
        for k, name in enumerate(c.CLASSES):
            values = curves[key][g, k]
            values = np.where(values > 0, values, np.nan) if log else values
            ax.plot(
                t[keep], values[keep], color=CLASS_COLOR[name], lw=2, solid_capstyle="round", solid_joinstyle="round"
            )
        ax.set_title(f"{GROUPS[g]} ({curves['n'][g]} slides)", loc="left", color=INK, fontsize=9)
        ax.grid(axis="y", alpha=0.6)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(FuncFormatter(thousands))
            ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_xlim(0, x_max)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 3, 5, 10]))
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(thousands))
    for ax in axes[1]:
        ax.set_xlabel("density threshold t (cells per mm²)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=CLASS_COLOR[name], lw=2) for name in c.CLASSES]
    fig.legend(handles, c.CLASSES, loc="outside upper center", ncol=len(c.CLASSES), frameon=False)
    return png_bytes(fig)


def fig_aucs(readouts: dict, hold_json: dict) -> bytes:
    """Left: each arm's macro AUC (development averaged over the repeats, each repeat, holdout). Right: the paired
    differences against D+C with their 95% intervals."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]}
    )
    for i, arm in enumerate(ARMS):
        row = len(ARMS) - 1 - i
        per = readouts["arms"][arm]["auc_per_repeat"]
        left.scatter(per, np.full(len(per), row - 0.22), s=12, color=CONTEXT, zorder=2, lw=0)
        left.plot([np.mean(per)] * 2, [row - 0.34, row - 0.10], color=INK2, lw=1.5, zorder=3)
        left.scatter([readouts["arms"][arm]["macro_auc"]], [row], s=56, color=SERIES[0], ec=SURFACE, lw=1.5, zorder=4)
        left.scatter(
            [hold_json["arms"][arm]["macro_auc"]],
            [row],
            s=56,
            marker="D",
            color=SERIES[1],
            ec=SURFACE,
            lw=1.5,
            zorder=4,
        )
    left.set_yticks(range(len(ARMS)), [ARM_LABEL[arm] for arm in ARMS][::-1])
    left.set_ylim(-0.6, len(ARMS) - 0.5)
    left.set_xlabel("macro AUC over the seven conditions")
    left.grid(axis="x", alpha=0.6)
    left.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=SERIES[0], ms=7, label="development, repeats averaged"),
            Line2D([], [], marker="D", ls="", color=SERIES[1], ms=6, label="holdout, scored once"),
            Line2D([], [], marker="o", ls="", color=CONTEXT, ms=4, label="each development repeat"),
            Line2D([], [], marker="|", ls="", color=INK2, ms=9, mew=1.5, label="their mean"),
        ],
        frameon=False,
        fontsize=7.8,
        loc="outside upper center",
        ncol=4,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    shown = [v for arm in ARMS for v in (*readouts["arms"][arm]["auc_per_repeat"], hold_json["arms"][arm]["macro_auc"])]
    left.set_xlim(min(shown) - 0.006, max(shown) + 0.006)
    names = list(COMPARISON_LABEL)
    for i, name in enumerate(names):
        row = len(names) - 1 - i
        for offset, color, marker, section in (
            (0.12, SERIES[0], "o", readouts["comparisons"]),
            (-0.12, SERIES[1], "D", hold_json["comparisons"]),
        ):
            diff, (lo, hi) = section[name]["difference"], section[name]["interval"]
            right.errorbar(
                [diff],
                [row + offset],
                xerr=[[diff - lo], [hi - diff]],
                fmt=marker,
                ms=6.5 if marker == "o" else 5.5,
                color=color,
                mec=SURFACE,
                ecolor=color,
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
    right.axvline(0, color=INK2, lw=1, zorder=1)
    right.set_yticks(range(len(names)), [COMPARISON_LABEL[name] for name in names][::-1])
    right.set_ylim(-0.6, len(names) - 0.4)
    right.set_xlabel("difference in macro AUC, with its 95% interval")
    right.grid(axis="x", alpha=0.6)
    right.tick_params(axis="y", length=0)
    return png_bytes(fig)


# ---------------------------------------------------------------- the report


def main() -> None:
    c.check_topocyte()
    inputs = c.load_inputs()
    results = {
        name: json.loads((c.RESULTS / f"{name}.json").read_text())
        for name in ("spike", "gram", "development", "holdout", "verify")
    }
    spike, gram_json, dev_json, hold_json, verify = results.values()
    readouts = dev_json["readouts"]
    assert verify["all_ok"] and all(check["ok"] for check in verify["checks"]), "a check of verify_results.py failed"
    assert verify["script_md5"] == md5(c.HERE / "verify_results.py"), "verify.json is not from this verify_results.py"
    for name in ("run_study.py", "folds.py", "common.py"):  # the readouts and the holdout ran the current code
        assert readouts["md5"][name] == hold_json["md5"][name] == md5(c.HERE / name)
    assert list(readouts["arms"]) == list(hold_json["arms"]) == list(ARMS)
    ids, labels = inputs["slide_ids"], inputs["labels"]
    dev, hold = np.flatnonzero(inputs["dev"]), np.flatnonzero(inputs["holdout"])
    y = labels[dev]
    with np.load(c.RESULTS / "development.npz", allow_pickle=False) as saved:
        split, dropped, chosen = saved["folds"], saved["dropped"], saved["k"]
        assert [str(s) for s in saved["slide_ids"]] == [ids[i] for i in dev]
        repeat_sd = {arm: float(np.std(saved[arm], axis=0, ddof=1).mean()) for arm in ARMS}
    with np.load(c.RESULTS / "holdout.npz", allow_pickle=False) as saved:
        hold_dropped, hold_k = saved["dropped"], int(saved["k"])
        assert [str(s) for s in saved["dev_slide_ids"]] == [ids[i] for i in dev]
    with np.load(c.RESULTS / "gram.npz", allow_pickle=False) as saved:
        assert [str(s) for s in saved["slide_ids"]] == ids
        grams = {name: saved[f"gram_{name}"] for name in c.CLASSES}
    pathology = pd.read_csv(c.SIB_DENSITY.parent / "pathology.csv", keep_default_na=False)
    assert list(pathology["slide_id"]) == ids and [GROUPS[g] for g in labels] == list(pathology["group"])
    curves = mean_curves(inputs)
    fractions = variance_fractions(grams, dev)

    n_dev, n_hold, n_all = dev.size, hold.size, len(ids)
    n_repeats, n_splits = split.shape[0], dropped.shape[1]
    n_folds = n_repeats * n_splits
    dev_n = np.bincount(y, minlength=len(GROUPS))
    hold_n = np.bincount(labels[hold], minlength=len(GROUPS))
    assert dict(zip(GROUPS, hold_n.tolist(), strict=True)) == hold_json["holdout_per_group"]

    # ---------------- the readouts and the verdict
    arms_dev, arms_hold = readouts["arms"], hold_json["arms"]
    auc = {arm: arms_dev[arm]["macro_auc"] for arm in ARMS}
    hauc = {arm: arms_hold[arm]["macro_auc"] for arm in ARMS}
    rep_mean = {arm: arms_dev[arm]["auc_repeat_mean"] for arm in ARMS}
    rep_sd = {arm: arms_dev[arm]["auc_repeat_sd"] for arm in ARMS}
    dev_cmp, hold_cmp = readouts["comparisons"], hold_json["comparisons"]
    diff, (lo, hi) = dev_cmp[PRIMARY]["difference"], dev_cmp[PRIMARY]["interval"]
    hdiff, (hlo, hhi) = hold_cmp[PRIMARY]["difference"], hold_cmp[PRIMARY]["interval"]
    diff4, (lo4, hi4) = dev_cmp["curves4 - dc"]["difference"], dev_cmp["curves4 - dc"]["interval"]
    hdiff4, (hlo4, hhi4) = hold_cmp["curves4 - dc"]["difference"], hold_cmp["curves4 - dc"]["interval"]
    verdict = hold_json["verdict"]["verdict"]
    assert verdict == "level with D+C" and lo <= 0 <= hi, "the prose below states the level outcome"
    assert (
        hold_json["verdict"]["development_interval"] == [lo, hi] and hold_json["verdict"]["holdout_difference"] == hdiff
    )
    assert diff < 0 and hdiff < 0 and auc["curves"] < auc["dc"] and hauc["curves"] < hauc["dc"]
    assert lo4 <= 0 <= hi4 and diff4 < diff  # k = 4: its interval contains 0 too, and it trails the chosen k
    assert hlo4 <= 0 <= hhi4 and hlo <= 0 <= hhi
    gate_dev, gate_hold = dev_json["dc_gate"]["max_abs_difference"], hold_json["gate"]["max_abs_difference"]
    assert max(gate_dev, gate_hold) <= 1e-12 and readouts["gate"]["macro_auc"] == auc["dc_unfiltered"]
    assert hold_json["gate"]["macro_auc"] == hauc["dc_unfiltered"]
    filter_cost, filter_gain = auc["dc_unfiltered"] - auc["dc"], hauc["dc"] - hauc["dc_unfiltered"]
    assert filter_cost > 0 and filter_gain > 0  # the filter cost D+C on development and helped it on the holdout
    per_repeat_gap = rep_mean["curves"] - rep_mean["dc"]
    gain = {arm: auc[arm] - rep_mean[arm] for arm in ARMS}
    assert per_repeat_gap < diff < 0 and all(value > 0 for value in gain.values())
    assert gain["curves"] == max(gain.values()) and repeat_sd["curves"] == max(repeat_sd.values())
    assert repeat_sd["curves"] > repeat_sd["dc"]

    # ---------------- the drops
    trained = split[:, None, :] != np.arange(n_splits)[None, :, None]  # (repeat, fold, slide): a training slide there
    assert not (dropped & ~trained).any()
    places, n_dropped = int(trained.sum()), int(dropped.sum())
    per_fold, n_train = dropped.sum(axis=2), trained.sum(axis=2)
    share_fold = per_fold / n_train
    kept_fold = n_train - per_fold
    group_places = np.array([trained[..., y == g].sum() for g in range(len(GROUPS))])
    group_dropped = np.array([dropped[..., y == g].sum() for g in range(len(GROUPS))])
    group_share = group_dropped / group_places
    drops, trains = dropped.sum(axis=(0, 1)), trained.sum(axis=(0, 1))
    assert len(set(trains.tolist())) == 1  # every slide trains in the same number of outer folds
    n_trains = int(trains[0])
    never, ever, always = int(np.sum(drops == 0)), int(np.sum(drops > 0)), int(np.sum(drops == trains))
    smallest_kept = np.array(
        [
            [np.sum(trained[r, f] & ~dropped[r, f] & (y == g)) for g in range(len(GROUPS))]
            for r, f in np.ndindex(trained.shape[:2])
        ]
    ).min(axis=0)
    assert smallest_kept.min() >= n_splits  # every group can fill the inner folds
    hold_drop_group = np.array([np.sum(hold_dropped & (y == g)) for g in range(len(GROUPS))])
    assert hold_drop_group.tolist() == list(hold_json["fit"]["dropped_per_group"].values())
    n_hold_dropped = int(hold_dropped.sum())
    hold_always, hold_ever = int(np.sum(hold_dropped & (drops == trains))), int(np.sum(hold_dropped & (drops > 0)))
    assert hold_ever == n_hold_dropped and hold_always == always  # the holdout fit dropped every always-dropped slide
    flag = pathology["quality_flag"].to_numpy()[dev]
    flagged_ever, flagged_never = int(np.sum((flag != "") & (drops > 0))), int(np.sum((flag != "") & (drops == 0)))
    low, high = int(np.argmin(group_share)), int(np.argmax(group_share))

    # ---------------- the chosen k
    k_values, k_counts = np.unique(chosen, return_counts=True)
    assert {str(k): int(n) for k, n in zip(k_values, k_counts, strict=True)} == dev_json["chosen_k"]
    assert hold_k == hold_json["fit"]["chosen_k"]
    k_never = [k for k in c.K_GRID if k not in k_values]
    k_top = sorted(k_values[np.argsort(-k_counts, kind="stable")[:2]].tolist())
    k_top_share = int(k_counts[np.isin(k_values, k_top)].sum())

    # ---------------- the curves, described
    share0 = curves["fractions"][:, :, 0]  # each group's mean class shares at t = 0 (every cell)
    epi, strom = c.CLASSES.index("epithelial"), c.CLASSES.index("stromal")
    cirrhosis, healthy = GROUPS.index("cirrhosis"), GROUPS.index("healthy")
    assert int(np.argmin(share0[:, epi])) == cirrhosis and int(np.argmax(share0[:, strom])) == cirrhosis
    raw = [fractions[f"{name} none"] for name in c.CLASSES]
    cosine = [fractions[f"{name} cosine"] for name in c.CLASSES]
    first_raw, first_cos = [f[0] for f in raw], [f[0] for f in cosine]
    assert min(first_raw) > max(first_cos)  # the raw blocks' first component holds more than any cosine block's
    nine = [float(f.sum()) for f in fractions.values()]
    assert math.isclose(c.MAD_FACTOR, 3 * 1.4826)  # as section 1 states it

    # ---------------- the runs
    runs = {log: resources(log) for log, _, _ in STAGES}
    built = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    head = git("log", "-1", "--format=%h", "--", ".") or "uncommitted"
    versions = {**dev_json["versions"], "h5py": gram_json["versions"]["h5py"], "matplotlib": matplotlib.__version__}
    assert dev_json["versions"] == readouts["versions"] == hold_json["versions"]

    # ---------------- summary
    group_drops = ", ".join(f"{html.escape(g)} {pct(group_dropped[i], group_places[i])}" for i, g in enumerate(GROUPS))
    hold_drops = ", ".join(f"{html.escape(g)} {hold_drop_group[i]} of {dev_n[i]}" for i, g in enumerate(GROUPS))
    summary = f"""
<p><strong>Verdict, by the rule fixed before any result: {html.escape(verdict)}.</strong> On the {n_dev} development
slides ({n_repeats} repeats of stratified {n_splits}-fold), the fPCA of the class curves at the chosen k reaches macro
AUC {num(auc["curves"])} against {num(auc["dc"])} for D+C trained on the same cleaned slides: a difference of
{signed(diff)}, 95% interval {interval([lo, hi])}, which contains 0. On the {n_hold}-slide holdout, scored once, the
curves reach {num(hauc["curves"])} against {num(hauc["dc"])} ({signed(hdiff)}, {interval([hlo, hhi])}); with the
development interval containing 0, the rule does not read the holdout, which leans the same way.</p>
<p><strong>The outlier filter dropped {pct(n_dropped, places)} of the training slides</strong> in the development
folds: {n_dropped:,} of {places:,} training-slide places in the {n_folds} outer folds, {per_fold.min()}–{per_fold.max()}
per fold. By group: {group_drops}. <strong>The holdout fit dropped {n_hold_dropped} of the {n_dev} development
slides</strong> ({pct(n_hold_dropped, n_dev)}): {hold_drops}. Both arms of the comparison train on the same cleaned
slides; the test slides are always scored.</p>
<p><strong>Per repeat, the gap is larger.</strong> The pre-registered readout scores the out-of-fold probabilities
averaged over the {n_repeats} repeats. Scored repeat by repeat, the curves reach {num(rep_mean["curves"])} ±
{num(rep_sd["curves"])} against {num(rep_mean["dc"])} ± {num(rep_sd["dc"])} for D+C (mean ± SD over the repeats). The
curves' probabilities vary more from repeat to repeat (a slide's probability has an SD over the repeats of
{num(repeat_sd["curves"])} on average, D+C's {num(repeat_sd["dc"])}), and averaging raises their macro AUC by
{num(gain["curves"])}, D+C's by {num(gain["dc"])}.</p>
<p>Beside the verdict: the curves at k = 4 reach {num(auc["curves4"])} ({signed(diff4)}, {interval([lo4, hi4])}) on
the development slides and {num(hauc["curves4"])} on the holdout ({signed(hdiff4)}, {interval([hlo4, hhi4])}). The
filter cost D+C {num(filter_cost)} of development macro AUC ({num(auc["dc"])} against {num(auc["dc_unfiltered"])}
unfiltered) and gained it {num(filter_gain)} on the holdout ({num(hauc["dc"])} against
{num(hauc["dc_unfiltered"])}).</p>
"""

    # ---------------- section 1: the question and the rulings
    group_rows = [[html.escape(g), str(dev_n[i]), str(hold_n[i])] for i, g in enumerate(GROUPS)]
    group_rows.append(["all", str(n_dev), str(n_hold)])
    k_grid = ", ".join(str(k) for k in c.K_GRID)
    question = f"""
<p>Simon's question (2026-11-23): does a functional PCA of each slide's class curves separate the liver pathology
groups better than D+C? D+C is the cell-type study's representation to beat: {inputs["DC"].shape[1]} features, the
slide's density summaries and its cells of each class at each of the ten density deciles (macro AUC
{num(hold_json["gate"]["macro_auc"])} on that study's {n_hold}-slide holdout).</p>
<p>A slide's <em>class curve</em> counts its cells of one class ({", ".join(c.CLASSES)}) whose
{curves["k_nn"]}-nearest-neighbour density is at or above a threshold t, for every t in cells per mm²; the liver cell
viewer draws them. At the ten deciles the four curves give C's counts, and their sum D's kept shares. The curve is the
reduced Betti curve of the bars (0, ρ_i), topocyte's <code>BettiCurve</code> of dimension 0, so the L2 inner product of
two curves is exactly Σ_i Σ_j min(ρ_i, ρ_j): each class's {n_all} × {n_all} Gram matrix is computed once, exactly, and
every fold's fPCA is linear algebra on it. No curve is evaluated on a grid in any computation.</p>
<p>Simon's rulings (design.md, committed before any result):</p>
<ul>
<li><strong>Groups:</strong> the {len(GROUPS)} pathology groups of the pathology study; the macro AUC scores the
{len(CONDITIONS)} conditions, not unclassified.</li>
<li><strong>Outliers</strong> are filtered on the training slides only, inside each fold: in each of the
{len(c.CLASSES) * len(c.NORMALIZATIONS)} blocks (four classes × raw and cosine), a training slide's squared distance to
its group's mean curve, divided by the block's median, summed over the blocks; a slide is dropped above its group's
median + 3 × 1.4826 × MAD (the MAD scaled to a normal SD). The test slides are always scored, and D+C trains on the
same cleaned slides.</li>
<li><strong>Components:</strong> best against best decides. The number k of MFA components of the curves is chosen by
inner stratified {n_splits}-fold cross-validation on the cleaned training slides from {k_grid}; k = 4 is reported
beside it.</li>
<li><strong>The rule, fixed in advance:</strong> beats D+C if the development 95% interval of curves at the chosen k
minus D+C lies above 0 and the holdout difference is positive; not confirmed if the interval lies above 0 and the
holdout difference does not; level with D+C if the interval contains 0; below D+C if it lies below 0.</li>
</ul>
{table(["group", "development slides", "holdout slides"], group_rows)}
"""

    # ---------------- section 2: the mean curves
    decile_rows = []
    for g, group in enumerate(GROUPS):
        for k, name in enumerate(c.CLASSES):
            decile_rows.append([f"{html.escape(group)}: {name}", *(f"{v:,.0f}" for v in curves["at_deciles"][g, k])])
    decile_header = [
        "group: class",
        *(
            f"d{j} <span class='unit'>{v:,.0f}</span>" if v >= 10 else f"d{j} <span class='unit'>{v:.2f}</span>"
            for j, v in enumerate(curves["deciles"])
        ),
    ]
    curves_section = f"""
<p>Each panel shows one group's mean class curves over its development slides (the holdout slides stay out of every
description). The displayed thresholds run from 0 to {curves["thresholds"][-1]:,.0f} cells per mm² in steps of
{STEP:.0f}, past the cohort's {100 * TOP_LEVEL:.1f}th density percentile ({curves["top"]:,.0f}); each count is exact
at its threshold. A grid of thresholds is used only here, to draw the curves. The faint vertical lines are the cohort's
deciles d1–d9, where each slide's curves are C's counts (asserted against the cell-type study's
<code>composition.npz</code>).</p>
{
        figure(
            fig_mean_curves(curves, "counts", "mean cells per slide at or above t", True, curves["thresholds"][-1]),
            "Mean class curves per group, counts, log scale",
            "The mean over the group's development slides of the cells of each class at or above the threshold t, on "
            "a log scale, which shows the four classes and the high-density tails alike.",
        )
    }
{
        figure(
            fig_mean_curves(
                curves,
                "fractions",
                "mean share of the slide's cells",
                False,
                math.ceil(curves["top_linear"] / 1000) * 1000,
            ),
            "Mean class curves per group, divided by the slide's cells",
            "The same curves, each slide's divided by its cell count before the mean (the liver cell viewer's stacked "
            f"view does the same for one slide), on a linear scale, up to the cohort's {100 * LINEAR_LEVEL:.0f}th "
            f"density percentile ({curves['top_linear']:,.0f}) rounded up to a thousand.",
        )
    }
<p>At t = 0 every cell counts: cirrhosis has the smallest share of epithelial cells
({num(share0[cirrhosis, epi], 2)} of a slide's cells, against {num(share0[healthy, epi], 2)} in healthy) and the
largest of stromal cells ({num(share0[cirrhosis, strom], 2)}, against {num(share0[healthy, strom], 2)}).</p>
<details><summary>the mean curves at the deciles d0–d9 (cells per slide; d0 is the cohort's minimum density, so it
counts every cell)</summary>{table(decile_header, decile_rows)}</details>
"""

    # ---------------- section 3: the drops
    drop_rows = []
    for i, g in enumerate(GROUPS):
        members = y == i
        drop_rows.append(
            [
                html.escape(g),
                str(dev_n[i]),
                f"{group_places[i]:,}",
                f"{group_dropped[i]:,}",
                pct(group_dropped[i], group_places[i]),
                str(int(np.sum(drops[members] > 0))),
                str(int(np.sum(drops[members] == trains[members]))),
                f"{hold_drop_group[i]} of {dev_n[i]}",
            ]
        )
    drop_rows.append(
        [
            "all",
            str(n_dev),
            f"{places:,}",
            f"{n_dropped:,}",
            pct(n_dropped, places),
            str(ever),
            str(always),
            f"{n_hold_dropped} of {n_dev}",
        ]
    )
    order = sorted(np.flatnonzero(drops > 0), key=lambda i: (-drops[i], y[i], ids[dev[i]]))
    slide_rows = [
        [
            html.escape(ids[dev[i]]),
            html.escape(GROUPS[y[i]]),
            f"{drops[i]} of {n_trains}",
            "dropped" if hold_dropped[i] else "",
            html.escape(flag[i]),
            html.escape(pathology["notes"].iloc[dev[i]]),
        ]
        for i in order
    ]
    drops_section = f"""
<p>In every outer fold the filter reads only the training slides' labels. Over the {n_folds} outer folds it dropped
{n_dropped:,} of the {places:,} training-slide places ({pct(n_dropped, places)}): {per_fold.min()}–{per_fold.max()}
per fold (median {np.median(per_fold):.0f}) of {n_train.min()}–{n_train.max()} training slides
({pct(share_fold.min(), 1)}–{pct(share_fold.max(), 1)}), leaving {kept_fold.min()}–{kept_fold.max()}. The smallest
group, {html.escape(GROUPS[int(np.argmin(smallest_kept))])}, kept at least {smallest_kept.min()} of its training slides
in every fold, enough for the inner {n_splits}-fold. By group the share dropped runs from {
        pct(group_dropped[low], group_places[low])
    }
({html.escape(GROUPS[low])}) to {pct(group_dropped[high], group_places[high])} ({html.escape(GROUPS[high])}).</p>
<p>Each development slide trains in {n_trains} of the {n_folds} outer folds. Of the {n_dev}, {never} were never
dropped, {ever} at least once and {always} in all {n_trains}. The holdout fit, on all {n_dev} development slides,
dropped {n_hold_dropped} ({pct(n_hold_dropped, n_dev)}): all {always} that every development fold dropped, and
{n_hold_dropped - always} of the others dropped at least once. {flagged_ever} of the {ever} slides ever dropped carry
the pathology study's quality flag, against {flagged_never} of the {never} never dropped.</p>
{
        table(
            [
                "group",
                "development slides",
                "training-slide places",
                "dropped",
                "share",
                "slides ever dropped",
                f"dropped in all {n_trains}",
                "holdout fit: dropped",
            ],
            drop_rows,
        )
    }
<h3>The {ever} slides dropped at least once</h3>
<p>Ordered by how often they were dropped; "holdout fit" marks the {n_hold_dropped} the holdout fit dropped. Notes and
flags are the pathology study's (<code>pathology.csv</code>).</p>
{
        table(
            ["slide", "group", "dropped in the folds it trained in", "holdout fit", "quality flag", "pathology notes"],
            slide_rows,
            "notes",
        )
    }
"""

    # ---------------- section 4: the fractions of variance
    variance_rows = [
        [html.escape(block), *(f"{100 * v:.2f}" for v in values), f"{100 * values.sum():.2f}"]
        for block, values in fractions.items()
    ]
    variance_section = f"""
<p>Each block's fPCA fitted on all {n_dev} development slides (<code>folds.BlockFPCA</code>, as in every fold), for
description only: the folds fit their own. A raw block's first component holds
{pct(min(first_raw), 1)}–{pct(max(first_raw), 1)} of the block's variance, a cosine block's
{pct(min(first_cos), 1)}–{pct(max(first_cos), 1)}; the first {c.N_COMPONENTS} hold at least {pct(min(nine), 1, 2)} in
every block. The MFA divides each block by the SD of its first component and concatenates the
{len(fractions) * c.N_COMPONENTS} columns.</p>
{
        table(
            [
                "block (class, normalization)",
                *(f"PC{j + 1} <span class='unit'>%</span>" for j in range(c.N_COMPONENTS)),
                f"first {c.N_COMPONENTS} <span class='unit'>%</span>",
            ],
            variance_rows,
        )
    }
"""

    # ---------------- section 5: the chosen k
    k_rows = [[str(k), str(n), pct(n, n_folds, 0)] for k, n in zip(k_values, k_counts, strict=True)]
    k_section = f"""
<p>The inner cross-validation chose k in each outer fold from {k_grid}, the highest pooled inner macro AUC winning,
ties to the smaller k. It chose k = {k_top[0]} or {k_top[1]} in {k_top_share} of the {n_folds} outer folds, and never
{listing(k_never)}. The holdout fit, on all {n_dev} development slides (inner
seed {hold_json["fit"]["inner_seed"]}), chose k = {hold_k}.</p>
{table(["k", "outer folds", "share"], k_rows)}
"""

    # ---------------- section 6: the AUCs
    auc_rows = [
        [
            ARM_LABEL[arm],
            num(auc[arm]),
            f"{num(rep_mean[arm])} ± {num(rep_sd[arm])}",
            num(repeat_sd[arm], 4),
            num(hauc[arm]),
        ]
        for arm in ARMS
    ]
    condition_rows = [
        [
            html.escape(condition),
            *(num(arms_dev[arm]["auc_per_condition"][condition]) for arm in ARMS),
            *(num(arms_hold[arm]["auc_per_condition"][condition]) for arm in ARMS),
        ]
        for condition in CONDITIONS
    ]
    cmp_rows = [
        [
            COMPARISON_LABEL[name],
            signed(dev_cmp[name]["difference"]),
            interval(dev_cmp[name]["interval"]),
            signed(hold_cmp[name]["difference"]),
            interval(hold_cmp[name]["interval"]),
        ]
        for name in COMPARISON_LABEL
    ]
    auc_section = f"""
{
        figure(
            fig_aucs(readouts, hold_json),
            "Macro AUC per arm and the paired differences against D+C",
            f"Left: each arm's macro AUC over the {len(CONDITIONS)} conditions; development from the out-of-fold "
            f"probabilities averaged over the {n_repeats} repeats (the pre-registered readout), each repeat's in grey "
            "with their mean; holdout scored once. Right: the paired differences against D+C trained on the same "
            f"cleaned slides, with 95% intervals over {readouts['bootstrap']['draws']:,} bootstrap draws of the "
            f"slides stratified by group (seed {readouts['bootstrap']['seed']}).",
        )
    }
{
        table(
            [
                "arm",
                "development <span class='unit'>repeats averaged</span>",
                "development <span class='unit'>per repeat, mean ± SD</span>",
                "probabilities' SD over the repeats <span class='unit'>mean over slides and groups</span>",
                "holdout",
            ],
            auc_rows,
        )
    }
<p>The four arms differ only in their features and training slides: the same logistic regression (the cell-type
study's: multinomial L2, C = 1, balanced class weights, features standardized on the training slides) predicts the
same test slides. The unfiltered D+C reproduces the cell-type study's improvement phase: its probabilities equal that
phase's (largest difference {gate_dev:g} on the development slides, {gate_hold:g} on the holdout), and so do its macro
AUCs, {num(readouts["gate"]["macro_auc"], 4)} and {num(hold_json["gate"]["macro_auc"], 4)}.</p>
{
        table(
            ["comparison", "development", "95% interval", "holdout", "95% interval"],
            cmp_rows,
        )
    }
<p class="note">{html.escape(hold_json["note"])}</p>
<h3>Each condition's AUC</h3>
{
        table(
            [
                "condition",
                *(f"{ARM_LABEL[arm]} <span class='unit'>development</span>" for arm in ARMS),
                *(f"{ARM_LABEL[arm]} <span class='unit'>holdout</span>" for arm in ARMS),
            ],
            condition_rows,
        )
    }
"""

    # ---------------- section 7: the verdict
    verdict_section = f"""
<p><strong>{html.escape(verdict[0].upper() + verdict[1:])}.</strong> The rule reads the development interval of curves
at the chosen k minus D+C, {interval([lo, hi])}, which contains 0: the curves' fPCA neither beats D+C nor falls below
it on the development slides. The holdout difference enters the rule only when the interval lies above 0; it was
{signed(hdiff)} ({interval([hlo, hhi])}).</p>
<p>The rule as <code>holdout.json</code> records it: {html.escape(hold_json["verdict"]["rule"])}</p>
"""

    # ---------------- section 8: methods and run
    run_rows = []
    for log, command, (result, keys) in STAGES:
        named = lookup(results[result.removesuffix(".json")], keys)
        run = runs[log]
        run_rows.append(
            [f"<code>{command}</code>", run["finished"], duration(run["seconds"]), f"{run['rss_gb']:.2f} GB", named]
        )
    provenance = next(ch for ch in verify["checks"] if ch["check"].startswith("the script md5s"))
    commit_of = {(row[0], row[1]): row[3] for row in provenance["measured"]}
    check_rows = [
        [html.escape(ch["check"]), html.escape(json.dumps(ch["measured"], ensure_ascii=False))]
        for ch in verify["checks"]
    ]
    methods_section = f"""
<ul>
<li><strong>Curves and Gram matrices</strong> (<code>curves.py</code>): each class's curve per slide from the
pathology study's float64 densities and the cell-type study's classes; each class's exact Gram matrix over the
{n_all} slides by <code>betti_inner_product_matrix</code> (topocyte {c.TOPOCYTE_COMMIT}), after a gate: the counts
at the ten deciles equal the cell-type study's <code>composition.npz</code>. The timing spike on the first
{spike["slides"]} slides extrapolated {num(spike["extrapolated_hours"], 2)} h; the Gram took
{duration(runs["gram.log"]["seconds"])}.</li>
<li><strong>Folds</strong> (<code>folds.py</code>, <code>run_study.py</code>): the cell-type improvement phase's
development folds ({n_repeats} repeats of stratified {n_splits}-fold, seed {dev_json["folds"]["seed"]}); in each, the
filter on the training slides, then each block's fPCA by topocyte's <code>fpca_from_gram</code> on the kept slides
({c.N_COMPONENTS} components) with an exact projection of the other slides, then the MFA (each block divided by the SD
of its first component, up to {max(c.K_GRID)} components by SVD), and the four arms' logistic regressions.</li>
<li><strong>Readouts:</strong> each arm's macro AUC over the {len(CONDITIONS)} conditions from the probabilities
averaged over the repeats; the paired differences with 95% intervals over {readouts["bootstrap"]["draws"]:,} stratified
bootstrap draws (seed {readouts["bootstrap"]["seed"]}). The holdout stage fitted everything once on the {n_dev}
development slides, the holdout labels hidden, scored the {n_hold} holdout slides at {hold_json["scored_at"]} and
wrote <code>holdout.json</code> with mode "x".</li>
</ul>
{table(["stage", "finished", "wall time", "peak RSS", "md5 its json names"], run_rows)}
<p class="note">On an M4 with {gram_json["workers"]} cores; the times and memory come from
<code>/usr/bin/time -l</code> in <code>logs/</code>. Python {versions["python"]}, numpy {versions["numpy"]}, scipy
{versions["scipy"]}, scikit-learn {versions["scikit-learn"]}, h5py {versions["h5py"]}, matplotlib
{versions["matplotlib"]}; topocyte {c.TOPOCYTE_COMMIT} from a clean checkout first on <code>PYTHONPATH</code>. The
development run's json names <code>run_study.py</code> as committed in
{commit_of[("development.json", "run_study.py")]}; the readouts and the holdout ran the current file, committed in
{commit_of[("holdout.json", "run_study.py")]} before either ran (<code>verify_results.py</code> checks each md5
against its commit).</p>
<h3>Checks</h3>
<p><code>verify_results.py</code>'s docstring says what each check takes another route for and what it does not
redo.</p>
{table(["check", "measured"], check_rows, "wrap1")}
"""

    body = f"""
<h1>Class-density curves against density and composition</h1>
<p class="subtitle">The fPCA of each slide's four class curves over the {curves["k_nn"]}-NN density threshold, against
D+C, on {n_all} GTEx liver slides in {len(GROUPS)} pathology groups</p>
<p class="meta">Built {built} from the study folder at {html.escape(head)}; {len(verify["checks"])} checks of
<code>verify_results.py</code> pass.</p>
<div class="summary">{summary}</div>
<h2>1. The question and the rulings</h2>
{question}
<h2>2. Each group's mean class curves</h2>
{curves_section}
<h2>3. The slides the filter dropped</h2>
{drops_section}
<h2>4. Fractions of variance per block</h2>
{variance_section}
<h2>5. The chosen k</h2>
{k_section}
<h2>6. AUCs per arm and the paired differences</h2>
{auc_section}
<h2>7. Verdict</h2>
{verdict_section}
<h2>8. Methods and run</h2>
{methods_section}
"""
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Class-curve fPCA</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )
    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
