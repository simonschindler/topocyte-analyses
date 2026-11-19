#!/usr/bin/env python
"""Build the self-contained HTML report of the cell-group study from the saved results.

Reads ``results/`` (composition, slices, spaces, UMAPs, readouts,
comparisons, gates) and the sibling study's groups and all-cell fractions of
variance, and writes ``report.html`` with every figure embedded as a base64
PNG. No number in the prose is typed by hand: all are formatted from those
files, and every sentence that states a direction or a ranking is asserted
against the numbers it states.

Colour follows the sibling report: each pathology group keeps its hue, and a
figure with more than three groups is drawn in facet rows that colour
healthy and two other groups (a triple that passes the all-pairs palette
check) over the rest in grey. Heatmaps use one sequential hue.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common as c  # noqa: E402

OUT = c.HERE / "report.html"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
RULE = "#e1e0d9"
SURFACE = "#fcfcfb"
CONTEXT = "#c3c2b7"
ACCENT = "#2a78d6"
COLOR = {
    "healthy": "#2a78d6",
    "steatosis": "#1baf7a",
    "congestion": "#eb6834",
    "fibrosis": "#eda100",
    "cirrhosis": "#e34948",
    "inflammation/hepatitis": "#008300",
    "necrosis": "#e87ba4",
}
FACETS = (
    ("healthy · steatosis · congestion", ("healthy", "steatosis", "congestion")),
    ("healthy · fibrosis · cirrhosis", ("healthy", "fibrosis", "cirrhosis")),
    ("healthy · inflammation/hepatitis · necrosis", ("healthy", "inflammation/hepatitis", "necrosis")),
)
MODEL_LABEL = {"h0_none": "H0 raw", "h1_none": "H1 raw", "h0_cosine": "H0 cosine", "h1_cosine": "H1 cosine"}
SET_LABEL = {"all": "all cells", "epithelial": "epithelial", "stromal": "stromal", "immune": "immune", "other": "other"}
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2, "grid.color": RULE, "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def png_bytes(fig: plt.Figure, dpi: int = 110) -> bytes:
    """The figure as a 128-colour palette PNG (sibling report)."""
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    palette = image.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    palette.save(out, format="PNG", optimize=True)
    return out.getvalue()


def img_tag(data: bytes, alt: str) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(data).decode("ascii")}" alt="{html.escape(alt)}">'


def html_table(header: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def fmt_interval(values: list[float], digits: int = 3) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]".replace("-", "−")


def signed(value: float, digits: int = 3) -> str:
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{value:+.{digits}f}".replace("-", "−")


def space_label(name: str, aliases: dict[str, str]) -> str:
    for alias, full in aliases.items():
        if name == full:
            return alias
    kind, sets, models = name.split(":")
    set_part = " + ".join(SET_LABEL[s] for s in sets.split("+"))
    model_part = "all models" if models == "+".join(c.MODEL_NAMES) else " + ".join(MODEL_LABEL[m] for m in models.split("+"))
    return f"{set_part} · {model_part}"


# ---------------------------------------------------------------- figures


def figure_composition(kept_by_group: np.ndarray, labels: np.ndarray) -> bytes:
    """Median share of each cell group among a slide's kept cells, by slice, one line per pathology group."""
    share = kept_by_group / kept_by_group.sum(axis=2, keepdims=True)  # (slices, slides, groups)
    fig, axes = plt.subplots(len(FACETS), len(c.CELL_GROUPS), figsize=(13.5, 8.2), constrained_layout=True, sharex=True)
    x = np.arange(c.N_SLICES)
    for row, (facet_title, members) in enumerate(FACETS):
        for col, group in enumerate(c.CELL_GROUPS):
            ax = axes[row, col]
            for cond_index, cond in enumerate(c.CONDITIONS):
                median = np.median(share[:, labels == cond_index, col], axis=1)
                if cond in members:
                    ax.plot(x, median, color=COLOR[cond], lw=2.0, marker="o", ms=4, mec=SURFACE, mew=0.8, zorder=3)
                else:
                    ax.plot(x, median, color=CONTEXT, lw=1.0, zorder=1)
            ax.grid(alpha=0.6)
            ax.set_xticks(x)
            if row == len(FACETS) - 1:
                ax.set_xlabel("density slice")
            if col == 0:
                ax.set_ylabel(f"{facet_title}\nmedian share", fontsize=8)
            if row == 0:
                ax.set_title(f"{group} cells", loc="left", color=INK, fontsize=9.5)
    handles = [Line2D([], [], color=COLOR[cond], lw=2, marker="o", ms=4, label=cond) for cond in c.CONDITIONS]
    handles.append(Line2D([], [], color=CONTEXT, lw=1, label="the other groups of the row"))
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("Share of each cell group among a slide's kept cells, median per pathology group, at each density slice",
                 x=0.01, ha="left", fontsize=10, color=INK)
    return png_bytes(fig)


def figure_scatter(scores: np.ndarray, evr: np.ndarray, labels: np.ndarray, embedding: np.ndarray, title: str) -> bytes:
    """Every pair of the first four components and the UMAP, one facet row per group triple (sibling layout)."""
    group = np.array([c.GROUP_ORDER[i] for i in labels])
    counts = {g: int(np.sum(group == g)) for g in c.GROUP_ORDER}
    panels = [(scores[:, i], scores[:, j], f"PC{i + 1} ({100 * evr[i]:.1f}%)", f"PC{j + 1} ({100 * evr[j]:.1f}%)")
              for i, j in PAIRS]
    panels.append((embedding[:, 0], embedding[:, 1], "UMAP 1", "UMAP 2"))
    fig, axes = plt.subplots(len(FACETS), len(panels), figsize=(15.2, 7.2), constrained_layout=True)
    for row, (facet_title, members) in enumerate(FACETS):
        context = ~np.isin(group, members)
        for col, (xv, yv, xlabel, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(xv[context], yv[context], s=9, color=CONTEXT, linewidths=0, alpha=0.7, zorder=1, rasterized=True)
            for rank, name in enumerate(sorted(members, key=lambda g: -counts[g])):
                chosen = group == name
                ax.scatter(xv[chosen], yv[chosen], s=20, color=COLOR[name], edgecolor=SURFACE, linewidths=0.6, zorder=2 + rank)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.5)
            ax.set_xlabel(xlabel, fontsize=7.5)
            ax.set_ylabel(ylabel, fontsize=7.5)
            if col == len(panels) - 1:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if row == 0:
                    ax.set_title("UMAP of the weighted blocks", loc="left", color=INK, fontsize=8.5)
            else:
                ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3), useMathText=True)
            if col == 0:
                ax.set_title(facet_title, loc="left", color=INK, fontsize=8.5)
    by_facet = list(dict.fromkeys(name for _, members in FACETS for name in members))
    handles = [Line2D([], [], ls="none", marker="o", ms=6, color=COLOR[n], mec=SURFACE, label=f"{n} ({counts[n]})") for n in by_facet]
    handles.append(Line2D([], [], ls="none", marker="o", ms=5, color=CONTEXT,
                          label=f"other slides, unclassified ({counts['unclassified']}) among them"))
    ncol = 4
    rows = -(-len(handles) // ncol)
    handles = [handles[r * ncol + col] for col in range(ncol) for r in range(rows) if r * ncol + col < len(handles)]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=ncol, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    return png_bytes(fig)


def figure_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, value_label: str,
                   marks: dict[tuple[int, int], str] | None = None, figsize: tuple[float, float] = (13.5, 10.5)) -> bytes:
    """One sequential hue; every cell's value printed; named cells outlined."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    finite = matrix[np.isfinite(matrix)]
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=finite.min(), vmax=finite.max())
    mid = (finite.min() + finite.max()) / 2
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color=SURFACE if value > mid else INK)
    for (i, j), label in (marks or {}).items():
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.8))
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.5, pad=0.01)
    bar.set_label(value_label, color=INK2)
    bar.outline.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    return png_bytes(fig)


def figure_intervals(rows: list[tuple[str, float, list[float]]], title: str, xlabel: str, zero: bool = False) -> bytes:
    """Dot and 95% interval per row, one hue."""
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.2), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, (lo, hi)) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=ACCENT, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([value], [yi], s=46, color=ACCENT, edgecolor=SURFACE, linewidths=1.4, zorder=3)
        ax.annotate(f"{value:.3f}", (hi, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    if zero:
        ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.grid(axis="x", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    return png_bytes(fig)


def section_improvements() -> str:
    """Summary of the improvement phase (details: improvements.md); asserted against its JSON files."""
    dev_path, hold_path = c.RESULTS / "improve_dev.json", c.RESULTS / "improve_holdout.json"
    if not dev_path.exists():
        return "<p>Not run yet.</p>"
    dev = json.loads(dev_path.read_text())
    n_ideas = sum(entry["role"] == "idea" for entry in dev["ideas"])
    text = (f"<p>Exploratory, after the main readouts: {n_ideas} modelling ideas (and their baselines, controls and "
            f"diagnostics) were scored on the {dev['dev_slides']} development slides only; the rule for choosing which reach "
            "the holdout was fixed before any readout (design.md), and the 118 holdout slides were scored once. "
            "Every idea, its development readouts and the full holdout table are in <code>improvements.md</code>.</p>")
    if not hold_path.exists():
        return text + "<p>The holdout has not been scored yet.</p>"
    hold = json.loads(hold_path.read_text())
    rows = []
    for name, row in hold["results"].items():
        rows.append([html.escape(name), f"{row['macro_auc']:.3f} <span class='unit'>{fmt_interval(row['auc_interval'])}</span>",
                     f"{signed(row['vs_everything4']['difference'])} <span class='unit'>{fmt_interval(row['vs_everything4']['interval'])}</span>",
                     f"{signed(row['vs_DC']['difference'])} <span class='unit'>{fmt_interval(row['vs_DC']['interval'])}</span>",
                     f"{row['purity_macro']:.2f}" if "purity_macro" in row else "—"])
    table = html_table(["holdout entry", "macro AUC", "vs everything (4)", "vs D+C features", "purity ×chance"], rows)
    groups = ", ".join(f"{g} {n}" for g, n in hold["holdout_per_group"].items())
    return text + (f"<p>Holdout ({hold['holdout_slides']} slides: {html.escape(groups)}), chosen by the fixed rule: "
                   f"{', '.join(f'<code>{html.escape(n)}</code>' for n in hold['chosen']) or 'none'}. With two "
                   "inflammation/hepatitis and four necrosis slides, the holdout intervals understate the uncertainty.</p>"
                   + table)


# ---------------------------------------------------------------- main


def main() -> None:
    labels_table = pd.read_csv(c.SOURCE / "pathology.csv")
    labels = np.array([c.GROUP_ORDER.index(g) for g in labels_table["group"]])
    with np.load(c.RESULTS / "composition.npz", allow_pickle=False) as data:
        kept_by_group = data["kept_by_group"]
        categories = [str(x) for x in data["categories"]]
        category_group = data["category_group"]
    spaces_meta = json.loads((c.RESULTS / "spaces.json").read_text())
    aliases = spaces_meta["aliases"]
    with np.load(c.RESULTS / "spaces.npz", allow_pickle=False) as data:
        names = [str(n) for n in data["names"]]
        space_scores = {name: data[f"s{i}__scores"] for i, name in enumerate(names)}
        space_evr = {name: data[f"s{i}__evr"] for i, name in enumerate(names)}
        dropped = [str(d) for d in data["dropped_blocks"]]
        if not np.array_equal(data["labels"], labels):
            raise RuntimeError("spaces.npz and pathology.csv disagree on the groups")
    with np.load(c.RESULTS / "umap.npz", allow_pickle=False) as data:
        umaps = {str(n): data[f"u{i}"] for i, n in enumerate(data["names"])}
    readouts = {row["job"]: row for row in json.loads((c.RESULTS / "readouts.json").read_text())["rows"]}
    comparisons = json.loads((c.RESULTS / "comparisons.json").read_text())
    slices = [json.loads((c.RESULTS / f"slice_{k}.json").read_text()) for k in range(c.N_SLICES)]
    gates = [json.loads((c.RESULTS / f"gate_{k}.json").read_text()) for k in (9, 8)]
    assert all(g["passed"] for g in gates)
    evr_blocks = {}
    for k in range(c.N_SLICES):
        with np.load(c.SOURCE / f"slice_{k}.npz", allow_pickle=False) as sibling, \
                np.load(c.RESULTS / f"slice_{k}.npz", allow_pickle=False) as own:
            for m in c.MODEL_NAMES:
                evr_blocks[("all", m, k)] = sibling[f"{m}_evr"]
                for g in c.CELL_GROUPS:
                    if f"{g}_{m}_evr" in own.files:
                        evr_blocks[(g, m, k)] = own[f"{g}_{m}_evr"]

    def job(alias_or_name: str, k: int) -> dict:
        return readouts[f"{aliases.get(alias_or_name, alias_or_name)}@{k}"]

    n_slides = labels.size
    n_cells = int(kept_by_group[0].sum())
    counts = {g: int(np.sum(labels == i)) for i, g in enumerate(c.GROUP_ORDER)}

    # ---------------- section: cell groups
    totals = kept_by_group[0].sum(axis=0)
    type_rows = []
    for j, group in enumerate(c.CELL_GROUPS):
        members = [categories[i] for i in range(len(categories)) if category_group[i] == j]
        type_rows.append([group, ", ".join(members), f"{int(totals[j]):,}", f"{100 * totals[j] / n_cells:.1f}%",
                          f"{int(kept_by_group[9, :, j].min()):,}", f"{int(np.median(kept_by_group[9, :, j])):,}"])
    type_table = html_table(["group", "HistoPLUS types", "cells", "share", "fewest at slice 9", "median at slice 9"], type_rows, "left2")
    share = kept_by_group / kept_by_group.sum(axis=2, keepdims=True)
    comp_rows = []
    for i, cond in enumerate(c.GROUP_ORDER):
        comp_rows.append([f"{cond} ({counts[cond]})"] + [
            f"{100 * np.median(share[0, labels == i, j]):.1f} / {100 * np.median(share[9, labels == i, j]):.1f}"
            for j in range(len(c.CELL_GROUPS))])
    comp_table = html_table(["pathology group"] + [f"{g} <span class='unit'>% slice 0 / 9</span>" for g in c.CELL_GROUPS], comp_rows)
    immune_dense = [np.median(share[9, labels == i, 2]) for i in range(len(c.CONDITIONS))]
    immune_top = c.CONDITIONS[int(np.argmax(immune_dense))]

    # ---------------- section: per-group fPCA fractions of variance
    fve_rows = []
    for p in c.POINT_SETS:
        for m in c.MODEL_NAMES:
            values = [evr_blocks.get((p, m, k)) for k in range(c.N_SLICES)]
            fve_rows.append([f"{SET_LABEL[p]}, {MODEL_LABEL[m]}"] + [f"{v[0]:.3f}" if v is not None else "dropped" for v in values])
    fve_table = html_table(["point set, model"] + [f"slice {k}" for k in range(c.N_SLICES)], fve_rows)

    # ---------------- section: key spaces
    key_names = spaces_meta["key_spaces"]
    scatter_figures = []
    umap_purity: dict[str, float] = {}
    for name in key_names:
        label = space_label(name, aliases)
        r4, r20 = readouts[f"{name}@4"], readouts[f"{name}@20"]
        umap_purity[name] = c.macro_purity(c.purity(umaps[name], labels), labels)[0]
        png = figure_scatter(space_scores[name], space_evr[name], labels, umaps[name],
                             f"{label}: every pair of the first four components and a UMAP; one row per group triple")
        scatter_figures.append(
            f"<figure>{img_tag(png, f'{label}: scatter matrix of PC1 to PC4 and a UMAP, by pathology group')}"
            f"<figcaption><strong>{html.escape(label)}</strong>: macro AUC {r4['macro_auc']:.3f} (4 components) / "
            f"{r20['macro_auc']:.3f} (20); purity {r4['purity_macro']:.2f} / {r20['purity_macro']:.2f} times chance, "
            f"{umap_purity[name]:.2f} in the UMAP plane (descriptive). "
            "The first six columns are the pairs of the first four components; the last is a UMAP of the weighted "
            "blocks (no units). Each row colours healthy and two other groups over all remaining slides in grey."
            "</figcaption></figure>")

    # ---------------- section: readouts across spaces (heatmaps)
    set_subsets = [s for r in range(1, 6) for s in __import__("itertools").combinations(c.POINT_SETS, r)]
    model_subsets = [s for r in range(1, 5) for s in __import__("itertools").combinations(c.MODEL_NAMES, r)]
    row_labels = [" + ".join(SET_LABEL[s] for s in subset) for subset in set_subsets]
    col_labels = [" + ".join(MODEL_LABEL[m] for m in subset) for subset in model_subsets]

    def grid(metric: str, k: int) -> np.ndarray:
        out = np.full((len(set_subsets), len(model_subsets)), np.nan)
        for i, sets in enumerate(set_subsets):
            for j, models in enumerate(model_subsets):
                out[i, j] = readouts[f"mfa:{'+'.join(sets)}:{'+'.join(models)}@{k}"][metric]
        return out

    marks = {}
    for alias in ("all-cell", "groups", "everything"):
        _, sets, models = aliases[alias].split(":")
        marks[(set_subsets.index(tuple(sets.split("+"))), model_subsets.index(tuple(models.split("+"))))] = alias
    heatmaps = []
    mfa_best = {}
    for metric, metric_label in (("macro_auc", "macro AUC"), ("purity_macro", "purity, times chance")):
        for k in (4, 20):
            matrix = grid(metric, k)
            best = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            mfa_best[(metric, k)] = (f"mfa:{'+'.join(set_subsets[best[0]])}:{'+'.join(model_subsets[best[1]])}", float(matrix[best]))
            png = figure_heatmap(matrix, row_labels, col_labels,
                                 f"{metric_label}, first {k} components, every combined space (outlined: all-cell, groups, everything)",
                                 metric_label, marks)
            heatmaps.append(f"<figure>{img_tag(png, f'Heatmap of {metric_label} at {k} components over 465 combined spaces')}"
                            f"<figcaption>{metric_label.capitalize()} of the {len(set_subsets) * len(model_subsets)} combined "
                            f"spaces on their first {k} components: rows are the point sets combined, columns the models. "
                            "Outlined: the three named spaces.</figcaption></figure>")
    block_rows = [f"{SET_LABEL[p]}, {MODEL_LABEL[m]}" for p in c.POINT_SETS for m in c.MODEL_NAMES]
    block_matrix = np.full((len(block_rows), c.N_SLICES), np.nan)
    for i, (p, m) in enumerate((p, m) for p in c.POINT_SETS for m in c.MODEL_NAMES):
        for k in range(c.N_SLICES):
            key = f"block:{p}:{m}:{k}@4"
            if key in readouts:
                block_matrix[i, k] = readouts[key]["macro_auc"]
    block_png = figure_heatmap(block_matrix, block_rows, [f"slice {k}" for k in range(c.N_SLICES)],
                               "macro AUC, first 4 components of each single block (one point set, one model, one slice)",
                               "macro AUC", None, figsize=(10.5, 7.8))
    block_best = np.unravel_index(np.nanargmax(block_matrix), block_matrix.shape)

    # ---------------- comparisons
    p4, p20, sec = comparisons["primary_k4"], comparisons["primary_k20"], comparisons["secondary"]
    verdict_text = {
        "enhanced": "enhanced: both intervals lie above 0",
        "mixed": "mixed: exactly one interval lies above 0",
        "not enhanced": "not enhanced: neither interval lies above 0",
    }[p4["verdict"]]
    below = [name for name, flag in (("macro AUC", p4["below_zero"]["auc"]), ("purity", p4["below_zero"]["purity"])) if flag]
    below_text = f" The {' and the '.join(below)} interval lies entirely below 0: everything separates worse there." if below else ""
    primary_rows = []
    for key, result in (("k = 4 (fixed in advance)", p4), ("k = 20 (descriptive)", p20)):
        primary_rows.append([key, "macro AUC", f"{result['auc']['a']:.3f}", f"{result['auc']['b']:.3f}",
                             signed(result["auc"]["difference"]), fmt_interval(result["auc"]["interval"]), result["verdict"] if "4" in key else ""])
        primary_rows.append([key, "purity", f"{result['purity']['a']:.2f}", f"{result['purity']['b']:.2f}",
                             signed(result["purity"]["difference"], 2), fmt_interval(result["purity"]["interval"], 2), ""])
    primary_table = html_table(["components", "readout", "everything", "all-cell", "difference", "95% interval", "verdict"], primary_rows)
    secondary_text = (
        f"The <code>D+C</code> features alone reach a macro AUC of {sec['auc']['b']:.3f}; with the first 20 components of "
        f"<code>groups</code> added, {sec['auc']['a']:.3f}, a difference of {signed(sec['auc']['difference'])} "
        f"{fmt_interval(sec['auc']['interval'])}. "
        + ("Per-group topology adds to density and composition by the rule fixed in advance: the interval lies above 0."
           if sec["adds"] else
           "By the rule fixed in advance, per-group topology does not add to density and composition: the interval does not lie above 0.")
    )
    assert sec["adds"] == (sec["auc"]["interval"][0] > 0)
    ladder_png = figure_intervals([(row["label"], row["auc"], row["interval"]) for row in comparisons["ladder"]],
                                  "Ladder: macro AUC of nested inputs, 20 × stratified 5-fold, 95% bootstrap interval", "macro AUC")
    ladder_rows = [[row["label"], f"{row['auc']:.3f}", fmt_interval(row["interval"]),
                    f"{row['purity']:.2f}" if row["purity"] is not None else "—"] for row in comparisons["ladder"]]
    ladder_table = html_table(["input", "macro AUC", "95% interval", "purity (×chance)"], ladder_rows)
    diff_label = {row["job"]: row["label"] for row in comparisons["ladder"]}
    diff_rows = [[f"{diff_label[d['a']]} − {diff_label[d['b']]}", signed(d["auc"]["difference"]), fmt_interval(d["auc"]["interval"])]
                 for d in comparisons["ladder_differences"]]
    diff_table = html_table(["difference", "macro AUC", "95% interval"], diff_rows)
    vs_rows = []
    jobs_vs = list(comparisons["vs_healthy"])
    for cond_index, cond in enumerate(c.CONDITIONS[1:]):
        vs_rows.append([cond] + [f"{comparisons['vs_healthy'][j][cond_index]['auc']:.2f} "
                                 f"<span class='unit'>{fmt_interval(comparisons['vs_healthy'][j][cond_index]['interval'], 2)}</span>"
                                 for j in jobs_vs])
    vs_table = html_table(["condition vs healthy"] + [html.escape(j.replace("features:", "")) for j in jobs_vs], vs_rows)

    # ---------------- headline numbers, asserted
    allcell4, every4 = job("all-cell", 4), job("everything", 4)
    every20 = job("everything", 20)
    assert np.isclose(p4["auc"]["a"], every4["macro_auc"]) and np.isclose(p4["auc"]["b"], allcell4["macro_auc"])
    ladder = {row["job"]: row for row in comparisons["ladder"]}
    dc_auc = ladder["features:D+C"]["auc"]
    topo_vs_dc = "above" if every20["macro_auc"] > dc_auc else "below"
    assert (every20["macro_auc"] > dc_auc) == (topo_vs_dc == "above")
    best_auc4_name, best_auc4 = mfa_best[("macro_auc", 4)]
    best_auc20_name, best_auc20 = mfa_best[("macro_auc", 20)]
    n_warn = sum(row["n_convergence_warnings"] for row in readouts.values())
    total_runtime = sum(s["runtime_seconds"] for s in slices) / 60

    comp4 = readouts["C@4"]
    comp_feat = ladder["features:C"]
    cp = comparisons["composition_pca_k4"]
    assert np.isclose(cp["auc"]["a"], comp4["macro_auc"]) and np.isclose(cp["auc"]["b"], every4["macro_auc"])
    comp_vs_topology = "at least as well as" if comp_feat["auc"] >= every20["macro_auc"] else "less well than"
    assert (comp_feat["auc"] >= every20["macro_auc"]) == (comp_vs_topology == "at least as well as")
    late = {(d["a"], d["b"]): d for d in comparisons["ladder_differences"]}[("features:D+C+everything20", "features:D+C+all-cell20")]

    def relation(interval: list[float], up: str, same: str, down: str) -> str:
        return up if interval[0] > 0 else down if interval[1] < 0 else same

    comp4_auc_word = relation(cp["auc"]["interval"], "a higher", "a similar", "a lower")
    comp4_purity_word = relation(cp["purity"]["interval"], "higher", "similar", "lower")
    if p4["verdict"] == "enhanced" and not sec["adds"]:
        assert late["auc"]["interval"][0] <= 0 <= late["auc"]["interval"][1]
        reading = (
            "<p><strong>Reading.</strong> Splitting the cells by type makes the liver conditions easier to tell apart than "
            "all-cell topology does, on both readouts. The number of cells of each group at each density level accounts for "
            "the gain: per-group topology adds nothing detectable to a linear readout of density and composition "
            f"({signed(sec['auc']['difference'])} {fmt_interval(sec['auc']['interval'])}); with density and composition in the "
            f"model, the four groups add {signed(late['auc']['difference'])} {fmt_interval(late['auc']['interval'])} to all-cell "
            f"topology; and the composition alone separates the conditions {comp_vs_topology} the everything space.</p>")
    else:
        reading = ""
    comp4_text = (
        f" At the same four components, the PCA of the composition alone (log cells per group and slice, no topology) "
        f"reaches {comp4_auc_word} macro AUC, {comp4['macro_auc']:.3f} ({signed(cp['auc']['difference'])} "
        f"{fmt_interval(cp['auc']['interval'])} against everything), and {comp4_purity_word} purity, {comp4['purity_macro']:.2f} "
        f"times chance ({signed(cp['purity']['difference'], 2)} {fmt_interval(cp['purity']['interval'], 2)})."
    )
    summary = reading + (
        f"<p><strong>Fixed in advance — {verdict_text}.</strong> On the first four components, the space of all five "
        f"point sets (<code>everything</code>) reaches a macro AUC of {every4['macro_auc']:.3f} against "
        f"{allcell4['macro_auc']:.3f} for all-cell topology, a difference of {signed(p4['auc']['difference'])} "
        f"{fmt_interval(p4['auc']['interval'])}; purity {every4['purity_macro']:.2f} against {allcell4['purity_macro']:.2f} "
        f"times chance, {signed(p4['purity']['difference'], 2)} {fmt_interval(p4['purity']['interval'], 2)}.{below_text}{comp4_text}</p>"
        f"<p><strong>Composition guard.</strong> {secondary_text}</p>"
        f"<p>Density summaries and cell counts per group reach {dc_auc:.3f}, the cell counts alone {comp_feat['auc']:.3f}; the everything space on 20 components "
        f"lies {topo_vs_dc} that, at {every20['macro_auc']:.3f}. Among all 465 combined spaces the highest macro AUC is "
        f"{best_auc4:.3f} on four components (<em>{html.escape(space_label(best_auc4_name, aliases))}</em>) and {best_auc20:.3f} "
        f"on twenty (<em>{html.escape(space_label(best_auc20_name, aliases))}</em>); these maxima are picked after the fact "
        "and carry no interval.</p>"
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liver pathology per cell group</title>
<style>
  :root {{ --ink: {INK}; --ink2: {INK2}; --muted: {MUTED}; --rule: {RULE}; --surface: {SURFACE}; --page: #f9f9f7; --accent: {ACCENT}; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ --ink: #f0efec; --ink2: #c3c2b7; --muted: #a3a29b; --rule: #383835; --surface: #1f1f1e; --page: #1a1a19; --accent: #3987e5; }}
  }}
  :root[data-theme="dark"] {{ --ink: #f0efec; --ink2: #c3c2b7; --muted: #a3a29b; --rule: #383835; --surface: #1f1f1e; --page: #1a1a19; --accent: #3987e5; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--page); color: var(--ink); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.55; font-size: 16px; }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 36px 16px 90px; }}
  h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.3rem; margin: 44px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }}
  h3 {{ font-size: 1.05rem; margin: 30px 0 8px; }}
  p, li {{ max-width: 80ch; }}
  .subtitle {{ color: var(--ink2); font-size: 1.05rem; margin: 0 0 8px; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin: 0 0 28px; }}
  .summary {{ background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--accent); padding: 16px 20px; margin: 24px 0; border-radius: 4px; }}
  .summary p {{ margin: 0 0 8px; }} .summary p:last-child {{ margin: 0; }}
  figure {{ margin: 22px 0; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 3px; }}
  figcaption {{ color: var(--ink2); font-size: 0.85rem; margin-top: 8px; }}
  .table-wrap {{ overflow-x: auto; margin: 16px 0 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; font-variant-numeric: tabular-nums; background: var(--surface); }}
  th, td {{ border-bottom: 1px solid var(--rule); padding: 7px 8px; text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .left2 th:nth-child(2), .left2 td:nth-child(2) {{ text-align: left; white-space: normal; }}
  th {{ color: var(--ink2); font-weight: 600; vertical-align: bottom; }}
  .unit {{ color: var(--muted); font-weight: 400; font-size: 0.72rem; }}
  .note {{ color: var(--ink2); font-size: 0.85rem; }}
  code {{ font-size: 0.86em; background: var(--rule); padding: 1px 4px; border-radius: 3px; }}
  details {{ margin: 14px 0; }} summary {{ cursor: pointer; color: var(--ink2); }}
  ul {{ margin: 8px 0 8px 22px; padding: 0; }} li {{ margin: 4px 0; }}
</style>
</head>
<body>
<main>
  <h1>Liver pathology per cell group</h1>
  <p class="subtitle">Does each cell group's own topology, at the same tissue-density levels, make the liver conditions easier to tell apart?</p>
  <p class="meta">Report generated {dt.date.today().isoformat()} from the saved results of <code>run_celltype_slices.py</code>,
  <code>spaces.py</code> and <code>readouts.py</code>; per-group persistent homology and fPCA {total_runtime:.1f} min; topocyte
  {html.escape(slices[0]['topocyte_commit'][:7])}. The design, with every ruling and the rule for the comparisons fixed in advance, is
  <code>design.md</code>, committed before any readout existed.</p>

  <div class="summary">{summary}</div>

  <h2>1. Question and set-up</h2>
  <p>The sibling Betti-curve study cut each of the {n_slides} GTEx liver slides ({n_cells:,} cells) at ten levels of the
  all-cell 10-nearest-neighbour density and found the leading combined component to be tissue density. Here the cells kept at
  each level are split by HistoPLUS cell type into four groups, and each group gets its own alpha persistent homology and
  Betti-curve fPCA (H0 and H1, raw and cosine, nine components) at the same ten levels: 200 blocks of scores with the sibling's
  40 all-cell blocks. Two readouts measure how well a space separates the seven pathology groups: cross-validated logistic
  regression (macro one-vs-rest AUC) and the purity of each slide's 15 nearest neighbours relative to chance. Spaces are
  compared on the same number of leading components.</p>
  {type_table}
  <p class="note">Groups as ruled (design.md). Every slide keeps at least {int(kept_by_group.min()):,} cells of every group at
  every slice; {"no block was dropped" if not dropped else "dropped blocks: " + html.escape(", ".join(dropped))}.</p>

  <h2>2. Composition across density slices</h2>
  <p>A raw H0 Betti curve starts at the number of points, so each group's raw H0 model carries its cell count; the composition
  is therefore one of the baselines below. At the densest slice the immune share is highest in {html.escape(immune_top)}
  slides (median {100 * max(immune_dense):.1f}%).</p>
  <figure>{img_tag(figure_composition(kept_by_group, labels), "Median share of each cell group by slice, per pathology group")}
  <figcaption>Median share of each cell group among the slide's kept cells at each slice, per pathology group; each row colours
  healthy and two other groups over the others in grey.</figcaption></figure>
  <details><summary>Table: median shares at slice 0 and slice 9</summary>{comp_table}</details>

  <h2>3. Per-group fPCA</h2>
  <p class="note">First component's fraction of variance of every block (the all-cell rows are the sibling study's).</p>
  {fve_table}

  <h2>4. Key spaces</h2>
  <p>Each combined space is a multiple factor analysis of its blocks over the ten slices, as in the sibling study: each block
  divided by the standard deviation of its first score, then PCA of the concatenation. The six spaces below are all-cell
  topology, each group over all four models, and everything.</p>
  {''.join(scatter_figures)}

  <h2>5. Fixed in advance</h2>
  <p><strong>Primary.</strong> <code>everything</code> against <code>all-cell</code> on the first four components: enhanced if the
  95% paired bootstrap intervals of both differences lie above 0, mixed if one does, not enhanced if neither does.</p>
  {primary_table}
  <p><strong>Secondary (composition guard).</strong> {secondary_text}</p>

  <h2>6. Ladder</h2>
  <p>Logistic regression on nested inputs: density summaries (<code>D</code>: log median density, log cell count and the share
  of cells kept at each slice), composition (<code>C</code>: log cells of each group at each slice), both, the topology spaces on
  20 components, and <code>D+C</code> with each topology space's first 20 components.</p>
  <figure>{img_tag(ladder_png, "Macro AUC with 95% intervals for the ladder of nested inputs")}
  <figcaption>Macro AUC of each input, with its 95% bootstrap interval.</figcaption></figure>
  {ladder_table}
  <p class="note">Paired differences between rungs (same folds, same bootstrap draws):</p>
  {diff_table}

  <h2>7. Every combined space</h2>
  <p>Both readouts for all {len(set_subsets) * len(model_subsets)} combinations of point sets and models, on 4 and 20
  components. These are descriptive: with 465 spaces, the best cell is the largest of many noisy numbers.</p>
  {''.join(heatmaps)}
  <figure>{img_tag(block_png, "Heatmap of macro AUC for every single block")}
  <figcaption>Macro AUC on the first four components of each single block. The highest is
  {SET_LABEL[c.POINT_SETS[block_best[0] // 4]]}, {MODEL_LABEL[c.MODEL_NAMES[block_best[0] % 4]]} at slice {block_best[1]}
  ({block_matrix[block_best]:.3f}).</figcaption></figure>

  <h2>8. Each condition against healthy</h2>
  <p class="note">AUC of log p(condition) − log p(healthy) between the condition's slides and the healthy slides, from the
  out-of-fold probabilities, with 95% bootstrap intervals. Inflammation/hepatitis has {counts['inflammation/hepatitis']} slides and
  necrosis {counts['necrosis']}: their intervals are wide.</p>
  {vs_table}

  <h2>9. Improvement phase</h2>
  {section_improvements()}

  <h2>Methods and reproduction</h2>
  <ul>
    <li><code>run_celltype_slices.py groups</code> counts each group's cells per slide and slice; <code>slice K</code> runs
    <code>tc.ph.compute_persistent_homology</code> (alpha, max_dim 1) on each slide's cells of each group with ρ at or above the
    sibling's decile K, then <code>tc.ph.FPCA</code> (nine components, matrix-free ARPACK as topocyte chooses) exactly as the
    sibling did; <code>gate K</code> reran that path on all cells at slices 9 and 8 and matched the sibling's scores (minimum
    |r| {min(min(g[m]['min_abs_corr'] for m in c.MODEL_NAMES) for g in gates):.9f}).</li>
    <li><code>spaces.py</code> builds the {len(names)} spaces; the all-cell space reproduces the sibling's combined space (minimum
    |r| {spaces_meta['all_cell_vs_sibling_min_abs_corr']:.9f}).</li>
    <li><code>readouts.py</code>: logistic regression (multinomial, L2, C = 1, balanced class weights, features standardized within
    each training fold, eight classes with unclassified as its own and never scored), 20 repeats of stratified 5-fold with the same
    folds for every space; macro AUC from each slide's out-of-fold probabilities averaged over the repeats; purity from the 15
    nearest neighbours (Euclidean, on the components as fitted) over the chance share (n<sub>g</sub> − 1)/(N − 1); intervals from
    4000 bootstrap draws of slides stratified by group, the same draws for every comparison. lbfgs raised {n_warn} convergence
    warnings over all fits.</li>
    <li><code>verify_results.py</code> recomputes each result by an independent route; <code>build_report.py</code> writes this
    page. The cohort was opened read only. A seeded fifth of the slides (<code>results/holdout.csv</code>) is reserved for the
    improvement phase and is not treated specially here.</li>
  </ul>
</main>
</body>
</html>
"""
    OUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
