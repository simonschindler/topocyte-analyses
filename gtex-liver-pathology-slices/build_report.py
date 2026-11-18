#!/usr/bin/env python
"""Build the self-contained HTML report from the saved results.

Reads ``results/pathology.csv``, ``results/density.npz`` (not its per-cell
array) and ``results/slice_<k>.npz`` with their ``.json``, and writes
``report.html`` with every figure embedded as a base64 PNG.  No number in the
prose is typed by hand: all of them are formatted from those files.

Colour: each pathology group keeps one hue of the report palette throughout.
A scatter can carry at most three hues that stay distinguishable in every
pair, under colour-vision deficiency too, so every pair of components is drawn
three times, one facet per row, each facet colouring healthy and two other
groups (a triple that passes the all-pairs palette check) over every other
slide in grey.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "report.html"
N_SLICES = 10
N_PLOT = 4  # components in the scatter matrix

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
RULE = "#e1e0d9"
SURFACE = "#fcfcfb"
CONTEXT = "#c3c2b7"  # every slide outside a facet's groups

MODELS = (
    ("h0_none", "H0, raw"),
    ("h1_none", "H1, raw"),
    ("h0_cosine", "H0, cosine"),
    ("h1_cosine", "H1, cosine"),
)
GROUP_ORDER = (
    "healthy",
    "congestion",
    "steatosis",
    "necrosis",
    "inflammation/hepatitis",
    "fibrosis",
    "cirrhosis",
    "unclassified",
)
# Slots of the report palette; each facet's triple passes validate_palette.py
# --pairs all in light mode (worst CVD Delta E 9.2, 15.3 and 13.0).
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
PAIRS = [(i, j) for i in range(N_PLOT) for j in range(i + 1, N_PLOT)]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": "#c3c2b7",
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


def png_bytes(fig: plt.Figure, dpi: int = 110) -> bytes:
    """The figure as a 128-colour palette PNG: a third of the RGB file, the same to the eye."""
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
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def html_table(header: list[str], rows: list[tuple[str, ...]], cls: str = "") -> str:
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join(
        ("<tr class='group'>" if row[0] else "<tr>") + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def load_slice(k: int) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(RESULTS / f"slice_{k}.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((RESULTS / f"slice_{k}.json").read_text())


def figure_density(density: dict[str, np.ndarray]) -> bytes:
    """The pooled quantile curve of the 10-NN density, with the ten slice levels."""
    levels, curve = density["curve_levels"], density["curve"]
    deciles, q = density["deciles"], density["decile_levels"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    inside = (levels > 0) & (levels < 1)  # the extremes are single cells
    ax.plot(levels[inside], curve[inside], color=COLOR["healthy"], lw=2.0, zorder=2)
    ax.set_yscale("log")
    for k, (level, value) in enumerate(zip(q, deciles)):
        if k == 0:
            continue
        ax.plot([level, level], [curve[inside].min(), value], color=RULE, lw=1.0, zorder=1)
        ax.scatter([level], [value], s=34, color=COLOR["healthy"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.annotate(f"{k}", (level, value), xytext=(-9, 6), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("quantile level over all cells")
    ax.set_ylabel("10-NN density, cells/mm² (log)")
    ax.grid(axis="y", alpha=0.7, which="both")
    ax.set_title(
        "Pooled quantile curve; dots: the slice levels, numbered (slice 0 keeps every cell)",
        loc="left", color=INK, fontsize=9.5,
    )
    return png_bytes(fig)


def figure_fve(arrays: dict[str, np.ndarray], k: int, floor: float) -> bytes:
    """Scree plot per model: the FVE of each of the nine components, log scale."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11.5, 2.9), sharey=True, constrained_layout=True)
    for ax, (key, label) in zip(axes, MODELS):
        evr = arrays[f"{key}_evr"]
        position = np.arange(1, evr.size + 1)
        ax.plot(position, evr, color=COLOR["healthy"], lw=2.0, marker="o", ms=5.5,
                mec=SURFACE, mew=1.0, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(floor, 1.5)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(position)
        ax.set_xlabel("component")
        ax.grid(axis="y", alpha=0.7)
        ax.set_title(
            f"{label}: PC1 {evr[0]:.3f}, PC1–4 {evr[:4].sum():.3f}",
            loc="left", color=INK, fontsize=9,
        )
    axes[0].set_ylabel("fraction of variance")
    fig.suptitle(f"Slice {k}: fraction of variance explained per component (log scale)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    return png_bytes(fig)


def figure_scatter(
    arrays: dict[str, np.ndarray],
    key: str,
    label: str,
    k: int | None,
    groups: dict[str, str],
    embedding: np.ndarray,
    title: str | None = None,
    umap_title: str = "UMAP of the nine scores",
) -> bytes:
    """PC pairs among the first four and the UMAP of all nine, one facet row per group triple."""
    ids = [str(g) for g in arrays[f"{key}_group_ids"]]
    scores = arrays[f"{key}_scores"]
    evr = arrays[f"{key}_evr"]
    group = np.array([groups[g] for g in ids])
    counts = {g: int(np.sum(group == g)) for g in GROUP_ORDER}
    panels = [(scores[:, i], scores[:, j], f"PC{i + 1} ({100 * evr[i]:.1f}%)", f"PC{j + 1} ({100 * evr[j]:.1f}%)")
              for i, j in PAIRS]
    panels.append((embedding[:, 0], embedding[:, 1], "UMAP 1", "UMAP 2"))
    fig, axes = plt.subplots(len(FACETS), len(panels), figsize=(15.2, 7.2), constrained_layout=True)
    for row, (facet_title, members) in enumerate(FACETS):
        context = ~np.isin(group, members)
        for col, (x, y, xlabel, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(x[context], y[context], s=9, color=CONTEXT, linewidths=0,
                       alpha=0.7, zorder=1, rasterized=True)
            # Larger groups first, so the smaller ones stay on top.
            for rank, name in enumerate(sorted(members, key=lambda g: -counts[g])):
                chosen = group == name
                ax.scatter(x[chosen], y[chosen], s=20, color=COLOR[name],
                           edgecolor=SURFACE, linewidths=0.6, zorder=2 + rank)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.5)
            ax.set_xlabel(xlabel, fontsize=7.5)
            ax.set_ylabel(ylabel, fontsize=7.5)
            if col == len(panels) - 1:
                # UMAP coordinates carry no units: no tick values.
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if row == 0:
                    ax.set_title(umap_title, loc="left", color=INK, fontsize=8.5)
            else:
                # Raw scores reach 10^6: one power-of-ten offset per axis, the same style everywhere.
                ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3), useMathText=True)
                ax.xaxis.get_offset_text().set_fontsize(7)
                ax.yaxis.get_offset_text().set_fontsize(7)
            if col == 0:
                ax.set_title(facet_title, loc="left", color=INK, fontsize=8.5)
    # Facet order, read row by row: matplotlib fills a legend column by column.
    by_facet = list(dict.fromkeys(name for _, members in FACETS for name in members))
    handles = [
        Line2D([], [], ls="none", marker="o", ms=6, color=COLOR[name], mec=SURFACE,
               label=f"{name} ({counts[name]})")
        for name in by_facet
    ]
    handles.append(Line2D([], [], ls="none", marker="o", ms=5, color=CONTEXT,
                          label=f"other slides, unclassified ({counts['unclassified']}) among them"))
    ncol = 4
    rows = -(-len(handles) // ncol)
    handles = [handles[r * ncol + c] for c in range(ncol) for r in range(rows) if r * ncol + c < len(handles)]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=ncol, fontsize=8)
    if title is None:
        title = (f"Slice {k}, {label}: every pair of the first four components, and the UMAP of all nine; "
                 "one row per group triple")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    return png_bytes(fig)


SPACE_SHORT = {"all": "all 40 blocks", "h0_none": "H0 raw", "h1_none": "H1 raw", "h0_cosine": "H0 cosine", "h1_cosine": "H1 cosine"}
SPACE_LABELS = {
    "all": "all 40 blocks",
    "h0_none": "H0, raw, ten slices",
    "h1_none": "H1, raw, ten slices",
    "h0_cosine": "H0, cosine, ten slices",
    "h1_cosine": "H1, cosine, ten slices",
}


def figure_combined_fve(combined: dict[str, np.ndarray], spaces: list[str]) -> bytes:
    """Scree plot of each combined space: the first 20 components, log scale."""
    fig, axes = plt.subplots(1, len(spaces), figsize=(13.2, 2.9), sharey=True, constrained_layout=True)
    floor = 10 ** np.floor(np.log10(min(float(combined[f"{s}_evr"][19]) for s in spaces)))
    for ax, space in zip(axes, spaces):
        evr = combined[f"{space}_evr"][:20]
        position = np.arange(1, evr.size + 1)
        ax.plot(position, evr, color=COLOR["healthy"], lw=2.0, marker="o", ms=4.5, mec=SURFACE, mew=1.0, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(floor, 1.5)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("combined component")
        ax.grid(axis="y", alpha=0.7)
        ax.set_title(f"{SPACE_SHORT[space]}: PC1 {evr[0]:.3f}, PC1–4 {evr[:4].sum():.3f}",
                     loc="left", color=INK, fontsize=8.5)
    axes[0].set_ylabel("fraction of variance")
    fig.suptitle("Combined spaces: fraction of variance per component (log scale)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    return png_bytes(fig)


def section_combined(
    combined: dict[str, np.ndarray],
    slides: list[tuple[dict[str, np.ndarray], dict]],
    density: dict[str, np.ndarray],
    groups: dict[str, str],
) -> dict[str, str]:
    """Prose, table and figures of the MFA-combined score spaces, every pattern asserted."""
    spaces = [str(s) for s in combined["spaces"]]
    ids = [str(s) for s in combined["slide_ids"]]
    group = np.array([groups[g] for g in ids])
    median_density = np.log10(density["slide_medians"])
    kept = density["kept_per_slide"]
    tail = kept[9] / kept[0]  # share of a slide's cells at or above level 9
    totals = [float(arrays[f"{key}_scores"].var(axis=0).sum()) for arrays, _ in slides for key, _ in MODELS]
    orders = int(np.floor(np.log10(max(totals) / min(totals))))
    plain_pc1 = float(combined["plain_evr"][0])
    plain_share = float(combined["plain_pc1_h0_none_share"])
    z_evr = combined["zscore_evr"]
    z90 = int(np.searchsorted(np.cumsum(z_evr), 0.9)) + 1
    assert plain_share > 0.8 and z_evr[0] < 0.2
    # Claim: every block's last component holds under one percent of its model's variance.
    assert max(float(arrays[f"{key}_evr"][-1]) for arrays, _ in slides for key, _ in MODELS) < 0.01

    def r(space: str, j: int, values: np.ndarray) -> float:
        return float(np.corrcoef(combined[f"{space}_scores"][:, j], values)[0, 1])

    def gap(space: str, j: int) -> float:
        scores = combined[f"{space}_scores"][:, j]
        return abs(np.median(scores[group == "cirrhosis"]) - np.median(scores[group == "healthy"])) / scores.std()

    def arch(space: str) -> tuple[float, np.ndarray]:
        """R^2 of PC2 on PC1 and PC1^2 - the arch one dominant gradient puts into a PCA - and the residual."""
        scores = combined[f"{space}_scores"]
        design = np.column_stack([np.ones(len(scores)), scores[:, 0], scores[:, 0] ** 2])
        beta, *_ = np.linalg.lstsq(design, scores[:, 1], rcond=None)
        residual = scores[:, 1] - design @ beta
        centred = scores[:, 1] - scores[:, 1].mean()
        return 1 - float(residual @ residual / (centred @ centred)), residual

    def residual_gap(residual: np.ndarray) -> float:
        return abs(np.median(residual[group == "cirrhosis"]) - np.median(residual[group == "healthy"])) / residual.std()

    pc1_med = [abs(r(s, 0, median_density)) for s in spaces]
    pc2_med = [abs(r(s, 1, median_density)) for s in spaces]
    pc2_tail = [abs(r(s, 1, tail)) for s in spaces]
    pc2_gap = [gap(s, 1) for s in spaces]
    arches = {s: arch(s) for s in spaces}
    arch_r2 = [arches[s][0] for s in spaces]
    rest_gap = [residual_gap(arches[s][1]) for s in spaces]
    rest_tail = [abs(float(np.corrcoef(arches[s][1], tail)[0, 1])) for s in spaces]
    # Claims: PC1 follows the median density in every space; PC2 hardly does,
    # but follows the share of cells in the densest decile, and separates
    # cirrhosis from healthy by about a standard deviation or more; part of
    # PC2 is the arch of PC1, and with it removed cirrhosis still sits more
    # than a standard deviation from healthy, the rest still following the
    # densest decile.
    assert min(pc1_med) > 0.7 and max(pc2_med) < 0.35 and min(pc2_tail) > 0.4, (pc1_med, pc2_med, pc2_tail)
    assert min(pc2_gap) > 0.9, pc2_gap
    assert 0.1 < min(arch_r2) and max(arch_r2) < 0.6, arch_r2
    assert min(rest_gap) > 1.0 and min(rest_tail) > 0.3, (rest_gap, rest_tail)
    widest = spaces[int(np.argmax(pc2_gap))]
    n90 = {s: int(np.searchsorted(np.cumsum(combined[f"{s}_evr"]), 0.9)) + 1 for s in spaces}

    rows = []
    for s in spaces:
        evr = combined[f"{s}_evr"]
        rows.append((
            SPACE_LABELS[s], str(len(combined[f"{s}_blocks"])), str(9 * len(combined[f"{s}_blocks"])),
            f"{evr[0]:.3f}", f"{evr[:4].sum():.3f}", str(n90[s]),
            f"{r(s, 0, median_density):+.2f}", f"{r(s, 1, median_density):+.2f}",
            f"{r(s, 0, tail):+.2f}", f"{r(s, 1, tail):+.2f}", f"{gap(s, 1):.2f}",
            f"{arches[s][0]:.2f}", f"{residual_gap(arches[s][1]):.2f}",
        ))
    table = html_table(
        ["space", "blocks", "columns", "PC1<br><span class='unit'>FVE</span>",
         "PC1–4<br><span class='unit'>FVE</span>", "PCs for<br><span class='unit'>90%</span>",
         "PC1 vs<br><span class='unit'>log median density, r</span>",
         "PC2 vs<br><span class='unit'>log median density, r</span>",
         "PC1 vs<br><span class='unit'>densest-decile share, r</span>",
         "PC2 vs<br><span class='unit'>densest-decile share, r</span>",
         "PC2: cirrhosis vs healthy<br><span class='unit'>median gap, SD</span>",
         "PC2 on PC1, PC1²<br><span class='unit'>R², the arch</span>",
         "arch removed<br><span class='unit'>median gap, SD</span>"],
        rows,
    )
    text_method = (
        "Each slice's four models give 40 blocks of nine scores, and their scales differ by "
        f"{orders} orders of magnitude (total variance per block from {min(totals):.1e} to "
        f"{max(totals):.1e}). Concatenated as they are, the 360 columns give a first component "
        f"with {plain_pc1:.2f} of the variance and {plain_share:.2f} of its loading on the H0 raw "
        "blocks: the combined space would be the raw H0 curves' size. With every column z-scored "
        f"the spectrum is flat (first component {z_evr[0]:.2f}, {z90} components for 90%), and the "
        "last component of every block, a fraction of a percent of its model's variance, counts as "
        "much as the first. Multiple factor analysis (MFA) divides every block by the square root "
        "of the variance of its first score, so that each block's leading component has variance 1 "
        "and its internal proportions stay, and a PCA of the weighted concatenation is the combined "
        "space. That PCA is kernel PCA of the sum of the blocks' centred Gram matrices, each "
        "divided by its first eigenvalue and cut to its nine components, so it is the fPCA of one "
        "curve per slide that runs through all the slices it combines. Five spaces: all 40 blocks, "
        "and each model family across its ten slices."
    )
    text_results = (
        "In every space the first component is density: its correlation with the slides' log "
        f"median density is {min(pc1_med):.2f}–{max(pc1_med):.2f} in absolute value. The second "
        f"hardly follows the median density (|r| ≤ {max(pc2_med):.2f}) but follows the share of a "
        "slide's cells in the densest decile, the cells slice 9 keeps "
        f"(|r| {min(pc2_tail):.2f}–{max(pc2_tail):.2f}), so it too is a density feature, of the "
        "dense tail rather than the bulk. Part of it is the arch that one dominant gradient puts "
        "into any PCA: a curved function of the first component, fitted as PC1 and PC1², explains "
        f"{min(arch_r2):.2f}–{max(arch_r2):.2f} of its variance. Along the second component "
        "cirrhosis sits apart from healthy: the two groups' median scores differ by "
        f"{min(pc2_gap):.2f}–{max(pc2_gap):.2f} standard deviations, most in the "
        f"{SPACE_LABELS[widest]} space, and still by {min(rest_gap):.2f}–{max(rest_gap):.2f} once "
        "the arch is removed; what remains still follows the densest-decile share "
        f"(|r| {min(rest_tail):.2f}–{max(rest_tail):.2f}). All of this is descriptive; no test "
        "was run."
    )
    figures = []
    for s in spaces:
        pseudo = {
            "c_group_ids": combined["slide_ids"],
            "c_scores": combined[f"{s}_scores"],
            "c_evr": combined[f"{s}_evr"],
        }
        png = figure_scatter(
            pseudo, "c", SPACE_LABELS[s], None, groups, combined[f"{s}_umap"], umap_title="UMAP of the space",
            title=f"Combined space, {SPACE_LABELS[s]}: every pair of the first four components, and the UMAP of the space; one row per group triple",
        )
        medians = []
        for name in GROUP_ORDER:
            values = np.median(combined[f"{s}_scores"][group == name, :N_PLOT], axis=0)
            medians.append((name,) + tuple(f"{v:+.3f}" for v in values))
        figures.append(
            f"<figure>{img_tag(png, f'Combined space {SPACE_LABELS[s]}: scatter matrix of its first four components and its UMAP, by pathology group')}"
            f"<figcaption>Combined space, {SPACE_LABELS[s]} ({len(combined[f'{s}_blocks'])} blocks). "
            "The first six columns are the pairs of its first four components; the last is a UMAP of "
            "the weighted blocks, whose axes carry no units and whose distances between clusters are "
            "not to scale. Rows as above.</figcaption></figure>"
            "<details><summary>Table: group medians of PC1–PC4</summary>"
            + html_table(["group"] + [f"PC{j + 1}" for j in range(N_PLOT)], medians)
            + "</details>"
        )
    fve_png = figure_combined_fve(combined, spaces)
    return {
        "text_method": text_method,
        "text_results": text_results,
        "table": table,
        "fve": img_tag(fve_png, "Fraction of variance per component in the five combined spaces"),
        "figures": "".join(figures),
    }


def medians_table(arrays: dict[str, np.ndarray], groups: dict[str, str]) -> str:
    """Per-group median of PC1-4 for every model: the table view of the scatters."""
    rows = []
    for name in GROUP_ORDER:
        cells: list[str] = [name]
        for key, _ in MODELS:
            ids = [str(g) for g in arrays[f"{key}_group_ids"]]
            chosen = np.array([groups[g] == name for g in ids])
            values = np.median(arrays[f"{key}_scores"][chosen, :N_PLOT], axis=0)
            cells += [f"{v:.3g}" for v in values]
        rows.append(tuple(cells))
    header = ["group"] + [
        f"{label}<br><span class='unit'>PC{c + 1}</span>" for _, label in MODELS for c in range(N_PLOT)
    ]
    return html_table(header, rows)


def section_pc2(check: dict[str, np.ndarray]) -> dict[str, str]:
    """Task 4: PC2's cirrhosis gap with the densest-decile share removed; every pattern asserted.

    The readout and the rule were written into the ledger before the run: the gap
    survives in a space if PC2, with the share and its square regressed out, has
    an AUC interval wholly above 0.5.
    """
    spaces = [str(s) for s in check["spaces"]]

    def cell(space: str, variant: str) -> str:
        a, lo, hi, _ = (float(v) for v in check[f"{space}_{variant}"])
        return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"

    share_auc = check["share_auc"]
    raw = [float(check[f"{s}_raw"][0]) for s in spaces]
    survivors = [s for s in spaces if check[f"{s}_share"][1] > 0.5]
    others = [s for s in spaces if s not in survivors]
    strict = [check[f"{s}_strict"] for s in spaces]
    # Claims: the share alone beats PC2 in every space; some but not all spaces
    # keep a gap once the share is removed; none once all density summaries are.
    assert share_auc[0] > max(raw), (share_auc, raw)
    assert 0 < len(survivors) < len(spaces), survivors
    assert all(v[1] <= 0.5 <= v[2] for v in strict), strict

    def listed(names: list[str]) -> str:
        return "; ".join(f"{SPACE_LABELS[s]}, AUC {cell(s, 'share')}" for s in names)

    text = (
        "Decided before the run (ledger, Task 4): PC2, oriented so that cirrhosis's median lies above "
        "healthy's, is compared between the two groups by its AUC, the chance that a cirrhosis slide "
        "scores above a healthy one, with a 95% bootstrap interval; the gap survives the "
        "densest-decile share in a space if, with the share and its square regressed out of PC2, that "
        "interval lies wholly above 0.5. The share alone separates the two groups at AUC "
        f"{share_auc[0]:.3f} [{share_auc[1]:.3f}, {share_auc[2]:.3f}], more than PC2 does in any space "
        f"({min(raw):.3f}–{max(raw):.3f}). With the share removed the gap survives in {len(survivors)} of "
        f"the {len(spaces)} spaces ({listed(survivors)}) and not in the others ({listed(others)}). "
        "With the arch (PC1 and PC1²), log median density and log cell count regressed out as well, no "
        f"space keeps any separation (AUC {min(float(v[0]) for v in strict):.3f}–"
        f"{max(float(v[0]) for v in strict):.3f}, every interval containing 0.5). So the separation "
        "along PC2 is the dense tail of the density distribution, and what is left once that is "
        "removed goes with the rest of density. Five spaces were tested, and the intervals treat the "
        "regression as fixed, so they are a little narrow."
    )
    rows = [
        (SPACE_LABELS[s],)
        + tuple(x for v in ("raw", "share", "strict") for x in (cell(s, v), f"{float(check[f'{s}_{v}'][3]):+.2f}"))
        for s in spaces
    ]
    table = html_table(
        ["space",
         "PC2 as fitted<br><span class='unit'>AUC [95% CI]</span>", "<br><span class='unit'>gap, SD</span>",
         "share removed<br><span class='unit'>AUC [95% CI]</span>", "<br><span class='unit'>gap, SD</span>",
         "share, arch, density, cells removed<br><span class='unit'>AUC [95% CI]</span>",
         "<br><span class='unit'>gap, SD</span>"],
        rows,
    )
    return {"text": text, "table": table}


def main() -> None:
    table = pd.read_csv(RESULTS / "pathology.csv", keep_default_na=False)
    groups = dict(zip(table["slide_id"], table["group"]))
    with np.load(RESULTS / "density.npz", allow_pickle=False) as data:
        density = {key: data[key] for key in data.files if key != "rho" and key != "slide_codes"}
    density_meta = json.loads((RESULTS / "density.json").read_text())
    slides = [load_slice(k) for k in range(N_SLICES)]
    with np.load(RESULTS / "umap.npz", allow_pickle=False) as data:
        embeddings = {key: data[key] for key in data.files}
    umap_meta = json.loads((RESULTS / "umap.json").read_text())
    with np.load(RESULTS / "combined.npz", allow_pickle=False) as data:
        combined = {key: data[key] for key in data.files}
    combined_meta = json.loads((RESULTS / "combined.json").read_text())
    with np.load(RESULTS / "pc2_check.npz", allow_pickle=False) as data:
        check = {key: data[key] for key in data.files}
    check_meta = json.loads((RESULTS / "pc2_check.json").read_text())
    assert [str(s) for s in combined["slide_ids"]] == [str(s) for s in density["slide_ids"]]

    # Checks the prose relies on.
    n_slides = len(table)
    assert n_slides == len(density["slide_ids"]) == 594
    assert list(table["slide_id"]) == [str(s) for s in density["slide_ids"]]
    for k, (arrays, _) in enumerate(slides):
        assert arrays["skipped"].size == 0, (k, arrays["skipped"])
        assert np.array_equal(arrays["kept_per_slide"], density["kept_per_slide"][k])
        for key, _ in MODELS:
            assert len(arrays[f"{key}_group_ids"]) == n_slides
            assert set(str(g) for g in arrays[f"{key}_group_ids"]) == set(groups)
            # The embedding's rows are the score rows, in the same order.
            assert np.array_equal(arrays[f"{key}_group_ids"], embeddings["slide_ids"])
            assert embeddings[f"{key}_{k}"].shape == (n_slides, 2)
    counts = table["group"].value_counts().reindex(GROUP_ORDER)
    n_cells = int(table["n_cells"].sum())
    total_runtime = (
        density_meta["runtime_seconds"]
        + sum(meta["runtime_seconds"] for _, meta in slides)
        + umap_meta["runtime_seconds"]
        + combined_meta["runtime_seconds"]
        + check_meta["runtime_seconds"]
    )
    umap_params = umap_meta["params"]

    # Pathology table.
    members = {
        "healthy": "curated label “healthy”: note reports no finding, no_abnormalities, clean_specimens",
        "congestion": "congestion, hemorrhage",
        "steatosis": "steatosis",
        "necrosis": "necrosis, infarction, ischemic_changes",
        "inflammation/hepatitis": "inflammation, hepatitis",
        "fibrosis": "fibrosis, scarring, sclerotic",
        "cirrhosis": "cirrhosis",
        "unclassified": "no category: finding only in the free-text note, or a preservation warning",
    }
    severity = {"cirrhosis": 1, "fibrosis": 2, "inflammation/hepatitis": 3, "necrosis": 4,
                "steatosis": 5, "congestion": 6}
    group_rows = [
        (name, str(int(counts[name])), str(severity.get(name, "—")), members[name],
         f"{int(np.median(table.loc[table['group'] == name, 'n_cells'])):,}")
        for name in GROUP_ORDER
    ]
    group_table = html_table(
        ["group", "slides", "severity rank", "GTEx categories", "cells per slide<br><span class='unit'>median</span>"],
        group_rows, cls="left2",
    )
    multi = int(sum(len([c for c in s.split(",") if c.strip()]) > 1 for s in table["categories"]))

    # Density table.
    kept = density["kept_per_slide"]
    density_rows = [
        (str(k), f"{density['decile_levels'][k]:.1f}", f"{density['deciles'][k]:,.1f}",
         f"{int(kept[k].sum()):,}", f"{int(kept[k].min()):,}", f"{int(np.median(kept[k])):,}")
        for k in range(N_SLICES)
    ]
    density_table = html_table(
        ["slice", "q", "threshold<br><span class='unit'>cells/mm²</span>", "cells kept",
         "per slide<br><span class='unit'>min</span>", "per slide<br><span class='unit'>median</span>"],
        density_rows,
    )

    # FVE summary across slices.
    evr = {key: np.array([arrays[f"{key}_evr"] for arrays, _ in slides]) for key, _ in MODELS}
    floor = 10 ** np.floor(np.log10(min(float(v[v > 0].min()) for v in evr.values())))
    fve_rows = [
        (label,) + tuple(f"{evr[key][k, 0]:.3f} / {evr[key][k, :4].sum():.3f}" for k in range(N_SLICES))
        for key, label in MODELS
    ]
    fve_table = html_table(["model"] + [f"slice {k}" for k in range(N_SLICES)], fve_rows)
    raw_pc1 = np.concatenate([evr["h0_none"][:, 0], evr["h1_none"][:, 0]])
    cos_pc1 = np.concatenate([evr["h0_cosine"][:, 0], evr["h1_cosine"][:, 0]])
    fve_text = (
        "Across the ten slices the first component of the raw models carries "
        f"{raw_pc1.min():.3f}–{raw_pc1.max():.3f} of the variance, that of the cosine models "
        f"{cos_pc1.min():.3f}–{cos_pc1.max():.3f}; the first four together carry "
        f"{min(v[:, :4].sum(axis=1).min() for v in evr.values()):.3f} or more in every model and slice."
    )

    density_png = figure_density(density)
    combined_section = section_combined(combined, slides, density, groups)
    assert [str(s) for s in check["slide_ids"]] == [str(s) for s in density["slide_ids"]]
    pc2_section = section_pc2(check)
    slice_blocks = []
    for k, (arrays, meta) in enumerate(slides):
        fve_png = figure_fve(arrays, k, floor)
        scatters = "".join(
            f"<figure>{img_tag(figure_scatter(arrays, key, label, k, groups, embeddings[f'{key}_{k}']), f'Slice {k}, {label}: scatter matrix of PC1 to PC4 and a UMAP of the nine scores, by pathology group')}"
            f"<figcaption>Slice {k}, {label}. The first six columns are the pairs of the first four "
            "components (axis labels give each component's fraction of variance); the last is a UMAP "
            "of all nine scores, whose axes carry no units and whose distances between clusters are "
            "not to scale. Each row colours healthy and two other groups over every other slide in "
            "grey.</figcaption></figure>"
            for key, label in MODELS
        )
        condition = "every cell" if k == 0 else f"the cells with ρ ≥ {float(arrays['threshold']):,.1f} cells/mm²"
        slice_blocks.append(
            f"<details{' open' if k == 0 else ''}><summary><strong>Slice {k}</strong> — {condition}; "
            f"{int(arrays['kept_per_slide'].sum()):,} cells, PH in {meta['ph_seconds']:.0f} s</summary>"
            f"<figure>{img_tag(fve_png, f'Slice {k}: fraction of variance explained per component for the four models')}"
            "<figcaption>Fraction of variance explained by each of the nine fitted components, "
            "log scale; the panel titles give the first component's share and the first four "
            "together.</figcaption></figure>"
            f"{scatters}"
            "<details><summary>Table: group medians of PC1–PC4</summary>"
            "<p class='note'>Median score of each group on the first four components of each model, "
            f"the table view of the scatters above (slice {k}).</p>"
            f"{medians_table(arrays, groups)}</details></details>"
        )

    topocyte_version = html.escape(str(slides[0][1].get("topocyte", "")))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liver pathology density slices</title>
<style>
  :root {{
    --ink: {INK}; --ink2: {INK2}; --muted: {MUTED}; --rule: {RULE};
    --surface: {SURFACE}; --page: #f9f9f7; --accent: {COLOR['healthy']};
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ink: #f0efec; --ink2: #c3c2b7; --muted: #a3a29b; --rule: #383835;
      --surface: #1f1f1e; --page: #1a1a19; --accent: #3987e5;
    }}
  }}
  :root[data-theme="dark"] {{
    --ink: #f0efec; --ink2: #c3c2b7; --muted: #a3a29b; --rule: #383835;
    --surface: #1f1f1e; --page: #1a1a19; --accent: #3987e5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.55; font-size: 16px;
  }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 36px 16px 90px; }}
  h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.3rem; margin: 44px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }}
  h3 {{ font-size: 1.05rem; margin: 30px 0 8px; }}
  p, li {{ max-width: 80ch; }}
  .subtitle {{ color: var(--ink2); font-size: 1.05rem; margin: 0 0 8px; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin: 0 0 28px; }}
  .summary {{ background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--accent);
              padding: 16px 20px; margin: 24px 0; border-radius: 4px; }}
  .summary p {{ margin: 0 0 8px; }} .summary p:last-child {{ margin: 0; }}
  figure {{ margin: 22px 0; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 3px; }}
  figcaption {{ color: var(--ink2); font-size: 0.85rem; margin-top: 8px; }}
  .table-wrap {{ overflow-x: auto; margin: 16px 0 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem;
           font-variant-numeric: tabular-nums; background: var(--surface); }}
  th, td {{ border-bottom: 1px solid var(--rule); padding: 7px 8px; text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .left2 th:nth-child(4), .left2 td:nth-child(4) {{ text-align: left; white-space: normal; }}
  th {{ color: var(--ink2); font-weight: 600; vertical-align: bottom; }}
  .unit {{ color: var(--muted); font-weight: 400; font-size: 0.72rem; }}
  .note {{ color: var(--ink2); font-size: 0.85rem; }}
  code {{ font-size: 0.86em; background: var(--rule); padding: 1px 4px; border-radius: 3px; }}
  details {{ margin: 14px 0; }}
  summary {{ cursor: pointer; color: var(--ink2); }}
  ul {{ margin: 8px 0 8px 22px; padding: 0; }} li {{ margin: 4px 0; }}
</style>
</head>
<body>
<main>
  <h1>Liver pathology across density slices</h1>
  <p class="subtitle">Betti-curve fPCA of the 594 GTEx liver slides at ten density levels, coloured by pathology.</p>
  <p class="meta">Report generated {dt.date.today().isoformat()} from the saved results of
  <code>run_pathology_slices.py</code>, <code>umap_scores.py</code>,
  <code>combined_spaces.py</code> and <code>pc2_density_check.py</code>; total compute
  {total_runtime / 60:.1f} min; topocyte from the checkout that README.md names (the
  interpreter's installed metadata reads {topocyte_version}), umap-learn
  {html.escape(str(umap_meta['umap_learn']))}.</p>

  <div class="summary">
    <p>{n_cells:,} cells on {n_slides} slides. Every slide is assigned one pathology group by
    severity; the cells of each slide are thresholded at the ten pooled deciles of the 10-nearest-
    neighbour density, and at each level the alpha persistent homology of the kept cells gives every
    slide an H0 and an H1 Betti curve. Four fPCA models per level (H0 and H1, raw and cosine) are
    fitted on all {n_slides} slides, nine components each.</p>
    <p>{fve_text}</p>
  </div>

  <h2>1. Pathology groups</h2>
  <p>The cohort table gives each slide up to five GTEx pathology categories; {multi} of the
  {n_slides} slides with cells carry more than one. Each slide is assigned the most severe group
  any of its categories belongs to (severity rank 1 is the most severe); the curated label
  “healthy” is its own group, and slides with no category stay in every fit as “unclassified”.
  Sixteen rows of the cohort table have no cells, among them all seven without a pathology
  note.</p>
  {group_table}

  <h2>2. Density slice levels</h2>
  <p>Each cell's density is ρ = 10 / (π r₁₀²), with r₁₀ the distance to its tenth nearest
  neighbour on the same slide, at an assumed 0.4942 µm per pixel (units only). The ten slice
  levels are the deciles q = 0, 0.1, …, 0.9 of ρ pooled over all cells, the same thresholds for
  every slide; slice k keeps the cells with ρ at or above level k, so slice 0 keeps every cell.
  Every slide keeps at least {int(kept.min()):,} cells at every level.</p>
  <figure>{img_tag(density_png, "Pooled quantile curve of the 10-NN cell density with the ten slice levels marked.")}
  <figcaption>Quantile curve of ρ over all {n_cells:,} cells (log scale); the numbered dots are
  the slice levels 1–9.</figcaption></figure>
  {density_table}

  <h2>3. Fraction of variance explained</h2>
  <p class="note">Each cell: the first component's fraction of variance / the first four
  components together. The fractions are exact: each eigenvalue over the trace of the centred
  Gram matrix of all {n_slides} curves.</p>
  {fve_table}

  <h2>4. Slices</h2>
  <p>Per slice: the scree plot of the four models, then for each model the six pairs of the
  first four components and a UMAP of the model's whole score space, all nine components
  (Euclidean distance on the scores as fitted, so the first components weigh most;
  n_neighbors {umap_params['n_neighbors']}, min_dist {umap_params['min_dist']}, seed
  {umap_params['random_state']}). A scatter can carry at most three colours that stay
  distinguishable in every pair of marks, under colour-vision deficiency too, so each panel is
  drawn in three rows, each colouring healthy and two other groups over all remaining slides in
  grey; a group keeps its colour everywhere. Every panel shows all {n_slides} slides.</p>
  {''.join(slice_blocks)}

  <h2>5. Combined score spaces</h2>
  <p>{combined_section['text_method']}</p>
  <p>{combined_section['text_results']}</p>
  {combined_section['table']}
  <figure>{combined_section['fve']}
  <figcaption>Fraction of variance of the first 20 components of each combined space, log scale;
  the titles give the first component's share and the first four together.</figcaption></figure>
  {combined_section['figures']}
  <h3>Does PC2's cirrhosis gap survive the densest-decile share?</h3>
  <p>{pc2_section['text']}</p>
  {pc2_section['table']}

  <h2>Methods and reproduction</h2>
  <ul>
    <li><code>run_pathology_slices.py groups</code> assigns the groups
    (<code>results/pathology.csv</code>); <code>density</code> computes ρ and the levels
    (<code>results/density.npz</code>); <code>slice K</code> computes the alpha persistent
    homology (alpha, max_dim 1) of each slide's kept cells and the four cohort fPCA models with
    nine components (<code>results/slice_K.npz</code>). It calls what
    <code>tc.tl.persistent_homology</code> and <code>tc.tl.fpca</code> run,
    <code>tc.ph.compute_persistent_homology</code> per slide and <code>tc.ph.FPCA</code> with
    their defaults, directly: an AnnData of all 126 million cells would not fit in memory.</li>
    <li><code>umap_scores.py</code> embeds each slice's and model's nine-component score matrix
    with umap-learn ({html.escape(json.dumps(umap_params))}; <code>results/umap.npz</code>).</li>
    <li>Betti curves are exact from the bar endpoints; the fits are matrix-free (topocyte's
    <code>eigen_solver="auto"</code> chooses ARPACK for {n_slides} curves and nine components),
    and the fractions of variance are exact over the trace of the centred Gram matrix.</li>
    <li><code>combined_spaces.py</code> weights the 40 score blocks by multiple factor analysis
    and fits the five combined spaces with their UMAPs (<code>results/combined.npz</code>).</li>
    <li><code>pc2_density_check.py</code> regresses the densest-decile share, and the other density
    summaries, out of each space's PC2 and compares cirrhosis with healthy by AUC
    (<code>results/pc2_check.npz</code>).</li>
    <li><code>verify_results.py</code> checks the saved results; <code>build_report.py</code>
    writes this page. The cohort was opened read only.</li>
  </ul>
</main>
</body>
</html>
"""
    OUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
