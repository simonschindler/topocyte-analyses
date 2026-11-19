#!/usr/bin/env python
"""Build the self-contained HTML report from the saved results.

Reads the sibling study's ``results/pathology.csv``, ``results/density.npz``
(not its per-cell array) and, for the comparison, its slice, combined and
PC2-check results; and this study's ``results/slice_<k>.npz`` with their
``.json``, ``validation.json``, ``umap.npz``, ``combined.npz`` and
``pc2_check.npz``.  Writes ``report.html`` with every figure embedded as a
base64 PNG.  No number in the prose is typed by hand: all of them are
formatted from those files.

The H1 cosine model's third and fourth components move with the draw of the
random features (``validate_kernel.py``); every axis, table column and
caption that shows them carries a dagger.

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
SOURCE = HERE.parent / "gtex-liver-pathology-slices" / "results"
OUT = HERE / "report.html"
N_SLICES = 10
N_PLOT = 4  # components in the scatter matrix
# Components (0-based) that move with the draw of the random features: ledger Task 1.
UNSTABLE = {"h1_cosine": (2, 3)}
DAGGER = "†"

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
    flagged = UNSTABLE.get(key, ())

    def axis(c: int) -> str:
        return f"PC{c + 1}{DAGGER if c in flagged else ''} ({100 * evr[c]:.1f}%)"

    panels = [(scores[:, i], scores[:, j], axis(i), axis(j)) for i, j in PAIRS]
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
                # Raw scores reach 10^4: one power-of-ten offset per axis, the same style everywhere.
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


def space_stats(combined: dict[str, np.ndarray], density: dict[str, np.ndarray], group: np.ndarray) -> dict[str, dict]:
    """Per combined space: what PC1 and PC2 follow, PC2's cirrhosis gap, and the arch.

    ``*_median_r``: correlation with the slides' log10 median density;
    ``*_share_r``: with the densest-decile share (a slide's cells kept at
    slice 9 over all its cells); ``pc2_gap``: |median(cirrhosis) -
    median(healthy)| on PC2 in standard deviations; ``arch_r2``: R^2 of PC2 on
    PC1 and PC1^2, the arch one dominant gradient puts into any PCA; and the
    gap and share correlation of what is left of PC2 once the arch is removed.
    """
    median_density = np.log10(density["slide_medians"])
    kept = density["kept_per_slide"]
    share = kept[9] / kept[0]
    cirrhosis, healthy = group == "cirrhosis", group == "healthy"

    def gap(values: np.ndarray) -> float:
        return abs(np.median(values[cirrhosis]) - np.median(values[healthy])) / values.std()

    def r(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    stats = {}
    for space in [str(s) for s in combined["spaces"]]:
        scores = combined[f"{space}_scores"]
        pc1, pc2 = scores[:, 0], scores[:, 1]
        design = np.column_stack([np.ones(len(pc1)), pc1, pc1**2])
        beta, *_ = np.linalg.lstsq(design, pc2, rcond=None)
        rest = pc2 - design @ beta
        centred = pc2 - pc2.mean()
        stats[space] = {
            "pc1_median_r": r(pc1, median_density),
            "pc2_median_r": r(pc2, median_density),
            "pc1_share_r": r(pc1, share),
            "pc2_share_r": r(pc2, share),
            "pc2_gap": gap(pc2),
            "arch_r2": 1 - float(rest @ rest / (centred @ centred)),
            "arch_removed_gap": gap(rest),
            "arch_removed_share_r": r(rest, share),
        }
    return stats


def sci(value: float) -> str:
    """A positive number as m × 10^e in HTML, one decimal in the mantissa."""
    exponent = int(np.floor(np.log10(value)))
    return f"{value / 10**exponent:.1f} × 10<sup>{exponent}</sup>".replace("<sup>-", "<sup>−")


def joined(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def interval(lo: float, hi: float, digits: int = 3) -> str:
    """lo–hi, or one value when the two agree at the digits shown."""
    a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
    return a if a == b else f"{a}–{b}"


# A space's name inside a sentence.
SPACE_IN_TEXT = {"all": "all-blocks", "h0_none": "H0 raw", "h1_none": "H1 raw", "h0_cosine": "H0 cosine",
                 "h1_cosine": "H1 cosine"}


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
    groups: dict[str, object],
) -> dict[str, object]:
    """Prose, table and figures of the MFA-combined score spaces, every pattern asserted."""
    spaces = [str(s) for s in combined["spaces"]]
    ids = [str(s) for s in combined["slide_ids"]]
    group = np.array([groups[g] for g in ids])
    totals = [float(arrays[f"{key}_scores"].var(axis=0).sum()) for arrays, _ in slides for key, _ in MODELS]
    orders = int(np.floor(np.log10(max(totals) / min(totals))))
    plain_pc1 = float(combined["plain_evr"][0])
    plain_share = float(combined["plain_pc1_h0_none_share"])
    z_evr = combined["zscore_evr"]
    z90 = int(np.searchsorted(np.cumsum(z_evr), 0.9)) + 1
    assert plain_share > 0.8 and z_evr[0] < 0.2
    # Claim: every block's last component holds a small share of its model's variance.
    last = max(float(arrays[f"{key}_evr"][-1]) for arrays, _ in slides for key, _ in MODELS)
    assert last < 0.02, last

    stats = space_stats(combined, density, group)
    # Spaces whose PC1 is density, and among them those whose PC2 is the dense tail.
    dense = [s for s in spaces if abs(stats[s]["pc1_median_r"]) > 0.7]
    pc1_med = [abs(stats[s]["pc1_median_r"]) for s in dense]
    gap_dense = [stats[s]["pc2_gap"] for s in dense]
    tail = [s for s in dense if abs(stats[s]["pc2_share_r"]) > 0.4 and abs(stats[s]["pc2_median_r"]) < 0.3]
    off_tail = [s for s in dense if s not in tail]
    arch_r2 = [stats[s]["arch_r2"] for s in spaces]
    h1c = stats["h1_cosine"]
    # Claims: PC1 follows the median density in every space but the H1 cosine
    # one, where PC1 follows the densest-decile share and PC2 the median
    # density; along PC2 cirrhosis sits more than a standard deviation from
    # healthy where PC1 is density, less than half of one in the H1 cosine
    # space; there PC2 follows the densest-decile share and not the median
    # density, except in one space, where it follows the median density more;
    # PC2 is hardly an arch of PC1.
    assert [s for s in spaces if s not in dense] == ["h1_cosine"], dense
    assert abs(h1c["pc1_share_r"]) > 0.7 and abs(h1c["pc1_median_r"]) < 0.35 and abs(h1c["pc2_median_r"]) > 0.7, h1c
    assert min(gap_dense) > 1.0 and h1c["pc2_gap"] < 0.5, (gap_dense, h1c)
    assert len(off_tail) == 1, off_tail
    off = stats[off_tail[0]]
    assert abs(off["pc2_median_r"]) > abs(off["pc2_share_r"]) and abs(off["pc2_share_r"]) < 0.3, off
    assert max(arch_r2) < 0.2, arch_r2
    widest = dense[int(np.argmax(gap_dense))]
    n90 = {s: int(np.searchsorted(np.cumsum(combined[f"{s}_evr"]), 0.9)) + 1 for s in spaces}

    def dagger_share(space: str, component: int) -> float:
        """Share of a combined component's squared loading on the dagger columns (H1 cosine PC3-4)."""
        columns, flags = [], []
        for block, weight in zip(combined[f"{space}_blocks"], combined[f"{space}_weights"]):
            k, name = str(block).split(":")
            scores = slides[int(k)][0][f"{name}_scores"]
            columns.append(scores * weight)
            flags += [c in UNSTABLE.get(name, ()) for c in range(scores.shape[1])]
        weighted = np.hstack(columns)
        weighted = weighted - weighted.mean(axis=0)
        score = combined[f"{space}_scores"][:, component]
        loading = score @ weighted / (score @ score)  # the component's unit loading vector
        assert abs(loading @ loading - 1) < 1e-8
        return float(np.sum(loading[np.array(flags)] ** 2))

    daggers = {s: (dagger_share(s, 0), dagger_share(s, 1)) for s in ("all", "h1_cosine")}
    dagger_max = max(max(v) for v in daggers.values())
    # Claim: the dagger columns hardly load on PC1 or PC2 of the spaces that hold them.
    assert dagger_max < 0.01, daggers

    rows = []
    for s in spaces:
        evr = combined[f"{s}_evr"]
        rows.append((
            SPACE_LABELS[s], str(len(combined[f"{s}_blocks"])), str(9 * len(combined[f"{s}_blocks"])),
            f"{evr[0]:.3f}", f"{evr[:4].sum():.3f}", str(n90[s]),
            f"{stats[s]['pc1_median_r']:+.2f}", f"{stats[s]['pc2_median_r']:+.2f}",
            f"{stats[s]['pc1_share_r']:+.2f}", f"{stats[s]['pc2_share_r']:+.2f}", f"{stats[s]['pc2_gap']:.2f}",
            f"{stats[s]['arch_r2']:.2f}", f"{stats[s]['arch_removed_gap']:.2f}",
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
        "blocks: the combined space would be the raw H0 blocks alone. With every column z-scored "
        f"the spectrum is flat (first component {z_evr[0]:.2f}, {z90} components for 90%), and the "
        f"last component of every block, at most {100 * last:.1f}% of its model's variance, counts as "
        "much as the first. Multiple factor analysis (MFA) divides every block by the square root "
        "of the variance of its first score, so that each block's leading component has variance 1 "
        "and its internal proportions stay, and a PCA of the weighted concatenation is the combined "
        "space. That PCA is kernel PCA of the sum of the blocks' centred Gram matrices, each "
        "divided by its first eigenvalue and cut to its nine components: one kernel PCA over all "
        "the slices it combines. Five spaces: all 40 blocks, and each model family across its ten "
        "slices."
    )
    def short(names: list[str]) -> str:
        return ", ".join(SPACE_SHORT[s] for s in names)

    tail_share = [abs(stats[s]["pc2_share_r"]) for s in tail]
    tail_median = [abs(stats[s]["pc2_median_r"]) for s in tail]
    text_results = (
        f"In {len(dense)} of the {len(spaces)} spaces ({short(dense)}) the first component is density: "
        f"its correlation with the slides' log median density is {min(pc1_med):.2f}–{max(pc1_med):.2f} "
        "in absolute value. In the H1 cosine space the first two swap: PC1 follows the share of a "
        "slide's cells in the densest decile, the cells slice 9 keeps "
        f"(|r| {abs(h1c['pc1_share_r']):.2f}, against {abs(h1c['pc1_median_r']):.2f} for the median "
        f"density), and PC2 follows the median density (|r| {abs(h1c['pc2_median_r']):.2f}). Along the "
        "second component cirrhosis sits apart from healthy where the first is density: the two "
        f"groups' median scores differ by {min(gap_dense):.2f}–{max(gap_dense):.2f} standard "
        f"deviations, most in the {SPACE_IN_TEXT[widest]} space, against {h1c['pc2_gap']:.2f} in the "
        f"H1 cosine space. In the {joined([SPACE_IN_TEXT[s] for s in tail])} spaces that second "
        "component follows the densest-decile share "
        f"(|r| {min(tail_share):.2f}–{max(tail_share):.2f}) rather than the median density "
        f"(|r| ≤ {max(tail_median):.2f}): a density feature of the dense tail rather than of the "
        f"bulk. In the {SPACE_IN_TEXT[off_tail[0]]} space it follows the median density more "
        f"(|r| {abs(off['pc2_median_r']):.2f}, against {abs(off['pc2_share_r']):.2f} for the share). "
        "The second component is hardly the arch that one dominant gradient puts into any PCA: PC1 "
        f"and PC1² explain {min(arch_r2):.2f}–{max(arch_r2):.2f} of its variance. The H1 cosine "
        f"blocks' PC3{DAGGER} and PC4{DAGGER} carry at most {100 * dagger_max:.1f}% of the squared "
        "loadings of PC1 and PC2 in the all-blocks and H1 cosine spaces, so the combined spaces' "
        "leading components rest on components that do not move with the draw. All of this is "
        "descriptive; no test was run."
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
        "dense": dense,
        "gap_dense": gap_dense,
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
        f"{label}<br><span class='unit'>PC{c + 1}{DAGGER if c in UNSTABLE.get(key, ()) else ''}</span>"
        for key, label in MODELS for c in range(N_PLOT)
    ]
    return html_table(header, rows)


def section_pc2(check: dict[str, np.ndarray]) -> dict[str, object]:
    """PC2's cirrhosis gap with the densest-decile share removed; every pattern asserted.

    The readout and the rule are the Betti study's (its ledger, Task 4), adopted
    here before any slice was fitted (ledger, Task 1): the gap
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
    reversed_share = [s for s in spaces if check[f"{s}_share"][2] < 0.5]
    vanished = [s for s in spaces if s not in survivors and s not in reversed_share]
    strict = {s: check[f"{s}_strict"] for s in spaces}
    strict_below = [s for s in spaces if strict[s][2] < 0.5]
    # Claims: the share alone beats PC2 in every space; some but not all spaces
    # keep the gap once the share is removed, in some it vanishes and in some it
    # reverses; with all density summaries removed no space keeps cirrhosis
    # above healthy, and some reverse.
    assert share_auc[0] > max(raw), (share_auc, raw)
    assert 0 < len(survivors) < len(spaces), survivors
    assert vanished and reversed_share, (vanished, reversed_share)
    assert all(v[1] <= 0.5 for v in strict.values()) and strict_below, strict

    def listed(names: list[str], variant: str) -> str:
        return "; ".join(f"{SPACE_LABELS[s]}, AUC {cell(s, variant)}" for s in names)

    text = (
        "Decided before the Betti study's run (its ledger, Task 4) and adopted here unchanged before any "
        "slice was fitted (ledger, Task 1): PC2, oriented so that cirrhosis's median lies above "
        "healthy's, is compared between the two groups by its AUC, the chance that a cirrhosis slide "
        "scores above a healthy one, with a 95% bootstrap interval; the gap survives the "
        "densest-decile share in a space if, with the share and its square regressed out of PC2, that "
        "interval lies wholly above 0.5. The share alone separates the two groups at AUC "
        f"{share_auc[0]:.3f} [{share_auc[1]:.3f}, {share_auc[2]:.3f}], more than PC2 does in any space "
        f"({min(raw):.3f}–{max(raw):.3f}). With the share removed the gap survives in {len(survivors)} of "
        f"the {len(spaces)} spaces ({listed(survivors, 'share')}), vanishes in "
        f"{listed(vanished, 'share')}, and reverses in {listed(reversed_share, 'share')}: there "
        "cirrhosis now scores below healthy. With the arch (PC1 and PC1²), log median density and log "
        "cell count regressed out as well, no space keeps cirrhosis above healthy (AUC "
        f"{min(float(v[0]) for v in strict.values()):.3f}–{max(float(v[0]) for v in strict.values()):.3f}), "
        f"and in {len(strict_below)} ({listed(strict_below, 'strict')}) the interval lies wholly below "
        "0.5, the gap reversed. So, as with the Betti curves, the separation along PC2 is density: the "
        "share alone separates the groups better than PC2 does, and what PC2 keeps beyond the share "
        "goes with the rest of density; once all of it is regressed out, what is left points, if "
        "anywhere, the other way. Five spaces were tested, and the intervals treat the regression as "
        "fixed, so they are a little narrow."
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
    return {"text": text, "table": table, "survivors": survivors, "strict_below": strict_below}


def section_kernel(slides: list[tuple[dict[str, np.ndarray], dict]], validation: dict) -> dict[str, str]:
    """The kernel, its approximation, the bandwidths and the measured error; every pattern asserted."""
    first = slides[0][0]
    n_features = int(first["h0_features"].shape[1])
    bandwidth = np.array([arrays["bandwidth"] for arrays, _ in slides])
    finite = np.array([np.median(arrays["n_finite"], axis=0) for arrays, _ in slides])
    pairs_slice0 = float(np.median((2 * first["n_finite"].astype(np.float64)) ** 2))
    rows = [
        (str(k), f"{bandwidth[k, 0]:.4g}", f"{bandwidth[k, 1]:.4g}",
         f"{int(finite[k, 0]):,}", f"{int(finite[k, 1]):,}",
         f"{meta['h0_feature_seconds']:.0f}", f"{meta['h1_feature_seconds']:.0f}")
        for k, (arrays, meta) in enumerate(slides)
    ]
    table = html_table(
        ["slice", "bandwidth h, H0<br><span class='unit'>px²</span>",
         "bandwidth h, H1<br><span class='unit'>px²</span>",
         "finite H0 bars<br><span class='unit'>median per slide</span>",
         "finite H1 bars<br><span class='unit'>median per slide</span>",
         "H0 features<br><span class='unit'>seconds</span>", "H1 features<br><span class='unit'>seconds</span>"],
        rows,
    )
    small = validation["small_slides"]
    v0, v1 = validation["h0"], validation["h1"]
    e0, e1 = v0["exact"], v1["exact"]
    sweep = {int(m): part["mean"] for m, part in v1["feature_count"].items()}
    counts = sorted(sweep)
    draws = validation["stability"]["draws"]
    corr = {name: np.array([d["abs_corr_pc1_4"] for d in draws[name]]) for name in draws}
    others = ("h0_none", "h1_none", "h0_cosine")
    others_min = min(float(corr[name].min()) for name in others)
    h1c = corr["h1_cosine"]
    equivalence = max(v0["equivalence"]["relative_difference"], v1["equivalence"]["relative_difference"])
    # Claims: the features are gudhi's kernel_approx; H0 entries within a few
    # percent, H1 entries off by about a tenth, and the H1 entries the emptier;
    # the H1 error falls slowly with more features; the three other models keep
    # PC1-4 across draws, the H1 cosine model PC1-2 but not PC3-4.
    assert equivalence < 1e-10, equivalence
    assert e0["relative_error_max"] < 0.05 and 0.05 < e1["relative_error_median"] < 0.25, (e0, e1)
    assert e1["fill_median"] < e0["fill_median"], (e0, e1)
    assert all(sweep[a] > sweep[b] for a, b in zip(counts, counts[1:])) and sweep[counts[-1]] > 0.005, sweep
    assert others_min > 0.95, others_min
    assert h1c[:, :2].min() > 0.95 and h1c[:, 2:].max() < 0.95, h1c

    text_kernel = (
        "The persistence scale-space kernel of Reininghaus and colleagues (CVPR 2015), as gudhi "
        f"{html.escape(validation['gudhi'])} implements it, puts a Gaussian of bandwidth h on every point "
        "of a diagram and a negative copy on the point's mirror image across the diagonal, and takes "
        "the inner product of two such sums: k(D, E) = ½ Σ<sub>p ∈ D ∪ D̄</sub> Σ<sub>q ∈ E ∪ Ē</sub> "
        "w(p) w(q) exp(−|p − q|² / 2h²) / (√(2π) h), with w = +1 on or above the diagonal and −1 below "
        "it. A point near the diagonal nearly cancels with its mirror image, so short bars weigh little, "
        "and points much farther apart than h hardly interact. An exact entry needs one Gaussian per "
        "pair of points of the two diagrams and their mirror images, a median "
        f"{sci(pairs_slice0)} per H0 entry at slice 0, so "
        "the kernel goes through gudhi's own approximation, random Fourier features: "
        f"{n_features:,} random cosines z(p) (scikit-learn's <code>RBFSampler</code>, which gudhi's "
        "<code>kernel_approx</code> takes) give each slide one vector φ(D) = Σ<sub>p ∈ D ∪ D̄</sub> "
        "w(p) z(p), and k(D, E) ≈ c φ(D) · φ(E) with c = 1 / (2√(2π) h). Kernel PCA of the slides is "
        "then the PCA of their vectors √c φ (raw), or of the same vectors scaled to unit length "
        "(cosine, k(D, E) / √(k(D, D) k(E, E))). The points are the alpha filtration's squared radii "
        "in px², the units the Betti curves used; each slide's one essential H0 bar is left out, since "
        "the kernel takes only finite points. The bandwidth h of a slice and dimension is the median "
        "persistence (death − birth) of all its finite bars, pooled over the slides: set before any "
        "fit, and not tuned against the groups."
    )
    stable_text = ", ".join(f"PC{j + 1} {interval(h1c[:, j].min(), h1c[:, j].max())}" for j in range(4))
    text_error = (
        "How good the approximation is was measured at slice 9 on the six slides with the fewest kept "
        f"cells ({min(small['h0_finite_bars']):,}–{max(small['h0_finite_bars']):,} finite H0 bars, "
        f"{min(small['h1_finite_bars']):,}–{max(small['h1_finite_bars']):,} finite H1 bars), against "
        "gudhi's exact kernel (<code>validate_kernel.py</code>). The once-per-slide vectors reproduce "
        f"gudhi's own <code>kernel_approx</code> route to a relative {equivalence:.0e}. Against the "
        f"exact kernel the saved features (seed 0) put the H0 entries off by a median "
        f"{100 * e0['relative_error_median']:.1f}% (at most {100 * e0['relative_error_max']:.1f}%) and "
        f"their cosines by at most {e0['cosine_abs_error_max']:.4f}; the H1 entries by a median "
        f"{100 * e1['relative_error_median']:.1f}% (at most {100 * e1['relative_error_max']:.1f}%) and "
        f"their cosines by up to {e1['cosine_abs_error_max']:.3f}, against off-diagonal cosines of "
        f"{e1['exact_cosine_min']:.2f}–{e1['exact_cosine_max']:.2f}. The H1 kernel is the harder one: "
        f"an entry is a median {sci(e1['fill_median'])} of the largest value two diagrams of its sizes "
        f"can give (every Gaussian at its peak), against {sci(e0['fill_median'])} for H0, so the same "
        "random-feature noise weighs more. More features help slowly: the H1 median error is "
        + joined([f"{100 * sweep[m]:.1f}%" for m in counts])
        + " with " + joined([f"{m:,}" for m in counts]) + " features "
        "(mean of five draws each). What matters downstream is whether the components move with the draw. "
        "Kernel PCA of all slides at slice 9 from three other draws (1,000 features with another seed, "
        "4,000 features with two seeds) correlates with the saved scores at |r| ≥ "
        f"{others_min:.3f} on PC1–4 of the H0 raw, H1 raw and H0 cosine models. The H1 cosine model "
        f"keeps PC1 and PC2 but not PC3 and PC4 (|r| {stable_text}): those two are limited by the "
        f"approximation, and carry a dagger ({DAGGER}) wherever they appear. The ledger (Task 1) kept "
        f"{n_features:,} features with these numbers at hand. They were measured at slice 9 only."
    )
    return {"text_kernel": text_kernel, "text_error": text_error, "table": table}


def section_comparison(
    slides: list[tuple[dict[str, np.ndarray], dict]],
    betti: list[dict[str, np.ndarray]],
    combined: dict[str, np.ndarray],
    betti_combined: dict[str, np.ndarray],
    check: dict[str, np.ndarray],
    betti_check: dict[str, np.ndarray],
    density: dict[str, np.ndarray],
    groups: dict[str, str],
) -> dict[str, object]:
    """The key numbers of the Betti study and this one side by side, and how far their components agree."""
    ours = [arrays for arrays, _ in slides]
    for a, b in zip(ours, betti):
        for key, _ in MODELS:
            assert np.array_equal(a[f"{key}_group_ids"], b[f"{key}_group_ids"])
    assert np.array_equal(combined["slide_ids"], betti_combined["slide_ids"])
    spaces = [str(s) for s in combined["spaces"]]
    assert spaces == [str(s) for s in betti_combined["spaces"]] == [str(s) for s in betti_check["spaces"]]
    group = np.array([groups[str(g)] for g in combined["slide_ids"]])
    studies = {"betti": betti, "pssk": ours}
    spaces_of = {"betti": betti_combined, "pssk": combined}
    checks = {"betti": betti_check, "pssk": check}
    stats = {x: space_stats(spaces_of[x], density, group) for x in studies}

    def span(values) -> str:
        values = [float(v) for v in values]
        return f"{min(values):.3f}–{max(values):.3f}"

    def both(fn) -> tuple[str, str]:
        return fn("betti"), fn("pssk")

    def survivors(x: str) -> list[str]:
        return [s for s in spaces if checks[x][f"{s}_share"][1] > 0.5]

    def intervals(x: str, variant: str) -> str:
        above = sum(float(checks[x][f"{s}_{variant}"][1]) > 0.5 for s in spaces)
        below = sum(float(checks[x][f"{s}_{variant}"][2]) < 0.5 for s in spaces)
        return f"{above} above 0.5, {below} below"

    rows = [
        (f"{label}: PC1's fraction of variance",)
        + both(lambda x, key=key: span(a[f"{key}_evr"][0] for a in studies[x]))
        for key, label in MODELS
    ] + [
        ("PC1–4's fraction of variance, lowest model and slice",)
        + both(lambda x: f"{min(float(a[f'{k}_evr'][:4].sum()) for a in studies[x] for k, _ in MODELS):.3f}"),
        ("Combined, all 40 blocks: PC1's fraction of variance",)
        + both(lambda x: f"{float(spaces_of[x]['all_evr'][0]):.3f}"),
        ("Combined: spaces whose PC1 follows log median density, |r| > 0.7",)
        + both(lambda x: f"{sum(abs(stats[x][s]['pc1_median_r']) > 0.7 for s in spaces)} of {len(spaces)}"),
        ("Combined: PC1 with log median density, |r|",)
        + both(lambda x: span(abs(stats[x][s]["pc1_median_r"]) for s in spaces)),
        ("Combined: PC2 with the densest-decile share, |r|",)
        + both(lambda x: span(abs(stats[x][s]["pc2_share_r"]) for s in spaces)),
        ("Combined: PC2, cirrhosis vs healthy, median gap in SD",)
        + both(lambda x: span(stats[x][s]["pc2_gap"] for s in spaces)),
        ("Combined: PC2 on PC1 and PC1², R² (the arch)",)
        + both(lambda x: span(stats[x][s]["arch_r2"] for s in spaces)),
        ("PC2 check: PC2 as fitted, AUC",)
        + both(lambda x: span(checks[x][f"{s}_raw"][0] for s in spaces)),
        ("PC2 check: share removed, AUC intervals",)
        + both(lambda x: intervals(x, "share")),
        ("PC2 check: share, arch, density and cells removed, AUC",)
        + both(lambda x: span(checks[x][f"{s}_strict"][0] for s in spaces)),
        ("PC2 check: share, arch, density and cells removed, AUC intervals",)
        + both(lambda x: intervals(x, "strict")),
    ]
    table = html_table(["", "Betti curves", "persistence scale-space kernel"], rows, cls="wrap1")

    def agree(a: np.ndarray, b: np.ndarray, j: int) -> float:
        return abs(float(np.corrcoef(a[:, j], b[:, j])[0, 1]))

    per_slice = {
        key: np.array([[agree(ours[k][f"{key}_scores"], betti[k][f"{key}_scores"], j) for j in range(2)]
                       for k in range(N_SLICES)])
        for key, _ in MODELS
    }
    slice_rows = [
        (label,) + tuple(f"{per_slice[key][k, 0]:.2f} / {per_slice[key][k, 1]:.2f}" for k in range(N_SLICES))
        for key, label in MODELS
    ]
    slice_table = html_table(["model"] + [f"slice {k}" for k in range(N_SLICES)], slice_rows)
    per_space = {s: [agree(combined[f"{s}_scores"], betti_combined[f"{s}_scores"], j) for j in range(2)] for s in spaces}
    space_rows = [(SPACE_LABELS[s], f"{per_space[s][0]:.2f}", f"{per_space[s][1]:.2f}") for s in spaces]
    space_table = html_table(["combined space", "PC1, |r|", "PC2, |r|"], space_rows)
    higher = {
        key: sum(float(ours[k][f"{key}_evr"][0]) > float(betti[k][f"{key}_evr"][0]) for k in range(N_SLICES))
        for key, _ in MODELS
    }
    raw_agree = min(float(per_slice[key][:, 0].min()) for key in ("h0_none", "h1_none"))
    cosine = {key: per_slice[key][:, 0] for key in ("h0_cosine", "h1_cosine")}
    near = [s for s in spaces if s != "h1_cosine"]
    arch = {x: span(stats[x][s]["arch_r2"] for s in spaces) for x in studies}
    kept = {x: survivors(x) for x in studies}
    # Claims: the kernel puts more variance into PC1 than the Betti curves at
    # most slices of every model; the raw models' PC1 agree at every slice;
    # each cosine model's PC1 agrees at slice 0 and not in the denser half;
    # the combined PC1 agrees in every space but the H1 cosine one.
    assert min(higher.values()) >= 8, higher
    assert raw_agree > 0.9, raw_agree
    for key, values in cosine.items():
        assert values[0] > 0.85 and values[5:].max() < 0.6, (key, values)
    assert min(per_space[s][0] for s in near) > 0.8 and per_space["h1_cosine"][0] < 0.3, per_space
    assert all(float(checks[x][f"{s}_strict"][1]) <= 0.5 for x in studies for s in spaces)
    def where(key: str) -> str:
        lower = [k for k in range(N_SLICES)
                 if float(ours[k][f"{key}_evr"][0]) <= float(betti[k][f"{key}_evr"][0])]
        return "" if not lower else " (not at slice " + joined([str(k) for k in lower]) + ")"

    grouped = {}
    for key, _ in MODELS:
        grouped.setdefault(higher[key], []).append(key)
    phrases = [
        ("at every slice for " if n == N_SLICES else f"at {n} of the {N_SLICES} slices for ")
        + joined([SPACE_IN_TEXT[key] + where(key) for key in keys])
        for n, keys in sorted(grouped.items(), reverse=True)
    ]
    text = (
        "The persistence scale-space kernel puts more of each model's variance into the first "
        "component than the Betti curves do: " + "; ".join(phrases)
        + ". In the raw models the two kernels see the same leading structure: their PC1 scores "
        f"correlate at |r| ≥ {raw_agree:.2f} at every slice. The cosine models agree at the sparsest "
        f"slice (PC1 |r| {cosine['h0_cosine'][0]:.2f} for H0, {cosine['h1_cosine'][0]:.2f} for H1, at "
        "slice 0) and part as the slices get denser (at most "
        f"{cosine['h0_cosine'][5:].max():.2f} and {cosine['h1_cosine'][5:].max():.2f} at slices 5–9). "
        "The combined spaces follow: PC1 agrees at |r| "
        f"{min(per_space[s][0] for s in near):.2f}–{max(per_space[s][0] for s in near):.2f} in every "
        f"space but the H1 cosine one ({per_space['h1_cosine'][0]:.2f}), whose PC1 here follows the "
        "densest-decile share (section 6). Under the same pre-registered rule the PC2 gap survives "
        f"the densest-decile share in {len(kept['pssk'])} of {len(spaces)} spaces "
        f"(Betti curves: {len(kept['betti'])}), and survives all density summaries in none, in either "
        f"study. The arch, {arch['betti']} of PC2's variance with the Betti curves, is {arch['pssk']} "
        "here."
    )
    return {
        "text": text,
        "table": table,
        "slice_table": slice_table,
        "space_table": space_table,
        "raw_agree": raw_agree,
        "survivors": kept,
    }


def main() -> None:
    table = pd.read_csv(SOURCE / "pathology.csv", keep_default_na=False)
    groups = dict(zip(table["slide_id"], table["group"]))
    with np.load(SOURCE / "density.npz", allow_pickle=False) as data:
        density = {key: data[key] for key in data.files if key != "rho" and key != "slide_codes"}
    slides = [load_slice(k) for k in range(N_SLICES)]
    validation = json.loads((RESULTS / "validation.json").read_text())
    betti = []
    for k in range(N_SLICES):
        with np.load(SOURCE / f"slice_{k}.npz", allow_pickle=False) as data:
            betti.append({key: data[key] for key in data.files})
    with np.load(SOURCE / "combined.npz", allow_pickle=False) as data:
        betti_combined = {key: data[key] for key in data.files}
    with np.load(SOURCE / "pc2_check.npz", allow_pickle=False) as data:
        betti_check = {key: data[key] for key in data.files}
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
    # The density step is the sibling study's and is not counted.
    total_runtime = (
        sum(meta["runtime_seconds"] for _, meta in slides)
        + validation["runtime_seconds"]
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
    kernel_section = section_kernel(slides, validation)
    combined_section = section_combined(combined, slides, density, groups)
    assert [str(s) for s in check["slide_ids"]] == [str(s) for s in density["slide_ids"]]
    pc2_section = section_pc2(check)
    comparison = section_comparison(slides, betti, combined, betti_combined, check, betti_check, density, groups)
    dagger_note = {
        key: f" {DAGGER} PC{UNSTABLE[key][0] + 1} and PC{UNSTABLE[key][1] + 1} of this model move with the draw "
        "of the random features (section 3)."
        for key in UNSTABLE
    }
    slice_blocks = []
    for k, (arrays, meta) in enumerate(slides):
        fve_png = figure_fve(arrays, k, floor)
        scatters = "".join(
            f"<figure>{img_tag(figure_scatter(arrays, key, label, k, groups, embeddings[f'{key}_{k}']), f'Slice {k}, {label}: scatter matrix of PC1 to PC4 and a UMAP of the nine scores, by pathology group')}"
            f"<figcaption>Slice {k}, {label}. The first six columns are the pairs of the first four "
            "components (axis labels give each component's fraction of variance); the last is a UMAP "
            "of all nine scores, whose axes carry no units and whose distances between clusters are "
            "not to scale. Each row colours healthy and two other groups over every other slide in "
            f"grey.{dagger_note.get(key, '')}</figcaption></figure>"
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
    n_features = int(slides[0][0]["h0_features"].shape[1])
    gap_dense = combined_section["gap_dense"]
    summary_results = (
        "Combined across slices by multiple factor analysis, the first component is density in "
        f"{len(combined_section['dense'])} of the 5 spaces, and along the second cirrhosis sits "
        f"{min(gap_dense):.2f}–{max(gap_dense):.2f} standard deviations from healthy there; with the "
        f"densest-decile share removed the gap survives in {len(pc2_section['survivors'])} of 5 spaces "
        f"(Betti curves: {len(comparison['survivors']['betti'])}), with all density summaries removed "
        f"in none, and reverses in {len(pc2_section['strict_below'])}. Against the Betti curves, the raw "
        f"models' first components agree at every slice (|r| ≥ {comparison['raw_agree']:.2f}); the "
        "cosine models' agree at the sparsest slice and part at the dense ones. The approximate "
        f"kernel is within {100 * validation['h0']['exact']['relative_error_max']:.1f}% of the exact "
        f"one for H0 and about {100 * validation['h1']['exact']['relative_error_median']:.0f}% off for "
        f"H1, whose cosine model's PC3 and PC4 move with the draw ({DAGGER})."
    )
    comparison_text = comparison["text"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liver pathology PSSK slices</title>
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
  .wrap1 td:first-child {{ white-space: normal; min-width: 16em; }}
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
  <h1>Liver pathology across density slices, with the persistence scale-space kernel</h1>
  <p class="subtitle">Kernel PCA of the 594 GTEx liver slides' persistence diagrams at ten density
  levels, coloured by pathology: the Betti-curve study repeated with gudhi's
  <code>PersistenceScaleSpaceKernel</code>.</p>
  <p class="meta">Report generated {dt.date.today().isoformat()} from the saved results of
  <code>run_pssk_slices.py</code>, <code>validate_kernel.py</code>, <code>umap_scores.py</code>,
  <code>combined_spaces.py</code> and <code>pc2_density_check.py</code>, with the groups, densities
  and slice levels of the sibling study <code>../gtex-liver-pathology-slices</code>; total compute
  {total_runtime / 60:.1f} min (the sibling study's density step not counted); topocyte from the
  checkout that README.md names (the interpreter's installed metadata reads {topocyte_version}),
  gudhi {html.escape(str(validation['gudhi']))}, umap-learn
  {html.escape(str(umap_meta['umap_learn']))}.</p>

  <div class="summary">
    <p>{n_cells:,} cells on {n_slides} slides, grouped by pathology and thresholded at the ten pooled
    deciles of the 10-nearest-neighbour density exactly as in the Betti-curve study; at each level the
    alpha persistent homology of the kept cells gives every slide an H0 and an H1 persistence
    diagram. Where that study compared the diagrams through their Betti curves, this one uses the
    persistence scale-space kernel, approximated by {n_features:,} random Fourier features: four
    kernel PCA models per level (H0 and H1, raw and cosine) on all {n_slides} slides, nine
    components each.</p>
    <p>{fve_text}</p>
    <p>{summary_results}</p>
  </div>

  <h2>1. Pathology groups</h2>
  <p>The groups are the sibling study's. The cohort table gives each slide up to five GTEx pathology
  categories; {multi} of the {n_slides} slides with cells carry more than one. Each slide is assigned
  the most severe group any of its categories belongs to (severity rank 1 is the most severe); the
  curated label “healthy” is its own group, and slides with no category stay in every fit as
  “unclassified”.</p>
  {group_table}

  <h2>2. Density slice levels</h2>
  <p>The levels are the sibling study's. Each cell's density is ρ = 10 / (π r₁₀²), with r₁₀ the
  distance to its tenth nearest neighbour on the same slide, at an assumed 0.4942 µm per pixel
  (units only). The ten slice levels are the deciles q = 0, 0.1, …, 0.9 of ρ pooled over all cells,
  the same thresholds for every slide; slice k keeps the cells with ρ at or above level k, so slice 0
  keeps every cell. Every slide keeps at least {int(kept.min()):,} cells at every level.</p>
  <figure>{img_tag(density_png, "Pooled quantile curve of the 10-NN cell density with the ten slice levels marked.")}
  <figcaption>Quantile curve of ρ over all {n_cells:,} cells (log scale); the numbered dots are
  the slice levels 1–9.</figcaption></figure>
  {density_table}

  <h2>3. The kernel</h2>
  <p>{kernel_section['text_kernel']}</p>
  {kernel_section['table']}
  <p class="note">Bandwidth of each slice and dimension, the median number of finite bars per slide,
  and the time the features of all {n_slides} slides took.</p>
  <h3>How close the approximation is</h3>
  <p>{kernel_section['text_error']}</p>

  <h2>4. Fraction of variance explained</h2>
  <p class="note">Each cell: the first component's fraction of variance / the first four
  components together. Each fraction is an eigenvalue of the approximate kernel's centred Gram
  matrix over its trace, the full spectrum of all {n_slides} slides.</p>
  {fve_table}

  <h2>5. Slices</h2>
  <p>Per slice: the scree plot of the four models, then for each model the six pairs of the
  first four components and a UMAP of the model's whole score space, all nine components
  (Euclidean distance on the scores as fitted, so the first components weigh most;
  n_neighbors {umap_params['n_neighbors']}, min_dist {umap_params['min_dist']}, seed
  {umap_params['random_state']}). A scatter can carry at most three colours that stay
  distinguishable in every pair of marks, under colour-vision deficiency too, so each panel is
  drawn in three rows, each colouring healthy and two other groups over all remaining slides in
  grey; a group keeps its colour everywhere. Every panel shows all {n_slides} slides. The H1 cosine
  model's PC3 and PC4 carry a dagger ({DAGGER}): they move with the draw of the random features
  (section 3).</p>
  {''.join(slice_blocks)}

  <h2>6. Combined score spaces</h2>
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

  <h2>7. Comparison with the Betti-curve study</h2>
  <p>{comparison_text}</p>
  {comparison['table']}
  <p class="note">Ranges run over the ten slices (the first four rows) or the five combined
  spaces (the others).</p>
  <h3>How far the components agree</h3>
  <p class="note">The absolute correlation between the two studies' scores over the {n_slides}
  slides (signs are arbitrary in both): PC1 / PC2 for each model and slice, then for each combined
  space.</p>
  {comparison['slice_table']}
  {comparison['space_table']}

  <h2>Methods and reproduction</h2>
  <ul>
    <li>The groups (<code>results/pathology.csv</code>), the densities and the slice levels
    (<code>results/density.npz</code>) are the sibling study's, read from
    <code>../gtex-liver-pathology-slices</code>.</li>
    <li><code>run_pssk_slices.py slice K</code> computes the alpha persistent homology (max_dim 1)
    of each slide's kept cells with <code>tc.ph.compute_persistent_homology</code>, as the sibling
    study did, and stops unless every slide's numbers of H0 and H1 bars equal that study's; then the
    bandwidths, the random features ({n_features:,} per dimension, <code>random_state</code> 0) and
    the four kernel PCA models, by SVD of the centred feature matrix
    (<code>results/slice_K.npz</code>). An AnnData of all 126 million cells would not fit in
    memory, so it calls topocyte's PH directly.</li>
    <li><code>validate_kernel.py</code> compares the approximation with gudhi's exact kernel and
    with other draws of the features at slice 9 (<code>results/validation.json</code>;
    section 3).</li>
    <li><code>umap_scores.py</code> embeds each slice's and model's nine-component score matrix
    with umap-learn ({html.escape(json.dumps(umap_params))}; <code>results/umap.npz</code>).</li>
    <li><code>combined_spaces.py</code> weights the 40 score blocks by multiple factor analysis
    and fits the five combined spaces with their UMAPs (<code>results/combined.npz</code>).</li>
    <li><code>pc2_density_check.py</code> regresses the densest-decile share, and the other density
    summaries, out of each space's PC2 and compares cirrhosis with healthy by AUC
    (<code>results/pc2_check.npz</code>).</li>
    <li><code>verify_results.py</code> checks the saved results, among them slice 9 recomputed from
    the cohort and every kernel PCA recomputed through its Gram matrix; <code>build_report.py</code>
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
