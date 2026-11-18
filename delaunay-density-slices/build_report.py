#!/usr/bin/env python
"""Build the self-contained HTML report from the saved slice results.

Reads ``results/density.npz``, ``results/slice_0.npz`` ... ``slice_9.npz`` and
the per-slice scatter PNGs, and writes ``report.html`` with every figure
embedded as a base64 data URI.  No number in the prose is typed by hand: all of
them are formatted from those files.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
OUT = HERE / "report.html"

# Validated categorical pair (blue/orange, all-pairs CVD and normal-vision
# pass) and neutral ink for the control series.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CONTROL = "#52514e"
MUTED = "#898781"
RULE = "#e1e0d9"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"

DIMS = ("h0", "h1")
NORMS = ("none", "cosine")
MODELS = (
    ("h0_none", "H0, raw"),
    ("h1_none", "H1, raw"),
    ("h0_cosine", "H0, cosine"),
    ("h1_cosine", "H1, cosine"),
)
MODEL_COLOR = {"h0_none": BLUE, "h1_none": ORANGE, "h0_cosine": BLUE, "h1_cosine": ORANGE}
SEED0 = 0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def binomial_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1.0 - p) / n))


def png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def img_tag(data: bytes, alt: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def figure_density(density: dict[str, np.ndarray]) -> bytes:
    rho = np.sort(density["rho"])
    deciles = density["deciles"]
    n = rho.size
    xs = np.geomspace(max(float(rho[0]), 1e-3), float(rho[-1]), 800)
    survival = 1.0 - np.searchsorted(rho, xs, side="left") / n

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    for ax in (ax_full, ax_zoom):
        ax.plot(xs, survival, color=BLUE, lw=2)
        for d in deciles:
            ax.axvline(d, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=0)
        ax.scatter(
            deciles,
            1.0 - np.searchsorted(rho, deciles, side="left") / n,
            s=18, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=0.8,
            label="decile thresholds",
        )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.7)
        ax.set_xlabel("density γ (cells/mm², log scale)")
    ax_full.set_xlim(float(rho[0]) * 0.8, float(rho[-1]) * 1.25)
    ax_full.annotate(
        f"q0 = {float(deciles[0]):.1f}", xy=(float(deciles[0]), 1.0),
        xytext=(6, -10), textcoords="offset points", fontsize=7, color=MUTED,
    )
    ax_full.set_title("Full range", loc="left", color=INK, fontsize=9)
    ax_zoom.set_xlim(float(deciles[1]) * 0.8, float(deciles[-1]) * 1.12)
    ax_zoom.set_xticks(deciles[1:])
    ax_zoom.set_xticklabels([f"{d:.0f}" for d in deciles[1:]],
                            rotation=90, fontsize=7)
    ax_zoom.minorticks_off()
    ax_zoom.set_title("Detail at the decile thresholds", loc="left",
                      color=INK, fontsize=9)
    ax_full.set_ylabel("fraction of cells with γ ≥ x")
    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=1)
    fig.suptitle(f"Pooled 10-NN density of all {n:,} cells", fontsize=10,
                 color=INK, x=0.01, ha="left")
    return png_bytes(fig)


def figure_kept(density: dict[str, np.ndarray], slices: list[dict[str, np.ndarray]]) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    starts = np.arange(len(slices))
    for label, color in (("healthy", BLUE), ("cirrhosis", ORANGE)):
        series = np.array([s["kept_fraction"][s["group_label"] == label] for s in slices])
        for slide in range(series.shape[1]):
            ax.plot(starts, series[:, slide], color=color, lw=0.7, alpha=0.18, zorder=1)
        ax.plot(starts, np.median(series, axis=1), color=color, lw=2.4, zorder=3,
                label=f"{label} slides (median)")
    ax.set_xticks(starts)
    ax.set_xlabel("density slice k (keeps γ ≥ the pooled decile d_k)")
    ax.set_ylabel("fraction of the slide's cells kept")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.7)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Per-slide kept fraction for the ten thresholds", loc="left",
                 color=INK, fontsize=10)
    return png_bytes(fig)


def figure_summary(slices: list[dict[str, np.ndarray]]) -> bytes:
    starts = np.arange(len(slices))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True,
                             constrained_layout=True)
    for ax, normalization in zip(axes, NORMS):
        for name, dim in (("h0", 0), ("h1", 1)):
            key = f"{name}_{normalization}"
            color = MODEL_COLOR[key]
            means, sds, seed0 = [], [], []
            for s in slices:
                values = s[f"{key}_accuracy"]
                mean, sd = mean_sd(values)
                means.append(mean)
                sds.append(sd)
                seed0.append(float(values[SEED0]))
            means = np.array(means)
            sds = np.array(sds)
            ax.fill_between(starts, means - sds, means + sds, color=color, alpha=0.12,
                            linewidth=0)
            ax.plot(starts, means, color=color, lw=2, label=f"H{dim}")
            ax.scatter(starts, seed0, s=14, color=color, zorder=4,
                       edgecolor=SURFACE, linewidth=0.6)
            ax.axhline(means[0], color=color, lw=0.9, ls=":", alpha=0.8)
        control_mean = np.array([mean_sd(s["control_accuracy"])[0] for s in slices])
        control_sd = np.array([mean_sd(s["control_accuracy"])[1] for s in slices])
        ax.fill_between(starts, control_mean - control_sd, control_mean + control_sd,
                        color=CONTROL, alpha=0.10, linewidth=0)
        ax.plot(starts, control_mean, color=CONTROL, lw=2, ls="--",
                label="density-only control")
        ax.axhline(control_mean[0], color=CONTROL, lw=0.9, ls=":", alpha=0.8)
        ax.set_xticks(starts)
        ax.set_xlabel("density slice k")
        ax.set_title(
            "Raw Betti curves" if normalization == "none"
            else "Cosine-normalised Betti curves",
            loc="left", color=INK, fontsize=10,
        )
        ax.grid(axis="y", alpha=0.7)
    axes[0].set_ylabel("held-out accuracy (mean ± SD over 20 seeds)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Held-out accuracy per slice; dotted lines mark slice 0", fontsize=10,
                 color=INK, x=0.01, ha="left")
    return png_bytes(fig)


def figure_combined(
    slices: list[dict[str, np.ndarray]], combined: dict[str, np.ndarray]
) -> bytes:
    fig, ax = plt.subplots(figsize=(9.2, 3.9), constrained_layout=True)
    positions = np.arange(len(MODELS))[::-1]
    for (key, label), ypos in zip(MODELS, positions):
        base_mean, _ = mean_sd(slices[0][f"{key}_accuracy"])
        mean, sd = mean_sd(combined[f"{key}_accuracy"])
        color = MODEL_COLOR[key]
        ax.plot([base_mean, mean], [ypos, ypos], color=RULE, lw=3, zorder=1)
        ax.scatter([base_mean], [ypos], s=42, color=MUTED, zorder=3,
                   label="slice 0 (same model)" if ypos == positions[0] else None)
        ax.errorbar(mean, ypos, xerr=sd, color=color, capsize=3, lw=1, zorder=2)
        ax.scatter([mean], [ypos], s=52, color=color, edgecolor=SURFACE,
                   linewidth=0.8, zorder=4,
                   label="all ten slices" if ypos == positions[0] else None)
    control_mean, control_sd = mean_sd(combined["control_accuracy"])
    ax.axvspan(control_mean - control_sd, control_mean + control_sd,
               color=CONTROL, alpha=0.08, linewidth=0, zorder=0)
    ax.axvline(control_mean, color=CONTROL, lw=1.4, ls="--", zorder=1,
               label="density-only control (ten kept fractions)")
    ax.set_yticks(positions, [label for _, label in MODELS])
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.grid(axis="x", alpha=0.7)
    ax.set_xlabel("held-out accuracy (mean over 20 seeds; error bar ± 1 SD)")
    ax.set_title("First four scores from every slice, 40 features per model",
                 loc="left", color=INK, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=3)
    return png_bytes(fig)


# Section 7 figures: colour is the homology dimension, as everywhere in this
# report; a filled marker is a raw model, an open one a cosine model.
FAMILY_FILLED = {"h0_none": True, "h1_none": True, "h0_cosine": False, "h1_cosine": False}


def _family_errorbar(ax, x, means, lo, hi, key, *, horizontal=False, marker="o",
                     label=None) -> None:
    color = MODEL_COLOR[key]
    lower = np.asarray(means) - np.asarray(lo)
    upper = np.asarray(hi) - np.asarray(means)
    kwargs = dict(
        fmt=marker, ms=5.5, color=color, mec=color, mew=1.2,
        mfc=color if FAMILY_FILLED[key] else SURFACE,
        ecolor=color, elinewidth=1.0, capsize=2.5, label=label, zorder=3,
    )
    if horizontal:
        ax.errorbar(means, x, xerr=[lower, upper], **kwargs)
    else:
        ax.errorbar(x, means, yerr=[lower, upper], **kwargs)


def figure_adds_combined(adds: dict[str, np.ndarray]) -> bytes:
    """Combined ten-slice models against the density arm: accuracy and log-loss."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True,
                             constrained_layout=True)
    positions = np.arange(len(MODELS))[::-1].astype(float)
    arms = (("nested", "o", 0.15), ("pc24", "^", -0.15))
    panels = (
        (axes[0], "", "Accuracy",
         "accuracy difference against the density arm"),
        (axes[1], "_ll", "Log-loss",
         "log-loss reduction against the density arm"),
    )
    for ax, suffix, title, xlabel in panels:
        ax.axvline(0.0, color=MUTED, ls="--", lw=1.0, zorder=0)
        for (key, _), ypos in zip(MODELS, positions):
            for kind, marker, offset in arms:
                mean, lo, hi = (float(v) for v in adds[f"ci_{kind}{suffix}_{key}_all"])
                _family_errorbar(ax, [ypos + offset], [mean], [lo], [hi], key,
                                 horizontal=True, marker=marker)
        ax.grid(axis="x", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
    axes[0].set_yticks(positions, [label for _, label in MODELS])
    axes[0].set_ylim(-0.6, len(MODELS) - 0.4)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=5.5, color=INK2,
               label="density + the four scores of every slice"),
        Line2D([], [], ls="none", marker="^", ms=5.5, color=INK2,
               label="scores 2–4 of every slice, no density"),
        Line2D([], [], color=MUTED, ls="--", lw=1.0, label="no difference"),
    ]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=3)
    fig.suptitle(
        "Combined ten-slice models against the density arm "
        "(mean over 20 seeds, 95% paired bootstrap over slides)",
        fontsize=10, color=INK, x=0.01, ha="left",
    )
    return png_bytes(fig)


def figure_adds_per_slice(adds: dict[str, np.ndarray]) -> bytes:
    """Per-slice nested and residual arms, each in accuracy and log-loss."""
    panels = (
        ("nested", "Density + scores against density: accuracy",
         "accuracy difference"),
        ("nested_ll", "Density + scores against density: log-loss",
         "log-loss reduction"),
        ("res_chance", "Residualised scores against chance: accuracy",
         "accuracy above chance"),
        ("res_ll", "Residualised scores against chance: log-loss",
         "log-loss reduction"),
    )
    n_slices = 10
    starts = np.arange(n_slices)
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True,
                             constrained_layout=True)
    for ax, (kind, title, ylabel) in zip(axes.ravel(), panels):
        ax.axhline(0.0, color=MUTED, ls="--", lw=1.0, zorder=0)
        for (key, label), offset in zip(MODELS, offsets):
            ci = np.array([adds[f"ci_{kind}_{key}_{k}"] for k in starts])
            _family_errorbar(ax, starts + offset, ci[:, 0], ci[:, 1], ci[:, 2], key,
                             label=label if ax is axes[0, 0] else None)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.7)
        ax.set_xticks(starts)
    for ax in axes[1]:
        ax.set_xlabel("density slice k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=4)
    fig.suptitle(
        "Per slice: mean over 20 seeds and 95% paired bootstrap interval over slides",
        fontsize=10, color=INK, x=0.01, ha="left",
    )
    return png_bytes(fig)


def tally_cis(adds: dict[str, np.ndarray], kind: str) -> dict[str, object]:
    """Count per-slice intervals wholly above or below zero; find the largest."""
    above: list[tuple[int, str]] = []
    below: list[tuple[int, str]] = []
    positive = np.zeros(10, dtype=int)
    best: dict[str, object] | None = None
    for key, label in MODELS:
        for k in range(10):
            mean, lo, hi = (float(v) for v in adds[f"ci_{kind}_{key}_{k}"])
            if lo > 0:
                above.append((k, label))
            if hi < 0:
                below.append((k, label))
            if mean > 0:
                positive[k] += 1
            if best is None or mean > best["delta"]:
                best = {"delta": mean, "lo": lo, "hi": hi, "label": label, "slice": k}
    return {"above": above, "below": below, "positive": positive, "best": best}


def html_table(header: list[str], rows: list[tuple[str, ...]], cls: str = "") -> str:
    """A table; a row whose first cell is non-empty opens a group."""
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join(
        ("<tr class='group'>" if row[0] else "<tr>")
        + "".join(f"<td>{cell}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    return (
        f"<div class='table-wrap'><table class='{cls}'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def fmt_ci(values: np.ndarray) -> str:
    mean, lo, hi = (float(v) for v in values)
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def list_by_slice(pairs: list[tuple[int, str]]) -> str:
    by_slice: dict[int, list[str]] = {}
    for k, label in pairs:
        by_slice.setdefault(k, []).append(label.replace(", ", " "))
    return "; ".join(
        f"slice {k}: {', '.join(labels)}" for k, labels in sorted(by_slice.items())
    )


def section7(adds: dict[str, np.ndarray], combined: dict[str, np.ndarray]) -> dict[str, str]:
    """Tables and prose for section 7, corrected 2026-11-17.

    Every number is read from ``density_adds.npz``.  Each sentence that states a
    pattern rather than a number is guarded by an assertion, so new results that
    break the pattern stop the build instead of contradicting the text.
    """
    n_slices = int(adds["n_slices"])
    n_components = 4
    per_slice = int(adds["rich_scores_k0"].shape[1]) + 1
    n_all = int(adds["rich_scores_all"].shape[1]) + 1
    short = {key: label.replace(", ", " ") for key, label in MODELS}
    y = adds["y"]
    cells = np.rint(10.0 ** adds["log_cells"])
    chance_acc = float(adds["chance_correct"].mean())
    base_ll = float(adds["base_logloss"].mean())

    def metrics(field: str, suffix: str) -> dict[str, tuple[float, float]]:
        return {
            name: mean_sd(adds[f"{field}_{name}_{suffix}"])
            for name in ("acc", "logloss", "auc")
        }

    def pm(pair: tuple[float, float]) -> str:
        return f"{pair[0]:.3f} ± {pair[1]:.3f}"

    def ci(name: str) -> tuple[float, float, float]:
        mean, lo, hi = (float(v) for v in adds[name])
        return mean, lo, hi

    density = metrics("A", "all")
    size = metrics("S", "all")

    # Table 7a: combined models, absolute.
    rows = [
        ("no scores", "chance (training-fold majority; prevalence for log-loss)", "0",
         f"{chance_acc:.3f}", f"{base_ll:.3f}", "—"),
        ("", "slide size alone (log₁₀ cells)", "1",
         pm(size["acc"]), pm(size["logloss"]), pm(size["auc"])),
        ("", "density arm", str(n_all),
         pm(density["acc"]), pm(density["logloss"]), pm(density["auc"])),
    ]
    pc1_acc = {}
    for key, label in MODELS:
        n_scores = int(combined[f"{key}_scores"].shape[1])
        n_pc1 = n_scores // n_components
        nested = metrics(f"B_{key}", "all")
        pc1 = metrics(f"P1_{key}", "all")
        pc24 = metrics(f"P24_{key}", "all")
        pc1_acc[key] = pc1["acc"]
        rows += [
            (label, "scores alone (section 6)", str(n_scores),
             pm(mean_sd(combined[f"{key}_accuracy"])), "—", "—"),
            ("", "density + scores", str(n_all + n_scores),
             pm(nested["acc"]), pm(nested["logloss"]), pm(nested["auc"])),
            ("", "PC1 only", str(n_pc1),
             pm(pc1["acc"]), pm(pc1["logloss"]), pm(pc1["auc"])),
            ("", "PCs 2–4 only", str(n_scores - n_pc1),
             pm(pc24["acc"]), pm(pc24["logloss"]), pm(pc24["auc"])),
        ]
    table_combined = html_table(
        ["model", "arm", "features", "accuracy", "log-loss", "AUC"], rows, cls="left2 grouped"
    )

    # Table 7b: combined models against the density arm.
    table_deltas = html_table(
        [
            "model",
            "density + scores<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "density + scores<br><span class='unit'>log-loss reduction [95% CI]</span>",
            "PCs 2–4 only<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "PCs 2–4 only<br><span class='unit'>log-loss reduction [95% CI]</span>",
        ],
        [
            (
                label,
                fmt_ci(adds[f"ci_nested_{key}_all"]),
                fmt_ci(adds[f"ci_nested_ll_{key}_all"]),
                fmt_ci(adds[f"ci_pc24_{key}_all"]),
                fmt_ci(adds[f"ci_pc24_ll_{key}_all"]),
            )
            for key, label in MODELS
        ],
    )

    # Table 7c: per slice, folded away.
    slice_rows = []
    for k in range(n_slices):
        density_k = mean_sd(adds[f"A_acc_{k}"])
        for index, (key, label) in enumerate(MODELS):
            slice_rows.append(
                (
                    str(k) if index == 0 else "",
                    label,
                    pm(density_k) if index == 0 else "",
                    pm(mean_sd(adds[f"B_{key}_acc_{k}"])),
                    fmt_ci(adds[f"ci_nested_{key}_{k}"]),
                    fmt_ci(adds[f"ci_nested_ll_{key}_{k}"]),
                    pm(mean_sd(adds[f"R_{key}_acc_{k}"])),
                    fmt_ci(adds[f"ci_res_chance_{key}_{k}"]),
                    fmt_ci(adds[f"ci_res_ll_{key}_{k}"]),
                    pm(mean_sd(adds[f"R_{key}_auc_{k}"])),
                )
            )
    table_slices = html_table(
        [
            "slice", "model",
            "density arm<br><span class='unit'>accuracy</span>",
            "density + scores<br><span class='unit'>accuracy</span>",
            "Δ accuracy<br><span class='unit'>[95% CI]</span>",
            "log-loss reduction<br><span class='unit'>[95% CI]</span>",
            "residual<br><span class='unit'>accuracy</span>",
            "above chance<br><span class='unit'>[95% CI]</span>",
            "log-loss reduction<br><span class='unit'>vs prevalence [95% CI]</span>",
            "residual<br><span class='unit'>AUC</span>",
        ],
        slice_rows,
        cls="left2 grouped",
    )

    # Combined prose.
    nested = {key: ci(f"ci_nested_{key}_all") for key, _ in MODELS}
    nested_ll = {key: ci(f"ci_nested_ll_{key}_all") for key, _ in MODELS}
    pc24 = {key: ci(f"ci_pc24_{key}_all") for key, _ in MODELS}
    nested_means = [value[0] for value in nested.values()]
    best_nested = max(nested, key=lambda key: nested[key][0])
    nested_excluding = sum(lo > 0 or hi < 0 for _, lo, hi in nested.values())
    # Claim: the scores add nothing measurable to the combined density arm.
    assert all(lo <= 0.0 for _, lo, _ in nested.values()), "a combined nested CI is above 0"
    ll_up = sum(mean < 0 for mean, _, _ in nested_ll.values())
    ll_up_beyond = sum(hi < 0 for _, _, hi in nested_ll.values())
    auc_change = [
        float(np.mean(adds[f"B_{key}_auc_all"] - adds["A_auc_all"])) for key, _ in MODELS
    ]
    pc24_means = [value[0] for value in pc24.values()]
    pc24_below = sum(hi < 0 for _, _, hi in pc24.values())
    pc24_above = sum(lo > 0 for _, lo, _ in pc24.values())
    r2 = adds["pc_r2"]
    best_pc1 = max(pc1_acc, key=lambda key: pc1_acc[key][0])
    if nested_excluding == 0:
        nested_excl_text = "none of the four intervals excludes zero"
    else:
        nested_excl_text = f"{nested_excluding} of the four intervals exclude zero"
    ll_up_text = "all four" if ll_up == len(MODELS) else f"{ll_up} of the four"
    pc24_above_text = "none" if pc24_above == 0 else str(pc24_above)
    # Claim: the best single-family PC1 is level with the density arm.
    pc1_gap = pc1_acc[best_pc1][0] - density["acc"][0]
    assert abs(pc1_gap) < density["acc"][1], f"PC1 vs density arm {pc1_gap:+.3f}"
    text_combined = (
        "Slide size alone, log₁₀ of the slide's cell count, classifies at "
        f"{pm(size['acc'])} (AUC {size['auc'][0]:.3f}): cirrhosis slides carry more "
        f"cells (median {np.median(cells[y == 1]):,.0f} against "
        f"{np.median(cells[y == 0]):,.0f}), and raw Betti curves grow with the "
        "number of cells. The density arm now includes slide size and reaches "
        f"{pm(density['acc'])} over the ten slices (log-loss "
        f"{density['logloss'][0]:.3f}, AUC {density['auc'][0]:.3f}). Adding every "
        f"slice's four scores changes accuracy by {min(nested_means):+.3f} to "
        f"{max(nested_means):+.3f} across the four families, and {nested_excl_text}; "
        f"the best is {short[best_nested]} at "
        f"{fmt_ci(adds[f'ci_nested_{best_nested}_all'])}. Log-loss rises in "
        f"{ll_up_text} families, and in {ll_up_beyond} the whole interval of the "
        f"reduction lies below zero; AUC changes by {min(auc_change):+.3f} to "
        f"{max(auc_change):+.3f}. Scores 2–4 alone, without any density feature, "
        f"differ from the density arm by {min(pc24_means):+.3f} to "
        f"{max(pc24_means):+.3f} in accuracy; {pc24_below} of the four intervals lie "
        f"wholly below zero and {pc24_above_text} above. The first score is "
        "density-shaped: the slice's density arm explains a median "
        f"{float(np.median(r2[:, :, 0])):.2f} of its variance across slices and "
        f"families (scores 2–4: {float(np.median(r2[:, :, 1:])):.2f}), and "
        f"{short[best_pc1]} PC1 alone, {n_slices} features, reaches "
        f"{pm(pc1_acc[best_pc1])}, level with the {n_all}-feature density arm."
    )

    # Per-slice nested prose.
    tally_nested = tally_cis(adds, "nested")
    tally_nested_ll = tally_cis(adds, "nested_ll")
    density_k = np.array([float(adds[f"A_acc_{k}"].mean()) for k in range(n_slices)])
    mean_gain = np.array(
        [np.mean([adds[f"ci_nested_{key}_{k}"][0] for key, _ in MODELS])
         for k in range(n_slices)]
    )
    gain_r = float(np.corrcoef(density_k, mean_gain)[0, 1])
    middle = list(range(4, 9))
    nested_k = {
        (k, key): float(adds[f"B_{key}_acc_{k}"].mean())
        for k in range(n_slices) for key, _ in MODELS
    }
    best_middle = max((k_key for k_key in nested_k if k_key[0] in middle),
                      key=nested_k.get)
    best_density_slice = int(np.argmax(density_k))
    # Claims: the gain falls as the slice's density arm gets stronger; every
    # point estimate at slices 4-8 is positive; the intervals above zero all sit
    # there, those below zero at slices 0-1; and no middle-slice nested model
    # reaches the best single-slice density arm.
    assert gain_r < -0.5, f"gain/density correlation {gain_r:.2f}"
    assert all(tally_nested["positive"][k] == len(MODELS) for k in middle)
    assert tally_nested["above"] and all(k in middle for k, _ in tally_nested["above"])
    assert all(k in (0, 1) for k, _ in tally_nested["below"])
    assert nested_k[best_middle] < density_k[best_density_slice]
    if tally_nested_ll["above"]:
        ll_slice_text = (
            f"{len(tally_nested_ll['above'])} intervals lie above zero "
            f"and {len(tally_nested_ll['below'])} below"
        )
    else:
        ll_slice_text = (
            "no interval lies above zero and "
            f"{len(tally_nested_ll['below'])} lie below"
        )
    text_slices = (
        "Per slice the picture differs. A single slice's density arm is strongest at "
        f"slice {best_density_slice} ({density_k[best_density_slice]:.3f}) and weakest "
        f"at slice {int(np.argmin(density_k))} ({density_k.min():.3f}), and the "
        "scores help most where it is weakest (correlation of a slice's mean Δ with "
        f"its density-arm accuracy r = {gain_r:.2f}). At slices "
        f"{middle[0]}–{middle[-1]} all {len(middle) * len(MODELS)} point estimates are "
        f"positive and {len(tally_nested['above'])} intervals lie wholly above zero "
        f"({list_by_slice(tally_nested['above'])}); "
        f"{len(tally_nested['below'])} lie wholly below, all at slices 0–1 "
        f"({list_by_slice(tally_nested['below'])}). Yet the middle slices' nested "
        f"models reach at most {nested_k[best_middle]:.3f} "
        f"({short[best_middle[1]]}, slice {best_middle[0]}), short of the "
        f"{density_k[best_density_slice]:.3f} that the density arm of slice "
        f"{best_density_slice} reaches alone, and the combined model, which holds "
        "every slice's density summary, gains nothing from the scores. What the "
        "middle slices' scores add is information that density at other thresholds "
        f"already carries. In log-loss, {ll_slice_text}."
    )

    # Per-slice residual prose.
    tally_res = tally_cis(adds, "res_chance")
    tally_res_ll = tally_cis(adds, "res_ll")
    residual = {
        (k, key): adds[f"R_{key}_acc_{k}"].mean()
        for k in range(n_slices) for key, _ in MODELS
    }
    residual_auc = {
        (k, key): adds[f"R_{key}_auc_{k}"].mean()
        for k in range(n_slices) for key, _ in MODELS
    }
    top_auc = max(residual_auc, key=residual_auc.get)
    dense = (8, 9)
    low = [value for (k, _), value in residual.items() if k not in dense]
    high = [value for (k, _), value in residual.items() if k in dense]
    above_dense = sum(k in dense for k, _ in tally_res["above"])
    # Claims: residuals beat chance somewhere and are never below it; slices 8-9
    # hold most of the intervals above chance and the higher residual
    # accuracies; log-loss never clears the prevalence.
    assert tally_res["above"] and not tally_res["below"]
    assert above_dense > len(tally_res["above"]) / 2
    assert np.mean(high) > np.mean(low) + 0.05
    assert not tally_res_ll["above"]
    # Claim: the residual signal is strongest at the top slice, where every
    # nested point estimate is negative.
    top = n_slices - 1
    residual_slice = [
        np.mean([residual[(k, key)] for key, _ in MODELS]) for k in range(n_slices)
    ]
    assert int(np.argmax(residual_slice)) == top
    assert tally_nested["positive"][top] == 0
    text_residual = (
        "The residualised scores are the direct test of the proxy claim: if the "
        "scores were a linear function of the density features, the residuals "
        "would sit at chance. In accuracy, "
        f"{len(tally_res['above'])} of the 40 intervals lie wholly above chance "
        f"({list_by_slice(tally_res['above'])}) and none below. Residual accuracy is "
        f"{min(low):.3f}–{max(low):.3f} at slices 0–7 and "
        f"{min(high):.3f}–{max(high):.3f} at slices 8–9, where {above_dense} of the 8 "
        f"intervals clear chance; AUC reaches {residual_auc[top_auc]:.3f} "
        f"({short[top_auc[1]]}, slice {top_auc[0]}). In log-loss none of the 40 "
        "intervals clears the prevalence baseline: the protocol picks C for accuracy, "
        "and its residual probabilities are no sharper than the class prevalence. "
        f"So the scores carry signal that a slice's {per_slice} density features do "
        "not explain linearly, strongest at the two densest thresholds. Yet at "
        f"slice {top}, where it is strongest, all four nested point estimates are "
        "negative, and the combined model gains nothing. Each arm makes 40 "
        "correlated comparisons on the same slides; the patterns across adjacent "
        "slices carry more weight than any single interval."
    )

    headline = (
        "Section 7, corrected on 2026-11-17, separates two claims that its first "
        "version merged. Persistent homology is not a linear proxy for the density "
        "summary: "
        f"residualised on {per_slice} density features per slice, slide size "
        "included, the scores still classify above chance in "
        f"{len(tally_res['above'])} of 40 slice/family comparisons by accuracy "
        f"({above_dense} of the 8 at slices 8–9), though in none by log-loss. But "
        "what they carry beyond density does not improve on density: with every "
        f"slice's density summary in the model ({pm(density['acc'])}), adding the "
        f"scores changes accuracy by at most {max(nested_means):+.3f}, with no "
        "interval above zero, and per slice they lift the weaker density arms of "
        f"slices {middle[0]}–{middle[-1]} only to {nested_k[best_middle]:.3f}, below "
        f"the {density_k[best_density_slice]:.3f} that the density arm of slice "
        f"{best_density_slice} reaches alone."
    )
    return {
        "per_slice": str(per_slice),
        "n_all": str(n_all),
        "table_combined": table_combined,
        "table_deltas": table_deltas,
        "table_slices": table_slices,
        "text_combined": text_combined,
        "text_slices": text_slices,
        "text_residual": text_residual,
        "headline": headline,
        "size_acc": pm(size["acc"]),
    }


# Section 8: the H1 death/birth ratio.  Colour stays the homology dimension, so
# the ratio, an H1 quantity, is orange; section 7's H1 Betti values are grey
# references.  Filled raw, open cosine, as in section 7.
RATIO_FAMILIES = (("none", "raw"), ("cosine", "cosine"))


def _errorbar(ax, x, means, lo, hi, *, color, filled, marker="o", label=None) -> None:
    means = np.asarray(means)
    ax.errorbar(
        x, means, yerr=[means - np.asarray(lo), np.asarray(hi) - means],
        fmt=marker, ms=5.5, color=color, mec=color, mew=1.2,
        mfc=color if filled else SURFACE, ecolor=color, elinewidth=1.0,
        capsize=2.5, label=label, zorder=3,
    )


def figure_ratio(ratio: dict[str, np.ndarray], adds: dict[str, np.ndarray]) -> bytes:
    """Nested and ratio-alone intervals per slice, with section 7's H1 Betti beside."""
    from matplotlib.lines import Line2D

    slices = [int(k) for k in ratio["slices"]]
    positions = np.arange(len(slices), dtype=float)
    panels = (
        ("nested", "Density + scores against density: accuracy", "accuracy difference", True),
        ("nested_ll", "Density + scores against density: log-loss", "log-loss reduction", True),
        ("alone_chance", "Ratio scores alone against chance: accuracy",
         "accuracy above chance", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), constrained_layout=True)
    for ax, (kind, title, ylabel, with_betti) in zip(axes, panels):
        ax.axhline(0.0, color=MUTED, ls="--", lw=1.0, zorder=0)
        for index, (m, _) in enumerate(RATIO_FAMILIES):
            filled = m == "none"
            ci = np.array([ratio[f"ci_{kind}_{m}_{k}"] for k in slices])
            offset = (-0.21 + 0.14 * index) if with_betti else (-0.09 + 0.18 * index)
            _errorbar(ax, positions + offset, ci[:, 0], ci[:, 1], ci[:, 2],
                      color=ORANGE, filled=filled)
            if with_betti:
                betti = np.array([adds[f"ci_{kind}_h1_{m}_{k}"] for k in slices])
                _errorbar(ax, positions + 0.07 + 0.14 * index, betti[:, 0], betti[:, 1],
                          betti[:, 2], color=MUTED, filled=filled, marker="^")
        ax.set_xticks(positions, [str(k) for k in slices])
        ax.set_xlabel("density slice k")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.grid(axis="y", alpha=0.7)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=5.5, color=ORANGE, label="H1 ratio, raw"),
        Line2D([], [], ls="none", marker="o", ms=5.5, color=ORANGE, mfc=SURFACE,
               mew=1.2, label="H1 ratio, cosine"),
        Line2D([], [], ls="none", marker="^", ms=5.5, color=MUTED,
               label="H1 Betti (section 7), raw"),
        Line2D([], [], ls="none", marker="^", ms=5.5, color=MUTED, mfc=SURFACE,
               mew=1.2, label="H1 Betti (section 7), cosine"),
    ]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4)
    fig.suptitle(
        "Per slice: mean over 20 seeds and 95% paired bootstrap interval over slides",
        fontsize=10, color=INK, x=0.01, ha="left",
    )
    return png_bytes(fig)


def figure_ratio_quantiles(ratio: dict[str, np.ndarray]) -> bytes:
    """Per-slide quantiles of log pi by class, one panel per quantile level."""
    slices = [int(k) for k in ratio["slices"]]
    levels = [float(v) for v in ratio["levels"]]
    y = ratio["y"]
    positions = np.arange(len(slices), dtype=float)
    rng = np.random.default_rng(0)  # display jitter only
    fig, axes = plt.subplots(1, len(levels), figsize=(11, 3.9), constrained_layout=True)
    for column, (ax, level) in enumerate(zip(axes, levels)):
        for label, name, color, shift in ((0, "healthy", BLUE, -0.17),
                                          (1, "cirrhosis", ORANGE, 0.17)):
            for x, k in zip(positions, slices):
                values = ratio[f"log_pi_quantiles_{k}"][y == label, column]
                jitter = rng.uniform(-0.09, 0.09, values.size)
                ax.scatter(x + shift + jitter, values, s=7, color=color, alpha=0.45,
                           linewidth=0, zorder=2,
                           label=f"{name} slides" if x == 0 and column == 0 else None)
                ax.plot([x + shift - 0.13, x + shift + 0.13], [np.median(values)] * 2,
                        color=INK, lw=1.6, zorder=3,
                        label="class median" if x == 0 and column == 0 and label == 1
                        else None)
        ax.set_xticks(positions, [str(k) for k in slices])
        ax.set_xlabel("density slice k")
        ax.set_title(f"q = {level:g}", loc="left", color=INK, fontsize=9.5)
        ax.grid(axis="y", alpha=0.7)
    axes[0].set_ylabel("log π quantile of the slide's H1 bars")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=3,
               markerscale=2.2)
    fig.suptitle("Quantiles of log π = ½ log(death / birth) per slide, by class",
                 fontsize=10, color=INK, x=0.01, ha="left")
    return png_bytes(fig)


def ratio_tables(ratio: dict[str, np.ndarray], adds: dict[str, np.ndarray]) -> dict[str, str]:
    """Section 8's tables, every number read from ``ratio_slices.npz``."""
    slices = [int(k) for k in ratio["slices"]]
    y = ratio["y"]

    def pm(values: np.ndarray) -> str:
        mean, sd = mean_sd(values)
        return f"{mean:.3f} ± {sd:.3f}"

    rows = []
    for k in slices:
        for index, (m, label) in enumerate(RATIO_FAMILIES):
            rows.append(
                (
                    ("0 (primary)" if k == 0 else str(k)) if index == 0 else "",
                    label,
                    pm(ratio[f"A_acc_{k}"]) if index == 0 else "",
                    pm(ratio[f"Q_{m}_acc_{k}"]),
                    pm(ratio[f"Q_{m}_auc_{k}"]),
                    fmt_ci(ratio[f"ci_alone_chance_{m}_{k}"]),
                    pm(ratio[f"N_{m}_acc_{k}"]),
                    fmt_ci(ratio[f"ci_nested_{m}_{k}"]),
                    fmt_ci(ratio[f"ci_nested_ll_{m}_{k}"]),
                )
            )
    table_arms = html_table(
        [
            "slice", "ratio",
            "density arm<br><span class='unit'>accuracy</span>",
            "ratio alone<br><span class='unit'>accuracy</span>",
            "ratio alone<br><span class='unit'>AUC</span>",
            "ratio alone<br><span class='unit'>above chance [95% CI]</span>",
            "density + ratio<br><span class='unit'>accuracy</span>",
            "density + ratio<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "density + ratio<br><span class='unit'>log-loss reduction [95% CI]</span>",
        ],
        rows,
        cls="left2 grouped",
    )

    def r2_cell(values: np.ndarray) -> str:
        return " · ".join(f"{float(v):.2f}" for v in values)

    betti_index = {"none": 1, "cosine": 3}  # h1_none, h1_cosine in section 7's pc_r2
    r2_rows = []
    for i, k in enumerate(slices):
        r2_rows.append(
            (str(k),)
            + tuple(r2_cell(ratio["ratio_r2"][i, f]) for f in range(len(RATIO_FAMILIES)))
            + tuple(r2_cell(adds["pc_r2"][k, betti_index[m]]) for m, _ in RATIO_FAMILIES)
        )
    table_r2 = html_table(
        [
            "slice",
            "H1 ratio, raw<br><span class='unit'>PC1 · PC2 · PC3 · PC4</span>",
            "H1 ratio, cosine<br><span class='unit'>PC1 · PC2 · PC3 · PC4</span>",
            "H1 Betti, raw<br><span class='unit'>PC1 · PC2 · PC3 · PC4</span>",
            "H1 Betti, cosine<br><span class='unit'>PC1 · PC2 · PC3 · PC4</span>",
        ],
        r2_rows,
    )

    bar_rows = []
    for k in slices:
        n_bars = ratio[f"n_bars_{k}"]
        per_cell = n_bars / ratio[f"kept_{k}"]
        quantiles = ratio[f"log_pi_quantiles_{k}"]
        cells = [str(k)]
        for label in (0, 1):
            cells.append(f"{np.median(n_bars[y == label]):,.0f}")
        for label in (0, 1):
            cells.append(f"{np.median(per_cell[y == label]):.3f}")
        for level in range(quantiles.shape[1]):
            cells.append(
                f"{np.median(quantiles[y == 0, level]):.3f} / "
                f"{np.median(quantiles[y == 1, level]):.3f}"
            )
        bar_rows.append(tuple(cells))
    levels = [float(v) for v in ratio["levels"]]
    table_bars = html_table(
        ["slice",
         "H1 bars<br><span class='unit'>healthy</span>",
         "H1 bars<br><span class='unit'>cirrhosis</span>",
         "bars per cell<br><span class='unit'>healthy</span>",
         "bars per cell<br><span class='unit'>cirrhosis</span>"]
        + [f"log π, q{level:g}<br><span class='unit'>healthy / cirrhosis</span>"
           for level in levels],
        bar_rows,
    )
    return {"table_arms": table_arms, "table_r2": table_r2, "table_bars": table_bars}


def section8(
    ratio: dict[str, np.ndarray],
    adds: dict[str, np.ndarray],
    slices: list[dict[str, np.ndarray]],
) -> dict[str, str]:
    """Tables and prose for section 8, added 2026-11-17.

    Every number is read from ``ratio_slices.npz`` (and, for the Betti
    references, ``density_adds.npz``).  The primary sentence has one branch per
    outcome of the pre-registered rule, written before the results were seen;
    every other sentence that states a pattern is guarded by an assertion.
    """
    ks = [int(k) for k in ratio["slices"]]
    secondary = [k for k in ks if k != 0]
    short = dict(RATIO_FAMILIES)

    # Checks the design list states.
    gram_rel = max(float(ratio[f"gram_check_rel_{k}"]) for k in ks)
    deviation = 0.0
    for k in ks:
        assert np.array_equal(ratio[f"A_correct_{k}"], adds[f"A_correct_{k}"])
        for m, _ in RATIO_FAMILIES:
            got = ratio[f"betti_{m}_scores_{k}"]
            want = slices[k][f"h1_{m}_scores"][:, :4]
            signs = np.sign(np.sum(got * want, axis=0))
            scale = np.max(np.abs(want), axis=0)
            deviation = max(deviation, float(np.max(np.abs(got * signs - want) / scale)))
    gate = "bit for bit" if deviation == 0.0 else f"to {deviation:.0e} (relative)"

    def pm(values: np.ndarray) -> str:
        mean, sd = mean_sd(values)
        return f"{mean:.3f} ± {sd:.3f}"

    def ci(name: str) -> tuple[float, float, float]:
        mean, lo, hi = (float(v) for v in ratio[name])
        return mean, lo, hi

    # Primary test, pre-registered: slice 0, both normalizations.
    nested0 = {m: ci(f"ci_nested_{m}_0") for m, _ in RATIO_FAMILIES}
    adds_primary = [m for m, (_, lo, _) in nested0.items() if lo > 0]
    primary_numbers = (
        f"At slice 0, the pre-registered test, the density arm reaches "
        f"{pm(ratio['A_acc_0'])}. With the four ratio scores added it reaches "
        f"{pm(ratio['N_none_acc_0'])} raw and {pm(ratio['N_cosine_acc_0'])} cosine, "
        f"a change in accuracy of {fmt_ci(ratio['ci_nested_none_0'])} and "
        f"{fmt_ci(ratio['ci_nested_cosine_0'])}"
    )
    if adds_primary:
        which = " and ".join(short[m] for m in adds_primary)
        text_primary = (
            f"{primary_numbers}. The {which} interval lies wholly above zero, so by "
            "the pre-registered rule the ratio adds beyond density."
        )
        headline = (
            "Section 8 tests one topological quantity that does not change with the "
            "scale of the cells, the death/birth ratio of the H1 bars. At slice 0, "
            "its pre-registered test, adding four ratio scores to the density arm "
            f"changes accuracy by {fmt_ci(ratio['ci_nested_none_0'])} (raw) and "
            f"{fmt_ci(ratio['ci_nested_cosine_0'])} (cosine): the ratio adds beyond "
            "density."
        )
    else:
        text_primary = (
            f"{primary_numbers}. Neither interval lies wholly above zero, so by the "
            "pre-registered rule the ratio does not add beyond density."
        )
        headline = (
            "Section 8 tests one topological quantity that does not change with the "
            "scale of the cells, the death/birth ratio of the H1 bars. At slice 0, "
            "its pre-registered test, adding four ratio scores to the density arm "
            f"changes accuracy by {fmt_ci(ratio['ci_nested_none_0'])} (raw) and "
            f"{fmt_ci(ratio['ci_nested_cosine_0'])} (cosine), and neither interval "
            "lies above zero: by the pre-registered rule, the ratio does not add beyond "
            "density either."
        )
    # Claim: of the two log-loss intervals, only the cosine one lies above zero.
    ll_above = [m for m, _ in RATIO_FAMILIES if ci(f"ci_nested_ll_{m}_0")[1] > 0]
    assert ll_above == ["cosine"], ll_above
    text_primary += (
        f" The log-loss reductions are {fmt_ci(ratio['ci_nested_ll_none_0'])} and "
        f"{fmt_ci(ratio['ci_nested_ll_cosine_0'])}. The cosine interval lies above "
        "zero, but the rule counts accuracy, and the post-hoc check below finds that "
        "this reduction does not hold with standardized features. Alone, the four ratio scores "
        f"classify at {pm(ratio['Q_none_acc_0'])} raw and "
        f"{pm(ratio['Q_cosine_acc_0'])} cosine (AUC "
        f"{mean_sd(ratio['Q_none_auc_0'])[0]:.3f} and "
        f"{mean_sd(ratio['Q_cosine_auc_0'])[0]:.3f}), "
        f"{fmt_ci(ratio['ci_alone_chance_none_0'])} and "
        f"{fmt_ci(ratio['ci_alone_chance_cosine_0'])} above chance. The post-hoc "
        "checks below refit these arms too."
    )

    # Secondary slices, R2 and tails: written once the results were in, every
    # pattern the prose states asserted here.
    above_candidates = [(k, m) for k in secondary for m, _ in RATIO_FAMILIES]
    assert len(above_candidates) == 8
    above = [(k, m) for k, m in above_candidates if ci(f"ci_nested_{m}_{k}")[1] > 0]
    assert above == [(6, "none")], above
    assert np.array_equal(
        np.round(ratio["ci_nested_none_6"], 3), np.round(ratio["ci_alone_density_none_6"], 3)
    )
    raw_gap = max(
        abs(ratio[f"N_none_acc_{k}"].mean() - ratio[f"Q_none_acc_{k}"].mean()) for k in secondary
    )
    cosine_gap = max(
        abs(ratio[f"N_cosine_acc_{k}"].mean() - ratio[f"A_acc_{k}"].mean()) for k in ks
    )
    assert raw_gap < 0.006 and cosine_gap < 0.004, (raw_gap, cosine_gap)
    assert abs(ratio["N_none_acc_0"].mean() - ratio["Q_none_acc_0"].mean()) > 0.03
    worse_ll = [k for k in secondary if ci(f"ci_nested_ll_cosine_{k}")[2] < 0]
    assert worse_ll == [8, 9], worse_ll
    assert all(ci(f"ci_nested_ll_cosine_{k}")[1] <= 0 for k in secondary)
    betti_above = [k for k in secondary if adds[f"ci_nested_h1_cosine_{k}"][1] > 0]
    assert betti_above == [6, 7], betti_above
    text_secondary = (
        "Of the eight secondary comparisons, slices 6–9 in both normalizations, one "
        f"accuracy interval lies wholly above zero: raw at slice 6, "
        f"{fmt_ci(ratio['ci_nested_none_6'])}. It is not density plus ratio. At slice 6 "
        f"the four raw ratio scores alone reach {pm(ratio['Q_none_acc_6'])}, the "
        f"nested arm {pm(ratio['N_none_acc_6'])}, and the nested arm's interval against "
        "the density arm is the ratio-alone arm's to the last digit shown: the nested fit "
        "classifies with the ratio scores and leaves out the density arm. The same "
        f"holds at every secondary slice: the raw nested arm stays within {raw_gap:.3f} "
        f"of the raw ratio scores alone, and the cosine nested arm, at every slice "
        f"including slice 0, within {cosine_gap:.3f} of the density arm. Each nested "
        "fit follows one of its two blocks, and the next subsection shows that the "
        "scale of the features decides which. With the cosine scores added, the "
        "density arm's log-loss gets worse at slices 8 and 9 "
        f"({fmt_ci(ratio['ci_nested_ll_cosine_8'])} and "
        f"{fmt_ci(ratio['ci_nested_ll_cosine_9'])}) and improves at no secondary slice. "
        "Section 7's cosine H1 Betti scores, by comparison, add accuracy at slices 6 "
        f"and 7 ({fmt_ci(adds['ci_nested_h1_cosine_6'])} and "
        f"{fmt_ci(adds['ci_nested_h1_cosine_7'])}). Were the eight comparisons "
        "independent, chance alone would put at least one interval above zero in "
        f"{1 - 0.975 ** len(above_candidates):.0%} of runs."
    )

    # How much of each ratio score the density arm explains, against the Betti scores.
    betti_index = [1, 3]  # h1_none, h1_cosine in section 7's pc_r2
    ratio_r2 = ratio["ratio_r2"]
    betti_r2 = adds["pc_r2"][ks][:, betti_index]
    ratio_pc1, ratio_rest = float(np.median(ratio_r2[..., 0])), float(np.median(ratio_r2[..., 1:]))
    betti_pc1, betti_rest = float(np.median(betti_r2[..., 0])), float(np.median(betti_r2[..., 1:]))
    assert abs(ratio_pc1 - betti_pc1) < 0.05 and abs(ratio_rest - betti_rest) < 0.05
    cosine_pc1 = float(np.median(ratio_r2[:, 1, 0]))
    assert cosine_pc1 > 0.8, cosine_pc1
    evr_min = min(float(ratio[f"ratio_none_evr_{k}"][0]) for k in ks)
    count_r = min(
        abs(float(np.corrcoef(ratio[f"ratio_none_scores_{k}"][:, 0], ratio[f"n_bars_{k}"])[0, 1]))
        for k in ks
    )
    assert evr_min > 0.998 and count_r > 0.998, (evr_min, count_r)
    text_r2 = (
        "Hardly more than the Betti curves are. Over the five slices and both "
        "normalizations, the density arm explains a median R² of "
        f"{ratio_pc1:.2f} of the first ratio score and {ratio_rest:.2f} of scores 2–4; "
        f"for the H1 Betti scores of the same slices the medians are {betti_pc1:.2f} "
        f"and {betti_rest:.2f}. Part of this is the bar count: the first raw score "
        f"carries at least {np.floor(evr_min * 1000) / 10:.1f}% of the variance at every slice and "
        f"correlates with the number of bars at |r| ≥ {np.floor(count_r * 1000) / 1000:.3f}, "
        "and the number of bars grows with the number of cells. But the cosine curves, "
        "where the count is divided out, are no freer of density: the median R² of "
        f"their first score is {cosine_pc1:.2f}. The ratio does not change when one "
        "slide is rescaled, and the slides differ in more than scale: where the cells "
        "are denser they are also arranged differently, and π follows the arrangement. "
        "Cells are also far from the independent points of the universality "
        "conjecture; nuclei cannot overlap, and how tightly they pack changes with "
        "density."
    )

    per_cell = {k: ratio[f"n_bars_{k}"] / ratio[f"kept_{k}"] for k in ks}
    y = ratio["y"]

    def class_medians(values: np.ndarray) -> tuple[float, float]:
        return float(np.median(values[y == 0])), float(np.median(values[y == 1]))

    tails = {}
    for k in ks:
        quantiles = ratio[f"log_pi_quantiles_{k}"]
        tails[k] = {
            "bars": class_medians(per_cell[k]),
            "q50": class_medians(quantiles[:, 0]),
            "q99": class_medians(quantiles[:, 2]),
            "q999": class_medians(quantiles[:, 3]),
            "max": class_medians(ratio[f"max_log_pi_{k}"]),
        }
    assert [float(v) for v in ratio["levels"]] == [0.5, 0.9, 0.99, 0.999]
    for k in secondary:
        for key in ("bars", "q99", "q999", "max"):
            healthy, cirrhosis = tails[k][key]
            assert cirrhosis > healthy, (k, key)
        healthy, cirrhosis = tails[k]["q50"]
        assert abs(cirrhosis - healthy) < 0.005, (k, healthy, cirrhosis)
    gap_max = {k: tails[k]["max"][1] - tails[k]["max"][0] for k in ks}
    assert gap_max[0] < min(gap_max[k] for k in secondary)
    for key in ("q99", "q999"):
        assert abs(tails[0][key][1] - tails[0][key][0]) < 0.02, key
    assert tails[0]["bars"][0] > tails[0]["bars"][1]
    # Upper bound on the share of each curve's squared L2 norm, the Gram diagonal
    # sum_ij min(v_i, v_j), that lies beyond the 0.99 quantile: there S(t) counts
    # at most 0.01 n + 1 bars, for t up to the largest log pi.
    tail_share = max(
        float(np.max(
            (0.01 * ratio[f"n_bars_{k}"] + 1.0) ** 2
            * (ratio[f"max_log_pi_{k}"] - ratio[f"log_pi_quantiles_{k}"][:, 2])
            / np.diag(ratio[f"ratio_gram_{k}"])
        ))
        for k in ks
    )
    assert tail_share < 0.02, tail_share

    def pair(k: int, key: str) -> str:
        healthy, cirrhosis = tails[k][key]
        return f"{healthy:.3f} against {cirrhosis:.3f}"

    text_tails = (
        "Where the classes do differ is the upper tail. At slices 6–9, cirrhotic "
        "slides have more H1 bars per cell (class medians "
        f"{pair(6, 'bars')} at slice 6, {pair(9, 'bars')} at slice 9, healthy first) "
        "and longer-lived ones: the class medians of the 0.99 and 0.999 quantiles of "
        "log π and of its maximum are higher for cirrhosis at each of these slices, "
        "while the median bar barely differs (at most "
        f"{max(abs(tails[k]['q50'][1] - tails[k]['q50'][0]) for k in secondary):.3f} "
        "in log π). At slice 0 the tails nearly coincide: the 0.99 quantile is "
        f"{pair(0, 'q99')}, the 0.999 quantile {pair(0, 'q999')}, the maximum differs "
        f"by {gap_max[0]:.2f} against {min(gap_max[k] for k in secondary):.2f}–"
        f"{max(gap_max[k] for k in secondary):.2f} at slices 6–9, and healthy slides "
        f"have slightly more bars per cell ({pair(0, 'bars')}). The fPCA compares whole "
        "survival curves in L², and the tail where the classes separate is a small part "
        "of them: beyond the 0.99 quantile lie one bar in a hundred and at most "
        f"{np.ceil(tail_share * 1000) / 10:.1f}% of any slide's squared norm ∫S(t)² dt, so "
        "the scores are shaped by the bulk of short-lived bars."
    )

    return {
        "gram_rel": f"{gram_rel:.0e}",
        "gate": gate,
        "text_primary": text_primary,
        "text_secondary": text_secondary,
        "text_r2": text_r2,
        "text_tails": text_tails,
        "headline": headline,
        **ratio_tables(ratio, adds),
    }


# Section 8's post-hoc check, added 2026-11-17 after its results were seen: the
# nested arms refitted with standardized features (scaled_nested.py).
SCORE_SETS = (
    ("ratio", "none", "H1 ratio, raw"),
    ("ratio", "cosine", "H1 ratio, cosine"),
    ("betti", "none", "H1 Betti, raw"),
    ("betti", "cosine", "H1 Betti, cosine"),
)


def figure_scaled(scaled: dict[str, np.ndarray]) -> bytes:
    """The scores' weight share before and after standardizing, and the standardized intervals."""
    from matplotlib.lines import Line2D

    slices = [int(k) for k in scaled["slices"]]
    positions = np.arange(len(slices), dtype=float)
    offsets = (-0.21, -0.07, 0.07, 0.21)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    panels = (
        (axes[0, 0], "share", "P", "Protocol fits: the scores' share of the weight",
         "share of Σ |coef| × SD"),
        (axes[0, 1], "share", "Z", "Standardized fits: the scores' share of the weight",
         "share of Σ |coef| × SD"),
        (axes[1, 0], "nested", "Z", "Standardized: density + scores against density, accuracy",
         "accuracy difference"),
        (axes[1, 1], "nested_ll", "Z", "Standardized: density + scores against density, log-loss",
         "log-loss reduction"),
    )
    for ax, kind, prefix, title, ylabel in panels:
        for offset, (source, m, _) in zip(offsets, SCORE_SETS):
            if kind == "share":
                values = np.array(
                    [scaled[f"{prefix}_{source}_{m}_share_{k}"].ravel() for k in slices]
                )
                mid = np.median(values, axis=1)
                lo, hi = np.quantile(values, 0.25, axis=1), np.quantile(values, 0.75, axis=1)
            else:
                ci = np.array([scaled[f"ci_{kind}_{source}_{m}_{k}"] for k in slices])
                mid, lo, hi = ci[:, 0], ci[:, 1], ci[:, 2]
            _errorbar(ax, positions + offset, mid, lo, hi,
                      color=ORANGE if source == "ratio" else MUTED, filled=m == "none",
                      marker="o" if source == "ratio" else "^")
        if kind == "share":
            ax.set_ylim(-0.03, 1.03)
        else:
            ax.axhline(0.0, color=MUTED, ls="--", lw=1.0, zorder=0)
        ax.set_xticks(positions, [str(k) for k in slices])
        ax.set_xlabel("density slice k")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.grid(axis="y", alpha=0.7)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=5.5, color=ORANGE, label="H1 ratio, raw"),
        Line2D([], [], ls="none", marker="o", ms=5.5, color=ORANGE, mfc=SURFACE,
               mew=1.2, label="H1 ratio, cosine"),
        Line2D([], [], ls="none", marker="^", ms=5.5, color=MUTED, label="H1 Betti, raw"),
        Line2D([], [], ls="none", marker="^", ms=5.5, color=MUTED, mfc=SURFACE,
               mew=1.2, label="H1 Betti, cosine"),
    ]
    fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=4)
    return png_bytes(fig)


def section8_posthoc(
    scaled: dict[str, np.ndarray], ratio: dict[str, np.ndarray], refit7: dict[str, np.ndarray]
) -> dict[str, str]:
    """Prose and table for section 8's post-hoc feature-scale check.

    Every number is read from ``scaled_nested.npz``, except the protocol arm's
    log-loss interval it is compared with (``ratio_slices.npz``).  The check and this prose
    were written after section 8's results were seen; every pattern the prose
    states is asserted here.
    """
    ks = [int(k) for k in scaled["slices"]]
    secondary = [k for k in ks if k != 0]

    def pm(values: np.ndarray) -> str:
        mean, sd = mean_sd(values)
        return f"{mean:.3f} ± {sd:.3f}"

    def share(prefix: str, source: str, m: str, k: int) -> np.ndarray:
        return scaled[f"{prefix}_{source}_{m}_share_{k}"].ravel()

    def ci(kind: str, source: str, m: str, k: int) -> np.ndarray:
        return scaled[f"ci_{kind}_{source}_{m}_{k}"]

    # Why the protocol's nested test is scale-sensitive.
    density_sd = scaled["density_sd_0"]
    assert np.sum(density_sd == 0.0) == 1
    d_lo, d_hi = float(density_sd[density_sd > 0].min()), float(density_sd.max())
    cos_sd, raw_sd = scaled["score_sd_ratio_cosine_0"], scaled["score_sd_ratio_none_0"]
    assert cos_sd.max() < 0.25 * d_lo and raw_sd.min() > 100 * d_hi
    cos0 = share("P", "ratio", "cosine", 0)
    assert np.median(cos0) < 0.01 and cos0.max() < 0.02, (np.median(cos0), cos0.max())
    raw_medians = [float(np.median(share("P", "ratio", "none", k))) for k in secondary]
    assert min(raw_medians) > 0.999, raw_medians
    small_c = [float(np.mean(scaled[f"P_ratio_none_C_{k}"] <= 1.001e-3)) for k in secondary]
    betti_sd = np.concatenate([scaled[f"score_sd_betti_none_{k}"] for k in secondary])
    betti_medians = [float(np.median(share("P", "betti", "none", k))) for k in secondary]
    assert min(betti_medians) > 0.8, betti_medians
    text_scale = (
        "The nested arms fit the density features and the scores in one L2-penalized "
        "logistic regression, each feature on its own scale, as section 7 does. The "
        "penalty is not scale-free: a feature with a small spread across slides needs a "
        "large coefficient to move the prediction, and the penalty charges for the "
        "coefficient. At slice 0 the density features have standard deviations from "
        f"{d_lo:.3f} to {d_hi:.3f} across slides (the kept fraction, 1 on every slide, is "
        f"constant), the four cosine ratio scores {cos_sd.min():.4f} to "
        f"{cos_sd.max():.4f}, and the four raw ratio scores {raw_sd.min():,.0f} to "
        f"{raw_sd.max():,.0f}. To see what the fits make of this, each fold's weight "
        "share of the scores is measured: Σ |β<sub>j</sub>| SD<sub>j</sub> over the four "
        "score columns divided by the same sum over all columns, SD<sub>j</sub> being the "
        "column's spread in the training fold. In the protocol fits at slice 0 the cosine "
        f"ratio scores carry a median {np.median(cos0):.3f} of the weight, at most "
        f"{cos0.max():.3f}: the cosine nested arm is the density arm. At slices 6–9 the "
        "raw ratio scores carry a median 1.000 at every slice, and the inner "
        f"cross-validation picks C ≤ 10⁻³ in {100 * min(small_c):.0f}–"
        f"{100 * max(small_c):.0f}% of the folds: the fit drops the density arm, which is "
        "why the raw nested arm there is the ratio-alone arm. Section 7's raw H1 Betti "
        f"scores, with standard deviations from {betti_sd.min():,.0f} to "
        f"{betti_sd.max():,.0f}, crowd out the density arm in the same way (median share "
        f"{min(betti_medians):.2f}–{max(betti_medians):.2f} at slices 6–9). The "
        "pre-registered test was therefore weaker than intended: its cosine arm could not "
        "show a gain from scores it gave almost no weight, and at slices 6–9 its raw arms "
        "compared the ratio with density rather than density plus ratio with density."
    )

    # The check.
    scaling = np.array([scaled[f"ci_scaling_density_{k}"] for k in ks])
    assert np.all((scaling[:, 1] <= 0) & (scaling[:, 2] >= 0))
    z_medians = [float(np.median(share("Z", s, m, k))) for s, m, _ in SCORE_SETS for k in ks]
    assert min(z_medians) > 0.04, min(z_medians)
    text_check = (
        "This check was not pre-registered; it was chosen after the results above were "
        "seen. <code>scaled_nested.py</code> refits every nested arm and the density arm "
        "with a StandardScaler, fitted on the outer training fold only, in front of the "
        "same estimator, with the same folds, seeds and bootstrap indices, for the ratio "
        "and, for comparison, section 7's H1 Betti scores. Refitted without the scaler, "
        "every nested arm equals the one reported above, seed for seed. Standardizing the "
        f"density arm alone moves its accuracy by {scaling[:, 0].min():+.3f} to "
        f"{scaling[:, 0].max():+.3f} across the five slices, no interval excluding zero. "
        f"Standardized, the scores carry a median {min(z_medians):.2f} to "
        f"{max(z_medians):.2f} of the weight, so neither block is shut out by its scale."
    )

    # The results.
    for s, m, _ in SCORE_SETS:
        assert ci("nested", s, m, 0)[1] <= 0 and ci("nested_ll", s, m, 0)[1] <= 0, (s, m)
    protocol_ll = ratio["ci_nested_ll_cosine_0"]
    assert protocol_ll[1] > 0 and np.array_equal(ratio["slices"], scaled["slices"])
    above = [(k, s, m) for k in secondary for s, m, _ in SCORE_SETS if ci("nested", s, m, k)[1] > 0]
    assert above == [
        (6, "ratio", "none"), (6, "betti", "none"), (7, "betti", "none"),
        (8, "betti", "none"), (8, "betti", "cosine"),
    ], above
    assert not any(ci("nested_ll", s, m, k)[1] > 0 for k in ks for s, m, _ in SCORE_SETS)
    betti_gains = [(k, s, m) for k, s, m in above if s == "betti"]
    assert all(ci("nested_ll", s, m, k)[0] < 0 for k, s, m in betti_gains)
    assert ci("nested_ll", "ratio", "none", 6)[1] < 0 < ci("nested_ll", "ratio", "none", 6)[2]
    assert ci("nested", "ratio", "cosine", 9)[2] < 0 and ci("nested_ll", "ratio", "cosine", 9)[2] < 0
    # Context from section 7's refit (scaled_section7.npz): the gains sit where the
    # slice's own standardized density arm is weak, and the best of them only
    # matches the best standardized density arm of any slice.
    gain_acc = {(k, s, m): float(scaled[f"Z_{s}_{m}_acc_{k}"].mean()) for k, s, m in above}
    best_gain = max(gain_acc, key=gain_acc.get)
    density_z = {k: float(refit7[f"Z_density_acc_{k}"].mean()) for k in range(10)}
    best_k = max(density_z, key=density_z.get)
    gain_slices = sorted({k for k, _, _ in above})
    weak = [density_z[k] for k in gain_slices]
    assert all(np.array_equal(refit7[f"Z_density_correct_{k}"], scaled[f"Z_density_correct_{k}"]) for k in ks)
    assert abs(gain_acc[best_gain] - density_z[best_k]) < 0.005, (gain_acc[best_gain], density_z[best_k])
    assert max(weak) < density_z[best_k] - 0.05
    gain_label = next(label for s, m, label in SCORE_SETS if (s, m) == best_gain[1:])
    text_posthoc_context = (
        " Section 7's refit with the same scaler narrows it further: these gains sit where "
        "the slice's own standardized density arm is weak "
        f"({min(weak):.3f}–{max(weak):.3f} at slices {gain_slices[0]}–{gain_slices[-1]}, "
        f"against {density_z[best_k]:.3f} at slice {best_k}), and the best of them, "
        f"{gain_acc[best_gain]:.3f} ({gain_label.replace(', ', ' ')}, slice {best_gain[0]}), "
        f"only matches what the density arm of slice {best_k} reaches alone."
    )
    text_posthoc = (
        "At slice 0 the result stands. With standardized features the ratio scores "
        f"change accuracy by {fmt_ci(ci('nested', 'ratio', 'none', 0))} raw and "
        f"{fmt_ci(ci('nested', 'ratio', 'cosine', 0))} cosine, the H1 Betti scores by "
        f"{fmt_ci(ci('nested', 'betti', 'none', 0))} and "
        f"{fmt_ci(ci('nested', 'betti', 'cosine', 0))}, and no interval lies above zero, "
        "in accuracy or in log-loss: the cosine ratio arm's log-loss reduction, "
        f"{fmt_ci(protocol_ll)} in the protocol fit, becomes "
        f"{fmt_ci(ci('nested_ll', 'ratio', 'cosine', 0))}. At slices 6–9, five of the sixteen accuracy "
        "intervals now lie wholly above zero: the raw ratio scores at slice 6 "
        f"({fmt_ci(ci('nested', 'ratio', 'none', 6))}), the raw H1 Betti scores at "
        f"slices 6, 7 and 8 ({fmt_ci(ci('nested', 'betti', 'none', 6))}, "
        f"{fmt_ci(ci('nested', 'betti', 'none', 7))} and "
        f"{fmt_ci(ci('nested', 'betti', 'none', 8))}) and the cosine Betti scores at "
        f"slice 8 ({fmt_ci(ci('nested', 'betti', 'cosine', 8))}). None comes with a "
        "log-loss interval above zero: for the four Betti gains the mean log-loss gets "
        "worse ("
        + ", ".join(fmt_ci(ci("nested_ll", s, m, k)) for k, s, m in betti_gains)
        + "), and the ratio's reduction at slice 6, "
        f"{fmt_ci(ci('nested_ll', 'ratio', 'none', 6))}, includes zero. The cosine ratio "
        "scores make both worse at slice 9 "
        f"({fmt_ci(ci('nested', 'ratio', 'cosine', 9))} in accuracy, "
        f"{fmt_ci(ci('nested_ll', 'ratio', 'cosine', 9))} in log-loss). So standardizing "
        "leaves the pre-registered conclusion where it was: at slice 0 neither the ratio "
        "nor the Betti curves add measurably to density. At the densest slices, "
        "topology on standardized features may add accuracy, but that rests on a choice "
        "made after the results were seen, on sixteen uncorrected comparisons and on no "
        "gain in log-loss. It is a lead for a pre-registered test, not a finding."
        + text_posthoc_context
    )
    headline = (
        "A post-hoc check found the nested test, here and in section 7, sensitive to the "
        "scale of the features. Refitted with standardized features, the ratio still adds "
        f"nothing measurable at slice 0 ({fmt_ci(ci('nested', 'ratio', 'none', 0))} raw, "
        f"{fmt_ci(ci('nested', 'ratio', 'cosine', 0))} cosine); at slices "
        f"{gain_slices[0]}–{gain_slices[-1]} some ratio and H1 Betti arms gain accuracy "
        "over their slice's density arm, none log-loss, and the best only matches slice "
        f"{best_k}'s density arm: at most a lead for a pre-registered test."
    )

    rows = []
    for k in ks:
        for index, (source, m, label) in enumerate(SCORE_SETS):
            rows.append(
                (
                    str(k) if index == 0 else "",
                    label,
                    f"{np.median(share('P', source, m, k)):.3f} → "
                    f"{np.median(share('Z', source, m, k)):.3f}",
                    pm(scaled[f"Z_density_acc_{k}"]) if index == 0 else "",
                    pm(scaled[f"Z_{source}_{m}_acc_{k}"]),
                    fmt_ci(ci("nested", source, m, k)),
                    fmt_ci(ci("nested_ll", source, m, k)),
                )
            )
    table_posthoc = html_table(
        [
            "slice", "scores",
            "score weight share<br><span class='unit'>protocol → standardized</span>",
            "density arm, standardized<br><span class='unit'>accuracy</span>",
            "density + scores, standardized<br><span class='unit'>accuracy</span>",
            "density + scores, standardized<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "density + scores, standardized<br><span class='unit'>log-loss reduction [95% CI]</span>",
        ],
        rows,
        cls="left2 grouped",
    )
    return {
        "text_scale": text_scale,
        "text_check": text_check,
        "text_posthoc": text_posthoc,
        "headline": headline,
        "table_posthoc": table_posthoc,
    }


def section8_alone(alone: dict[str, np.ndarray]) -> dict[str, str]:
    """Prose and table for the post-hoc refit of section 8's ratio-alone arms.

    Every number is read from ``scaled_ratio_alone.npz``.  The check was chosen
    after section 8's results were seen; how to read it (held back if the
    standardized accuracy interval against the protocol arm lies wholly above
    zero) was written into the ledger before it ran.  Every pattern the prose
    states is asserted here.
    """
    ks = [int(k) for k in alone["slices"]]
    top = float(alone["cs_grid"][-1])

    def pm(values: np.ndarray) -> str:
        mean, sd = mean_sd(values)
        return f"{mean:.3f} ± {sd:.3f}"

    def ci(kind: str, m: str, k: int) -> np.ndarray:
        return alone[f"ci_{kind}_{m}_{k}"]

    def top_share(field: str, m: str, k: int) -> float:
        return float(np.mean(np.isclose(alone[f"{field}_{m}_C_{k}"], top)))

    cos_sd = np.concatenate([alone[f"score_sd_cosine_{k}"] for k in ks])
    cos_top = [top_share("P_Q", "cosine", k) for k in ks]
    # Claims: the raw arms never choose the top C, the standardized cosine arms
    # never do, and no arm is held back by the reading fixed before the run.
    assert all(top_share("P_Q", "none", k) == 0.0 for k in ks)
    assert all(top_share("Z_Q", "cosine", k) == 0.0 for k in ks)
    held = [(k, m) for k in ks for m, _ in RATIO_FAMILIES if ci("scaling", m, k)[1] > 0]
    assert not held, held
    scaling = {(k, m): ci("scaling", m, k) for k in ks for m, _ in RATIO_FAMILIES}
    largest = max(scaling, key=lambda key: scaling[key][0])
    assert largest == (0, "cosine"), largest
    ll_above = [(k, m) for k in ks for m, _ in RATIO_FAMILIES if ci("scaling_ll", m, k)[1] > 0]
    ll_below = [(k, m) for k in ks for m, _ in RATIO_FAMILIES if ci("scaling_ll", m, k)[2] < 0]
    assert ll_above == [(0, "cosine")] and not ll_below, (ll_above, ll_below)
    means = [float(v[0]) for v in scaling.values()]
    text = (
        "The ratio-alone arms of the results table meet the scale problem in another form. With "
        "one block of features, a common spread only shifts the C the fit needs, as "
        "long as that C lies on the grid. The cosine ratio scores, with standard "
        f"deviations from {cos_sd.min():.4f} to {cos_sd.max():.3f} across slides, sit "
        "at its edge: in the protocol fits the cosine arms choose the grid's largest "
        f"C, 10⁴, in {100 * min(cos_top):.0f}–{100 * max(cos_top):.0f}% of the folds "
        f"({100 * cos_top[ks.index(0)]:.0f}% at slice 0), the raw arms never. Since "
        "the inner cross-validation keeps the first of equally good values, a fold at "
        "the top preferred it to every smaller C. This check, too, was chosen after the "
        "results were seen; <code>scaled_ratio_alone.py</code> refits the ten arms with "
        "a StandardScaler fitted on the outer training fold, with the same folds, seeds "
        "and bootstrap indices. Refitted without it, every arm equals the one in the "
        "results table; standardized, no cosine fold chooses the top C. The reading was written "
        "into the ledger before the run: an arm is held back by its scale if its "
        "standardized accuracy interval against the protocol arm lies wholly above "
        f"zero. None is. Accuracy changes by {min(means):+.3f} to {max(means):+.3f}, the "
        f"largest at slice 0, cosine ({fmt_ci(scaling[largest])}). Log-loss improves "
        f"there ({fmt_ci(ci('scaling_ll', 'cosine', 0))}; AUC "
        f"{float(alone['P_Q_cosine_auc_0'].mean()):.3f} → "
        f"{float(alone['Z_Q_cosine_auc_0'].mean()):.3f}); elsewhere no log-loss interval "
        "excludes zero. So the ratio-alone accuracies in the results table stand; "
        "at slice 0 the cosine arm's probabilities are sharper with standardized scores. "
        "Standardizing also evens out the four scores' spreads within the block, so the "
        "change is not the grid's reach alone."
    )
    rows = []
    for k in ks:
        for index, (m, label) in enumerate(RATIO_FAMILIES):
            rows.append(
                (
                    str(k) if index == 0 else "",
                    label,
                    pm(alone[f"P_Q_{m}_acc_{k}"]),
                    f"{100 * top_share('P_Q', m, k):.0f}%",
                    pm(alone[f"Z_Q_{m}_acc_{k}"]),
                    fmt_ci(ci("scaling", m, k)),
                    fmt_ci(ci("scaling_ll", m, k)),
                    fmt_ci(ci("alone_chance", m, k)),
                )
            )
    table = html_table(
        [
            "slice", "ratio",
            "ratio alone, protocol<br><span class='unit'>accuracy</span>",
            "protocol folds<br><span class='unit'>at the top C</span>",
            "ratio alone, standardized<br><span class='unit'>accuracy</span>",
            "standardized − protocol<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "standardized − protocol<br><span class='unit'>log-loss reduction [95% CI]</span>",
            "standardized<br><span class='unit'>above chance [95% CI]</span>",
        ],
        rows,
        cls="left2 grouped",
    )
    return {"text": text, "table": table}


# Section 7's post-hoc refit, added 2026-11-17 after section 8's check: every
# nested arm of section 7 refitted with standardized features
# (scaled_section7.py).
SECTION7_SUFFIXES = tuple(str(k) for k in range(10)) + ("all",)


def figure_scaled_section7(refit: dict[str, np.ndarray]) -> bytes:
    """Section 7's nested arms: weight shares before and after standardizing, standardized intervals."""
    positions = np.append(np.arange(10, dtype=float), 10.8)
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    panels = (
        (axes[0, 0], "share", "P", "Protocol fits: the scores' share of the weight",
         "share of Σ |coef| × SD"),
        (axes[0, 1], "share", "Z", "Standardized fits: the scores' share of the weight",
         "share of Σ |coef| × SD"),
        (axes[1, 0], "nested", "Z", "Standardized: density + scores against density, accuracy",
         "accuracy difference"),
        (axes[1, 1], "nested_ll", "Z", "Standardized: density + scores against density, log-loss",
         "log-loss reduction"),
    )
    for ax, kind, prefix, title, ylabel in panels:
        for (key, label), offset in zip(MODELS, offsets):
            if kind == "share":
                values = np.array(
                    [refit[f"{prefix}_{key}_share_{s}"].ravel() for s in SECTION7_SUFFIXES]
                )
                mid = np.median(values, axis=1)
                lo, hi = np.quantile(values, 0.25, axis=1), np.quantile(values, 0.75, axis=1)
            else:
                ci = np.array([refit[f"ci_{kind}_{key}_{s}"] for s in SECTION7_SUFFIXES])
                mid, lo, hi = ci[:, 0], ci[:, 1], ci[:, 2]
            _family_errorbar(ax, positions + offset, mid, lo, hi, key,
                             label=label if ax is axes[0, 0] else None)
        if kind == "share":
            ax.set_ylim(-0.03, 1.03)
        else:
            ax.axhline(0.0, color=MUTED, ls="--", lw=1.0, zorder=0)
        ax.axvline(9.9, color=RULE, lw=1.0, zorder=0)
        ax.set_xticks(positions, [str(k) for k in range(10)] + ["all"])
        ax.set_xlabel("density slice k; all = the combined ten-slice model")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.grid(axis="y", alpha=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=4)
    return png_bytes(fig)


def section7_posthoc(refit: dict[str, np.ndarray], adds: dict[str, np.ndarray]) -> dict[str, str]:
    """Prose and table for section 7's post-hoc refit with standardized features.

    Every number is read from ``scaled_section7.npz`` and, for the protocol arms
    it is set against, ``density_adds.npz``.  The refit was chosen after section
    8's results were seen; how to read it (the combined models first) was
    written into the ledger before it ran.  Every pattern the prose states is
    asserted here.
    """
    short = {key: label.replace(", ", " ") for key, label in MODELS}
    raw = ("h0_none", "h1_none")
    cosine = ("h0_cosine", "h1_cosine")
    slices = [str(k) for k in range(10)]

    def pm(values: np.ndarray) -> str:
        mean, sd = mean_sd(values)
        return f"{mean:.3f} ± {sd:.3f}"

    def ci(kind: str, key: str, s: str) -> np.ndarray:
        return refit[f"ci_{kind}_{key}_{s}"]

    def median_share(prefix: str, key: str, s: str) -> float:
        return float(np.median(refit[f"{prefix}_{key}_share_{s}"]))

    # Why: the spreads, and where the protocol fits put the weight.
    density_sd = np.concatenate([refit[f"density_sd_{s}"] for s in SECTION7_SUFFIXES])
    d_lo, d_hi = float(density_sd[density_sd > 0].min()), float(density_sd.max())
    raw_sd = np.concatenate([refit[f"score_sd_{key}_{s}"] for key in raw for s in SECTION7_SUFFIXES])
    cos_sd = np.concatenate(
        [refit[f"score_sd_{key}_{s}"] for key in cosine for s in SECTION7_SUFFIXES]
    )
    assert raw_sd.min() > 100 * d_hi
    raw_shares = [median_share("P", key, s) for key in raw for s in slices]
    h0_raw_shares = [median_share("P", "h0_none", s) for s in slices]
    cos_shares = [median_share("P", key, s) for key in cosine for s in slices]
    # Claim: in the combined raw models the scores take all the weight, at a
    # C so small that the density features are effectively left out.
    all_raw_share = [median_share("P", key, "all") for key in raw]
    all_raw_small_c = [float(np.mean(refit[f"P_{key}_C_all"] <= 1.001e-3)) for key in raw]
    assert min(all_raw_share) > 0.999 and min(all_raw_small_c) >= 0.8
    all_raw_ll = [float(adds[f"ci_nested_ll_{key}_all"][0]) for key in raw]
    scaling = np.array([refit[f"ci_scaling_density_{s}"] for s in SECTION7_SUFFIXES])
    assert np.all((scaling[:, 1] <= 0) & (scaling[:, 2] >= 0))
    z_shares = [median_share("Z", key, s) for key, _ in MODELS for s in SECTION7_SUFFIXES]
    text_scale = (
        "Section 8's post-hoc check found the protocol's nested fits sensitive to the "
        "spread of the features: the L2 penalty charges a feature with a small spread "
        "more for the same effect on the prediction. The nested arms of this section "
        "mix the same scales. Across the ten slices, the density features have standard "
        f"deviations from {d_lo:.4f} to {d_hi:.3f} across slides, the raw scores from "
        f"{raw_sd.min():,.0f} to {raw_sd.max():,.0f} and the cosine scores from "
        f"{cos_sd.min():.3f} to {cos_sd.max():.3f}. In the fits above, the raw scores "
        f"carry a median {min(raw_shares):.2f}–{max(raw_shares):.2f} of the "
        f"classifier's weight per slice (H0 raw {min(h0_raw_shares):.2f}–"
        f"{max(h0_raw_shares):.2f}) and the cosine scores {min(cos_shares):.2f}–"
        f"{max(cos_shares):.2f}. In the two combined raw models the scores carry all of "
        f"it, and the inner cross-validation picks C ≤ 10⁻³ in "
        f"{100 * min(all_raw_small_c):.0f}–{100 * max(all_raw_small_c):.0f}% of the "
        "folds: those arms classify with the scores alone, and their log-loss is "
        f"{-all_raw_ll[0]:.3f} (H0) and {-all_raw_ll[1]:.3f} (H1) worse than the density "
        "arm's. "
        "This check was not pre-registered; it was chosen after section 8's results "
        "were seen, and how to read it was written into the ledger before it ran: the "
        "combined models first, as in this section's verdict. "
        "<code>scaled_section7.py</code> refits all 44 nested arms and the eleven "
        "density arms with a StandardScaler, fitted on the outer training fold only, in "
        "front of the same estimator, with the same folds, seeds and bootstrap indices. "
        "Refitted without the scaler, every nested arm equals the one above, seed for "
        "seed, and the standardized arms that section 8 also refits (H1 at slices 0 and "
        "6–9) come out the same. Standardizing a density arm alone moves it by "
        f"{scaling[:, 0].min():+.3f} to {scaling[:, 0].max():+.3f}, no interval excluding "
        f"zero; standardized, the scores carry a median {min(z_shares):.2f} to "
        f"{max(z_shares):.2f} of the weight."
    )

    # The combined models, the main comparison.
    combined_ci = {key: ci("nested", key, "all") for key, _ in MODELS}
    combined_ll = {key: ci("nested_ll", key, "all") for key, _ in MODELS}
    combined_means = [float(v[0]) for v in combined_ci.values()]
    assert all(v[1] <= 0 for v in combined_ci.values()), "a standardized combined CI is above 0"
    combined_below = [key for key, v in combined_ci.items() if v[2] < 0]
    assert combined_below == ["h1_cosine"], combined_below
    assert all(v[0] < 0 for v in combined_ll.values())
    combined_ll_below = [key for key, v in combined_ll.items() if v[2] < 0]
    assert combined_ll_below == ["h1_cosine"], combined_ll_below

    # Per slice, against the protocol tallies of section 7.
    tally = tally_cis(refit, "nested")
    tally_ll = tally_cis(refit, "nested_ll")
    protocol = tally_cis(adds, "nested")
    above_slices = sorted({k for k, _ in tally["above"]})
    assert above_slices == list(range(above_slices[0], above_slices[-1] + 1)), above_slices
    protocol_slices = sorted({k for k, _ in protocol["above"]})
    assert protocol_slices == list(range(protocol_slices[0], protocol_slices[-1] + 1))
    assert len(tally["below"]) == 1 and not tally_ll["above"]
    density_k = np.array([float(refit[f"Z_density_acc_{s}"].mean()) for s in slices])
    mean_gain = np.array(
        [np.mean([ci("nested", key, s)[0] for key, _ in MODELS]) for s in slices]
    )
    gain_r = float(np.corrcoef(density_k, mean_gain)[0, 1])
    assert gain_r < -0.5, gain_r
    nested_k = {
        (k, key): float(refit[f"Z_{key}_acc_{k}"].mean())
        for k in range(above_slices[0], above_slices[-1] + 1)
        for key, _ in MODELS
    }
    best_middle = max(nested_k, key=nested_k.get)
    best_density = int(np.argmax(density_k))
    assert nested_k[best_middle] < density_k[best_density]
    below_k, below_label = tally["below"][0][0], short[
        next(key for key, label in MODELS if label == tally["below"][0][1])
    ]
    text_results = (
        "The combined models do not change. With every slice's density summary "
        f"standardized ({pm(refit['Z_density_acc_all'])}), adding all forty scores "
        f"changes accuracy by {min(combined_means):+.3f} to {max(combined_means):+.3f} "
        "across the four families; no interval lies above zero, and H1 cosine's lies "
        f"below ({fmt_ci(combined_ci['h1_cosine'])}). Log-loss gets worse in all four "
        f"({min(float(v[0]) for v in combined_ll.values()):+.3f} to "
        f"{max(float(v[0]) for v in combined_ll.values()):+.3f}), H1 cosine's interval "
        "wholly below zero. Per slice, "
        f"{len(tally['above'])} of the 40 accuracy intervals now lie wholly above zero, "
        f"all at slices {above_slices[0]}–{above_slices[-1]} "
        f"({list_by_slice(tally['above'])}), against {len(protocol['above'])} at slices "
        f"{protocol_slices[0]}–{protocol_slices[-1]} in the fits above; one lies below "
        f"({below_label}, slice {below_k}). The gain again follows the weakness of the "
        "slice's density arm (correlation of a slice's mean Δ with its standardized "
        f"density-arm accuracy r = {gain_r:.2f}), and the best nested arm at slices "
        f"{above_slices[0]}–{above_slices[-1]}, {nested_k[best_middle]:.3f} "
        f"({short[best_middle[1]]}, slice {best_middle[0]}), stays below the "
        f"{density_k[best_density]:.3f} that the standardized density arm of slice "
        f"{best_density} reaches alone. In log-loss no interval lies above zero and "
        f"{len(tally_ll['below'])} lie below ({list_by_slice(tally_ll['below'])}). So "
        "standardizing leaves this section's conclusion where it was: the scores lift "
        "the weaker density arms of single slices, and with every slice's density "
        "summary in the model they add nothing."
    )
    headline = (
        "Refitted post-hoc with every feature standardized in the training fold "
        "(section 8 shows why the scale matters), section 7's nested arms tell the same "
        f"story: the combined models gain nothing ({min(combined_means):+.3f} to "
        f"{max(combined_means):+.3f}, no interval above zero), and per slice the scores "
        f"lift only the weaker density arms of slices {above_slices[0]}–"
        f"{above_slices[-1]}, to at most {nested_k[best_middle]:.3f}, below the "
        f"{density_k[best_density]:.3f} of slice {best_density}'s standardized density "
        "arm."
    )

    rows = []
    for s in SECTION7_SUFFIXES:
        for index, (key, label) in enumerate(MODELS):
            rows.append(
                (
                    ("all (combined)" if s == "all" else s) if index == 0 else "",
                    label,
                    f"{median_share('P', key, s):.3f} → {median_share('Z', key, s):.3f}",
                    pm(refit[f"Z_density_acc_{s}"]) if index == 0 else "",
                    pm(refit[f"Z_{key}_acc_{s}"]),
                    fmt_ci(ci("nested", key, s)),
                    fmt_ci(ci("nested_ll", key, s)),
                )
            )
    table = html_table(
        [
            "slice", "model",
            "score weight share<br><span class='unit'>protocol → standardized</span>",
            "density arm, standardized<br><span class='unit'>accuracy</span>",
            "density + scores, standardized<br><span class='unit'>accuracy</span>",
            "density + scores, standardized<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "density + scores, standardized<br><span class='unit'>log-loss reduction [95% CI]</span>",
        ],
        rows,
        cls="left2 grouped",
    )
    return {"text_scale": text_scale, "text_results": text_results, "headline": headline, "table": table}


def section7_single_block(single: dict[str, np.ndarray], adds: dict[str, np.ndarray]) -> dict[str, str]:
    """Prose and table for the post-hoc refit of section 7's residual and combined PC arms.

    Every number is read from ``scaled_single_block.npz`` and, for the protocol
    tallies it is set against, ``density_adds.npz``.  The check was chosen after
    the results were seen; how to read it was written into the ledger (Task 10)
    before it ran.  Every pattern the prose states is asserted here.
    """
    top, bottom = float(single["cs_grid"][-1]), float(single["cs_grid"][0])
    raw = ("h0_none", "h1_none")
    slices = range(10)

    def pm(values: np.ndarray) -> str:
        mean, sd = mean_sd(values)
        return f"{mean:.3f} ± {sd:.3f}"

    def share_at(field: str, suffix: str, value: float) -> float:
        return float(np.mean(np.isclose(single[f"{field}_C_{suffix}"], value)))

    # Where the protocol residual fits put C: the raw residuals below the grid's
    # reach, the cosine H0 ones at its top.
    raw_sd_min = min(float(single[f"res_sd_{key}_{k}"].min()) for key in raw for k in slices)
    raw_sd_max = max(float(single[f"res_sd_{key}_{k}"].max()) for key in raw for k in slices)
    weakest = bottom * raw_sd_min**2
    assert weakest > top, weakest
    exponent = int(np.floor(np.log10(weakest)))
    weakest_text = (
        f"{weakest / 10**exponent:.1f} × 10"
        + "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(digit)] for digit in str(exponent))
    )
    raw_bottom = [share_at(f"P_R_{key}", str(k), bottom) for key in raw for k in slices]
    assert min(raw_bottom) >= 0.95, min(raw_bottom)
    assert all(share_at(f"P_R_{key}", str(k), top) == 0.0 for key in raw for k in slices)
    h0_cos_top = [share_at("P_R_h0_cosine", str(k), top) for k in slices]
    h1_cos_top = [share_at("P_R_h1_cosine", str(k), top) for k in slices]
    assert max(h1_cos_top) <= 0.05
    z_top = max(share_at(f"Z_R_{key}", str(k), top) for key, _ in MODELS for k in slices)
    z_raw_bottom = max(share_at(f"Z_R_{key}", str(k), bottom) for key in raw for k in slices)

    # The reading fixed before the run, for the residual arms.
    scaling = {(k, key): single[f"ci_scaling_R_{key}_{k}"] for k in slices for key, _ in MODELS}
    assert not [kk for kk, v in scaling.items() if v[1] > 0]
    lost = [kk for kk, v in scaling.items() if v[2] < 0]
    assert lost == [(1, "h1_none"), (2, "h1_none")], lost
    scaling_means = [float(v[0]) for v in scaling.values()]
    # Section 7's residual claims, re-counted.
    tally = tally_cis(single, "res_chance")
    tally_ll = tally_cis(single, "res_ll")
    protocol = tally_cis(adds, "res_chance")
    assert tally["above"] == protocol["above"] and not tally["below"] and not tally_ll["above"]
    dense = sum(k in (8, 9) for k, _ in tally["above"])

    # The combined PC arms.
    pc_scaling = {
        (part, key): single[f"ci_scaling_{part}_{key}_all"] for part in ("PC1", "PC24") for key, _ in MODELS
    }
    held = [pk for pk, v in pc_scaling.items() if v[1] > 0]
    assert held == [("PC24", "h0_none"), ("PC24", "h1_none")], held
    held_ll = [float(single[f"ci_scaling_ll_{part}_{key}_all"][0]) for part, key in held]
    assert min(held_ll) > 1.0
    pc_bottom = [share_at(f"P_PC24_{key}", "all", bottom) for key in raw]
    pc24 = {key: single[f"ci_pc24_{key}_all"] for key, _ in MODELS}
    assert not [key for key, v in pc24.items() if v[1] > 0]
    pc24_below = [key for key, v in pc24.items() if v[2] < 0]
    assert pc24_below == ["h0_cosine"], pc24_below
    protocol_below = [key for key, _ in MODELS if adds[f"ci_pc24_{key}_all"][2] < 0]
    assert protocol_below == ["h0_none", "h1_none"], protocol_below
    pc1_h0_cos = single["ci_pc1_h0_cosine_all"]
    assert pc1_h0_cos[1] <= 0 <= pc1_h0_cos[2]
    pc24_means = [float(v[0]) for v in pc24.values()]

    text = (
        "The residual and PC arms hold one block of features each, where a common spread "
        "only shifts the C the fit needs, as long as that C lies on the grid. They meet both "
        f"of its edges. The raw residuals, with standard deviations of {raw_sd_min:,.0f} to "
        f"{raw_sd_max:,.0f}, lie beyond its bottom: on them even the smallest C, 10⁻⁴, is a "
        f"penalty no stronger than C = {weakest_text} on standardized residuals, above the "
        "grid's top, and the protocol fits choose that smallest C in "
        f"{100 * min(raw_bottom):.0f}–{100 * max(raw_bottom):.0f}% of their folds. The cosine "
        "H0 residuals sit at its top: their fits choose C = 10⁴ in up to "
        f"{100 * max(h0_cos_top):.0f}% of the folds (the cosine H1 ones in at most "
        f"{100 * max(h1_cos_top):.0f}%). <code>scaled_single_block.py</code> refits the 40 "
        "residual arms, standardizing the residuals after the same in-fold "
        "residualisation, and the eight combined PC arms, with a StandardScaler fitted on "
        "the outer training fold; refitted without it, every arm equals the saved one. "
        "Standardized, at most "
        f"{100 * z_top:.0f}% of a residual arm's folds choose the top C, and the raw residual "
        f"arms keep the smallest C in up to {100 * z_raw_bottom:.0f}% of their folds. By the "
        "reading written "
        "into the ledger before the run, an arm is held back by its scale if its "
        "standardized accuracy interval against the protocol arm lies wholly above zero. "
        f"No residual arm is: accuracy changes by {min(scaling_means):+.3f} to "
        f"{max(scaling_means):+.3f}, and two raw H1 arms, at slices 1 and 2, lose "
        f"({fmt_ci(scaling[(1, 'h1_none')])} and {fmt_ci(scaling[(2, 'h1_none')])}). The "
        "residual claims above come back unchanged: the same "
        f"{len(tally['above'])} of 40 intervals lie above chance ({dense} of the 8 at "
        "slices 8–9), none below, and in log-loss none clears the prevalence. Two PC arms "
        "are held back: scores 2–4 of every slice, raw, whose fits had chosen the smallest "
        f"C in {100 * min(pc_bottom):.0f}–{100 * max(pc_bottom):.0f}% of the folds, gain "
        f"{fmt_ci(pc_scaling[('PC24', 'h0_none')])} (H0) and "
        f"{fmt_ci(pc_scaling[('PC24', 'h1_none')])} (H1) in accuracy, and their log-loss "
        f"falls by {held_ll[0]:.3f} and {held_ll[1]:.3f}. They still fall short of the "
        f"standardized density arm ({fmt_ci(pc24['h0_none'])} and "
        f"{fmt_ci(pc24['h1_none'])}); across the four families scores 2–4 alone differ from "
        f"it by {min(pc24_means):+.3f} to {max(pc24_means):+.3f}, with one interval (H0 "
        "cosine) wholly below zero and none above, against two below in the protocol fits. "
        "H0 cosine PC1 alone stays level with the density arm "
        f"({fmt_ci(pc1_h0_cos)}). So section 7's reading of these arms stands."
    )
    headline = (
        "The residual and PC arms, refitted the same way, leave the residual tallies "
        f"unchanged ({len(tally['above'])} of 40 above chance, none by log-loss) and scores "
        "2–4 alone below the density arm."
    )

    rows = []
    for k in slices:
        for index, (key, label) in enumerate(MODELS):
            rows.append(
                (
                    f"residual, slice {k}" if index == 0 else "",
                    label,
                    pm(single[f"P_R_{key}_acc_{k}"]),
                    f"{100 * share_at(f'P_R_{key}', str(k), top):.0f}%",
                    pm(single[f"Z_R_{key}_acc_{k}"]),
                    fmt_ci(single[f"ci_scaling_R_{key}_{k}"]),
                    fmt_ci(single[f"ci_scaling_ll_R_{key}_{k}"]),
                    fmt_ci(single[f"ci_res_chance_{key}_{k}"]),
                    fmt_ci(single[f"ci_res_ll_{key}_{k}"]),
                )
            )
    for part, title in (("PC1", "PC1 of every slice"), ("PC24", "PCs 2–4 of every slice")):
        for index, (key, label) in enumerate(MODELS):
            rows.append(
                (
                    title if index == 0 else "",
                    label,
                    pm(single[f"P_{part}_{key}_acc_all"]),
                    f"{100 * share_at(f'P_{part}_{key}', 'all', top):.0f}%",
                    pm(single[f"Z_{part}_{key}_acc_all"]),
                    fmt_ci(single[f"ci_scaling_{part}_{key}_all"]),
                    fmt_ci(single[f"ci_scaling_ll_{part}_{key}_all"]),
                    fmt_ci(single[f"ci_{part.lower()}_{key}_all"]),
                    fmt_ci(single[f"ci_{part.lower()}_ll_{key}_all"]),
                )
            )
    table = html_table(
        [
            "arm", "model",
            "protocol<br><span class='unit'>accuracy</span>",
            "protocol folds<br><span class='unit'>at the top C</span>",
            "standardized<br><span class='unit'>accuracy</span>",
            "standardized − protocol<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "standardized − protocol<br><span class='unit'>log-loss reduction [95% CI]</span>",
            "standardized vs baseline<br><span class='unit'>Δ accuracy [95% CI]</span>",
            "standardized vs baseline<br><span class='unit'>log-loss reduction [95% CI]</span>",
        ],
        rows,
        cls="left2 grouped",
    )
    return {"text": text, "headline": headline, "table": table}


def combined_verdict(
    slices: list[dict[str, np.ndarray]],
    combined: dict[str, np.ndarray],
    n_slides: int,
) -> dict[str, object]:
    control_mean, control_sd = mean_sd(combined["control_accuracy"])
    records: list[dict[str, object]] = []
    for key, label in MODELS:
        base_mean, base_sd = mean_sd(slices[0][f"{key}_accuracy"])
        mean, sd = mean_sd(combined[f"{key}_accuracy"])
        noise_slice0 = 2.0 * np.sqrt(
            binomial_se(mean, n_slides) ** 2 + binomial_se(base_mean, n_slides) ** 2
            + sd**2 + base_sd**2
        )
        noise_control = 2.0 * np.sqrt(
            binomial_se(mean, n_slides) ** 2
            + binomial_se(control_mean, n_slides) ** 2
            + sd**2 + control_sd**2
        )
        diff_slice0 = mean - base_mean
        diff_control = mean - control_mean
        records.append(
            {
                "key": key,
                "label": label,
                "mean": mean,
                "sd": sd,
                "base_mean": base_mean,
                "base_sd": base_sd,
                "diff_slice0": diff_slice0,
                "noise_slice0": noise_slice0,
                "diff_control": diff_control,
                "noise_control": noise_control,
                "margin": min(diff_slice0 - noise_slice0, diff_control - noise_control),
            }
        )
    hits = [
        record
        for record in records
        if record["diff_slice0"] > record["noise_slice0"]
        and record["diff_control"] > record["noise_control"]
    ]
    return {
        "records": records,
        "hits": hits,
        "control_mean": control_mean,
        "control_sd": control_sd,
        "highest": max(records, key=lambda record: record["mean"]),
        "largest_lift": max(records, key=lambda record: record["diff_slice0"]),
    }


def accuracy_cell(values: np.ndarray) -> str:
    mean, sd = mean_sd(values)
    return f"{float(values[SEED0]):.3f} / {mean:.3f} ± {sd:.3f}"


def verdict(slices: list[dict[str, np.ndarray]], n_slides: int) -> dict[str, object]:
    hits: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    beat_slice0 = 0
    beat_control = 0
    total = 0
    for key, label in MODELS:
        baseline = slices[0][f"{key}_accuracy"]
        baseline_mean, baseline_sd = mean_sd(baseline)
        for k in range(1, len(slices)):
            values = slices[k][f"{key}_accuracy"]
            mean, sd = mean_sd(values)
            control = slices[k]["control_accuracy"]
            control_mean, control_sd = mean_sd(control)
            noise_slice0 = 2.0 * np.sqrt(
                binomial_se(mean, n_slides) ** 2
                + binomial_se(baseline_mean, n_slides) ** 2
                + sd**2 + baseline_sd**2
            )
            noise_control = 2.0 * np.sqrt(
                binomial_se(mean, n_slides) ** 2
                + binomial_se(control_mean, n_slides) ** 2
                + sd**2 + control_sd**2
            )
            diff_slice0 = mean - baseline_mean
            diff_control = mean - control_mean
            margin = min(diff_slice0 - noise_slice0, diff_control - noise_control)
            record = {
                "key": key,
                "label": label,
                "slice": k,
                "mean": mean,
                "sd": sd,
                "diff_slice0": diff_slice0,
                "noise_slice0": noise_slice0,
                "diff_control": diff_control,
                "noise_control": noise_control,
                "margin": margin,
            }
            total += 1
            if diff_slice0 > noise_slice0:
                beat_slice0 += 1
            if diff_control > noise_control:
                beat_control += 1
            if diff_slice0 > noise_slice0 and diff_control > noise_control:
                hits.append(record)
            if best is None or record["margin"] > best["margin"]:
                best = record
    return {
        "hits": hits,
        "best": best,
        "beat_slice0": beat_slice0,
        "beat_control": beat_control,
        "total": total,
    }


def section56_posthoc(
    ph: dict[str, np.ndarray],
    slices: list[dict[str, np.ndarray]],
    combined: dict[str, np.ndarray],
    n_slides: int,
) -> dict[str, str]:
    """Prose and table for the post-hoc refit of the PH-only arms behind the verdict.

    Every number is read from ``scaled_ph_only.npz``, whose unscaled refits give
    back the per-seed accuracies in ``slice_<k>.npz`` and ``combined.npz``.  The
    check was chosen after the results were seen; how to read it (the verdict
    rule re-applied unchanged with every arm standardized) was written into the
    ledger (Task 11) before it ran.  Every pattern the prose states is asserted.
    """
    top, bottom = float(ph["cs_grid"][-1]), float(ph["cs_grid"][0])
    keys = [key for key, _ in MODELS]
    label_of = {key: label for key, label in MODELS} | {"control": "control"}
    suffixes = [str(k) for k in range(10)] + ["all"]
    arms = keys + ["control"]

    # The protocol arms here are the report's, seed by seed.
    for k in range(10):
        for key in keys:
            assert np.array_equal(ph[f"P_{key}_acc_{k}"], slices[k][f"{key}_accuracy"])
        assert np.array_equal(ph[f"P_control_acc_{k}"], slices[k]["control_accuracy"])
    for key in keys:
        assert np.array_equal(ph[f"P_{key}_acc_all"], combined[f"{key}_accuracy"])
    assert np.array_equal(ph["P_control_acc_all"], combined["control_accuracy"])

    def share_at(arm: str, s: str, value: float) -> float:
        return float(np.mean(np.isclose(ph[f"P_{arm}_C_{s}"], value)))

    # Where the protocol fits put C.
    raw = ("h0_none", "h1_none")
    raw_sd = [float(ph[f"feature_sd_{key}_all"].min()) for key in raw]
    raw_sd_max = max(float(ph[f"feature_sd_{key}_all"].max()) for key in raw)
    slice_raw_bottom = [share_at(key, str(k), bottom) for key in raw for k in range(10)]
    all_raw_bottom = [share_at(key, "all", bottom) for key in raw]
    slice_cos_top = [share_at(key, str(k), top) for key in ("h0_cosine", "h1_cosine") for k in range(10)]
    control_top = share_at("control", "all", top)
    assert min(slice_raw_bottom) >= 0.95 and min(all_raw_bottom) >= 0.5

    # The per-arm reading of the earlier checks.
    scaling = {(s, arm): ph[f"ci_scaling_{arm}_{s}"] for s in suffixes for arm in arms}
    held = [key for key, v in scaling.items() if v[1] > 0]
    assert held == [("all", "h1_none")], held
    lost = [key for key, v in scaling.items() if v[2] < 0]
    assert lost and all(s != "all" for s, _ in lost), lost
    others = [float(v[0]) for key, v in scaling.items() if key not in held]
    held_ci = scaling[("all", "h1_none")]
    held_ll = float(ph["ci_scaling_ll_h1_none_all"][0])
    h0_raw = scaling[("all", "h0_none")]
    h0_raw_ll = ph["ci_scaling_ll_h0_none_all"]
    assert h0_raw[1] <= 0 <= h0_raw[2] and h0_raw_ll[1] > 0
    lost_text = "; ".join(
        f"slice {s}: " + ", ".join(label_of[arm].replace(", ", " ") for t, arm in lost if t == s)
        for s in dict.fromkeys(s for s, _ in lost)
    )

    # The verdict rule, re-applied unchanged with every arm standardized.
    z_slices = [
        {**{f"{key}_accuracy": ph[f"Z_{key}_acc_{k}"] for key in keys},
         "control_accuracy": ph[f"Z_control_acc_{k}"]}
        for k in range(10)
    ]
    z_combined = {**{f"{key}_accuracy": ph[f"Z_{key}_acc_all"] for key in keys},
                  "control_accuracy": ph["Z_control_acc_all"]}
    decision = verdict(z_slices, n_slides)
    protocol = verdict(slices, n_slides)
    combined_decision = combined_verdict(z_slices, z_combined, n_slides)
    assert not decision["hits"] and not combined_decision["hits"]
    best, best_protocol = decision["best"], protocol["best"]
    assert (best["slice"], best["key"]) == (best_protocol["slice"], best_protocol["key"])
    near = max(combined_decision["records"], key=lambda record: record["margin"])
    assert near["key"] == "h1_none"
    assert near["diff_slice0"] > near["noise_slice0"] and near["diff_control"] < near["noise_control"]

    text = (
        "The arms behind the verdict feed the fPCA scores to the protocol estimator as "
        "they come, as <code>fpca_classify</code> does by design, and the checks of "
        "sections 7 and 8 found such arms held back where the features' spread puts the "
        "C they need off the grid. These arms meet both edges of it. The raw scores, "
        f"with standard deviations from {min(raw_sd):,.0f} to {raw_sd_max:,.0f}, lie "
        "beyond its bottom: the per-slice raw arms choose its smallest C, 10⁻⁴, in "
        f"{100 * min(slice_raw_bottom):.0f}–{100 * max(slice_raw_bottom):.0f}% of the "
        "folds and the combined raw models in "
        f"{100 * min(all_raw_bottom):.0f}–{100 * max(all_raw_bottom):.0f}%; the per-slice "
        f"cosine arms choose its largest, 10⁴, in up to {100 * max(slice_cos_top):.0f}% "
        f"of the folds, and the combined control in {100 * control_top:.0f}%. "
        "<code>scaled_ph_only.py</code> refits all 55 arms — the 40 four-score arms, the "
        "ten per-slice controls, the four combined models and the combined control — "
        "with a StandardScaler fitted on the outer training fold, as in section 7; "
        "refitted without it, every arm gives back the per-seed accuracies and confusion "
        "matrices that <code>fpca_classify</code> and <code>combined_classify.py</code> "
        "stored. By the per-arm reading of the earlier checks, one arm is held back by "
        f"its scale: the combined H1 raw model, {fmt_ci(held_ci)} in accuracy "
        f"({float(ph['P_h1_none_acc_all'].mean()):.3f} → "
        f"{float(ph['Z_h1_none_acc_all'].mean()):.3f}), its log-loss {held_ll:.3f} "
        "better; the combined H0 raw model gains "
        f"{fmt_ci(h0_raw)} in accuracy and {float(h0_raw_ll[0]):.3f} in log-loss. "
        f"Elsewhere accuracy changes by {min(others):+.3f} to {max(others):+.3f}, and "
        f"{len(lost)} intervals lie below zero ({lost_text}). The verdict rule, "
        "re-applied unchanged with every arm standardized — the reading written into the "
        "ledger before the run — gives the same verdict. None of the "
        f"{decision['total']} (slice, model) combinations beats both slice 0 and its "
        "own control by more than the noise; the closest approach is again slice "
        f"{best['slice']} ({best['label']}), {best['mean']:.3f} ± {best['sd']:.3f}, "
        f"{best['diff_slice0']:+.3f} over slice 0 (needs {best['noise_slice0']:.3f}) and "
        f"{best['diff_control']:+.3f} over its control (needs "
        f"{best['noise_control']:.3f}), a margin of {best['margin']:+.3f} against "
        f"{best_protocol['margin']:+.3f} before. Of the {decision['total']}, "
        f"{decision['beat_slice0']} beat slice 0 alone ({protocol['beat_slice0']} before) and "
        f"{decision['beat_control']} their control alone "
        f"({protocol['beat_control']} before). No combined model clears both bars: the "
        f"standardized combined {near['label'].replace(', ', ' ')} model, {near['mean']:.3f} ± "
        f"{near['sd']:.3f}, clears the slice-0 bar ({near['diff_slice0']:+.3f}, needs "
        f"{near['noise_slice0']:.3f}) but beats the combined control "
        f"({combined_decision['control_mean']:.3f} ± {combined_decision['control_sd']:.3f}) "
        f"by only {near['diff_control']:+.3f} (needs {near['noise_control']:.3f}). So the "
        "verdict does not rest on the protocol's use of raw scores."
    )
    headline = (
        "Refitted post-hoc with every feature standardized in the training fold, the "
        "verdict is the same: no (slice, model) pair and no combined model clears both "
        f"bars, and the closest approach is again slice {best['slice']} "
        f"({best['label']}), margin {best['margin']:+.3f}."
    )

    rows = []
    for s in suffixes:
        rows.append(
            ("all (combined)" if s == "all" else s,)
            + tuple(
                f"{float(ph[f'P_{arm}_acc_{s}'].mean()):.3f} → "
                f"{float(ph[f'Z_{arm}_acc_{s}'].mean()):.3f}"
                for arm in arms
            )
        )
    table = html_table(["slice"] + [label for _, label in MODELS] + ["density-only control"], rows)
    return {"text": text, "headline": headline, "table": table}


def main() -> None:
    density = load_npz(RESULTS / "density.npz")
    slices = [load_npz(RESULTS / f"slice_{k}.npz") for k in range(10)]
    metas = [json.loads((RESULTS / f"slice_{k}.json").read_text()) for k in range(10)]
    combined = load_npz(RESULTS / "combined.npz")
    combined_meta = json.loads((RESULTS / "combined.json").read_text())
    adds = load_npz(RESULTS / "density_adds.npz")
    adds_meta = json.loads((RESULTS / "density_adds.json").read_text())
    ratio = load_npz(RESULTS / "ratio_slices.npz")
    ratio_meta = json.loads((RESULTS / "ratio_slices.json").read_text())
    scaled = load_npz(RESULTS / "scaled_nested.npz")
    scaled_meta = json.loads((RESULTS / "scaled_nested.json").read_text())
    refit7 = load_npz(RESULTS / "scaled_section7.npz")
    refit7_meta = json.loads((RESULTS / "scaled_section7.json").read_text())
    alone = load_npz(RESULTS / "scaled_ratio_alone.npz")
    alone_meta = json.loads((RESULTS / "scaled_ratio_alone.json").read_text())
    single = load_npz(RESULTS / "scaled_single_block.npz")
    single_meta = json.loads((RESULTS / "scaled_single_block.json").read_text())
    ph_only = load_npz(RESULTS / "scaled_ph_only.npz")
    ph_only_meta = json.loads((RESULTS / "scaled_ph_only.json").read_text())

    n_cells = int(density["rho"].size)
    slide_labels = [str(x) for x in density["slide_labels"]]
    n_healthy = sum(label == "healthy" for label in slide_labels)
    n_cirrhosis = sum(label == "cirrhosis" for label in slide_labels)
    counts = density["slide_counts"]
    deciles = density["deciles"]
    total_runtime = (
        sum(meta["runtime_seconds"] for meta in metas)
        + combined_meta["runtime_seconds"]
        + adds_meta["runtime_seconds"]
        + ratio_meta["runtime_seconds"]
        + scaled_meta["runtime_seconds"]
        + refit7_meta["runtime_seconds"]
        + alone_meta["runtime_seconds"]
        + single_meta["runtime_seconds"]
        + ph_only_meta["runtime_seconds"]
    )

    # Slice-0 reproduction of tutorial 04, read from the saved results.
    tutorial = {
        key: float(slices[0][f"{key}_accuracy"][SEED0]) for key, _ in MODELS
    }
    n_slides = int(slices[0]["group_ids"].size)
    min_kept = int(min(int(s["kept_per_slide"].min()) for s in slices))
    se_values = [
        binomial_se(mean_sd(s[f"{key}_accuracy"])[0], n_slides)
        for s in slices
        for key, _ in MODELS
    ]
    se_lo, se_hi = min(se_values), max(se_values)
    combined_result = combined_verdict(slices, combined, n_slides)
    s56 = section56_posthoc(ph_only, slices, combined, n_slides)

    # Section 7: what the scores add beyond density.  Corrected 2026-11-17; the
    # first version (density_adds of 2026-11-16) is kept in superseded/.
    s7 = section7(adds, combined)
    adds_combined_png = figure_adds_combined(adds)
    adds_slice_png = figure_adds_per_slice(adds)
    s7p = section7_posthoc(refit7, adds)
    s7b = section7_single_block(single, adds)
    scaled7_png = figure_scaled_section7(refit7)
    old_density_features = int(adds["rich_scores_all"].shape[1])

    # Section 8: the H1 death/birth ratio, added 2026-11-17.
    s8 = section8(ratio, adds, slices)
    ratio_png = figure_ratio(ratio, adds)
    ratio_quantiles_png = figure_ratio_quantiles(ratio)
    s8p = section8_posthoc(scaled, ratio, refit7)
    s8a = section8_alone(alone)
    for key in ("y", "group_ids"):
        assert np.array_equal(alone[key], ratio[key])
    for k in (int(k) for k in alone["slices"]):
        for m, _ in RATIO_FAMILIES:
            assert np.array_equal(alone[f"P_Q_{m}_correct_{k}"], ratio[f"Q_{m}_correct_{k}"])
    scaled_png = figure_scaled(scaled)
    n_train_min = n_slides - int(np.ceil(n_slides / 5))

    # Density summary statements, computed from the saved arrays.
    caption_example = (
        f"γ ≥ {float(deciles[5]):.0f} cells/mm² keeps "
        f"{float(np.median(slices[5]['kept_fraction'][slices[5]['group_label'] == 'healthy'])):.2f} "
        f"of a healthy slide's cells (median) and "
        f"{float(np.median(slices[5]['kept_fraction'][slices[5]['group_label'] == 'cirrhosis'])):.2f} "
        f"of a cirrhosis slide's."
    )

    density_png = figure_density(density)
    kept_png = figure_kept(density, slices)
    summary_png = figure_summary(slices)
    combined_png = figure_combined(slices, combined)
    n_features = int(combined["h0_none_scores"].shape[1])

    # Table.
    header = (
        "<tr><th>slice</th><th>q</th><th>threshold<br><span class='unit'>cells/mm²</span></th>"
        "<th>cells kept</th>"
        + "".join(f"<th>{label}</th>" for _, label in MODELS)
        + "<th>density-only control</th></tr>"
    )
    body_rows = []
    for k, s in enumerate(slices):
        q = f"{0.1 * k:.1f}"
        cells = f"{int(s['kept_total']):,}"
        cells_frac = 100.0 * float(s["kept_total"]) / n_cells
        threshold = float(s["threshold"])
        threshold_text = f"{threshold:.1f}" if threshold < 1 else f"{threshold:.0f}"
        row = (
            f"<tr><td>{k}</td><td>{q}</td><td>{threshold_text}</td>"
            f"<td>{cells}<br><span class='unit'>{cells_frac:.1f}%</span></td>"
        )
        for key, _ in MODELS:
            row += f"<td>{accuracy_cell(s[f'{key}_accuracy'])}</td>"
        row += f"<td>{accuracy_cell(s['control_accuracy'])}</td></tr>"
        body_rows.append(row)
    table = (
        "<div class='table-wrap'><table><thead>" + header + "</thead><tbody>"
        + "".join(body_rows) + "</tbody></table></div>"
    )

    slice_figures = []
    for k, s in enumerate(slices):
        png = (FIGURES / f"slice_{k:02d}.png").read_bytes()
        kept = int(s["kept_total"])
        frac = 100.0 * kept / n_cells
        threshold = float(s["threshold"])
        if k == 0:
            condition = "all cells"
            alt_condition = "all cells"
            rounding_note = (
                f" Its q0 threshold is the minimum density, {threshold:.1f} "
                "cells/mm², which the panel title rounds to 0."
            )
        else:
            condition = f"γ ≥ {threshold:.0f} cells/mm²"
            alt_condition = f"rho >= {threshold:.0f} cells/mm^2"
            rounding_note = ""
        slice_figures.append(
            "<figure>"
            + img_tag(
                png,
                f"Density slice {k}: {alt_condition}, "
                f"{kept} of {n_cells} cells kept; fPCA score scatters PC1 vs PC2.",
            )
            + f"<figcaption><strong>Slice {k}.</strong> Keeps {condition}; "
            f"{kept:,} cells ({frac:.1f}% of the cohort). One point per slide, "
            f"coloured by class; rows H0/H1, columns raw/cosine.{rounding_note}"
            "</figcaption></figure>"
        )

    # Combined experiment table and result sentence.
    combined_records = combined_result["records"]
    combined_rows = []
    for record in combined_records:
        key = record["key"]
        seed0 = float(combined[f"{key}_accuracy"][SEED0])
        combined_rows.append(
            f"<tr><td>{record['label']}</td>"
            f"<td>{record['base_mean']:.3f} ± {record['base_sd']:.3f}</td>"
            f"<td>{seed0:.3f}</td>"
            f"<td>{record['mean']:.3f} ± {record['sd']:.3f}</td>"
            f"<td>{record['diff_slice0']:+.3f}</td>"
            f"<td>{record['diff_control']:+.3f}</td></tr>"
        )
    combined_rows.append(
        "<tr><td>density-only control</td><td>—</td>"
        f"<td>{float(combined['control_accuracy'][SEED0]):.3f}</td>"
        f"<td>{combined_result['control_mean']:.3f} ± "
        f"{combined_result['control_sd']:.3f}</td><td>—</td><td>—</td></tr>"
    )
    combined_table = (
        "<div class='table-wrap'><table><thead><tr><th>model</th>"
        "<th>slice 0<br><span class='unit'>mean ± SD</span></th>"
        "<th>all ten<br><span class='unit'>seed 0</span></th>"
        "<th>all ten<br><span class='unit'>mean ± SD</span></th>"
        "<th>vs slice 0</th><th>vs control</th></tr></thead><tbody>"
        + "".join(combined_rows)
        + "</tbody></table></div>"
    )
    if combined_result["hits"]:
        hit = combined_result["hits"][0]
        combined_text = (
            f"The combined model {hit['label']} reaches {hit['mean']:.3f} ± "
            f"{hit['sd']:.3f}, {hit['diff_slice0']:+.3f} over its slice-0 baseline "
            f"and {hit['diff_control']:+.3f} over the combined control, clearing "
            f"both noise margins ({hit['noise_slice0']:.3f} and "
            f"{hit['noise_control']:.3f})."
        )
    else:
        highest = combined_result["highest"]
        lift = combined_result["largest_lift"]
        combined_text = (
            "Combining the ten slices into one "
            f"{n_features}-feature model does not change the verdict. The highest "
            f"combined mean is {highest['label']} at {highest['mean']:.3f} ± "
            f"{highest['sd']:.3f} ({highest['diff_slice0']:+.3f} over its slice-0 "
            "baseline); the combined density-only control reaches "
            f"{combined_result['control_mean']:.3f} ± "
            f"{combined_result['control_sd']:.3f}. The largest lift over a "
            f"slice-0 baseline is {lift['label']} "
            f"({lift['diff_slice0']:+.3f}), but no combined model beats both "
            "bars by more than the noise."
        )

    # Verdict.
    decision = verdict(slices, n_slides)
    best = decision["best"]
    assert best is not None
    if decision["hits"]:
        hit = decision["hits"][0]
        verdict_title = "Yes — one slice clears both bars"
        verdict_text = (
            f"Slice {hit['slice']}, {hit['label']}, reaches "
            f"{hit['mean']:.3f} ± {hit['sd']:.3f} (mean ± SD over seeds), "
            f"{hit['diff_slice0']:+.3f} over slice 0 and "
            f"{hit['diff_control']:+.3f} over its density-only control, and both "
            f"gaps exceed the stated noise margins "
            f"({hit['noise_slice0']:.3f} and {hit['noise_control']:.3f})."
        )
        if len(decision["hits"]) > 1:
            verdict_text += (
                f" {len(decision['hits'])} of the {decision['total']} slice/model "
                "combinations clear both bars."
            )
    else:
        verdict_title = "No"
        verdict_text = (
            f"None of the {decision['total']} (slice, model) combinations beats both "
            "slice 0 and its own density-only control by more than the noise. "
            f"The closest approach is slice {best['slice']} ({best['label']}): "
            f"{best['mean']:.3f} ± {best['sd']:.3f}, {best['diff_slice0']:+.3f} "
            f"over slice 0 (needs {best['noise_slice0']:.3f}) and "
            f"{best['diff_control']:+.3f} over its control "
            f"(needs {best['noise_control']:.3f}); the tighter margin is "
            f"{best['margin']:+.3f}. Across the grid, "
            f"{decision['beat_slice0']} of the {decision['total']} combinations "
            f"{'beats' if decision['beat_slice0'] == 1 else 'beat'} slice 0 alone "
            f"and {decision['beat_control']} "
            f"{'beats' if decision['beat_control'] == 1 else 'beat'} their control "
            "alone; neither alone identifies a slice worth adding."
        )

    control0 = slices[0]["control_accuracy"]
    control0_mean, _ = mean_sd(control0)
    control1_mean, _ = mean_sd(slices[1]["control_accuracy"])
    last_control_mean, _ = mean_sd(slices[-1]["control_accuracy"])
    last_best_key, last_best_label = max(
        MODELS, key=lambda model: mean_sd(slices[-1][f"{model[0]}_accuracy"])[0]
    )
    last_best_mean, _ = mean_sd(slices[-1][f"{last_best_key}_accuracy"])

    # The report.
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Density slices of the GTEx liver cohort</title>
<style>
  :root {{
    --ink: {INK}; --ink2: {INK2}; --muted: {MUTED}; --rule: {RULE};
    --surface: {SURFACE}; --page: #f9f9f7; --blue: {BLUE}; --orange: {ORANGE};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.55; font-size: 16px;
  }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 36px 22px 90px; }}
  h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 4px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.3rem; margin: 44px 0 12px; padding-bottom: 6px;
        border-bottom: 1px solid var(--rule); }}
  h3 {{ font-size: 1.05rem; margin: 28px 0 8px; }}
  p, li {{ max-width: 78ch; }}
  .subtitle {{ color: var(--ink2); font-size: 1.05rem; margin: 0 0 8px; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin: 0 0 28px; }}
  .verdict {{ background: var(--surface); border: 1px solid var(--rule);
              border-left: 4px solid var(--orange); padding: 18px 22px;
              margin: 26px 0; border-radius: 4px; }}
  .verdict h2 {{ border: 0; margin: 0 0 6px; font-size: 1.15rem; }}
  .verdict p {{ margin: 0; }}
  figure {{ margin: 26px 0; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 3px; }}
  figcaption {{ color: var(--ink2); font-size: 0.85rem; margin-top: 8px; }}
  .table-wrap {{ overflow-x: auto; margin: 18px 0 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem;
           font-variant-numeric: tabular-nums; background: var(--surface); }}
  th, td {{ border-bottom: 1px solid var(--rule); padding: 7px 8px;
            text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--ink2); font-weight: 600; vertical-align: bottom; }}
  .left2 th:nth-child(2), .left2 td:nth-child(2) {{ text-align: left; }}
  .grouped tr.group td {{ border-top: 1px solid #c3c2b7; }}
  .unit {{ color: var(--muted); font-weight: 400; font-size: 0.72rem; }}
  .correction {{ background: var(--surface); border: 1px solid var(--rule);
                 border-left: 4px solid var(--muted); padding: 12px 18px;
                 margin: 18px 0; border-radius: 4px; }}
  .correction p {{ margin: 0 0 6px; }}
  .correction p:last-child {{ margin-bottom: 0; }}
  .note {{ color: var(--ink2); font-size: 0.85rem; }}
  code {{ font-size: 0.86em; background: #f0efec; padding: 1px 4px; border-radius: 3px; }}
  details {{ margin: 16px 0; }}
  summary {{ cursor: pointer; color: var(--ink2); }}
  ul {{ margin: 8px 0 8px 22px; padding: 0; }}
  li {{ margin: 4px 0; }}
</style>
</head>
<body>
<main>
  <h1>Density slices of the GTEx liver cohort</h1>
  <p class="subtitle">Is the function-Delaunay bifiltration worth adopting in
  topocyte? Ten horizontal slices with γ = 10-nearest-neighbour density.</p>
  <p class="meta">Report generated {dt.date.today().isoformat()} from the saved
  results of <code>run_slices.py</code>, <code>combined_classify.py</code>,
  <code>density_adds.py</code>, <code>ratio_slices.py</code>,
  <code>scaled_nested.py</code>, <code>scaled_section7.py</code>,
  <code>scaled_ratio_alone.py</code>, <code>scaled_single_block.py</code> and
  <code>scaled_ph_only.py</code>; section 7
  corrected, and section 8 and the post-hoc checks of sections 5–8 added, on
  2026-11-17; total compute
  {total_runtime / 60:.1f} min. Library: topocyte
  {html.escape(str(metas[0]["topocyte"]))}, scikit-learn
  {html.escape(str(metas[0]["sklearn"]))}.</p>

  <div class="verdict">
    <h2>Verdict: {verdict_title}</h2>
    <p>{verdict_text}</p>
    <p>{combined_text} {s56['headline']}</p>
    <p>{s7['headline']} {s7p['headline']} {s7b['headline']}</p>
    <p>{s8['headline']} {s8p['headline']}</p>
  </div>

  <h2>1. Question</h2>
  <p>Alonso et al. (<em>Delaunay bifiltrations of functions on point clouds</em>,
  SODA 2024) turn a function on a point cloud into a two-parameter persistence
  module. For topocyte, the candidate function is the local cell density
  γ(x) = 10 / (π r₁₀(x)²), the number of cells per mm² within the distance to
  the 10th nearest other cell. This report measures whether fixing γ at ten
  horizontal levels and running the existing alpha pipeline on each level adds
  anything to the full-cohort analysis of tutorial 04 — specifically, whether
  any slice separates healthy from cirrhotic liver slides better than both the
  unsliced cohort and a density-only control.</p>

  <h2>2. Method</h2>
  <h3>Why no bifiltration code is needed</h3>
  <p>The paper proves that its function-Delaunay bifiltration is topologically
  equivalent to the sublevel-offset bifiltration, the union of r-balls around
  the cells whose γ passes the threshold. The horizontal slice at density d
  therefore has exactly the barcode of the alpha filtration of the cells with
  γ ≥ d. Every slice here is that alpha filtration, computed by
  <code>tc.tl.persistent_homology(..., filtration="alpha", groupby="slide_id")</code>;
  no <code>function_delaunay</code>, <code>multipers</code> or CGAL code is
  involved.</p>

  <h3>Cohort and density</h3>
  <p>The GTEx liver cohort has {n_cells:,} cells on
  {n_healthy + n_cirrhosis} H&amp;E slides ({n_healthy} healthy,
  {n_cirrhosis} cirrhosis), {int(counts.min()):,} to {int(counts.max()):,} cells
  per slide. For every cell, a <code>scipy.spatial.cKDTree</code> query with
  k = 11 gives the distance to the 10th nearest <em>other</em> cell of the same
  slide; r₁₀ is converted with 0.4942 µm/px and γ is reported in cells/mm². No
  subsampling, no approximation, no grid. The ten thresholds are the pooled
  deciles of γ, q = 0, 0.1, …, 0.9, applied unchanged to every slide; slice k
  keeps every cell with γ ≥ d_k, and slice 0 keeps all cells.</p>

  <h3>Per-slice analysis</h3>
  <p>Each slice is processed independently: persistent homology (alpha complex,
  H0 and H1, one computation per slide, all {n_slides} slides present in every
  slice), then four fPCA models — H0 and H1, each raw
  (<code>normalization="none"</code>) and cosine
  (<code>normalization="cosine"</code>) — and a logistic-regression classifier
  on the first four scores of each model. The held-out estimate is tutorial
  04's: stratified 5-fold cross-validation with the regularisation strength
  chosen by an inner 5-fold cross-validation, <code>n_components=4</code>,
  <code>scoring="accuracy"</code>. In sections 4–5 each (slice, model) pair is
  fitted and classified on its own; section 6 adds one classifier that
  concatenates the ten slices; section 7 asks what the scores add beyond a
  density summary of each slice and the slide's size; section 8 repeats that
  test for the death/birth ratio of the H1 bars, which does not change with
  the scale of the cells. Seeds 0–19 repeat the
  classification; seed 0 is tutorial 04's protocol.</p>

  <h3>Density-only control</h3>
  <p>The control uses one feature per slide: the fraction of that slide's cells
  the slice keeps. It is classified by exactly the same estimator and the same
  outer folds as the fPCA models (<code>LogisticRegressionCV(Cs=10,
  inner_cv=5, scoring="accuracy", max_iter=1000)</code> inside
  <code>StratifiedKFold(5, shuffle=True, random_state=seed)</code>). A slice
  counts as useful only if it beats both slice 0 and its own control by more
  than the noise: twice the combined binomial standard error at n = {n_slides}
  slides ({se_lo:.3f}–{se_hi:.3f}) plus the seed-to-seed standard deviation.</p>

  <h3>Assumptions and proposed choices</h3>
  <ul>
    <li><strong>Pixel size.</strong> 0.4942 µm/px is assumed for every slide;
    only the example slides' SVS headers were checked. Because the thresholds
    are deciles of the pooled density, a common scale factor cancels between γ
    and its deciles: slice membership depends only on the ranking of densities,
    so this assumption changes the axis labels and no result.</li>
    <li><strong>fPCA is cohort-wide.</strong> As in tutorial 04, each slice's
    fPCA is fitted on all {n_slides} slides, the held-out ones included. It never sees
    the labels, but the accuracies are not a fully external estimate.</li>
    <li><strong>The density arm is a summary.</strong> Section 7's density arm
    has {s7['per_slice']} features per slice — kept fraction, seven log-density
    quantiles, their mean and SD, and the slide's cell count — not the full
    density field. It carries {s7['per_slice']} numbers per slice against four
    scores, which biases the nested comparison against finding an increment
    from PH.</li>
    <li><strong>Log-loss judges untuned probabilities.</strong> The protocol
    chooses the regularisation strength for accuracy, so its probabilities are
    not tuned for log-loss, and log-loss punishes confident mistakes hard. It is
    reported beside accuracy, not in its place.</li>
    <li><strong>Proposed choices, open to overrule.</strong> Repeating the
    classification and the control for seeds 0–19, reporting seed 0 and the mean
    ± SD over seeds; and scatter plots in PC1 vs PC2 only.</li>
  </ul>

  <h3>Checks before this report</h3>
  <p>Slice 0, seed 0 reproduces tutorial 04's held-out accuracies exactly:
  H0 raw {tutorial['h0_none']:.3f}, H1 raw
  {tutorial['h1_none']:.3f}, H0 cosine
  {tutorial['h0_cosine']:.3f}, H1 cosine
  {tutorial['h1_cosine']:.3f}. All {n_slides} slides are present in
  every slice, no slide falls below {min_kept:,} kept cells, and the pooled
  deciles and kept fractions match the read-only probe of 2026-11-16.
  <code>verify_results.py</code> asserts all of this.</p>

  <h2>3. Density and the ten thresholds</h2>
  <p>The pooled deciles are
  {", ".join(f"{d:.0f}" for d in deciles)} cells/mm² (q = 0 to 0.9). The
  distribution is broad: the top decile starts at {float(deciles[-1]):,.0f}
  cells/mm², and even the smallest slide keeps
  {int(slices[-1]['kept_per_slide'].min()):,} cells there. Cirrhotic tissue is
  denser than healthy tissue, which is exactly why the control exists:
  {caption_example} The per-slide spread is shown in the second figure.</p>
  <figure>{img_tag(density_png, "Empirical survival function of the pooled 10-NN density with the ten decile thresholds marked.")}
  <figcaption>Pooled density of all cells on a log axis, drawn as the exact
  empirical survival function (fraction of cells with γ ≥ x) evaluated at 800
  log-spaced display points. The left panel spans the full range, including the
  isolated low-q0 threshold; the right panel zooms on the nine upper deciles.
  Dashed lines and orange dots mark the thresholds.</figcaption></figure>
  <figure>{img_tag(kept_png, "Per-slide kept fraction against slice index, coloured by class.")}
  <figcaption>Fraction of each slide's cells kept at each threshold, one line
  per slide (thin) and the class median (thick). Cirrhotic slides retain more
  cells at every threshold above the minimum, so a slice's accuracy must be
  compared with what density alone already encodes.</figcaption></figure>

  <h2>4. The ten slices</h2>
  <p>Each panel shows the first two fPCA scores of every slide, coloured by
  class. Slice 0 is all cells and reproduces tutorial 04; the later slices keep
  {float(slices[-1]['kept_total']) / n_cells:.1%} of the cells at q = 0.9.</p>
  {''.join(slice_figures)}

  <h2>5. Accuracy across slices</h2>
  <figure>{img_tag(summary_png, "Held-out accuracy against slice index for the four models and the control.")}
  <figcaption>Held-out accuracy against slice index for the four fPCA models
  and the density-only control. Lines are the mean over 20 seeds, bands ± 1 SD,
  dots the seed-0 result; dotted horizontal lines mark each series' slice-0
  level. At slice 0 the control's feature is constant (kept fraction 1), so it
  can only predict the majority class
  ({control0_mean:.3f}).</figcaption></figure>
  <p>At the highest threshold, q = 0.9, the density-only control alone reaches
  {last_control_mean:.3f}, against {last_best_mean:.3f} for the best fPCA model
  ({last_best_label}). The control rises steadily from {control1_mean:.3f} at
  slice 1, so the topological scores add little once the retained fraction of
  cells is known.</p>

  <h3>Table</h3>
  <p class="note">Each fPCA cell is seed-0 accuracy / mean ± SD over the 20
  seeds; the control column is the same format.</p>
  {table}

  <h2>6. One classifier on all ten slices</h2>
  <p>The models above were fitted and classified one slice at a time. This
  experiment asks the complementary question: do the ten slices together carry
  more than any single one? For each fPCA family, the first four scores that
  every slice's model gives a slide are concatenated over the ten slices —
  {n_features} features per slide — and classified with the same estimator,
  outer folds and seeds. One combined model is used per family rather than one
  160-feature model, because raw and cosine scores, and H0 and H1 scores,
  differ by orders of magnitude and the protocol uses raw scores without
  standardisation. The matching control concatenates the ten kept fractions of
  every slide.</p>
  <figure>{img_tag(combined_png, "Combined-slice accuracy against slice 0 and the combined density control.")}
  <figcaption>Each row is one fPCA family: grey dot the slice-0 mean, coloured
  dot the all-ten-slices mean with ± 1 SD over seeds, dashed line and band the
  combined density-only control.</figcaption></figure>
  <h3>Table</h3>
  <p class="note">Mean ± SD over the 20 seeds unless the column says seed 0.
  The last two columns are descriptive deltas; the noise margins are in the
  text.</p>
  {combined_table}
  <p>{combined_text}</p>
  <h3>Feature scale (post-hoc)</h3>
  <p>{s56['text']}</p>
  <details>
    <summary>Protocol and standardized accuracies</summary>
    <p class="note">Mean held-out accuracy over the 20 seeds, protocol →
    standardized; the last row is the combined ten-slice models and
    control.</p>
    {s56['table']}
  </details>

  <h2>7. Does PH add beyond density?</h2>
  <p>The control in section 5 uses one feature per slide per slice — the kept
  fraction — and it comes close to the best fPCA model. That raises the
  question whether the scores encode anything beyond the slide's density. This
  section replaces the kept fraction with a stronger density arm and runs three
  tests on the saved scores; no persistent homology is recomputed.</p>
  <div class="correction">
    <p><strong>Corrected on 2026-11-17.</strong> The first version of this
    section (2026-11-16) concluded that persistent homology acts as a density
    proxy. That conclusion rested on three flaws.</p>
    <p>Its combined residual arm regressed {n_features} scores on
    {old_density_features} density features with about {n_train_min} training
    slides per fold. The regression fits the training slides exactly, so the
    arm's near-chance accuracy was an artifact; it is dropped. It also judged
    the residual arm against the density arm instead of chance, the null of the
    proxy claim. And its density arm left out slide size, which alone classifies
    at {s7['size_acc']}.</p>
    <p>This version adds slide size to the density arm, tests the residuals per
    slice against chance, and reports log-loss and AUC beside accuracy. The
    score-only arms reproduce the first run exactly; its files are kept in
    <code>superseded/</code>.</p>
  </div>
  <h3>Design</h3>
  <ul>
    <li><strong>Density arm.</strong> {s7['per_slice']} features per slide per
    slice: the kept fraction, the 0.05, 0.10, 0.25, 0.50, 0.75, 0.90 and 0.95
    quantiles of log₁₀γ among the kept cells, their mean and standard deviation,
    and log₁₀ of the slide's cell count — {s7['n_all']} features over the ten
    slices, where slide size enters once. The fraction carries the slide-level
    density, the quantiles the within-slide density field, and the cell count
    the slide size that raw Betti curves scale with.</li>
    <li><strong>Slide size alone.</strong> log₁₀ of the cell count as the only
    feature.</li>
    <li><strong>Nested.</strong> The density arm plus the first four scores; Δ
    is its held-out result minus the density arm's on the same folds and
    seeds.</li>
    <li><strong>Residual, per slice only.</strong> Each of the four scores is
    regressed on the slice's {s7['per_slice']} density features inside every
    outer training fold, and the residuals are classified. If the scores were a
    linear function of these density features, this arm would sit at chance:
    the training fold's majority
    class for accuracy, its prevalence for log-loss. A combined version would
    regress {n_features} scores on {s7['n_all']} features with about
    {n_train_min} training slides, so it is not computed.</li>
    <li><strong>PC breakdown.</strong> PC1 alone — total mass and extent,
    density-shaped — against PCs 2–4, the shape directions, both without
    density features.</li>
    <li><strong>Metrics.</strong> Accuracy, log-loss and AUC, all from the same
    out-of-fold predictions. Log-loss differences are reported as reductions,
    the density arm's (or chance's) log-loss minus the arm's, so that positive
    favours the scores in every column.</li>
    <li><strong>Uncertainty.</strong> Paired bootstrap over the {n_slides}
    slides (4000 draws) of per-slide values averaged over the 20 seeds, so the
    intervals measure slide sampling, not only fold noise. One set of
    resampling indices serves every comparison.</li>
  </ul>
  <h3>Combined results</h3>
  <p>Mean ± SD over the 20 seeds; the ten-slice models concatenate the same
  features as section 6. The scores-alone row repeats section 6, whose
  probabilities were not saved.</p>
  {s7['table_combined']}
  <p class="note">Against the density arm: mean and 95% paired bootstrap
  interval over the {n_slides} slides.</p>
  {s7['table_deltas']}
  <figure>{img_tag(adds_combined_png, "Combined models against the density arm: accuracy difference and log-loss reduction per fPCA family, with bootstrap intervals.")}
  <figcaption>Each row is one fPCA family in the combined model: H0 blue, H1
  orange, filled raw, open cosine. Circles: the density arm plus every slice's four scores;
  triangles: scores 2–4 of every slice alone, without density. Left, accuracy
  difference; right, log-loss reduction; positive favours the scores. Bars are
  95% paired bootstrap intervals over slides.</figcaption></figure>
  <p>{s7['text_combined']}</p>
  <h3>Per slice</h3>
  <figure>{img_tag(adds_slice_png, "Per-slice nested and residual arms in accuracy and log-loss, four fPCA families per slice, with bootstrap intervals.")}
  <figcaption>Top: each slice's density arm plus its four scores, against the
  density arm alone. Bottom: the four scores residualised on the density arm,
  against chance (training-fold majority for accuracy, prevalence for
  log-loss). Blue H0, orange H1; filled raw, open cosine. Bars are 95% paired
  bootstrap intervals over slides; positive favours the scores.</figcaption></figure>
  <p>{s7['text_slices']}</p>
  <p>{s7['text_residual']}</p>
  <details>
    <summary>Per-slice numbers</summary>
    <p class="note">Accuracy and AUC are mean ± SD over the 20 seeds; Δ,
    log-loss reduction and above-chance columns are means with 95% paired
    bootstrap intervals over the {n_slides} slides.</p>
    {s7['table_slices']}
  </details>
  <h3>Feature scale (post-hoc)</h3>
  <p>{s7p['text_scale']}</p>
  <figure>{img_tag(scaled7_png, "The scores' share of the classifier's weight in section 7's nested fits, as fitted and standardized, and the standardized nested arms against the standardized density arm in accuracy and log-loss, per slice and for the combined models.")}
  <figcaption>Blue H0, orange H1; filled raw, open cosine; the rightmost
  group is the combined ten-slice model. Top: the scores' share of Σ |coef| ×
  SD in each fitted fold, median and interquartile range over the 100 fits
  (20 seeds × 5 folds), left with the features as they come (the arms above),
  right standardized in the training fold. Bottom: the standardized density
  arm plus the scores against the standardized density arm, in accuracy and
  log-loss reduction; bars are 95% paired bootstrap intervals over slides,
  positive favours the scores.</figcaption></figure>
  <p>{s7p['text_results']}</p>
  <details>
    <summary>Standardized nested arms</summary>
    <p class="note">Weight shares are medians over the 100 fitted folds.
    Accuracy is mean ± SD over the 20 seeds; Δ and log-loss reduction are
    against the standardized density arm, means with 95% paired bootstrap
    intervals over the {n_slides} slides.</p>
    {s7p['table']}
  </details>
  <p>{s7b['text']}</p>
  <details>
    <summary>Residual and PC arms, as fitted and standardized</summary>
    <p class="note">Accuracy is mean ± SD over the 20 seeds. The top-C column
    is the share of the 100 protocol fits (20 seeds × 5 folds) that chose
    C = 10⁴, the grid's largest. Δ accuracy and log-loss reduction are the
    standardized arm against the protocol arm; the last two columns are the
    standardized arm against its baseline (residual arms: chance, and the
    training-fold prevalence for log-loss; PC arms: the standardized density
    arm), means with 95% paired bootstrap intervals over the {n_slides}
    slides.</p>
    {s7b['table']}
  </details>

  <h2>8. Topology with density removed: the H1 death/birth ratio</h2>
  <p>Section 7 finds nothing usable in the Betti-curve scores beyond density. A
  Betti curve counts the bars alive at each radius, and both the count and the
  radii follow the cells: a denser slide holds more cycles, born and dying at
  smaller radii. This section tests one quantity built to escape both. For an
  H1 bar born at radius b and dying at radius d, the ratio π = d / b does not
  change when every distance is multiplied by the same factor, so neither a
  uniform change of density nor the assumed pixel size moves it. For cycles of
  noise, more is conjectured: in large samples of independent, identically
  distributed points in a fixed dimension, the distribution of π tends to one
  limit, whatever the distribution the points are drawn from (Bobrowski and
  Skraba's "weak universality", arXiv:2207.03926, published in <em>Scientific
  Reports</em> in 2023; supported by simulations, not proven). Cells are not
  independent points, so how far π escapes density here is measured, not
  assumed.</p>
  <p>The decision rule was written into the ledger (<code>progress.md</code>,
  Task 7) before the run; only a smoke run of slice 9 had been seen. The
  primary test is slice 0, all cells, in both normalizations: two
  comparisons. The ratio adds beyond density if a nested accuracy interval lies
  wholly above zero. Slices 6–9 are secondary and exploratory, eight
  comparisons. With {n_slides} slides, a true gain below about 0.03 can go
  undetected.</p>
  <h3>Design</h3>
  <ul>
    <li><strong>Ratio.</strong> π = √(death / birth) for every finite H1 bar of
    the slide's alpha filtration. GUDHI's alpha filtration values are squared
    radii, and alpha diagrams equal Čech diagrams, so π is the ratio of the
    Čech death and birth radii.</li>
    <li><strong>Curve.</strong> Each slide's curve is S(t), the number of its
    H1 bars with log π &gt; t, for t ≥ 0. S is the Betti curve of the bars
    [0, log π), so <code>topocyte.ph.FPCA(dim=1)</code> fits it exactly from
    the bar endpoints, without a grid. The inner product of two slides' curves
    is Σᵢⱼ min(uᵢ, vⱼ) over their log π values uᵢ and vⱼ; for every pair of
    slides in every slice, the fitted Gram matrix matches this closed form to
    {s8['gram_rel']} (relative).</li>
    <li><strong>Normalization.</strong> Raw (<code>"none"</code>) keeps the bar
    count, which grows with the number of cells. Cosine divides it out exactly:
    S = N·F, with N the number of bars and F the empirical survival function of
    log π, so S / ‖S‖ = F / ‖F‖. The curve weights every bar by its log π, so
    a long-lived cycle counts more than a short-lived one.</li>
    <li><strong>Slices.</strong> Slice 0 (all cells) and slices 6–9. No earlier
    run saved the bars, so persistent homology is recomputed as before. As a
    gate, the recomputed diagrams give back the saved H1 Betti-curve fPCA
    scores of every slice {s8['gate']}.</li>
    <li><strong>Arms.</strong> The estimator, outer folds and 20 seeds of
    section 7: the slice's density arm ({s7['per_slice']} features, identical
    to section 7's), the first four ratio scores alone, and the density arm
    plus the four ratio scores. Intervals are paired bootstrap intervals over
    the {n_slides} slides with section 7's resampling indices.</li>
  </ul>
  <h3>Results</h3>
  <p>{s8['text_primary']}</p>
  {s8['table_arms']}
  <figure>{img_tag(ratio_png, "Per-slice intervals for the H1 death/birth ratio: density plus ratio against density in accuracy and log-loss, and the ratio alone against chance, with section 7's H1 Betti results beside.")}
  <figcaption>Orange circles: the H1 death/birth ratio, filled raw, open
  cosine. Grey triangles: section 7's H1 Betti curves, the same density arm
  plus four Betti scores. Left and centre: density arm plus four scores against
  the density arm alone, in accuracy and log-loss reduction. Right: the four
  ratio scores alone against chance (training-fold majority). Bars are 95%
  paired bootstrap intervals over slides; positive favours the
  scores.</figcaption></figure>
  <p>{s8['text_secondary']}</p>
  <h3>Feature scale and the nested test (post-hoc)</h3>
  <p>{s8p['text_scale']}</p>
  <p>{s8p['text_check']}</p>
  <figure>{img_tag(scaled_png, "The scores' share of the classifier's weight in the protocol and standardized nested fits, and the standardized nested arms against the standardized density arm in accuracy and log-loss, per slice.")}
  <figcaption>Orange circles: the H1 death/birth ratio; grey triangles: section
  7's H1 Betti curves; filled raw, open cosine. Top: the four scores' share of
  Σ |coef| × SD in each fitted fold, median and interquartile range over the
  100 fits (20 seeds × 5 folds), left with the features as they come (the
  arms above), right standardized in the training fold. Bottom: the
  standardized density arm plus four scores against the standardized density
  arm, in accuracy and log-loss reduction; bars are 95% paired bootstrap
  intervals over slides, positive favours the scores.</figcaption></figure>
  <p>{s8p['text_posthoc']}</p>
  <details>
    <summary>Standardized arms per slice</summary>
    <p class="note">Weight shares are medians over the 100 fitted folds.
    Accuracy is mean ± SD over the 20 seeds; Δ and log-loss reduction are
    against the standardized density arm, means with 95% paired bootstrap
    intervals over the {n_slides} slides.</p>
    {s8p['table_posthoc']}
  </details>
  <p>{s8a['text']}</p>
  <details>
    <summary>Ratio-alone arms, as fitted and standardized</summary>
    <p class="note">Accuracy is mean ± SD over the 20 seeds. The top-C column
    is the share of the 100 protocol fits (20 seeds × 5 folds) that chose
    C = 10⁴, the grid's largest. Δ accuracy and log-loss reduction are the
    standardized arm against the protocol arm, and the last column the
    standardized arm against chance (training-fold majority), means with 95%
    paired bootstrap intervals over the {n_slides} slides.</p>
    {s8a['table']}
  </details>
  <h3>How density-free is the ratio?</h3>
  <p>{s8['text_r2']}</p>
  <p class="note">R² of each of the first four scores on the slice's density
  arm (in-sample least squares, {s7['per_slice']} features), for the ratio and,
  from section 7, for the H1 Betti curves.</p>
  {s8['table_r2']}
  <p>{s8['text_tails']}</p>
  <figure>{img_tag(ratio_quantiles_png, "Per-slide quantiles of log pi by class for five slices and four quantile levels.")}
  <figcaption>Each dot is one slide's quantile of log π over its H1 bars,
  blue healthy, orange cirrhosis, with a small horizontal jitter; black ticks
  are the class medians. Panels are the quantile levels q = 0.5, 0.9, 0.99 and
  0.999.</figcaption></figure>
  <details>
    <summary>Bar counts and log π quantiles per slice</summary>
    <p class="note">Class medians over slides.</p>
    {s8['table_bars']}
  </details>

  <h2>9. Verdict</h2>
  <p>{verdict_text} {combined_text} {s56['headline']}</p>
  <p>{s7['headline']} {s7p['headline']} {s7b['headline']}</p>
  <p>{s8['headline']} {s8p['headline']}</p>

  <h2>Reproduction</h2>
  <details>
    <summary>Files and commands</summary>
    <ul>
      <li><code>run_slices.py density</code> — writes <code>results/density.npz</code>.</li>
      <li><code>run_slices.py slice K</code> — writes <code>results/slice_K.npz</code>,
      <code>results/slice_K.json</code> and <code>figures/slice_K.png</code>.</li>
      <li><code>verify_results.py</code> — asserts the measured numbers, the
      slice-0 tutorial match, the combined feature blocks, the density-arm
      blocks and slide size, the consistency of every section-7 arm's
      probabilities, accuracy, log-loss and AUC, the chance baselines, one
      independently recomputed arm, and that the score-only arms reproduce the
      superseded run; for section 8, the recomputed diagrams against the saved
      scores, the density arm against section 7's, the Gram matrix against the
      closed form, two independently recomputed arms and one recomputed
      interval; for the post-hoc checks, the unscaled refits against the saved
      arms, the standardized arms that two refits share against each other,
      six standardized arms against scikit-learn's pipelines (one through the
      in-fold residualisation), one unscaled arm against
      <code>fpca_classify</code>'s call, five folds' chosen C and two of their
      weight shares, and fourteen recomputed intervals.</li>
      <li><code>combined_classify.py</code> — fits the all-slices models and
      their density-only control.</li>
      <li><code>density_adds.py</code> — the density arm with slide size, slide
      size alone, and the nested, per-slice residual and PC-breakdown arms, in
      accuracy, log-loss and AUC; writes <code>results/density_adds.npz</code>.
      The first run of 2026-11-16 and its report are kept in
      <code>superseded/</code>.</li>
      <li><code>ratio_slices.py</code> — recomputes the H1 diagrams of slices
      0 and 6–9, checks them against the saved Betti-curve scores, fits the
      fPCA of the death/birth-ratio curves and checks its Gram matrix against
      the closed form, and runs the density, ratio-alone and nested arms;
      writes <code>results/ratio_slices.npz</code>.</li>
      <li><code>scaled_nested.py</code> — section 8's post-hoc check: refits
      the nested arms of slices 0 and 6–9 (ratio and H1 Betti) to record each
      fold's C and the scores' weight share, checks them against the saved
      arms, and repeats them and the density arm with features standardized
      in the training fold; reads only saved results and writes
      <code>results/scaled_nested.npz</code>.</li>
      <li><code>scaled_section7.py</code> — the same check for every nested
      arm of section 7 (four families at slices 0–9 and the four combined
      models): checks the unscaled refits against the saved arms and the
      standardized ones against <code>scaled_nested.py</code>'s where both
      refit the same arm; writes <code>results/scaled_section7.npz</code>.</li>
      <li><code>scaled_ratio_alone.py</code> — the same check for section 8's
      ten ratio-alone arms: records each fold's chosen C, checks the unscaled
      refits against the saved arms and refits them standardized; writes
      <code>results/scaled_ratio_alone.npz</code>.</li>
      <li><code>scaled_single_block.py</code> — the same check for section 7's
      residual arms (standardizing the residuals after the in-fold
      residualisation) and its combined PC1 and PCs 2–4 arms; writes
      <code>results/scaled_single_block.npz</code>.</li>
      <li><code>scaled_ph_only.py</code> — the same check for the arms behind
      the verdict: the four-score arms and density-only controls of every
      slice and the combined models and control; checks the unscaled refits
      against the per-seed accuracies and confusion matrices that
      <code>fpca_classify</code> and <code>combined_classify.py</code> stored;
      writes <code>results/scaled_ph_only.npz</code>.</li>
      <li><code>build_report.py</code> — this page; every number and figure is
      read from the saved results.</li>
    </ul>
    <p class="note">The zarr store was read with <code>mode="r"</code> under a
    sandbox profile that denies writes below <code>~/data</code>; topocyte
    itself is unchanged. All Betti curves and fPCA components are exact from
    bar endpoints, the kNN search is exact and the quantiles use every cell.</p>
  </details>
</main>
</body>
</html>
"""
    OUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
