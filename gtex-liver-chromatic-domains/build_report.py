#!/usr/bin/env python
"""Build the self-contained HTML report of the chromatic-domains pilot from its saved results.

Run from this folder with the study venv and ``PYTHONPATH=<topocyte checkout at 18e3aa5>``
(``common.check_topocyte``). It reads ``results/`` (``pilot.csv``, the bars jsons,
``features.npz``, ``domains.json``, ``preflight.json``, ``verify.json``), the bars files for the
mean residual curves, and ``logs/features.log`` for the features run's time and memory, and
writes ``report.html`` with every figure embedded as a 128-colour PNG (the sibling reports'
palette). It writes nothing else. About 90 s, peak RSS about 7 GB (the mean residual curves).

No number in the prose is typed by hand: each is formatted from those files, and every sentence
that states a direction, a ranking or an outcome is asserted against the numbers it states. Three
kinds of value have no results file and are constants below, named as such: the provenance of
the saved results (commits and md5 sums, checked against the files and the git history), one
measurement recorded in ``progress.md`` (the serial time per centre behind the speed-up of 8
processes), and the design's estimates that the run is compared with.

Three things are computed here, all descriptive (the controller's rulings for Task 8):
- the stratum's mean residual curve of each block (T8-b): the exact mean, every bar kept with
  weight 1/n, of the all-slides fit sample's scaled residual curves as ``features.py`` builds them,
  drawn as steps at its own breakpoints, never on a grid;
- the mechanism of the level-1 outcome (T8-d): each half's 2-means split direction, the minority's
  share and composition, and the crossings behind prediction strength at k = 2;
- a post-hoc check outside the rule (T8-e): prediction strength at k = 2 on three random half/half
  splits of all centres, at level 1 and in the all-slides level-2 basis. It cannot change the verdict.

Wording: the level-1 compartments are composition strata, never modes (Simon, 2026-11-21).
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import html
import io
import json
import re
import subprocess

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as c  # noqa: E402
import curves  # noqa: E402
import domains  # noqa: E402
import features  # noqa: E402

OUT = c.HERE / "report.html"
N = 30 * c.N_CENTRES
INK, INK2, MUTED, RULE, SURFACE, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
ACCENT = "#2a78d6"
HALF = {"A": "#2a78d6", "B": "#eb6834"}  # the dataviz palette's first two slots, validated as a pair
L1_NAMES = ("log10 cells", *c.CLASSES)
SPLIT_MARK = 0x5B17  # T8-e: SeedSequence([SEED, 0x5B17, i, 1]), i = 1..3
# Provenance of the saved results (progress.md, Tasks 5-7, and the controller's ruling); check_provenance checks it.
BARS_COMMIT, FEATURES_COMMIT, DOMAINS_COMMIT = "e8db8a4", "da7e578", "381d0ce"
FEATURES_BLOB, FEATURES_MD5 = "f7ae634", "48ef03203e58aca9de7cfebfae1200d9"
DOMAINS_MD5 = "964d508240c2c32d9883e5f68e07dd34"
# Measured once, without a results file (progress.md, Task 5): a centre of GTEX-YJ89-1426 took 0.50 s serially
# (8 centres), and phimaker 0.5.0 ran up to 10 threads per six-pack call although the port passes num_threads=1.
SERIAL_SECONDS_PER_CENTRE, PHIMAKER_THREADS = 0.50, 10
# The design's estimates (design.md, Compute): about 12 h serial, about 2 h on 8 processes.
DESIGN_SERIAL_HOURS, DESIGN_HOURS = 12, 2

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
  figcaption { color: var(--ink2); font-size: 0.85rem; margin-top: 8px; }
  .table-wrap { overflow-x: auto; margin: 16px 0 6px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.82rem; font-variant-numeric: tabular-nums;
          background: var(--surface); }
  th, td { border-bottom: 1px solid var(--rule); padding: 7px 8px; text-align: right; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .left2 th:nth-child(2), .left2 td:nth-child(2), .left2 th:nth-child(3), .left2 td:nth-child(3) {
    text-align: left; white-space: normal; }
  .label2 th:nth-child(2), .label2 td:nth-child(2) { text-align: left; }
  .wrap1 td:first-child { white-space: normal; min-width: 16rem; }
  th { color: var(--ink2); font-weight: 600; vertical-align: bottom; white-space: normal; }
  .unit { color: var(--muted); font-weight: 400; font-size: 0.72rem; }
  .note { color: var(--ink2); font-size: 0.85rem; }
  code { font-size: 0.86em; background: var(--rule); padding: 1px 4px; border-radius: 3px; }
  details { margin: 14px 0; } summary { cursor: pointer; color: var(--ink2); }
  ul { margin: 8px 0 8px 22px; padding: 0; } li { margin: 6px 0; }
"""


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


def arrow(direction: str) -> str:
    return direction.replace("->", "→")


def radius_um(alpha2: float) -> float:
    """The radius in µm of a filtration value α² in px² (0.4942 µm/px)."""
    return float(np.sqrt(alpha2)) * c.PX_UM


# ---------------------------------------------------------------- inputs


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(c.HERE), *args], capture_output=True, text=True, check=True).stdout.strip()


def check_provenance() -> dict:
    """The code that wrote the saved results is the code here (md5 sums, git), and the features run's record."""

    def md5(name: str) -> str:
        return hashlib.md5((c.HERE / name).read_bytes()).hexdigest()

    blob = git("rev-parse", f"{FEATURES_COMMIT}:gtex-liver-chromatic-domains/features.py")
    assert blob.startswith(FEATURES_BLOB) and md5("features.py") == FEATURES_MD5, "features.py changed"
    assert md5("domains.py") == DOMAINS_MD5, "domains.py changed since it wrote domains.json"
    assert git("cat-file", "-t", BARS_COMMIT) == git("cat-file", "-t", DOMAINS_COMMIT) == "commit"
    log = (c.LOGS / "features.log").read_text().splitlines()
    assert f"features.py md5 {FEATURES_MD5}" in log[0], "logs/features.log is not the run of this features.py"
    wrote = re.search(r"in (\d+) s; peak RSS ([\d.]+) GB", next(line for line in log if line.startswith("wrote")))
    user = re.search(r"([\d.]+) user", next(line for line in log if " real " in line)).group(1)
    return {
        "seconds": int(wrote.group(1)),
        "peak_rss_gb": float(wrote.group(2)),
        "cpu_seconds": float(user),
        "started": re.search(r"started (\S+ \S+),", log[0]).group(1),
    }


def load() -> dict:
    pilot = pd.read_csv(c.RESULTS / "pilot.csv")
    metas = [json.loads((c.RESULTS / "bars" / f"{s}.json").read_text()) for s in pilot["slide_id"]]
    with np.load(c.RESULTS / "features.npz", allow_pickle=False) as z:
        F = {key: z[key] for key in z.files if not key.startswith("blk_obs_")}
    return {
        "pilot": pilot,
        "metas": metas,
        "F": F,
        "S": json.loads((c.RESULTS / "domains.json").read_text()),
        "PF": json.loads((c.RESULTS / "preflight.json").read_text()),
        "V": json.loads((c.RESULTS / "verify.json").read_text()),
        "run": check_provenance(),
    }


# ---------------------------------------------------------------- the descriptive analyses


def mean_residual_curves(pilot: pd.DataFrame, s0_all: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """T8-b: per block, the exact mean of the all-slides fit sample's 12,000 scaled residual curves.

    Each curve as ``features.py`` builds it: labelling 0 minus the mean of labellings 1..19,
    divided by sqrt(s² + s0_all²). Every bar is kept with weight 1/n (``curves.combine``), and the
    mean's step table (its breakpoints and the value from each to the next) comes from
    ``curves._steps``, with compensated sums. No curve is evaluated on a grid.
    """
    out = {}
    for block in range(len(c.BLOCKS)):
        pieces = []
        for _, slide in pilot.iterrows():
            with np.load(c.RESULTS / "bars" / f"{slide['slide_id']}.npz", allow_pickle=False) as z:
                labs = features.labellings(z["bars"], z["offsets"], block)
            fit = [features.take(lab, features.fit_sample(int(slide["slide_index"]))) for lab in labs]
            residual = features.residual(fit, 0)
            pieces.append(features.scaled(residual, features.spread(fit, 0), float(s0_all[block]) ** 2))
        joined = features.concat(pieces)
        one = curves.CurveSet(joined.b, joined.d, joined.w, np.array([0, joined.b.size]))
        mean = curves.combine([one], [1.0 / len(joined)])
        x, v, count = curves._steps(mean.b, mean.d, mean.w, mean.offsets)
        out[block] = (x[: count[0]].copy(), v[: count[0]].copy())
        print(f"  mean residual curve, block {block}: {len(joined):,} curves, {count[0]:,} breakpoints", flush=True)
    return out


def split_mechanism(L1: np.ndarray, half: np.ndarray, ps_curve: dict) -> dict:
    """T8-d: each prediction-strength direction at k = 2, both halves in the training half's z-coordinates.

    The training half's 2-means and the test half's own 2-means (``domains.kmeans_fit``, as
    ``prediction_strength`` fits them); the |cosine| of their centroid differences; each one's
    minority share and mean level-1 features; where each one's boundary (the bisector of its two
    centroids) crosses the training half's split direction, in the training half's z-units; and
    the crossings, the test clusters' centres that the training centroids assign to the other side.
    Prediction strength recomputed from the crossings must equal the saved PS(2).
    """
    out = {}
    for train, test in (("A", "B"), ("B", "A")):
        Xt, Xs = L1[half == train], L1[half == test]
        zt, zs = domains.zscore(Xt, Xt), domains.zscore(Xs, Xt)
        mt, ms = domains.kmeans_fit(zt, 2), domains.kmeans_fit(zs, 2)
        own, assigned = ms.labels_, mt.predict(zs)
        minority_test, minority_train = int(np.argmin(np.bincount(own))), int(np.argmin(np.bincount(mt.labels_)))
        dt_ = mt.cluster_centers_[minority_train] - mt.cluster_centers_[1 - minority_train]
        ds_ = ms.cluster_centers_[minority_test] - ms.cluster_centers_[1 - minority_test]
        u = dt_ / np.linalg.norm(dt_)
        shares = []
        for label in (minority_test, 1 - minority_test):
            members = own == label
            other_side = 1 - minority_train if label == minority_test else minority_train
            n, k = int(members.sum()), int(np.count_nonzero(assigned[members] == other_side))
            shares.append({"n": n, "crossing": k, "share": ((n - k) * (n - k - 1) + k * (k - 1)) / (n * (n - 1))})
        ps = min(s["share"] for s in shares)
        assert abs(ps - ps_curve[f"{train}->{test}"][0]) <= 1e-12, "PS(2) from the crossings differs from domains.json"
        assert abs(ps - domains.ps_from_labels(own, assigned)) <= 1e-15
        out[f"{train}->{test}"] = {
            "cos": float(abs(dt_ @ ds_) / (np.linalg.norm(dt_) * np.linalg.norm(ds_))),
            "train_minority_share": float(np.mean(mt.labels_ == minority_train)),
            "test_minority_share": float(np.mean(own == minority_test)),
            "boundary_offset": float((mt.cluster_centers_.mean(axis=0) - ms.cluster_centers_.mean(axis=0)) @ u),
            "minority": shares[0],
            "majority": shares[1],
            "ps": ps,
            "profiles": {
                f"half {test}'s own minority": Xs[own == minority_test].mean(axis=0),
                f"half {test}'s own majority": Xs[own != minority_test].mean(axis=0),
                f"half {train}'s minority": Xt[mt.labels_ == minority_train].mean(axis=0),
                f"half {train}'s majority": Xt[mt.labels_ != minority_train].mean(axis=0),
            },
        }
    return out


def random_splits(L1: np.ndarray, L2_all: np.ndarray, half: np.ndarray) -> dict:
    """T8-e, post hoc: PS(2) both ways on three random half/half splits of all centres; the slide halves in the
    all-slides level-2 basis beside them."""
    rows = []
    for i in (1, 2, 3):
        perm = np.random.default_rng(np.random.SeedSequence([c.SEED, SPLIT_MARK, i, 1])).permutation(N)
        X = np.zeros(N, bool)
        X[perm[: N // 2]] = True
        _, _, directions = domains.level1_inputs(L1[X], L1[~X])
        level1 = [domains.prediction_strength(train, test, 2) for train, test in directions]
        level2 = [
            domains.prediction_strength(L2_all[X], L2_all[~X], 2),
            domains.prediction_strength(L2_all[~X], L2_all[X], 2),
        ]
        rows.append(
            {
                "split": i,
                "level1": level1,
                "level2": level2,
                "slides": int(np.unique(np.flatnonzero(X) // c.N_CENTRES).size),
            }
        )
    A, B = half == "A", half == "B"
    halves = [
        domains.prediction_strength(L2_all[A], L2_all[B], 2),
        domains.prediction_strength(L2_all[B], L2_all[A], 2),
    ]
    return {"rows": rows, "halves_level2_all": halves}


# ---------------------------------------------------------------- figures


def fig_rule(choice: dict, title: str, null: dict | None = None) -> bytes:
    """Both halves' gap curves with ±s_k, and the prediction strength of both directions at every k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9), constrained_layout=True)
    for h in "AB":
        gap, s = np.array(choice[f"gap_{h}"]["gap"]), np.array(choice[f"gap_{h}"]["s"])
        k = np.arange(1, gap.size + 1)
        label = f"half {h}: stops at k = {choice[f'gap_{h}']['k']}"
        ax1.errorbar(
            k,
            gap,
            yerr=s,
            color=HALF[h],
            lw=2,
            marker="o",
            ms=5,
            mec=SURFACE,
            mew=0.8,
            capsize=2.5,
            label=label,
            zorder=3,
        )
    if null is not None:
        for h, style in (("A", "--"), ("B", ":")):
            gap = np.array(null[f"gap_{h}"]["gap"])
            ax1.plot(
                np.arange(1, gap.size + 1),
                gap,
                color=MUTED,
                ls=style,
                lw=1.4,
                marker="s",
                ms=3.5,
                zorder=2,
                label=f"composition null, half {h}: stops at k = {null[f'gap_{h}']['k']}",
            )
    ax1.axhline(0, color=CONTEXT, lw=0.8, zorder=1)
    ax1.set_xticks(k)
    ax1.set_xlabel("k (clusters)")
    ax1.set_ylabel("Gap(k), ± s_k")
    ax1.grid(alpha=0.6)
    ax1.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax1.set_title("gap statistic against a matched Gaussian (B = 20)", loc="left", color=INK, fontsize=9)
    for direction, h in (("A->B", "A"), ("B->A", "B")):
        ps = choice["ps_curve"][direction]
        ax2.plot(
            np.arange(2, len(ps) + 2),
            ps,
            color=HALF[h],
            lw=2,
            marker="o",
            ms=5,
            mec=SURFACE,
            mew=0.8,
            zorder=3,
            label=f"trained on half {h}, tested on half {direction[-1]}",  # Helvetica Neue has no arrow glyph
        )
    if null is not None:
        ax2.scatter(
            [null["k"]] * 2,
            null["ps"],
            color=MUTED,
            marker="s",
            s=22,
            zorder=2,
            label=f"composition null at its k = {null['k']}",
        )
    ax2.axhline(domains.PS_MIN, color=INK2, lw=1, ls=":", zorder=1)
    ax2.annotate(
        f"{domains.PS_MIN}: needed in both directions",
        (len(ps) + 1, domains.PS_MIN),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=7.8,
        color=INK2,
    )
    ax2.set_xticks(np.arange(2, len(ps) + 2))
    ax2.set_ylim(0, 1.04)
    ax2.set_xlabel("k (clusters)")
    ax2.set_ylabel("prediction strength")
    ax2.grid(alpha=0.6)
    ax2.legend(frameon=False, fontsize=7.8, loc="lower left")
    ax2.set_title("prediction strength between the halves", loc="left", color=INK, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    return png_bytes(fig)


def fig_l1_ecdf(L1: np.ndarray, half: np.ndarray) -> bytes:
    """The empirical distribution function of each level-1 feature per half: exact steps at the data."""
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0), constrained_layout=True, sharey=True)
    for j, (ax, name) in enumerate(zip(axes, L1_NAMES, strict=True)):
        for h in "AB":
            x = np.sort(L1[half == h, j])
            ax.plot(
                x, np.arange(1, x.size + 1) / x.size, drawstyle="steps-post", color=HALF[h], lw=1.6, label=f"half {h}"
            )
        ax.grid(alpha=0.6)
        ax.set_xlabel(name if j == 0 else f"{name} fraction")
    axes[0].set_ylabel("share of the half's centres ≤ x")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Level-1 features of each centre's ball, per half (empirical distribution functions)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    return png_bytes(fig)


def fig_mean_residuals(mean_curves: dict) -> bytes:
    """The stratum's mean scaled residual curve of each block, drawn as exact steps on topocyte's logarithmic
    filtration axis (plain 1-2-5 labels)."""
    from topocyte.pl.diagram import _log_x_axis

    fig, axes = plt.subplots(4, 4, figsize=(13.5, 10.6), constrained_layout=True)
    for block, (t, module, p) in enumerate(c.BLOCKS):
        ax = axes[t, c.MODULES.index(module) * 2 + p]
        x, v = mean_curves[block]
        ax.axhline(0, color=INK2, lw=0.7, zorder=1)
        ax.plot(x, v, drawstyle="steps-post", color=ACCENT, lw=1.0, zorder=2)
        _log_x_axis(ax)
        ax.set_xlim(x[x > 0].min(), x.max())
        ax.grid(alpha=0.5)
        ax.tick_params(labelsize=7)
        if t == 0:
            ax.set_title(f"{module} H{p}", loc="left", color=INK, fontsize=9.5)
        if t == 3:
            ax.set_xlabel("filtration value α² (px²)")
        if module == "domain" and p == 0:
            ax.set_ylabel(f"{c.CLASSES[t]}\nmean scaled residual", fontsize=8)
    fig.suptitle(
        "Mean scaled residual curve of each block over the 12,000 fit-sample centres (observed minus the "
        "mean of 19 random labellings, over sqrt(s² + s0²))",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    return png_bytes(fig)


# ---------------------------------------------------------------- the rule's curves


def rule_table(choice: dict) -> str:
    """The table view of a rule application's curves."""
    rows = []
    for k in range(1, len(choice["gap_A"]["gap"]) + 1):
        row = [str(k)]
        for h in "AB":
            row += [num(choice[f"gap_{h}"]["gap"][k - 1], 4), num(choice[f"gap_{h}"]["s"][k - 1], 4)]
        row += [num(choice["ps_curve"][d][k - 2]) if k >= 2 else "—" for d in ("A->B", "B->A")]
        rows.append(row)
    return table(["k", "Gap, A", "s, A", "Gap, B", "s, B", "PS A→B", "PS B→A"], rows)


def margin(choice: dict, h: str) -> float:
    """Gap(1) − (Gap(2) − s_2): positive where the half's gap stops at k = 1."""
    g = choice[f"gap_{h}"]
    return g["gap"][0] - (g["gap"][1] - g["s"][1])


def rising(choice: dict, h: str) -> bool:
    """Whether the half's gap never stops: Gap(k) < Gap(k+1) − s_(k+1) at every k < k_max."""
    g, s = choice[f"gap_{h}"]["gap"], choice[f"gap_{h}"]["s"]
    return all(g[k - 1] < g[k] - s[k] for k in range(1, len(g)))


def best_ps(choice: dict) -> tuple[int, float, float]:
    """The k whose weaker direction has the largest prediction strength, and both directions there."""
    curve = choice["ps_curve"]
    k = 2 + int(np.argmax(np.minimum(curve["A->B"], curve["B->A"])))
    return k, curve["A->B"][k - 2], curve["B->A"][k - 2]


# ---------------------------------------------------------------- main


def main() -> None:
    c.check_topocyte()
    d = load()
    pilot, metas, F, S, PF, V, run = d["pilot"], d["metas"], d["F"], d["S"], d["PF"], d["V"], d["run"]
    half, L1 = F["half"], F["L1"]
    A = half == "A"
    assert V["passed"] and V["n_checks"] == len(V["checks"]) and all(check["passed"] for check in V["checks"])
    level1, comps = S["level1"], S["compartments"]
    assert len(comps) == 1 and level1["k"] == 1 and S["control_k"] == [1] and S["control_passed"]
    stratum = comps[0]
    control, level2 = stratum["control"], stratum["level2"]
    null = PF["descriptive"][0]
    assert null["case"].startswith("composition null") and PF["passed"]
    rule = S["settings"]["rule"]
    k_max1, k_max2, min_gain = rule["k_max_strata"], rule["k_max_subdomains"], rule["min_gain"]
    # The rule's outcome, asserted against its own curves.
    ps_at = {
        name: [ch["ps_curve"][x][len(ch["gap_A"]["gap"]) - 2] for x in ("A->B", "B->A")]
        for name, ch in (("level1", level1), ("level2", level2))
    }
    for name, ch, k_max in (("level1", level1, k_max1), ("level2", level2, k_max2)):
        assert ch["k"] == 1 and ch["k_A"] == ch["k_B"] == k_max and rising(ch, "A") and rising(ch, "B")
        assert ch["ps"] == ps_at[name] and all(p < domains.PS_MIN for p in ch["ps"])
        # No k reaches 0.8 in both directions, so prediction strength at another k would not have changed k.
        assert max(np.minimum(ch["ps_curve"]["A->B"], ch["ps_curve"]["B->A"])) < domains.PS_MIN
        assert ch["ps_curve"]["A->B"][0] < ch["ps_curve"]["B->A"][0]  # A->B is the weaker direction at k = 2
    assert (
        control["k"] == control["k_A"] == control["k_B"] == 1 and margin(control, "A") > 0 and margin(control, "B") > 0
    )
    assert control["gap_B"]["largest_gain"] == control["gap_B"]["gap"][-1] - control["gap_B"]["gap"][0]
    assert S["verdict"] == {"verdict": "does not add", "reason": "no compartment with subdomains"}
    assert stratum["guard"] is None and stratum["contiguity"] is None and level2["ari"] is None
    k1, ps1_ab, ps1_ba = best_ps(level1)
    k2, ps2_ab, ps2_ba = best_ps(level2)
    assert k1 == k2 == 2
    assert level2["ps_curve"]["B->A"][1] >= domains.PS_MIN > level2["ps_curve"]["A->B"][1]
    gain = {
        (name, h): ch[f"gap_{h}"]["gap"][-1] - ch[f"gap_{h}"]["gap"][0]
        for name, ch in (("level1", level1), ("level2", level2))
        for h in "AB"
    }
    for (name, h), value in gain.items():
        assert abs(value - (level1 if name == "level1" else level2)[f"gap_{h}"]["largest_gain"]) < 1e-12
    assert gain["level1", "A"] > gain["level1", "B"] and gain["level2", "A"] > gain["level2", "B"]
    assert null["k"] == null["k_A"] == null["k_B"] and all(p >= domains.PS_MIN for p in null["ps"])
    assert null["k"] < k_max1 and not rising(null, "A")

    # ---------------- slides, balls and compute
    cells = np.rint(10 ** L1[:, 0]).astype(np.int64)
    assert np.allclose(np.log10(cells), L1[:, 0], rtol=0, atol=1e-12)
    bars_total, cod_total = sum(m["bars"] for m in metas), sum(m["codomain_bars"] for m in metas)
    bars_wall, bars_bytes = sum(m["wall_seconds"] for m in metas), sum(m["npz_bytes"] for m in metas)
    workers = metas[0]["workers"]
    yj = next(m for m in metas if m["slide_id"] == "GTEX-YJ89-1426")
    speedup = workers * SERIAL_SECONDS_PER_CENTRE / yj["seconds_per_centre"]["median"]
    bars_started, bars_finished = min(m["started"] for m in metas), max(m["finished"] for m in metas)
    assert all(m["workers"] == workers for m in metas)
    assert speedup < DESIGN_SERIAL_HOURS / DESIGN_HOURS and bars_wall / 3600 > DESIGN_HOURS
    slide_rows = []
    for (_, slide), m in zip(pilot.iterrows(), metas, strict=True):
        assert m["slide_id"] == slide["slide_id"] and m["slide_index"] == slide["slide_index"]
        cells_range, slowest = f"{m['ball_cells']['min']}–{m['ball_cells']['max']}", m["seconds_per_centre"]["max"]
        balls = f"{m['ball_cells']['median']:.0f} <span class='unit'>({cells_range})</span>"
        timing = f"{m['seconds_per_centre']['median']:.2f} <span class='unit'>(max {slowest:.2f})</span>"
        slide_rows.append(
            [
                str(slide["slide_index"]),
                slide["slide_id"],
                slide["condition"],
                slide["half"],
                f"{m['slide_cells']:,}",
                balls,
                timing,
                f"{m['wall_seconds']:.0f}",
                f"{m['bars'] / 1e6:.1f}M",
            ]
        )
    slides_table = table(
        [
            "index",
            "slide",
            "condition",
            "half",
            "cells",
            "ball cells, median (range)",
            "s per centre, median",
            "wall, s",
            "bars",
        ],
        slide_rows,
    )
    halves_per_condition = pilot.groupby("condition", sort=False)["half"].agg(
        lambda h: f"{(h == 'A').sum()} / {(h == 'B').sum()}"
    )

    # ---------------- level 1: the stratum, the halves, the mechanism and the random splits
    means = {f: np.array(level1["zscore"][f]["mean"]) for f in ("A", "B", "all")}
    sds = {f: np.array(level1["zscore"][f]["sd"]) for f in ("A", "B", "all")}
    assert np.allclose(means["all"], L1.mean(axis=0), atol=1e-12) and np.allclose(
        means["A"], L1[A].mean(axis=0), atol=1e-12
    )
    assert np.allclose(list(stratum["mean_L1"].values()), means["all"], atol=1e-12)
    std_diff = (means["A"] - means["B"]) / np.sqrt((sds["A"] ** 2 + sds["B"] ** 2) / 2)
    feature_names = [name if j == 0 else f"{name} fraction" for j, name in enumerate(L1_NAMES)]
    profile_rows = [
        [feature_names[j]]
        + [f"{means[f][j]:.3f} <span class='unit'>({sds[f][j]:.3f})</span>" for f in ("all", "A", "B")]
        + [num(means["A"][j] - means["B"][j]), num(std_diff[j], 2)]
        for j in range(len(L1_NAMES))
    ]
    profile_table = table(
        ["feature", "stratum, mean (SD)", "half A", "half B", "A − B", "A − B in pooled SDs"], profile_rows
    )
    biggest = int(np.argmax(np.abs(std_diff)))
    assert means["A"][0] > means["B"][0] and means["A"][4] > means["B"][4] and means["A"][1] < means["B"][1]
    mech = split_mechanism(L1, half, level1["ps_curve"])
    ab, ba = mech["A->B"], mech["B->A"]
    assert abs(ab["ps"] - ps1_ab) < 1e-12 and abs(ba["ps"] - ps1_ba) < 1e-12
    for m in mech.values():  # every minority: fewer epithelial, more stromal, immune and other cells, and more cells
        for kind in ("own", ""):
            keys = [key for key in m["profiles"] if ("own" in key) == (kind == "own")]
            minority, majority = (m["profiles"][key] for key in sorted(keys, key=lambda key: "majority" in key))
            assert minority[1] < majority[1] and all(minority[j] > majority[j] for j in (0, 2, 3, 4))
    # A's boundary lies further toward the minority's end than B's, in both halves' coordinates.
    assert ab["boundary_offset"] > 0 > ba["boundary_offset"]
    assert (
        ab["minority"]["crossing"] > 0 == ab["majority"]["crossing"]
        and ba["majority"]["crossing"] > 0 == ba["minority"]["crossing"]
    )
    mech_rows = []
    for direction, m in mech.items():
        mech_rows.append(
            [
                arrow(direction),
                f"{m['cos']:.3f}",
                f"{100 * m['train_minority_share']:.1f}%",
                f"{100 * m['test_minority_share']:.1f}% <span class='unit'>({m['minority']['n']:,})</span>",
                num(m["boundary_offset"], 2),
                f"{m['minority']['crossing']:,} of {m['minority']['n']:,}",
                f"{m['majority']['crossing']:,} of {m['majority']['n']:,}",
                f"{m['minority']['share']:.3f} / {m['majority']['share']:.3f}",
                f"{m['ps']:.3f}",
            ]
        )
    mech_table = table(
        [
            "direction",
            "|cos| of the split directions",
            "training half's minority",
            "test half's own minority",
            "training boundary − test boundary (z-units)",
            "test minority on the training majority's side",
            "test majority on the training minority's side",
            "pairs kept together, minority / majority",
            "PS(2)",
        ],
        mech_rows,
    )
    profile_split = table(
        ["cluster (direction)"] + [n + (" (mean)" if j == 0 else "") for j, n in enumerate(L1_NAMES)],
        [
            [f"{key} ({arrow(direction)})"] + [f"{value:.3f}" for value in profile]
            for direction, m in mech.items()
            for key, profile in m["profiles"].items()
        ],
    )
    splits = random_splits(L1, F["L2_obs_all"], half)
    low = min(min(r["level1"] + r["level2"]) for r in splits["rows"])
    assert all(r["slides"] == 30 for r in splits["rows"]) and low >= domains.PS_MIN
    assert max(splits["halves_level2_all"][0], ps1_ab, ps2_ab) < domains.PS_MIN  # the slide halves fail A->B
    split_rows = [
        [
            f"random split {r['split']}",
            f"{r['level1'][0]:.3f} / {r['level1'][1]:.3f}",
            f"{r['level2'][0]:.3f} / {r['level2'][1]:.3f}",
        ]
        for r in splits["rows"]
    ]
    split_rows += [
        ["slide halves A→B / B→A (the rule's bases)", f"{ps1_ab:.3f} / {ps1_ba:.3f}", f"{ps2_ab:.3f} / {ps2_ba:.3f}"],
        [
            "slide halves A→B / B→A, level 2 in the all-slides basis",
            "—",
            f"{splits['halves_level2_all'][0]:.3f} / {splits['halves_level2_all'][1]:.3f}",
        ],
    ]
    split_table = table(
        ["halves", "level 1, PS(2) one way / the other", "level 2, PS(2) one way / the other"], split_rows
    )

    # ---------------- level 2: the mean residual curves; the control
    mean_curves = mean_residual_curves(pilot, F["s0_all"])
    summary_of = {}
    for block, (x, v) in mean_curves.items():
        summary_of[block] = {
            "integral": float(np.sum(v[:-1] * np.diff(x))),
            "max": float(v.max()),
            "at_max": float(x[np.argmax(v)]),
            "min": float(v.min()),
            "at_min": float(x[np.argmin(v)]),
        }
    h0_rest = [b for b, (t, _, p) in enumerate(c.BLOCKS) if t > 0 and p == 0]
    image_h1 = [b for b, (_, module, p) in enumerate(c.BLOCKS) if module == "image" and p == 1]
    domain_h1 = [b for b, (_, module, p) in enumerate(c.BLOCKS) if module == "domain" and p == 1]
    assert all(summary_of[b]["integral"] < 0 and -summary_of[b]["min"] > summary_of[b]["max"] for b in h0_rest)
    assert all(summary_of[b]["integral"] > 0 and summary_of[b]["max"] > -summary_of[b]["min"] for b in image_h1)
    assert all(
        summary_of[b]["max"] > 0 > summary_of[b]["min"] and summary_of[b]["at_max"] < summary_of[b]["at_min"]
        for b in domain_h1
    )

    def span(blocks: list[int], key: str) -> str:
        """The range of a curve feature's α² over ``blocks``, in px², and the radii in µm."""
        low, high = min(summary_of[b][key] for b in blocks), max(summary_of[b][key] for b in blocks)
        return f"{low:,.0f} to {high:,.0f} px², radii {radius_um(low):.1f} to {radius_um(high):.1f} µm"

    curve_rows = [
        [
            f"{b}",
            f"{c.CLASSES[t]} {module} H{p}",
            f"{len(mean_curves[b][0]):,}",
            num(summary_of[b]["integral"], 2),
            f"{summary_of[b]['max']:.2e} <span class='unit'>at {summary_of[b]['at_max']:,.0f}</span>",
            f"{summary_of[b]['min']:.2e} <span class='unit'>at {summary_of[b]['at_min']:,.0f}</span>",
        ]
        for b, (t, module, p) in enumerate(c.BLOCKS)
    ]
    curve_table = table(
        ["block", "class, module, degree", "breakpoints", "integral", "maximum (at α², px²)", "minimum (at α², px²)"],
        curve_rows,
        "label2",
    )
    breakpoints = [len(x) for x, _ in mean_curves.values()]
    ctl_sd, obs_sd = float(np.mean(F["L2_ctl_all"].std(axis=0))), float(np.mean(F["L2_obs_all"].std(axis=0)))
    assert ctl_sd < obs_sd

    # ---------------- methods: per-block table, MFA, verification
    fits = features.FITS
    block_rows = []
    for b, (t, module, p) in enumerate(c.BLOCKS):
        block_rows.append(
            [
                str(b),
                f"{c.CLASSES[t]} {module} H{p}",
                " / ".join(f"{F[f's0_{f}'][b]:.4g}" for f in fits),
                " / ".join(f"{F[f'evr_{f}'][b].sum():.3f}" for f in fits),
                f"{F['evr_all'][b][0]:.3f}",
                f"{F['zero_spread_nonzero'][b]:,}",
                f"{F['gate_gram'][b]:.1e}",
                f"{max(F[f'gate_scores_{f}'][b] for f in fits):.1e}",
            ]
        )
    block_table = table(
        [
            "block",
            "class, module, degree",
            "s0, px: A / B / all",
            "9 components' FVE: A / B / all",
            "first component (all)",
            "zero spread, non-zero residual",
            "gate 1",
            "gate 2",
        ],
        block_rows,
        "label2",
    )
    assert all(np.all(F[f"kept_{f}"] == features.N_COMPONENTS) and not np.any(F[f"dropped_{f}"]) for f in fits)
    mfa_rows = [
        [
            f,
            f"{F[f'mfa_evr_{f}'].sum():.3f}",
            ", ".join(f"{v:.3f}" for v in F[f"mfa_evr_{f}"][:5]),
            f"{F[f'mfa_evr_{f}'][-1]:.3f}",
        ]
        for f in fits
    ]
    mfa_table = table(["fit", "first 20 components explain", "first five", "20th"], mfa_rows)
    zero_total = int(F["zero_spread_nonzero"].sum())
    zero_image_h1 = int(F["zero_spread_nonzero"][image_h1].sum())
    assert zero_image_h1 > zero_total / 2
    verify_rows = [
        [
            html.escape(ch["check"]),
            html.escape(ch["sample"]),
            html.escape(ch["tolerance"]),
            f"{ch['measured']:.0f}" if ch["tolerance"] == "descriptive" else f"{ch['measured']:.3g}",
        ]
        for ch in V["checks"]
    ]
    verify_table = table(["check (all passed)", "sample", "tolerance", "measured"], verify_rows, "left2 wrap1")
    sixpack = next(ch for ch in V["checks"] if ch["check"].startswith("six-packs: saved labelling-0"))
    n_gated = sum(ch["tolerance"] != "descriptive" for ch in V["checks"])
    halves_fit = max(ch["measured"] for ch in V["checks"] if ch["check"].startswith(("fPCA A: fit", "fPCA B: fit")))

    # ---------------- figures
    fig_level1 = figure(
        fig_rule(
            level1,
            "Level 1, composition strata: the rule's curves, with the pre-flight's composition null "
            "(cluster-free, reported only) for contrast",
            null,
        ),
        "Gap and prediction-strength curves at level 1",
        f"Left: each half's gap statistic Gap(k) with ±s<sub>k</sub>, k = 1..{k_max1}; grey, the composition null of "
        "the pre-flight, a cluster-free Dirichlet–multinomial cloud of the same size. Right: prediction strength at "
        f"k = 2..{k_max1} in both directions; grey squares, the null's at its k. Values in the table below.",
    )
    fig_ecdf = figure(
        fig_l1_ecdf(L1, half),
        "Distribution of the level-1 features per half",
        "Empirical distribution function of each level-1 feature over each half's 22,500 centres.",
    )
    fig_level2 = figure(
        fig_rule(level2, "Level 2, subdomains inside the one stratum: the rule's curves"),
        "Gap and prediction-strength curves at level 2",
        f"As at level 1, on the 20 MFA components in each half's own basis, k = 1..{k_max2}.",
    )
    fig_control = figure(
        fig_rule(
            control, "The control (labelling 1 against labellings 2..19, in the observed bases): the rule's curves"
        ),
        "Gap and prediction-strength curves of the control",
        "As at level 2. Prediction strength is drawn for completeness: with k = 1 in both halves the rule does not "
        "read it.",
    )
    fig_means = figure(
        fig_mean_residuals(mean_curves),
        "Mean scaled residual curve of each block",
        "Rows: the class t of the inclusion (class t → all cells); columns: the module and the degree. Each curve is "
        "the exact mean of the 12,000 fit-sample centres' scaled residual curves, drawn as steps at its own "
        f"breakpoints ({min(breakpoints):,} to {max(breakpoints):,} per block) on a logarithmic filtration axis. "
        "Above 0, the observed classes keep more bars alive at that value than their random labellings.",
    )

    # ---------------- prose
    summary = (
        "<p><strong>Under the rule fixed in advance, topology does not add.</strong> Level 1 gives one composition "
        f"stratum: each half's gap statistic rises at every k up to {k_max1}, but prediction strength at "
        f"k = {k_max1} is {level1['ps'][0]:.3f} (A→B) and {level1['ps'][1]:.3f} (B→A), below {domains.PS_MIN}, so "
        f"k = 1. The control finds no structure (k = 1), so level 2 runs. It gives no subdomains: the gap rises to "
        f"k = {k_max2} in both halves, and prediction strength there is {level2['ps'][0]:.3f} and "
        f"{level2['ps'][1]:.3f}. The verdict is “{S['verdict']['verdict']}”, with the reason "
        f"“{S['verdict']['reason']}”: the pairwise guard had nothing to test.</p>"
        f"<p><strong>No split recurs across the slide halves.</strong> The best k at both levels is 2, with "
        f"prediction strength {ps1_ab:.3f} / {ps1_ba:.3f} at level 1 and {ps2_ab:.3f} / {ps2_ba:.3f} at level 2. At "
        f"level 1 both halves split along the same direction (|cos| {ab['cos']:.3f}) into a minority of fuller "
        "balls with fewer epithelial and more stromal, immune and other cells, but half A's boundary lies further "
        f"toward the minority: {ab['minority']['crossing']:,} of half B's {ab['minority']['n']:,} minority centres "
        "fall on half A's majority side. Post hoc, on random half/half splits that mix the slides, prediction "
        f"strength at k = 2 is at least {low:.3f} in every direction at both levels: the split recurs within a "
        "mixed set of slides and moves between the two slide halves.</p>"
        f"<p><code>verify_results.py</code> recomputed every saved result by another route: {n_gated} checks "
        "passed.</p>"
    )

    rulings = [
        (
            "2026-11-20",
            "Simon",
            "Two levels: compartments from density and composition, with no topology; subdomains "
            "inside each, from chromatic topology against a reference.",
        ),
        (
            "2026-11-20",
            "Simon",
            "The pairwise guard, fixed in advance: the subdomains must not be predictable from "
            "Delaunay contacts per class pair, density and composition.",
        ),
        (
            "2026-11-20",
            "Simon",
            "Random labelling within the ball as the reference; the closed 150 µm ball; per-class domain and "
            "image Betti curves; a pilot of 30 slides with 1,500 centres each; recurrence across two halves of "
            "the slides and an H&amp;E atlas as the checks.",
        ),
        (
            "2026-11-20",
            "Simon, after the builder check",
            "k by the gap statistic against a Gaussian with the data's mean and covariance, confirmed by "
            "prediction strength of at least 0.8 in both directions, else k = 1; residuals scaled by the "
            "moderated spread.",
        ),
        (
            "2026-11-21",
            "Claude, before any study data (the k-rule correction)",
            "A half's k is where the gap stops rising: Tibshirani's stopping rule from k = 1, then the gain "
            "floor. The first formal version skipped k = 1 and split skewed cluster-free clouds.",
        ),
        (
            "2026-11-21",
            "Simon (the floor)",
            f"Gains Gap(k) − Gap(1) count from {min_gain}, raised from 0.02 before any "
            "study data, after skewed cluster-free clouds reached 0.02.",
        ),
        (
            "2026-11-21",
            "Simon (strata)",
            "Level-1 compartments are reported as composition strata, bins of a continuum that recur across "
            "the halves, never as modes; a cluster-free composition null joins the pre-flight, reported and "
            "not gated.",
        ),
    ]
    rulings_table = table(["date", "who", "ruling"], [list(row) for row in rulings], "left2")

    section1 = f"""
  <h2>1. Question and rulings</h2>
  <p>The cell-type study found that the per-type topology of whole slides re-describes density and composition.
  This pilot asks the next question at the scale where the liver is organised: around each cell, does the
  arrangement of cell types relative to each other, measured by the chromatic persistent homology of the cell's
  150 µm neighbourhood, define recurring microanatomical domains that density, composition and pairwise contacts
  do not? The design fixed two levels. <strong>Level 1</strong> groups centres by the density and composition of
  their balls, with no topology. <strong>Level 2</strong> looks inside each level-1 group for subdomains in
  chromatic topology measured against random labellings of the same cells.</p>
  <p>For each ball and class t, the <em>six-pack</em> of the inclusion (class t → all cells) holds six persistence
  modules. The <em>domain</em> (the class's own alpha persistent homology) and the <em>image</em> (the class's
  features that persist among all cells), in degrees 0 and 1, determine the other modules' Betti curves together
  with the codomain, which does not depend on the labels. So 4 classes × 2 modules × 2 degrees, the 16
  <em>blocks</em>, carry every residual of the six-packs. A centre's residual curve in a block is its observed
  reduced Betti curve minus the mean of 19 random labellings (positions and class counts kept), divided by
  sqrt(s² + s0²): its own spread over the references, moderated by the block's median. Each block's residuals go
  through an exact fPCA (9 components) and the 16 blocks through a multiple factor analysis (20 components): the
  level-2 features. The rulings:</p>
  {rulings_table}
  <p class="note">Every ruling, with what it costs if wrong, is in <code>design.md</code> and
  <code>progress.md</code>. The design, the plan and the rule for k were committed before any centre was
  computed.</p>"""

    section2 = f"""
  <h2>2. Pilot slides and compute</h2>
  <p>Thirty slides, five per condition, were drawn from the cell-type study's development slides without a quality
  flag. Within each condition a seeded split put 3 + 2 slides into halves A and B, alternating, so each half holds
  15 slides and every condition appears in both (A / B:
  {html.escape(", ".join(f"{k} {v}" for k, v in halves_per_condition.items()))}). Each slide gives 1,500 centre
  cells drawn at random, {N:,} in all. A centre's ball holds every cell within 150 µm (303.52 px at an assumed
  0.4942 µm/px): from {cells.min()} to {cells.max()} cells, median {np.median(cells):.0f}.</p>
  <p>The six-packs of 20 labellings per centre, {20 * N:,} chromatic complexes, took {bars_wall:,.0f} s
  ({bars_wall / 3600:.1f} h) of slide wall time on {workers} processes, {bars_started.replace("T", " ")} to
  {bars_finished[11:16]}: {bars_total:,} finite domain and image bars and {cod_total:,} codomain bars,
  {bars_bytes / 1e9:.2f} GB. <code>features.py</code> took {run["seconds"]:,} s ({run["cpu_seconds"]:,.0f} s of
  CPU) with a peak RSS of {run["peak_rss_gb"]:.2f} GB, <code>domains.py</code> {S["seconds"]["total"]:.0f} s, the
  pre-flight's re-run {PF["seconds"]:.0f} s, and <code>verify_results.py</code> {V["seconds"]["total"]:,.0f} s
  (peak RSS {V["peak_rss_gb"]:.2f} GB).</p>
  <details><summary>Table: the 30 pilot slides and their balls</summary>{slides_table}</details>"""

    section3 = f"""
  <h2>3. One composition stratum</h2>
  <p>Level 1 describes each ball by log10 of its cell count and its four class fractions, z-scored: each half with
  its own means and SDs for its gap statistic, both halves with the training half's in each prediction-strength
  direction. Each half's gap statistic rises at every k from 1 to {k_max1}, by {gain["level1", "A"]:.3f} in half
  A and {gain["level1", "B"]:.3f} in half B, so neither half stops before k<sub>max</sub> and
  k<sub>A</sub> = k<sub>B</sub> = {k_max1}. Prediction strength at k = {k_max1} is {level1["ps"][0]:.3f} (A→B)
  and {level1["ps"][1]:.3f} (B→A), under {domains.PS_MIN}, so k = 1: one composition stratum of all
  {stratum["size"]:,} centres, {stratum["size_A"]:,} per half on {stratum["slides_A"]} slides each. At no k from 2
  to {k_max1} does prediction strength reach {domains.PS_MIN} in both directions; the best is k = {k1}
  ({ps1_ab:.3f} / {ps1_ba:.3f}), so reading it at another k would not have changed the outcome. A map of one
  stratum would carry one colour, so the report draws none.</p>
  {fig_level1}
  <details><summary>Table: the level-1 gap and prediction-strength curves</summary>{rule_table(level1)}</details>
  <p><strong>Against the composition null.</strong> The pre-flight's cluster-free composition cloud of the same
  size (log10 of Poisson(120) + 1 cells and the fractions of a multinomial draw from Dirichlet(6, 2, 1, 1),
  reported and not gated) has a gap that rises to k = {null["k"]} and stops there, with gains of
  {null["gap_A"]["largest_gain"]:.3f} and {null["gap_B"]["largest_gain"]:.3f}, and prediction strength confirms
  {null["k"]} strata in both directions ({null["ps"][0]:.3f} / {null["ps"][1]:.3f}). In that cloud the rule reports
  bins of a continuum; in the real compositions the gap never stops rising and no split recurs across the
  halves.</p>
  <p><strong>The stratum and the halves.</strong> The stratum's mean ball holds 10<sup>{means["all"][0]:.3f}</sup>
  ≈ {10 ** means["all"][0]:.0f} cells, {100 * means["all"][1]:.1f}% epithelial, {100 * means["all"][2]:.1f}%
  stromal, {100 * means["all"][3]:.1f}% immune and {100 * means["all"][4]:.1f}% other. The halves differ: half A's
  balls hold more cells (mean log10 {means["A"][0]:.3f} against {means["B"][0]:.3f}), more of class other
  ({means["A"][4]:.3f} against {means["B"][4]:.3f}) and fewer epithelial cells ({means["A"][1]:.3f} against
  {means["B"][1]:.3f}). The largest difference, in pooled SDs, is in {feature_names[biggest]}
  ({num(std_diff[biggest], 2)}).</p>
  {profile_table}
  {fig_ecdf}
  <h3>Why no split recurs (descriptive)</h3>
  <p>In each prediction-strength direction both halves are in the training half's z-coordinates. At k = 2 they
  split along nearly the same direction: the |cosine| between the two halves' centroid differences is
  {ab["cos"]:.3f} (A→B) and {ba["cos"]:.3f} (B→A). Every split cuts off a minority of fuller balls with fewer
  epithelial and more stromal, immune and other cells: {100 * ab["test_minority_share"]:.1f}% of half B in A's
  coordinates, {100 * ab["train_minority_share"]:.1f}% of half A in its own. Along the shared direction, half A's
  boundary lies further toward the minority than half B's, by {ab["boundary_offset"]:.2f} z-units in A's
  coordinates and {-ba["boundary_offset"]:.2f} in B's. So half A's centroids put {ab["minority"]["crossing"]:,} of
  half B's {ab["minority"]["n"]:,} minority centres
  ({100 * ab["minority"]["crossing"] / ab["minority"]["n"]:.0f}%) on the majority side, and only
  {ab["minority"]["share"]:.3f} of that cluster's pairs stay together: that is PS(2) A→B. In the other direction
  half B's centroids put {ba["majority"]["crossing"]:,} of half A's {ba["majority"]["n"]:,} majority centres on
  the minority side, and PS(2) B→A is {ba["ps"]:.3f}.</p>
  {mech_table}
  {profile_split}
  <h3>Post hoc: random half/half splits (outside the rule)</h3>
  <p>Not part of the rule, and it cannot change the verdict: prediction strength at k = 2 on three random
  half/half splits of all {N:,} centres, each random half holding centres of all 30 slides, at level 1 (z-scored
  as the rule does) and at level 2 in the all-slides basis, the one basis both random halves share. Every
  direction reaches at least {low:.3f}. Within a mixed set of slides the 2-means split is stable; between the two
  slide halves it is not, at either level, and the all-slides basis does not change that.</p>
  {split_table}
  <p class="note">Seeds: <code>SeedSequence([SEED, 0x5B17, i, 1])</code>, i = 1..3; k-means as in the rule
  (<code>n_init=20</code>, <code>random_state=0</code>). The rule's level-2 values use each training half's own
  basis; the last row puts both slide halves in the all-slides basis.</p>"""

    section4 = f"""
  <h2>4. No subdomains</h2>
  <p>Level 2 clusters the stratum's centres on their 20 MFA components, in each half's own basis for the rule.
  Each half's gap rises at every k up to k<sub>max</sub> = {k_max2}, by {gain["level2", "A"]:.3f} in half A and
  {gain["level2", "B"]:.3f} in half B, so k<sub>A</sub> = k<sub>B</sub> = {k_max2}. Prediction strength at
  k = {k_max2} is {level2["ps"][0]:.3f} (A→B) and {level2["ps"][1]:.3f} (B→A), so k = 1: no subdomains. The best
  k is again {k2} ({ps2_ab:.3f} / {ps2_ba:.3f}); at k = 3 the B→A direction reaches
  {level2["ps_curve"]["B->A"][1]:.3f} but A→B only {level2["ps_curve"]["A->B"][1]:.3f}. As at level 1, half A's
  gains exceed half B's and A→B is the weaker direction at k = 2, and the post-hoc random splits of section 3
  show the same picture at this level.</p>
  {fig_level2}
  <details><summary>Table: the level-2 gap and prediction-strength curves</summary>{rule_table(level2)}</details>
  <p><strong>The stratum's mean residual curves.</strong> The mean of the 12,000 fit-sample centres' scaled
  residual curves shows, block by block, how the observed classes differ from their random labellings on
  average. In degree 0, the domain and image curves of the stromal, immune and other classes have negative
  integrals and dip furthest at α² of {span(h0_rest, "at_min")}. There the observed cells of each of these classes
  fall into fewer components, on their own and among all cells, than the same number of shuffled cells. In degree
  1, the image curves of all four classes have positive integrals and peak at α² of {span(image_h1, "at_max")}:
  there more loops of one class's cells stay holes among all cells than under shuffling. The domain H1 curves of all
  four classes rise above 0 before they fall below it: maxima at α² of {span(domain_h1, "at_max")}; minima at
  {span(domain_h1, "at_min")}. With no subdomains these are the stratum's only curves; there is no split of
  them to compare.</p>
  {fig_means}
  <details><summary>Table: each block's mean residual curve</summary>{curve_table}</details>"""

    section5 = f"""
  <h2>5. The control</h2>
  <p>The control asks whether the coordinates and the rule find structure in pure shuffle noise. Labelling 1 stands
  in for the observed classes, and its residual is taken against labellings 2..19, scaled with its own spread and
  the observed fit's s0, then scored in the observed fPCA components and projected with the observed MFA. It must
  give k = 1. It does: each half's gap stops at k = 1 (Gap(1) − (Gap(2) − s<sub>2</sub>) is
  {signed(margin(control, "A"))} in half A and {signed(margin(control, "B"))} in half B), so the control passed
  and level 2 ran. Half B's largest gain, {control["gap_B"]["largest_gain"]:.4f} at k = {k_max2}, comes after its
  gap has stopped. In the all-slides basis the control's level-2 columns have a mean SD of {ctl_sd:.2f}, against
  {obs_sd:.2f} for the observed.</p>
  {fig_control}
  <details><summary>Table: the control's gap and prediction-strength curves</summary>{rule_table(control)}</details>"""

    section6 = f"""
  <h2>6. The pairwise guard and the verdict</h2>
  <p>The guard, fixed in advance, runs in every composition stratum with subdomains. In each direction it compares
  how well the test half's subdomains are predicted by the training half's own centroids (stability) and by a
  gradient-boosted classifier on the 15 pairwise-contact, density and composition features. Topology adds in a
  stratum if stability wins with a 95% bootstrap interval above 0 in both directions, and the study's verdict is
  “adds” if such a stratum holds at least 10% of the centres. With no stratum of k ≥ 2 there was nothing to test,
  and the guard did not run. The verdict is <strong>“{S["verdict"]["verdict"]}”</strong>, with the reason
  “{S["verdict"]["reason"]}”.</p>"""

    section7 = """
  <h2>7. Contiguity</h2>
  <p>Spatial contiguity would compare, for each subdomain, the share of each centre's five nearest same-slide
  centres that share its subdomain with its expectation when the labels are reassigned at random within the
  stratum and slide. With no subdomains it was not computed.</p>"""

    section8 = """
  <h2>8. The H&amp;E atlas</h2>
  <p>The atlas was planned for compartments and subdomains: twelve random centres per group with their H&amp;E
  crops, labels hidden first, and Simon's words for each group recorded in the ledger. With one composition stratum
  and no subdomains there are no groups to compare. Whether to build it is Simon's decision.</p>"""

    section9 = f"""
  <h2>9. Methods and reproduction</h2>
  <ul>
    <li><strong>Balls and labellings.</strong> Per slide, 1,500 centres drawn without replacement; a centre's ball
    is every cell within δ = 150 µm, boundary included. Labelling 0 is the observed classes (the cell-type study's
    four groups of HistoPLUS types); labellings 1..19 permute them within the ball. Every generator uses
    <code>np.random.SeedSequence</code> with a 4-entry key whose last entry is non-zero.</li>
    <li><strong>Six-packs.</strong> The numba port of chalc's chromatic Delaunay–Čech filtration (faces to dimension
    2; values are squared smallest-enclosing-ball radii in px², topocyte's α² convention) and phimaker 0.5.0's
    <code>sixpack_from_inclusion</code>, one boundary matrix per complex, essential bars left out. The port passes
    <code>num_threads=1</code>, but phimaker runs up to {PHIMAKER_THREADS} threads inside each call
    (<code>progress.md</code>, Task 5). The pairing is unique for a fixed column order, so the results do not
    depend on the threads, and one ball per slide recomputes bit for bit (verification below). The throughput
    does: a centre of GTEX-YJ89-1426 took {SERIAL_SECONDS_PER_CENTRE:.2f} s serially against
    {yj["seconds_per_centre"]["median"]:.2f} s on {workers} processes, a speed-up of {speedup:.1f}×, not the
    ~{DESIGN_SERIAL_HOURS // DESIGN_HOURS}× the design assumed, and the bars took {bars_wall / 3600:.1f} h instead of
    ~{DESIGN_HOURS} h.</li>
    <li><strong>Residual curves.</strong> Exact signed step functions (observed bars with weight 1, reference bars
    with −1/19), divided by sqrt(s² + s0²), with s² the centre's spread over its 19 references and s0² the median
    positive s² over the fitting set's fit sample (400 centres per slide). Exact bar-endpoint arithmetic throughout:
    no grid and no quadrature. {zero_total:,} centre-blocks had s = 0 with a non-zero residual, {zero_image_h1:,} of
    them in the four image H1 blocks; the moderated spread keeps them finite.</li>
    <li><strong>fPCA.</strong> Per block and fitting set (half A, half B, all slides), the exact Gram matrix of the
    fit sample's scaled residuals and topocyte's <code>fpca_from_gram</code> (raw, divisor n). A block keeps at
    most 9 components. The effective rank threshold is <code>fpca_from_gram</code>'s default: an eigenvalue must
    exceed 1e-10 of the Gram matrix's trace, which is stricter than the design's 1e-12 of the largest eigenvalue,
    so that rule never binds. All {3 * len(c.BLOCKS)} fits kept 9 components and no block was dropped. Every
    centre is scored by exact evaluation against the component step functions.</li>
    <li><strong>MFA</strong> as in the sibling studies: each block divided by the SD of its first score on the fit
    sample, the 144 columns concatenated, PCA with the sibling sign rule, 20 components kept.</li>
    <li><strong>The rule for k</strong> (<code>domains.py</code>): per half, the gap statistic against B = 20 draws
    from a Gaussian with the half's mean and covariance (W<sub>1</sub> the total sum of squares, W<sub>k</sub> the
    k-means inertia, <code>n_init=20</code>, seed 0); the half's k by Tibshirani's stopping rule from k = 1 with a
    gain floor of {min_gain}; k is the smaller of the halves' and stands only with prediction strength of at least
    {domains.PS_MIN} in both directions. k<sub>max</sub> is {k_max1} at level 1 and {k_max2} at level 2.</li>
  </ul>
  <h3>Per block</h3>
  {block_table}
  <p class="note">s0 is in px: a squared L² norm integrates a count over squared radii (px²). FVE: the fraction of
  variance of the 9 components. Gate 1: <code>curves.inner_matrix</code> against topocyte's
  <code>betti_inner_product_matrix</code> (unit weights, 200 curves, relative); gate 2: the fit sample's scores
  against <code>fpca_from_gram</code>'s (relative to the largest score), the largest of the three fits.</p>
  {mfa_table}
  <h3>Verification</h3>
  <p><code>verify_results.py</code> recomputes each saved result by a route independent of the one that made it:
  {n_gated} checks passed, beside {V["n_checks"] - n_gated} descriptive count, in {V["seconds"]["total"]:,.0f} s.
  The saved labelling-0 bars match chalc
  15.0.1 to {sixpack["measured"]:.1e} of each ball's scale on {sixpack["sample"].split(",")[0]}; the halves' fits
  match <code>fpca_from_gram</code> to {halves_fit:.1e} of the largest score. The all-slides fPCA was refitted for
  blocks {" and ".join(map(str, V["samples"]["all_fit_blocks"]))} only (a refit of all 16 takes about 40 min more);
  the rule reads only the halves' bases.</p>
  <details><summary>Table: every check</summary>{verify_table}</details>
  <h3>Provenance</h3>
  <ul>
    <li>topocyte {c.TOPOCYTE_COMMIT}, a clean checkout named first on <code>PYTHONPATH</code>
    (<code>common.check_topocyte</code>); the study venv's packages are listed in
    <code>results/venv_freeze.txt</code>.</li>
    <li>The bars: <code>run_pilot.py bars</code> as committed in {BARS_COMMIT}, run on {bars_started[:10]}.</li>
    <li><code>results/features.npz</code>: <code>features.py</code> blob {FEATURES_BLOB} at {FEATURES_COMMIT} (md5
    {FEATURES_MD5[:8]}…, the first line of <code>logs/features.log</code>), started {run["started"]}.</li>
    <li><code>results/domains.json</code> and <code>.npz</code>: <code>domains.py</code> md5 {DOMAINS_MD5[:8]}…
    at {DOMAINS_COMMIT}; the pre-flight re-run at {DOMAINS_COMMIT}.</li>
    <li>This page checks the md5 sums of <code>features.py</code> and <code>domains.py</code> against those runs'
    before it is built.</li>
  </ul>"""

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chromatic domains pilot</title>
<style>{STYLE}</style>
</head>
<body>
<main>
  <h1>Chromatic domains in GTEx liver: the pilot</h1>
  <p class="subtitle">Does the local chromatic topology of 150 µm neighbourhoods define microanatomical domains
  beyond density, composition and pairwise contacts?</p>
  <p class="meta">Report generated {dt.date.today().isoformat()} from the saved results of <code>run_pilot.py</code>,
  <code>features.py</code>, <code>domains.py</code> and <code>verify_results.py</code>; topocyte {c.TOPOCYTE_COMMIT}.
  The design, with every ruling and the decision rules fixed in advance, is <code>design.md</code>; the ledger is
  <code>progress.md</code>.</p>
  <div class="summary">{summary}</div>
  {section1}
  {section2}
  {section3}
  {section4}
  {section5}
  {section6}
  {section7}
  {section8}
  {section9}
</main>
</body>
</html>
"""
    OUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
