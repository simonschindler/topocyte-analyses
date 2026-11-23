#!/usr/bin/env python
"""Build the self-contained HTML report of the e-means neighbourhood study from its saved results.

Run from this folder with the study venv after ``verify_results.py`` and ``check_kmeans.py``. Reads
``results/features.json``, ``results/gap.json``, ``results/clusters.json``, ``results/verify.json`` (it checks
that the last comes from this folder's ``verify_results.py`` and that every check passed) and
``results/kmeans_check.json`` (from this folder's ``check_kmeans.py``), and the logs in ``logs/`` for the runs'
time and memory (``/usr/bin/time -l``). Writes ``report.html`` with its figure embedded as a 128-colour PNG (the
sibling reports' palette), and nothing else. A few seconds.

No number in the prose is typed by hand: each is formatted from those files or from ``common``, and every
sentence that states a direction or an outcome is asserted against the numbers it states.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import re
import subprocess

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as c  # noqa: E402

OUT = c.HERE / "report.html"
INK, INK2, MUTED, RULE, SURFACE, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # the dataviz palette's first two slots, validated as a pair (sibling reports)
WORSE = 0.002  # log ratio of W_k above which the prose names a fit as worse than k-means++'s
TITLES = {
    "composition": "composition: the ball's four class fractions",
    "density": "density: the ball's fractions kept at thresholds 1–9",
    "composition+density": "composition + density, equal weight per block",
}

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
  .wrap1 td:first-child { white-space: normal; min-width: 18rem; }
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
    return f'<figure><img src="data:image/png;base64,{src}" alt="{html.escape(alt)}"><figcaption>{caption}</figcaption></figure>'


def table(header: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def num(value: float, digits: int = 3) -> str:
    """A number with a typographic minus and no "−0.000"."""
    return f"{0:.{digits}f}" if round(value, digits) == 0 else f"{value:.{digits}f}".replace("-", "−")


def signed(value: float, digits: int = 4) -> str:
    return f"{0:.{digits}f}" if round(value, digits) == 0 else f"{value:+.{digits}f}".replace("-", "−")


def percent(log_ratio: float, digits: int = 2) -> str:
    """A log ratio as a signed percentage difference."""
    return signed(100 * (np.exp(log_ratio) - 1), digits) + "%"


def resources(log: str) -> dict:
    """Wall time, peak RSS and peak memory footprint of the run that ``/usr/bin/time -l`` closed the log with."""
    text = (c.HERE / "logs" / log).read_text()
    real = float(re.findall(r"([\d.]+) real", text)[-1])
    rss = int(re.findall(r"(\d+)\s+maximum resident set size", text)[-1])
    footprint = int(re.findall(r"(\d+)\s+peak memory footprint", text)[-1])
    return {"seconds": real, "rss_gb": rss / 1e9, "footprint_gb": footprint / 1e9}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(c.HERE), *args], capture_output=True, text=True, check=True).stdout.strip()


def fig_gap(sets: dict, checks: dict, n_refs: int, rows: int) -> bytes:
    """Per set: Gap(k) with ±s_k and Tibshirani's k̂ ringed, and the k-means++ check's Gap(k) with ±1 SE; below,
    the data's log W_k and the references' mean."""
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), constrained_layout=True, sharex=True)
    for col, name in enumerate(c.SETS):
        r, kc = sets[name], checks[name]
        k, gap, s = np.array(r["k"]), np.array(r["gap"]), np.array(r["s"])
        k_hat = r["k_tibshirani"]
        top, bottom = axes[0, col], axes[1, col]
        top.errorbar(
            k,
            kc["gap_kmeans"],
            yerr=kc["gap_kmeans_se"],
            color=INK2,
            lw=1.2,
            ls="--",
            marker="o",
            ms=4.5,
            mfc=SURFACE,
            mec=INK2,
            mew=1.0,
            capsize=2,
            zorder=2,
            label=f"k-means++, {rows:,} cells and reference points",
        )
        top.errorbar(
            k,
            gap,
            yerr=s,
            color=SERIES[0],
            lw=2,
            marker="o",
            ms=5,
            mec=SURFACE,
            mew=0.8,
            capsize=2.5,
            zorder=3,
            label="e-means, every cell (the run)",
        )
        top.plot([k_hat], [gap[k_hat - 1]], marker="o", ms=12, mfc="none", mec=INK, mew=1.1, zorder=4)
        top.annotate(
            f"k̂ = {k_hat}",
            (k_hat, gap[k_hat - 1]),
            xytext=(-10, -16) if k_hat == len(k) else (-10, 6),  # below-left at k_max, clear of the k - 1 marker
            textcoords="offset points",
            ha="right",
            color=INK,
        )
        top.axhline(0, color=CONTEXT, lw=0.8, zorder=1)
        top.set_title(TITLES[name], loc="left", color=INK, fontsize=9)
        top.set_ylabel("Gap(k), ± s_k" if col == 0 else "")
        top.grid(alpha=0.6)
        if col == 0:
            top.legend(frameon=False, fontsize=7.8, loc="lower right")
        bottom.plot(k, r["log_w"], color=SERIES[0], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=0.8, label="data")
        bottom.plot(
            k,
            r["log_w_reference_mean"],
            color=SERIES[1],
            lw=2,
            marker="s",
            ms=4,
            mec=SURFACE,
            mew=0.8,
            label=f"Gaussian reference, mean of {n_refs}",
        )
        bottom.set_xticks(k)
        bottom.set_xlabel("k (clusters)")
        bottom.set_ylabel("log W_k" if col == 0 else "")
        bottom.grid(alpha=0.6)
        if col == 0:
            bottom.legend(frameon=False, fontsize=7.8, loc="upper right")
    return png_bytes(fig)


def gap_table(r: dict) -> str:
    gap, s = np.array(r["gap"]), np.array(r["s"])
    rows = []
    for i, k in enumerate(r["k"]):
        stops = "" if i + 1 == len(gap) else ("yes" if gap[i] >= gap[i + 1] - s[i + 1] else "no")
        rows.append(
            [
                str(k),
                num(r["log_w"][i], 4),
                num(r["log_w_reference_mean"][i], 4),
                num(gap[i], 4),
                num(s[i], 5),
                "" if i == 0 else signed(gap[i] - gap[i - 1]),
                stops,
            ]
        )
    header = ["k", "log W_k", "mean log W*_k", "Gap(k)", "s_k", "Gap(k) − Gap(k − 1)", "Gap(k) ≥ Gap(k + 1) − s_{k+1}"]
    return table(header, rows)


def profile_table(clusters: list[dict]) -> str:
    rows = []
    for label, row in enumerate(clusters):
        rows.append(
            [
                str(label),
                f"{row['n']:,}",
                f"{100 * row['share']:.1f}%",
                f"{row['cells_per_ball']:.0f}",
                *(f"{row['class_fractions'][name]:.3f}" for name in c.CLASSES),
                *(f"{value:.3f}" for value in row["kept_fractions"][1:]),
            ]
        )
    header = [
        "label",
        "cells",
        "share",
        "cells per ball",
        *c.CLASSES,
        *(f"kept {j}" for j in range(1, c.N_THRESHOLDS)),
    ]
    return table(header, rows)


def kmeans_table(r: dict, kc: dict) -> str:
    rows = [
        [
            str(k),
            f"{kc['w_run_centroids'][i]:.5f}",
            f"{kc['w_kmeans'][i]:.5f}",
            f"{percent(kc['log_ratio_run_to_kmeans'][i])} ± {100 * kc['log_ratio_se'][i]:.2f}%",
            num(r["gap"][i], 4),
            f"{num(kc['gap_kmeans'][i], 4)} ± {kc['gap_kmeans_se'][i]:.4f}",
        ]
        for i, k in enumerate(kc["k"])
    ]
    header = [
        "k",
        "W_k, the run's centroids",
        "W_k, k-means++",
        "run against k-means++ <span class='unit'>(± paired SE)</span>",
        "Gap(k), the run",
        "Gap(k), k-means++ <span class='unit'>(± SE)</span>",
    ]
    return table(header, rows)


def main() -> None:
    feats = json.loads((c.RESULTS / "features.json").read_text())
    gap_all = json.loads((c.RESULTS / "gap.json").read_text())
    clusters = json.loads((c.RESULTS / "clusters.json").read_text())
    verify = json.loads((c.RESULTS / "verify.json").read_text())
    kcheck = json.loads((c.RESULTS / "kmeans_check.json").read_text())
    assert verify["all_ok"] and all(check["ok"] for check in verify["checks"])
    assert verify["script_md5"] == c.md5(c.HERE / "verify_results.py"), "verify.json is not from this verify_results.py"
    settings, sets = gap_all["settings"], gap_all["sets"]
    assert list(sets) == list(c.SETS) and settings["script_md5"] == c.md5(c.HERE / "cluster.py")
    assert feats["script_md5"] == c.md5(c.HERE / "features.py")
    assert kcheck["settings"]["script_md5"] == c.md5(c.HERE / "check_kmeans.py"), "not from this check_kmeans.py"
    assert kcheck["settings"]["cluster_md5"] == settings["script_md5"] and list(kcheck["sets"]) == list(c.SETS)
    checks, rows = kcheck["sets"], kcheck["settings"]["rows"]
    assert kcheck["settings"]["reference_points"] == rows  # the prose names one number for both samples
    n_refs, k_max = settings["n_references"], settings["k_max"]
    run_features, run_cluster = resources("features.log"), resources("cluster.log")
    run_check = resources("check_kmeans.log")

    # ---------------- the outcome, asserted against the curves it states
    lines, worse_ratios = [], []
    for name in c.SETS:
        r, kc = sets[name], checks[name]
        gap = np.array(r["gap"])
        steps = np.diff(gap)
        rises = bool(np.all(steps > 0))
        best = int(np.argmax(steps)) + 2  # the k whose step from k - 1 is largest
        k_t, k_p, k_km = r["k_tibshirani"], r["k_pilot_rule"], kc["k_tibshirani"]
        if rises:
            assert k_t == k_max
        else:
            assert k_t == int(np.argmax(steps <= 0)) + 1  # the rule stops where Gap first does not rise
        shape = (
            f"rises at every k, so Tibshirani's rule stops at k_max = {k_t}"
            if rises
            else f"stops rising first at k = {k_t} (Tibshirani's k̂)"
        )
        ratio = np.array(kc["log_ratio_run_to_kmeans"])  # entry k - 1 is k clusters
        worse = [k for k in range(2, k_max + 1) if ratio[k - 1] > WORSE]
        worse_ratios += [ratio[k - 1] for k in worse]
        close = max(abs(ratio[k - 1]) for k in range(2, k_max + 1) if k not in worse)
        fits = f"the run's fits are within {100 * (np.exp(close) - 1):.2f}% of k-means++'s W_k at every k" + (
            " but " + " and ".join(f"k = {k} ({percent(ratio[k - 1])})" for k in worse) if worse else ""
        )
        gap_km = np.array(kc["gap_kmeans"])
        if k_km == k_t and k_t < k_max:
            assert k_t not in worse and k_t + 1 not in worse
            outcome = (
                f"Tibshirani's rule gives the same k̂: the run's fits at k = {k_t} and {k_t + 1} match k-means++'s, "
                f"whose gap falls by {num(gap_km[k_t - 1] - gap_km[k_t], 4)} there (e-means: "
                f"{num(gap[k_t - 1] - gap[k_t], 4)})."
            )
        elif k_km == k_t:
            outcome = "Tibshirani's rule gives the same k̂."
        else:
            assert k_t + 1 in worse and k_km > k_t
            outcome = (
                f"Tibshirani's rule gives k̂ = {k_km}, not {k_t}: e-means' stop at k = {k_t} comes from its fit at "
                f"k = {k_t + 1}."
            )
        lines.append(
            f"<p><strong>{html.escape(name)}</strong> ({len(r['features']['columns'])} columns): Gap(k) {shape}; "
            f"the pilot's rule gives k̂ = {k_p}. The largest step is from k = {best - 1} to {best} "
            f"({signed(steps[best - 2])}); Gap({k_max}) = {num(gap[-1], 4)}. On {rows:,} of its cells, {fits}. "
            f"With k-means++ on those cells and {rows:,} reference points instead, {outcome}</p>"
        )
    s_max = max(max(sets[name]["s"]) for name in c.SETS)
    n_fits = len(c.SETS) * (k_max - 1)
    summary = "".join(lines) + (
        f"<p>s_k measures only how the {n_refs} references vary (at most {num(s_max, 4)}), not how far a fit is "
        f"from the best partition. In {len(worse_ratios)} of the {n_fits} data fits e-means settled "
        f"{percent(min(worse_ratios))} to {percent(max(worse_ratios))} above k-means++'s W_k: their labels are a "
        "worse partition than k-means++'s, and the run's Gap(k) at those k is too low by about as much.</p>"
    )

    # ---------------- sections
    gap_sections = "".join(f"<h3>{html.escape(TITLES[name])}</h3>{gap_table(sets[name])}" for name in c.SETS)
    profile_sections = []
    for name in c.SETS:
        k_t = sets[name]["k_tibshirani"]
        others = "".join(
            f"<h4>k = {k}</h4>{profile_table(clusters[name]['clusters'][str(k)])}"
            for k in range(2, k_max + 1)
            if k != k_t
        )
        profile_sections.append(
            f"<h3>{html.escape(TITLES[name])}: k̂ = {k_t}</h3>{profile_table(clusters[name]['clusters'][str(k_t)])}"
            f"<details><summary>every other k</summary>{others}</details>"
        )
    kmeans_sections = "".join(
        f"<h3>{html.escape(TITLES[name])}</h3>{kmeans_table(sets[name], checks[name])}" for name in c.SETS
    )
    inrun_rows = [
        [name, k, f"{v['e_means']:.5f}", f"{v['kmeans']:.5f}", f"{v['ratio']:.4f}"]
        for name in c.SETS
        for k, v in sets[name]["check_against_kmeans"].items()
    ]
    small = [checks[name]["subsample_emeans_k4"] for name in c.SETS]
    assert len({(x["total_steps"], x["batch_size"]) for x in small}) == 1
    run_recipes = {
        (f["recipe"]["total_steps"], f["recipe"]["batch_size"])
        for name in c.SETS
        for f in sets[name]["fits"]
        if "recipe" in f
    }
    assert len(run_recipes) == 1
    ((run_steps, run_batch),) = run_recipes
    reproduced = max(
        abs(now / was - 1) for name in c.SETS for was, now in checks[name]["reproduces_run_kmeans"].values()
    )
    small_reproduced = max(abs(x["mse"] / x["logged_mse"] - 1) for x in small)
    trace = max(abs(kc["trace_check"]["rebuilt_w1"] / kc["trace_check"]["run_w1"] - 1) for kc in checks.values())
    max_d = max(len(sets[name]["features"]["columns"]) for name in c.SETS)
    recipe = sets[c.SETS[0]]["fits"][1]["recipe"]
    recipe_rows = []
    for name in c.SETS:
        fits = [f for f in sets[name]["fits"] if "recipe" in f]
        batches = sorted({f["recipe"]["batch_size"] for f in fits})
        alphas = sorted({f["recipe"]["alpha0"] for f in fits})
        revived = sum(f["revived"] for f in fits)
        seconds = [f["seconds"] for f in fits]
        recipe_rows.append(
            [
                name,
                ", ".join(f"{b:,}" for b in batches),
                ", ".join(f"{a:.4g}" for a in alphas),
                str(revived),
                f"{min(seconds):.0f}–{max(seconds):.0f}",
                f"{sets[name]['seconds'] / 60:.0f}",
            ]
        )
    verify_rows = [
        [html.escape(ch["check"]), html.escape(json.dumps(ch["measured"], ensure_ascii=False))]
        for ch in verify["checks"]
    ]
    versions = {**feats["versions"], **settings["versions"], **kcheck["settings"]["versions"]}
    built = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    head = git("log", "-1", "--format=%h", "--", ".") or "uncommitted"

    body = f"""
<h1>e-means clusters of every liver cell's 150 µm neighbourhood</h1>
<p class="subtitle">{feats["n_cells"]:,} cells of {feats["n_slides"]} GTEx liver slides, three feature sets,
gap statistic for k = 1–{k_max}</p>
<p class="meta">Built {built} from the study folder at {html.escape(head)}; {len(verify["checks"])} checks of
<code>verify_results.py</code> pass.</p>
<div class="summary">{summary}</div>

<h2>Gap statistic</h2>
{
        figure(
            fig_gap(sets, checks, n_refs, rows),
            "Gap curves and log W_k of the three feature sets",
            f"Top: Gap(k) = mean log W*_k − log W_k with ± s_k (too small to see), Tibshirani's k̂ ringed; dashed, the "
            f"same gap from k-means++ on {rows:,} cells and {rows:,} reference points, ± 1 SE (section e-means against "
            f"k-means++). Bottom: the data's log W_k (every cell) and the mean over {n_refs} Gaussian references of "
            f"{settings['n_reference_points']:,} points each; the gap is the distance between the two curves.",
        )
    }
<p>The reference is a Gaussian with the set's mean and covariance (the chromatic pilot's rule). It reads any
departure from a Gaussian shape as structure, bounded and skewed fractions included, so a gap that keeps rising
says the cloud is not Gaussian, not that it holds {k_max} separate groups. Tibshirani's k̂ is the smallest k with
Gap(k) ≥ Gap(k + 1) − s_(k+1), else k_max; the pilot's rule keeps it only if Gap(k̂) − Gap(1) ≥
{settings["min_gain"]}.</p>
{gap_sections}

<h2>Clusters</h2>
<p>Every cell's label is its nearest final centroid. A cluster's row gives its cells, its mean cells per ball,
and the mean over its cells of the ball's class fractions and of the fractions kept at density thresholds 1–9
(threshold 0 keeps every cell). Labels are e-means' own numbering, as in <code>labels_&lt;set&gt;.npz</code> and
<code>clusters.json</code>.</p>
{"".join(profile_sections)}

<h2>e-means against k-means++</h2>
<p><code>check_kmeans.py</code> takes cluster.py's {rows:,} random cells of each set and compares W_k of the run's
centroids, which e-means fitted to every cell, with W_k of scikit-learn's k-means++
(<code>KMeans(k, n_init=20)</code>) fitted to those cells, both as the mean squared distance over the same cells. It
then draws {rows:,} points of the set's Gaussian reference, fits k-means++ to them the same way, and computes the
gap from the two k-means++ curves; Tibshirani's rule uses the run's s_k. A Gap's SE comes from the spread of the
squared distances in each sample, the paired SE from the per-cell differences between the two fits. k-means++ is
fitted to the cells it is scored on, which favours it slightly (of the order of k·d/n of W_k, at most
{100 * k_max * max_d / rows:.3f}% here).</p>
{kmeans_sections}
<details><summary>cluster.py's own check (gap.json, <code>check_against_kmeans</code>)</summary>
<p>cluster.py refitted e-means to the {rows:,} cells alone, a separate fit: e-means sizes its batch and steps by
the sample, {small[0]["total_steps"]:,} steps of {small[0]["batch_size"]:,} cells there against the run's
{run_steps:,} steps of {run_batch:,} on every cell. These ratios therefore say nothing about the run's fits; the
tables above replace them.</p>
{table(["set", "k", "e-means W_k, refitted to the subsample", "k-means++ W_k", "ratio"], inrun_rows)}
</details>

<h2>Method and run</h2>
<ul>
<li>Ball: the closed {feats["radius_um"]:.0f} µm ball around every cell ({feats["radius_px"]:.2f} px at the assumed
{feats["px_um"]} µm per pixel), the cell included; mean {feats["mean_cells_per_ball"]:.1f} cells, at most
{feats["max_count"]}.</li>
<li>Density thresholds: the pathology study's deciles of the 10-NN density, {
        num(feats["thresholds_cells_per_mm2"][1], 0)
    }
to {num(feats["thresholds_cells_per_mm2"][-1], 0)} cells per mm² for thresholds 1–9; a cell is kept at
threshold j when its density is at least d_j.</li>
<li>Features: every column z-scored over all cells; the combined set's blocks divided by √(their columns).</li>
<li>e-means: <code>faiss.contrib.e_means.NumpyEmeans</code> (faiss {versions["faiss"]}), {settings["n_epochs"]}
passes, recipe <code>{html.escape(recipe["name"])}</code> with its {html.escape(recipe["batch_law"])} batch law
({html.escape(recipe["batch_law_reason"])}).</li>
</ul>
{table(["set", "batch size", "α0", "clusters revived", "seconds per fit", "minutes for the set"], recipe_rows)}
<p class="note">features.py: {run_features["seconds"] / 60:.1f} min, peak RSS {run_features["rss_gb"]:.1f} GB;
cluster.py: {run_cluster["seconds"] / 3600:.1f} h, peak RSS {run_cluster["rss_gb"]:.1f} GB, peak memory footprint
{run_cluster["footprint_gb"]:.1f} GB; check_kmeans.py: {run_check["seconds"] / 60:.0f} min (M4, 10 cores, 16 GB).
Python {versions["python"]}, numpy {versions["numpy"]}, scipy {versions["scipy"]}, faiss {versions["faiss"]},
scikit-learn {versions["scikit-learn"]}.</p>

<h2>Checks</h2>
<p><code>verify_results.py</code> ({verify["seconds"] / 60:.0f} min); its docstring says what each check takes
another route for and what it does not redo.</p>
{table(["check", "measured"], verify_rows, "wrap1")}
<p><code>check_kmeans.py</code> rebuilt cluster.py's subsample from <code>counts.npy</code> and the run's column
moments: k-means++ reproduces the W_4 and W_10 the run logged (largest relative difference {reproduced:.1e}),
e-means refitted to the subsample reproduces the run's W_4 there ({small_reproduced:.1e}), and the rebuilt features'
total variance equals the run's W_1 ({trace:.1e}).</p>
"""
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>e-means neighbourhoods</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )
    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
