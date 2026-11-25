#!/usr/bin/env python
"""Build the self-contained HTML report of the Betti FPCR benchmark from its saved results.

Run from this folder after ``verify_results.py``, with topocyte's checkout at ``18e3aa5`` first on ``PYTHONPATH``
(``README.md``, Run)::

    $PY build_report.py

It calls ``common.check_topocyte`` and ``common.check_upstream`` first, and refuses (``RuntimeError``) unless
``results/verify.json``:

- comes from this folder's ``verify_results.py`` (the md5 it records);
- records every check passed;
- is newer than every stage's json, by the dates they record: a verifier that failed or crashed after a stage ran
  again leaves an earlier ``verify.json`` in place, which is not trusted;
- records, for every data file read here, the sha256 of the bytes read here: ``results/arms.npz`` and upstream's three
  result tables and four curvature pickles.

It reads the stages' jsons in ``results/``; ``results/arms.npz`` through ``run_study.load_arms``, which refuses an npz
that ``arms.json`` does not vouch for; upstream's three tables (``results_interpolation.csv``,
``results_curvature.csv``, ``results_fluid.csv``) with pandas and ``float_precision="round_trip"``, asserted equal to
the reported values in ``readouts.json``; the curvature clouds through ``inputs.curvature_data`` (the whitelist
loader), for their coordinates; and the logs in ``logs/``, for the stages' wall time and peak RSS
(``/usr/bin/time -l``). It runs no stage and fits nothing.

Computed here, for description only, never for a verdict: from ``arms.npz``, the fluid cells in which arm F's R² lies
below ``FAIL_BELOW``, with the (K0, K1) they chose; the median, the share above 0 and the mean without those cells of
Δ(F, K) over the fluid cells; for interpolation, the median of Δ(F, K) over the datasets, how many lie above 0, and how
much of the worst training size's mean comes from its datasets whose K0 + K1 lies within ``NEAR_CAP`` of the cap
n − 2; arm F's chosen K0 + K1 per fluid training size; and whether the curvature predictions have the sign of the true
curvature.

Writes ``report.html``, with its figures embedded as 128-colour PNGs in the sibling reports' palette and look, and
nothing else. A few seconds.

Every result in the prose is formatted from those files or from ``common``, and every sentence that states a direction
or an outcome is asserted against the numbers it states. The counts of the gates (their records in the stages' jsons),
of the build's rulings (numbered in ``progress.md``) and of the reported Turkeš et al. methods are derived, not typed.
Written into the text by hand are only labels and the design's fixed parameters: the paper's table and figure numbers
and arXiv id, the ledger's dates and ruling numbers, the 95% of the intervals (asserted against the bootstrap's
percentiles), the bandwidth grid's [0.1, 2], and the curvature targets' linspace(−2, 2, n) (asserted against the data).
``README.md`` is asserted to state, as this report formats them: in its status line, the verdicts with their intervals
and gate G1's worst differences; elsewhere, its table of the three pairs' readouts row by row, the number of rulings,
the curvature MSEs, the interpolation and fluid information notes, the G3 observation, the npz sizes and the table of
the stages' runs.
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator  # noqa: E402

import common as c  # noqa: E402
import inputs as inp  # noqa: E402
import matrices as mat  # noqa: E402
import readouts as ro  # noqa: E402
import run_study as rs  # noqa: E402

OUT = c.HERE / "report.html"
INK, INK2, MUTED, RULE, SURFACE, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
# The dataviz palette's first three slots, which pass its all-pairs check on the light surface: one colour per arm in
# every figure. The third (aqua, arm B) is below 3:1 against the surface, so every arm also has its own marker, and
# every figure has a table beside it.
ARM_COLOR = {"K": "#2a78d6", "F": "#eb6834", "B": "#1baf7a"}
ARM_MARKER = {"K": "o", "F": "s", "B": "^"}
ARM_NAME = {"K": "K: PD_kernel reimplemented", "F": "F: Betti FPCR", "B": "B: Betti NW"}
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b")  # its blue ramp
TASKS, ARMS = ro.TASKS, ro.ARMS
TASK_NAME = {"interp": "interpolation", "curv": "curvature", "fluid": "fluid"}
PAIRS = ("F_vs_K", "B_vs_K", "F_vs_B")  # the primary pair, then the exploratory pairs as readouts.json keys them
PAIR_LABEL = {"F_vs_K": "Δ(F, K): primary", "B_vs_K": "Δ(B, K): representation", "F_vs_B": "Δ(F, B): estimator"}
PAIR_MARKER = {"F_vs_K": "o", "B_vs_K": "s", "F_vs_B": "D"}
PAIR_COLOR = {"F_vs_K": INK, "B_vs_K": INK2, "F_vs_B": MUTED}  # pairs are not arms: ink, not the arms' colours
VERDICT_TEXT = {"beats": "beats", "below": "is below", "level": "is level with"}
SIDES = ("above 0", "below 0", "containing 0")  # where an interval lies (``side``)
# The one reported Turkeš et al. method whose input is not a geodesic distance matrix: upstream's
# curvature/Turkes_Experiment/turkes_curvature_experiment.py (lines 217-237, 477, 537, read as text) feeds it the
# stored coordinates as 3D point clouds, the 2D ones padded with z = 0 (data_construction.calculate_3d_point_clouds).
POINT_NET = "point_net"
FAIL_BELOW = -10.0  # the information note's threshold on arm F's fluid R² (progress.md, the readouts line)
NEAR_CAP = 2  # the note counts the failed cells that chose K0 + K1 within this of n − 2
FLUID_YLIM = (-0.6, 1.0)  # the fluid figure's R² axis; a mean below it is marked at the bottom, its value written
PAPER = f"splits 0–{c.PAPER_SPLITS - 1}"  # the paper's fluid splits
ALL = f"splits 0–{c.FLUID_SPLITS - 1}"  # this study's
FLUID_ROWS = (("paper_splits", f"the paper's {PAPER}"), ("all_splits", ALL))
STAGE_JSONS = ("upstream", "inputs", "matrices", "arms", "readouts")
STAGES = (  # (command, log, json, the script whose md5 the table shows): the runs whose outputs are read here
    ("fetch_upstream.py", None, "upstream", "fetch_upstream.py"),
    ("inputs.py", "inputs.log", "inputs", "inputs.py"),
    ("matrices.py", "matrices.log", "matrices", "matrices.py"),
    ("run_study.py arms", "arms.log", "arms", "run_study.py"),
    ("run_study.py readouts", "readouts.log", "readouts", "readouts.py"),
    ("verify_results.py", "verify.log", "verify", "verify_results.py"),
)
CSV = {  # upstream's reported tables, by this folder's path, as verify.json keys their sha256
    "interp": "upstream/interpolation/results_interpolation.csv",
    "curv": "upstream/curvature/results_curvature.csv",
    "fluid": "upstream/fluid/results_fluid.csv",
}
CURVATURE_PICKLES = tuple(
    f"upstream/curvature/data/{name}.pkl" for name in ("point_clouds", "labels", "train_indices", "test_indices")
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
  .wrap1 th, .wrap1 td { white-space: normal; text-align: left; vertical-align: top; }
  .wrap1 td:nth-child(2) { min-width: 26rem; }
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


def minus(text: str) -> str:
    return text.replace("-", "−")


def lower_first(text: str) -> str:
    """``text`` with its first letter in lower case, to continue a sentence."""
    return text[:1].lower() + text[1:]


def num(value: float, digits: int = 4) -> str:
    """A number to ``digits`` decimals, with a typographic minus and no "−0.0000"."""
    return f"{0:.{digits}f}" if round(value, digits) == 0 else minus(f"{value:.{digits}f}")


def sig(value: float, digits: int = 4, plus: bool = False) -> str:
    """``value`` to ``digits`` significant digits, trailing zeros kept, with a typographic minus, and a plus sign if
    ``plus`` and it is positive: the readouts' format (``-0.02401``, ``+0.001110``, ``-15.34``)."""
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    decimals = max(digits - 1 - exponent, 0)
    if abs(round(value, decimals)) >= 10 ** (exponent + 1):  # rounding carried into the next power of ten
        decimals = max(decimals - 1, 0)
    return minus(f"{value:{'+' if plus else ''}.{decimals}f}")


def delta_text(readout: dict) -> str:
    """A readout's mean Δ and 95% interval: ``-0.02401 [-0.03837, -0.01141]``, typographic minus."""
    return f"{sig(readout['mean'], plus=True)} [{sig(readout['lo'], plus=True)}, {sig(readout['hi'], plus=True)}]"


def scientific(value: float, digits: int = 1) -> str:
    """``3.6e-16`` rather than ``3.61e-16`` or ``3.6e-016``; ``digits=0`` gives a tolerance's ``1e-9``."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def markdown(text: str) -> str:
    """``text`` escaped for HTML, its ```code``` spans as ``<code>``."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))


def pct(part: float, whole: float, digits: int = 1) -> str:
    return f"{100 * part / whole:.{digits}f}%"


def listing(items) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = [str(item) for item in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def mean_sd(stats: dict) -> str:
    return f"{num(stats['mean'])} ± {num(stats['std'])}"


def definition(task: str, x: str, y: str) -> str:
    """Δ(x, y) of ``task``, signed so that Δ > 0 favours ``x`` (``readouts.delta``)."""
    return f"MSE({y}) − MSE({x})" if task == "curv" else f"R²({x}) − R²({y})"


def side(readout: dict) -> str:
    """Where a readout's 95% interval lies with respect to 0."""
    return "below 0" if readout["hi"] < 0 else "above 0" if readout["lo"] > 0 else "containing 0"


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def when(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def git(*args: str, folder=c.HERE) -> str:
    return subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, check=True).stdout.strip()


def resources(log: str | None) -> dict:
    """Wall time and peak RSS of the run that ``/usr/bin/time -l`` closed ``log`` with; ``{}`` if it holds none."""
    if log is None or not (c.LOGS / log).exists():
        return {}
    text = (c.LOGS / log).read_text()
    real, rss = re.findall(r"([\d.]+) real", text), re.findall(r"(\d+)\s+maximum resident set size", text)
    return {"seconds": float(real[-1]), "rss_gb": int(rss[-1]) / 1e9} if real and rss else {}


def progress_rulings() -> int:
    """How many rulings ``progress.md`` numbers in its section "Rulings during the build", asserted to run 1, 2, …"""
    text = (c.HERE / "progress.md").read_text()
    section = text.split("## Rulings during the build", 1)[1].split("\n## ", 1)[0]
    numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", section, flags=re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1)), f"progress.md's rulings are not numbered 1, 2, …: {numbers}"
    return len(numbers)


def check_readme(claims: dict[str, str], status: dict[str, str]) -> None:
    """``README.md`` states each claim as this report formats it, a line break reading as a space, and its status
    line, the paragraph that opens with ``**Status:``, states each of ``status``. ``AssertionError`` naming every claim
    it does not state."""
    raw = (c.HERE / "README.md").read_text()
    start = raw.find("**Status:")
    assert start >= 0, "README.md has no status line, a paragraph opening with **Status:"
    end = raw.find("\n\n", start)
    line = " ".join(raw[start : len(raw) if end < 0 else end].split())
    text = " ".join(raw.split())
    missing = [f"its status line, {what}: {claim}" for what, claim in status.items() if claim not in line]
    missing += [f"{what}: {claim}" for what, claim in claims.items() if claim not in text]
    assert not missing, "README.md does not state, as this report formats them:\n" + "\n".join(missing)


# ---------------------------------------------------------------- what the report may rely on


def check_verified(jsons: dict[str, dict], arms_sha: str) -> None:
    """Refuse (``RuntimeError``) unless ``verify.json`` vouches for what this report reads (module docstring)."""
    verify = jsons["verify"]
    checks = verify["checks"]
    if not (
        verify["passed"] is True
        and checks
        and all(check["passed"] is True for check in checks)
        and verify["checks_passed"] == verify["checks_total"] == len(checks)
    ):
        raise RuntimeError("results/verify.json does not record every check passed; run verify_results.py")
    if verify["provenance"]["md5"]["verify_results.py"] != md5(c.HERE / "verify_results.py"):
        raise RuntimeError("results/verify.json is not from this folder's verify_results.py; run it again")
    newer = [name for name in STAGE_JSONS if when(jsons[name]["date"]) >= when(verify["date"])]
    if newer:
        stale = ", ".join(f"{name}.json" for name in newer)
        raise RuntimeError(
            f"results/verify.json ({verify['date']}) is not newer than {stale}: a stage ran after the verifier; run "
            "verify_results.py again"
        )
    read = {"results/arms.npz": arms_sha} | {
        path: sha256(c.HERE / path) for path in (*CSV.values(), *CURVATURE_PICKLES)
    }
    differing = [path for path, digest in read.items() if verify["sha256"].get(path) != digest]
    if differing:
        raise RuntimeError(f"results/verify.json did not check these files as read here: {differing}; run it again")
    if jsons["readouts"]["arms_npz_sha256"] != arms_sha:
        raise RuntimeError("results/readouts.json was not computed from this results/arms.npz; run the readouts stage")


def check_reported(tables: dict) -> None:
    """The reported values in ``readouts.json``'s tables equal upstream's CSVs, read exactly (pandas, round_trip)."""
    interp = pd.read_csv(c.HERE / CSV["interp"], float_precision="round_trip").set_index("n_train")
    for row in tables["interp"]:
        for method, stats in row["reported"].items():
            assert stats == {stat: float(interp.loc[row["n_train"], f"{method}_R2_{stat}"]) for stat in stats}
    curv = pd.read_csv(c.HERE / CSV["curv"], float_precision="round_trip")
    assert tables["curv"]["reported"] == dict(zip(curv["method"], curv["MSE"].astype(float).tolist(), strict=True))
    fluid = pd.read_csv(c.HERE / CSV["fluid"], float_precision="round_trip")
    reported = {(float(t), int(n)): {"mean": float(m), "std": float(s)} for t, n, m, s in fluid.itertuples(index=False)}
    assert {(row["t"], row["n_train"]): row["reported"] for row in tables["fluid"]} == reported


def pair(readouts: dict, name: str) -> dict:
    """The readouts of the pair ``name`` (``PAIRS``): the primary's, or an exploratory pair's."""
    return readouts["primary"] if name == "F_vs_K" else readouts["exploratory"][name]


# ---------------------------------------------------------------- figures


def fig_deltas(readouts: dict) -> bytes:
    """One panel per task: the mean Δ of the primary pair and of the two exploratory pairs, with 95% intervals."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11.5, 2.7), constrained_layout=True)
    for ax, task in zip(axes, TASKS, strict=True):
        for i, name in enumerate(PAIRS):
            r, y = pair(readouts, name)[task], len(PAIRS) - 1 - i
            ax.errorbar(
                [r["mean"]],
                [y],
                xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                fmt=PAIR_MARKER[name],
                ms=7 if name == "F_vs_K" else 6,
                color=PAIR_COLOR[name],
                mec=SURFACE,
                ecolor=PAIR_COLOR[name],
                elinewidth=2,
                capsize=3,
                zorder=3,
            )
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        ax.set_yticks(range(len(PAIRS)), [PAIR_LABEL[name] for name in PAIRS][::-1])
        ax.tick_params(axis="y", length=0, labelleft=ax is axes[0])
        ax.set_ylim(-0.6, len(PAIRS) - 0.4)
        verdict = readouts["primary"][task]["verdict"]
        ax.set_title(f"{TASK_NAME[task]}: F {VERDICT_TEXT[verdict]} K", loc="left", color=INK, fontsize=9)
        ax.set_xlabel(f"Δ in {'MSE' if task == 'curv' else 'R²'}; Δ > 0 favours the first arm")
        ax.grid(axis="x", alpha=0.6)
    return png_bytes(fig)


def fig_interp(readouts: dict) -> bytes:
    """Left, as the paper's Figure 6: each arm's R² per training size, mean ± SD over the runs, beside the reported
    PD_kernel and PI_kernel. Right: each pair's Δ per training size, with its 95% interval."""
    rows = readouts["tables"]["interp"]
    n = np.array([row["n_train"] for row in rows], dtype=float)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]}
    )
    dodge = {"K": -1.5, "F": 0.0, "B": 1.5}
    pi = [row["reported"]["PI_kernel"] for row in rows]
    left.errorbar(
        n + 3.0,
        [s["mean"] for s in pi],
        yerr=[s["std"] for s in pi],
        color=MUTED,
        ls="--",
        marker="D",
        ms=4.5,
        lw=1.4,
        elinewidth=1.0,
        capsize=2,
        label="PI_kernel, as reported",
        zorder=2,
    )
    for arm in ARMS:
        left.errorbar(
            n + dodge[arm],
            [row[arm]["mean"] for row in rows],
            yerr=[row[arm]["std"] for row in rows],
            color=ARM_COLOR[arm],
            marker=ARM_MARKER[arm],
            ms=5.5,
            lw=2,
            elinewidth=1.1,
            capsize=2,
            mec=SURFACE,
            mew=0.6,
            label=ARM_NAME[arm],
            zorder=3,
        )
    left.scatter(
        n + dodge["K"],
        [row["reported"]["PD_kernel"]["mean"] for row in rows],
        s=80,
        facecolor="none",
        edgecolor=INK,
        lw=1.1,
        zorder=4,
        label="PD_kernel, as reported",
    )
    left.set_xticks(n, [f"{v:.0f}" for v in n])
    left.set_xlabel("training size n")
    left.set_ylabel(f"R² on the {c.INTERP_N_TEST} test clouds, mean ± SD over the {c.INTERP_RUNS} runs")
    left.grid(axis="y", alpha=0.6)
    left.legend(frameon=False, fontsize=7.8, loc="lower right")
    pair_dodge = {"F_vs_K": -1.5, "B_vs_K": 0.0, "F_vs_B": 1.5}
    for name in PAIRS:
        by = pair(readouts, name)["interp"]["by_size"]
        m = np.array([b["mean"] for b in by])
        lo, hi = np.array([b["lo"] for b in by]), np.array([b["hi"] for b in by])
        right.errorbar(
            n + pair_dodge[name],
            m,
            yerr=[m - lo, hi - m],
            fmt=PAIR_MARKER[name],
            ms=5.5,
            color=PAIR_COLOR[name],
            mec=SURFACE,
            ecolor=PAIR_COLOR[name],
            elinewidth=1.6,
            capsize=2,
            label=PAIR_LABEL[name],
            zorder=3,
        )
    right.axhline(0, color=INK2, lw=1, zorder=1)
    right.set_xticks(n, [f"{v:.0f}" for v in n])
    right.set_xlabel("training size n")
    right.set_ylabel("Δ in R², with its 95% interval")
    right.grid(axis="y", alpha=0.6)
    right.legend(frameon=False, fontsize=7.8, loc="lower right")
    return png_bytes(fig)


def fig_curvature(arms: dict[str, np.ndarray], mse: dict[str, float]) -> bytes:
    """Each arm's predictions of the 50 test clouds' curvature against the true curvature."""
    y = arms["curv_y_test"]
    fig, axes = plt.subplots(1, len(ARMS), figsize=(11.5, 3.7), sharex=True, sharey=True, constrained_layout=True)
    reach = 1.1 * float(max(np.abs(y).max(), *(np.abs(arms[f"curv_pred_{arm}"]).max() for arm in ARMS)))
    for ax, arm in zip(axes, ARMS, strict=True):
        ax.plot([-reach, reach], [-reach, reach], color=CONTEXT, lw=1, zorder=1)
        ax.axhline(0, color=RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        ax.scatter(
            y,
            arms[f"curv_pred_{arm}"],
            s=26,
            color=ARM_COLOR[arm],
            marker=ARM_MARKER[arm],
            edgecolor=SURFACE,
            linewidths=0.6,
            zorder=3,
        )
        ax.set_title(f"{ARM_NAME[arm]}, MSE {num(mse[arm])}", loc="left", color=INK, fontsize=9)
        ax.set_xlabel("true curvature K of the test cloud")
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.grid(alpha=0.5)
    axes[0].set_ylabel("predicted curvature")
    return png_bytes(fig)


def fig_fluid(readouts: dict) -> bytes:
    """As the paper's Figure 9: each arm's R² per training size, mean ± SD over the splits, one panel per time; the
    paper's splits 0–4 above, beside the reported PD_kernel, and all 100 splits below."""
    rows = readouts["tables"]["fluid"]
    low, high = FLUID_YLIM
    fig, axes = plt.subplots(
        len(FLUID_ROWS), len(c.FLUID_TIMES), figsize=(13.5, 6.0), sharex=True, sharey=True, constrained_layout=True
    )
    dodge = {"K": 0.9, "F": 1.0, "B": 1.1}  # multiplicative, on the log axis
    for r, (key, label) in enumerate(FLUID_ROWS):
        for col, t in enumerate(c.FLUID_TIMES):
            ax = axes[r, col]
            cells = [row for row in rows if row["t"] == t]
            n = np.array([row["n_train"] for row in cells], dtype=float)
            for arm in ARMS:
                m = np.array([row[key][arm]["mean"] for row in cells])
                s = np.array([row[key][arm]["std"] for row in cells])
                x, inside = n * dodge[arm], m >= low
                ax.errorbar(
                    x[inside],
                    m[inside],
                    yerr=s[inside],
                    color=ARM_COLOR[arm],
                    marker=ARM_MARKER[arm],
                    ms=5,
                    lw=1.8,
                    elinewidth=1.0,
                    capsize=2,
                    mec=SURFACE,
                    mew=0.5,
                    zorder=3,
                )
                for xi, mi in zip(x[~inside], m[~inside], strict=True):  # below the axis: marked, its value written
                    ax.plot([xi], [low], marker="v", ms=7, color=ARM_COLOR[arm], mec=SURFACE, clip_on=False, zorder=4)
                    ax.annotate(
                        num(mi, 2),
                        (xi, low),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=INK2,
                    )
            if key == "paper_splits":
                ax.scatter(
                    n * dodge["K"],
                    [row["reported"]["mean"] for row in cells],
                    s=70,
                    facecolor="none",
                    edgecolor=INK,
                    lw=1.0,
                    zorder=5,
                )
            ax.set_xscale("log")
            ax.set_xticks(n, [f"{v:.0f}" for v in n])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylim(low, high)
            ax.grid(axis="y", alpha=0.6)
            if r == 0:
                ax.set_title(f"t = {t:g}", loc="left", color=INK, fontsize=9)
            if r == len(FLUID_ROWS) - 1:
                ax.set_xlabel("training size n")
            if col == 0:
                ax.set_ylabel(f"{label}\nR² on {c.FLUID_N_TEST} test simulations")
    handles = [
        Line2D([], [], color=ARM_COLOR[arm], marker=ARM_MARKER[arm], mec=SURFACE, lw=1.8, label=ARM_NAME[arm])
        for arm in ARMS
    ]
    handles.append(Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=INK, label="PD_kernel, as reported"))
    handles.append(
        Line2D([], [], ls="", marker="v", ms=7, color=MUTED, label=f"mean below {num(low, 1)}, value written")
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=8)
    return png_bytes(fig)


def fig_failures(total: np.ndarray, failed: np.ndarray, cap: int, n: int) -> bytes:
    """Arm F's chosen K0 + K1 in the fluid cells of training size ``n``, those whose R² lies below ``FAIL_BELOW``
    stacked on the others."""
    values = np.arange(cap + 1)
    ok = np.array([np.sum((total == v) & ~failed) for v in values])
    bad = np.array([np.sum((total == v) & failed) for v in values])
    fig, ax = plt.subplots(figsize=(11.5, 2.9), constrained_layout=True)
    ax.bar(
        values,
        ok,
        width=0.8,
        color=CONTEXT,
        edgecolor=SURFACE,
        linewidth=1,
        label=f"R² at or above {num(FAIL_BELOW, 0)}",
    )
    ax.bar(
        values,
        bad,
        width=0.8,
        bottom=ok,
        color=ARM_COLOR["F"],
        edgecolor=SURFACE,
        linewidth=1,
        label=f"R² below {num(FAIL_BELOW, 0)}",
    )
    ax.annotate(
        f"n − 2 = {cap}: the most\nthe eligibility rule allows",
        (cap, ok[cap] + bad[cap]),
        xytext=(-12, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK2,
    )
    ax.set_xticks(values)
    ax.set_xlim(-0.7, cap + 0.7)
    ax.set_xlabel(f"K0 + K1 that arm F chose (the fluid cells of training size n = {n})")
    ax.set_ylabel("cells")
    ax.grid(axis="y", alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    return png_bytes(fig)


def fig_chosen(chosen: dict) -> bytes:
    """How often arm F chose each (K0, K1): interpolation's datasets and fluid's cells, on a log colour scale."""
    cmap = LinearSegmentedColormap.from_list("palette_blue", SEQUENTIAL)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    for ax, task, unit in zip(axes, ("interp", "fluid"), ("datasets", "cells"), strict=True):
        grid = np.zeros((c.K_MAX + 1, c.K_MAX + 1))
        for row in chosen[task]:
            grid[row["k0"], row["k1"]] = row["count"]
        image = ax.imshow(
            np.ma.masked_equal(grid, 0),
            origin="lower",
            cmap=cmap,
            aspect="equal",
            norm=LogNorm(vmin=1, vmax=grid.max()),
        )
        ax.set_xlabel("K1: H1 scores in the regression")
        ax.set_ylabel("K0: H0 scores in the regression")
        ax.set_xticks(range(0, c.K_MAX + 1, 2))
        ax.set_yticks(range(0, c.K_MAX + 1, 2))
        ax.set_title(
            f"{TASK_NAME[task]}: {int(grid.sum()):,} {unit}, {len(chosen[task])} distinct (K0, K1)",
            loc="left",
            color=INK,
            fontsize=9,
        )
        bar = fig.colorbar(image, ax=ax, shrink=0.85)
        bar.set_label(unit)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        bar.ax.yaxis.set_minor_formatter(NullFormatter())
    return png_bytes(fig)


# ---------------------------------------------------------------- the report


def main() -> None:
    c.check_topocyte()
    c.check_upstream()
    jsons = {name: json.loads((c.RESULTS / f"{name}.json").read_text()) for name in (*STAGE_JSONS, "verify")}
    arms, arms_sha = rs.load_arms()
    check_verified(jsons, arms_sha)
    upstream_json, inputs_json, matrices_json, arms_json, readouts, verify = (
        jsons[name] for name in (*STAGE_JSONS, "verify")
    )
    tables = readouts["tables"]
    check_reported(tables)
    assert readouts["primary"]["pair"] == ["F", "K"] and list(readouts["exploratory"]) == ["B_vs_K", "F_vs_B"]
    assert [pair(readouts, name)["pair"] for name in PAIRS] == [["F", "K"], ["B", "K"], ["F", "B"]]
    boot = readouts["bootstrap"]
    assert (boot["n_boot"], boot["seed"], boot["percentiles"]) == (c.N_BOOT, c.BOOT_SEED, [2.5, 97.5])

    # ---------------- the verdicts
    primary = {task: readouts["primary"][task] for task in TASKS}
    verdicts = {task: primary[task]["verdict"] for task in TASKS}
    for task, r in primary.items():  # the rule, read again: the verdict is where the interval lies
        assert verdicts[task] == {"below 0": "below", "above 0": "beats", "containing 0": "level"}[side(r)]
    assert set(verdicts.values()) == {"below"}, "the summary, the status line and the index state 'below' thrice"
    verdict_phrase = "is below `PD_kernel` (arm K) on all three tasks"
    task_claims = {
        task: f"{TASK_NAME[task]} {delta_text(primary[task])} (Δ = {definition(task, 'F', 'K')})" for task in TASKS
    }

    # ---------------- gate G1, and the other gates
    g1 = arms_json["g1"]
    g1_worst = max(g1[task]["worst_abs_diff"] for task in TASKS)
    g1_values = sum(g1[task]["compared"] for task in TASKS)
    (g1_check,) = [check for check in verify["checks"] if check["family"] == "g1"]
    g1_exact = g1_check["worst"]
    assert g1["passed"] and g1_worst <= g1["tolerance"] == c.G1_TOL and g1_exact <= g1_check["tolerance"]
    assert g1_check["measured"]["compared"] == {task: g1[task]["compared"] for task in TASKS}
    g1_phrase = f"to {scientific(g1_worst)} at worst ({scientific(g1_exact)} with the CSVs read exactly)"
    g2, g4, g3 = matrices_json["g2"], matrices_json["g4"], arms_json["g3"]
    assert g2["passed"] and g3["passed"] and g4["passed"] and inputs_json["g5"]["passed"]
    g3_worst = max(g3["worst_relative"].values())
    g3_cell = max(g3["cells"], key=lambda cell: cell["worst_relative_all_columns"])
    g3_all = g3_cell["worst_relative_all_columns"]
    assert g3_all == max(g3["worst_relative_all_columns"].values()) and g3_all > g3["tolerance"] >= g3_worst
    assert g3_cell["task"] == "fluid" and all(cell["rank"] == cell["rank_transform"] for cell in g3["cells"])
    g3_phrase = f"up to {scientific(g3_all)} of the largest score"
    g4_cell = max(g4["per_time_and_dimension"], key=lambda cell: cell["relative_to_largest_entry"])
    g4_worst = g4_cell["relative_to_largest_entry"]
    g4_entry = max(cell["entrywise"]["worst_relative"] for cell in g4["per_time_and_dimension"])
    assert g4_worst <= g4["tolerance"] < g4_entry  # entry by entry the shifted Grams differ beyond the tolerance
    g5 = inputs_json["g5"]
    g5_clouds = sum(g5["rips_clouds_compared"].values())
    assert not any(g5["rips_clouds_differing_from_ripser"].values()) and not g5["fluid_diagrams_missing"]
    assert g5["interpolation_equals_its_seed_0_regeneration"]
    n_gates = len([g1, g2, g3, g4, g5])  # every gate's record, each asserted passed above
    rulings = progress_rulings()

    # ---------------- the exploratory pairs: Δ(F, K) = Δ(F, B) + Δ(B, K), mean by mean
    fk, bk, fb = ({task: pair(readouts, name)[task] for task in TASKS} for name in PAIRS)
    for task in TASKS:
        assert math.isclose(fk[task]["mean"], fb[task]["mean"] + bk[task]["mean"], rel_tol=1e-12, abs_tol=1e-15)
    # interpolation: the representation's part lies below 0, the estimator's contains 0
    assert side(bk["interp"]) == "below 0" and side(fb["interp"]) == "containing 0"
    # curvature: the representation's part contains 0, the estimator's lies below 0 and is most of the gap
    assert side(bk["curv"]) == "containing 0" and side(fb["curv"]) == "below 0"
    curv_share = fb["curv"]["mean"] / fk["curv"]["mean"]
    assert 0.5 < curv_share < 1
    # fluid: both lie below 0, the estimator's part is most of the mean, and every size's interval lies below 0
    assert side(bk["fluid"]) == "below 0" and side(fb["fluid"]) == "below 0"
    fluid_share = fb["fluid"]["mean"] / fk["fluid"]["mean"]
    assert 0.5 < fluid_share < 1
    by_size = {name: pair(readouts, name)["fluid"]["by_size"] for name in PAIRS}
    for name in PAIRS:
        assert [b["n_train"] for b in by_size[name]] == list(c.FLUID_SIZES)
        assert all(side(b) == "below 0" for b in by_size[name])
    n_small = c.FLUID_SIZES[0]
    rest_bk = [b["mean"] for b in by_size["B_vs_K"]]
    rest_fb = [b["mean"] for b in by_size["F_vs_B"][1:]]
    rest_fk = [b["mean"] for b in by_size["F_vs_K"][1:]]
    small_fb, small_fk = by_size["F_vs_B"][0], by_size["F_vs_K"][0]
    assert small_fk["mean"] < min(rest_fk) and small_fb["mean"] < min(rest_fb)  # the smallest size is the worst
    # at the larger sizes the two parts are of the same order: within a factor of 2 of each other
    assert 0.5 < min(map(abs, rest_fb)) / max(map(abs, rest_bk)) and max(map(abs, rest_fb)) / min(map(abs, rest_bk)) < 2
    # interpolation's Δ(F, B) contains 0 overall as a cancellation: by size, its interval lies above 0 at some sizes
    # and below 0 at others
    fb_sizes = {where: [b["n_train"] for b in fb["interp"]["by_size"] if side(b) == where] for where in SIDES}
    assert fb_sizes["above 0"] and fb_sizes["below 0"]
    cancellation = (
        f"Δ(F, B)'s interval lies above 0 at n = {listing(fb_sizes['above 0'])} and below 0 at n = "
        f"{listing(fb_sizes['below 0'])}"
    )
    explore = {
        "interp": f"In interpolation the gap is the representation's: on the Betti curves' L² distances K's own "
        f"estimator falls below K, Δ(B, K) = {delta_text(bk['interp'])}, while F cannot be told apart from B, "
        f"Δ(F, B) = {delta_text(fb['interp'])}, an interval containing 0. That is a cancellation: by size, "
        f"{cancellation}.",
        "curv": f"In curvature it is the estimator's: B cannot be told apart from K, Δ(B, K) = "
        f"{delta_text(bk['curv'])}, an interval containing 0, while F falls below B, Δ(F, B) = "
        f"{delta_text(fb['curv'])}, {pct(curv_share, 1, 0)} of F's mean gap to K.",
        "fluid": f"In fluid both parts lie below 0: Δ(B, K) = {delta_text(bk['fluid'])} and Δ(F, B) = "
        f"{delta_text(fb['fluid'])}. The estimator's part is {pct(fluid_share, 1, 0)} of the mean, through the "
        f"training size n = {n_small}, where Δ(F, B) = {delta_text(small_fb)}. At the larger sizes the two parts are "
        f"of the same order: Δ(B, K) lies between {sig(min(rest_bk))} and {sig(max(rest_bk))} at every training "
        f"size, and Δ(F, B) between {sig(min(rest_fb))} and {sig(max(rest_fb))} at the larger ones, every interval "
        "below 0. Here the representation's part also holds the essential births, which W2 compares and the Betti "
        "curves drop.",
    }

    # ---------------- the fluid cells in which arm F fails: information, not a readout
    r2 = {arm: arms[f"fluid_r2_{arm}"] for arm in ARMS}  # (times, splits, sizes)
    k_fluid = arms["fluid_k_F"]  # (times, splits, sizes, 2)
    total = k_fluid.sum(axis=-1)
    delta = r2["F"] - r2["K"]
    failed = r2["F"] < FAIL_BELOW
    n_failed, n_cells = int(failed.sum()), delta.size
    assert {c.FLUID_SIZES[si] for si in np.nonzero(failed)[2]} == {n_small}, "the note says: all at the smallest size"
    cap = n_small - 2  # K0 + K1 ≤ n − 2, the eligibility rule's cap
    assert total[:, :, 0].max() <= cap
    at_cap, near_cap = int(np.sum(total[failed] == cap)), int(np.sum(total[failed] >= cap - NEAR_CAP))
    assert at_cap > n_failed / 2  # "most of them"
    worst = np.unravel_index(int(np.argmin(r2["F"])), r2["F"].shape)
    worst_value, worst_k = float(r2["F"][worst]), tuple(int(k) for k in k_fluid[worst])
    median_delta, n_above = float(np.median(delta)), int(np.sum(delta > 0))
    mean_without = float(delta[~failed].mean())
    assert median_delta < 0 and mean_without < 0 and n_above < n_cells / 2  # the note: the sign does not rest on them
    assert math.isclose(float(delta.mean()), fk["fluid"]["mean"], rel_tol=1e-12)
    medians = {arm: float(np.median(r2[arm])) for arm in ARMS}
    assert medians["F"] < medians["B"] < medians["K"]
    below0, below1 = int(np.sum(r2["F"] < 0)), int(np.sum(r2["F"] < -1))
    fail_phrases = {
        "the failed cells": f"All {n_failed} cells where F's R² lies below {num(FAIL_BELOW, 0)} have n = {n_small}",
        "their (K0, K1)": f"{at_cap} of them chose K0 + K1 = {cap} = n − 2, the most the eligibility rule allows, and "
        f"{near_cap} chose {cap - NEAR_CAP} or more",
        "the fluid readout by size": f"n = {n_small} {delta_text(small_fk)}, and from {sig(min(rest_fk))} to "
        f"{sig(max(rest_fk))} at the other sizes, every interval below 0",
        "the sign": f"without them the mean Δ(F, K) is {sig(mean_without)}, the median over all {n_cells:,} cells is "
        f"{sig(median_delta)}, and F's R² lies above K's in {n_above} of the {n_cells:,} cells "
        f"({pct(n_above, n_cells)})",
    }
    worst_text = (
        f"{minus(f'{worst_value:,.0f}')} (t = {c.FLUID_TIMES[worst[0]]:g}, split {worst[1]}, n = "
        f"{c.FLUID_SIZES[worst[2]]}, (K0, K1) = {worst_k})"
    )

    # ---------------- interpolation
    interp_rows = tables["interp"]
    assert [row["n_train"] for row in interp_rows] == list(c.INTERP_SIZES)
    sizes_below = [b["n_train"] for b in fk["interp"]["by_size"] if side(b) == "below 0"]
    sizes_level = [b["n_train"] for b in fk["interp"]["by_size"] if side(b) == "containing 0"]
    assert sizes_below and sizes_level and len(sizes_below) + len(sizes_level) == len(c.INTERP_SIZES)
    over_pi = [row["F"]["mean"] - row["reported"]["PI_kernel"]["mean"] for row in interp_rows]
    assert min(over_pi) > 0  # F's mean lies above the reported PI_kernel's at every size
    # for information, not a readout: arm F's interpolation gap dataset by dataset (datasets in size order, then runs)
    interp_delta = arms["interp_r2_F"] - arms["interp_r2_K"]
    interp_n = np.repeat(c.INTERP_SIZES, c.INTERP_RUNS)
    interp_k_total = arms["interp_k_F"].sum(axis=1)
    assert math.isclose(float(interp_delta.mean()), fk["interp"]["mean"], rel_tol=1e-12)
    interp_median, interp_above = float(np.median(interp_delta)), int(np.sum(interp_delta > 0))
    assert interp_median < 0 and interp_above < interp_delta.size / 2  # most datasets lie below K
    worst_size = min(fk["interp"]["by_size"], key=lambda b: b["mean"])  # the size with the most negative mean Δ
    n_worst, at = worst_size["n_train"], np.flatnonzero(interp_n == worst_size["n_train"])
    assert math.isclose(float(interp_delta[at].mean()), worst_size["mean"], rel_tol=1e-12)
    interp_cap = n_worst - 2  # K0 + K1 ≤ n − 2
    near = at[interp_k_total[at] >= interp_cap - NEAR_CAP]  # that size's near-saturated datasets
    share_near = float(interp_delta[near].sum() / interp_delta[at].sum())
    assert near.size and np.all(interp_delta[near] < 0) and share_near > 0.5 and interp_k_total[at].max() <= interp_cap
    assert set(near.tolist()) == set(at[np.argsort(interp_delta[at])[: near.size]].tolist())  # they are its worst
    # over all sizes: the lowest Δ(F, K) belong to near-saturated fits, as many as come before the first that is not
    by_delta = np.argsort(interp_delta, kind="stable")
    near_cap = (interp_k_total >= interp_n - 2 - NEAR_CAP)[by_delta]
    assert near_cap[0] and not near_cap.all()
    worst_ids = by_delta[: int(np.argmin(near_cap))]  # argmin finds the first False
    assert set(near.tolist()) <= set(worst_ids.tolist())
    interp_phrases = {
        "the worst datasets": f"the {worst_ids.size} interpolation datasets with the lowest Δ(F, K) ("
        f"{listing(f'{i} at n = {interp_n[i]}' for i in worst_ids)}) all chose K0 + K1 within {NEAR_CAP} of n − 2",
        "the interpolation median": f"The median Δ(F, K) over the {interp_delta.size} datasets is "
        f"{sig(interp_median)}, and F's R² lies above K's in {interp_above} of them",
        "the interpolation size": f"n = {n_worst}'s mean Δ(F, K), {sig(worst_size['mean'])}, comes "
        f"{pct(share_near, 1, 0)} from datasets {listing(near.tolist())}, whose K0 + K1 is "
        f"{listing(interp_k_total[near].tolist())} against the cap of {interp_cap}",
        "the cancellation": cancellation,
    }

    # ---------------- curvature
    curv = tables["curv"]
    mse = {arm: curv[arm] for arm in ARMS}
    assert mse["K"] == curv["reported"]["PD_kernel"] and mse["K"] < mse["B"] < mse["F"]
    turkes = {method: value for method, value in curv["reported"].items() if method != "PD_kernel"}
    assert max(mse["K"], mse["B"]) < min(turkes.values())  # K and B lie below all of them
    # upstream's Turkes_Experiment (read as text): point_net reads the stored coordinates as 3D clouds, the 2D ones
    # padded with z = 0; every other method a geodesic distance matrix
    assert POINT_NET in turkes
    n_turkes, n_geodesic = len(turkes), len(turkes) - 1
    turkes_lower = [method for method, value in turkes.items() if value < mse["F"]]
    turkes_higher = [method for method, value in turkes.items() if value > mse["F"]]
    assert turkes_lower and turkes_higher and len(turkes_lower) + len(turkes_higher) == len(turkes)
    clouds = inp.curvature_data()
    y_curv = clouds["y"]
    assert np.array_equal(y_curv[clouds["test"]], arms["curv_y_test"])
    assert np.array_equal(y_curv[clouds["train"]], np.linspace(-2, 2, clouds["train"].size))
    dims = np.array([X.shape[1] for X in clouds["clouds"]])
    points = {X.shape[0] for X in clouds["clouds"]}
    positive, negative = y_curv > 0, y_curv < 0
    assert np.all(positive | negative) and np.array_equal(dims == 3, positive) and np.array_equal(dims == 2, negative)
    for X, k in zip(clouds["clouds"], y_curv, strict=True):  # on the sphere of radius 1/√K; in the disk of 1/√−K
        radii = np.linalg.norm(X, axis=1) * math.sqrt(abs(k))
        assert np.allclose(radii, 1, rtol=0, atol=1e-9) if k > 0 else radii.max() < 1
    signs_right = {arm: int(np.sum(np.sign(arms[f"curv_pred_{arm}"]) == np.sign(arms["curv_y_test"]))) for arm in ARMS}
    n_test_curv = arms["curv_y_test"].size
    assert all(right == n_test_curv for right in signs_right.values())  # every arm gets every test cloud's sign right
    k_curv = tuple(int(k) for k in arms["curv_k_F"])
    assert readouts["chosen_k"]["curv"] == [{"k0": k_curv[0], "k1": k_curv[1], "count": 1}]

    # ---------------- the chosen (K0, K1)
    chosen = readouts["chosen_k"]
    k_interp = arms["interp_k_F"]
    n_interp = np.repeat(c.INTERP_SIZES, c.INTERP_RUNS)
    for task, k in (("interp", k_interp), ("fluid", k_fluid)):  # readouts.json's counts are the arrays'
        pairs_, counts = np.unique(k.reshape(-1, 2), axis=0, return_counts=True)
        assert chosen[task] == [
            {"k0": int(a), "k1": int(b), "count": int(n)} for (a, b), n in zip(pairs_, counts, strict=True)
        ]
    top = {task: max(chosen[task], key=lambda row: row["count"]) for task in ("interp", "fluid")}
    zero_fluid = next((row["count"] for row in chosen["fluid"] if (row["k0"], row["k1"]) == (0, 0)), 0)
    interp_total = k_interp.sum(axis=1)
    interp_at_cap = int(np.sum(interp_total == n_interp - 2))
    size_rows = []
    for si, n in enumerate(c.FLUID_SIZES):
        t = total[:, :, si]
        zero = int(np.sum((k_fluid[:, :, si] == 0).all(axis=-1)))
        size_rows.append(
            [
                str(n),
                f"{np.median(t):g}",
                str(int(t.max())),
                f"{int(np.sum(t == n - 2))} of {t.size}",
                str(zero),
                str(int(np.sum(failed[:, :, si]))),
            ]
        )

    # ---------------- the caveats' numbers
    fluid_inputs = inputs_json["fluid"]
    assert fluid_inputs["essential_per_diagram"] == [[1, 2]] and fluid_inputs["essential_is_one_h0_two_h1"]
    empty = fluid_inputs["empty_finite_per_time"]
    empty_h0, empty_h1 = ([empty[key][h] for key in empty] for h in ("h0", "h1"))
    paper = {name: pair(readouts, name)["fluid"]["paper_splits"] for name in PAIRS}
    assert paper["F_vs_K"]["splits"] == list(range(c.PAPER_SPLITS)) and side(paper["F_vs_K"]) == "below 0"
    n_fluid = arms["fluid_y"].size

    # ---------------- the runs
    provenance = next(check for check in verify["checks"] if check["family"] == "provenance")
    commit_of = {(row[0], row[2]): row[1] for row in provenance["measured"]["rows"]}
    stage_rows, readme_rows, untimed = [], [], []
    for command, log, name, script in STAGES:
        record, run = jsons[name], resources(log)
        if not run:
            untimed.append(command)
        recorded = record["provenance"]["md5"][script]
        note = "" if recorded == md5(c.HERE / script) else f" (as committed in {commit_of[(f'{name}.json', script)]})"
        seconds = record.get("seconds")
        wall = (
            f"{run['seconds']:.2f} s"
            if run
            else f"{seconds} s inside the stage"
            if isinstance(seconds, int | float)
            else "not logged"
        )
        rss = f"{run['rss_gb']:.2f} GB" if run else "not logged"
        date = record["date"][:16]
        stage_rows.append([f"<code>{command}</code>", date, wall, rss, f"<code>{script}</code> {recorded}{note}"])
        readme_rows.append(f"| `{command}` | {date} | {wall} | {rss} | `{script}` {recorded}{note} |")
    seconds = arms_json["seconds"]
    arm_seconds = [
        [
            ARM_NAME[arm],
            *(f"{seconds[arm][task]:.1f}" for task in TASKS),
            f"{sum(seconds[arm][task] for task in TASKS):.1f}",
        ]
        for arm in ARMS
    ]
    fluid_share_time = {arm: seconds[arm]["fluid"] / sum(seconds[arm][task] for task in TASKS) for arm in ARMS}
    assert all(share > 0.9 for share in fluid_share_time.values())  # the fluid cells take nearly all of each arm's time
    versions = {**verify["provenance"]["versions"], "matplotlib": matplotlib.__version__}
    upstream_date = git("log", "-1", "--format=%cs", folder=c.UPSTREAM)
    assert upstream_json["sha"] == c.UPSTREAM_SHA == verify["upstream_clean_at"]
    npz = {"matrices.npz": matrices_json["npz"], "arms.npz": arms_json["npz"]}
    npz_text = {name: f"{record['arrays']} arrays, {record['bytes'] / 1e6:.1f} MB" for name, record in npz.items()}
    for name in npz:  # the npz files are the ones verify.json checked
        assert verify["sha256"][f"results/{name}"] == npz[name]["sha256"]

    results_rows = {  # README's table of the readouts, row by row
        task: f"| {TASK_NAME[task]} | {delta_text(fk[task])}, **{verdicts[task]}** | {delta_text(bk[task])} | "
        f"{delta_text(fb[task])} |"
        for task in TASKS
    }
    check_readme(
        {
            **{f"the readouts' row of {task}": row for task, row in results_rows.items()},
            "G1's values": f"the {g1_values} values behind the paper's Tables 1–3",
            "the curvature MSEs": f"curvature MSE, {num(mse['F'])}, is {mse['F'] / mse['K']:.1f} times K's, "
            f"{num(mse['K'])}",
            "the Turkeš methods": f"F's below {len(turkes_higher)} of them",
            "the curvature signs": f"every arm predicts the sign of K right for all {n_test_curv} test clouds",
            "the Turkeš inputs": f"{n_geodesic} of those methods used geodesic distance matrices",
            **fail_phrases,
            **interp_phrases,
            "the rulings": f"the {rulings} rulings made during the build",
            "the G3 observation": g3_phrase,
            **{f"{name}'s size": text for name, text in npz_text.items()},
            **{f"the run of {row[0]}": readme_row for row, readme_row in zip(STAGES, readme_rows, strict=True)},
        },
        {
            "the verdicts": verdict_phrase,
            **{f"the {task} readout": claim for task, claim in task_claims.items()},
            "gate G1": g1_phrase,
        },
    )

    built = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    head = git("log", "-1", "--format=%h", "--", ".") or "uncommitted"
    dirty = ", with uncommitted changes" if git("status", "--porcelain", "--", ".") else ""

    # ---------------- summary
    readout_list = "; ".join(f"{task_claims[task]}" for task in TASKS)
    summary = f"""
<p><strong>Verdict, by the rule fixed before any result: Betti-curve fPCA regression (arm F)
{markdown(verdict_phrase)}.</strong> Δ is the paired difference between F and arm K, <code>PD_kernel</code>
reimplemented from upstream's code (read, never run), signed so that Δ > 0 favours F; its 95% interval is a
percentile bootstrap of {boot["n_boot"]:,} resamples. Mean Δ and 95% interval: {html.escape(readout_list)}. Each
interval lies below 0, which the rule reads as below <code>PD_kernel</code>.</p>
<p><strong>The comparison stands on the paper's own footing.</strong> The reimplementation reproduces upstream's
published results, the {g1_values} values behind the paper's Tables 1–3, {html.escape(g1_phrase)}, against gate G1's
tolerance of {scientific(g1["tolerance"], 0)}, so the verdict does not rest on a weaker reimplementation. All
{n_gates} gates passed.</p>
<p><strong>Exploratory, no verdict: where F's gap comes from.</strong> Arm B runs K's estimator on the Betti curves'
L² distances, so Δ(B, K) is the representation's part of the gap and Δ(F, B) the estimator's; their means add up to
Δ(F, K). {html.escape(explore["interp"])} {html.escape(explore["curv"])} {html.escape(explore["fluid"])}</p>
<p><strong>For information, not a readout: arm F fails badly in a few near-saturated fits.</strong> In fluid,
{html.escape(lower_first(fail_phrases["the failed cells"]))}; {html.escape(fail_phrases["their (K0, K1)"])}. The fluid
readout by training size shows it: {html.escape(fail_phrases["the fluid readout by size"])}. They make the size of
fluid's mean Δ, not its sign: {html.escape(fail_phrases["the sign"])}. In interpolation,
{html.escape(lower_first(interp_phrases["the interpolation median"]))};
{html.escape(interp_phrases["the interpolation size"])}.</p>
"""

    # ---------------- section 1: the question
    rule_text = html.escape(readouts["rule"])
    question = f"""
<p>Simon's request (2026-11-24), with the paper's PDF: "Benchmark fPCA regression using inner product space of betti
curves against their reported performances". Then: "whether Betti fPCA regression beats their kernel; download the
repo. report in topocyte analyses." He chose approach B of <code>design.md</code>: the head-to-head below, with an
exploratory arm.</p>
<p><strong>The paper.</strong> S. Aggarwal, J. Nichols and K. Turner, <em>Kernel regression using persistence
diagrams</em>, arXiv:2611.25295. Its estimator, <code>PD_kernel</code>, is Nadaraya–Watson regression with a Gaussian
kernel on the distance W2(H0) + W2(H1) between persistence diagrams, its bandwidth chosen by leave-one-out (LOO)
cross-validation. The paper compares it with persistence images (<code>PI_kernel</code>) and, for curvature, with the
reported results of {n_turkes} methods of Turkeš, Montúfar and Otter (2022). Its code and data, <em>upstream</em>,
were cloned at <code>{c.UPSTREAM_SHA}</code> ({upstream_date}) and read as data only: no upstream code ran, and its
pickles went through a whitelist unpickler.</p>
<p><strong>Three arms see the same diagrams, training sets and test sets:</strong></p>
<ul>
<li><strong>K, <code>PD_kernel</code> reimplemented:</strong> Nadaraya–Watson on W2(H0) + W2(H1), transcribed from
upstream's code, which was read and never run. Each W2 is exact (an optimal assignment, scipy); fluid uses upstream's
cached distances, which gate G2 checked. The bandwidth is the first LOO minimizer over {c.N_BANDWIDTHS} values in
[0.1, 2] × the median positive training distance.</li>
<li><strong>F, Betti FPCR (primary):</strong> the reduced Betti curves of H0 and H1 (topocyte's
<code>BettiCurve</code>, finite bars only), topocyte's exact fPCA per dimension fitted on the training curves
(normalization "none"), the test curves projected exactly, then ordinary least squares of the target on the first K0
H0 scores and K1 H1 scores, with an intercept. (K0, K1) is chosen on the training set by closed-form LOO among
0 ≤ K_d ≤ min({c.K_MAX}, components), K0 + K1 ≤ n − 2, every leverage below 1 − {scientific(1 - c.LEVERAGE_CAP, 0)};
(0, 0) predicts the training mean. The fluid bars are shifted by {c.FLUID_SHIFT:g} so that none lies below 0, where
topocyte's inner product does not reach (gate G4). No curve is evaluated on a grid.</li>
<li><strong>B, Betti NW (exploratory):</strong> K's estimator on L2(H0) + L2(H1), the exact L² distances between the
same Betti curves. Against K it isolates the representation, against F the estimator.</li>
</ul>
<p><strong>The tasks</strong>, the paper's Tables 1–3:</p>
<ul>
<li><strong>Interpolation:</strong> clouds of {inp._INTERP_POINTS} points (cos θ + U, (1 − t) cos θ + t sin θ + U), the
target t; for each training size n in {c.INTERP_SIZES[0]}, {c.INTERP_SIZES[1]}, …, {c.INTERP_SIZES[-1]},
{c.INTERP_RUNS} runs, each with its own training set and {c.INTERP_N_TEST} test clouds: {
        len(interp_rows) * c.INTERP_RUNS
    } datasets. R² on the test clouds.</li>
<li><strong>Curvature:</strong> {clouds["train"].size} training clouds and {clouds["test"].size} test clouds of
{listing(sorted(points))} points from the unit disk on a surface of constant curvature K, the target K; one dataset.
The mean squared error (MSE) on the test clouds.</li>
<li><strong>Fluid:</strong> {n_fluid} simulations of a 2D unsteady flow ensemble, the target ln Re; each sample's
diagrams are the vorticity's sublevel persistence (upstream's), at times t = {
        listing(f"{t:g}" for t in c.FLUID_TIMES)
    }. Split s tests on {c.FLUID_N_TEST} simulations and trains on n = {listing(c.FLUID_SIZES)} of the
others; the paper runs splits 0–{c.PAPER_SPLITS - 1}, this study {c.FLUID_SPLITS} (0–{c.FLUID_SPLITS - 1}), all of
which feed the fluid readout. R² on the test simulations.</li>
</ul>
<p><strong>The rule, fixed in advance,</strong> as <code>readouts.json</code> records it: {rule_text}</p>
"""

    # ---------------- section 2: the verdicts
    verdict_rows = [
        [
            TASK_NAME[task],
            f"Δ = {definition(task, 'F', 'K')}",
            sig(primary[task]["mean"], plus=True),
            f"[{sig(primary[task]['lo'], plus=True)}, {sig(primary[task]['hi'], plus=True)}]",
            f"<strong>{html.escape(VERDICT_TEXT[verdicts[task]])} <code>PD_kernel</code></strong>",
        ]
        for task in TASKS
    ]
    delta_defs = html.escape("; ".join(f"{TASK_NAME[task]}: {readouts['delta'][task]}" for task in TASKS))
    verdict_section = f"""
{
        figure(
            fig_deltas(readouts),
            "The paired differences per task, with 95% intervals",
            "Each task's mean paired difference Δ with its 95% interval: the primary pair F against K, the one the "
            "rule reads, and the exploratory pairs B against K and F against B. Δ(X, Y) is signed so that Δ > 0 "
            "favours X: in R² for interpolation and fluid, in MSE for curvature (Y's MSE minus X's).",
        )
    }
{table(["task", "Δ(F, K)", "mean Δ", "95% interval", "verdict"], verdict_rows)}
<p class="note">The definitions and interval recipes, as <code>readouts.json</code> records them: {delta_defs}.</p>
"""

    # ---------------- section 3: the exploratory pairs
    explore_rows = [
        [
            TASK_NAME[task],
            delta_text(fk[task]),
            delta_text(bk[task]),
            side(bk[task]),
            delta_text(fb[task]),
            side(fb[task]),
        ]
        for task in TASKS
    ]
    explore_section = f"""
<p>Arm B is <code>PD_kernel</code>'s estimator on the Betti curves' L² distances. Δ(B, K) therefore compares two
representations under one estimator, and Δ(F, B) two estimators on one representation; for the means, Δ(F, K) =
Δ(F, B) + Δ(B, K) (asserted). The intervals come from the same recipes and carry no verdict.</p>
{
        table(
            ["task", "Δ(F, K)", "Δ(B, K): representation", "its interval", "Δ(F, B): estimator", "its interval"],
            explore_rows,
        )
    }
<ul>
<li>{html.escape(explore["interp"])}</li>
<li>{html.escape(explore["curv"])}</li>
<li>{html.escape(explore["fluid"])}</li>
</ul>
"""

    # ---------------- section 4: interpolation
    t1_rows = [
        [
            str(row["n_train"]),
            mean_sd(row["reported"]["PD_kernel"]),
            mean_sd(row["reported"]["PI_kernel"]),
            *(mean_sd(row[arm]) for arm in ARMS),
        ]
        for row in interp_rows
    ]
    by_size_rows = [
        [str(b["n_train"]), *(delta_text(pair(readouts, name)["interp"]["by_size"][i]) for name in PAIRS)]
        for i, b in enumerate(fk["interp"]["by_size"])
    ]
    interp_section = f"""
<p>In upstream's code, and so here, t = 0 gives a line and t = 1 a circle, the noise-free points (cos θ, cos θ)
and (cos θ, sin θ); the paper's text swaps the two. Each arm's R² per training size, mean ± SD (ddof 1) over the
{c.INTERP_RUNS} runs, beside the paper's reported values. K's columns equal the reported <code>PD_kernel</code> to the
last bits (gate G1).</p>
{
        figure(
            fig_interp(readouts),
            "Interpolation: R² per training size per arm, and the paired differences per size",
            "Left, as the paper's Figure 6: each arm's R² on the test clouds, mean ± SD over the runs of each size, "
            "beside the reported <code>PD_kernel</code> (hollow circles, on K's) and <code>PI_kernel</code> "
            "(grey, dashed). Right: each pair's mean Δ per size, its 95% interval resampling that size's runs. The "
            "series are dodged sideways so that their bars do not overlap.",
        )
    }
{table(["n", "PD_kernel, reported", "PI_kernel, reported", "K", "F", "B"], t1_rows)}
<p>By size, Δ(F, K)'s interval lies below 0 at n = {listing(sizes_below)} and contains 0 at the other
{len(sizes_level)} sizes; the rule reads only the readout over all {len(interp_rows) * c.INTERP_RUNS} datasets. F's
mean R² lies above the reported <code>PI_kernel</code>'s at every size, by {num(min(over_pi), 3)} to
{num(max(over_pi), 3)}.</p>
{table(["n", "Δ(F, K)", "Δ(B, K)", "Δ(F, B)"], by_size_rows)}
<h3>For information: arm F's interpolation gap</h3>
<p>Not a readout. {html.escape(interp_phrases["the interpolation median"])}. {
        html.escape(interp_phrases["the interpolation size"])
    }: fits that use nearly all the components the eligibility rule allows, as in fluid at
n = {n_small} (section 6). Over all sizes, {html.escape(interp_phrases["the worst datasets"])}. The exploratory
Δ(F, B), whose interval over all {interp_delta.size} datasets contains 0, is a cancellation: by size,
{html.escape(cancellation)}.</p>
"""

    # ---------------- section 5: curvature
    point_net_input = "the stored coordinates as 3D clouds, the 2D ones padded with z = 0"
    curv_rows = [
        [
            html.escape(method),
            num(value),
            "Turkeš et al., as reported",
            point_net_input if method == POINT_NET else "geodesic distance matrices",
        ]
        for method, value in turkes.items()
    ]
    rips_input = "the Rips filtration of the stored coordinates"
    curv_rows.append(["PD_kernel", num(curv["reported"]["PD_kernel"]), "the paper, as reported", rips_input])
    curv_rows += [[ARM_NAME[arm], num(mse[arm]), "this study", rips_input] for arm in ARMS]
    curv_section = f"""
<p>One training set of {clouds["train"].size} clouds, K = linspace(−2, 2, {clouds["train"].size}), and
{clouds["test"].size} test clouds. Arm K's MSE equals the reported <code>PD_kernel</code>'s bit for bit; F's is
{mse["F"] / mse["K"]:.1f} times K's, and B's lies between them. F chose (K0, K1) = {k_curv}.</p>
{table(["method", "MSE on the test clouds", "source", "input"], curv_rows)}
<p>Of the {n_turkes} reported methods of Turkeš et al., {n_geodesic} used geodesic distance matrices rather than these
inputs, and {POINT_NET} {point_net_input} (see Caveats). F's MSE lies below those of {listing(turkes_higher)}, and
above those of {listing(turkes_lower)}; K's and B's lie below all {n_turkes}.</p>
{
        figure(
            fig_curvature(arms, mse),
            "Curvature: predicted against true curvature, per arm",
            f"Each arm's predictions of the {n_test_curv} test clouds against their true curvature K; the grey "
            "diagonal is a perfect prediction. Every arm predicts the sign of K right for every test cloud: their "
            "errors lie within each sign.",
        )
    }
"""

    # ---------------- section 6: fluid
    t3_rows = [
        [
            f"{row['t']:g}",
            str(row["n_train"]),
            mean_sd(row["reported"]),
            *(mean_sd(row["paper_splits"][arm]) for arm in ARMS),
            *(mean_sd(row["all_splits"][arm]) for arm in ARMS),
        ]
        for row in tables["fluid"]
    ]
    time_rows = [
        [f"{b['t']:g}", *(delta_text(pair(readouts, name)["fluid"]["by_time"][i]) for name in PAIRS)]
        for i, b in enumerate(fk["fluid"]["by_time"])
    ]
    size_delta_rows = [
        [str(b["n_train"]), *(delta_text(pair(readouts, name)["fluid"]["by_size"][i]) for name in PAIRS)]
        for i, b in enumerate(fk["fluid"]["by_size"])
    ]
    size_delta_rows.append([f"splits 0–{c.PAPER_SPLITS - 1} alone", *(delta_text(paper[name]) for name in PAIRS)])
    assert all(side(b) == "below 0" for b in fk["fluid"]["by_time"])
    off_axis = {  # the figure's means below its axis, which its caption names: arm F's at the smallest size only
        (key, arm, row["n_train"])
        for row in tables["fluid"]
        for key, _ in FLUID_ROWS
        for arm in ARMS
        if row[key][arm]["mean"] < FLUID_YLIM[0]
    }
    assert off_axis == {(key, "F", n_small) for key, _ in FLUID_ROWS}
    assert all(
        row[key]["F"]["mean"] < FLUID_YLIM[0]
        for row in tables["fluid"]
        if row["n_train"] == n_small
        for key, _ in FLUID_ROWS
    )  # at every time
    fluid_section = f"""
<p>Each arm's R² per time and training size, mean ± SD (ddof 1, as upstream's) over the paper's splits
0–{c.PAPER_SPLITS - 1}, which is the paper's Table 3, and over all {c.FLUID_SPLITS} splits, which the readout uses.
Upstream reports <code>PD_kernel</code> only.</p>
{
        figure(
            fig_fluid(readouts),
            "Fluid: R² per training size per arm, one panel per time",
            f"As the paper's Figure 9: each arm's R² on the {c.FLUID_N_TEST} test simulations, mean ± SD over the "
            f"splits; above, the paper's splits 0–{c.PAPER_SPLITS - 1} beside the reported <code>PD_kernel</code> "
            f"(hollow circles, on K's); below, all {c.FLUID_SPLITS} splits. A mean below {num(FLUID_YLIM[0], 1)} "
            f"is marked at the bottom with its value: arm F's at n = {n_small}, at every time.",
        )
    }
{
        table(
            [
                "t",
                "n",
                f"PD_kernel, reported <span class='unit'>{PAPER}</span>",
                *(f"{arm} <span class='unit'>{PAPER}</span>" for arm in ARMS),
                *(f"{arm} <span class='unit'>{ALL}</span>" for arm in ARMS),
            ],
            t3_rows,
        )
    }
<h3>The paired differences by time and by training size</h3>
<p>Descriptive, with the readout's recipe (ruling 13 in <code>progress.md</code>): by time, the interval resamples
the {c.FLUID_SPLITS} splits, each by its mean Δ over the sizes; by size, each by its mean Δ over the times; the last
row is the readout on the paper's splits alone. Δ(F, K)'s interval lies below 0 at every time and every size.</p>
{table(["t", "Δ(F, K)", "Δ(B, K)", "Δ(F, B)"], time_rows)}
{table(["training size n", "Δ(F, K)", "Δ(B, K)", "Δ(F, B)"], size_delta_rows)}
<h3>For information: arm F's failures at n = {n_small}</h3>
<p>Not a readout. {html.escape(fail_phrases["the failed cells"])}; {html.escape(fail_phrases["their (K0, K1)"])}. The
worst is {html.escape(worst_text)}. Over all {n_cells:,} fluid cells F's R² has median {num(medians["F"], 3)} (K
{num(medians["K"], 3)}, B {num(medians["B"], 3)}); it lies below 0 in {below0} cells and below −1 in {below1}. The
readout by size shows the failures: {html.escape(fail_phrases["the fluid readout by size"])}. They make the size of
fluid's mean Δ, not its sign: {html.escape(fail_phrases["the sign"])}.</p>
{
        figure(
            fig_failures(total[:, :, 0], failed[:, :, 0], cap, n_small),
            f"Arm F's chosen K0 + K1 at n = {n_small}, the failed cells stacked",
            f"The K0 + K1 that arm F chose in the {total[:, :, 0].size} fluid cells of training size n = {n_small} "
            f"({len(c.FLUID_TIMES)} times × {c.FLUID_SPLITS} splits), the cells whose R² lies below "
            f"{num(FAIL_BELOW, 0)} in F's colour. A model with K0 + K1 near n − 2 leaves the LOO few residual degrees "
            "of freedom.",
        )
    }
"""

    # ---------------- section 7: the chosen (K0, K1)
    chosen_section = f"""
<p>Arm F's (K0, K1), chosen on each training set by LOO. Curvature, one training set: {k_curv}. Interpolation:
{len(chosen["interp"])} distinct pairs over {k_interp.shape[0]} datasets, ({top["interp"]["k0"]},
{top["interp"]["k1"]}) the most frequent ({top["interp"]["count"]}); {interp_at_cap} datasets chose K0 + K1 = n − 2.
Fluid: {len(chosen["fluid"])} distinct pairs over {n_cells:,} cells, ({top["fluid"]["k0"]}, {top["fluid"]["k1"]}) the
most frequent ({top["fluid"]["count"]:,}); (0, 0), the training mean, in {zero_fluid}.</p>
{
        figure(
            fig_chosen(chosen),
            "How often arm F chose each (K0, K1)",
            "The count of each (K0, K1) that arm F chose, on a log colour scale; blank: never chosen. Left, the "
            f"{k_interp.shape[0]} interpolation datasets; right, the {n_cells:,} fluid cells. K_d is at most "
            f"{c.K_MAX} by design.",
        )
    }
{
        table(
            [
                "fluid training size n",
                "median K0 + K1",
                "largest K0 + K1",
                "cells at n − 2",
                "cells at (0, 0)",
                f"cells with F's R² below {num(FAIL_BELOW, 0)}",
            ],
            size_rows,
        )
    }
"""

    # ---------------- section 8: the gates and the rulings
    gate_rows = [
        [
            "G1",
            f"arm K against upstream's published results: {g1_values} values (interpolation's mean and SD per "
            f"size, curvature's MSE, fluid's R² of {PAPER})",
            scientific(g1_worst),
            f"{scientific(g1['tolerance'], 0)} absolute",
            f"{scientific(g1_exact)} with the CSVs read exactly",
        ],
        [
            "G2",
            f"this study's W2 against upstream's cached fluid distances: {g2['checks']} checks of {g2['pairs']} "
            f"same-time pairs, {g2['large_checks']} with more than {mat.G2_LARGE_BARS:,} finite bars on both sides",
            scientific(g2["worst_relative"]),
            f"{scientific(g2['tolerance'], 0)} relative",
            "",
        ],
        [
            "G3",
            f"the projected test scores against <code>FPCA.transform</code>: {len(g3['cells'])} cells × "
            f"dimensions, the first min({g3['k_max']}, rank) columns",
            scientific(g3_worst),
            f"{scientific(g3['tolerance'], 0)} of the largest score",
            f"all columns: {scientific(g3_all)}",
        ],
        [
            "G4",
            f"the fluid Gram matrices at shifts {listing(f'{s:g}' for s in g4['shifts'])}, per time and dimension",
            scientific(g4_worst),
            f"{scientific(g4['tolerance'], 0)} of the largest entry",
            f"entry by entry: {scientific(g4_entry)} of the entry",
        ],
        [
            "G5",
            f"inputs: the {g5['interpolation_datasets']} interpolation datasets equal their seed-0 regeneration; "
            f"{g5_clouds:,} clouds' Rips diagrams equal <code>ripser</code>'s as pairs; "
            f"{g5['fluid_diagrams_expected']:,} fluid diagrams present",
            "none differ",
            "exact",
            "",
        ],
    ]
    g3_where = f"fluid, t = {g3_cell['time']:g}, H{g3_cell['h']}"
    g4_where = f"t = {g4_cell['time']:g}, H{g4_cell['h']}"
    gates_section = f"""
{table(["gate", "what it checks", "worst", "tolerance", "for information"], gate_rows, "wrap1")}
<p>Every gate passed before any readout, and arms F and B ran only after K had passed G1. Two gates needed a reference
that <code>design.md</code> did not name; the rulings below (<code>progress.md</code>, Rulings during the build, items
7–9) were confirmed by Simon on 2026-11-25:</p>
<ul>
<li><strong>G3 compares the first min(K_MAX, rank) score columns per dimension,</strong> the only ones arm F can use
(K_d ≤ {c.K_MAX}), relative to the largest score over them: worst {scientific(g3_worst)}. Over all columns the worst is
{scientific(g3_all)} ({g3_where}), kept in <code>arms.json</code> for information. <strong>A follow-up for topocyte,
not for this study:</strong> topocyte's <code>FPCA.transform</code> is inaccurate in near-null components,
{html.escape(g3_phrase)}, where it also misses its own fitted scores (ruling 8). Arm F's scores come from the exact
projection on the Gram matrix, and it uses none of the columns past the first {c.K_MAX}.</li>
<li><strong>G4 is relative to each matrix's largest entry:</strong> worst {scientific(g4_worst)} ({g4_where}). Entry by
entry the two shifts differ by up to {scientific(g4_entry)} of the entry, on entries whose bars overlap over tiny
lengths: the rounding of the shifted endpoints, which no shift avoids, not a part of a bar clipped below 0, which G4 is
for (ruling 7 gives the measurements, against the exact full-line values computed in rationals).</li>
</ul>
<p><code>progress.md</code> holds all {rulings} rulings of the build, each with its reason; none changes a number of
<code>design.md</code>. Ruling 14 records that G3 checks one curvature cell, the task's single training set, where
<code>design.md</code> asked for at least 3 cells per task.</p>
"""

    # ---------------- section 9: the caveats
    caveats_section = f"""
<ul>
<li><strong>The curvature inputs.</strong> The input of <code>PD_kernel</code>, and so of every arm here, is the
Euclidean Rips filtration of upstream's stored coordinates: 3D Cartesian coordinates on the sphere of radius 1/√K for
K &gt; 0 ({int(positive.sum())} of the {y_curv.size} clouds), 2D coordinates in the Poincaré disk of radius 1/√−K for
K &lt; 0 ({int(negative.sum())}); both asserted on the clouds. Of the {n_turkes} Turkeš et al. methods of Table 2,
{n_geodesic} used geodesic distance matrices instead, so their rows compare other inputs; {POINT_NET} read the stored
coordinates themselves as 3D clouds, the 2D ones padded with z = 0 (upstream's
<code>curvature/Turkes_Experiment/turkes_curvature_experiment.py</code>, read as text). Every arm predicts the sign of
K right for all {n_test_curv} test clouds.</li>
<li><strong>The essential bars that F and B drop.</strong> topocyte's reduced Betti curves hold finite bars only.
In the Rips tasks the only essential bar is H0's, born at 0 in every cloud (the filtration ends in one simplex over
all the points), so nothing is lost. In fluid every diagram has one essential bar in H0 and two in H1, whose births
W2 matches and compares, and which F and B never see; and some diagrams have no finite bar in a dimension at all, an
empty curve (per time, {listing(empty_h0)} in H0 and {listing(empty_h1)} in H1, of {n_fluid}).</li>
<li><strong>The single curvature training set.</strong> Curvature has one dataset, generated without a seed upstream:
its interval resamples the test clouds only, and says nothing of another training set.</li>
<li><strong>The {c.FLUID_SPLITS} fluid splits share the same {n_fluid} simulations.</strong> The fluid interval bounds
the mean Δ over random splits of these simulations, not over new simulations. On the paper's splits
0–{c.PAPER_SPLITS - 1} alone, Δ(F, K) is {delta_text(paper["F_vs_K"])}.</li>
<li><strong>The paper's swap of line and circle.</strong> The paper's text swaps the interpolation's two ends; in
upstream's code, and so here, t = 0 is a line and t = 1 a circle (section 4).</li>
</ul>
"""

    # ---------------- section 10: methods and run
    check_rows = [
        [
            html.escape(check["family"]),
            html.escape(check["check"]),
            *(
                "0" if value == 0 else scientific(value, 1 if key == "worst" else 0)
                for key, value in (("worst", check["worst"]), ("tolerance", check["tolerance"]))
            ),
        ]
        for check in verify["checks"]
    ]
    methods_section = f"""
<p>The stages, in the order they ran; each row is the run whose outputs this report reads (<code>progress.md</code>
lists the earlier, superseded runs). The wall time and peak RSS come from <code>/usr/bin/time -l</code> in
<code>logs/</code>; {listing(f"<code>{command}</code>" for command in untimed)} ran without it. Each json records
the md5 of the study's scripts its stage ran, and <code>verify_results.py</code> checks each one against the script in
the commit that last changed that json; where a script has changed since, the table names that commit.</p>
{table(["stage", "finished", "wall time", "peak RSS", "md5 its json names"], stage_rows)}
<p>Arm by arm, the seconds of the arms stage (<code>arms.json</code>); the fluid cells take nearly all of each arm's
time, and gate G3 took {seconds["g3"]:.1f} s:</p>
{
        table(
            [
                "arm",
                *(f"{TASK_NAME[task]} <span class='unit'>s</span>" for task in TASKS),
                "all <span class='unit'>s</span>",
            ],
            arm_seconds,
        )
    }
<p class="note">{html.escape(verify["platform"])}; Python {versions["python"]}, numpy {versions["numpy"]}, scipy
{versions["scipy"]}, pandas {versions["pandas"]}, ripser {versions["ripser"]}, scikit-learn
{versions["scikit-learn"]}, matplotlib {versions["matplotlib"]}. topocyte {c.TOPOCYTE_COMMIT}, from a clean checkout
first on <code>PYTHONPATH</code>; upstream {c.UPSTREAM_SHA} ({upstream_date}), a clean clone in
<code>upstream/</code>. <code>results/matrices.npz</code>: {npz_text["matrices.npz"]}; <code>results/arms.npz</code>:
{npz_text["arms.npz"]}; both sha256 sums are recorded in their jsons and in <code>verify.json</code>.</p>
<h3>Checks by other routes</h3>
<p><code>verify_results.py</code> ran at {verify["date"]}, after every stage, and passed {verify["checks_passed"]} of
{verify["checks_total"]} checks; its docstring says what each check takes another route for and what it does not
redo.</p>
{table(["family", "check", "worst", "tolerance"], check_rows, "wrap1")}
"""

    body = f"""
<h1>Betti-curve fPCA regression against the persistence-diagram kernel</h1>
<p class="subtitle">Functional principal component regression on topocyte's exact Betti-curve fPCA (arm F) against
<code>PD_kernel</code> of Aggarwal, Nichols and Turner (arm K, reimplemented from their code), with an exploratory
Nadaraya–Watson arm on Betti L² distances (arm B), on the paper's three tasks</p>
<p class="meta">Built {built} from the study folder at {html.escape(head)}{dirty}; {verify["checks_passed"]} checks of
<code>verify_results.py</code> pass.</p>
<div class="summary">{summary}</div>
<h2>1. The question, the arms and the rule</h2>
{question}
<h2>2. The verdicts</h2>
{verdict_section}
<h2>3. Exploratory: the representation or the estimator?</h2>
{explore_section}
<h2>4. Interpolation: the paper's Table 1 and Figure 6</h2>
{interp_section}
<h2>5. Curvature: the paper's Table 2</h2>
{curv_section}
<h2>6. Fluid: the paper's Table 3 and Figure 9</h2>
{fluid_section}
<h2>7. Arm F's chosen (K0, K1)</h2>
{chosen_section}
<h2>8. The gates and the rulings that shaped them</h2>
{gates_section}
<h2>9. Caveats</h2>
{caveats_section}
<h2>10. Methods and run</h2>
{methods_section}
"""
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Betti FPCR benchmark</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )
    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
