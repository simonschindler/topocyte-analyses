// The viewer's page: the slide list filtered by pathology group, an OpenSeadragon canvas (the slide's H&E
// tiles, or a blank canvas of the cloud's size) and the WebGL layer of the slide's cells.
import { CurvesPanel, binOf, classCounts, valueClassCounts } from "/static/curves.js";
import { MatrixPanel } from "/static/matrix.js";
import { PointsOverlay } from "/static/overlay.js";
import { CLUSTERS, DENSITY, PALETTE, colourScale, rgb } from "/static/palette.js";

const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
const CELL_UM = 10; // a cell's drawn diameter once zoomed in far enough
const SCALE_MAX_CSS = 150; // the scale bar's longest length on screen
const BACKGROUND = 15; // the code of the cells that are not centres, in a TDA clustering
const GREY = { light: "#c3c2b7", dark: "#52514e" }; // those cells, drawn behind the centres
const CENTRE_SIZE = 2.5; // TDA centres are drawn this much larger than the cells
const API = 3; // the /api/meta "api" this page is written for (serve.py's API)
// The layers of each cell's k nearest other cells, all four from one request (/api/neighbourhood/<id>).
const NEIGHBOURHOOD = ["homotypic", "entropy", "excess", "kl"];
const USES_NEIGHBOURS = [...NEIGHBOURHOOD, "coloc"]; // the layers the k and label controls apply to
const NONE = 15; // the code of a cell without a value (no Voronoi cell) when the slide's percentiles cut the bins
const LOWEST = -3.4e38; // below every value: where a cell has none, and the threshold that hides nothing
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString("en-US");

const state = {
  meta: null, // /api/meta
  slide: null, // the slide being shown or loading
  info: null, // /api/slide/<id> of the shown slide
  counts: null, // cells per type code on the shown slide
  types: null, // the shown slide's type code per cell
  classes: null, // the shown slide's class code per cell, for the class curves
  clusters: null, // /api/cluster/<id> of the shown slide, with its labels decoded
  density: null, // the shown slide's density bin per cell (/api/density/<id>)
  rho: null, // the shown slide's 10-NN density per cell, float32
  densityCounts: null, // cells per density bin on the shown slide
  voronoi: null, // the shown slide's Voronoi density codes and cells per code (/api/voronoi/<id>), once fetched
  neighbourhood: null, // the shown slide's NEIGHBOURHOOD layers for one k and label set, once fetched
  coloc: null, // the shown slide's colocalization excess for one k, label set, neighbour type and score
  colocSummary: null, // its pooled excess by focal and neighbour type, for the matrix panel
  matrixClosed: false, // the matrix panel was closed for this choice of the colouring
  layer: "types", // "types" | "classes" | "clusters" | "density" | "voronoi", or one of NEIGHBOURHOOD
  scale: { map: "viridis", mode: "fixed", from: 5, to: 95 }, // the value layers' colour map and bins
  thresholdSource: null, // the values the overlay's threshold compares (state.rho, or a value layer's)
  hidden: {
    types: new Set(), classes: new Set(), clusters: new Set(), density: new Set(), voronoi: new Set(),
    homotypic: new Set(), entropy: new Set(), excess: new Set(), kl: new Set(), coloc: new Set(),
  },
};
let viewer = null;
let overlay = null;
let curves = null; // the class curves panel
let matrix = null; // the colocalization matrix panel

function setStatus(text) { $("status").textContent = text; }

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return response;
}

// --- theme and colours ------------------------------------------------------------------------------------

function storedTheme() {
  try { return localStorage.getItem("theme"); } catch { return null; }
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("theme", theme); } catch { /* private window: the theme is not remembered */ }
  applyColors();
  curves?.setColours(classColours(), state.meta.classes);
  renderMatrix();
}

// The class colours of the page's theme, for the curves panel (it sits on the page surface, not on the H&E).
function classColours() {
  const theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  return state.meta.classes.map((name) => PALETTE.classes[theme][name]);
}

function showCurves(open) {
  $("curves").hidden = !open;
  $("curves-open").hidden = open;
  try { localStorage.setItem("curves", open ? "shown" : "hidden"); } catch { /* not remembered */ }
}

// Light colours on the light theme and on H&E, dark colours on the dark canvas.
function paletteMode() {
  const onSlide = Boolean(state.info?.wsi) && $("hne").checked;
  return onSlide || document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

// A density bin's range in cells per mm²: below the second decile, between two deciles, or from the top one on.
function densityRange(j, joiner = "–") {
  const d = state.meta.density.deciles;
  const per = (v) => fmt(Math.round(v));
  if (j === 0) return `< ${per(d[1])}`;
  if (j === d.length - 1) return `≥ ${per(d[j])}`;
  return `${per(d[j])}${joiner}${per(d[j + 1])}`;
}

// A bin of a value in [0, 1]: from j/10 up to (j + 1)/10, the last one with 1 itself.
function unitRange(j, joiner = "–") {
  return `${(j / 10).toFixed(1)}${joiner}${j === 9 ? "1" : ((j + 1) / 10).toFixed(1)}`;
}

function unitTitle(j) {
  const what = state.layer === "homotypic" ? "Homotypic fraction" : "Local entropy";
  const top = j === 9 ? "1 included" : `${((j + 1) / 10).toFixed(1)} not included`;
  return `${what} from ${(j / 10).toFixed(1)} to ${top}.`;
}

// A homotypic excess bin: between two edges, below the first or from the last on, with its sign.
function excessRange(j, joiner = " to ") {
  const edges = state.meta.neighbourhood.excess_edges;
  const s = (v) => (v > 0 ? `+${v.toFixed(1)}` : `−${(-v).toFixed(1)}`);
  if (j === 0) return `< ${s(edges[0])}`;
  if (j === edges.length) return `≥ ${s(edges[j - 1])}`;
  return `${s(edges[j - 1])}${joiner}${s(edges[j])}`;
}

// A KL divergence bin in nats: from j to j + 1 steps, the last one open above.
function klRange(j, joiner = "–") {
  const step = state.meta.neighbourhood.kl_step;
  const v = (x) => String(Number((x * step).toFixed(2)));
  return j === 9 ? `≥ ${v(9)}` : `${v(j)}${joiner}${v(j + 1)}`;
}

function densityTitle(j) {
  const kept = j === 0 ? "kept only in density slice 0, every cell" : `kept in density slices 0 to ${j}`;
  return `Cohort decile ${j + 1} of the 10-NN density: ${densityRange(j, " to ")} cells per mm²; ${kept}.`;
}

function voronoiTitle(j) {
  return `Voronoi density in cohort decile ${j + 1} of the 10-NN density: ${densityRange(j, " to ")} cells per mm².`;
}

function excessTitle(j) {
  const middle = state.meta.neighbourhood.excess_edges.length / 2;
  const side = j === middle ? "about chance" : j < middle ? "fewer of its own kind than chance"
    : "more of its own kind than chance";
  return `Homotypic excess ${excessRange(j, " up to ")}: ${side}.`;
}

function colocTitle(j) {
  return `Colocalization excess ${excessRange(j, " up to ")}.`;
}

function klTitle(j) {
  return `KL divergence of the neighbours' mix from the slide's: ${klRange(j, " up to ")} nats.`;
}

// The value layers, each cell coloured by a number. Fixed bins: the server's codes, with their labels and titles.
// digits: the fewest decimals to print; centre: where a diverging map puts its neutral; map: its own map, which
// the colour menu's "by layer" gives (blue where none).
const VALUES = {
  density: {
    what: "10-NN density", name: "10-NN density", unit: "cells per mm²", digits: 0, bins: 10, label: densityRange,
    title: densityTitle,
  },
  voronoi: {
    what: "Voronoi density", name: "Voronoi density", unit: "cells per mm²", digits: 0, bins: 10,
    label: densityRange, title: voronoiTitle, noneLabel: "no Voronoi cell",
    noneTitle: "On the convex hull of the slide's cells, where the Voronoi cell is unbounded, or at the position "
      + "of another cell, which shares it.",
    noneCode: () => state.meta.density.voronoi_none,
  },
  homotypic: {
    what: "Homotypic fraction", name: "homotypic fraction", unit: "", digits: 2, bins: 10, label: unitRange,
    title: unitTitle,
  },
  entropy: {
    what: "Local entropy", name: "local entropy", unit: "", digits: 2, bins: 10, label: unitRange, title: unitTitle,
  },
  excess: {
    what: "Homotypic excess", name: "homotypic excess", unit: "", digits: 2, bins: 11, label: excessRange,
    title: excessTitle, centre: 0,
    map: "diverging", signed: true,
  },
  kl: {
    what: "KL divergence", name: "KL divergence", unit: "nats", digits: 2, bins: 10, label: klRange, title: klTitle,
  },
  coloc: {
    what: "Colocalization excess", name: "colocalization excess", unit: "", digits: 2, bins: 11, label: excessRange,
    title: colocTitle, centre: 0, map: "diverging", signed: true, noneLabel: "no estimate",
    noneTitle: "The null gives this count no variance: the type is missing among the slide's other cells, or it "
      + "is all of them.",
    noneCode: () => state.meta.colocalization.none,
  },
};
const VALUE_LAYERS = Object.keys(VALUES);

// The shown slide's values of a value layer: float32, NaN where a cell has none.
function valuesOf(layer) {
  if (layer === "density") return state.rho;
  if (layer === "voronoi") return state.voronoi?.values ?? null;
  if (layer === "coloc") return state.coloc?.values ?? null;
  return state.neighbourhood?.[layer]?.values ?? null;
}

// The server's fixed bins of a value layer: its codes and the cells per code.
function fixedOf(layer) {
  if (layer === "density") return state.density && { codes: state.density, counts: state.densityCounts };
  if (layer === "voronoi") return state.voronoi;
  if (layer === "coloc") return state.coloc;
  return state.neighbourhood?.[layer] ?? null;
}

function mapOf(layer) {
  return state.scale.map === "auto" ? (VALUES[layer].map ?? "blue") : state.scale.map;
}

const sortedValues = new WeakMap(); // a layer's values -> their finite values, sorted

function sortedOf(values) {
  if (!sortedValues.has(values)) sortedValues.set(values, values.filter((v) => !Number.isNaN(v)).sort());
  return sortedValues.get(values);
}
const scaledBins = new WeakMap(); // a layer's values -> the last bins cut at their percentiles, with its settings

// The q-quantile of sorted values, interpolated between neighbours as numpy's default.
function quantile(sorted, q) {
  const x = q * (sorted.length - 1);
  const i = Math.floor(x);
  return i + 1 < sorted.length ? sorted[i] + (x - i) * (sorted[i + 1] - sorted[i]) : sorted[i];
}

// A value layer's bins: the server's fixed ones, or n equal steps between the slide's two percentiles, the first
// and last open (11 for a diverging map, so that one step sits in the middle, else 10). A diverging map on a layer
// with a centre widens the range to be even about it, so the neutral stays there.
function binsOf(layer) {
  if (state.scale.mode === "fixed") {
    const f = fixedOf(layer);
    const none = VALUES[layer].noneCode?.() ?? null;
    return f && { n: VALUES[layer].bins, codes: f.codes, counts: f.counts, none };
  }
  const values = valuesOf(layer);
  if (!values) return null;
  const diverging = mapOf(layer) === "diverging";
  const n = diverging ? 11 : 10;
  const { from, to } = state.scale;
  const key = `${from}|${to}|${n}|${diverging}`;
  const cached = scaledBins.get(values);
  if (cached?.key === key) return cached.bins;
  const sorted = sortedOf(values);
  let lo = sorted.length ? quantile(sorted, from / 100) : 0;
  let hi = sorted.length ? quantile(sorted, to / 100) : 0;
  const { centre } = VALUES[layer];
  if (diverging && centre !== undefined) {
    const half = Math.max(centre - lo, hi - centre);
    [lo, hi] = [centre - half, centre + half];
  }
  const width = (hi - lo) / n;
  const codes = new Uint8Array(values.length);
  const counts = new Array(NONE + 1).fill(0);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const j = Number.isNaN(v) ? NONE : width > 0 ? Math.min(n - 1, Math.max(0, Math.floor((v - lo) / width))) : n >> 1;
    codes[i] = j;
    counts[j] += 1;
  }
  const bins = { n, codes, counts, none: VALUES[layer].noneLabel ? NONE : null, lo, hi, width, from, to };
  scaledBins.set(values, { key, bins });
  return bins;
}

// A value as its layer prints it, with enough decimals to tell steps of the given width apart.
function valueText(layer, v, width) {
  const { digits, signed } = VALUES[layer];
  const places = Math.min(4, Math.max(digits, width > 0 ? Math.ceil(-Math.log10(width)) : 0));
  const text = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: places, maximumFractionDigits: places });
  if (Number(text.replaceAll(",", "")) === 0) return text;
  return v < 0 ? `−${text}` : signed ? `+${text}` : text;
}

function scaledRange(layer, bins, j, joiner = "–") {
  const edge = (i) => valueText(layer, bins.lo + i * bins.width, bins.width);
  if (j === 0) return `< ${edge(1)}`;
  if (j === bins.n - 1) return `≥ ${edge(j)}`;
  return `${edge(j)}${VALUES[layer].signed ? " to " : joiner}${edge(j + 1)}`;
}

function scaledTitle(layer, bins, j) {
  const { what, unit } = VALUES[layer];
  return `${what} ${scaledRange(layer, bins, j, " up to ")}${unit ? ` ${unit}` : ""}: step ${j + 1} of ${bins.n} `
    + `between the slide's percentiles ${bins.from} and ${bins.to}.`;
}

// The legend of a value layer: its bins in its colour map, and the cells without a value where there are such.
function valueEntries(mode) {
  const layer = state.layer;
  const spec = VALUES[layer];
  const bins = binsOf(layer);
  const scaled = state.scale.mode !== "fixed" && bins;
  const entries = colourScale(mapOf(layer), bins ? bins.n : spec.bins, mode).map((hex, j) => ({
    key: j,
    hex,
    n: bins ? bins.counts[j] : 0,
    label: scaled ? scaledRange(layer, bins, j) : spec.label(j),
    title: scaled ? scaledTitle(layer, bins, j) : spec.title(j),
  }));
  if (spec.noneLabel) {
    const none = bins?.none ?? spec.noneCode();
    const n = bins ? bins.counts[none] : 0;
    entries.push({ key: none, label: spec.noneLabel, hex: GREY[mode], n, title: spec.noneTitle });
  }
  return entries;
}

// What a value layer shows, and how its colours are cut.
function valueNote(layer) {
  const bins = binsOf(layer);
  const { um_per_px: um, k_nn: kNN } = state.meta.density;
  let text;
  let fixed;
  if (layer === "density") {
    text = `Each cell's ${kNN}-nearest-neighbour density in cells per mm² (the pathology study's density.npz, at `
      + `${um} µm per pixel).`;
    fixed = "In the cohort's ten deciles: the thresholds of the density slices.";
  } else if (layer === "voronoi") {
    text = "Each cell's Voronoi density, 1 / the area of its Voronoi cell among the slide's cells, in cells per mm² "
      + `(at ${um} µm per pixel, as the 10-NN density). ${fmt(bins ? bins.counts[bins.none] : 0)} cells have no `
      + "Voronoi cell of their own (grey): on the convex hull it is unbounded, and cells at one position share "
      + "theirs. Cells at the edge of the tissue have large Voronoi cells, so low densities.";
    fixed = "In the 10-NN density's bins, the cohort's ten deciles.";
  } else if (layer === "coloc") {
    const { k, by, of, score } = state.coloc;
    const kind = by === "types" ? "HistoPLUS type" : "class";
    const target = of === "own" ? `its own ${kind}` : `${state.meta.colocalization.labels[by][Number(of)]} cells`;
    text = `Each cell's hypergeometric count excess of ${target} among its ${k} nearest other cells: the count less `
      + "what random labelling of the slide's other cells gives, over the room it has: k (1 − q) on the gain scale, "
      + "k q on the loss scale, each focal × neighbour pair's scale fixed by the sign of its summed excess over the "
      + `${state.meta.colocalization.n_slides} cohort slides at this k. `
      + (score === "shrunk"
        ? "Shrunk toward the slide's pooled excess by empirical Bayes (DerSimonian–Laird between-cell variance), "
          + "so noisy estimates count less."
        : `Unshrunk: ${fmt(bins ? bins.counts[bins.none] : 0)} cells have no estimate (grey), where the null has no `
          + "variance.");
    fixed = "In bins of 0.2 about 0, −0.1 to +0.1.";
  } else {
    const { k, by, nLabels } = state.neighbourhood;
    const [kind, kinds] = by === "types" ? ["HistoPLUS type", "HistoPLUS types"] : ["class", "classes"];
    const step = state.meta.neighbourhood.kl_step;
    text = {
      homotypic: `Each cell's homotypic fraction: the share of its ${k} nearest other cells that are of its own `
        + `${kind} (1: inside a sheet of its own kind; about 0.5: on a border between two; 0: alone among others).`,
      entropy: `Each cell's local Shannon entropy: of the ${kinds} of its ${k} nearest other cells, divided by `
        + `log ${nLabels} for the ${nLabels} ${kinds} (Pielou's evenness; 0: all of one ${kind}; 1: every ${kind} `
        + "in equal shares).",
      excess: `Each cell's homotypic excess: the share of its ${k} nearest other cells of its own ${kind}, less the `
        + `slide's share of cells of that ${kind}. 0 is what chance gives; above, more of its own ${kind} around it `
        + "than the slide's mix; below, fewer.",
      kl: `Each cell's KL divergence: how far the mix of ${kinds} among its ${k} nearest other cells is from the `
        + `slide's own mix, in nats (0: the same mix; neighbours all of one ${kind} that makes up a share s of the `
        + "slide score log(1/s): 0.69 for half the slide, 2.3 for a tenth).",
    }[layer] + " The neighbours are taken among all the slide's cells.";
    fixed = {
      homotypic: "In bins of 0.1.",
      entropy: "In bins of 0.1.",
      excess: "In bins of 0.2 about chance, −0.1 to +0.1.",
      kl: `In bins of ${step} nats, the last from ${9 * step} on.`,
    }[layer];
  }
  const even = VALUES[layer].centre !== undefined && mapOf(layer) === "diverging" ? ", widened to be even about 0" : "";
  const scale = state.scale.mode === "fixed" || !bins ? fixed
    : `In ${bins.n} equal steps between the slide's percentiles ${bins.from} and ${bins.to}, `
      + `${valueText(layer, bins.lo, bins.width)} and ${valueText(layer, bins.hi, bins.width)}${even}; the first `
      + "and last steps also take the values beyond.";
  return `${text} ${scale}`;
}

function legendEntries() {
  const { types, type_class: typeClass, classes } = state.meta;
  const mode = paletteMode();
  const count = (t) => (state.counts ? state.counts[t] : 0);
  if (VALUE_LAYERS.includes(state.layer)) return valueEntries(mode);
  if (state.layer === "clusters") {
    const { emeans } = state.clusters;
    const entries = state.clusters.clusters.map((cluster, j) => ({
      key: j,
      label: emeans ? `#${cluster.label}` : `C${j + 1}`,
      hex: CLUSTERS[mode][j],
      n: cluster.n,
      mix: cluster.fractions.map((f, k) => ({ f, hex: PALETTE.classes[mode][classes[k]] })),
      title: emeans ? emeansTitle(cluster) : `${Math.round(cluster.cells_per_ball)} cells per ball on average; `
        + cluster.fractions.map((f, k) => `${classes[k]} ${f.toFixed(2)}`).join(", "),
    }));
    if (state.clusters.tda) {
      const others = state.types.length - state.clusters.n_centres;
      entries.push({ key: BACKGROUND, label: "other cells", hex: GREY[mode], n: others, background: true });
    }
    return entries;
  }
  if (state.layer === "types") {
    return Object.keys(PALETTE.types[mode]).map((name) => {
      const t = types.indexOf(name);
      if (t < 0) throw new Error(`the cohort has no cell type ${name}`);
      return { key: t, label: name, hex: PALETTE.types[mode][name], n: count(t) };
    });
  }
  return classes.map((name, k) => ({
    key: k,
    label: name,
    hex: PALETTE.classes[mode][name],
    n: types.reduce((sum, _, t) => sum + (typeClass[t] === k ? count(t) : 0), 0),
  }));
}

// A saved e-means cluster's profile over every cohort cell, which is what defines it.
function emeansTitle(cluster) {
  const { classes } = state.meta;
  const median = cluster.kept_fractions[5]; // threshold 5 is the median of the cohort's 10-NN density
  return `Saved label ${cluster.label}. Over the cohort: ${(100 * cluster.cohort_share).toFixed(1)}% of cells, `
    + `${Math.round(cluster.cells_per_ball)} cells per ball; `
    + cluster.fractions.map((f, k) => `${classes[k]} ${f.toFixed(2)}`).join(", ")
    + `; ${(100 * median).toFixed(0)}% of the ball's cells at or above the cohort's median density.`;
}

function applyColors() {
  if (!state.meta) return;
  const { types, type_class: typeClass, classes } = state.meta;
  const mode = paletteMode();
  const table = new Float32Array(16 * 4);
  if (VALUE_LAYERS.includes(state.layer)) {
    const hidden = state.hidden[state.layer];
    for (const e of valueEntries(mode)) table.set([...rgb(e.hex), hidden.has(e.key) ? 0 : 1], e.key * 4);
    overlay?.setColors(table);
    renderLegend();
    return;
  }
  if (state.layer === "clusters") {
    const sizes = new Float32Array(16).fill(1);
    state.clusters.clusters.forEach((_, j) => {
      table.set([...rgb(CLUSTERS[mode][j]), state.hidden.clusters.has(j) ? 0 : 1], j * 4);
      if (state.clusters.tda) sizes[j] = CENTRE_SIZE;
    });
    if (state.clusters.tda) {
      table.set([...rgb(GREY[mode]), state.hidden.clusters.has(BACKGROUND) ? 0 : 2], BACKGROUND * 4);
    }
    overlay?.setColors(table, sizes);
    renderLegend();
    return;
  }
  types.forEach((name, t) => {
    const k = typeClass[t];
    const hex = state.layer === "types" ? PALETTE.types[mode][name] : PALETTE.classes[mode][classes[k]];
    if (!hex) throw new Error(`no colour for ${name}`);
    const shown = state.layer === "types" ? !state.hidden.types.has(t) : !state.hidden.classes.has(k);
    table.set([...rgb(hex), shown ? 1 : 0], t * 4);
  });
  overlay?.setColors(table);
  renderLegend();
}

function renderLegend() {
  const entries = legendEntries();
  const hidden = state.hidden[state.layer];
  const total = entries.reduce((sum, e) => sum + (e.background ? 0 : e.n), 0);
  const list = $("legend");
  list.replaceChildren(...entries.map((e) => {
    const row = document.createElement("li");
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !hidden.has(e.key);
    box.addEventListener("change", () => {
      if (box.checked) hidden.delete(e.key); else hidden.add(e.key);
      applyColors();
    });
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = e.hex;
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = e.label;
    const n = document.createElement("span");
    n.className = "count";
    const share = total ? ((100 * e.n) / total).toFixed(1) : "0.0";
    n.textContent = !state.counts ? "" : e.background ? fmt(e.n) : `${fmt(e.n)} · ${share}%`;
    label.append(box, swatch, name, n);
    if (e.title) label.title = e.title;
    if (e.mix) {
      const mix = document.createElement("span");
      mix.className = "mix";
      mix.append(...e.mix.map(({ f, hex }) => {
        const part = document.createElement("span");
        part.style.width = `${(100 * f).toFixed(2)}%`;
        part.style.background = hex;
        return part;
      }));
      label.append(mix);
    }
    const only = document.createElement("button");
    only.type = "button";
    only.className = "only";
    only.textContent = "only";
    only.title = `Show only ${e.label}`;
    only.addEventListener("click", () => {
      const solo = entries.every((o) => (o.key === e.key) !== hidden.has(o.key));
      hidden.clear();
      if (!solo) entries.forEach((o) => { if (o.key !== e.key) hidden.add(o.key); });
      applyColors();
    });
    row.append(label, only);
    return row;
  }));
  const note = $("legend-note");
  note.hidden = !VALUE_LAYERS.includes(state.layer);
  if (!note.hidden) note.textContent = valueNote(state.layer);
}

// --- slide list -------------------------------------------------------------------------------------------

function renderGroups() {
  const select = $("group");
  const all = new Option(`All groups (${state.meta.slides.length})`, "");
  select.replaceChildren(all, ...state.meta.groups.map((g) => new Option(`${g.name} (${g.n_slides})`, g.name)));
  const withSlide = state.meta.slides.filter((s) => s.wsi).length;
  $("hne-only").disabled = withSlide === 0;
  $("hne-only-label").textContent = `Only slides with H&E (${withSlide})`;
}

function slideLabel(s) {
  return `${s.slide_id} · ${fmt(s.n_cells)} cells${s.pilot ? " · pilot" : ""}${s.wsi ? " · H&E" : ""}`;
}

// A slide's option in the list, with its GTEx pathology categories and notes as the tooltip.
function slideOption(s) {
  const option = new Option(slideLabel(s), s.slide_id);
  option.title = [s.categories, s.notes].filter(Boolean).join(": ") || "no pathology categories or notes";
  return option;
}

function renderPathology(id) {
  const s = state.meta.slides.find((x) => x.slide_id === id);
  $("pathology").hidden = !s;
  if (!s) return;
  $("pathology-group").textContent = `Pathology group: ${s.group}`;
  $("pathology-flag").textContent = s.quality_flag ? `flag: ${s.quality_flag}` : "";
  $("pathology-donor").textContent = `Donor: ${s.sex || "sex unknown"}, age ${s.age_bracket || "unknown"}; `
    + `Hardy scale: ${s.hardy_scale || "unknown"}`;
  $("pathology-categories").textContent = `GTEx categories: ${s.categories || "none"}`;
  $("pathology-notes").textContent = s.notes ? `Notes: ${s.notes}` : "No pathology notes.";
}

function renderSlides() {
  const group = $("group").value;
  const onlyHne = $("hne-only").checked;
  const slides = state.meta.slides.filter((s) => (!group || s.group === group) && (!onlyHne || s.wsi));
  const list = $("slides");
  if (group) {
    list.replaceChildren(...slides.map(slideOption));
  } else {
    const blocks = state.meta.groups.map((g) => {
      const members = slides.filter((s) => s.group === g.name);
      if (!members.length) return null;
      const block = document.createElement("optgroup");
      block.label = g.name;
      block.append(...members.map(slideOption));
      return block;
    }).filter(Boolean);
    list.replaceChildren(...blocks);
  }
  $("n-slides").textContent = `${slides.length} slide${slides.length === 1 ? "" : "s"}`;
  if (!slides.length) return;
  if (slides.some((s) => s.slide_id === state.slide)) {
    list.value = state.slide;
  } else {
    list.value = slides[0].slide_id;
    loadSlide(slides[0].slide_id);
  }
}

// The per-cell codes of a layer: cluster labels, density bins, or type codes for types and classes.
function codesOf(layer) {
  if (layer === "clusters") return state.clusters?.labels ?? null;
  if (VALUE_LAYERS.includes(layer)) return binsOf(layer)?.codes ?? null;
  return state.types;
}

// Layers whose values the server computes for the shown slide on first use; each switches once they are here.
const LAZY = {
  voronoi: () => showVoronoi(),
  homotypic: () => showNeighbourhood("homotypic"),
  entropy: () => showNeighbourhood("entropy"),
  excess: () => showNeighbourhood("excess"),
  kl: () => showNeighbourhood("kl"),
  coloc: () => showColoc(),
};

// After a failed request: check the layer still shown again, and say why.
function lazyFailed(message) {
  document.querySelector(`input[name=layer][value=${state.layer}]`).checked = true;
  $("neighbourhood").hidden = !USES_NEIGHBOURS.includes(state.layer);
  $("coloc").hidden = state.layer !== "coloc";
  $("matrix").hidden = state.layer !== "coloc" || state.matrixClosed;
  $("scale").hidden = !VALUE_LAYERS.includes(state.layer);
  renderLegend();
  $("legend-note").hidden = false;
  $("legend-note").textContent = message;
}

// /api/neighbourhood/<id>: [n u32], then n f32 values and then n u8 bins of each NEIGHBOURHOOD layer in turn.
function parseNeighbourhood(buffer, n, k, by) {
  if (new DataView(buffer).getUint32(0, true) !== n) throw new Error("values for another number of cells");
  const part = (j) => {
    const values = new Float32Array(buffer, 4 + 4 * n * j, n);
    const codes = new Uint8Array(buffer, 4 + 16 * n + n * j, n);
    const counts = new Array(NONE + 1).fill(0);
    for (let i = 0; i < n; i++) counts[codes[i]] += 1;
    return { values, codes, counts };
  };
  const nLabels = state.meta.neighbourhood.n_labels[by];
  if (buffer.byteLength !== 4 + 5 * NEIGHBOURHOOD.length * n) throw new Error("values of another layout");
  const layers = Object.fromEntries(NEIGHBOURHOOD.map((name, j) => [name, part(j)]));
  return { k, by, nLabels, ...layers };
}

// /api/colocalization/<id>: [n u32][n f32 value][n u8 bin], NaN and its own bin where a cell has no estimate.
function parseValueBlob(buffer, n) {
  if (new DataView(buffer).getUint32(0, true) !== n) throw new Error("values for another number of cells");
  const codes = new Uint8Array(buffer, 4 + 4 * n, n);
  const counts = new Array(NONE + 1).fill(0);
  for (let i = 0; i < n; i++) counts[codes[i]] += 1;
  return { values: new Float32Array(buffer, 4, n), codes, counts };
}

// The colocalization excess for the k and label set of the neighbourhood controls, of the cell's own type or the
// one chosen, shrunk or raw, with its pooled matrix for the panel. A later choice wins over a request still running.
async function showColoc() {
  const id = state.slide;
  if (!state.info || state.info.slide_id !== id) return; // the slide's own open handler calls again
  const { k_min: kMin, k_max: kMax } = state.meta.colocalization;
  const k = Number($("nb-k").value);
  const by = $("nb-by").value;
  const of = $("coloc-of").value;
  const score = $("coloc-score").value;
  const note = $("legend-note");
  if (!Number.isInteger(k) || k < kMin || k > kMax) {
    note.hidden = false;
    note.textContent = `k must be a whole number from ${kMin} to ${kMax}.`;
    return;
  }
  const have = state.coloc;
  if (!have || have.k !== k || have.by !== by || have.of !== of || have.score !== score) {
    note.hidden = false;
    note.textContent = `Computing the colocalization excess of ${fmt(state.info.n_cells)} cells …`;
    try {
      const slide = `/api/colocalization/${encodeURIComponent(id)}`;
      const [buffer, summary] = await Promise.all([
        fetchOk(`${slide}?k=${k}&by=${by}&of=${of}&score=${score}`).then((r) => r.arrayBuffer()),
        fetchOk(`${slide}/summary?k=${k}&by=${by}`).then((r) => r.json()),
      ]);
      const same = Number($("nb-k").value) === k && $("nb-by").value === by && $("coloc-of").value === of
        && $("coloc-score").value === score;
      if (state.slide !== id || !same) return;
      state.coloc = { k, by, of, score, ...parseValueBlob(buffer, state.types.length) };
      state.colocSummary = summary;
      renderMatrix();
    } catch (err) {
      if (state.slide === id) lazyFailed(`Could not compute the colocalization excess: ${err.message}`);
      return;
    }
  }
  if ($("layer-coloc").checked) setLayer("coloc");
}

// The pooled excess of the shown slide as a focal × neighbour matrix, in the colour map the menu gives the
// colocalization excess, for the page's theme.
function renderMatrix() {
  const s = state.colocSummary;
  if (!s || !matrix) return;
  const theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
  $("matrix-title").textContent = `Pooled excess on this slide, k = ${s.k}, by ${s.by === "types" ? "type" : "class"}`;
  matrix.render({
    labels: s.labels,
    values: s.pooled_excess,
    colours: colourScale(mapOf("coloc"), 11, theme),
    none: GREY[theme],
    describe: (b, a) => {
      const v = s.pooled_excess[b][a];
      const head = `${s.labels[b]} cells (${fmt(s.n_cells[b])}) · ${s.labels[a]} neighbours: `;
      if (v === null) return `${head}no estimate`;
      return `${head}pooled excess ${signed(v)} ± ${Math.sqrt(s.pooled_variance[b][a]).toFixed(3)}, median of the `
        + `shrunk ${signed(s.median[b][a])}; ${s.gain[b][a] ? "gain" : "loss"} scale`;
    },
  });
}

// The neighbour types the colouring can show: the cell's own, or each type or class of the label set.
function renderColocTargets() {
  const names = state.meta.colocalization?.labels[$("nb-by").value] ?? [];
  const options = names.map((name, j) => new Option(name, String(j)));
  const own = $("nb-by").value === "types" ? "its own type" : "its own class";
  $("coloc-of").replaceChildren(new Option(own, "own"), ...options);
}

// Both values come from one request for the k and label set in the controls; a later choice wins over an
// earlier request still running.
async function showNeighbourhood(layer) {
  const id = state.slide;
  if (!state.info || state.info.slide_id !== id) return; // the slide's own open handler calls again
  const { k_min: kMin, k_max: kMax } = state.meta.neighbourhood;
  const k = Number($("nb-k").value);
  const by = $("nb-by").value;
  const note = $("legend-note");
  if (!Number.isInteger(k) || k < kMin || k > kMax) {
    note.hidden = false;
    note.textContent = `k must be a whole number from ${kMin} to ${kMax}.`;
    return;
  }
  const have = state.neighbourhood;
  if (!have || have.k !== k || have.by !== by) {
    note.hidden = false;
    note.textContent = `Finding the ${k} nearest cells of each of ${fmt(state.info.n_cells)} cells …`;
    try {
      const url = `/api/neighbourhood/${encodeURIComponent(id)}?k=${k}&by=${by}`;
      const buffer = await fetchOk(url).then((r) => r.arrayBuffer());
      if (state.slide !== id || Number($("nb-k").value) !== k || $("nb-by").value !== by) return;
      state.neighbourhood = parseNeighbourhood(buffer, state.types.length, k, by);
    } catch (err) {
      if (state.slide === id) lazyFailed(`Could not compute the neighbourhoods: ${err.message}`);
      return;
    }
  }
  if ($(`layer-${layer}`).checked) setLayer(layer);
}

// /api/voronoi/<id>: [n u32][n f32 Voronoi density][n u8 code], the code VORONOI_NONE for cells without one.
function parseVoronoi(buffer, n) {
  if (new DataView(buffer).getUint32(0, true) !== n) throw new Error("Voronoi densities for another number of cells");
  const codes = new Uint8Array(buffer, 4 + 4 * n, n);
  const counts = new Array(state.meta.density.voronoi_none + 1).fill(0);
  for (let i = 0; i < n; i++) counts[codes[i]] += 1;
  return { values: new Float32Array(buffer, 4, n), codes, counts };
}

// The Voronoi layer: the server triangulates the shown slide's cells on first use, which takes a few seconds on
// a large slide; the layer switches once the densities are here, if it is still the one chosen.
async function showVoronoi() {
  const id = state.slide;
  if (!state.info || state.info.slide_id !== id) return; // the slide's own open handler calls again
  const note = $("legend-note");
  if (!state.voronoi) {
    note.hidden = false;
    note.textContent = `Computing the Voronoi cells of ${fmt(state.info.n_cells)} cells …`;
    try {
      const buffer = await fetchOk(`/api/voronoi/${encodeURIComponent(id)}`).then((r) => r.arrayBuffer());
      if (state.slide !== id) return;
      state.voronoi = parseVoronoi(buffer, state.types.length);
    } catch (err) {
      if (state.slide === id) lazyFailed(`Could not compute the Voronoi densities: ${err.message}`);
      return;
    }
  }
  if ($("layer-voronoi").checked) setLayer("voronoi");
}

// The per-cell codes the overlay holds now. Compared with this, not with the codes of the layer shown before: a
// lazy layer's new values (another k, label set or neighbour type) replace its codes before setLayer runs.
let shownCodes = null;

function uploadCodes(codes) {
  if (!overlay || !codes || codes === shownCodes) return;
  overlay.setCodes(codes);
  shownCodes = codes;
}

function setLayer(layer) {
  state.layer = layer;
  document.querySelector(`input[name=layer][value=${layer}]`).checked = true;
  $("matrix").hidden = layer !== "coloc" || state.matrixClosed;
  showPercentiles(layer, state.scale.mode);
  uploadCodes(codesOf(layer));
  if (thresholdSource() !== state.thresholdSource) syncThreshold();
  applyColors();
}

// --- composition clusters ---------------------------------------------------------------------------------

function clusterMethod(r) {
  if (r.emeans) {
    const worse = r.worse_than_kmeans == null ? ""
      : ` At this k the fit's W_k is ${(100 * Math.expm1(r.worse_than_kmeans)).toFixed(2)}% above k-means++'s `
        + "(kmeans_check.json): a worse partition than k-means++ finds.";
    return `Saved e-means clusters (k = ${r.k}) of every cohort cell's ${r.radius_um} µm ball on ${r.set}: `
      + `${r.features.length} columns z-scored over the whole cohort (${r.file}). #j is the saved label; the `
      + `colours go to the largest clusters over the cohort first. Tibshirani's k̂ = ${r.k_tibshirani}.${worse} `
      + `${r.seconds} s.`;
  }
  if (r.tda) {
    return `k-means (k = ${r.k}, ${r.n_init} starts, seed ${r.seed}) of the slide's ${fmt(r.n_centres)} sampled `
      + `centres on the pilot's level-2 features: ${r.features} of their residual chromatic Betti curves `
      + `(${r.radius_um} µm balls, ${r.n_labellings} random labellings). The pilot found no subdomains that `
      + `recur across slides, so these clusters are exploratory; ${r.seconds} s.`;
  }
  const scale = r.um_source === "assumed" ? `, ${r.um_per_px} µm/px assumed` : "";
  return `k-means (k = ${r.k}, ${r.n_init} starts, seed ${r.seed}) of ${r.features.join(", ")} in every cell's `
    + `${r.radius_um} µm ball, z-scored on this slide${scale}; ${r.seconds} s.`;
}

function clearClusters() {
  state.clusters = null;
  state.hidden.clusters.clear();
  $("layer-clusters").disabled = true;
  if (state.layer === "clusters") state.layer = "types";
  document.querySelector(`input[name=layer][value=${state.layer}]`).checked = true;
  $("cluster-note").textContent = defaultNote();
}

function defaultNote() {
  const m = state.meta.cluster;
  if ($("method").value === "emeans") {
    return "Colours every cell by its saved cluster from e-means on every cohort cell's ball "
      + "(gtex-liver-emeans-neighbourhoods); the gap statistic's k̂ is in brackets.";
  }
  if ($("method").value === "tda") {
    return `Clusters the slide's ${fmt(m.tda_centres)} sampled centres by their saved chromatic TDA features `
      + `(${m.tda_radius_um} µm balls, ${m.tda_labellings} random labellings); pilot slides only.`;
  }
  return "Clusters every cell by its ball: log10 cells (with density) and the four class fractions, as the "
    + `pilot's level 1 (radius ${m.radius_um} µm there).`;
}

// TDA needs a pilot slide and e-means the e-means study's results; the radius and density belong to
// composition clusters only.
function updateMethod() {
  const pilot = Boolean(state.info?.pilot);
  $("method").querySelector("option[value=tda]").disabled = !pilot;
  $("method").querySelector("option[value=emeans]").disabled = !state.meta.emeans;
  if (!pilot && $("method").value === "tda") $("method").value = "composition";
  if (!state.meta.emeans && $("method").value === "emeans") $("method").value = "composition";
  const method = $("method").value;
  $("radius").disabled = method !== "composition";
  $("density").disabled = method !== "composition";
  $("emeans-field").hidden = method !== "emeans";
}

function decode(base64) {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

async function runClusters() {
  const id = state.slide;
  const { k_min: kMin, k_max: kMax, radius_min_um: rMin, radius_max_um: rMax } = state.meta.cluster;
  const k = Number($("k").value);
  const radius = Number($("radius").value);
  if (!state.info || state.info.slide_id !== id) return;
  if (!Number.isInteger(k) || k < kMin || k > kMax) {
    $("cluster-note").textContent = `k must be a whole number from ${kMin} to ${kMax}.`;
    return;
  }
  const method = $("method").value;
  const tda = method === "tda";
  if (method === "composition" && (!Number.isFinite(radius) || radius < rMin || radius > rMax)) {
    $("cluster-note").textContent = `The radius must be from ${rMin} to ${rMax} µm.`;
    return;
  }
  const density = $("density").checked;
  const set = $("emeans-set").value;
  $("run").disabled = true;
  $("cluster-note").textContent = {
    tda: `Clustering ${fmt(state.meta.cluster.tda_centres)} centres with k = ${k} …`,
    emeans: `Reading the saved e-means labels of ${set}, k = ${k} …`,
  }[method] ?? `Clustering ${fmt(state.info.n_cells)} cells with k = ${k} in ${radius} µm balls …`;
  try {
    const url = {
      tda: `/api/tda/${encodeURIComponent(id)}?k=${k}`,
      emeans: `/api/emeans/${encodeURIComponent(id)}?set=${encodeURIComponent(set)}&k=${k}`,
    }[method] ?? `/api/cluster/${encodeURIComponent(id)}?k=${k}&radius=${radius}&density=${density ? 1 : 0}`;
    const result = await fetchOk(url).then((r) => r.json());
    if (state.slide !== id) return;
    let labels = decode(result.labels);
    if (result.tda) {
      const rows = new DataView(decode(result.centre_rows).buffer);
      const centres = labels;
      labels = new Uint8Array(state.types.length).fill(BACKGROUND);
      for (let i = 0; i < centres.length; i++) labels[rows.getUint32(4 * i, true)] = centres[i];
    }
    if (labels.length !== state.types.length) throw new Error(`${labels.length} labels for ${state.types.length} cells`);
    if (state.layer === "clusters") uploadCodes(state.types); // the old labels leave before the new ones come
    state.layer = "types";
    state.clusters = { ...result, labels };
    state.hidden.clusters.clear();
    $("layer-clusters").disabled = false;
    $("cluster-note").textContent = clusterMethod(result);
    setLayer("clusters");
  } catch (err) {
    if (state.slide === id) $("cluster-note").textContent = `Could not cluster: ${err.message}`;
  } finally {
    $("run").disabled = false;
  }
}

// --- density threshold --------------------------------------------------------------------------------

// The threshold at a slider step (the cohort's density quantile at step / 1000; the deciles at steps 0, 100, ...,
// 900), and the density bin it falls in, by density_codes' rule.
function thresholdAt(step) {
  const { curve, deciles } = state.meta.density;
  const t = curve[step];
  return { t, bin: binOf(t, deciles) };
}

// Hide the cells below the threshold in every layer, as the overlay decides it: by bin where the bins differ,
// by float32 density inside the threshold's bin.
// The attribute the slider thresholds: the colouring's values, or the 10-NN density under a categorical colouring.
function thresholdOf() {
  return VALUE_LAYERS.includes(state.layer) ? state.layer : "density";
}

function thresholdSource() {
  const on = thresholdOf();
  return on === "density" ? state.rho : valuesOf(on);
}

const thresholdArrays = new WeakMap(); // a layer's values -> the overlay's bins (all 0) and values (none: LOWEST)

// What the overlay thresholds for a value layer: its values as float32, a cell without one at LOWEST, all in bin 0.
function thresholdArraysOf(source) {
  if (!thresholdArrays.has(source)) {
    const values = Float32Array.from(source, (v) => (Number.isNaN(v) ? LOWEST : v));
    thresholdArrays.set(source, { bins: new Uint8Array(source.length), values });
  }
  return thresholdArrays.get(source);
}

// Give the overlay what the slider thresholds: the 10-NN density's cohort bins and densities (exact at the density
// slices), or the colouring's own values, where a cell without one is below every threshold. The class curves
// follow.
function syncThreshold() {
  if (!overlay || !state.density) return;
  const on = thresholdOf();
  const source = thresholdSource();
  if (on === "density") {
    overlay.setThresholdValues(state.density, state.rho);
  } else if (source) {
    const { bins, values } = thresholdArraysOf(source);
    overlay.setThresholdValues(bins, values);
  }
  state.thresholdSource = source;
  updateCurves();
  applyThreshold();
}

// The first index of sorted values at or above t.
function lowerBound(sorted, t) {
  let [lo, hi] = [0, sorted.length];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Hide the cells below the threshold in every layer, as the overlay decides it. The 10-NN density runs over the
// cohort's quantiles (its deciles are the density slices): by bin where the bins differ, by float32 density inside
// the threshold's bin. Any other value runs over the shown slide's own quantiles, by its float32 value.
function applyThreshold() {
  const step = Number($("threshold").value);
  const on = thresholdOf();
  $("threshold-label").textContent = `Show only cells with ${VALUES[on].name} at or above`;
  if (on !== "density") {
    applyValueThreshold(on, step);
    return;
  }
  const { t, bin } = thresholdAt(step);
  overlay?.setThreshold(bin, t);
  curves?.setMarker(step > 0 ? step / 1000 : null);
  const bins = state.density;
  const rho = state.rho;
  if (!bins) {
    $("threshold-note").textContent = "";
    return;
  }
  const tf = Math.fround(t);
  let shown = 0;
  for (let i = 0; i < bins.length; i++) if (bins[i] > bin || (bins[i] === bin && rho[i] >= tf)) shown += 1;
  const n = bins.length;
  if (step === 0) {
    $("threshold-note").textContent = `No threshold: every cell is shown (${fmt(n)}).`;
    return;
  }
  const slice = step % 100 === 0 ? `, density slice ${step / 100}` : "";
  $("threshold-note").textContent = `≥ ${fmt(Math.round(t))} cells per mm² (cohort percentile `
    + `${(step / 10).toFixed(1)}${slice}): ${fmt(shown)} of ${fmt(n)} cells shown (${((100 * shown) / n).toFixed(1)}%).`;
}

// A value layer's threshold at a slider step: the slide's quantile at step / 1000 as float32, LOWEST at step 0.
function valueThreshold(sorted, step) {
  return step === 0 || !sorted.length ? LOWEST : Math.fround(quantile(sorted, step / 1000));
}

function applyValueThreshold(on, step) {
  const values = valuesOf(on);
  if (!values || state.thresholdSource !== values) {
    overlay?.setThreshold(0, LOWEST);
    curves?.setMarker(null);
    $("threshold-note").textContent = "";
    return;
  }
  const sorted = sortedOf(values);
  const n = values.length;
  const t = valueThreshold(sorted, step);
  overlay?.setThreshold(0, t);
  curves?.setMarker(step > 0 ? curvesAt(values, t) : null);
  if (step === 0) {
    $("threshold-note").textContent = `No threshold: every cell is shown (${fmt(n)}).`;
    return;
  }
  const shown = sorted.length - lowerBound(sorted, t);
  const { unit } = VALUES[on];
  const width = sorted.length ? (sorted[sorted.length - 1] - sorted[0]) / 1000 : 0;
  const none = n - sorted.length ? `; the ${fmt(n - sorted.length)} cells without a value are hidden` : "";
  $("threshold-note").textContent = `≥ ${valueText(on, t, width)}${unit ? ` ${unit}` : ""} (this slide's percentile `
    + `${(step / 10).toFixed(1)}): ${fmt(shown)} of ${fmt(n)} cells shown (${((100 * shown) / n).toFixed(1)}%)${none}.`;
}

// --- class curves -----------------------------------------------------------------------------------------

const curveData = new WeakMap(); // a threshold source's values -> its class curves, by the range they span
const VALUE_STEPS = 501; // the thresholds along a value colouring's class curves
let shownCurves = null; // the class curves on show: the values they threshold and their data

// The caption of the class curves' x axis for the attribute the slider thresholds.
function curvesCaption(on) {
  if (on === "density") return "threshold: cohort density percentile (every 10 = a density slice)";
  const { name, unit } = VALUES[on];
  const short = unit === "cells per mm²" ? "cells/mm²" : unit;
  return `threshold: ${name}${short ? ` (${short})` : ""}, this slide's p${state.scale.from}–p${state.scale.to}`;
}

// Round values between lo and hi for ticks: steps of 1, 2 or 5 × 10^k, about six of them.
function niceTicks(lo, hi, count = 6) {
  const span = hi - lo;
  if (!(span > 0)) return { step: 0, values: [lo] };
  let step = 10 ** Math.floor(Math.log10(span / count));
  const err = (count * step) / span;
  if (err <= 0.15) step *= 10;
  else if (err <= 0.35) step *= 5;
  else if (err <= 0.75) step *= 2;
  const values = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) values.push(Number(v.toPrecision(12)));
  return { step, values };
}

// A tick label: the value as its layer prints it, to the ticks' step, shortened from 10,000 on (12.3K).
function tickText(layer, v, step) {
  if (Math.abs(v) < 1e4) return valueText(layer, v, step);
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

// The class curves over the 10-NN density: the slider's cohort quantiles, counted by the overlay's rule.
function densityCurves() {
  const { curve, deciles } = state.meta.density;
  const nClasses = state.meta.classes.length;
  const counts = classCounts({ classes: state.classes, bins: state.density, rho: state.rho, nClasses, curve, deciles });
  return {
    counts,
    nCells: state.classes.length,
    caption: curvesCaption("density"),
    ticks: Array.from({ length: 11 }, (_, j) => ({ at: j / 10, label: String(10 * j) })),
    describe: (s) => `percentile ${(s / 10).toFixed(1)} · ≥ ${fmt(Math.round(curve[s]))} cells per mm²`,
  };
}

// The class curves over a value colouring: the value itself, at equal steps from the shown slide's lower to its
// upper percentile of the colour menus, counted by the overlay's rule (float32 values; a cell without one is never
// above a threshold here, as none is the slider's "no threshold").
function valueCurves(on, source) {
  const sorted = sortedOf(source);
  const { from, to } = state.scale;
  const lo = sorted.length ? quantile(sorted, from / 100) : 0;
  const hi = sorted.length ? quantile(sorted, to / 100) : 0;
  const span = hi - lo;
  const thresholds = Array.from({ length: VALUE_STEPS }, (_, s) => Math.fround(lo + (span * s) / (VALUE_STEPS - 1)));
  const counts = valueClassCounts({
    classes: state.classes, values: thresholdArraysOf(source).values, nClasses: state.meta.classes.length, thresholds,
  });
  const { unit } = VALUES[on];
  const ticks = niceTicks(lo, hi);
  const percentile = (t) => (100 * lowerBound(sorted, t)) / Math.max(1, sorted.length);
  return {
    counts,
    nCells: state.classes.length,
    caption: curvesCaption(on),
    ticks: ticks.values.map((v) => ({ at: span > 0 ? (v - lo) / span : 0.5, label: tickText(on, v, ticks.step) })),
    describe: (s) => `≥ ${valueText(on, thresholds[s], span / (VALUE_STEPS - 1))}${unit ? ` ${unit}` : ""} · this `
      + `slide's percentile ${percentile(thresholds[s]).toFixed(1)}`,
    lo,
    hi,
  };
}

// Where a value colouring's threshold falls along its class curves' x axis, 0 to 1, or null off it.
function curvesAt(values, t) {
  const data = shownCurves?.values === values ? shownCurves.data : null;
  if (!data || !(data.hi > data.lo)) return null;
  const at = (t - data.lo) / (data.hi - data.lo);
  return at >= 0 && at <= 1 ? at : null;
}

// The class curves for the attribute the slider thresholds, once its values are here.
function updateCurves() {
  if (!curves) return;
  const on = thresholdOf();
  const values = state.thresholdSource;
  if (!values || !state.classes) {
    shownCurves = null;
    curves.clear(curvesCaption(on));
    return;
  }
  const key = on === "density" ? "density" : `${state.scale.from}|${state.scale.to}`;
  if (!curveData.has(values)) curveData.set(values, new Map());
  const cached = curveData.get(values);
  if (!cached.has(key)) cached.set(key, on === "density" ? densityCurves() : valueCurves(on, values));
  shownCurves = { values, data: cached.get(key) };
  curves.setData(shownCurves.data);
}

// The percentile fields show where they act: under slide-percentile bins, and for the class curves of a value
// colouring other than the 10-NN density.
function showPercentiles(layer, mode) {
  $("pct").hidden = mode !== "percentiles" && !(VALUE_LAYERS.includes(layer) && layer !== "density");
}

// --- scale bar --------------------------------------------------------------------------------------------

// The longest of 1, 2 or 5 × 10^k that is not above maxUm.
function niceLength(maxUm) {
  const power = 10 ** Math.floor(Math.log10(maxUm));
  return [5, 2, 1].map((m) => m * power).find((um) => um <= maxUm) ?? power;
}

function updateScaleBar() {
  const scale = $("scalebar");
  const item = viewer.world.getItemAt(0);
  if (!state.info || !item) {
    scale.hidden = true;
    return;
  }
  const rect = item.viewportToImageRectangle(viewer.viewport.getBounds(true));
  const umPerCss = (state.info.um_per_px * rect.width) / viewer.container.clientWidth;
  const um = niceLength(SCALE_MAX_CSS * umPerCss);
  scale.querySelector(".bar").style.width = `${um / umPerCss}px`;
  scale.querySelector(".label").textContent = `${Number(um.toPrecision(3)).toLocaleString("en-US")} µm`;
  scale.hidden = false;
}

// --- one slide --------------------------------------------------------------------------------------------

function applyHne() {
  const item = viewer.world.getItemAt(0);
  if (item) item.setOpacity(state.info?.wsi && $("hne").checked ? 1 : 0);
}

function slideStatus(info) {
  const where = info.wsi ? ($("hne").checked ? "on its H&E" : "H&E hidden") : "no slide file";
  const scale = `${info.um_per_px} µm/px${info.um_source === "assumed" ? " (assumed)" : ""}`;
  return `${info.slide_id} · ${info.group}${info.pilot ? " (pilot)" : ""} · ${fmt(info.n_cells)} cells · ${where} · ${scale}`;
}

async function loadSlide(id) {
  state.slide = id;
  renderPathology(id);
  setStatus(`Loading ${id} …`);
  try {
    const [info, buffer, density] = await Promise.all([
      fetchOk(`/api/slide/${encodeURIComponent(id)}`).then((r) => r.json()),
      fetchOk(`/api/cells/${encodeURIComponent(id)}`).then((r) => r.arrayBuffer()),
      fetchOk(`/api/density/${encodeURIComponent(id)}`).then((r) => r.arrayBuffer()),
    ]);
    if (state.slide !== id) return; // a later choice won
    const n = new DataView(buffer).getUint32(0, true);
    const xy = new Float32Array(buffer, 4, 2 * n);
    const types = new Uint8Array(buffer, 4 + 8 * n, n);
    const counts = new Array(state.meta.types.length).fill(0);
    for (let i = 0; i < n; i++) counts[types[i]] += 1;
    if (new DataView(density).getUint32(0, true) !== n) throw new Error(`density for another number of cells`);
    const rho = new Float32Array(density, 4, n);
    const bins = new Uint8Array(density, 4 + 4 * n, n);
    const binCounts = new Array(DENSITY.light.length).fill(0);
    for (let i = 0; i < n; i++) binCounts[bins[i]] += 1;
    const source = info.wsi
      ? {
        width: info.width, height: info.height, tileSize: info.tile_size, tileOverlap: 0,
        minLevel: 0, maxLevel: info.max_level,
        getTileUrl: (level, x, y) => `/api/tile/${encodeURIComponent(id)}/${level}/${x}_${y}.jpeg`,
      }
      : { width: info.width, height: info.height, tileSize: 256, tileOverlap: 0, getTileUrl: () => BLANK };
    viewer.addOnceHandler("open", () => {
      if (state.slide !== id) return;
      state.info = info;
      state.counts = counts;
      state.types = types;
      state.classes = Uint8Array.from(types, (t) => state.meta.type_class[t]);
      state.density = bins;
      state.rho = rho;
      state.densityCounts = binCounts;
      state.voronoi = null; // these are fetched for this slide when their layer is chosen
      state.neighbourhood = null;
      state.coloc = null;
      state.colocSummary = null;
      const lazy = Object.keys(LAZY).find((layer) => state.layer === layer || $(`layer-${layer}`).checked);
      if (lazy && state.layer === lazy) state.layer = "types"; // until this slide's values are here
      updateMethod();
      clearClusters();
      overlay ??= new PointsOverlay(viewer, CELL_UM / info.um_per_px);
      overlay.cellPx = CELL_UM / info.um_per_px;
      shownCodes = codesOf(state.layer);
      overlay.setPoints(xy, shownCodes, bins, rho);
      state.thresholdSource = rho;
      syncThreshold();
      $("hne").disabled = !info.wsi;
      applyHne();
      applyColors();
      updateScaleBar();
      setStatus(slideStatus(info));
      if (lazy) {
        $(`layer-${lazy}`).checked = true; // clearClusters checked the layer shown meanwhile
        LAZY[lazy]();
      }
    });
    viewer.addOnceHandler("open-failed", (event) => {
      if (state.slide === id) setStatus(`Could not open ${id}: ${event.message}`);
    });
    viewer.open(source);
  } catch (err) {
    if (state.slide === id) setStatus(`Could not load ${id}: ${err.message}`);
  }
}

// --- start ------------------------------------------------------------------------------------------------

async function init() {
  const dark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = storedTheme() || (dark ? "dark" : "light");
  viewer = OpenSeadragon({
    id: "viewer",
    prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
    showNavigationControl: false,
    zoomPerClick: 1,
    maxZoomPixelRatio: 4,
    visibilityRatio: 0.2,
  });
  viewer.addHandler("update-viewport", updateScaleBar);
  viewer.addHandler("resize", updateScaleBar);
  state.meta = await fetchOk("/api/meta").then((r) => r.json());
  if (state.meta.api !== API) {
    throw new Error("serve.py was started before this page changed (or after): stop it and start it again");
  }
  renderGroups();
  $("group").addEventListener("change", renderSlides);
  $("hne-only").addEventListener("change", renderSlides);
  $("slides").addEventListener("change", (event) => loadSlide(event.target.value));
  document.querySelectorAll("input[name=layer]").forEach((radio) => radio.addEventListener("change", () => {
    $("neighbourhood").hidden = !USES_NEIGHBOURS.includes(radio.value);
    $("coloc").hidden = radio.value !== "coloc";
    if (radio.value === "coloc") state.matrixClosed = false;
    $("scale").hidden = !VALUE_LAYERS.includes(radio.value);
    showPercentiles(radio.value, state.scale.mode);
    const next = VALUE_LAYERS.includes(radio.value) ? radio.value : "density";
    if (next !== thresholdOf()) {
      $("threshold").value = "0"; // a new attribute to threshold starts with none
      applyThreshold();
    }
    if (radio.value in LAZY) LAZY[radio.value](); else setLayer(radio.value);
  }));
  const { k, k_min: kMin, k_max: kMax } = state.meta.neighbourhood;
  Object.assign($("nb-k"), { value: String(k), min: String(kMin), max: String(kMax) });
  const neighbours = () => {
    const layer = USES_NEIGHBOURS.find((name) => $(`layer-${name}`).checked);
    if (layer) LAZY[layer]();
  };
  $("nb-k").addEventListener("change", neighbours);
  $("nb-by").addEventListener("change", () => {
    renderColocTargets(); // the label set changes the neighbour types to choose from
    neighbours();
  });
  matrix = new MatrixPanel({ chart: $("matrix-chart"), tip: $("matrix-tip"), scale: $("matrix-scale") });
  $("layer-coloc").disabled = !state.meta.colocalization;
  if (!state.meta.colocalization) {
    $("layer-coloc").parentElement.title = "results/colocalization_scale.json is missing: run fit_colocalization.py";
  }
  renderColocTargets();
  for (const id of ["coloc-of", "coloc-score"]) {
    $(id).addEventListener("change", () => { if ($("layer-coloc").checked) showColoc(); });
  }
  $("matrix-close").addEventListener("click", () => {
    state.matrixClosed = true;
    $("matrix").hidden = true;
  });
  const rescale = () => {
    const from = Number($("pct-from").value);
    const to = Number($("pct-to").value);
    showPercentiles(state.layer, $("scale-mode").value);
    if (!(from >= 0 && to <= 100 && from < to)) {
      $("legend-note").hidden = false;
      $("legend-note").textContent = "The percentiles run from 0 to 100, the first below the second.";
      return;
    }
    state.scale = { map: $("cmap").value, mode: $("scale-mode").value, from, to };
    VALUE_LAYERS.forEach((layer) => state.hidden[layer].clear()); // the bins, and so the toggles, change
    const codes = VALUE_LAYERS.includes(state.layer) ? codesOf(state.layer) : null;
    uploadCodes(codes);
    applyColors();
    renderMatrix(); // the matrix takes the colour map too
    updateCurves(); // a value colouring's class curves span the percentiles
    applyThreshold(); // and the slider's mark on them moves with them
  };
  for (const id of ["cmap", "scale-mode", "pct-from", "pct-to"]) $(id).addEventListener("change", rescale);
  $("emeans-set").replaceChildren(...(state.meta.emeans ?? []).map((s) => new Option(
    `${s.name} (${s.columns} columns, k̂ ${s.k_tibshirani})`, s.name,
  )));
  $("k").min = String(state.meta.cluster.k_min);
  $("k").max = String(state.meta.cluster.k_max);
  $("radius").min = String(state.meta.cluster.radius_min_um);
  $("radius").max = String(state.meta.cluster.radius_max_um);
  $("radius").value = String(state.meta.cluster.radius_um);
  $("radius").addEventListener("keydown", (event) => { if (event.key === "Enter") runClusters(); });
  $("run").addEventListener("click", runClusters);
  $("method").addEventListener("change", () => {
    updateMethod();
    if (!state.clusters) $("cluster-note").textContent = defaultNote();
  });
  $("k").addEventListener("keydown", (event) => { if (event.key === "Enter") runClusters(); });
  clearClusters();
  $("threshold").addEventListener("input", applyThreshold);
  curves = new CurvesPanel({
    chart: $("curves-chart"), legend: $("curves-legend"), tip: $("curves-tip"), select: $("curves-view"),
  });
  updateCurves(); // the caption until a slide is here
  curves.setColours(classColours(), state.meta.classes);
  $("curves-close").addEventListener("click", () => showCurves(false));
  $("curves-open").addEventListener("click", () => showCurves(true));
  let stored = null;
  try { stored = localStorage.getItem("curves"); } catch { /* private window */ }
  showCurves(stored !== "hidden");
  $("size").addEventListener("input", (event) => overlay?.setPointScale(Number(event.target.value)));
  $("alpha").addEventListener("input", (event) => overlay?.setAlpha(Number(event.target.value)));
  $("hne").addEventListener("change", () => {
    applyHne();
    applyColors();
    if (state.info) setStatus(slideStatus(state.info));
  });
  $("theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  renderSlides();
  rescale(); // the colour menus as shown (a browser may restore them on reload), then the colours
  $("slides").focus();
}

init().catch((err) => setStatus(`Could not start: ${err.message}`));
