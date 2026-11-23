// The viewer's page: the slide list filtered by pathology group, an OpenSeadragon canvas (the slide's H&E
// tiles, or a blank canvas of the cloud's size) and the WebGL layer of the slide's cells.
import { PointsOverlay } from "/static/overlay.js";
import { CLUSTERS, DENSITY, PALETTE, rgb } from "/static/palette.js";

const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
const CELL_UM = 10; // a cell's drawn diameter once zoomed in far enough
const SCALE_MAX_CSS = 150; // the scale bar's longest length on screen
const BACKGROUND = 15; // the code of the cells that are not centres, in a TDA clustering
const GREY = { light: "#c3c2b7", dark: "#52514e" }; // those cells, drawn behind the centres
const CENTRE_SIZE = 2.5; // TDA centres are drawn this much larger than the cells
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString("en-US");

const state = {
  meta: null, // /api/meta
  slide: null, // the slide being shown or loading
  info: null, // /api/slide/<id> of the shown slide
  counts: null, // cells per type code on the shown slide
  types: null, // the shown slide's type code per cell
  clusters: null, // /api/cluster/<id> of the shown slide, with its labels decoded
  density: null, // the shown slide's density bin per cell (/api/density/<id>)
  densityCounts: null, // cells per density bin on the shown slide
  layer: "types", // "types" | "classes" | "clusters" | "density"
  hidden: { types: new Set(), classes: new Set(), clusters: new Set(), density: new Set() },
};
let viewer = null;
let overlay = null;

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

function densityTitle(j) {
  const kept = j === 0 ? "kept only in density slice 0, every cell" : `kept in density slices 0 to ${j}`;
  return `Cohort decile ${j + 1} of the 10-NN density: ${densityRange(j, " to ")} cells per mm²; ${kept}.`;
}

function legendEntries() {
  const { types, type_class: typeClass, classes } = state.meta;
  const mode = paletteMode();
  const count = (t) => (state.counts ? state.counts[t] : 0);
  if (state.layer === "density") {
    return DENSITY[mode].map((hex, j) => ({
      key: j,
      label: densityRange(j),
      hex,
      n: state.densityCounts ? state.densityCounts[j] : 0,
      title: densityTitle(j),
    }));
  }
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
  if (state.layer === "density") {
    DENSITY[mode].forEach((hex, j) => table.set([...rgb(hex), state.hidden.density.has(j) ? 0 : 1], j * 4));
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
  note.hidden = state.layer !== "density";
  if (!note.hidden) {
    const { k_nn: k, um_per_px: um } = state.meta.density;
    note.textContent = `Each cell's ${k}-nearest-neighbour density in cells per mm² (the pathology study's `
      + `density.npz, at ${um} µm per pixel), in the cohort's ten deciles: the thresholds of the density slices. `
      + "Low densities fade into the background.";
  }
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

function renderSlides() {
  const group = $("group").value;
  const onlyHne = $("hne-only").checked;
  const slides = state.meta.slides.filter((s) => (!group || s.group === group) && (!onlyHne || s.wsi));
  const list = $("slides");
  if (group) {
    list.replaceChildren(...slides.map((s) => new Option(slideLabel(s), s.slide_id)));
  } else {
    const blocks = state.meta.groups.map((g) => {
      const members = slides.filter((s) => s.group === g.name);
      if (!members.length) return null;
      const block = document.createElement("optgroup");
      block.label = g.name;
      block.append(...members.map((s) => new Option(slideLabel(s), s.slide_id)));
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
  if (layer === "density") return state.density;
  return state.types;
}

function setLayer(layer) {
  const before = codesOf(state.layer);
  state.layer = layer;
  document.querySelector(`input[name=layer][value=${layer}]`).checked = true;
  const codes = codesOf(layer);
  if (overlay && codes && codes !== before) overlay.setCodes(codes);
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
    if (state.layer === "clusters") overlay.setCodes(state.types); // the old labels leave before the new ones come
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
  setStatus(`Loading ${id} …`);
  try {
    const [info, buffer, density] = await Promise.all([
      fetchOk(`/api/slide/${encodeURIComponent(id)}`).then((r) => r.json()),
      fetchOk(`/api/cells/${encodeURIComponent(id)}`).then((r) => r.arrayBuffer()),
      fetchOk(`/api/density/${encodeURIComponent(id)}`).then((r) => r.json()),
    ]);
    if (state.slide !== id) return; // a later choice won
    const n = new DataView(buffer).getUint32(0, true);
    const xy = new Float32Array(buffer, 4, 2 * n);
    const types = new Uint8Array(buffer, 4 + 8 * n, n);
    const counts = new Array(state.meta.types.length).fill(0);
    for (let i = 0; i < n; i++) counts[types[i]] += 1;
    const bins = decode(density.codes);
    if (bins.length !== n) throw new Error(`${bins.length} density bins for ${n} cells`);
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
      state.density = bins;
      state.densityCounts = binCounts;
      updateMethod();
      clearClusters();
      overlay ??= new PointsOverlay(viewer, CELL_UM / info.um_per_px);
      overlay.cellPx = CELL_UM / info.um_per_px;
      overlay.setPoints(xy, codesOf(state.layer));
      $("hne").disabled = !info.wsi;
      applyHne();
      applyColors();
      updateScaleBar();
      setStatus(slideStatus(info));
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
  renderGroups();
  $("group").addEventListener("change", renderSlides);
  $("hne-only").addEventListener("change", renderSlides);
  $("slides").addEventListener("change", (event) => loadSlide(event.target.value));
  document.querySelectorAll("input[name=layer]").forEach((radio) => radio.addEventListener("change", () => {
    setLayer(radio.value);
  }));
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
  $("size").addEventListener("input", (event) => overlay?.setPointScale(Number(event.target.value)));
  $("alpha").addEventListener("input", (event) => overlay?.setAlpha(Number(event.target.value)));
  $("hne").addEventListener("change", () => {
    applyHne();
    applyColors();
    if (state.info) setStatus(slideStatus(state.info));
  });
  $("theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  renderSlides();
  applyColors();
  $("slides").focus();
}

init().catch((err) => setStatus(`Could not start: ${err.message}`));
