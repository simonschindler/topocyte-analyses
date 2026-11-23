// The viewer's page: the slide list filtered by pathology group, an OpenSeadragon canvas (the slide's H&E
// tiles, or a blank canvas of the cloud's size) and the WebGL layer of the slide's cells.
import { PointsOverlay } from "/static/overlay.js";
import { CLUSTERS, PALETTE, rgb } from "/static/palette.js";

const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
const CELL_UM = 10; // a cell's drawn diameter once zoomed in far enough
const SCALE_MAX_CSS = 150; // the scale bar's longest length on screen
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString("en-US");

const state = {
  meta: null, // /api/meta
  slide: null, // the slide being shown or loading
  info: null, // /api/slide/<id> of the shown slide
  counts: null, // cells per type code on the shown slide
  types: null, // the shown slide's type code per cell
  clusters: null, // /api/cluster/<id> of the shown slide, with its labels decoded
  layer: "types", // "types" | "classes" | "clusters"
  hidden: { types: new Set(), classes: new Set(), clusters: new Set() },
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

function legendEntries() {
  const { types, type_class: typeClass, classes } = state.meta;
  const mode = paletteMode();
  const count = (t) => (state.counts ? state.counts[t] : 0);
  if (state.layer === "clusters") {
    return state.clusters.clusters.map((cluster, j) => ({
      key: j,
      label: `C${j + 1}`,
      hex: CLUSTERS[mode][j],
      n: cluster.n,
      mix: cluster.fractions.map((f, k) => ({ f, hex: PALETTE.classes[mode][classes[k]] })),
      title: `${Math.round(cluster.cells_per_ball)} cells per ball on average; `
        + cluster.fractions.map((f, k) => `${classes[k]} ${f.toFixed(2)}`).join(", "),
    }));
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

function applyColors() {
  if (!state.meta) return;
  const { types, type_class: typeClass, classes } = state.meta;
  const mode = paletteMode();
  const table = new Float32Array(16 * 4);
  if (state.layer === "clusters") {
    state.clusters.clusters.forEach((_, j) => {
      table.set([...rgb(CLUSTERS[mode][j]), state.hidden.clusters.has(j) ? 0 : 1], j * 4);
    });
    overlay?.setColors(table);
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
  const total = entries.reduce((sum, e) => sum + e.n, 0);
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
    n.textContent = state.counts ? `${fmt(e.n)} · ${total ? ((100 * e.n) / total).toFixed(1) : "0.0"}%` : "";
    label.append(box, swatch, name, n);
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
      label.title = e.title;
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

// The per-cell codes follow the layer: cluster labels for "clusters", type codes otherwise.
function setLayer(layer) {
  const codesChange = (layer === "clusters") !== (state.layer === "clusters");
  state.layer = layer;
  document.querySelector(`input[name=layer][value=${layer}]`).checked = true;
  if (codesChange && overlay && state.types) {
    overlay.setCodes(layer === "clusters" ? state.clusters.labels : state.types);
  }
  applyColors();
}

// --- composition clusters ---------------------------------------------------------------------------------

function clusterMethod(r) {
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
  const { radius_um: radius } = state.meta.cluster;
  $("cluster-note").textContent = "Clusters every cell by its ball: log10 cells (with density) and the four class "
    + `fractions, as the pilot's level 1 (radius ${radius} µm there).`;
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
  if (!Number.isFinite(radius) || radius < rMin || radius > rMax) {
    $("cluster-note").textContent = `The radius must be from ${rMin} to ${rMax} µm.`;
    return;
  }
  const density = $("density").checked;
  $("run").disabled = true;
  $("cluster-note").textContent = `Clustering ${fmt(state.info.n_cells)} cells with k = ${k} in ${radius} µm balls …`;
  try {
    const result = await fetchOk(`/api/cluster/${encodeURIComponent(id)}?k=${k}&radius=${radius}&density=${density ? 1 : 0}`)
      .then((r) => r.json());
    if (state.slide !== id) return;
    const bytes = atob(result.labels);
    const labels = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) labels[i] = bytes.charCodeAt(i);
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
    const [info, buffer] = await Promise.all([
      fetchOk(`/api/slide/${encodeURIComponent(id)}`).then((r) => r.json()),
      fetchOk(`/api/cells/${encodeURIComponent(id)}`).then((r) => r.arrayBuffer()),
    ]);
    if (state.slide !== id) return; // a later choice won
    const n = new DataView(buffer).getUint32(0, true);
    const xy = new Float32Array(buffer, 4, 2 * n);
    const types = new Uint8Array(buffer, 4 + 8 * n, n);
    const counts = new Array(state.meta.types.length).fill(0);
    for (let i = 0; i < n; i++) counts[types[i]] += 1;
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
      clearClusters();
      overlay ??= new PointsOverlay(viewer, CELL_UM / info.um_per_px);
      overlay.cellPx = CELL_UM / info.um_per_px;
      overlay.setPoints(xy, types);
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
  $("k").min = String(state.meta.cluster.k_min);
  $("k").max = String(state.meta.cluster.k_max);
  $("radius").min = String(state.meta.cluster.radius_min_um);
  $("radius").max = String(state.meta.cluster.radius_max_um);
  $("radius").value = String(state.meta.cluster.radius_um);
  $("radius").addEventListener("keydown", (event) => { if (event.key === "Enter") runClusters(); });
  $("run").addEventListener("click", runClusters);
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
