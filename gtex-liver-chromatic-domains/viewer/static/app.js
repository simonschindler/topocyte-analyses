// The viewer's page: the slide list filtered by pathology group, an OpenSeadragon canvas (the slide's H&E
// tiles, or a blank canvas of the cloud's size) and the WebGL layer of the slide's cells.
import { PointsOverlay } from "/static/overlay.js";
import { PALETTE, rgb } from "/static/palette.js";

const BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
const CELL_UM = 10; // a cell's drawn diameter once zoomed in far enough
const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString("en-US");

const state = {
  meta: null, // /api/meta
  slide: null, // the slide being shown or loading
  info: null, // /api/slide/<id> of the shown slide
  counts: null, // cells per type code on the shown slide
  layer: "types", // "types" | "classes"
  hidden: { types: new Set(), classes: new Set() },
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

// --- one slide --------------------------------------------------------------------------------------------

function applyHne() {
  const item = viewer.world.getItemAt(0);
  if (item) item.setOpacity(state.info?.wsi && $("hne").checked ? 1 : 0);
}

function slideStatus(info) {
  const where = info.wsi ? ($("hne").checked ? "on its H&E" : "H&E hidden") : "no slide file";
  return `${info.slide_id} · ${info.group}${info.pilot ? " (pilot)" : ""} · ${fmt(info.n_cells)} cells · ${where}`;
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
      overlay ??= new PointsOverlay(viewer, CELL_UM / state.meta.um_per_px);
      overlay.setPoints(xy, types);
      $("hne").disabled = !info.wsi;
      applyHne();
      applyColors();
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
  state.meta = await fetchOk("/api/meta").then((r) => r.json());
  renderGroups();
  $("group").addEventListener("change", renderSlides);
  $("hne-only").addEventListener("change", renderSlides);
  $("slides").addEventListener("change", (event) => loadSlide(event.target.value));
  document.querySelectorAll("input[name=layer]").forEach((radio) => radio.addEventListener("change", () => {
    state.layer = radio.value;
    applyColors();
  }));
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
