// Class curves over the 10-NN density threshold, for the shown slide: how the cells of each class thin out as
// the threshold rises. The x axis is the threshold slider's scale, the cohort's density quantiles in steps of 0.1
// percentile, with the deciles (the density slices) exact at every hundredth step.
//
// Views:
//   stacked: of the slide's cells, the share that is of each class and at or above the threshold, stacked, so the
//            total is the share of the slide's cells kept: 1 without a threshold;
//   among:   of the cells at or above the threshold, the share in each class (the curves add up to 1);
//   own:     of each class's cells, the share at or above the threshold (each curve falls from 1).
//
// The counts follow the overlay's rule: by density bin where the bins differ (the server bins in float64) and by
// float32 density inside the threshold's own bin, so at the density slices they are the studies' kept counts.

export const VIEWS = {
  stacked: "Stacked: share of the slide's cells kept, by class",
  among: "Share of each class among the cells kept",
  own: "Share of each class's own cells kept",
};

const SPARSE = 0.01; // the "among" shares are drawn faded where fewer than this share of the slide's cells is left
const W = 384;
const H = 214;
const M = { left: 36, right: 8, top: 8, bottom: 30 };
const SVG = "http://www.w3.org/2000/svg";

// The density bin a threshold falls in: the number of deciles after the first at or below it (density_codes).
export function binOf(t, deciles) {
  let bin = 0;
  for (let j = 1; j < deciles.length; j++) if (deciles[j] <= t) bin = j;
  return bin;
}

function lowerBound(sorted, x) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// counts[k][s]: the slide's cells of class k at or above the threshold curve[s].
export function classCounts({ classes, bins, rho, nClasses, curve, deciles }) {
  const nBins = deciles.length;
  const n = classes.length;
  const size = new Int32Array(nClasses * nBins);
  for (let i = 0; i < n; i++) size[classes[i] * nBins + bins[i]] += 1;
  const groups = Array.from(size, (m) => new Float32Array(m));
  const fill = new Int32Array(nClasses * nBins);
  for (let i = 0; i < n; i++) {
    const g = classes[i] * nBins + bins[i];
    groups[g][fill[g]++] = rho[i];
  }
  groups.forEach((g) => g.sort());
  const above = Array.from({ length: nClasses }, (_, k) => {
    const out = new Float64Array(nBins); // cells of class k in the bins above b
    for (let b = nBins - 2; b >= 0; b--) out[b] = out[b + 1] + size[k * nBins + b + 1];
    return out;
  });
  const counts = Array.from({ length: nClasses }, () => new Float64Array(curve.length));
  for (let s = 0; s < curve.length; s++) {
    const bin = binOf(curve[s], deciles);
    const tf = Math.fround(curve[s]);
    for (let k = 0; k < nClasses; k++) {
      const g = groups[k * nBins + bin];
      counts[k][s] = above[k][bin] + g.length - lowerBound(g, tf);
    }
  }
  return counts;
}

// Each class's curve in a view, as arrays over the steps (NaN where undefined).
export function viewValues(view, counts, nCells) {
  if (view === "stacked") return counts.map((c) => Array.from(c, (v) => v / nCells));
  if (view === "among") {
    const totals = counts[0].map((_, s) => counts.reduce((sum, c) => sum + c[s], 0));
    return counts.map((c) => Array.from(c, (v, s) => (totals[s] > 0 ? v / totals[s] : NaN)));
  }
  return counts.map((c) => Array.from(c, (v) => (c[0] > 0 ? v / c[0] : NaN)));
}

function el(name, attributes = {}, style = {}) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  Object.assign(node.style, style);
  return node;
}

export class CurvesPanel {
  // parts: the panel's chart container, legend, tooltip, view select and threshold-to-text function.
  constructor({ chart, legend, tip, select, describe }) {
    Object.assign(this, { chart, legend, tip, select, describe });
    this.view = "stacked";
    this.data = null;
    this.colours = null;
    this.names = null;
    this.marker = 0;
    select.replaceChildren(...Object.entries(VIEWS).map(([value, label]) => new Option(label, value)));
    select.value = this.view;
    select.addEventListener("change", () => { this.view = select.value; this.render(); });
  }

  // counts from classCounts, the slide's cell count, the class names and the threshold curve.
  setData(data) { this.data = data; this.render(); }

  clear() { this.data = null; this.render(); }

  setColours(colours, names) { this.colours = colours; this.names = names; this.render(); }

  setMarker(percentile) { this.marker = percentile; this.placeMarker(); }

  x(p) { return M.left + (p / 100) * (W - M.left - M.right); }

  y(v) { return M.top + (1 - v) * (H - M.top - M.bottom); }

  render() {
    this.tip.hidden = true;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    this.svg = svg;
    for (let p = 0; p <= 100; p += 10) {
      svg.append(el("line", { x1: this.x(p), x2: this.x(p), y1: M.top, y2: H - M.bottom }, { stroke: "var(--rule)" }));
      const label = el("text", { x: this.x(p), y: H - M.bottom + 12, "text-anchor": "middle" });
      label.textContent = String(p);
      svg.append(label);
    }
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(el("line", { x1: M.left, x2: W - M.right, y1: this.y(v), y2: this.y(v) }, { stroke: "var(--rule)" }));
      const label = el("text", { x: M.left - 5, y: this.y(v) + 3.5, "text-anchor": "end" });
      label.textContent = String(v);
      svg.append(label);
    }
    const caption = el("text", { x: (M.left + W - M.right) / 2, y: H - 3, "text-anchor": "middle" });
    caption.textContent = "threshold: cohort density percentile (every 10 = a density slice)";
    svg.append(caption);
    if (this.data && this.colours) this.drawCurves(svg);
    this.markerLine = el("line", { y1: M.top, y2: H - M.bottom }, { stroke: "var(--ink)", strokeDasharray: "3 3" });
    svg.append(this.markerLine);
    this.cross = el("line", { y1: M.top, y2: H - M.bottom }, { stroke: "var(--ink2)", visibility: "hidden" });
    svg.append(this.cross);
    const hit = el("rect", { x: M.left, y: M.top, width: W - M.left - M.right, height: H - M.top - M.bottom }, {
      fill: "transparent",
    });
    hit.addEventListener("mousemove", (event) => this.hover(event));
    hit.addEventListener("mouseleave", () => { this.cross.style.visibility = "hidden"; this.tip.hidden = true; });
    svg.append(hit);
    this.chart.replaceChildren(svg);
    this.placeMarker();
    this.legend.replaceChildren(...(this.names ?? []).map((name, k) => {
      const item = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = "sw";
      swatch.style.background = this.colours[k];
      item.append(swatch, name);
      return item;
    }));
  }

  drawCurves(svg) {
    const { counts, nCells } = this.data;
    const values = viewValues(this.view, counts, nCells);
    const steps = values[0].length;
    const px = (s) => this.x((100 * s) / (steps - 1)).toFixed(2);
    if (this.view === "stacked") {
      let lower = new Float64Array(steps);
      values.forEach((v, k) => {
        const upper = lower.map((low, s) => low + v[s]);
        let d = "";
        for (let s = 0; s < steps; s++) d += `${s ? "L" : "M"}${px(s)},${this.y(upper[s]).toFixed(2)}`;
        for (let s = steps - 1; s >= 0; s--) d += `L${px(s)},${this.y(lower[s]).toFixed(2)}`;
        svg.append(el("path", { d: `${d}Z` }, { fill: this.colours[k], stroke: "var(--surface)", strokeWidth: "1" }));
        lower = upper;
      });
      return;
    }
    if (this.view === "among") {
      const totals = counts[0].map((_, s) => counts.reduce((sum, c) => sum + c[s], 0));
      const first = totals.findIndex((t) => t < SPARSE * nCells);
      if (first >= 0) {
        svg.append(el("rect", {
          x: px(first), y: M.top, width: (this.x(100) - Number(px(first))).toFixed(2), height: H - M.top - M.bottom,
        }, { fill: "var(--rule)", opacity: "0.6" }));
      }
    }
    values.forEach((v, k) => {
      let d = "";
      let pen = "M";
      for (let s = 0; s < steps; s++) {
        if (Number.isNaN(v[s])) { pen = "M"; continue; }
        d += `${pen}${px(s)},${this.y(v[s]).toFixed(2)}`;
        pen = "L";
      }
      if (d) svg.append(el("path", { d }, { fill: "none", stroke: this.colours[k], strokeWidth: "2" }));
    });
  }

  placeMarker() {
    if (!this.markerLine) return;
    const x = this.x(this.marker);
    this.markerLine.setAttribute("x1", x);
    this.markerLine.setAttribute("x2", x);
    this.markerLine.style.visibility = this.marker > 0 ? "visible" : "hidden";
  }

  hover(event) {
    if (!this.data) return;
    const box = this.svg.getBoundingClientRect();
    const xSvg = ((event.clientX - box.left) / box.width) * W;
    const steps = this.data.counts[0].length;
    const s = Math.max(0, Math.min(steps - 1, Math.round(((xSvg - M.left) / (W - M.left - M.right)) * (steps - 1))));
    const p = (100 * s) / (steps - 1);
    this.cross.setAttribute("x1", this.x(p));
    this.cross.setAttribute("x2", this.x(p));
    this.cross.style.visibility = "visible";
    const values = viewValues(this.view, this.data.counts, this.data.nCells);
    const pct = (v) => (Number.isNaN(v) ? "–" : `${(100 * v).toFixed(1)}%`);
    const rows = this.names.map((name, k) => `<div><span class="sw" style="background:${this.colours[k]}"></span>`
      + `${name} ${pct(values[k][s])}</div>`);
    if (this.view === "stacked") rows.push(`<div>kept ${pct(values.reduce((sum, v) => sum + v[s], 0))}</div>`);
    this.tip.innerHTML = `<div class="tip-head">${this.describe(s)}</div>${rows.join("")}`;
    this.tip.hidden = false;
    const left = ((this.x(p) / W) * box.width) + (p > 55 ? -this.tip.offsetWidth - 10 : 10);
    this.tip.style.left = `${left + this.chart.offsetLeft}px`;
    this.tip.style.top = `${this.chart.offsetTop + 6}px`;
  }
}
