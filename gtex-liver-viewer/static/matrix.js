// The colocalization panel: a slide's pooled excess as a focal type × neighbour type matrix (an SVG heatmap) in
// the colour map the page gives it, in steps even about 0, with a key under it and each square's numbers on hover.

const NS = "http://www.w3.org/2000/svg";
const KEY_HEIGHT = 8; // the key's swatches

function el(name, attrs = {}, style = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  Object.assign(node.style, style);
  return node;
}

export class MatrixPanel {
  // chart: the element the SVG goes in; tip: the hover box; scale: the line under it that names the axes.
  constructor({ chart, tip, scale }) {
    Object.assign(this, { chart, tip, scale });
  }

  // labels: K names; values: K × K pooled excess (null where none), rows focal, columns neighbour; colours: an odd
  // number of hex steps of a colour map from low to high, the middle one about 0; none: a square without a value;
  // describe(b, a): a square's tooltip.
  render({ labels, values, colours, none, describe }) {
    const k = labels.length;
    const size = k <= 4 ? 30 : 15;
    const left = Math.max(...labels.map((l) => l.length)) * 6.2 + 8;
    const top = left * 0.72 + 6;
    const finite = values.flat().filter((v) => v !== null);
    const limit = finite.length ? Math.max(...finite.map(Math.abs)) : 0;
    const n = colours.length;
    const step = (v) => {
      if (v === null) return none;
      if (limit === 0) return colours[n >> 1];
      const j = Math.floor(((v + limit) / (2 * limit)) * n);
      return colours[Math.min(n - 1, Math.max(0, j))];
    };
    const swatch = Math.max(8, Math.min(14, Math.floor((k * size) / n))); // the key's steps, about the matrix's width
    const keyTop = top + k * size + 8;
    const height = limit > 0 ? keyTop + KEY_HEIGHT + 14 : top + k * size + 4;
    const svg = el("svg", { width: left + Math.max(k * size, n * swatch) + 4, height, role: "img" });
    labels.forEach((label, i) => {
      const row = el("text", { x: left - 6, y: top + i * size + size / 2 + 3.5, "text-anchor": "end" });
      row.textContent = label;
      const x = left + i * size + size / 2;
      const column = el("text", { x, y: top - 6, "text-anchor": "start", transform: `rotate(-45 ${x} ${top - 6})` });
      column.textContent = label;
      svg.append(row, column);
    });
    values.forEach((row, b) => row.forEach((v, a) => {
      const rect = el("rect", { x: left + a * size, y: top + b * size, width: size - 1, height: size - 1 },
        { fill: step(v) });
      rect.addEventListener("mousemove", (event) => this.showTip(event, describe(b, a)));
      rect.addEventListener("mouseleave", () => { this.tip.hidden = true; });
      svg.append(rect);
    }));
    if (limit > 0) { // the key: the steps from −limit to +limit, 0 in the middle one
      colours.forEach((hex, j) => svg.append(el("rect",
        { x: left + j * swatch, y: keyTop, width: swatch - 1, height: KEY_HEIGHT }, { fill: hex })));
      const end = left + n * swatch - 1;
      const marks = [[left, "start", `−${limit.toFixed(2)}`], [(left + end) / 2, "middle", "0"],
        [end, "end", `+${limit.toFixed(2)}`]];
      for (const [x, anchor, text] of marks) {
        const mark = el("text", { x, y: keyTop + KEY_HEIGHT + 11, "text-anchor": anchor });
        mark.textContent = text;
        svg.append(mark);
      }
    }
    this.chart.replaceChildren(svg);
    this.scale.textContent = limit > 0 ? "rows: focal cells, columns: their neighbours"
      : "no pooled excess on this slide";
  }

  showTip(event, text) {
    const box = this.chart.closest("section").getBoundingClientRect();
    this.tip.textContent = text;
    this.tip.hidden = false;
    this.tip.style.left = `${event.clientX - box.left + 12}px`;
    this.tip.style.top = `${event.clientY - box.top + 12}px`;
  }
}
