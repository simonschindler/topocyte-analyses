// The colocalization panel: a slide's pooled excess as a focal type × neighbour type matrix (an SVG heatmap), on
// a diverging scale even about 0, with each cell's numbers on hover.

const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, style = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  Object.assign(node.style, style);
  return node;
}

export class MatrixPanel {
  // chart: the element the SVG goes in; tip: the hover box; scale: the element under it that names the colours.
  constructor({ chart, tip, scale }) {
    Object.assign(this, { chart, tip, scale });
  }

  // labels: K names; values: K × K pooled excess (null where none), rows focal, columns neighbour; colours: 11
  // hex steps of the diverging map from the negative pole to the positive; describe(b, a): a cell's tooltip.
  render({ labels, values, colours, none, describe }) {
    const k = labels.length;
    const size = k <= 4 ? 30 : 15;
    const left = Math.max(...labels.map((l) => l.length)) * 6.2 + 8;
    const top = left * 0.72 + 6;
    const finite = values.flat().filter((v) => v !== null);
    const limit = finite.length ? Math.max(...finite.map(Math.abs)) : 0;
    const step = (v) => {
      if (v === null) return none;
      if (limit === 0) return colours[colours.length >> 1];
      const j = Math.floor(((v + limit) / (2 * limit)) * colours.length);
      return colours[Math.min(colours.length - 1, Math.max(0, j))];
    };
    const svg = el("svg", { width: left + k * size + 4, height: top + k * size + 4, role: "img" });
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
    this.chart.replaceChildren(svg);
    this.scale.textContent = limit > 0
      ? `blue −${limit.toFixed(2)} … grey 0 … red +${limit.toFixed(2)}; `
        + "rows: focal cells, columns: their neighbours"
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
