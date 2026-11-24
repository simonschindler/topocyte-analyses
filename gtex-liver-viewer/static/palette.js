// Colours of the four classes and the 14 HistoPLUS types, one set per background (light: light theme or H&E;
// dark: dark theme without H&E), in viridis's hues, to sit with the value colourings' default map.
//
// Classes: epithelial in viridis's blue (OKLCH hue 260), stromal in its teal-green (166) and immune in its yellow
// (102), with "other" a neutral grey (the dataviz palette's fold for "Other"). Each type takes a hue of viridis's,
// or next to its yellow end, near its class's: epithelial blue and purple; stromal teal-green, blue-teal, green and
// teal; immune yellow, lime, gold, olive and amber; the greys as before. Lightness and chroma were searched per
// background inside the dataviz lightness band (light 0.43-0.77, dark 0.48-0.67) for the widest separation, except
// that on the dark canvas the immune shades run brighter (L up to 0.90), as viridis's yellow end, so that the rare
// immune cells stand out. Checked with the dataviz skill's validate_palette.js: the classes on all pairs, as a
// scatter needs, worst CVD ΔE 10.2 light / 13.6 dark (target 8), worst normal-vision ΔE 16.3 / 16.8 (floor 15);
// the types on adjacent pairs, worst CVD ΔE 9.4 / 10.6 and normal-vision ΔE 15.8 / 16.1. The greys fail the chroma
// floor by design, and on the dark canvas the bright immune shades (and the light grey, as before) the band.
// Fourteen colours cannot all stay apart in one scatter (no palette can), so the legend's per-type toggles and
// solo are part of the encoding.

export const PALETTE = {
  classes: {
    light: { epithelial: "#5179b9", stromal: "#22ac81", immune: "#baa800", other: "#52514e" },
    dark: { epithelial: "#638bcd", stromal: "#00a075", immune: "#f2e04f", other: "#c3c2b7" },
  },
  types: {
    light: {
      "Epithelial": "#5179b9", "Cancer cell": "#6c318b",
      "Fibroblasts": "#22ac81", "Endothelial Cell": "#006b91", "Muscle Cell": "#286000", "Minor Stromal Cell": "#009191",
      "Lymphocytes": "#baa800", "Macrophages": "#3f5c00", "Neutrophils": "#9a6d00", "Plasmocytes": "#afb53d",
      "Eosinophils": "#ad5f00",
      "Red blood cell": "#52514e", "Apoptotic Body": "#807f7b", "Mitotic Figures": "#b0afac",
    },
    dark: {
      "Epithelial": "#638bcd", "Cancer cell": "#7b409b",
      "Fibroblasts": "#00a075", "Endothelial Cell": "#007199", "Muscle Cell": "#307100", "Minor Stromal Cell": "#009898",
      "Lymphocytes": "#f2e04f", "Macrophages": "#8cbb40", "Neutrophils": "#b07d00", "Plasmocytes": "#b8be48",
      "Eosinophils": "#bd6900",
      "Red blood cell": "#c3c2b7", "Apoptotic Body": "#919086", "Mitotic Figures": "#626158",
    },
  },
};

// Composition clusters, largest first: the dataviz reference palette's eight categorical slots in its fixed
// order (validated on adjacent pairs; in a scatter only the first three stay apart on every pair), then two
// darker (light) or lighter (dark) steps of slots 1 and 2 for k = 9 and 10. The legend's toggles and "only"
// buttons separate clusters that look alike.
export const CLUSTERS = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#184f95", "#ab3d06"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767", "#86b6ef", "#fe9068"],
};

// 10-NN density, the cohort's ten deciles from low to high: a sequential scale (magnitude, so one hue). The
// dataviz reference palette's blue ramp, steps 100 to 700, resampled in OKLab at ten equal steps of lightness:
// adjacent steps differ by ΔL 0.063 (validate_palette.js --ordinal: monotone, every gap >= 0.06, one hue).
// Light to dark on the light theme and on H&E, flipped on the dark canvas, so that low densities fade into the
// background; the faintest step is below 2:1 against the surface by design.
const DENSITY_RAMP = ["#cde2fb", "#aecef5", "#8ebbf1", "#6da7ec", "#4b92e6", "#2f7ddb", "#256abf", "#1b58a4", "#134688", "#0d366b"];
export const DENSITY = { light: DENSITY_RAMP, dark: [...DENSITY_RAMP].reverse() };

// Homotypic excess, eleven bins from below -0.9 to 0.9 and above: diverging around chance (the middle bin, the
// dataviz neutral grey), blue below and red above, the reference palette's diverging pair. Blue: the reference blue
// ramp resampled in OKLab at L 0.86, 0.75, 0.64, 0.53, 0.42 out from the neutral; red: the same L and chroma at the
// hue of the reference red slot (#e34948), all in sRGB. Each arm passes validate_palette.js --ordinal on monotone L,
// gaps >= 0.06 and one hue, on light and dark; the step next to the neutral fades into the surface by design (1.5:1
// on light, 1.9:1 for red on dark), as chance should. The two arms' first steps are close (ΔE 10.3, 7.7 protan);
// from the second step on they are 18.8 or more apart (13.8 protan). On the dark canvas the arms run the other
// way, dark next to the neutral and light at the poles.
const BLUE_ARM = ["#b8d4f6", "#7fb2ee", "#448de6", "#256bc0", "#164b90"]; // from the neutral out
const RED_ARM = ["#f4c3be", "#e8948d", "#d9615c", "#b23f3d", "#842928"];
export const DIVERGING = {
  light: [...[...BLUE_ARM].reverse(), "#f0efec", ...RED_ARM],
  dark: [...BLUE_ARM, "#383835", ...[...RED_ARM].reverse()],
};

// Viridis (matplotlib 3.11.2, colormaps["viridis"] at n equal steps from 0 to 1): dark purple to yellow, the same on
// every background.
const VIRIDIS = {
  10: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
  11: ["#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c", "#22a884", "#44bf70", "#7ad151", "#bddf26",
    "#fde725"],
};

export function rgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const fromLinear = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function oklab(hex) {
  const [r, g, b] = rgb(hex).map(toLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m
    + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

function hexOf([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const linear = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  return `#${linear.map((c) => Math.round(255 * Math.min(1, Math.max(0, fromLinear(c)))).toString(16).padStart(2, "0"))
    .join("")}`;
}

// n colours at equal steps along the anchors, interpolated in OKLab; the anchors themselves when n is their count.
function sample(anchors, n) {
  if (n === anchors.length) return [...anchors];
  const labs = anchors.map(oklab);
  return Array.from({ length: n }, (_, j) => {
    const x = (j / (n - 1)) * (labs.length - 1);
    const i = Math.min(Math.floor(x), labs.length - 2);
    const t = x - i;
    return hexOf(labs[i].map((v, k) => v + t * (labs[i + 1][k] - v)));
  });
}

// The colour maps a value layer can take, n colours from low to high for the background mode ("light": light theme
// or H&E, "dark": the dark canvas). Blue and diverging are as above at their own counts (10, 11) and resampled
// otherwise; viridis is matplotlib's at either count.
export const COLOUR_MAPS = ["blue", "viridis", "diverging"];

export function colourScale(name, n, mode) {
  if (name === "blue") return sample(DENSITY[mode], n);
  if (name === "diverging") return sample(DIVERGING[mode], n);
  if (name === "viridis") return VIRIDIS[n] ?? sample(VIRIDIS[11], n);
  throw new Error(`no colour map ${name}`);
}
