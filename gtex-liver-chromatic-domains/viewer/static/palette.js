// Colours of the four classes and the 14 HistoPLUS types, one set per background (light: light theme or H&E;
// dark: dark theme without H&E).
//
// Classes: categorical slots 1-3 of the dataviz reference palette (blue, aqua, orange), with "other" as a
// neutral grey (the palette's fold for "Other"). Validated with the dataviz skill's validate_palette.js on all
// pairs, as a scatter needs: worst CVD ΔE 9.2 light / 9.4 dark (target 8), worst normal-vision ΔE 21.8 light /
// 20.9 dark (floor 15). The grey fails the chroma floor by design.
//
// Types: each type is a shade of its class's colour (OKLCH lightness and hue steps), in the order below;
// adjacent shades clear CVD ΔE 9.9 and normal-vision ΔE 14.9–15.0 at worst. Fourteen colours cannot all stay
// apart in one scatter (no palette can), so the legend's per-type toggles and solo are part of the encoding.

export const PALETTE = {
  classes: {
    light: { epithelial: "#2a78d6", stromal: "#1baf7a", immune: "#eb6834", other: "#52514e" },
    dark: { epithelial: "#3987e5", stromal: "#199e70", immune: "#d95926", other: "#c3c2b7" },
  },
  types: {
    light: {
      "Epithelial": "#2a78d6", "Cancer cell": "#6641ac",
      "Fibroblasts": "#1baf7a", "Endothelial Cell": "#077971", "Muscle Cell": "#66cd87", "Minor Stromal Cell": "#0696ab",
      "Lymphocytes": "#eb6834", "Macrophages": "#af3005", "Neutrophils": "#fc9a0e", "Plasmocytes": "#d84e77",
      "Eosinophils": "#c5a600",
      "Red blood cell": "#52514e", "Apoptotic Body": "#807f7b", "Mitotic Figures": "#b0afac",
    },
    dark: {
      "Epithelial": "#3987e5", "Cancer cell": "#7250bb",
      "Fibroblasts": "#199e70", "Endothelial Cell": "#036a64", "Muscle Cell": "#5dbc7e", "Minor Stromal Cell": "#0e869a",
      "Lymphocytes": "#d95926", "Macrophages": "#9a2501", "Neutrophils": "#e78d14", "Plasmocytes": "#c63f6a",
      "Eosinophils": "#b49703",
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

export function rgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}
