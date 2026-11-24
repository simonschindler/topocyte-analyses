# Liver cell viewer

A browser viewer of the GTEx liver cohort's cells (594 slides, 126,219,124
cells). It shows any slide's point cloud, on its H&E where the slide file is at
hand, coloured by HistoPLUS cell type, by the cell-type study's four classes, by
10-nearest-neighbour or Voronoi density, by the homotypic fraction or local
entropy of its nearest cells and their counterparts against the slide's
composition, by its hypergeometric colocalization excess, or by clusters from
three of the studies here. It was built for the
[chromatic-domains pilot](../gtex-liver-chromatic-domains/README.md) on
2026-11-23 and moved to this folder on 2026-11-24, since the later studies use it
too.

It is adapted from `tissuegeometry/web/cosine_viewer`: that app's OpenSlide
DeepZoom tiles and WebGL point overlay, without its FastAPI backend, its
local-homology endpoints and its tissuegraphs and Rust imports. The server is the
Python standard library's; the page is OpenSeadragon 4.0.0 with a WebGL point
layer. The viewer reads everything read only and writes nothing.

## Run

From this folder:

```bash
../gtex-liver-chromatic-domains/.venv/bin/python serve.py                             # point clouds only
../gtex-liver-chromatic-domains/.venv/bin/python serve.py --wsi-dir ~/data/GTEX/wsi   # plus the H&E
```

When the server prints `open http://127.0.0.1:8765/  (Ctrl-C stops)`, open that
address in a browser; `--port` and `--host` change it. Ctrl-C in the terminal
stops the server. The page loads OpenSeadragon from cdnjs, so the browser needs
the internet. Python reads `serve.py` once, when the server starts, but the page
is read from disk at every load: after an update, stop the server and start it
again. The page checks this (`API` in both) and says so rather than failing
on its own.

- **Environment.** The viewer has no venv of its own. It runs in the
  chromatic-domains study's venv (Python 3.13; that study's README gives the
  recipe), whose `requirements.txt` holds everything the viewer imports: numpy,
  pandas, h5py, scipy and scikit-learn, and openslide-python with openslide-bin
  for `--wsi-dir`. It also imports that study's `common.py`, for the cohort's
  path, the four classes and the pilot's constants.
- **H&E.** With `--wsi-dir DIR`, a slide whose file `<slide_id>.svs` (or another
  format OpenSlide reads) is in DIR is shown on its H&E, and the server prints how
  many cohort slides have one; the other slides are drawn on a blank canvas. The
  cells' coordinates are the slide's level-0 pixels.
- **Tests.** `../gtex-liver-chromatic-domains/.venv/bin/python -m pytest -q -W error`
  runs `test_viewer.py` and `test_local_colocalization.py`: 51 tests on
  synthetic inputs, about 3 s.
- **Colocalization scale.** The colocalization excess needs
  `results/colocalization_scale.json`, which `fit_colocalization.py` writes from
  every cohort cell (reading the cohort read only). With the same venv, from this
  folder: `../gtex-liver-chromatic-domains/.venv/bin/python fit_colocalization.py`
  (6.7 min and a peak RSS of 3.5 GB on the M4; the JSON is tracked). Without it,
  that colouring is off.

## What it shows

### Slides and cell types

The slide list is filtered by pathology group (the pathology study's eight
groups in `pathology.csv`). The cells are coloured by HistoPLUS cell type or by
the four classes (epithelial, stromal, immune, other). The 14 type colours are
shades of their class's colour, so the legend's toggles and its "only" buttons
tell neighbouring types apart. The scale bar in µm (1, 2 or 5 × 10^k) uses the
slide file's pixel size where there is one, else the cohort's assumed 0.4942 µm
per pixel; the status line says which.

### Density and the threshold slider

"10-NN density" colours every cell by its 10-nearest-neighbour density from the
pathology study's `density.npz` (cells per mm², at the assumed 0.4942 µm per
pixel), binned at the cohort's ten deciles. The deciles are the density slices'
thresholds, so a cell in the bin that starts at decile j is kept in slices 0 to
j. The legend gives each bin's range and its cells on the slide. The blue scale
runs light to dark (dark to light on the dark canvas), so low densities fade
into the background. The densities are memory-mapped from the uncompressed
`density.npz` rather than loaded (1 GB for the cohort).

"Voronoi density" colours every cell by 1 / the area of its Voronoi cell among
the slide's cells, in cells per mm² at the same 0.4942 µm per pixel, in the same
ten bins (values below the second decile fall in the first bin, values from the
top decile on in the last), so the two colourings compare bin for bin. The
server computes it for the shown slide on first use, exactly, from the Delaunay
triangulation (`scipy.spatial.Delaunay`): a bounded Voronoi cell is the polygon
of the circumcentres of the triangles around its cell, of area
(1/8) Σ (cot α + cot β) |ab|² over the cell's Delaunay edges ab, where α and β
are the angles opposite ab (`voronoi_areas`; the tests check it against scipy's
Voronoi polygons and on square and hexagonal lattices). A cell on the convex
hull of the slide's cells has an unbounded Voronoi cell, and cells at one
position share theirs: these have no value and are drawn grey ("no Voronoi
cell"). On the largest slide (725,038 cells) the densities took 2.4 s, 44 cells
had none, and the median was 3,261 cells per mm², against 3,060 for the 10-NN
density. Cells next to empty space, at the tissue's edges or around a gap in it,
have large Voronoi cells and so low densities.

The slider under the legend hides every cell below a threshold of the attribute
the cells are coloured by; its label names it, and the note under it gives the
threshold and the cells shown.

- **10-NN density**, and under a colouring with no number to it (type, class,
  clusters): the slider runs over the cohort's density quantiles in steps of
  0.1 percentile (the pathology study's quantile curve), and its ten marked
  steps are the deciles, so at those steps it shows exactly the cells of
  density slices 0 to 9.
- **Every other value colouring** (Voronoi density and the four neighbourhood
  values): the slider runs over the shown slide's own quantiles of that value,
  in the same steps, its marks at the slide's deciles; a cell is shown when its
  value is at or above the threshold, as the page compares them (both float32),
  and a cell without a value (no Voronoi cell) is hidden by any threshold.

Choosing a colouring with another attribute starts it with no threshold; the
step otherwise stays, also from slide to slide.

### Neighbourhood composition

"Homotypic fraction" and "Local entropy" describe each cell's k nearest other
cells (k = 15 by default, 2 to 100, set under the colour choice; `cKDTree`, the
cell itself left out, also where another cell shares its position), by
HistoPLUS type or by class (the menu beside k):

- **Homotypic fraction:** the share of those neighbours of the cell's own type
  (or class): 1 inside a sheet of its own kind, about 0.5 on a border between
  two, 0 for a lone cell among others.
- **Local entropy:** the Shannon entropy of the neighbours' types (or classes),
  divided by the log of the number of types (14) or classes (4): Pielou's
  evenness, 0 when every neighbour is of one kind and 1 when every kind is there
  in equal shares.

The pair tells apart a cell in a core of its own kind (fraction 1, entropy 0), a
lone infiltrator (0 and 0), a cell on a border between two kinds (about 0.5, and
log 2 / log K) and one in a mixed hub (low, high). Both are coloured in bins of
0.1 on the densities' blue scale; the homotypic fraction is binned from its
counts of neighbours, so no fraction lands in a bin by rounding. A fixed k gives
every cell as many neighbours, so the entropy does not shrink where cells are
sparse, as it would within a fixed radius. The server computes both for a slide,
k and label set in one request on first use, with the two below: 0.65 s on the
largest slide (725,038 cells) at k = 15 by type.

Neither takes into account how common each type is on the slide: where
hepatocytes are 70% of the cells, they have a high homotypic fraction by
chance alone. Two more colourings measure against the slide's composition P,
the share of its cells of each type (or class):

- **Homotypic excess:** the homotypic fraction less P of the cell's own type, so
  0 is what chance gives; above 0, more of its own kind around the cell than
  the slide's mix, below 0, fewer. It is coloured on a diverging scale, red
  above and blue below, in 11 bins of 0.2 with chance (−0.1 to +0.1) in the
  middle, in the dataviz neutral grey. The scale's blue arm resamples the
  reference blue ramp and its red arm takes the same lightness and chroma at
  the reference red's hue (`palette.js` gives the checks).
- **KL divergence:** the Kullback–Leibler divergence D(p ‖ P) of the
  neighbours' composition p from P, in nats: 0 for the slide's own mix, and
  log(1/s) for neighbours all of one kind that makes up a share s of the slide
  (0.69 for half the slide, 2.3 for a tenth). It is computed exactly as the
  cross-entropy −(1/k) Σ log P of the neighbours' types less their entropy, and
  coloured on the blue scale in bins of 0.25 nats, the last from 2.25 on. On
  seven slides, healthy to cirrhotic, the medians were 0.09 to 0.6 nats and
  the 99th percentiles 1.1 to 2.7 (by type or by class, k = 15).

### Colour maps and bins

Under the colour choice, the six value colourings (the two densities and the four
neighbourhood values) share two menus:

- **colours:** *default* (the blue scale, and the diverging one for the
  homotypic excess), *blue*, *viridis* (matplotlib 3.11.2's) or *diverging* (blue
  to red about a neutral grey, as for the excess). Blue and the diverging map are
  resampled in OKLab where a layer needs another number of steps than their own
  (10 and 11).
- **bins:** *fixed*, each colouring's own bins as described above, or *slide
  percentiles*: 10 equal steps (11 for the diverging map, so that one sits in
  the middle) between two percentiles of the shown slide's values, 2 and 98 by
  default, the first and last steps also taking the values beyond. A diverging
  map on the homotypic excess widens the range to be even about 0, so chance
  stays in the neutral step. The percentiles are taken over all the slide's cells
  with a value, whatever the slider hides; the legend and the note give the
  steps' values.

Changing either resets the legend's toggles of the value colourings, since the
bins change with them. The slider and the class curves keep their own scales.

### Colocalization excess

"Colocalization excess" colours every cell by the hypergeometric count excess of
its k nearest other cells (`local_colocalization.py`), adapted from a
radius-graph specification after spicyR v2's conditional random labelling and
room normalisation to the viewer's k nearest neighbours (Simon, 2026-11-24):

- **The null.** For a cell of type B and a neighbour type A, the count o of A
  among its k neighbours is set against random labelling of the slide's other
  N − 1 cells, the cell's own position, label and k held fixed: hypergeometric,
  with q = n′ / (N − 1), where n′ is the slide's count of A less the cell itself
  when A = B, mean e = k q and variance v = k q (1 − q) (N − 1 − k) / (N − 2).
- **The scale.** Y = (o − e) / R, where the room R is k (1 − q) on the gain scale
  and k q on the loss scale. Each (B, A) pair's scale is fixed by the sign of its
  excess summed over all 594 cohort slides at the same k (`fit_colocalization.py`,
  for k = 2 to 100 and both label sets), never cell by cell or slide by slide.
  Y is not clipped, so that its null mean stays 0.
- **Shrunk or raw** (the menu beside it): shrunk, the default, pulls each cell's
  Y toward its (B, A) group's pooled excess over room on the slide, by
  τ² / (τ² + s²) with s² = v / R² and τ² the group's DerSimonian–Laird
  between-cell variance (0 below 5 valid cells), so noisy estimates count less.
  Raw is Y itself; a cell without an estimate, where the null has no variance
  (A missing among the slide's other cells, or all of them), is grey.
- **Excess of** (the menu): the cell's own type, the default, or any one type or
  class. k and the label set are the neighbourhood controls'.

It is coloured in 11 bins of 0.2 about 0 (−0.1 to +0.1 in the neutral grey) on
the diverging scale, or as the colour menus set. The panel at the top right gives
the shown slide's pooled excess by focal type (rows) and neighbour type
(columns): the precision-weighted mean of the cells' raw Y (`summarize_cloud`),
on the diverging scale even about 0 up to the largest value; hovering a square
gives its value with its standard error, the median shrunk Y and the pair's
scale. The server computes a slide, k and label set on first use (about 1.3 s on
the largest slide, 725,038 cells, at k = 15 by type) and keeps the last two.

### Pathology notes and donor

The box under the slide list shows the slide's pathology group, its donor's sex,
age bracket and Hardy scale (the circumstances of death, from the cohort table
`liver_cohort.csv`), its GTEx pathology categories and notes, and the cohort's
quality flag where there is one; the slide list shows the categories and notes
as a tooltip.

### Class curves

The panel in the slide view's bottom-right corner (it can be closed and
reopened) draws, for the shown slide, how each class thins out as the 10-NN
density threshold rises, on the slider's scale for that density, with the
slider's threshold marked while the slider is on the 10-NN density. Its menu has
three views:

- **stacked** (the default): of the slide's cells, the share that is of each
  class and at or above the threshold, stacked, so the stack starts at 1 and its
  height is the share of the slide's cells kept;
- **among the cells kept:** the share of each class among the cells at or above
  the threshold (the curves add up to 1; faded where fewer than 1% of the
  slide's cells are left);
- **each class's own cells:** the share of each class's cells at or above the
  threshold (each curve falls from 1).

Hovering gives the values at a threshold. The counts are computed in the
browser from the densities and bins the slider uses, by the same rule, so at
the density slices they are the studies' kept counts. The
[class-curve study](../gtex-liver-class-curves-fpca/README.md) runs a functional
PCA on the counts behind them, the cells of each class at or above every
threshold.

### Clusters

- **Composition clusters on this slide** runs k-means with the chosen k (2 to
  10) on every cell of the shown slide: the chromatic pilot's level-1 features
  of the cell's ball (log10 cells, unless "with density" is off, and the four
  class fractions; its `features.level1`), z-scored on that slide, with its
  `domains.py`'s `KMeans(k, n_init=20, random_state=0)`. The ball's radius is
  10 to 500 µm, 150 µm (the pilot's) by default. Clusters are numbered by size,
  largest first, and the legend shows each cluster's mean composition. The first
  run at a radius counts the balls; on the largest slide (725,038 cells) k = 10
  at 150 µm took 13 s, and k = 4 took 4.6 s at 500 µm and 3.0 s at 50 µm.
- **Chromatic TDA of the pilot's centres**, on the pilot's 30 slides, clusters
  the slide's 1,500 sampled centres instead, by their saved level-2 features
  (the pilot's `results/features.npz`, `L2_obs_all`: 20 MFA scores of the
  residual chromatic Betti curves, 150 µm balls, 19 random labellings), with
  `domains.py`'s k-means (`kmeans_fit(L2_obs_all[c], k)`). The centres are drawn
  larger over the other cells in grey, and the legend shows each cluster's mean
  ball composition. The pilot found no subdomains that recur across the slide
  halves, so these clusters are exploratory.
- **Cohort e-means clusters (saved)**, with the results of
  [gtex-liver-emeans-neighbourhoods](../gtex-liver-emeans-neighbourhoods/README.md)
  in place, colours every cell of the shown slide by its saved e-means label for
  the chosen feature set (composition, density or both) and k (2 to 10): that
  study's `results/labels_<set>.npz`, fitted to the 150 µm balls of all
  126,219,124 cohort cells, nothing refitted. The legend's `#j` is the saved
  label (as in that study's report), the colours go to the largest clusters over
  the cohort first, and each entry's tooltip gives the cluster's cohort profile
  (share of cells, cells per ball, class fractions, share of the ball's cells at
  or above the cohort's median density). The set menu shows Tibshirani's k̂; at
  the four (set, k) whose e-means fit is worse than k-means++'s
  (`kmeans_check.json`) the note says by how much. A set and k is read from its
  file on first use; the last three stay in memory (63 MB each).

## What it reads

All read only:

| file | what the viewer takes from it |
|---|---|
| `~/data/gtex_liver_cells/gtex_liver_cells/gtex_liver_cells.h5ad` (h5py mode "r") | each cell's slide, HistoPLUS type and position (`obsm/spatial`, level-0 pixels) |
| `liver_cohort.csv`, beside the `.h5ad` | each donor's sex, age bracket and Hardy scale |
| `../gtex-liver-pathology-slices/results/density.npz` | each slide's cohort rows, every cell's 10-NN density (memory-mapped), the deciles and the quantile curve |
| `../gtex-liver-pathology-slices/results/pathology.csv` | each slide's group, GTEx pathology categories and notes, and quality flag |
| `../gtex-liver-chromatic-domains/results/pilot.csv` and `features.npz` | the 30 pilot slides and their sampled centres' features |
| `../gtex-liver-emeans-neighbourhoods/results/labels_<set>.npz`, `clusters.json`, `gap.json`, `kmeans_check.json` | the saved e-means labels and their summaries; without them, that option is off |
| `results/colocalization_scale.json` | the cohort's summed excess per k, label set and (focal, neighbour) pair, which fixes each pair's scale; without it, that option is off |
| the slide files in `--wsi-dir` | the H&E tiles (OpenSlide DeepZoom, 256 px, no overlap) |

## Code

- `serve.py`, the server. `/` is the page and `/static/<file>` its scripts. The
  data endpoints are `/api/meta` (the groups, types, classes, settings, density
  deciles and slide table), `/api/slide/<id>`, `/api/cells/<id>` (positions and
  types, binary), `/api/density/<id>` (each cell's density and its bin,
  binary), `/api/voronoi/<id>` (each cell's Voronoi density and its bin,
  binary), `/api/neighbourhood/<id>?k=&by=` (each cell's homotypic fraction,
  local entropy, homotypic excess and KL divergence, and their bins, binary),
  `/api/colocalization/<id>?k=&by=&of=&score=` (each cell's colocalization
  excess and its bin, binary), `/api/colocalization/<id>/summary?k=&by=` (the
  pooled matrix), `/api/cluster/<id>?k=&density=&radius=`, `/api/tda/<id>?k=`,
  `/api/emeans/<id>?set=&k=` and `/api/tile/<id>/<level>/<col>_<row>.jpeg`. The
  last four slides' cells, Voronoi densities, neighbourhood values and slide
  files stay in memory, and the last two colocalization excesses.
- `local_colocalization.py`: `LocalColocalizationExtractor` (fit, transform,
  summarize_cloud), numpy and scipy only, with `test_local_colocalization.py`
  (the null by Monte Carlo over 200,000 relabellings, self-exclusion, small
  clouds, cells at one position, background cells, the frozen fit, shrinkage,
  the cohort sums); `fit_colocalization.py`, the cohort fit.
- `static/`: `index.html` (the page and its styles), `app.js` (the slide list,
  the controls and the OpenSeadragon canvas), `overlay.js` (the WebGL point
  layer, from cosine_viewer's `WsiGLOverlay`), `curves.js` (the class-curves
  panel), `matrix.js` (the colocalization matrix) and `palette.js` (the colour
  maps).
- `test_viewer.py`, the server's tests.
