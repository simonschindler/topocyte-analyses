# Liver cell viewer

A browser viewer of the GTEx liver cohort's cells (594 slides, 126,219,124
cells). It shows any slide's point cloud, on its H&E where the slide file is at
hand, coloured by HistoPLUS cell type, by the cell-type study's four classes, by
10-nearest-neighbour density, or by clusters from three of the studies here. It
was built for the
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
the internet.

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
  runs `test_viewer.py`: 22 tests on synthetic inputs, about 2 s.

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

The slider under the legend hides every cell below a density threshold, in any
colouring. It runs over the cohort's density quantiles in steps of 0.1
percentile (the pathology study's quantile curve), and its ten marked steps are
the deciles, so at those steps it shows exactly the cells of density slices 0
to 9; the note under it gives the threshold and the cells shown.

### Pathology notes and donor

The box under the slide list shows the slide's pathology group, its donor's sex,
age bracket and Hardy scale (the circumstances of death, from the cohort table
`liver_cohort.csv`), its GTEx pathology categories and notes, and the cohort's
quality flag where there is one; the slide list shows the categories and notes
as a tooltip.

### Class curves

The panel in the slide view's bottom-right corner (it can be closed and
reopened) draws, for the shown slide, how each class thins out as the density
threshold rises, on the slider's scale with the slider's threshold marked. Its
menu has three views:

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
| the slide files in `--wsi-dir` | the H&E tiles (OpenSlide DeepZoom, 256 px, no overlap) |

## Code

- `serve.py`, the server. `/` is the page and `/static/<file>` its scripts. The
  data endpoints are `/api/meta` (the groups, types, classes, settings, density
  deciles and slide table), `/api/slide/<id>`, `/api/cells/<id>` (positions and
  types, binary), `/api/density/<id>` (each cell's density and its bin,
  binary), `/api/cluster/<id>?k=&density=&radius=`, `/api/tda/<id>?k=`,
  `/api/emeans/<id>?set=&k=` and `/api/tile/<id>/<level>/<col>_<row>.jpeg`. The
  last four slides' cells and slide files stay in memory.
- `static/`: `index.html` (the page and its styles), `app.js` (the slide list,
  the controls and the OpenSeadragon canvas), `overlay.js` (the WebGL point
  layer, from cosine_viewer's `WsiGLOverlay`), `curves.js` (the class-curves
  panel) and `palette.js` (the type, class and cluster colours).
- `test_viewer.py`, the server's tests.
