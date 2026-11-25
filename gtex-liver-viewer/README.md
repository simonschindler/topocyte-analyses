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
too. Since 2026-11-25 it also opens the per-tissue GTEx objects of
`gtex-liver-cells` (one folder per GTEx Portal tissue site), without the liver
cohort's precomputed layers ([Other GTEx tissues](#other-gtex-tissues)).

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
../gtex-liver-chromatic-domains/.venv/bin/python serve.py --tissues ~/data/gtex_tissues        # GTEx tissue folders
../gtex-liver-chromatic-domains/.venv/bin/python serve.py --tissues ~/data/gtex_tissues/lung   # one of them
```

Without `--tissues` the viewer shows the liver cohort, as it always has.

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
- **Tissues.** With `--tissues`, the viewer shows GTEx tissue folders instead of
  the liver cohort: one folder, or a folder of them, among which the page's
  *Tissue* menu chooses (it opens `/?tissue=<slug>`; without one, the first
  folder by name). The terminal lists each folder found with its slides and
  cell types, any that cannot be opened (also greyed out in the menu, with the
  reason) and any that hold only one of the two files a tissue folder needs (a
  copy still running, say). [Other GTEx tissues](#other-gtex-tissues) says what
  the page shows there.
- **Tests.** `../gtex-liver-chromatic-domains/.venv/bin/python -m pytest -q -W error`
  runs `test_viewer.py`, `test_tissues.py` and `test_local_colocalization.py`:
  79 tests on synthetic inputs, about 4 s.
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
the four classes (epithelial, stromal, immune, other), in viridis's hues to sit
with the value colourings' default map: epithelial in its blue, stromal in its
teal-green, immune in its yellow (brightest on the dark canvas, so that the rare
immune cells stand out) and other in grey. Each type takes a hue near its
class's (`palette.js` gives the checks), and the legend's toggles and its "only"
buttons tell neighbouring types apart. The scale bar in µm (1, 2 or 5 × 10^k)
uses the slide file's pixel size where there is one, else the cohort's assumed
0.4942 µm per pixel; the status line says which.

### Density and the threshold slider

"10-NN density" colours every cell by its 10-nearest-neighbour density from the
pathology study's `density.npz` (cells per mm², at the assumed 0.4942 µm per
pixel), binned at the cohort's ten deciles. The deciles are the density slices'
thresholds, so a cell in the bin that starts at decile j is kept in slices 0 to
j. The legend gives each bin's range and its cells on the slide. Its own scale
(the colour menu's *by layer*, below) is blue, light to dark (dark to light on
the dark canvas), so that low densities fade into the background. The densities
are memory-mapped from the uncompressed `density.npz` rather than loaded (1 GB
for the cohort).

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
0.1 (blue by layer, as the densities); the homotypic fraction is binned from its
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
  the slide's mix, below 0, fewer. It takes 11 bins of 0.2 with chance (−0.1
  to +0.1) in the middle; its own scale (*by layer*) is diverging, red above and
  blue below about the dataviz neutral grey for chance. The scale's blue arm
  resamples the reference blue ramp and its red arm takes the same lightness and
  chroma at the reference red's hue (`palette.js` gives the checks).
- **KL divergence:** the Kullback–Leibler divergence D(p ‖ P) of the
  neighbours' composition p from P, in nats: 0 for the slide's own mix, and
  log(1/s) for neighbours all of one kind that makes up a share s of the slide
  (0.69 for half the slide, 2.3 for a tenth). It is computed exactly as the
  cross-entropy −(1/k) Σ log P of the neighbours' types less their entropy, and
  binned in steps of 0.25 nats, the last from 2.25 on (blue by layer). On
  seven slides, healthy to cirrhotic, the medians were 0.09 to 0.6 nats and
  the 99th percentiles 1.1 to 2.7 (by type or by class, k = 15).

### Colour maps and bins

Under the colour choice, the seven value colourings (the two densities, the four
neighbourhood values and the colocalization excess) share two menus:

- **colours:** *viridis* (matplotlib 3.11.2's), the default; *blue*; *diverging*
  (blue to red about a neutral grey); or *by layer*, each colouring's own scale
  as above: blue, and diverging for the homotypic and colocalization excesses.
  Blue and the diverging map are resampled in OKLab where a layer needs another
  number of steps than their own (10 and 11). On the two excesses in fixed
  bins, viridis puts 0 in its middle step, with no neutral.
- **bins:** *fixed*, each colouring's own bins as described above, or *slide
  percentiles*: 10 equal steps (11 for the diverging map, so that one sits in
  the middle) between two percentiles of the shown slide's values, 5 and 95 by
  default, the first and last steps also taking the values beyond. A diverging
  map on either excess widens the range to be even about 0, so that 0 stays in
  the neutral step. The percentiles are taken over all the slide's cells with a
  value, whatever the slider hides; the legend and the note give the steps'
  values. The same two percentiles bound the class curves of every value
  colouring but the 10-NN density (below), so their fields show under fixed
  bins too.

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

It takes 11 bins of 0.2 about 0 (−0.1 to +0.1 in the middle), coloured as the
colour menus set (*by layer*: diverging, the middle step in the neutral grey).
The panel at the top right gives the shown slide's pooled excess by focal type
(rows) and neighbour type (columns): the precision-weighted mean of the cells'
raw Y (`summarize_cloud`), in 11 steps even about 0 up to the largest value, in
the colour map the menu gives the cells, with a key under it; hovering a square
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
reopened) draws, for the shown slide, how each class thins out as a threshold
rises on the attribute the slider thresholds, with the slider's threshold
marked:

- under the 10-NN density, type, class and cluster colourings, the 10-NN
  density on the cohort's percentiles (the slider's scale), every 10 a density
  slice;
- under every other value colouring, that value itself, in 500 equal steps
  between the two percentiles of the shown slide that the colour menus set (5
  and 95 by default), so that a long tail does not squeeze the rest; the mark
  shows while the slider's threshold falls between them.

Its menu has three views:

- **stacked** (the default): of the slide's cells, the share that is of each
  class and at or above the threshold, stacked, so the stack starts at 1 and its
  height is the share of the slide's cells kept;
- **among the cells kept:** the share of each class among the cells at or above
  the threshold (the curves add up to 1; faded where fewer than 1% of the
  slide's cells are left);
- **each class's own cells:** the share of each class's cells at or above the
  threshold (each curve falls from 1).

Hovering gives the values at a threshold. The counts are computed in the
browser from the values the slider compares, by the same rule, so at the
density slices they are the studies' kept counts. The
[class-curve study](../gtex-liver-class-curves-fpca/README.md) runs a functional
PCA on the counts behind the density curves, the cells of each class at or
above every threshold.

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

## Other GTEx tissues

`gtex-liver-cells` builds one AnnData per GTEx Portal tissue site, in the liver
object's encoding, each in a folder of its own:

```
<ROOT>/                       ~/data/gtex_tissues (on the cluster: /nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_tissues)
├── tissues.csv               the build's manifest; not read
└── <slug>/                   one per tissue: liver, lung, skin_sun_exposed_lower_leg, ...
    ├── gtex_<slug>_cells.h5ad
    ├── slides.csv            one row per slide of the object, in its row order
    ├── provenance.json       not read
    └── centroids/            not read (may be absent)
```

A folder is a tissue folder when it holds both `gtex_<its name>_cells.h5ad` and
`slides.csv`; its name is the tissue's slug, which the page's address and menu
use. At the start the server checks each folder's `slides.csv` against its
object: the nine GTEx Portal columns and `n_cells`, `row_start` and `row_end`
are there, no slide is listed twice or without cells, `n_cells` is
`row_end − row_start`, the rows run without gaps from 0 to the object's last
cell, the object names every slide, and the file names one tissue. As for
liver, a slide's cells are read only if the object names the slide at the first
and the last of its rows. A folder that fails a check is listed but cannot be
opened.

In a tissue folder the page shows, as for liver:

- **Slides.** One list by slide ID, with no pathology groups to filter by. The
  box under it gives the tissue, the donor (its Subject ID, sex, age bracket and
  Hardy scale) and the GTEx pathology categories and notes from `slides.csv`,
  and "hidden on the GTEx Portal" where the Portal's Hidden is true. The H&E
  (`--wsi-dir`, one folder of slide files for every tissue) and the scale bar
  are as for liver.
- **Cell types and classes.** The legend calls them cell types (liver's are
  HistoPLUS's) and lists those of liver's 14 that the tissue has, in their
  colours, then its others. A cell type outside the cell-type study's four
  classes, whose map holds exactly liver's 14, counts as **other**: in the class
  colouring, the class curves, the neighbourhood values by class and the
  composition clusters' class fractions. It takes pink, orange or red in the type
  colouring (`EXTRA_TYPES` in `palette.js`, hues none of the 14 take, not
  validated against them), its legend entry and the classes' "other" say so in
  their tooltips, a note under the legend names them, and the terminal does at
  the start. Liver's map is unchanged. A tissue with more than 16 cell types
  cannot be opened: the page's colour table has 16 entries.
- **Voronoi density**, at GTEx's assumed 0.4942 µm per pixel as for liver. Its
  fixed bins are the shown slide's own ten deciles, each a tenth of its cells
  with a value, since a tissue has no cohort deciles; the legend gives each
  decile's range. *Slide percentiles* bins are as for liver.
- **Neighbourhood composition and composition clusters**, unchanged: they need
  nothing but the slide's cells. The label menu offers cell types or classes.
- **The threshold slider and the class curves**, under a value colouring (the
  Voronoi density and the four neighbourhood values) unchanged: the shown
  slide's own quantiles of that value. Under the type, class and cluster
  colourings, where for liver they run over the cohort's 10-NN density
  quantiles, a tissue folder has nothing to threshold: the slider is off, its
  note says to choose a value colouring, and the class curves panel stays empty
  with the same caption. This is the simplest honest fallback; thresholding the
  Voronoi density there instead would compute it for every slide shown and put
  two different densities under one slider.

Off in a tissue folder, its control greyed out with a tooltip that says why (the
server answers 404):

| layer | why it is the liver cohort's |
|---|---|
| 10-NN density, with its cohort deciles and the slider's cohort quantiles | the pathology study's `density.npz`, computed over the liver cohort |
| pathology groups | the pathology study's `pathology.csv`, of liver slides |
| chromatic TDA of the pilot's centres | the chromatic pilot's sampled centres, on 30 liver slides |
| cohort e-means clusters | fitted to every liver cohort cell's ball |
| colocalization excess | each pair's gain or loss scale, fitted over the liver cohort (`fit_colocalization.py`) |

Checked on 2026-11-25 on a temporary root whose `liver/` linked the liver
object, with a `slides.csv` from `liver_cohort.csv` and the row ranges of the
object's `obs/slide_id/codes`: on three slides (28,530 to 195,091 cells) the
tissue folder gave byte for byte the cells and neighbourhood values that the
liver cohort gives, the same composition clusters and Voronoi densities, and
Voronoi codes with a tenth of each slide's cells per decile.

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

With `--tissues`, instead, of each tissue folder (and none of the liver cohort's
files above):

| file | what the viewer takes from it |
|---|---|
| `<slug>/gtex_<slug>_cells.h5ad` (h5py mode "r") | each cell's slide, cell type and position (`obsm/spatial`, level-0 pixels), as from the liver object |
| `<slug>/slides.csv` | each slide's rows (`n_cells`, `row_start`, `row_end`), its donor (`Subject ID`, `Sex`, `Age Bracket`, `Hardy Scale`), its `Pathology Categories` and `Pathology Notes`, its `Hidden` flag, and the `Tissue` |
| the slide files in `--wsi-dir` | the H&E tiles, as for liver |

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
  files stay in memory, and the last two colocalization excesses. With
  `--tissues` every endpoint takes `tissue=<slug>` (without it, the first
  tissue), `/api/meta` adds the tissue (`tissue`: its slug, name and cell types
  outside the four classes) and every tissue folder found (`tissues`), and the
  caches are per tissue, of which only the two used last keep theirs, so that
  the server needs no more memory for many tissues than for two. A `Cohort` is
  the liver cohort or one tissue folder; the `Library` holds what the server
  shows.
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
- `test_viewer.py`, the server's tests, and `test_tissues.py`, those of the
  tissue folders: finding them, `slides.csv`, the class map's "other", the slide's
  own Voronoi deciles, a tissue's layers and the server's routing by tissue, on
  tissue objects that it writes with h5py in anndata's encoding.
