# What the topocyte analyses have done, and what they have taught us

A self-contained recap of every study in this repository up to 2026-11-23, for a
reader who has not met the project before. Each study's own folder has the full
detail: a README with the commands, `progress.md` with every decision in the
order it was made, and a `report.html` with the figures (kept out of git).

## The question behind all of it

Liver disease changes how cells are arranged in tissue. A pathologist sees this
in an H&E slide as scarring, nodules, fat, inflammation. The studies here ask
whether a mathematical description of that arrangement, the *topology* of the
cells' positions, tells us something about the tissue that two much simpler
descriptions do not:

- **density:** how tightly the cells are packed;
- **composition:** which kinds of cells are present, and in what proportions.

The topology is computed with [topocyte](https://github.com/rendeirolab/topocyte),
a Python package for the persistent homology of cells in tissue. The studies
that use it, and that do not belong in the package itself, live in this
repository.

The short answer so far: **at the level of whole slides, topology in the forms
tried here re-describes density and composition rather than adding to them.**
The most compact description that separates the liver conditions is the
composition of the cells at each density level. Locally, in 150 µm
neighbourhoods, the arrangement of cell types defines no recurring domains,
and neighbourhood composition varies as a continuum. Across the whole cohort,
three broad neighbourhood types emerge from composition and density together.
The sections below say how each of these was found and how firm it is.

## Background

### The data: the GTEx liver cohort

[GTEx](https://gtexportal.org) (the Genotype-Tissue Expression project)
collected tissue from post-mortem donors, including H&E-stained whole-slide
images of liver. Every cell on those slides was detected and classified by
HistoPLUS, a cell detector and classifier for H&E trained on tumour tissue. It
gives each cell a position and one of 14 types. The studies group the types
into four classes:

| class | HistoPLUS types |
|---|---|
| epithelial | Epithelial, Cancer cell |
| stromal | Fibroblasts, Endothelial Cell, Muscle Cell, Minor Stromal Cell |
| immune | Lymphocytes, Macrophages, Neutrophils, Plasmocytes, Eosinophils |
| other | Red blood cell, Apoptotic Body, Mitotic Figures |

The cohort holds **594 slides with cells, 126,219,124 cells in all**: about
212,000 per slide, at most 725,038. The cohort records no pixel size; every
slide file checked so far has 0.4942 µm per pixel, so the studies assume that
value and say so.

Each slide gets one pathology group from GTEx's "Pathology Categories". Where a
slide has several categories, the most severe wins, in the order cirrhosis >
fibrosis > inflammation/hepatitis > necrosis > steatosis > congestion:

| group | slides |
|---|---|
| healthy (curated: the pathology note reports no finding) | 66 |
| congestion | 155 |
| steatosis (fatty liver) | 202 |
| inflammation/hepatitis | 12 |
| fibrosis | 50 |
| cirrhosis | 57 |
| necrosis | 18 |
| unclassified (no category: a free-text finding or a preservation warning) | 34 |

### Topology of a cloud of cells

topocyte treats a slide as a cloud of points in the plane, one per cell, and
computes its **persistent homology**:

- **Growing discs.** Imagine a disc of radius r around every cell. As r grows,
  discs touch and merge. topocyte does this with the *alpha complex*, the exact
  version of this construction on the cells' Delaunay triangulation.
- **H0 (components).** At small r every cell is its own component; components
  merge as r grows. Each merge ends an H0 *bar*, and the radius at which it
  ends measures the gap that was bridged.
- **H1 (loops).** Some rings of cells close around empty space before the
  space fills in, for example a ring of cells around a lumen. Each such loop
  is an H1 bar, born when the ring closes and dying when the hole fills.
- **Betti curves.** The number of components (β0) or loops (β1) alive at each
  radius. A slide becomes two curves.
- **fPCA (functional PCA).** The curves of all slides are compared, and their
  main modes of variation give each slide a few scores, like an ordinary PCA
  of curves. topocyte computes the inner products between curves exactly from
  the bars' ends. It has two variants:
  - **raw:** the curves as they are, so their size (the number of cells)
    counts;
  - **cosine:** each curve scaled to unit length first, so only its shape
    counts.

A typical classification feeds a slide's first four fPCA scores to a logistic
regression, cross-validated and repeated over 20 random splits.

### Density, the confounder that runs through everything

Disease changes how densely the cells are packed. Persistent homology is
sensitive to exactly that: bar lengths depend on the spacing of the cells, and
bar counts on their number. So every study here measures topology *against
density*.

- **Density of a cell:** 10 divided by the area of the smallest disc around the
  cell that holds its 10 nearest neighbours (cells per mm²).
- **Density slices:** ten thresholds at the deciles of that density over all
  cells. Slice j keeps the cells at or above threshold j: slice 0 is every
  cell, slice 9 the densest tenth. Each slice gets its own persistent homology.
  These are the horizontal slices of the *function-Delaunay bifiltration*
  (density × radius; Alonso et al., SODA 2024).
- **Density summaries** of a slide, used as controls: for example the fraction
  of its cells kept at each slice, its median density, or the share of its
  cells in the densest decile.

Several studies combine many such views, such as 10 slices × 4 fPCA models.
They use **multiple factor analysis (MFA)**: each block of scores is scaled so
that no block dominates, then one PCA runs over all of them.

### Neighbourhoods

The local studies take the closed disc of radius **150 µm** around each cell
(303.5 pixels; the cell itself included) and describe it by:

- its **composition**, the fractions of the four classes among its cells;
- its **density profile**, the fraction of its cells kept at each density
  threshold.

## The studies in order

| dates | study | design | in one line |
|---|---|---|---|
| 2026-11-16/17 | [delaunay-density-slices](delaunay-density-slices/README.md) | verdict | density slices do not beat density for healthy vs cirrhosis |
| 2026-11-18 | [gtex-liver-pathology-slices](gtex-liver-pathology-slices/README.md) | descriptive | all 594 slides: PC1 is density, and cirrhosis separates along the densest-decile share |
| 2026-11-18/19 | [gtex-liver-pathology-pssk](gtex-liver-pathology-pssk/README.md) | descriptive | another kernel on the persistence diagrams gives the same picture |
| 2026-11-19/20 | [gtex-liver-celltype-slices](gtex-liver-celltype-slices/README.md) | verdict and holdout | topology per cell type helps only through the cell counts |
| 2026-11-20 to 23 | [gtex-liver-chromatic-domains](gtex-liver-chromatic-domains/README.md) | verdict (pilot) | local arrangement of cell types gives no recurring domains |
| 2026-11-23 | [gtex-liver-emeans-neighbourhoods](gtex-liver-emeans-neighbourhoods/README.md) | descriptive | every cell's neighbourhood clustered; three broad types |
| 2026-11-23 | [gtex-liver-class-curves-fpca](gtex-liver-class-curves-fpca/README.md) | verdict and holdout | the fPCA of each class's density curve does not beat D+C: level by the rule, below it in every repeat |
| 2026-11-18 | [gtex-graphcode-gnn](gtex-graphcode-gnn/README.md) | planned | design and related-work survey only |

"Verdict" studies fixed their decision rule before seeing any result;
"descriptive" ones report what the data look like and decide nothing.

### 1. Density slices: is a density-sliced topology worth adding to topocyte?

**Question.** Does persistent homology computed at ten density slices separate
healthy from cirrhotic livers better than persistent homology of all cells, and
better than density alone? This was also a test of whether topocyte should
adopt the function-Delaunay bifiltration.

**Done.** The 123 slides of topocyte's cirrhosis tutorial (66 healthy, 57
cirrhosis). Per slice: H0 and H1 Betti curves, four fPCA models (H0 and H1, raw
and cosine), classification on four scores over 20 repeated splits. Each slice,
and their combination, was compared with a density-only control. Two follow-ups
came after the first verdict, sections 7 and 8 of that study's report:

- its section 7 asked whether topology adds anything beyond density at all;
- its section 8 tested, against a rule written before the run, the H1
  death/birth ratio, a summary built to escape density.

**Results.**

- Slice 0 reproduces the tutorial exactly: accuracies 0.886, 0.780, 0.886 and
  0.886 for the four models at seed 0. Always guessing the larger group gives
  0.537.
- No slice and no combination of slices beats both slice 0 and its density
  control by more than the noise. The best combined model (H0 cosine,
  0.896 ± 0.015) sits level with the density control (0.883 ± 0.011) and with
  its own slice-0 baseline (0.895).
- A ten-number density summary alone classifies at 0.909 ± 0.014, above every
  topology-only model (0.871–0.896). Adding topology scores to it gains at most
  +0.002 (95% interval −0.022 to +0.028).
- The density features explain a median 86% of the variance of the first fPCA
  score.
- Topology is not a *linear* stand-in for density. After density is regressed
  out, some signal survives, mostly at the densest slices (8–9). But that
  signal adds nothing usable to the density summary.
- The death/birth ratio (pre-registered): at slice 0, adding its four scores to
  the density arm moves accuracy from 0.915 to 0.925, +0.010 (−0.025 to
  +0.049), so by the rule it does not add. Alone it classifies well (0.887,
  AUC 0.952), but density explains 87% of its first score's variance.
- Refitting every concluding model with standardized features, after the fact,
  changed no conclusion.

**Learned.** In this cohort, for healthy against cirrhosis, slide-level
persistent homology mostly re-encodes density, and topocyte gains nothing
practical from the density bifiltration. The density control became part of
every study that followed.

### 2. Liver pathology across density slices: where do the conditions sit?

**Question.** Not a classification, a map. With all 594 slides and all eight
groups, where do the conditions sit in the fPCA space of persistent homology,
and how does that change from slice to slice?

**Done.** The ten density slices of study 1, now over the whole cohort; at each
one the four fPCA models with nine components, plotted as variance explained,
pairs of components and UMAPs coloured by group. MFA then combined all 40
slice-by-model blocks into one space, and each model family's ten slices into
four more. A check fixed in advance asked whether the separation of cirrhosis
from healthy survives removing the share of a slide's cells in the densest
decile.

**Results.**

- In every combined space, **PC1 is density**: its correlation with the slide's
  median density is 0.77–0.95 in absolute value.
- Cirrhosis and healthy sit apart along **PC2** (median gap 1.00–2.07 SD).
  PC2 hardly follows the median density, but it follows the densest-decile
  share.
- That share alone separates cirrhosis from healthy with AUC 0.933
  (0.883–0.973), better than PC2 itself (0.742–0.918).
- With the share removed, the gap survives in 2 of the 5 spaces: all 40
  blocks, AUC 0.656, and H0 cosine, 0.761. With the share, the curvature of
  PC1, the median density and the cell count all removed, no space keeps any
  separation (AUC 0.395–0.509).

**Learned.** The conditions' layout in topology space is organised by density.
The cirrhosis signal is the extra density of a slide's densest cells. The
topology finds it, but one density number states it better.

### 3. The same map with another kernel (PSSK)

**Question.** Does the conclusion depend on how persistence diagrams are
compared? Betti curves forget which birth goes with which death. The
persistence scale-space kernel (Reininghaus et al., CVPR 2015) keeps the
pairing.

**Done.** Study 2 repeated with gudhi's kernel in place of the Betti-curve inner
product, followed by kernel PCA. An exact kernel needs about 10^11 Gaussian
terms per pair of slides at slice 0, so the study used gudhi's own
approximation, 1,000 random Fourier features per dimension. Checked against the
exact kernel at slice 9, the H0 kernel is within about 1% and the H1 kernel
about 13% off.

**Results.**

- The raw models' first component matches the Betti-curve study's at every
  slice (correlation 0.92 or more); the cosine models' only at the sparsest
  slices.
- PC1 is density in 4 of 5 combined spaces.
- The cirrhosis gap on PC2 survives removing the densest-decile share in 3 of
  5 spaces (Betti curves: 2), and removing all density summaries in none.
- The H1 cosine model's PC3 and PC4 change with the random draw of the
  features, so they are not to be read.

**Learned.** Keeping the birth-death pairing does not change the picture:
density organises the space, whichever way the diagrams are compared.

### 4. One topology per cell type

**Question.** Does it help to split the cells by type and compute topology for
each group separately? A loop of immune cells is a different thing from a
loop of all cells. The comparisons were fixed before any result:

- **primary:** do all cells plus the four groups separate the seven conditions
  better than all-cell topology?
- **composition guard:** does per-group topology add anything to density
  summaries and the number of cells of each type?

**Done.**

- **Topology:** each of the four classes got its own persistent homology and
  fPCA at the same ten slices. MFA combined them in 465 ways.
- **Readouts:**
  - the macro AUC of a cross-validated logistic regression over the seven
    conditions (unclassified is not scored);
  - the *purity* of each slide's 15 nearest neighbours, relative to chance.
- **Improvement phase:** an exploratory phase tried modelling ideas on 476
  development slides. A holdout of 118 slides was set aside before any result
  and scored exactly once.
- **PLS-DA:** a supervised projection, explored in depth.

**Results.**

- **Primary: yes.** At four components, all cells plus the groups beat all-cell
  topology: macro AUC +0.106 (0.073 to 0.138), purity +0.52 (0.27 to 0.79)
  times chance.
- **Guard: no.** Per-group topology adds 0.000 (−0.023 to +0.024) to density
  and cell counts. The counts of each type at each density level separate the
  conditions as well on their own.
- **Holdout, scored once:**

  | representation | macro AUC |
  |---|---|
  | all-cell topology, 4 components | 0.627 |
  | everything (all cells and groups), 4 components | 0.715 |
  | density and composition features, no topology (D+C) | 0.774 |
  | supervised LDA of D+C, 4 components, no topology | 0.768 |
  | supervised LDA of D+C plus 20 topology components | 0.784 |
  | supervised PLS-DA of 50 topology components | 0.781 |

  No idea beat D+C: the best was +0.010 (−0.027 to +0.051). Study 7 later
  scored the same 118 slides a second time, for the class curves, so a
  further study on them would take a third look.
- **PLS-DA in depth** (development slides only): nested cross-validation gives
  0.773 (0.734 to 0.810), never beyond composition. Its axes are mostly density
  and composition: D+C predicts 81% of the variance of the axis that separates
  cirrhosis. Topology with density and composition regressed out reaches 0.539.

**Learned.** In the study's own words: "For slide-level classification of these
conditions, cell-type composition across density levels is the parsimonious
representation. Any topological claim on these data should first beat it."

### 5. Chromatic domains: does the local arrangement of cell types define microanatomy?

**Question.** Move from whole slides to neighbourhoods. Around each cell, does
the arrangement of the cell types relative to one another define recurring
microanatomical domains that density, composition and pairwise contacts do not?

**Done.** A pilot on 30 slides, five from each of six conditions, with 1,500
sampled centre cells per slide and the 150 µm disc around each.

- **Level 1, compartments.** Clusters of the centres by density and
  composition alone, with no topology.
- **Level 2, subdomains.** Inside each compartment, clusters by *chromatic*
  persistent homology. This is topology that knows the colours: for each class,
  how the topology of that class's cells sits inside the topology of all cells.
  Each centre's curves were measured against 19 random relabellings of the
  same cells, so that only arrangement beyond chance counts.
- **Guard.** Subdomains would count only if contacts between class pairs,
  density and composition could not predict them.

The rule for the number of clusters, fixed in advance, had two steps:

1. k is where the **gap statistic**, measured against a Gaussian with the
   data's mean and covariance, stops rising.
2. **Prediction strength** must then confirm that the split recurs between two
   halves of the slides.

A control on shuffled labels had to find nothing. Computing the persistent
homology of every centre took 3.3 hours and 9.3 GB of bars.

**Results.**

- **Level 1:** in each half the gap rises all the way to k = 10, but no split
  recurs across the halves. Prediction strength is 0.327 and 0.276 at k = 10,
  and at best 0.650 and 0.926 at k = 2, below the 0.8 the rule asks of both.
  So there is one stratum.
- **Control:** passes, finding no structure in the shuffles.
- **Level 2:** no subdomains (prediction strength 0.338 and 0.361 at k = 6, at
  best 0.737 and 0.926).
- **Verdict: does not add.** The pairwise guard had nothing to test.
- **Post hoc:** random half-and-half splits of all centres reach prediction
  strength 0.95 or more at k = 2. Clouds with no clusters at all do the same
  (0.98–0.99), so this is no sign of a split. The two slide halves themselves
  differ: half A's neighbourhoods hold more cells, by 0.35 pooled SD in log10
  cells.
- **The design's fallback fails too.** The level-1 strata were to give a second
  study its regions, and there is only one. The planned H&E atlas was not
  built (Simon, 2026-11-23), since there are no groups to compare.

**Learned.**

- Neighbourhood composition in liver is a continuum, not a set of compartments.
  At this scale, the arrangement of cell types shows no recurring structure in
  30 slides.
- The design itself held lessons. Against a matched Gaussian, bounded
  compositions always look clustered: a cloud of compositions with no clusters
  gets k = 5 from the gap rule. Prediction strength alone splits elongated
  noise. And a replication across slide halves partly tests whether the halves
  differ.

### 6. e-means clusters of every cell's neighbourhood

**Question.** Over the whole cohort, how many kinds of 150 µm neighbourhood are
there by composition, by density profile, and by both? The clustering used
e-means, a recent mini-batch k-means (Bessa et al. 2026, arXiv 2611.09501), fast
enough for 126 million cells. The number of clusters was read off the gap
statistic for k = 1 to 10.

**Done.**

- **Neighbourhoods:** every cell's disc counted (8 minutes).
- **Three feature sets:**
  - composition: four fractions;
  - density profile: nine kept fractions, since threshold 0 keeps everything;
  - both, with equal weight per block.
- **Clustering:** e-means on all 126 million cells for each k. The gap
  statistic used 20 Gaussian references of 20 million points each, and the
  whole run took 2.5 hours.
- **Checks:** a verifier passed 40 checks, among them 1,200 neighbourhoods
  recounted by brute force. A separate check compared the fits with k-means++
  (20 starts) on 500,000 cells.

**Results.**

| set | gap stops at (e-means) | with k-means++ fits |
|---|---|---|
| composition | rises to k = 10 | 10 |
| density profile | 7 | 10 |
| composition + density | 3 | 3 |

- **Composition + density, k = 3:**
  - 37% of cells in sparse parenchyma: 105 cells per neighbourhood, 78%
    epithelial;
  - 48% in denser parenchyma: 164 cells, 79% epithelial;
  - 16% in dense mixed neighbourhoods: 223 cells; 30% epithelial, 30% stromal,
    17% immune, 22% other.

  On one cirrhotic slide looked at in the viewer (GTEX-1192X-1026), 81% of the
  cells fall in the dense mixed type.
- **Density's stop at 7 is an artefact.** e-means' fit at k = 8 is 4.4% worse
  than k-means++'s. In 4 of its 27 fits e-means settled 0.8–4.4% above
  k-means++, and within 0.03% everywhere else.

**Learned.**

- Composition alone is again a continuum against the Gaussian reference.
  Composition and density together give three broad neighbourhood types.
- A fast clustering method needs its fits checked before a gap curve is read.
  The gap's error bar s_k measures only how the references vary, not how good
  each fit is.

The liver cell viewer (below) can colour any slide by these clusters.

### 7. Class-density curves: does their fPCA beat D+C?

**Question.** For one slide and one cell class, the *class curve* counts the
class's cells whose 10-NN density is at or above a threshold t, for every t;
the liver cell viewer plots it. At the ten density deciles the four class
curves give exactly the counts behind D+C's composition features. Does a
functional PCA of the whole curves separate the eight pathology groups better
than D+C? The rule was fixed before any result.

**Done.**

- **Curves:** each class curve is a Betti curve of dimension 0, so the inner
  product of two curves is exact. Each class's 594 × 594 Gram matrix was
  computed once (about 10 minutes); no curve was evaluated on a grid.
- **Folds:** the cell-type study's 476 development slides, 20 repeats of
  stratified 5-fold. In each fold:
  - an outlier filter on the training slides only: a slide is dropped when its
    distance to its group's mean curves exceeds the group's median distance by
    more than 3 × 1.4826 MADs;
  - the fPCA of each class, raw and cosine, nine components each, combined by
    MFA;
  - logistic regression on the first k MFA components, k chosen by inner
    cross-validation (4 to 51), against D+C trained on the same kept slides.
- **Holdout:** the cell-type study's 118 holdout slides, scored once.
- **Checks:** a verifier passed 17 checks, among them one fold's fPCA refitted
  with topocyte's own estimator and every AUC recounted from the saved
  probabilities.

**Results.**

| arm | development macro AUC | holdout |
|---|---|---|
| class curves, k chosen by cross-validation | 0.750 | 0.769 |
| class curves, 4 components | 0.740 | 0.765 |
| D+C on the same cleaned training slides | 0.756 | 0.779 |
| D+C on all training slides | 0.764 | 0.774 |

- **Verdict: level with D+C,** by the rule fixed in advance: the development
  difference of the repeat-averaged readout, −0.0064 (−0.0241 to +0.0116),
  contains 0, so that readout cannot tell the two apart. The holdout leans the
  same way: −0.0101 (−0.0421 to +0.0169), an interval that the holdout's groups
  of 2 and 4 slides make too narrow.
- **The filter dropped 11.7% of the training slides** in the development folds
  (from 4.5% of fibrosis's to 17.6% of necrosis's), and 55 of the 476 in the
  holdout fit. It cost D+C 0.008 on the development slides and gained it 0.005
  on the holdout.
- **Scored repeat by repeat, the curves fall below D+C in all 20 repeats,** by
  0.006 to 0.049 (0.722 against 0.749 on average). Averaging each slide's
  probabilities over the repeats lifts them by 0.028 and D+C by only 0.007,
  which narrows the averaged readout's gap. That is in line with their model
  changing from fold to fold (k from 4 to 32).

**Learned.** The fPCA of the continuous class curves does not beat D+C. The
rule calls the two level, which means that its readout cannot tell them apart,
not that they are equal: the curves scored below D+C in all 20 repeats, by
0.006 to 0.049 at the chosen k and by 0.004 to 0.033 at k = 4, and on the
holdout at both. The
whole curves show no gain over the counts at ten density deciles, and D+C
stays the representation to beat.

### 8. Planned: graphcode GNN on all GTEx slides

Design stage only. The plan:

- **Model:** a graph neural network on *graphcodes* of each slide's density
  bifiltration, with each bar decorated by the cell-type composition of its
  cells.
- **Data and targets:** all GTEx slides, with four targets: pathology, age,
  RNA expression and organ.
- **Ablation ladder:** asks whether topology adds beyond density and
  composition.

A survey of 41 related works found:

- no published study that tests topology on real tissue while controlling for
  both density and composition;
- no graphcodes or multiparameter-persistence GNNs applied to histology.

The survey and studies 1–4 point to the same main risk: the gain from topology
may vanish once density and composition are controlled. The plan's ladder
already puts composition first; the cell-type study expects that rung to be
hard to beat and advises fixing the composition guard in advance there too.

### Tooling: the liver cell viewer

[gtex-liver-viewer](gtex-liver-viewer/README.md) shows any cohort slide's cells in
the browser, on the H&E where the slide file is available. Cells can be
coloured by:

- HistoPLUS type or class, with slides filtered by pathology group;
- each cell's 10-NN density, in the cohort's ten deciles (the density slices),
  or its Voronoi density (1 / the area of its Voronoi cell) in the same bins;
- the homotypic fraction and the local Shannon entropy of its k nearest cells,
  by type or by class, and against the slide's composition their excess over
  chance and their Kullback–Leibler divergence;
- composition clusters computed on the fly for one slide;
- the chromatic-domains pilot's TDA features;
- the saved e-means clusters of study 6.

A slider hides the cells below a threshold of the colouring's value (the 10-NN
density under a type, class or cluster colouring), a panel plots how each class
thins out as the 10-NN density threshold rises, and a box shows the
slide's GTEx pathology notes and its donor's sex, age bracket and Hardy scale.
It reads everything read only and writes nothing.

## What we have learned

### About liver tissue and topology

1. **Density comes first.** In every slide-level study, the first axis of the
   topology space is density. A ten-number density summary classifies healthy
   against cirrhosis at 0.909, better than any topology-only model (0.871 to
   0.896). The cirrhosis signal in the topology spaces is the share of a
   slide's cells in the cohort's densest decile.
2. **Composition comes second.** Splitting the cells by type helps, but through
   the number of cells of each type at each density level, not through their
   topology. Density plus composition (D+C) is the representation to beat:
   0.774 on unseen slides, and nothing with topology beat it beyond the noise.
   Nor does composition gain from its functional form: the fPCA of each class's
   whole curve over density does not beat D+C's 51 features (study 7).
3. **Topology is not a mere copy of density.** Some signal survives regressing
   density out, mostly among the densest cells. It just does not improve the
   classifications tried here.
4. **The choice of topological summary does not change this.** Betti curves or
   a pairing-aware kernel, raw or shape-only curves, all cells or per type: the
   conclusions stand.
5. **Locally, there are no discrete domains of arrangement.** Neighbourhood
   composition is a continuum. The chromatic topology of 150 µm neighbourhoods
   found no recurring subdomains. Over the whole cohort, composition and density
   together give three broad neighbourhood types; this is descriptive, and
   anatomical labels for them remain to be checked on H&E.

These findings concern this cohort, these labels (GTEx pathology categories,
which are coarse) and these representations. They say that topology has not yet
shown added value here, not that it cannot.

### About method

- **A density control and a composition control belong in every comparison.**
  They are cheap, and here they were decisive.
- **Fix the rule before the result, and keep a holdout you score once.** Study 4
  held 118 slides back from the start, and study 7 reused them for its own
  single look. Several rules were written before any
  run: studies 4, 5 and 7 for their verdicts, study 1 for the death/birth ratio,
  and studies 2 and 3 for the PC2 check. This kept the conclusions from
  drifting with the data.
- **Cluster-number rules can be fooled.**
  - A gap statistic against a matched Gaussian calls any bounded or skewed
    cloud clustered.
  - Prediction strength alone splits elongated noise.
  - Replication across slide halves partly measures how the halves differ.
  - Strata are not modes, and the reports say which they show.
- **Check the fits behind a curve.** e-means settled into worse optima at some k,
  and one stop of the gap curve came from such a fit. The gap's error bar did
  not show it; a paired check against k-means++ did.
- **Scale matters in penalised comparisons.** With features on very different
  scales, an L2-penalised nested test can under- or over-weight a block. Study 1
  refitted everything with standardized features to make sure.
- **Approximations move the minor components.** The random-feature kernel of
  study 3 reproduces the leading components, but the H1 cosine model's third
  and fourth components move with the draw.
- **Verify by another route, and look at everything.** Every study has a
  `verify_results.py` that recomputes the saved results independently, from
  brute-force recounts to refits. Together with looking at every figure and
  questioning surprising numbers, this caught bugs before they reached a
  conclusion. Examples:
  - a figure title that named the wrong panel;
  - an input silently cut to its first 50 columns;
  - a k-means check that measured a different fit from the one it claimed to.

## Open threads

- **Regions for a second local study.** The chromatic pilot was to supply them,
  and does not; where they come from is Simon's decision.
- **The graphcode GNN study.** Its ablation ladder puts composition first; its
  composition guard should be fixed in advance.
- **Future directions named by the cell-type study:**
  - a null that relabels cell types at random within each slide;
  - topology *between* types (chromatic bifiltrations, type-pair filtrations);
  - local and regional analysis with attribution maps (portal, lobular,
    septal);
  - targets where arrangement is the diagnosis (fibrosis stage, nodularity,
    with pathologist-reviewed labels and an external cohort);
  - representations that keep the birth-death pairing, tested against D+C
    first;
  - liver-specific cell types (hepatocytes, cholangiocytes, Kupffer and
    stellate cells) instead of a tumour-trained classifier.
- **The four weaker e-means fits** (composition k = 4, density k = 8,
  composition + density k = 6 and 9) could be refitted from more starting
  points, so that their saved labels match k-means++ quality.

## Glossary

- **Alpha complex:** the exact construction of growing discs around points,
  built on their Delaunay triangulation; its filtration gives persistent
  homology.
- **Bar, persistence diagram:** a topological feature's birth and death radius;
  a slide's bars form its diagram.
- **Betti curve:** the number of components (β0) or loops (β1) alive at each
  radius.
- **Chromatic persistent homology:** persistent homology that tracks how the
  topology of one class's cells sits inside that of all cells.
- **D+C:** density summaries and the number of cells of each type at each
  density level; no topology.
- **Density slice:** the cells at or above one of the ten decile thresholds of
  10-nearest-neighbour density; slice 0 is every cell.
- **e-means:** a mini-batch k-means whose batch size and step are set by
  scaling laws (Bessa et al. 2026).
- **fPCA:** functional principal component analysis of curves; raw keeps their
  size, cosine only their shape.
- **Gap statistic:** compares how much clustering tightens the data with how
  much it tightens a reference without clusters; k̂ is where the gain stops.
- **Holdout:** slides set aside before any result and scored once.
- **Macro AUC:** the one-vs-rest area under the ROC curve, averaged over the
  conditions.
- **MFA (multiple factor analysis):** a PCA over several blocks of scores,
  each scaled so that no block dominates.
- **PLS-DA, LDA:** supervised projections that look for the directions
  separating the labelled groups.
- **Prediction strength:** how well clusters found on one half of the data
  predict the clusters of the other half.
- **PSSK:** the persistence scale-space kernel, which compares diagrams bar by
  bar.
- **Purity ×chance:** the share of a slide's 15 nearest neighbours that share
  its group, divided by the share expected by chance.

## Where things are

- **Studies:** one folder each, with:
  - a README giving the run order and the topocyte commit used;
  - `progress.md`, the ledger of every ruling and result;
  - `verify_results.py` and `build_report.py`.
- **Not in git:** large outputs (`*.npz`, `*.npy`, reports and logs). The
  scripts regenerate them, except where a study's README says that its holdout
  was scored once and its outputs are the kept copy (study 7); the copies the
  reports were built from stay on Simon's machine and go to Zenodo at
  publication.
- **The cohort** (`~/data/gtex_liver_cells/`) is always opened read only.
- **Visibility:** the repository is private until the topocyte paper is
  published. As of 2026-11-23 its commits are on local `main`, and Simon
  pushes them.
