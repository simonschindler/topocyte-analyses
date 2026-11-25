# topocyte analyses

Studies and benchmarks run with [topocyte](https://github.com/rendeirolab/topocyte)
that do not belong in the package repository: the code that produced a result,
the ledger of how it was decided, and the builder of its report. The repository
is private until the topocyte paper is published; then it becomes public and the
large outputs are deposited on Zenodo.

[EXPERIMENTS.md](EXPERIMENTS.md) recaps every study below for a reader new to
the project: the background, what each study did and found, and what they
have taught us together.

## Layout

One folder per study, and one for the liver cell viewer, a tool the studies
share (see [Tools](#tools)). Each study's folder holds

- its scripts, run in the order its README gives;
- `progress.md`, the ledger: every ruling, with what it costs if wrong, and
  every result, in the order they were made;
- `verify_results.py`, which checks the saved results, and `build_report.py`,
  which writes `report.html` from them;
- `results/*.json`, the small metadata of each run (runtime, library versions).

Generated outputs stay out of git: `*.npz`, `*.npy`, `*.html`, `logs/`, `figures/`
and `superseded/`. The scripts regenerate them, except where a study's README says
that its holdout was scored once and its outputs are the kept copy
(gtex-liver-class-curves-fpca); each study's README says where the copy its report
was built from is kept.

## topocyte version

The scripts import topocyte from the source checkout named first on
`PYTHONPATH`, not from whatever the interpreter has installed, so a study's
README states the topocyte commits its results were made with. A checkout of a
given commit:

```bash
git -C /path/to/topocyte worktree add ../topocyte-bd1d5c0 bd1d5c0
```

## Studies

| study | question | verdict |
|---|---|---|
| [delaunay-density-slices](delaunay-density-slices/README.md) | Is the function-Delaunay bifiltration, sliced at density deciles, worth adopting in topocyte, and does persistent homology add to the GTEx liver classification beyond density? | No; the H1 death/birth ratio adds nothing beyond density either (pre-registered), and post-hoc refits with standardized features change no conclusion |
| [gtex-liver-pathology-slices](gtex-liver-pathology-slices/README.md) | Where do the pathologies of all 594 GTEx liver slides sit in the fPCA space of their Betti curves, at ten density slices? | Descriptive, no verdict: fractions of variance, every pair of the first four components and a UMAP per slice and model, coloured by eight severity-ranked pathology groups; combined across slices by multiple factor analysis, PC1 is density everywhere and cirrhosis sits apart from healthy along PC2, which is the densest-decile share: with it removed the gap survives in 2 of 5 spaces, with all density summaries removed in none |
| [gtex-graphcode-gnn](gtex-graphcode-gnn/README.md) | Do graphcodes of the cells' density bifiltration, decorated with cell-type composition, add to multi-task GTEx prediction (pathology, age, RNA, organ) beyond density and composition? | Planned: design stage, related-work survey only |
| [gtex-liver-pathology-pssk](gtex-liver-pathology-pssk/README.md) | What changes when the liver pathology study across density slices compares the persistence diagrams with gudhi's persistence scale-space kernel (random Fourier features) instead of the Betti-curve inner product? | Descriptive, no verdict: the raw models' leading components match the Betti-curve study's at every slice (PC1 correlation 0.92 or more), the cosine models' only at the sparsest slices; combined by multiple factor analysis, PC1 is density in 4 of 5 spaces, and the PC2 cirrhosis gap survives the densest-decile share in 3 of 5 (Betti curves: 2) and all density summaries in none; the approximate H1 kernel is about 13% off the exact one, and its cosine model's PC3 and PC4 move with the draw |
| [gtex-liver-celltype-slices](gtex-liver-celltype-slices/README.md) | Does splitting the cells by HistoPLUS cell type into four groups, each with its own alpha persistent homology at the same density slices, make the liver pathology groups easier to tell apart? | Against all-cell topology yes (fixed in advance: macro AUC +0.106 [0.073, 0.138] and purity +0.52 [0.27, 0.79] times chance at four components), but through composition: per-group topology adds nothing to density and cells per group (0.000 [−0.023, 0.024], fixed in advance), and the cell counts alone separate as well. Exploratory improvement phase, 118-slide holdout scored once: supervised four-component views (LDA, PLS-DA) beat the unsupervised everything space (holdout macro AUC up to 0.784 against 0.715) but not the density and composition features (0.774); see improvements.md. PLS-DA in depth (development slides): tuned by nested CV 0.773, never beyond composition, axes mostly density and composition; the discussion concludes that topology in this form re-describes density and composition and lists future directions. Its 118-slide holdout has since been scored once more, by the class-curve study |
| [gtex-liver-chromatic-domains](gtex-liver-chromatic-domains/README.md) | Does the local chromatic persistent homology of 150 µm neighbourhoods define microanatomical domains in GTEx liver beyond density, composition and pairwise contacts? | Pilot of 30 slides, rule fixed in advance: does not add. Level 1 gives one composition stratum: each half's gap statistic rises to k = 10, but no split recurs across the slide halves (prediction strength 0.327 / 0.276 at k = 10, at best 0.650 / 0.926 at k = 2); the control passes; level 2 gives no subdomains (0.338 / 0.361 at k = 6, at best 0.737 / 0.926), so the pairwise guard had nothing to test. Post hoc, on random half/half splits of all centres prediction strength at k = 2 is 0.95 or more at both levels, as on cluster-free clouds (0.98–0.99 in design.md and progress.md): no sign of a split, only that the two slide halves differ (half A's balls hold more cells, by 0.35 pooled SD in log10 cells). The design's fallback, the level-1 strata as the regions of a second study, does not hold either; where that study's regions come from is Simon's decision. The H&E atlas is not built (Simon, 2026-11-23): one stratum and no subdomains leave nothing to compare |
| [gtex-liver-emeans-neighbourhoods](gtex-liver-emeans-neighbourhoods/README.md) | How many clusters does e-means (Bessa et al. 2026) find in the composition, the density profile and both, of every GTEx liver cell's 150 µm ball, by the gap statistic for k = 1–10? | Descriptive, no verdict: against the matched Gaussian, composition's gap rises to k = 10; composition + density stops at k = 3 (sparse parenchyma, denser parenchyma, dense mixed neighbourhoods), and k-means++ agrees; density stops at k = 7 under e-means only, because its k = 8 fit is 4.4% worse than k-means++'s, and with k-means++ it rises to k = 10. In 4 of 27 fits e-means settled 0.8–4.4% above k-means++ in W_k |
| [gtex-liver-class-curves-fpca](gtex-liver-class-curves-fpca/README.md) | Does a functional PCA of each slide's four class curves (the cells of each class at or above every 10-NN density threshold) separate the eight liver pathology groups better than D+C, the cell-type study's density and composition features? | Level with D+C, by the rule fixed in advance, which means that its readout cannot tell the two apart, not that they are equal: development macro AUC 0.750 against 0.756 for D+C trained on the same cleaned slides, a difference of −0.0064 [−0.0241, +0.0116]; the cell-type study's 118-slide holdout, scored once for the curves, leans the same way, 0.769 against 0.779 (−0.0101 [−0.0421, +0.0169], an interval that its groups of 2 and 4 slides make too narrow). The curves scored below D+C in all 20 repeats, by 0.006 to 0.049 (0.722 against 0.749 on average), and on the holdout. The outlier filter dropped 11.7% of the training slides in the development folds and 55 of 476 in the holdout fit |
| [betti-fpca-regression-benchmark](betti-fpca-regression-benchmark/README.md) | Does functional principal component regression on topocyte's exact Betti-curve fPCA scores beat the persistence-diagram kernel regression of Aggarwal, Nichols and Turner (`PD_kernel`, arXiv:2611.25295) on the paper's three tasks, both seeing the same diagrams, training sets and test sets? | No: below `PD_kernel` on all three tasks, by the rule fixed in advance, each 95% interval below 0: interpolation R² −0.02401 [−0.03837, −0.01141], curvature MSE −0.2478 [−0.4340, −0.09194] (`PD_kernel`'s minus the regression's), fluid R² −3.283 [−5.812, −1.507], whose size comes from a few near-saturated fits at the smallest training size (median difference −0.2545). `PD_kernel`, reimplemented from the paper's public code (read, never run), matches all 146 values published with that code behind the paper's Tables 1–3 to 3.6e-16, so the verdict does not rest on a weaker reimplementation. Exploratory: the paper's estimator on Betti-curve distances puts the gap in the representation for interpolation, in the estimator for curvature, and in both for fluid. A topocyte follow-up: `FPCA.transform` is inaccurate in near-null components |

## Tools

[gtex-liver-viewer](gtex-liver-viewer/README.md) is the liver cell viewer. It
shows any cohort slide's cells in the browser, on the H&E where the slide file is
at hand, coloured by HistoPLUS cell type or class, by 10-NN or Voronoi density,
by the homotypic fraction or local entropy of its nearest cells (also against
the slide's composition), by its hypergeometric colocalization excess on a
scale fitted over the cohort, by composition clusters computed on the spot, or by
the chromatic-domains pilot's TDA clusters or the e-means study's saved
clusters. The value colourings take viridis (the default), blue or a diverging
map, in fixed bins or between two percentiles of the slide's values. A slider
hides the cells below a threshold of the colouring's value (the 10-NN density
for a type, class or cluster colouring), a panel draws the slide's class curves,
and a box gives the slide's GTEx pathology notes and its donor. From its folder:

```bash
../gtex-liver-chromatic-domains/.venv/bin/python serve.py --wsi-dir ~/data/GTEX/wsi
```

then open `http://127.0.0.1:8765/`. Its README says what each colouring shows and
which files the viewer reads; it writes nothing.
