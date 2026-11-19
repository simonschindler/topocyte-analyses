# topocyte analyses

Studies and benchmarks run with [topocyte](https://github.com/rendeirolab/topocyte)
that do not belong in the package repository: the code that produced a result,
the ledger of how it was decided, and the builder of its report. The repository
is private until the topocyte paper is published; then it becomes public and the
large outputs are deposited on Zenodo.

## Layout

One folder per study. Each holds

- its scripts, run in the order its README gives;
- `progress.md`, the ledger: every ruling, with what it costs if wrong, and
  every result, in the order they were made;
- `verify_results.py`, which checks the saved results, and `build_report.py`,
  which writes `report.html` from them;
- `results/*.json`, the small metadata of each run (runtime, library versions).

Generated outputs stay out of git: `*.npz`, `*.html`, `logs/`, `figures/` and
`superseded/`. The scripts regenerate them; each study's README says where the
copy its report was built from is kept.

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
