# GTEx graphcode GNN (planned)

Design stage: nothing is built yet. The planned study trains a graph neural
network on graphcodes of each slide's cell density bifiltration, with every bar
decorated by the cell-type composition of its cells, on all GTEx slides with
four heads (pathology categories, age, RNA expression, organ). An ablation
ladder asks whether topology adds anything beyond density and cell-type
composition.

## Contents

- `related_work.md`: literature survey compiled on 2026-11-18 by a Claude
  subagent through web search, 41 entries, each with its verification status.
  Two entries were re-checked against their abstracts afterwards: Natarajan
  et al. 2026 (arXiv 2603.03237) and Ali et al. 2025 (Nat Commun 16:8419).
