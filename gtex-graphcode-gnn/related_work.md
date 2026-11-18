# Related-work survey: composition-decorated density-bifiltration graphcodes on GTEx cell point clouds

Compiled 2026-11-18 by web search. I checked every entry by opening its abstract page, its full text or its bibliographic record (arXiv, DOI resolver, Europe PMC or PMC, Crossref, OpenAlex, proceedings page), and each entry gives its own status. Works the brief marked as already known (graphcodes 2405.14302, compressed graphcodes 2503.07368, RNAPath, TANGLE, PersLay, DeepSets, the function-Delaunay bifiltration) appear only in comparisons.

Verification legend:

- **Verified (full text)**: I read the methods or results sections.
- **Verified (abstract)**: I read the abstract or an indexed abstract, and claims stay within it.
- **Partly verified**: the bibliographic record is confirmed, but some details come from secondary sources (flagged).
- **Unverified**: the source could not be opened.

---

## Q1. Closest prior work and novelty risk

### 1a. Multiparameter persistence on cell point clouds (histology, spatial omics)

**1. Vipond O, Bull JA, Macklin PS, Tillmann U, Pugh CW, Byrne HM, Harrington HA (2021).** Multiparameter persistent homology landscapes identify immune cell spatial patterns in tumors. *PNAS* 118(41):e2102166118. https://doi.org/10.1073/pnas.2102166118

- **What they did:** They built a Rips radius × codensity (fifth nearest neighbour) bifiltration of cell centroids and summarised it with MP landscapes. The cells were CD8+, FoxP3+ and CD68+ cells from aligned IHC sections of 16 head-and-neck tumours, in 1.5 mm regions, plus an agent-based spheroid model. A regularised LDA on the landscapes discriminated cell types per region: 74.7% for CD8 vs FoxP3, 65.3% for CD8 vs CD68 and 86.3% for FoxP3 vs CD68, all on one tumour. In simulation, MP landscapes were more robust to cell-label noise than the pair-correlation function and one-parameter PH. There was no patient-level prediction and no cell-count or density covariate.
- **Relevance (overlap):** This is the highest overlap on the topology side, because density × radius MP of histology cell centroids is already published. Novelty must rest on four things: graphcodes with a GNN, decorated bars, sample-level multi-organ prediction, and density-controlled evaluation. The label-noise robustness argument is worth borrowing, since HistoPLUS classes are noisy.
- **Status:** Verified (full text, PMC8522280).

**2. Loiseaux D, Carrière M, Blumberg AJ (2023).** A framework for fast and stable representations of multiparameter persistent homology decompositions. *NeurIPS 2023.* https://arxiv.org/abs/2306.11170

- **What they did:** They vectorised MMA decompositions as S-CDR. One experiment uses Vipond et al.'s immunohistochemistry point clouds. They built sublevel filtrations of Gaussian KDEs of CD8 and CD68 on a 100×100 grid and classified which cell type each cloud represents. Scores (5-fold) were:
  - S-CDR 91.4 ± 1.6
  - Ripley's K 67.2 ± 2.3
  - one-parameter PH 60.7 ± 4.2
  - MP landscapes 65.3 ± 3.0

  MP kernels and MP images could not run at these point-cloud sizes.
- **Relevance (overlap and baseline):** This is the only quantitative ML benchmark of MP features on histology cell clouds that I found. It sets a precedent and shows that reviewers can ask for an MP-vectorisation baseline. The task is cell-type discrimination, not clinical prediction.
- **Status:** Verified (NeurIPS proceedings page; experiments section of the arXiv HTML).

**3. Benjamin K, Bhandari A, Kepple JD, ... Tillmann U, Harrington HA, Bull KR (2024).** Multiscale topology classifies cells in subcellular spatial transcriptomics. *Nature* 630:943-949. https://doi.org/10.1038/s41586-024-07563-1

- **What they did:** TopACT assigns cell types in Stereo-seq and Xenium data. They then computed Rips-codensity MP landscapes (H1) of TopACT-typed immune cells in patches around glomeruli in a lupus-nephritis mouse model. Landscapes averaged per condition suggested peripheral immune rings, which immunofluorescence confirmed. The analysis is descriptive: there is no classifier and no density baseline, and per-patch immune counts are reported only as separate support.
- **Relevance:** This is MP on typed cell positions in spatial omics, and its H1 "ring" reading is close to the planned H1 hole-boundary decoration. It is no novelty threat for supervised sample-level prediction.
- **Status:** Verified (full text, PMC11208150).

**4. Loiseaux D, Scoccola L, Carrière M, Botnan MB, Oudot S (2023).** Stable vectorization of multiparameter persistent homology using signed barcodes as measures. *NeurIPS 2023.* https://arxiv.org/abs/2306.03801

- **What they did:** They treated Hilbert and rank signed barcodes as signed measures and vectorised them by convolution or sliced-Wasserstein kernels. These beat MP landscapes, images and kernels on time-series, graph and point-cloud benchmarks, including DTM-filtered alpha and density-filtered Rips complexes. The paper has no cell data; Vipond et al. is cited only as motivation.
- **Relevance (baseline):** This gives a cheap, stable MP baseline on the same function-Delaunay bifiltration. TDA reviewers will ask whether the GAT on graphcodes beats it.
- **Status:** Verified (arXiv abstract; experiments section of the arXiv HTML).

### 1b. Persistence that carries point labels or attributes

**5. Stolz BJ, Dhesi J, Bull JA, Harrington HA, Byrne HM, Yoon IHR (2024).** Relational persistent homology for multispecies data with application to the tumor microenvironment. *Bull Math Biol* 86. https://doi.org/10.1007/s11538-024-01353-6 (arXiv 2308.06205)

- **What they did:** Dowker PH and multispecies witness PH encode the spatial relations between cell types. On synthetic agent-based tumour-microenvironment data (vessels, macrophages, tumour and necrotic cells), the features recover the dominant immune phenotype and cluster the model's parameter regimes.
- **Relevance (alternative decoration):** This is the main published "label-aware PH" for multi-type cells. It encodes types inside the complex, via type pairs, rather than as bar attributes, so it is both a possible baseline and a point to position against.
- **See also:** Torras-Pérez M, Yoon IHR, Weeratunga P, Ho L-P, Byrne HM, Tillmann U, Harrington HA (2025). Topology across scales on heterogeneous cell data. arXiv 2505.02717. It reweights and visualises persistence images for multi-type multiplexed imaging (COVID-19 lung, lupus mouse spleen), with no prediction.
- **Status:** Verified (arXiv abstracts; journal metadata from the PMC listing, PMC11408586).

**6. Cultrera di Montesano S, Draganov O, Edelsbrunner H, Saghafian M (2026).** Chromatic alpha complexes. *Foundations of Data Science* 8:30-62. https://doi.org/10.3934/fods.2025003 (arXiv 2212.03128)

- **What they did:** They define chromatic Delaunay and alpha complexes for coloured point sets. Kernel, image and cokernel persistence give stable multi-scale measures of how the colours mix, and software is provided.
- **Relevance (alternative):** This is a theory-grade alternative to composition decorations: colour mixing is measured through maps between complexes rather than as bar attributes. The abstract does not say whether the method scales to 14 HistoPLUS classes; a grouping (for example epithelial, immune, stromal) would probably be needed (my inference).
- **Status:** Verified (arXiv abstract; Crossref).

**7. Natarajan A, Chaplin T, Bull JA, Mulholland-Illingworth EJ, Leedham SJ, Byrne HM, Jimenez M-J, Harrington HA (2026).** Topology of multi-species localization. arXiv 2603.03237. https://arxiv.org/abs/2603.03237

- **What they did:** They propose a scalable way to quantify higher-order interactions among species: features shared by several species, features that vanish or appear when another species is added, and so on. They apply it to a simulated tumour microenvironment and to colorectal cancer samples across progression; the abstract reports qualitative findings only.
- **Related method:** It builds on chromatic Delaunay machinery from Natarajan A, Chaplin T, Brown A, Jimenez M-J, *Morse theory for chromatic Delaunay triangulations*, arXiv 2405.19303. That paper shows chromatic Delaunay-Čech and Delaunay-Rips filtrations collapse to the chromatic alpha filtration, which makes them cheaper.
- **Relevance (closest competitor):** This is the closest current competitor for "topology of typed cells" in human pathology. Per its abstract it is descriptive, with no sample-level learning. Position decorated graphcodes as the learning-oriented counterpart, and consider its descriptors on grouped classes as a baseline.
- **Status:** Verified (arXiv abstracts). Whether 2603.03237 uses the chalc package is not confirmed.

**8. Curry J, Hang H, Mio W, Needham T, Okutan OB (2022).** Decorated merge trees for persistent topology. *J Appl Comput Topol* 6(3):371-428. https://doi.org/10.1007/s41468-022-00089-3 (arXiv 2103.15804)

- **What they did:** They define merge trees whose components carry higher-degree homology, fusing π0 with Hn. They prove bottleneck and interleaving stability and compute alignments with Gromov-Wasserstein couplings. Applications include point clouds, time series, tumour images and graph alignment.
- **Relevance (framing):** This is the mathematical precedent for decorating H0 merges. Its decorations are homological, not label compositions. It is useful for framing, and its stability ideas could support a stability argument for composition decorations.
- **Status:** Verified (arXiv abstract; Crossref).

**9. Immonen J, Souza AH, Garg V (2023).** Going beyond persistent homology using persistent homology (RePHINE). *NeurIPS 2023.* https://arxiv.org/abs/2311.06152

- **What they did:** They characterise when component persistence of graphs with coloured vertices separates graphs, using "colour-separating sets". RePHINE combines vertex-level and edge-level persistence with colour information and plugs into message-passing GNNs, which improves graph classification.
- **Relevance (closest ML precedent):** This is the closest ML precedent for persistence pairs enriched with vertex-label information inside a GNN, the conceptual cousin of decorated bars. It differs from the planned design in four ways: it works on graphs rather than point clouds, uses a single filtration, does not aggregate compositions, and has no two-parameter structure.
- **Status:** Verified (arXiv abstract).

**10. Yoon IHR, Jenkins R, Colliver E, ... Sahai E (2024).** Deciphering the diversity and sequence of extracellular matrix and cellular spatial patterns in lung adenocarcinoma using topological data analysis. *bioRxiv.* https://doi.org/10.1101/2024.01.05.574362

- **What they did:** They applied PH to ECM architecture and cell positions, and Dowker PH to cancer-leukocyte segregation across scales, in 44 TRACERx lung adenocarcinoma specimens stained with picrosirius red and haematoxylin. ECM remodelling followed a trajectory associated with immune exclusion.
- **Relevance:** This is a real-tissue use of cross-type (Dowker) PH on brightfield stains. One idea to borrow: analyse topology jointly with conventional tumour-microenvironment features.
- **Status:** Verified (bioRxiv abstract).

### 1c. Persistence combined with GNNs or set networks for slide- or sample-level prediction

**11. Horn M, De Brouwer E, Moor M, Moreau Y, Rieck B, Borgwardt K (2022).** Topological graph neural networks (TOGL). *ICLR 2022.* https://arxiv.org/abs/2102.07835

- **What they did:** TOGL is a differentiable PH layer with learned vertex filtrations that fits into any GNN. It is more expressive than the WL test and improves graph and node classification.
- **Relevance (alternative design):** It offers an alternative to a fixed density × radius bifiltration: learn the filtration on a cell graph instead. It could become an extra rung if reviewers ask for learned filtrations, though its cost at WSI scale is a concern.
- **Status:** Verified (arXiv abstract).

**12. Obeid A, Sohail A, Boumaraf S, Liu X, Javed S, Almarzouqi H, Dias J, Bennamoun M, Werghi N, Elfadel I (2025).** PMIL: a topology module to improve MIL-based WSI classification. *IEEE ISCAS 2025.* https://doi.org/10.1109/ISCAS56072.2025.11043738

- **What they did:** They claim the first adaptable PH module inside MIL for WSI classification. The abstract does not say what the PH is computed on.
- **Relevance (low overlap):** It shows that persistence combined with MIL on WSIs is emerging. Nothing indicates cell-level or multiparameter persistence.
- **Status:** Verified (abstract and metadata only).

**13. Pothagoni S, Schweinhart B (2025; v3 2026).** Classification of histopathology slides with persistent homology convolutions. arXiv 2507.14378. https://arxiv.org/abs/2507.14378

- **What they did:** They compute local PH in a convolution-like operator to keep locality. The abstract reports that this beats conventionally trained models and is less sensitive to hyperparameters, but gives no numbers.
- **Relevance (low overlap):** Image-side topology for slides; complementary to the planned design.
- **Status:** Verified (abstract only).

I found no published application of graphcodes, or of any MP-plus-GNN pipeline, to cells, tissues or histology, in several searches covering 2024-2026.

---

## Q2. TDA on cell positions in pathology and spatial biology: does topology add beyond density and composition?

### Positive evidence (none of it controls for density and composition)

**14. Aukerman A, Carrière M, Chen C, Gardner K, Rabadán R, Vanguri R (2020).** Persistent homology based characterization of the breast cancer immune microenvironment: a feasibility study. *SoCG 2020*, LIPIcs 164:11. https://doi.org/10.4230/LIPIcs.SoCG.2020.11 (journal version: *J Comput Geom* 12(2), dated 2020 or 2021 in different records)

- **What they did:** They computed persistence diagrams from per-cell positions and stain intensities in quantitative multiplex IF of about 700 breast cancer patients (median follow-up 8.5 years) and related them to subtype and overall survival. The diagrams "outperform and complement" nearest-neighbour descriptors.
- **Relevance:** This is positive evidence at patient scale, but the comparison is against nearest-neighbour statistics, not against density or composition, and the abstract gives no effect sizes.
- **Status:** Verified (abstract). The journal version is confirmed only through OpenAlex.

**15. Singhal S, Li C, Aukerman A, Carrière M, ... Rabadan R, Saltz J, Chen C, Gardner K (2026).** Topology-based biomarkers accurately predict breast cancer outcome and survival. *Cancer Res* 86(8):2078-2093. https://doi.org/10.1158/0008-5472.CAN-25-1216

- **What they did:** Continuous topology scores of retained tissue order predicted survival better than conventional biomarkers and varied less across racial and ethnic groups. Combined with transcriptomics, they gave therapy-response signatures.
- **Relevance:** This is the strongest recent positive claim. The full text was not accessible, so I do not know whether the models adjusted for cell density or composition. Press summaries mention multiplex IF of tumour and immune cells and PD-L1 in more than 550 patients (unverified).
- **Status:** Verified (abstract via Europe PMC); further details unverified.

**16. Samorodnitsky S, Campbell K, Little A, Ling W, Zhao N, Chen Y-C, Wu MC (2026).** Detecting clinically relevant topological structures in multiplexed spatial proteomics using TopKAT. *Patterns* 7(1):101456. https://doi.org/10.1016/j.patter.2025.101456

- **What they did:** They compute Rips PH (H0 and H1) of cell centroids, compare diagrams with a lifespan-based dissimilarity, and feed the result into a kernel association test that handles survival, binary and continuous outcomes with clinical covariates.
  - In simulations with loop structure, the test had power 0.83-0.90, versus 0.52-0.69 for point-process methods.
  - In MIBI triple-negative breast cancer (38 patients), immune-cell topology was associated with survival (p = 0.011, with the distance chosen by a scan), where the SPOT, FunSpace and SPF methods found no association.
  - There was no cell-abundance adjustment.
- **Relevance:** Positive, but not controlled for density. Worth borrowing for an inferential check: run the same kernel test with density and composition as covariates.
- **Status:** Verified (full text, PMC12827733).

**17. Lawson P, Sholl AB, Brown JQ, Fasy BT, Wenk C (2019).** Persistent homology for the quantitative evaluation of architectural features in prostate cancer histology. *Sci Rep* 9:1139. https://doi.org/10.1038/s41598-018-36798-y

- **What they did:** They applied PH to glandular architecture in Gleason 3-5 images. Ranked persistence vectors clustered the images in line with Gleason progression and split patterns into subgroups.
- **Relevance:** The classic gland-architecture study. It is unsupervised and has no density control.
- **Status:** Verified (abstract via Europe PMC).

**18. Teramoto T, Shinohara T, Takiyama A (2020).** Computer-aided classification of hepatocellular ballooning in liver biopsies from patients with NASH using persistent homology. *Comput Methods Programs Biomed* 195:105614. https://doi.org/10.1016/j.cmpb.2020.105614

- **What they did:** They fed persistence images (H0 and H1) of H&E images from 79 NAFLD patients to linear classifiers. Accuracy exceeded 90% for NASH vs non-NASH, with a best AUC of 0.946 for NASH vs Matteoni type 2.
- **Relevance:** This is the only liver-pathology PH classifier I found. It is image-based, small and has no density baseline. I found no TDA study of liver fibrosis or cirrhosis from cell centroids.
- **Status:** Verified (abstract via Europe PMC).

**19. Yang J, Fang H, Dhesi J, Yoon IHR, Bull JA, Byrne HM, Harrington HA, Grindstaff G (2025).** Topological classification of tumour-immune interactions and dynamics. *J Math Biol* 91(3). https://doi.org/10.1007/s00285-025-02253-6

- **What they did:** They built four topological pipelines (Rips or radial filtrations, zigzag, vineyards) on simulated tumour and macrophage positions. Logistic regression detected perivascular-niche formation earlier than tumour-cell counts or the macrophage phenotype ratio did.
- **Relevance:** Topology beat simple counts, but only in agent-based simulations and without a combined model of counts plus topology.
- **Status:** Verified (abstracts via Europe PMC and arXiv).

### Negative or cautionary evidence

**20. Ali M, Richter S, Ertürk A, Fischer DS, Theis FJ (2025).** Graph neural networks learn emergent tissue properties from spatial molecular profiles. *Nat Commun* 16:8419. https://doi.org/10.1038/s41467-025-63758-8

- **What they did:** They trained GNNs on spatial cell graphs to predict tissue phenotypes. For breast-cancer grading, spatial information did not significantly improve on single-cell or pseudobulk baselines, though the GNNs still exposed interpretable spatial features.
- **Relevance:** This is the most direct published evidence that spatial structure may add nothing over composition for a sample-level label. It is consistent with the internal Betti-curve null result and makes rung (1) the bar to beat.
- **Status:** Verified (abstract via Europe PMC).

**21. Bhaskar D, Zhang WY, Volkening A, Sandstede B, Wong IY (2023).** Topological data analysis of spatial patterning in heterogeneous cell populations: clustering and sorting with varying cell-cell adhesion. *npj Syst Biol Appl* 9:43. https://doi.org/10.1038/s41540-023-00302-8

- **What they did:** Persistence images classified simulated two-population sorting patterns. Normalising the persistence images improved classification when proliferation changed cell numbers. They also tested adding per-cell-type information.
- **Relevance:** This explicitly shows that raw persistence summaries are confounded by cell number and density. It supports count-normalised bar features and count-matched controls.
- **Status:** Verified (abstract via Europe PMC; Crossref).

**22. Harlev I, Oukhanov T, Ben-Uri R, Keren L, Bagon S (2026).** Sampling-aware 3D spatial analysis in multiplexed imaging. arXiv 2604.07890 (CVPR 2026 CVMI workshop). https://arxiv.org/abs/2604.07890

- **What they did:** In simulation, 2D sections recovered global cell-type abundance reliably, while local clustering and interaction statistics showed high variance, worst for rare or regionally confined types. Real data showed the same section-to-section fluctuation.
- **Relevance:** Graphcode features are local, meso-scale statistics, so on single GTEx sections they may be noisier than composition. Their stability should be measured.
- **Status:** Verified (abstract).

### Synthesis for Q2

Indirect evidence also comes from:

- entries 1-3, where the MP studies have no density control;
- entry 24, where GTEx pathology categories are only weakly predictable from image texture;
- entries 33 and 36, where composition-type features are strong.

The positive studies compare topology with other spatial statistics (nearest-neighbour, Ripley's K, pair correlation, point-process tests), or with simple counts in simulation. None of them conditions on both density and composition in a combined model on real tissue. The negative and cautionary studies (Ali 2025, Bhaskar 2023, Harlev 2026) point the other way. The internal null result (Betti curves of density slices add nothing beyond density summaries for liver cirrhosis) is therefore consistent with the published record, not contradicted by it.

---

## Q3. GTEx histology machine learning

**23. Abila E, Buljan I, Zheng Y, ... Rendeiro AF (2026).** Histological aging signatures for monitoring tissue-specific aging and disease. *Nat Med* 32(9). https://doi.org/10.1038/s41591-026-04566-5 (preprint: https://doi.org/10.1101/2024.11.14.618081)

- **What they did:** They used 25,712 GTEx WSIs from 983 donors and 40 tissues. CNN backbones fine-tuned for tissue classification supplied features to per-tissue ridge regressions ("tissue clocks") with a mean error of 4.9 years. Age gaps were then linked to GTEx pathology notes, comorbidities and telomere length, and blood-based proxies were built.
- **Split:** Models were per tissue, evaluated by five-fold GroupKFold with donors as groups (preprint methods).
- **Relevance (baseline and protocol):** This is the age-regression baseline and split protocol reviewers will cite. It is image-based and per tissue; a multi-organ model needs donor-grouped splits across organs.
- **Status:** Verified (full-text methods of the preprint; Crossref for the journal version).

**24. Gallins P, Saghapour E, Zhou Y-H (2020).** Exploring the limits of combined image/'omics analysis for non-cancer histological phenotypes. *Front Genet* 11:555886. https://doi.org/10.3389/fgene.2020.555886

- **What they did:** They predicted six GTEx pathology categories in five tissues from 117 Haralick texture features and, separately, from bulk expression. The categories were lung fibrosis, liver steatosis, liver congestion, tibial-artery atherosclerosis, thyroid Hashimoto's and adipose fibrosis. Expression did as well as or better than images: liver steatosis AUC 0.73 from images vs 0.74-0.75 from expression, lung fibrosis 0.61-0.62 vs 0.63-0.65. Combining the two gave no gain.
- **Split:** 5-fold cross-validation per tissue; donor grouping is not stated (one sample per donor per tissue).
- **Relevance (baseline):** This is the only GTEx pathology-category prediction benchmark I found. It gives a weak image baseline and suggests that label noise caps achievable accuracy.
- **Status:** Verified (full text).

**25. Ash JT, Darnell G, Munro D, Engelhardt BE (2021).** Joint analysis of expression levels and histological images identifies genes associated with tissue morphology. *Nat Commun* 12. https://doi.org/10.1038/s41467-021-21727-x

- **What they did:** ImageCCA (a convolutional autoencoder plus sparse CCA) links image features with expression. In GTEx v6 it found image-morphology QTLs in thyroid and colon.
- **Relevance:** Early GTEx coupling of images and expression, which supports an RNA head. There is no prediction benchmark.
- **Status:** Verified (abstract via Europe PMC).

**26. Gao Y, Liang J, Tian M, Deng W, Wang L, Tao S, Mou T (2025).** Histology image analysis of 13 healthy tissues reveals molecular-histological correlations. *Sci Rep* 15:26812. https://doi.org/10.1038/s41598-025-11853-7

- **What they did:** They measured nuclear morphology with deep learning on 4,306 GTEx samples from 13 organs and found that nuclear features correlate with expression programmes.
- **Relevance:** This is the only GTEx study at nucleus level that I found. It is correlational, and it suggests nuclear morphology as an extra per-cell feature that the planned inputs omit.
- **Status:** Verified (abstract via Europe PMC).

**27. Ribeiro R, Moraes A, Moreno M, Ferreira PG (2024).** Integration of multi-modal datasets to estimate human aging. *Mach Learn* 113:7293-7317. https://doi.org/10.1007/s10994-024-06588-x

- **What they did:** They regressed age in GTEx lung and ovary from methylation, expression, telomere length and HIPT histology features. Methylation was best, with test MAE of 3.36 and 4.36 years; expression plus histology beat either alone.
- **Relevance:** A second histology-age reference point, small and limited to two tissues.
- **See also:** Chaudhary S, Voigts A, Bereket M, ..., Casale FP (2026). HistoGWAS. *Genome Biol* 27. https://doi.org/10.1186/s13059-026-04031-z. It uses foundation-model histology features as GWAS traits in 11 GTEx tissues and reports four loci (partly verified: Crossref record, with content from search summaries).
- **Status:** Verified (Crossref abstract).

### Gaps and a contamination risk

I found no GTEx study that does any of the following:

- trains one multi-task slide model across organs with pathology, age, RNA and organ heads;
- uses cell-type-labelled centroids as its only input;
- applies TDA.

**Contamination warning:** THREADS (entry 29) used GTEx slides with paired RNA in pretraining, and RNAPath was trained on GTEx. Image foundation models used as baselines may therefore have seen GTEx test slides.

---

## Q4. Supervised (molecular or multi-task) vs self-supervised slide pretraining

**28. Shao D, Chen RJ, Song AH, Runevic J, Lu MY, Ding T, Mahmood F (2025).** Do multiple instance learning models transfer? *ICML 2025* (PMLR 267). https://arxiv.org/abs/2506.09022

- **What they did:** They studied 11 MIL models across many pretraining tasks. Supervised pan-cancer pretraining (PC-108, a 108-class cross-entropy task) transferred across organs and tasks. The paper's comparisons with slide foundation models:

  | Comparison | vs CHIEF | vs GigaPath |
  |---|---|---|
  | k-NN, tasks won (of 15) | 12 | 13 |
  | k-NN, average gain | +5.9% | +9.7% |
  | Fine-tuning, average gain | +2.2% | +0.8% |
  | PC-108 pretraining set as share of theirs | 6.5% | 2.3% |

  The paper also notes that CHIEF's pretraining data overlapped 11 of the 15 downstream tasks.
- **Relevance:** This is the best direct evidence that supervised slide-level pretraining can match or beat self-supervised slide encoders with far less data. It also gives a reporting template (k-NN or linear probe plus fine-tuning) and a warning about overlap between pretraining labels and downstream tasks.
- **Status:** Verified (abstract, PMLR page, results section of the arXiv HTML). The text's GigaPath figures do not quite match the Table 2 averages as read from the HTML (about +9.5 instead of +9.7 for k-NN), so quote them with care.

**29. Vaidya A, Zhang A, Jaume G, ..., Mahmood F (2025).** Molecular-driven foundation model for oncologic pathology (THREADS). arXiv 2501.16652. https://arxiv.org/abs/2501.16652

- **What they did:** They contrastively aligned an ABMIL slide encoder with paired RNA profiles (scGPT-based encoder) or DNA profiles for 47,171 WSIs; the RNA side drew on MGH, TCGA and GTEx. Across 54 tasks, linear probing gained 9.9, 6.7 and 6.3 points over GigaPath, CHIEF and PRISM. There is no ablation that changes only the objective, so the gain is confounded with the patch encoder and the data.
- **Relevance:** Molecular supervision of slide representations works, which supports the RNA head. Note the GTEx overlap.
- **Status:** Verified (abstract; methods and results of the arXiv HTML).

**30. Nicke T, Schäfer JR, Höfener H, Feuerhake F, Merhof D, Kießling F, Lotz J (2025).** Tissue concepts: supervised foundation models in computational pathology. *Comput Biol Med* 186. https://doi.org/10.1016/j.compbiomed.2024.109621 (arXiv 2409.03519)

- **What they did:** One encoder trained with supervision on 16 tasks (912,000 patches) matched self-supervised encoders while using about 6% of their training patches, and beat ImageNet pretraining.
- **See also:** Mormont R, Geurts P, Marée R (2021). Multi-task pre-training of deep neural networks for digital pathology. *IEEE J Biomed Health Inform* 25(2). It pretrained on 22 tasks (about 900,000 images) and was better than or comparable to ImageNet.
- **Relevance:** Tile-level evidence that multi-task supervision is data-efficient compared with self-supervised learning.
- **Status:** Verified (abstracts; Crossref and Europe PMC).

**31. Ding T, Wagner SJ, Song AH, ..., Le LP, Mahmood F (2025).** A multimodal whole-slide foundation model for pathology (TITAN). *Nat Med* 31. https://arxiv.org/abs/2411.19666

- **What they did:** Visual self-supervised learning on 335,645 WSIs, plus alignment with pathology reports and 423,122 synthetic captions. It beats ROI-level and slide-level foundation models on linear probing, few-shot and zero-shot tasks, and retrieval.
- **Relevance (counterpoint):** Large-scale self-supervision with language is state of the art for image slide encoders. No comparable self-supervised model exists for cell-topology graphs, so asking for a graph-contrastive alternative is a fair reviewer question.
- **Status:** Verified (arXiv abstract; journal record via Europe PMC).

**32. Li J, Hu J, Sun Q, Yan R, Ouyang M, Guan T, Han A, He C, He Y (2025).** Can we simplify slide-level fine-tuning of pathology foundation models? (SiMLP). arXiv 2502.20823. https://arxiv.org/abs/2502.20823

- **What they did:** Mean pooling plus an MLP beat MIL by 3.52% on a pan-cancer task and was competitive with slide foundation models in few-shot settings.
- **Relevance:** Simple pooled baselines are strong, the analogue of rung (1). A mean-pooled image-feature reference model is worth including.
- **Status:** Verified (abstract).

### Synthesis for Q4

Supervised pretraining (multi-class, multi-task or molecular) matched or beat self-supervised encoders at small to moderate data scale (entries 28-30). The caveats:

- the evidence is confounded by data and architecture;
- gains shrink under fine-tuning;
- the largest self-supervised models with language (entry 31) lead overall;
- no evidence exists for cell-topology encoders specifically.

---

## Q5. Baselines on cell-level inputs (positions and types)

**33. Schürch CM, Bhate SS, Barlow GL, ..., Nolan GP (2020).** Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. *Cell* 182(5):1341-1359.e19. https://doi.org/10.1016/j.cell.2020.07.005

- **What they did:** CODEX imaging (56 markers, 140 regions, 35 patients) yielded nine cellular neighbourhoods. Coupling and fragmentation of these neighbourhoods were linked to outcome.
- **Relevance (expected baseline):** This is the origin of the standard cellular-neighbourhood analysis: cells are clustered by the cell-type composition of their surroundings, and per-sample neighbourhood frequencies serve as features. Reviewers expect it as a baseline.
- **Status:** Verified (abstract via Europe PMC). The abstract does not describe how neighbourhoods are constructed.

**34. Wu Z, Trevino AE, Wu E, ..., Mayer AT, Zou J (2022).** Graph deep learning for the characterization of tumour microenvironments from spatial protein profiles in tissue specimens (SPACE-GM). *Nat Biomed Eng* 6:1435-1448. https://doi.org/10.1038/s41551-022-00951-w

- **What they did:** A GNN on local cell subgraphs from CODEX data (40 markers; colorectal and head-and-neck cancer), pooled to patient outcomes. It was "substantially more accurate" than deep models built on the cell-type composition of local neighbourhoods.
- **Relevance (expected baseline):** A scalable recipe for a GNN on cells (sample local subgraphs, then pool), and positive evidence for spatial structure beyond neighbourhood composition (abstract-level, no numbers).
- **Status:** Verified (abstract via Europe PMC).

**35. Hu Y, Rong J, Xu Y, Xie R, Peng J, Gao L, Tan K (2024).** Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes (CytoCommunity). *Nat Methods* 21:267-278. https://doi.org/10.1038/s41592-023-02124-2

- **What they did:** A GNN maps cell phenotypes and their spatial graph to soft neighbourhoods. In its supervised mode, graph pooling and sample labels yield neighbourhoods that predict the condition.
- **Relevance (expected baseline):** Its input type matches the planned one (phenotype labels plus positions), and it has a sample-level supervised mode, so it is a ready learned-neighbourhood baseline.
- **Status:** Verified (abstract via Europe PMC).

**36. Diao JA, Wang JK, Chui WF, ..., Taylor-Weiner A (2021).** Human-interpretable image features derived from densely mapped cancer pathology slides predict diverse molecular phenotypes. *Nat Commun* 12. https://doi.org/10.1038/s41467-021-21896-9

- **What they did:** Cell-type and tissue-type maps of H&E WSIs were turned into 607 interpretable features that predicted molecular phenotypes (AUROC 0.601-0.864), on par with black-box models.
- **Relevance (baseline):** H&E cell-type composition and density features make a strong interpretable baseline, the published analogue of rung (1).
- **Status:** Verified (abstract via Europe PMC).

**37. Pati P, Jaume G, Foncubierta-Rodríguez A, ..., Gabrani M (2022).** Hierarchical graph representations in digital pathology (HACT-Net). *Med Image Anal* 75:102264. https://doi.org/10.1016/j.media.2021.102264

- **What they did:** Hierarchical cell-to-tissue graphs built from H&E nuclei and tissue regions, classified by a GNN on the BRACS dataset; it beat alternative methods and individual pathologists.
- **See also:** Wang Y, Wang YG, Hu C, ..., Song J (2022). Cell graph neural networks enable the precise prediction of patient survival in gastric cancer. *npj Precis Oncol* 6:45. It used mIHC cell graphs and reached AUC 0.960 for binary survival classes.
- **Relevance (expected baselines):** The canonical H&E and mIHC cell-graph GNN baselines.
- **Status:** Verified (abstracts via Europe PMC).

### Standard practice

Entries 20 and 33-37 show what is standard:

- composition or pseudobulk features;
- cellular-neighbourhood frequencies;
- cell-graph GNNs on k-NN or Delaunay graphs, with local subgraph sampling at scale;
- second-order spatial statistics such as Ripley's K or the pair-correlation function, used as baselines in entries 1, 2 and 16.

DeepSets or PointNet over cells did not turn up as an established baseline in this literature. Rung (2) is reasonable, but reviewers will not recognise it, so it does not replace a cell-graph GNN.

---

## Q6. Mass-based filtering of merge trees

**38. Stuetzle W (2003).** Estimating the cluster tree of a density by analyzing the minimal spanning tree of a sample. *J Classif* 20(1):25-47. https://doi.org/10.1007/s00357-003-0004-6

- **What they did:** Stuetzle introduced runt pruning, building on the runt test of Hartigan and Mohanty (1992).
  - The runt size of a dendrogram node is the smaller of the number of leaves of the two subtrees rooted there. For an MST edge it is the smaller of the two groups the edge connects.
  - A split counts as real if both daughters contain enough observations, that is, if the runt size exceeds a threshold.
  - The paper gives diagnostics, such as rootograms of runt sizes, for choosing the threshold.
- **Follow-up:** Stuetzle W, Nugent R (2010). A generalized single linkage method for estimating the cluster tree of a density. *J Comput Graph Stat* 19(2):397-418.
- **Relevance (exact match):** This matches the rule "keep a merge only if its smaller side has at least m points" exactly.
  - At a fixed density slice, the H0 merges of an alpha filtration (homotopy-equivalent to Čech) are the single-linkage dendrogram of the retained points, at scale 2r or up to a monotone reparametrisation, so runt pruning applies verbatim.
  - Cite Hartigan and Mohanty (1992) for the statistic and Stuetzle (2003) for the pruning.
- **Status:** Verified (definition read in the full text; Crossref). Hartigan and Mohanty (1992) is checked only as cited by Stuetzle; I did not open it.

**39. Campello RJGB, Moulavi D, Sander J (2013).** Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, LNCS 7819:160-172. https://doi.org/10.1007/978-3-642-37456-2_14; **Campello RJGB, Moulavi D, Zimek A, Sander J (2015).** Hierarchical density estimates for data clustering, visualization, and outlier detection. *ACM TKDD* 10(1). https://doi.org/10.1145/2733381

- **What they did:** HDBSCAN's condensed tree applies a min_cluster_size rule at every split:
  - A split is kept only if both children have at least min_cluster_size points.
  - If one child is smaller, its points "fall out" of the cluster and the larger child keeps the parent's identity.
  - If both children are smaller, the cluster ends there.

  Flat clusters are then chosen by a cluster-stability score: the sum over a cluster's points of how long each stays in it, measured in inverse distance. The scikit-learn implementation's condensing function states that it is "akin to the runt pruning procedure of Stuetzle".
- **Relevance (same criterion):** The criterion is the same, but HDBSCAN applies it to a single-linkage tree of mutual-reachability distances, which are density-smoothed, rather than Euclidean distances. Its stability score is a candidate extra vertex attribute.
- **Status:** Verified (Crossref for both; hdbscan "How HDBSCAN works" documentation; scikit-learn source `sklearn/cluster/_hdbscan/_tree.pyx`).

**40. Chazal F, Guibas LJ, Oudot SY, Skraba P (2013).** Persistence-based clustering in Riemannian manifolds (ToMATo). *J ACM* 60(6). https://doi.org/10.1145/2535927

- **What they did:** Mode-seeking clustering on a neighbourhood graph with a density estimate. Clusters whose prominence (persistence in density height) is below τ merge into their parent.
- **Relevance:** **Not a match**, because the criterion is prominence, not mass. A multi-parameter extension exists: Andrianirina L, Carrière M (2026), ToMAToMP, arXiv 2605.14824 (verified abstract).
- **Status:** Partly verified (Crossref record; the τ rule comes from the description in Huber, Kalisnik, Schnider, AuToMATo, TMLR 2025, arXiv 2408.06958).

**41. Carr H, Snoeyink J, van de Panne M (2004).** Simplifying flexible isosurfaces using local geometric measures. *IEEE Visualization 2004.* https://doi.org/10.1109/VISUAL.2004.96

- **What they did:** Contour-tree leaves are pruned in order of a local geometric measure: height (equivalent to persistence), volume or hypervolume. The paper cites earlier pruning by the approximate volume of the discarded subtree.
- **Relevance (related, not a match):** This is the visualisation literature's mass-based simplification. It prunes leaves iteratively rather than applying a per-merge runt test.
- **Status:** Verified (paper text).

### Answer to Q6

The rule "keep a merge only if its smaller side has at least m points" is runt pruning (Hartigan and Mohanty 1992; Stuetzle 2003). HDBSCAN's condensed tree with min_cluster_size = m applies the same test, on a density-smoothed metric. ToMATo does not match, because it prunes by prominence. Volume-based contour-tree simplification is a related iterative leaf-pruning scheme.

---

## Novelty assessment

### Not new (claims should avoid "first")

- **Density × radius MP of histology cell centroids** exists: Vipond 2021 and Benjamin 2024 (Rips × codensity), and Loiseaux, Carrière and Blumberg 2023 (KDE-sublevel bifiltrations of the same immune-cell data, with an ML benchmark). Slicing a bifiltration at fixed values of one parameter is also the standard route to MP vectorisations such as fibered barcodes and landscapes.
- **Label-aware topology for multi-type cells** exists: relational/Dowker PH (Stolz 2024; Yoon 2024) and chromatic alpha and multi-species localisation (Cultrera di Montesano 2026; Natarajan 2026).
- **Persistence pairs enriched with vertex colours inside a GNN** exist (RePHINE, 2023), and decorated merge trees exist as a mathematical object (Curry 2022).
- **Supervised and molecular slide pretraining** exists (Shao 2025; THREADS 2025).
- **GTEx-scale histology models** exist for age (tissue clocks), RNA (RNAPath) and pathology categories (Gallins 2020), all image-based.
- **The mass filter of merges** is runt pruning, the same rule as HDBSCAN's condensing.

### Apparently new (no precedent found)

- Graphcodes, or any pipeline combining MP and a GNN, applied to cells, tissue or histology.
- Per-bar decoration with the cell-type composition and log sizes of both merging components, and H1 decoration with the composition of the hole's outer boundary at birth, used as learnable vertex features.
- Supervised sample-level prediction from MP of cell centroids across about 30 organs, with multi-task heads and a donor-level split.
- A density- and composition-controlled ablation ladder for topology in pathology. The literature lacks one, so a careful result would be a contribution even if it is negative.

### Risk

- **Methodological novelty risk: moderate.** The decorations are a natural combination of graphcodes with label-aware TDA, and reviewers will cite RePHINE, chromatic and relational PH, and decorated merge trees.
- **Application novelty risk: low.** I found no TDA on GTEx and no MP-plus-GNN work on histology.
- **Main scientific risk:** the topology-specific gain may vanish once density and composition are controlled. The internal finding, Ali 2025 and Bhaskar 2023 all point this way. If it happens, the paper's claim would have to shift to "where decorated topology helps beyond composition (which organs, which labels)", and the decorations, being composition themselves, may carry most of whatever signal remains.

---

## Suggested design changes

### 1. Make density and composition control explicit and testable

- **Widen rung (1)** so it holds density summaries as well as counts: per-slice cell counts, tissue area, and per-class k-NN distance quantiles. Diao 2021 and Ali 2025 show such summaries are strong.
- **Add two perturbation controls**, motivated by Bhaskar 2023 (count confound), Ali 2025 (no gain over pseudobulk) and Harlev 2026 (local-statistic variance):
  - (a) Shuffle cell-type labels within each slide. This keeps geometry and density and destroys type arrangement, so it isolates what the decorations contribute.
  - (b) Resample positions from an inhomogeneous Poisson process with per-type intensity smoothed at a scale comparable to the density slices. This keeps the density map and composition and destroys finer arrangement, so it isolates what topology contributes.
- **Report incremental value over rung (1)** with paired, donor-level bootstrap confidence intervals.
- **Add an inferential check:** a TopKAT-style kernel association test (entry 16) with density and composition as covariates.

### 2. Add the baselines reviewers know

- Cellular-neighbourhood frequencies (entry 33).
- A cell-graph GNN on a Delaunay or k-NN graph with cell-type one-hot features, either SPACE-GM-style local subgraph sampling (entry 34) or supervised CytoCommunity (entry 35). Rung (2) can stay, but it does not replace a GNN in reviewers' eyes.
- Cross-type Ripley's K or L, or pair-correlation functions, at several radii (as in entries 1, 2 and 16).
- One-parameter PH per density slice, such as persistence images, which is the internal Betti-curve result formalised.
- An MP vectorisation of the same function-Delaunay bifiltration, such as Hilbert signed measures or MMA/S-CDR (entries 2 and 4). This separates "graphcode plus GAT" from "MP features in general".
- An image reference model on the same donor split: foundation-model tile features with mean pooling (entry 32) and ABMIL, reported alongside rather than as something to beat. Check for GTEx contamination of the chosen encoder (entry 29; RNAPath).

### 3. Pretraining protocol (following entry 28)

- Report k-NN or linear-probe results as well as fine-tuning results.
- Compare multi-task pretraining with training from scratch and with leave-one-head-out variants.
- When cirrhosis is presented as downstream transfer, keep cirrhosis and correlated liver pathology categories out of the pretraining heads, or hold out the liver donors. Otherwise frame cirrhosis explicitly as an in-pretraining task. Shao et al. flagged exactly this kind of overlap as inflating CHIEF's results.

### 4. Name and justify the m-filter (Q6)

- Call the m-filter runt pruning and cite Hartigan and Mohanty (1992) and Stuetzle (2003), noting the HDBSCAN equivalence (entries 38 and 39).
- Choose m with runt-size diagnostics (Stuetzle's rootograms) or a stability criterion.
- Consider adding HDBSCAN-style cluster stability of each component as an extra H0 vertex attribute.
- Contrast the filter with ToMATo's prominence threshold (entry 40).

### 5. Position the decorations

- **Positioning:** add a paragraph contrasting bar-level composition with relational/Dowker PH (entry 5) and chromatic kernel, image and cokernel persistence (entries 6 and 7), and cite RePHINE (entry 9) as the GNN precedent and decorated merge trees (entry 8) as the mathematical one.
- **Optional experiment:** use chromatic or Dowker features on a coarsened three-class grouping (epithelial, immune, stromal) as a baseline.

### 6. Stability and label noise

- Where a GTEx slide holds several tissue pieces, measure the piece-to-piece variability of graphcode-derived features against composition, for example as an intraclass correlation (motivated by entry 22).
- Test how sensitive the decorations are to HistoPLUS misclassification: relabel cells at random at controlled rates, or use class probabilities as soft compositions. Vipond 2021 argued that MP is robust to label noise; the decorations may not be.

### 7. Splits

- Keep all of a donor's organs on the same side of every split; the tissue clocks used GroupKFold by donor (entry 23).
- Stratify by organ and age bin, and report per-organ metrics as well as pooled ones.
