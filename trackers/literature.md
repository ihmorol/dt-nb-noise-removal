# Literature tracker

Status values: `to-read` → `reading` → `noted` (summary exists in notes/) → `cited` (used in a draft).
Full verified review lives in `../reference/literature-review-dt-nb-hybrid.md` — this file only tracks MY reading workflow.

## Must-know core (seed's two directions)

| ID | Paper | Why it matters to me | Status |
|---|---|---|---|
| L01 | Farid et al. 2014, ESWA 41(4) — seed | Both hybrids; my target of replication | noted |
| L02 | **Hall 2007, KBS 20(2):120–126** — tree-based attribute weighting filter | Near-identical to Alg. 2 (1/√d, root=1, exponents); seed does NOT cite it; my closest-prior + baseline | to-read |
| L03 | John 1995, KDD — Robust decision trees | Direct ancestor of Alg. 1 (delete-what-model-gets-wrong loop) | to-read |
| L04 | Brodley & Friedl 1999, JAIR 11 — Identifying mislabeled data | Shows single-algorithm filters (Alg. 1's design) are the weakest; consensus/majority = my C3 design | noted (E11's filter design; summary in `notes/committee-filter-literature.md`) |
| L05 | Wilson 1972, IEEE TSMC — Edited data | Theoretical root of instance filtering | to-read |
| L06 | Zaidi et al. 2013, JMLR 14 — Attribute weighting for NB | Formalizes Hall's weight; WANBIA baseline for C4 | to-read |
| L07 | Wong, Yang & Chen 2020, Inf. Sci. 520 — Instance-filter hybrids | Closest successor; beats Farid's hybrid; my filtering baseline | to-read |
| L08 | Sarker et al. 2017, AusDM — NB noise detection w/ threshold | Thresholded deletion = one C3 variant | to-read |
| L09 | Frénay & Verleysen 2014, IEEE TNNLS — Label-noise survey | Framing umbrella for the noise-filter family | to-read |
| L10 | Kohavi 1996, KDD — NBTree | Canonical DT-NB architecture; seed cites the idea, not the paper | to-read |
| L11 | Zhang & Sheng 2004, ICDM — Weighted NB ranking | Weighted-NB framework Alg. 2 plugs into | to-read |
| L12 | Farid, Harbi & Rahman 2010, IJNSA — NB+DT for IDS | Seed's own predecessor; shows author's lineage | to-read |

## Context layer (skim; cite without deep notes)

| ID | Paper | Status |
|---|---|---|
| L20 | Friedman et al. 1997 — TAN, Machine Learning 29 | to-read |
| L21 | Webb et al. 2005 — AODE, Machine Learning 58 | to-read |
| L22 | Ratanamahatana & Gunopulos 2003 — SBC, Applied AI 17 | to-read |
| L23 | Jiang et al. 2016 — DFW, Eng. Appl. AI 52 | to-read |
| L24 | Jiang et al. 2007 — Improving-NB survey, ADMA | to-read |
| L25 | Smith & Martinez 2011 — IJCNN (PRISM filter) | to-read |
| L26 | Verbaeten & Van Assche 2003 — MCS | to-read |
| L27 | Hassan & Farid 2025 — ICCIT (NBTree follow-up) | to-read |
| L28 | Zhang, Jiang & Yu 2021 — Attribute+instance weighted NB, Pattern Recognition (DOI 10.1016/j.patcog.2020.107674 — corrected 2026-09-17; earlier record had a nonexistent DOI) | to-read |
| L29 | Hall & Frank 2008 — DTNB, FLAIRS | to-read |

## Deep-research wave 2 (verified 2026-09-17 — one-line findings in `notes/deep-research-mutual-hybrid.md`)

Metadata spot-verified the same day via Crossref/arXiv/PubMed/publisher pages; corrections found during the audit (incl. L28's DOI) are recorded in `notes/verification-ledger-2026-09-17.md` — consult it before citing. Status `noted` = finding summarized in the deep-research report; read before citing.

### Combined / novelty-critical

| ID | Paper | Why it matters to me | Status |
|---|---|---|---|
| L30 | Latubessy et al. 2025, J. Artificial Intelligence and Technology 5 — mPIR + fine-tuned attribute-weighted NB | **Closest live "both knobs" prior** (instance reduction + attribute weighting, one classifier) — but NO tree, single gaming-disorder dataset, ~1.3–1.4% gains | noted |
| L31 | Jiang, G. et al. 2024, AAAI-38 — "Which is more effective in label noise cleaning, correction or filtering?" | Proves filtering over-cleans vs correction; FCF fusion wins — the theoretical backbone for E5's relabel/reweight arms | noted |
| L32 | Northcutt, Jiang & Chuang 2021, JAIR 72 — Confident learning | Confidence-based noise handling + noise-rate estimation (cleanlab); the principled version of F3 thresholds | noted |
| L33 | Samami et al. 2020, Physica A — high-agreement filtering (HAVF) | Grades noise: remove strong, RELABEL weak — template for the E5d arm | noted |
| L34 | Liu et al. 2024, IEEE Trans. Multimedia — preventing bias in sample selection | Sample-selection cleaning is class-biased under imbalance → motivates per-class thresholds + E6 class diagnostics | noted |
| L35 | Johnson & Khoshgoftaar 2022, ACM Computing Surveys — classifying big data with label noise | Reports NB as the most stable learner under label noise — supports NB-as-committee-member, NOT NB-as-sole-judge | noted |
| L36 | Garcia et al. 2019, KBS — new label noise injection methods | Random noise injection flatters filters; complexity-based injection degrades them — governs any synthetic-noise part of E5 | noted |

### Attribute weighting for NB (the post-2019 successors of Alg 2)

| ID | Paper | Why it matters to me | Status |
|---|---|---|---|
| L37 | Zhang, Jiang et al. 2023, IEEE TKDE — Multi-view attribute weighted NB (MAWNB) | Uses random trees to build weight views — closest living relative of tree-based weighting; strong E8 baseline | noted |
| L38 | Zhang & Jiang 2022, Neurocomputing — Fine-tuning attribute weighted NB (FTAWNB) | The weighting half of L30; gain-ratio fine-tuning | to-read |
| L39 | Jiang et al. 2019, IEEE TKDE — Correlation-based feature weighting (CFW) | Started the post-2019 weighting wave | to-read |
| L40 | Wu et al. 2026, UAI/PMLR — Matrix-view weighting (PMWNB) | Newest learned-weight NB (WANBIA-style objective); shows the field's direction | to-read |
| L41 | Zhang et al. 2026, Pattern Recognition — General dual-view instance weighted NB | Instance-side successor of L28 (attribute side not pursued there) — consistent with the retrieval finding that nobody pairs both knobs with a tree | to-read |
| L42 | Tong et al. 2023, J. Systems & Software — ARRAY triple feature-weighted transfer NB | Feature weighting + instance selection in one NB pipeline (transfer setting) | to-read |
| L43 | Xu 2019, JETAI — attribute-value-frequency instance weighting filter | Couples attribute stats to instance weights inside NB | to-read |

### Coupling the two levels (order / simultaneity / over-cleaning)

| ID | Paper | Why it matters to me | Status |
|---|---|---|---|
| L44 | García-Pedrajas et al. 2014, Evolutionary Computation — memetic simultaneous instance+feature selection | States the two selections are "interwoven" — the premise of the mutual hybrid, from the data-reduction school | noted |
| L45 | García-Pedrajas et al. 2021, Pattern Recognition — SI(FS)2 | Fast simultaneous selection, high-dimensional | to-read |
| L46 | Villuendas-Rey et al. 2024, Applied Sciences — ROFS simultaneous instance-attribute selection for noise | Deterministic simultaneous selection aimed at noise; benchmarks ISF/IREACE/CAISE | to-read |
| L47 | Kusy et al. 2024, KBS — fused instance+feature data reduction | Newest fusion pair | to-read |
| L48 | Tsai, Eberle & Chu 2013, KBS — GA priority between feature and instance selection | Direct evidence that relative emphasis/order changes outcomes | noted |
| L49 | Sáez et al. 2015, Information Sciences — SMOTE-IPF | Sharpest order effect: filter-before-resample strips minority borderline instances → order ablation E3b is justified | noted |
| L50 | Derrac et al. 2012, Information Sciences — fuzzy-rough FS + evolutionary IS | One-way interaction: adding FS improves IS | to-read |
| L51 | Fragoudis et al. 2005, IJCAI — integrating feature and instance selection (FIS) | Earliest explicit joint treatment | to-read |
| L52 | Zhu & Wu 2004, Artificial Intelligence Review — class noise vs attribute noise | Class noise hurts more — the two operations are not symmetric | noted |
| L73 | Khoshgoftaar & Rebours 2007, JCST 22(3):387–396 — Iterative-Partitioning Filter (IPF) | The original IPF (NOT Wilson & Martinez — verified 2026-10-02): repeat the k-fold vote and sum the votes; E11's committee uses its 3×3 voting | noted |

### Evaluation methodology (the C1/C4 arsenal)

| ID | Paper | Why it matters to me | Status |
|---|---|---|---|
| L53 | Demircioğlu 2021, Insights into Imaging — bias of feature selection outside CV | Quantifies the leakage: up to +0.15 AUC / +0.17 accuracy, worse when samples-per-feature drops — the audit's citation | noted |
| L54 | Kapoor & Narayanan 2023, Patterns — leakage and the reproducibility crisis | Modern umbrella ref for preprocessing leakage | noted |
| L55 | Cawley & Talbot 2010, JMLR — over-fitting in model selection | Canonical: selection outside evaluation is optimistically biased; prescribes nested CV | noted |
| L56 | Nadeau & Bengio 2003, Machine Learning — inference for generalization error | Corrected resampled t-test — required for my 10×10 repeated CV (E7) | noted |
| L57 | Demšar 2006, JMLR — statistical comparisons of classifiers | The multi-dataset comparison protocol for E7 | noted |
| L58 | García & Herrera 2008, JMLR — all-pairwise extension | Holm/Shaffer corrections for the full E7 system grid | to-read |
| L59 | Benavoli et al. 2017, JMLR — Bayesian classifier comparison (baycomp) | Optional upgrade for E7 | to-read |
| L60 | Bouke & Abdullah 2023, ESWA — preprocessing leakage on NSL-KDD/UNSW-NB15 | Leakage shown on MY benchmark's NSL-KDD — where the E3a gap should appear | noted |
| L61 | Roth 2026, arXiv (preprint) — leakage landscape over 2,047 tabular datasets | Caution: on low-dim tabular data some leakage classes are negligible → measure, don't assume (peer review status: preprint) | to-read |

### Recent NB-under-noise and DT-robustness (context)

| ID | Paper | Why it matters to me | Status |
|---|---|---|---|
| L62 | Yang et al. 2023, Electronics — three-way incremental NB | Accept/reject/defer instance handling = soft alternative to deletion | to-read |
| L63 | Zeng et al. 2023, Statistics and its Interface — improved NB with mislabeled data | From the seed's own citation network; NB under label noise without deletion | to-read |
| L64 | Zhu et al. 2023, Annals of Operations Research — NB reliability measurement for noisy labels | Same network; reliability-weighted NB | to-read |
| L65 | Veneri et al. 2022, KBS — HyCASTLE | Hybrid classification with typicality/labels/entropy | to-read |
| L66 | Wilton & Ye 2024, AAAI — robust loss functions for decision trees with noisy labels | Alternative to pre-filtering: absorb noise during tree growth | to-read |
| L67 | Mantovani et al. 2024, DMKD — better trees (hyperparameter tuning) | DT-side configuration evidence for the downstream tree | to-read |
| L68 | Zerhari et al. 2019, J. Intelligent & Fuzzy Systems — MIPCNF | Multi-iterative partitioning filter; "no single filter is universally best" | to-read |
| L69 | Pio et al. 2024, JIDM — meta-learning noise filter recommendation | Adaptive filter choice — anti-hard-coded-deletion argument | to-read |
| L70 | Brishti et al. 2025, ICT Express — imbalanced classification with label noise (review) | Systematic review of noise×imbalance | to-read |
| L71 | Hu et al. 2025, ICML — learning imbalanced data with beneficial label noise | Adding asymmetric noise can rebalance — deletion-only framing is incomplete | to-read |
| L72 | Szeghalmy & Fazekas 2024, KBS — noise filtering of imbalanced datasets | Filter×sampling interaction under imbalance | to-read |



## Discipline

- One `notes/` file per deep-read paper only (L01–L12); everything else gets 2 lines in its row here.
- A paper cannot reach `cited` until its row says `noted`.
