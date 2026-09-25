# Verification ledger — 2026-09-17 audit of the deep-research wave-2 evidence

*Scope: every load-bearing citation of `notes/deep-research-mutual-hybrid.md` and `notes/strategy-mutual-hybrid.md`, plus the seed paper itself, re-checked against primary sources on 2026-09-17. Where an earlier record was wrong, the correction is listed here and applied in the report. Read this file before citing anything from the report.*

## A. Corrections applied (earlier records were WRONG — do not reuse old values)

| # | Item | Wrong record (source of error) | Corrected record (verified) |
|---|---|---|---|
| C1 | Zhang "Attribute and instance weighted naive Bayes" | DOI `10.1016/j.patcog.2020.107725`; authors "Zhang, Y. et al." (Sep-13 review) | DOI **10.1016/j.patcog.2020.107674**; authors **Zhang, H., Jiang, L., Yu, L.** — Pattern Recognition, 2021 (Crossref) |
| C2 | Kapoor & Narayanan, "Leakage and the reproducibility crisis…" | DOI `10.1016/j.patter.2023.100794` (resolves to an unrelated article) | DOI **10.1016/j.patter.2023.100804** — Patterns, 2023 (Crossref title search) |
| C3 | Northcutt et al., "Confident learning" | authors "Northcutt, Athalye, Mueller" | authors **Northcutt, C.G., Jiang, L., Chuang, I.L.** — JAIR 72, 2021 (Crossref). (Athalye/Mueller belong to the different "Pervasive label errors" NeurIPS paper.) |
| C4 | "Attribute weighted naive Bayes classifier" CMC 2022 | authors "Foo, Chua, Ibrahim" | authors **Kalra, M., Kumar, V., Kaur, M., Ahmed Idris, S., Öztürk, Ş.** — CMC 71(1), 2022 (Crossref) |
| C5 | Zeng et al., "Improved naive Bayes with mislabeled data" | year 2023 | year **2024**; authors Zeng, Q., Zhu, Y., Zhu, X., Wang, F., Zhao — Statistics and Its Interface (DOI 10.4310/22-sii757, Crossref) |
| C6 | Zhu & Wu 2004, "Class noise vs. attribute noise" | DOI `10.1007/s10462-004-0751-x` | DOI **10.1007/s10462-004-0751-8** (Crossref title search) |
| C7 | "A new three-way incremental naive Bayes classifier" | authors "Yang, Ren, Zhang" | authors **Yang, Z., Ren, J., Zhang, Z., Sun, Y., Zhang, C.** — Electronics 12(7):1730, 2023 |
| C8 | IWHNB (Mathematics 2021) | authors "Yu, Gan, Chen" | authors **Yu, L., Gan, S., Chen, Y., Luo, D.** |
| C9 | García-Pedrajas 2014 (Evol. Comput.) | author order "Pérez-Rodríguez, de Haro-García" | order **García-Pedrajas, N., de Haro-García, A., Pérez-Rodríguez, J.** |
| C10 | SI(FS)² (Pattern Recognition 2021) | authors "Romero del Castillo et al." | **García-Pedrajas, N., Romero del Castillo, J.A., Cerruela-García, G.** |
| C11 | Xu, AVF instance weighting filter | author "W. Xu" only | **Xu, W., Jiang, L., Yu, L.** — JETAI (online 2018; vol 31, 2019) |
| C12 | Sample-selection bias (IEEE TMM 2024) | "Liu, Sheng, Sun, et al." | **Liu, Z., Sheng, V.S., Sun, T., Yao** (Crossref) |
| C13 | Hu et al. ICML 2025 | — (not yet in survey; now verified) | **Hu, G., Liu, F., Gong, M., Wang, G., Peng** — "Learning Imbalanced Data with Beneficial Label Noise," ICML 2025 (poster page + abstract) |
| C14 | HyCASTLE (KBS 2022) | "Veneri, M.D." | **Delli Veneri, M., Cavuoti, S., Abbruzzese, R., Brescia, M.** (DOI 10.1016/j.knosys.2022.108566) |
| C15 | CEUR AdaBoost+NB paper | authors "Ahammed et al." | **Ahammed, A., Harangi, B., Hajdu, A.** — CEUR Vol-2874 (conf. Nov 2020, proceedings 2021); full text extracted from the PDF |

## B. Load-bearing sources re-verified at primary level (2026-09-17)

| Citation | Canonical record | Status | Checked at |
|---|---|---|---|
| Farid et al. 2014, ESWA 41(4):1937–1946 — **seed** | local PDF `../../11th_trimester/bda/reference/Farid_2014_hybrid_DT_NB_multiclass.pdf` | **FULL TEXT extracted** | algorithms verbatim (below, §C) |
| Latubessy et al. 2025, JAIT 5:278–288 — closest "both knobs" prior | DOI 10.37965/jait.2025.0734; ojs.istp-press.com/jait/article/view/734 | **ABSTRACT/landing page read** (Crossref authors: Latubessy, A., Winarko, E., Musdholifah, A., Kusrohmaniah, S.) | no tree; mPIR + FTAWNB; gaming-disorder dataset; +1.28%/+1.4% accuracy — single domain, do not generalize |
| Demircioğlu 2021, Insights into Imaging 12:172 — leakage magnitude | DOI 10.1186/s13244-021-01115-1; PubMed 34817740 | **VERIFIED** (publisher paywalled; PubMed abstract + search-confirmed numbers) | bias up to +0.15 AUC-ROC / +0.29 AUC-F1 / +0.17 accuracy; grows as samples-per-feature falls |
| Jiang et al. AAAI 2024 — correction vs filtering | DOI 10.1609/aaai.v38i11.29183; authors Jiang, G., Zhang, J., Bai, X., Wang, W., Meng, D. | **VERIFIED** (Crossref + AAAI/ACM metadata; abstract-level theorem statement) | "dataset level, correction is more effective than filtering" (Theorem 5 per abstract coverage); filtering "overcleaning" |
| Zhang et al. 2021 AIWNB — "both knobs, no tree" | DOI 10.1016/j.patcog.2020.107674 (corrected, C1) | **VERIFIED** (Crossref) | authors/citation fixed; claim boundaries unchanged |
| Zhang et al. TKDE MAWNB — random-tree views | DOI 10.1109/TKDE.2022.3177634 | **VERIFIED** (Crossref: Zhang, H., Jiang, L., Zhang, W., Li, C.; issued 2022, print 2023) | random trees build auxiliary views; NOT the min-depth formula |
| Zhang et al. 2026, Pattern Recognition — dual-view instance weighted NB | DOI 10.1016/j.patcog.2025.112181 | **VERIFIED** (Crossref: Zhang, H., Meng, K., Lv, P., He, S., Xu, M.; issued 2026, online 2025) | instance-weighting successor line; do not claim it "dropped attribute weighting" — only that it is instance-side |
| Wu et al. UAI 2026 PMWNB | PMLR 337:7416–7432, proceedings.mlr.press/v337/wu26b.html | **VERIFIED** (page read: Siyao Wu, Huan Zhang, Kexin Meng, Zhipeng Ding, Pei Lv) | perturbation/matrix-view learned weights; 59 datasets claimed |
| Villuendas-Rey et al. 2024, Applied Sciences 14(18):8459 (ROFS) | DOI 10.3390/app14188459 | **METADATA-ONLY** — full text 403 from this environment | simultaneous instance+attribute selection for noise; **whether it compares orderings is NOT confirmable** — no order claim may be attributed to it |
| Wong et al. 2020, Inf. Sci. 520:445–455 | DOI 10.1016/j.ins.2020.02.021 | **VERIFIED** (Crossref) | instance-filtering routing hybrids, benchmarks Farid |
| Hall 2007, KBS 20(2):120–126 | DOI 10.1016/j.knosys.2006.11.008 | **VERIFIED** (Crossref) | the mandatory prior for Alg 2's formula |
| Brodley & Friedl 1999, JAIR 11 | DOI 10.1613/jair.606 | **VERIFIED** (Crossref) | single-model filter = weakest design |
| Sáez et al. 2015 SMOTE-IPF | DOI 10.1016/j.ins.2014.08.051 | **VERIFIED** (Crossref) | filter-after-resample ordering under imbalance |
| Tsai et al. 2013, KBS 39 | DOI 10.1016/j.knosys.2012.11.005 | **VERIFIED** (Crossref) | priority between FS/IS changes outcomes |
| García-Pedrajas et al. 2014, Evol. Comput. 22(1) | DOI 10.1162/evco_a_00102 | **VERIFIED** (Crossref) | "interwoven" joint selection |
| García-Pedrajas et al. 2021, Pattern Recognition 111 (SI(FS)²) | DOI 10.1016/j.patcog.2020.107723 | **VERIFIED** (Crossref) | fast simultaneous selection |
| Kusy & Zajdel 2024, KBS 296 | DOI 10.1016/j.knosys.2024.111844 | **VERIFIED** (Crossref; correct authors Kusy, M., Zajdel, R.) | fused reduction — **note**: not "Kusy et al." with unnamed co-authors |
| Jiang et al. 2019 CFW, IEEE TKDE 31(2) | DOI 10.1109/TKDE.2018.2836440 | **VERIFIED** (Crossref) | correlation-based weighting filter |
| Zhang & Jiang 2022 FTAWNB, Neurocomputing 488 | DOI 10.1016/j.neucom.2022.03.020 | **VERIFIED** (Crossref) | weighting half of the Latubessy pipeline |
| Ou et al. 2025 MS-WNBC, Inf. Sci. 721 | DOI 10.1016/j.ins.2025.122568 | **VERIFIED** (Crossref: Ou, G., He, Y., Fournier-Viger, P., Huang, J.) | multi-source weights |
| Tong et al. 2023 ARRAY, JSS 202 | DOI 10.1016/j.jss.2023.111721 | **VERIFIED** (Crossref: Tong, H., Lu, W., Xing, W., Wang, S.) | triple feature weighting + instance selection (transfer) |
| Samami et al. 2020, Physica A 540:124219 | DOI 10.1016/j.physa.2020.124219 | **VERIFIED** (Crossref) | graded remove/relabel |
| Liu et al. 2024, IEEE TMM 26 | DOI 10.1109/TMM.2024.3368910 | **VERIFIED** (Crossref, C12) | selection bias under imbalance |
| Brishti et al. 2025, ICT Express 11(6) | DOI 10.1016/j.icte.2025.09.011 | **VERIFIED** (Crossref: Brishti, F., Zhang, F., Mohammed, S., Bai, L., Wu, F.) | imbalance × label-noise review |
| Szeghalmy & Fazekas 2024, KBS 297 | DOI 10.1016/j.knosys.2024.112236 | **VERIFIED** (Crossref) | filter×imbalance comparative study |
| Zerhari et al. 2019 MIPCNF, JIFS 37(5) | DOI 10.3233/JIFS-190261 | **VERIFIED** (Crossref) | multi-partitioning iterative filter |
| Mantovani et al. 2024, DMKD 38 | DOI 10.1007/s10618-024-01002-5 | **VERIFIED** (Crossref: Gomes Mantovani, R., Horváth, T., Rossi, A., Cerri, R., Barbon Junior, S.) | tree hyperparameter study |
| Changpetch et al. 2021, Computation 9(9):99 | DOI 10.3390/computation9090099 | **VERIFIED** (Crossref) | tree discretization → NB inputs |
| Hassan & Farid 2025, ICCIT | DOI 10.1109/ICCIT68739.2025.11490552 | **VERIFIED** (Crossref) | seed group's NBTree pivot |
| Zhu et al. 2023, Annals of OR | DOI 10.1007/s10479-023-05671-1 | **VERIFIED** (Crossref: Zhu, Y., Wang, Y., Qin, L., Zhang, B., Shia) | reliability-weighted NB under noise |
| Varma & Simon 2006, BMC Bioinformatics 7:91 | DOI 10.1186/1471-2105-7-91 | **VERIFIED** (Crossref; **note: 2006, not 2007**) | CV-tuning optimism |
| Nadeau & Bengio 2003, Machine Learning 52(3) | DOI 10.1023/A:1024068626366 | **VERIFIED** (Crossref) | corrected resampled t-test |
| Bouke & Abdullah 2023, ESWA | DOI 10.1016/j.eswa.2023.120715 | **VERIFIED** (Crossref title match) | preprocessing leakage on intrusion data (NSL-KDD) |
| Wilton & Ye AAAI 2024 | arXiv:2312.12937 (Jonathan Wilton, Nan Ye) | **VERIFIED** (arXiv API: title + authors match) | robust tree losses |
| Roth 2026 leakage landscape | arXiv:2604.04199 (Simon Roth) | **VERIFIED as preprint** (arXiv API: title + author match) | 2,047 tabular datasets; not peer-reviewed — hedge accordingly |
| Fan et al. 2022 joint selection RL | arXiv:2205.07867 (Wei Fan, Kunpeng Liu, Hao Liu) | **VERIFIED** (arXiv API) | coupled decisions framing |
| Pio et al. 2024, JIDM 15(1) | journals-sol.sbc.org.br/index.php/jidm/article/view/3365 | **VERIFIED page exists** (HTTP 200, title matches); Crossref DOI lookup fails (SBC registry) — cite without DOI | filter-choice meta-learning |
| Ahammed et al. CEUR Vol-2874 paper 1 | ceur-ws.org/Vol-2874/paper1.pdf | **FULL TEXT read** (C15) | AdaBoost weight increments for NB — soft-weight analog |

## C. Seed-paper extraction record (direct, from the PDF)

Extracted 2026-09-17 with PyMuPDF (`_seed_extract.txt`, 57k chars, 10 pages). Verbatim facts now usable in the writeup:

- **Algorithm 1** (verbatim structure): compute priors P(Ci) and conditionals P(Aij|Ci); for each training instance find P(Ci|xi); *"if xi is misclassified, remove xi from D"*; then build the tree (C4.5-style DTBuild) on the remaining data.
- **Algorithm 2** (verbatim structure): build tree T; for each attribute Ai: *"if Ai is not tested in T, then Wi = 0; else d = minimum depth of Ai ∈ T, and Wi = 1/√d"*; then compute class conditional probabilities *"P(Aij|Ci)^Wi"* — the weights are exponents (their Eq. 14) — and classify with only the selected attributes.
- **Protocol text**: "10-fold cross validation" everywhere (19 mentions); the filter is described as acting on "the training set" — **the paper never states whether filtering/selection happens inside or outside the CV folds.** C1's two-variant audit (paper-style vs within-fold) is therefore framed as *testing the underspecified protocol*, not as accusing a stated one.
- **Implementation**: Java in NetBeans IDE 7.1 (not Weka), so `CART ≠ C4.5` remains a stated limitation for replication.
- The reference list contains no Hall (2007) and no Ratanamahatana & Gunopulos (Sep-13 review verified this against the same PDF; re-confirmed section text references only the Weka-era Hall).

## D. Verification limitations (honest boundaries of this audit)

1. Semantic Scholar API and DBLP were unreachable/rate-limited from this environment during the audit; verification leaned on Crossref, arXiv API, PubMed, publisher pages, and direct PDF extraction. Crossref does not cover AAAI/IEEE DOIs uniformly (AAAI/IEEE were verified where DOI resolution succeeded).
2. ROFS (MDPI) full text returned HTTP 403 — all ROFS claims are metadata-level only.
3. Several references in the survey originate from the Sep-13 verified review (pre-2019 canon); they were not all re-verified this session. The ones re-checked today are listed in §B and all matched.
4. The "empty cell" (no mutual DT–NB system) and "no controlled order study" statements are *retrieval-bounded*: they reflect the searches described in the report's Methodology section, not a proof of absence. Wording in the report was softened accordingly.
5. `10.1007/s10462-004-0751-x` (wrong) vs `-8` (correct) — a one-character DOI error class; assume similar risk anywhere a DOI was copied from secondary sources without resolution. Every DOI in the updated reference list resolved on 2026-09-17 except where marked METADATA-ONLY.

## E. Reading order (for the tracker's `to-read` rows)

1. **Before any coding:** Hall 2007 (L02) and the seed's Alg 2 extraction above; Brodley & Friedl 1999 (L04) — these govern the two baselines.
2. **Before E3a/E3b design:** Demircioğlu 2021 (L53), Kapoor & Narayanan 2023 (L54), Sáez 2015 (L49), Tsai 2013 (L48).
3. **Before E5 design:** Jiang AAAI 2024 (L31), Samami 2020 (L33), Liu 2024 (L34), Northcutt 2021 (L32).
4. **Before writing the positioning paragraph:** Latubessy 2025 (L30), Zhang/Jiang/Yu 2021 (L28 — corrected), MAWNB (L37), Wong 2020 (L07), Zhang 2026 dual-view (L41).
