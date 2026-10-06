# DT–NB filtering: predictive value versus evaluation effects

## Abstract

This focused research synthesis examines the existing DT–NB findings through three questions: whether composition improves classification, whether order changes outcomes, and what before-versus-within-CV cleaning can establish. It combines a primary-source literature check with a reproducible audit of the saved Weka experiment. The two combined tree methods have lower equal-dataset accuracy means than uncleaned C4.5; NB compositions stay close to their baseline and below the attribute-only method. Order changes removal decisions without yielding a consistent predictive advantage. Before-CV filtering gives substantially higher scores but changes both access to held-out information and the population evaluated. No comparison is significant after a joint Holm correction of 24 exploratory tests. The defensible paper is an implementation-explicit empirical evaluation with removal diagnostics; neither classifier superiority nor verified noise identification is supported.

## 1. Questions and scope

RQ1 asks whether ordered combinations improve on matched uncleaned and single-step classifiers. RQ2 asks whether order affects prediction or the filters’ outputs relative to independent composition. RQ3 asks how before-versus-within-CV cleaning changes reported scores and what causal interpretation is possible.

The intended reader is the supervisor or a reviewer of empirical classification work. This analysis concerns `results/EXP-E9_farid`, using Weka J48 and original attribute columns. Earlier sklearn/CART/one-hot pilots are a different experiment and are not mixed into its evidence. The aim is to assess a bounded paper contribution, not to establish conference acceptance.

## 2. Retrieval and numerical method

On 2026-10-03, three independent research perspectives checked (1) the seed algorithms, tree weighting and joint selection, (2) filtering criticism and counterevidence, and (3) evaluation and statistical inference. Searches used title and author queries plus terms such as “decision tree naive Bayes hybrid,” “instance feature selection priorities,” “mislabeled training data filtering,” and “feature selection before cross-validation.” Publisher, proceedings, author-institution and author-deposited primary sources were preferred. An additional independent verifier checked all 13 references actually cited in the manuscript.

The final synthesis uses 15 focused works: the 13 manuscript sources plus two modern counterexamples discussed only here. It is not an exhaustive systematic review, and no claim is made that all citing papers or all work through 2026 were screened. A paper enters only when its existence and attributed finding can be verified; full-text-level claims require accessible text. Recent but dissimilar filtering studies are retained as boundary evidence rather than added as untested benchmark competitors.

The numerical audit recomputes dataset means from 12,000 protocol-B and 10,000 protocol-A fold score records. All recomputed aggregates match `metrics.csv` within 1e-9. Four thousand unique B cleaning events are deduplicated before summarizing removals. Twelve contrasts on each of accuracy and macro-F1 produce 24 two-sided permutation signed-rank tests using ten paired dataset means. The family is post-hoc and exploratory. Holm correction is joint across both metrics. Scripts, source hashes, runtime versions and detailed outputs are recorded in `paper/manuscript/analysis.json`.

## 3. Taxonomy of the evidence

| Theme | Question answered | Main sources | What the sources do not establish |
|---|---|---|---|
| Composition and weighting | Which mechanism is already known; what can ordering add? | Farid, Hall, Tsai [1–3] | New weighting principle or first-ever order study |
| Difficulty and filtering | Does disagreement identify corrupted labels? | Brodley/Friedl, John, Northcutt, Jiang, Pleiss, Swayamdipta [4–9] | Corruption labels for this study or universal filtering failure |
| Evaluation and uncertainty | What is being estimated, and how should methods be compared? | Demircioglu, Cawley/Talbot, Kapoor/Narayanan, Demsar [10–13] | Isolated leakage effect from a changing test population |
| Implementation context | What dataset/tool lineage is used? | Witten, Tavallaee [14–15] | Exact reproduction of unpublished seed software |

## 4. Composition changes the data, but gain is conditional

Farid [1] proposes two independent hybrids, while Hall [2] develops the same minimum-depth weighting principle with bagged unpruned trees. The present single pruned-tree implementation must therefore be positioned as a specific empirical composition, not a new weighting mechanism. Tsai [3] already examines joint feature/instance selection priorities using genetic algorithms and other final classifiers. That supports asking an ordering question without claiming that ordering itself is a new research problem.

| Design | Judge or selector sees | Distinguishing feature | Boundary |
|---|---|---|---|
| Farid’s independent algorithms | Supplied training data | N before DT or tree weights before NB | Not the ordered composition tested here |
| Hall’s weighting | Bagged unpruned trees | Averaged minimum-depth weights | Not an exact single-tree reproduction |
| Current NA | N sees original rows; A sees survivors | Plain judge, then fitted selection | Both operations can reduce useful information |
| Current AN | A sees original rows; N sees selected attributes | Weighted judge after selection | Order and judge inputs change together |
| Current parallel reference | Both operations see the original fold | Independently fitted row and column decisions | Helps assess sequential dependence; not a pure causal interaction estimate |

The saved experiment gives NB 75.93% accuracy and A–NB 77.35%; NA–NB and AN–NB give 75.63% and 76.35%. C4.5 gives 86.33%, A–DT 86.19%, N–DT 81.03%, NA–DT 81.21%, and AN–DT 80.89%. Thus chaining does not preserve the attribute-only NB mean advantage, and the combined tree means stay below the uncleaned tree. This is descriptive evidence for these settings, not a population-level ranking.

AN–NB exceeds NA–NB on eight datasets, yet its mean difference is only +0.71 pp and raw signed-rank p=0.275. AN–DT is −0.32 pp on the mean, with raw p=0.922. Neither order can be declared a general winner. Parallel NB/DT means are 76.39%/81.05%; sequential methods provide no consistent mean advantage over that control.

## 5. A difficult observation is not necessarily noise

Brodley/Friedl [4] distinguish erroneous labels from valid exceptions that conflict with a filter’s learning bias; John [5] likewise discusses losses on underrepresented, noise-free patterns. Their procedures differ from the single in-sample NB disagreement rule studied here. Northcutt [6] and Jiang [7] investigate different approaches to label uncertainty and correction/filtering, so our negative mean results do not disprove their findings.

Counterevidence matters: Pleiss [8] reports benefits from calibrated filtering based on training dynamics, whereas Swayamdipta [9] finds value in difficult or ambiguous examples under particular NLP conditions. Together, these works bound the inference: the usefulness of filtering depends on how errors are identified and what the removed examples contribute. Their neural-network tasks do not directly predict the behavior of tabular NB/C4.5.

The current logs show plain filtering removing 47.31% of glass training rows but 4.14% on iris. For breast cancer, recurrence cases form 29.72% of training rows on average, yet plain NB removes 50.63% of them versus 14.51% of non-recurrence cases. Weighted filtering after A raises the recurrence removal rate to 58.97%. Entire classes disappear in some training folds, despite the two-class safety guard. These measurements expose selection behavior; they do not establish which labels are incorrect or causally prove why accuracy declines.

The attribute stage also responds to prior row removal. On glass, retained attributes fall from 97.00% when A sees the original fold to 61.00% after N. On vote, retention falls from 27.38% to 7.62%. That is direct evidence of altered fitted selection, even though it is not evidence of beneficial interaction.

## 6. Evaluation protocol changes both information and target population

Demircioglu [10] directly examines supervised feature selection before versus inside CV, while Cawley/Talbot [11] addresses broader selection bias and Kapoor/Narayanan [12] explains leakage and reproducibility concerns. These sources justify fold-local supervised fitting. They do not make our protocol contrast an isolated leakage experiment, because row filtering changes the test population as well.

NA–DT rises from 81.21% in B to 94.72% in A; AN–DT rises from 80.89% to 96.32%. The descriptive differences are +13.51 and +15.43 pp. Attribute-only differences are much smaller: A–NB +0.72 pp and A–DT +0.33 pp. The pattern is consistent with a strong sensitivity to row selection and cleaning placement, but the contrast changes several things at once. Calling all of it “leakage inflation” would overidentify the mechanism.

Demšar [13] motivates using one paired score per dataset for cross-dataset inference. All 24 exploratory tests are nonsignificant after Holm correction; the smallest adjusted p is 0.328. This does not establish equality. Large task-specific losses remain visible, and ten selected datasets provide limited precision. Repetition estimates split sensitivity, not ten times as many independent tasks.

## 7. Synthesis and paper positioning

RQ1 is answered by matched baselines and reference arms: neither composed order shows a consistent classification advantage under B. RQ2 is answered by both score contrasts and removal logs: fitted outputs depend on sequence, but predictive order preference depends on the final classifier and metric. RQ3 is answered by the A/B contrast: scores change substantially for row-filtering methods, but the difference combines information access, sample selection and fitting changes.

The strongest framing is an implementation-explicit audit of supervised DT–NB filtering, ordered composition and evaluation. Prior weighting and ordering literature rules out a broad technique-novelty claim. The lack of known corruption labels rules out demonstrated noise detection. The seed paper’s original cleaning placement remains unresolved, so no allegation about its protocol is supported.

The primary seed-PDF audit also found that its code description uses Weka-derived DT/NB Java components. An earlier historical note saying “not Weka” should not be relied on. Its prose aggregate for Algorithm 2 (86.7%) differs from the arithmetic mean of its Table 11 (83.423%). The present draft avoids quoting those aggregates as verified headline comparisons.

## 8. Open questions and readiness

A fixed held-out population with separately varied cleaning-information access would better separate selection from leakage. A known-corruption experiment could test whether the NB judge discards actual label errors or valid hard cases. A rare-class preservation comparison would assess the consequences of class disappearance without assuming those consequences from removal counts alone.

Current limits include legacy benchmarks, different available dataset sizes, globally constructed nominal vocabularies, literal-question-mark categories, variable macro-F1 class sets, rare classes under ten-fold splitting, no official NSL-KDD test evaluation, and no fresh full-model rerun. The empty-tree behavior differs between standalone A–NB protocols; no such event occurred in saved B records, but it should be repaired and tested before extending the experiment to other data.

The revised manuscript is a complete evidence-grounded draft. Its acceptance prospects require a named venue and supervisor judgment. The built-in LaTeX compiler failed with “Unable to find standard directories for platform,” so rendered page count, overlap, float placement and venue compliance remain unverified. The open source is standalone and was kept in the same editor; no replacement PDF was generated.

## References and primary retrieval links

[1] Farid et al., “Hybrid decision tree and naïve Bayes classifiers for multi-class classification tasks,” ESWA, 2014. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0957417413007100).

[2] Hall, “A decision tree-based attribute weighting filter for naive Bayes,” KBS, 2007. [Author’s working-paper text](https://researchcommons.waikato.ac.nz/bitstreams/f665d717-07a0-40e7-950f-539ff1dab4d8/download).

[3] Tsai et al., “Genetic algorithms in feature and instance selection,” KBS, 2013. [Publisher](https://www.sciencedirect.com/science/article/pii/S0950705112003140).

[4] Brodley and Friedl, “Identifying Mislabeled Training Data,” JAIR, 1999. [Author-deposited text](https://arxiv.org/pdf/1106.0219).

[5] John, “Robust Decision Trees: Removing Outliers from Databases,” KDD, 1995. [Proceedings PDF](https://cdn.aaai.org/KDD/1995/KDD95-044.pdf).

[6] Northcutt et al., “Confident Learning: Estimating Uncertainty in Dataset Labels,” JAIR, 2021. [DOI](https://doi.org/10.1613/jair.1.12125).

[7] Jiang et al., “Which Is More Effective in Label Noise Cleaning, Correction or Filtering?” AAAI, 2024. [DOI](https://doi.org/10.1609/aaai.v38i11.29183).

[8] Pleiss et al., “Identifying Mislabeled Data using the Area Under the Margin Ranking,” NeurIPS, 2020. [Proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf).

[9] Swayamdipta et al., “Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics,” EMNLP, 2020. [Official proceedings](https://aclanthology.org/2020.emnlp-main.746/).

[10] Demircioğlu, “Measuring the bias of incorrect application of feature selection when using cross-validation in radiomics,” Insights into Imaging, 2021. [Institutional full text](https://duepublico2.uni-due.de/servlets/MCRFileNodeServlet/duepublico_derivate_00077597/Insights_Imaging_2021_12_172.pdf).

[11] Cawley and Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” JMLR, 2010. [Primary record](https://jmlr.org/papers/v11/cawley10a.html).

[12] Kapoor and Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, 2023. [Institutional record](https://collaborate.princeton.edu/en/publications/leakage-and-the-reproducibility-crisis-in-machine-learning-based-/).

[13] Demšar, “Statistical Comparisons of Classifiers over Multiple Data Sets,” JMLR, 2006. [Primary record](https://jmlr.org/papers/v7/demsar06a.html).

[14] Witten et al., Data Mining: Practical Machine Learning Tools and Techniques, fourth edition, 2016. [Publisher](https://www.sciencedirect.com/book/9780128042915/data-mining).

[15] Tavallaee et al., “A detailed analysis of the KDD CUP 99 data set,” CISDA, 2009. [DOI](https://doi.org/10.1109/CISDA.2009.5356528).
