# Combining the two DT–NB hybrids: the 2026 landscape, and why the mutual lane is still open

*Deep-research report, 2026-09-17. Companion to `../reference/literature-review-dt-nb-hybrid.md` (Sep 13) — that file covers the pre-2019 canon in depth; this report covers what has happened since and what it means for the mutual-hybrid design.*

## Abstract

Farid et al. (2014) proposed two *separate* decision-tree/Naive-Bayes hybrids — an NB filter that deletes misclassified instances before tree induction, and a tree that weights attributes by inverse depth for NB. This report maps the 2019–2026 literature around a proposed **mutual combination** (NB filters instances → tree grows on cleaned data → tree weights attributes → weighted NB, in one loop), using five search perspectives (successor school, label-noise filtering, attribute weighting, instance×feature interaction, evaluation methodology) over a verified corpus of ~55 recent works plus the 410-paper citation dump of the seed. Three findings matter. First, **the closed mutual loop still does not exist in the published record**; the closest artifacts treat both knobs but drop one component each (no tree, no filtering, or no weighting). Second, recent evidence converges against the seed's hard deletion: single-model filters are the weakest design, correction beats filtering under comparable noise rates, and selection-based cleaning biases minority classes — so the mutual hybrid should be built **confidence-aware from the start**. Third, the evaluation methodology literature now quantifies exactly the leakage the seed's protocol invites (up to +0.15 AUC when selection is fit outside cross-validation), while also warning that the inflation on ordinary low-dimensional tabular data must be *measured*, not assumed. The report ends with open problems and points to the companion strategy note.

## 1. Introduction

The seed paper's two algorithms each fix one classifier with the other, in one direction only: Algorithm 1 lets NB judge the training set and delete every instance it misclassifies, then grows a tree on the cleaned set; Algorithm 2 lets a tree decide which attributes matter — weighted by `1/√(minimum depth)` — and plugs those weights into NB's product. The seed never ran them together, and the Sep-13 review verified that no published work does, either. The planned project claims that combination as its contribution, which makes three questions urgent:

- **RQ1 (novelty).** Has anyone since 2014 — especially since 2019 — published a system that chains instance-level noise filtering *and* attribute-level weighting/selection into one DT–NB classifier?
- **RQ2 (interaction evidence).** What does recent work say about (a) order and interaction effects when instance reduction and attribute reduction are applied to the same dataset, and (b) confidence-aware or soft alternatives to hard deletion?
- **RQ3 (defensible evaluation).** What protocol does the methodology literature require for such a claim to survive review, and is it feasible on a laptop?

The report is organized as five branches — model-level successors (§4), instance-level handling (§5), attribute-level handling (§6), coupling the two levels (§7), and evaluation methodology (§8) — followed by a cross-branch synthesis (§9), open problems (§10), and conclusions (§11). The companion note `strategy-mutual-hybrid.md` turns the findings into an implementable plan.

## 2. Methodology

**Sources.** The verified Sep-13 literature review (`../reference/literature-review-dt-nb-hybrid.md`) supplied the pre-2019 canon and the seed's citation network. The Semantic Scholar dump of all 410 papers citing the seed (`../../11th_trimester/bda/tmp/farid2014_citations.json`, fetched 2026-09-13) was re-mined by keyword. Five web-search perspectives then ran (2026-09-17): (1) the DT–NB successor school; (2) label-noise filtering, classical/tabular; (3) attribute weighting for NB; (4) instance×feature selection interaction and simultaneity; (5) evaluation methodology (leakage, significance testing). Candidates were verified against publisher pages, Crossref/DBLP records, PubMed/ACM/JMLR pages, or arXiv abstracts; the three load-bearing claims (the closest 2025 prior, the quantified leakage magnitude, the correction-vs-filtering theorem) were re-verified directly against primary sources during synthesis.

**Disclosed degradations.** Search agents were constrained to run serially (platform concurrency limit), and the Semantic Scholar API rate-limited most programmatic calls, so existence verification leaned on WebSearch and publisher pages. Three items rest on metadata only and are flagged inline. Google Scholar was not directly accessible; CJK-venue and ProQuest-only hits could not be verified and were discarded under the grey-zone rule.

**Corpus.** ~55 new works verified beyond the Sep-13 review (which held ~35), across 2019–2026 plus the pre-2019 anchors each branch requires. Inclusion favored: tabular/classical feasibility, 2019–2026 recency, and direct bearing on RQ1–RQ3. Application papers that merely *use* the hybrids (rainfall, landslide, graduation prediction, etc.) were excluded unless they carry methodological signal.

**Taxonomy rationale.** The two seed algorithms manipulate different parts of the training data — instances (Alg 1) and attributes (Alg 2). The natural axes are therefore *what is manipulated* (instances / attributes / both) × *how the two manipulations are coupled* (not at all / sequentially / simultaneously / mutually). That yields the five branches: each branch answers part of the RQ set, and the empty cells of the coupling axis are themselves findings (§10).

## 3. Taxonomy

| Branch | Manipulates | Coupling with the other level | Answers |
|---|---|---|---|
| §4 DT–NB hybrid successors | model structure | none — one-directional hybrids, ensembles, leaf-level hybrids | RQ1 |
| §5 Instance-level handling | training instances | alone (filters for any downstream model) | RQ2b |
| §6 Attribute-level handling for NB | attributes | alone (weights/selections for NB) | RQ1, RQ2b |
| §7 Coupling the levels | both | sequential, simultaneous, or joint — **not mutual, not DT–NB-specific** | RQ2a |
| §8 Honest evaluation | — | protocol requirements | RQ3 |

The cell **"mutual coupling of instance filtering and attribute weighting inside a DT–NB system"** is empty in every branch — that is the report's central observation.

## 4. DT–NB hybrid successors: where the school actually went (2019–2026)

This branch places the seed in its living lineage. The canonical architecture — a tree whose leaves hold local NB models — is Kohavi's NBTree [5], and the "tree inside NB" alternative is TAN [6]; both predate the seed and were never cited by it. The seed's own group has since pivoted *away* from combining its two algorithms: its 2025 ICCIT paper compares NBTree-style hybrids on ten UCI datasets without revisiting Algorithms 1–2 [9]; its 2023 ensemble work uses the hybrids as base learners under voting [10]; and its newer tree-induction and KNNTree papers generalize the "tree with another learner at the leaves" template to KNN [11], [12] — again, no instance filtering, no attribute weighting for NB.

Two successors matter more. Wong, Yang & Chen (2020) is the branch's methodological pivot: instead of deleting instances NB misclassifies, they keep them and route each test instance to one of three models built from DT and NB — and explicitly benchmark against the seed's hybrid, which they beat on twenty datasets [4]. This is the strongest published statement that **Algorithm 1's deletion premise is its weakest part**. And Changpetch et al. (2021) build a one-way tree→NB pipeline (trees discretize predictors, association rules add interactions, NB consumes both) for medical data [14] — a second instance of "the tree shapes NB's inputs" that stops short of any feedback loop.

Since 2019, nobody in this school composes the two seed algorithms. The nearest structural relative of a *mutual* system is outside the family entirely: black-box-guided tree induction frameworks [unconfirmed: one 2025 *Big Data Mining and Analytics* hit could not be verified past metadata and is excluded]. Within the verified record, the closest published "both knobs in one classifier" works are not DT–NB systems at all — they live in branches §6 and §7.

**Takeaway:** the school that produced the seed redirected its energy to ensembles and leaf-level hybrids; the two one-directional hybrids stand unreconciled.

## 5. Instance-level handling: the shift from hard deletion to confidence-aware handling

Algorithm 1 inherits a lineage: Wilson editing's local-consensus deletion [22], John's robust trees — the same delete-what-the-model-gets-wrong loop, whose own experiments already degraded on error-free domains like Tic-Tac-Toe [23] — and the canonical critique: Brodley & Friedl showed that a *single*-model filter is the weakest and most over-aggressive of the filter designs [3]. Later refinements replaced "one model disagreed" with stronger evidentiary standards: committees and ensembles [24], [25], and leave-one-out partitioning so that only instances no model can learn from the rest are removed [26]. The seed's filter sits at the aggressive extreme of this spectrum; Frénay & Verleysen's survey frames it as one unvalidated member of the cleansing family [27].

The 2019–2026 record pushes the same direction, with three converging results.

**Confidence and graded handling replace binary deletion.** Confident learning reframes noise handling as confidence-based pruning of predicted probabilities plus noise-rate estimation [29]. Samami et al.'s high-agreement filter grades instances by noise strength — removing strong noise, *relabeling* weak noise rather than deleting it [31]. The three-way decision literature formalizes accept/reject/defer handling of uncertain instances, including NB variants [19], and two 2023 works from the seed's own citation network improve NB under mislabeled data by reliability or generative corrections instead of deletion [42], [43]. Most pointedly, the AAAI-2024 analysis proves at dataset level that **correction is more effective than filtering because filtering is prone to over-cleaning** — removing true-labeled instances in excess of actual noise — and that fusing correction with filtering beats either alone [32].

**Cleaning is class-biased.** Under imbalance, selection-based cleaning systematically disadvantages minority classes: a 2024 IEEE TMM study demonstrates the bias and rebalances the selection [36]; the 2025 systematic review catalogs the interaction of label noise with imbalance [37]; and an ICML-2025 result shows asymmetric *added* noise can even rebalance imputed boundaries — deletion-only thinking is conceptually incomplete [38]. A 2024 KBS study compares filter behavior under imbalance directly [35]. This is the modern, citable form of John's 1995 observation that aggressive deletion throws away good data [23].

**Two evaluation cautions for any new filter.** Random label injection overstates filter performance; complexity-based noise injection degrades classical filters [33]. And no single filter is universally best — motivating adaptive or meta-learned filter choice [30], [41]. Meanwhile, robust-loss work shows trees can absorb label noise during growth itself, an alternative to pre-filtering entirely [39], and NB itself is reported as the *most stable* learner under label noise [34] — which supports using NB as a *committee member with confidence outputs*, but not as a sole judge.

**Takeaway:** the field has moved from "delete what one model got wrong" to "grade what a committee is unsure about, reweight or relabel the borderline, delete only the confident noise, and report per-class effects." Algorithm 1 is on the wrong side of that shift; the mutual hybrid should treat confidence-aware handling as the default, not the extension.

## 6. Attribute-level handling for NB: the dormant depth scheme and its active successors

Algorithm 2's mechanism has a precise pre-history — Langley & Sage's selection-only wrapper [45], SBC's top-3-levels tree selection [46], and above all Hall (2007), whose bagged-tree filter uses the *same* `1/√d` minimum-depth weighting the seed uses [2] — and a systematic study in Zaidi et al.'s WANBIA line, which formalizes Hall's filter and argues attribute weighting's real value is compensating for violated independence [48]. The exponent-in-product framework itself is Zhang & Sheng's [47], and the prominent non-tree weight source is Jiang's deep feature weighting [49].

What happened after 2019 is a decisive shift of weight sources. The Jiang group's correlation-based filter (TKDE 2019) [50] spawned fine-tuned gain-ratio weights [52], KL/information-gain weights [51], attribute augmentation [53], and — closest to a tree — **multi-view weighting that uses random trees to build auxiliary label views** for deriving weights (MAWNB, TKDE) [54], extended through multi-source weights [55] to a perturbation-matrix view with a WANBIA-style learned objective (UAI 2026) [56]. All of these are learning-based or MI-based; the *tree-depth* scheme survives only as a cited baseline.

The negative result is load-bearing: **targeted searches found no post-2019 paper that computes NB attribute weights from minimum tree depth.** Nobody has combined tree-derived depth weights with instance filtering either. The only attribute+instance combinations in the record avoid the tree: Zhang et al.'s attribute-and-instance weighted NB [16] and its 2025/26 successor, which dropped the attribute side and pursued instance weighting alone [17]; the value-frequency instance-weighting filter [57]; and ARRAY's triple feature weighting with instance selection in a transfer-NB setting [18]. The 2025 closest-prior — modified partial instance reduction feeding a fine-tuned attribute-weighted NB — combines both knobs but contains **no decision tree**, and its evidence is a single gaming-disorder dataset with ~1.3–1.4% gains [15].

**Takeaway:** two implications cut in opposite directions. Novelty of the Alg-2 pathway is *protected* (dormant since 2019, pre-dated by Hall [2] — any writeup must cite Hall as closest prior). But novelty alone is weaker if the mechanism is stale; the mutual hybrid's defense is that the tree's weights are computed **on data the NB filter has just cleaned**, a condition no weighting study has ever tested.

## 7. Coupling the two levels: simultaneity, order, and over-cleaning

This branch answers RQ2a and contains the strongest evidence that the mutual design's premise — the two operations interact — is real.

The data-reduction school states the interaction outright: instance and feature selection are "interwoven" problems whose separate treatment loses mutual benefits [62], and a line of simultaneous-selection algorithms exists precisely because of it — memetic joint selection [62], fast simultaneous selection for high-dimensional data [63], deterministic simultaneous selection aimed at noise [64], fused data-reduction algorithms [65], and a reinforcement-learning framing that treats the two selections as mutually coupled decisions [66]. Crucially, these are model-agnostic dataset reducers: none ties the joint selection to a DT–NB classifier, and none reports class-wise effects (only aggregate accuracy/storage).

The order question has direct, if scattered, evidence. Varying the relative priority of feature vs instance selection inside one GA changes outcomes [60]. Adding a feature-selection stage *improves* the instance-selection process itself [61]. The sharpest order effect comes from the adjacent resampling-filtering pipeline: SMOTE-IPF deliberately reverses the natural order — oversample first, filter after — because **filtering before resampling strips valuable borderline minority instances** [68]. Class noise is more harmful than attribute noise, so the two operations do not act symmetrically [58]. And modern practice simply sequences the operations (GP feature selection, then instance selection) without asking whether the order is right [67]; the earliest explicit joint treatment dates to Fragoudis et al. [59].

The over-cleaning result of branch §5 belongs here too: filtering removes clean instances in excess of true noise [32], and cleaning interacts with imbalance [35]–[38].

**Takeaway:** the interaction is asserted, exploited by model-agnostic reducers, and shown to have order effects in adjacent pipelines — but in the retrieved literature no controlled study of *noise-filter-first vs attribute-selection-first vs simultaneous* exists for a classifier family like DT–NB (the closest retrieved work is the GA priority study [60], the one-way improvement [61], and the resampling-order effect [68]). The order ablation is unclaimed by anything we could retrieve, cheap, and directly serves the mutual design's justification.

## 8. Honest evaluation: what the methodology school now demands

Fitting a filter or selector is learning; doing it outside the evaluation loop leaks. The canonical statements: tuning/selection on the same data as evaluation is optimistically biased, with effects comparable to the differences being measured [70]; using CV error for a CV-tuned model is substantially optimistic [69]; the modern reproducibility audit documents leakage across 294+ studies in 17 fields, with preprocessing fit on train+test jointly as a named leakage type [71].

Magnitude, specifically for filters: in radiomics, feature selection applied *before* CV rather than inside each fold inflates AUC-ROC by up to **0.15**, AUC-F1 by 0.29, and accuracy by 0.17, with bias growing as samples-per-feature falls [72]. Adjacent evidence: preprocessing leakage inflates apparent reliability of intrusion-detection models on NSL-KDD/UNSW-NB15 [73] — relevant because NSL-KDD is one of the seed's ten datasets — and oversampling before CV similarly overestimates performance [unconfirmed: Sci Rep 2024 follow-up seen in search snippets only].

One caution cuts the other way. A 2026 preprint measuring leakage across 2,047 tabular benchmarks finds *scaling-type* leakage negligible on ordinary low-dimensional data while *selection-type* leakage (peeking, seed cherry-picking) remains large [74]. The implication for the project: **the seed's inflation must be measured, not assumed** — on ten low-dimensional UCI datasets the leakage audit may show a modest gap, which is still a publishable honest-protocol contribution, but the paper's claims must be calibrated to whatever is measured.

For multi-dataset significance testing: Friedman/Nemenyi [76], the more powerful all-pairwise successors (Iman–Davenport, Holm, Shaffer) [77], the Bayesian alternative reporting probabilities of superiority [78], and — essential for repeated 10×10 CV — the corrected resampled t-test, since fold errors are dependent and naive t-tests have absurdly high Type-I error [75].

**Takeaway:** RQ3's answer is a package: refit every filter/selector inside every fold, repeat seeds, use corrected or Bayesian comparisons — and treat the measured leakage gap as a result with honest uncertainty, not a headline number.

## 9. Synthesis

Cross-branch, four tensions define the design space.

**T1 — The stable learner should not be the sole judge.** Branch §5 reports NB as the most stable learner under label noise [34], yet Algorithm 1 uses exactly one NB as the sole deletion judge — the design Brodley & Friedl ranked weakest [3]. The resolution is mechanical: NB's *probabilities* are the asset; put them in a committee (consensus/majority with tree- and instance-based members [25]) and threshold on confidence [29] rather than on a misclassification flag.

**T2 — "Don't throw away" converges from three branches.** Wong's routing keeps misclassified instances available and beats the seed [4]; AAAI-2024 proves filtering over-cleans relative to correction [32]; the imbalance studies show deletion biases minority classes [36]–[38]. A mutual hybrid should therefore implement deletion, reweighting, *and* relabeling as arms of the same experiment — not hard-code deletion.

**T3 — Interaction is asserted by one school, ignored by the other.** The data-reduction school builds simultaneous methods because the selections are interwoven [62]–[66]; the DT–NB school keeps the levels separate and unreconciled (§4). The mutual hybrid is exactly the bridge, and the order ablation (§7) is the missing controlled experiment that would justify it. The seed's own order is filter→weight; SMOTE-IPF's lesson [68] says order effects should be expected, not assumed away.

**T4 — Leakage is real, quantified elsewhere, and unmeasured here.** The +0.15-AUC radiomics bias [72] came from small-sample, high-dimensional data; the seed's ten datasets are low-dimensional [74]. The honest audit is therefore itself a contribution with a genuinely uncertain outcome — and NSL-KDD, the one larger dataset in the benchmark, is precisely where preprocessing leakage has been shown to matter before [73].

## 10. Open problems

1. **The mutual cell is empty in the retrieved record.** No verified work retrieved chains NB-based filtering → tree on cleaned data → tree-derived weights → weighted NB, iteratively or in one pass, with per-fold refitting. Closest published artifacts each miss one component: Latubessy 2025 (no tree) [15], Zhang 2021/2026 (no tree; the 2026 successor is instance-side) [16], [17], ARRAY 2023 (transfer NB, no tree) [18], Wong 2020 (no weighting) [4].
2. **No controlled order study was retrieved** for noise filtering × attribute selection inside classifier training — only priority effects inside a GA [60], a one-way improvement result [61], and an order effect in the adjacent resampling pipeline [68]. (Retrieval-bounded: ROFS [64] is simultaneous-by-design and its full text was not accessible to check whether it also compares orderings.)
3. **"Does cleaning change attribute relevance?" is untested.** Every weighting study (§6) derives weights from raw training data; no work derives tree-depth weights from *filtered* data — the mutual premise itself.
4. **Class-wise effects of joint reduction are unreported.** Simultaneous-selection papers report aggregates only (§7), while the imbalance evidence says class-wise is where the harm shows [36]–[38].
5. **Tree-depth weighting is dormant, not dead.** No post-2019 successor exists; whether the depth scheme on modern cleaned pipelines is competitive with MAWNB-style views [54] or learned weights [56] is open — and answering it only requires sklearn, not deep learning.

## 11. Conclusion

**RQ1 — Novelty:** no verified published system combines the seed's two hybrids into one mutually-reinforcing, DT–NB-specific classifier. The closest priors are Latubessy et al. (2025) [15], Zhang et al. (2021, 2026) [16], [17], and ARRAY (2023) [18] on the "both knobs" side, and Wong et al. (2020) [4] on the instance-handling side; Hall (2007) [2] remains the unavoidable prior for the weighting mechanism itself. The mutual lane is open, but every writeup must cite [2] and [15]–[18] and position against them.

**RQ2 — Interaction and soft handling:** the interaction is real and citable — interwoven selections [62], priority effects [60], order effects under imbalance [68], over-cleaning by filtering [32], class-biased cleaning [36]–[38]. Confidence-aware handling (committee + thresholds + reweighting/relabeling) is the evidence-backed default over hard NB-judge deletion; deletion should be one arm among several.

**RQ3 — Evaluation:** per-fold refitting of every filter/selector, repeated seeds, corrected resampled t-tests or Bayesian comparisons [70], [75], [78], with the leakage gap *measured* on the ten datasets rather than assumed from high-dimensional precedents [72], [74]. Everything required is CPU-scale sklearn on datasets ≤ 25k rows — feasible on a laptop, mapped concretely in `strategy-mutual-hybrid.md`.

**The survey's own contribution** is the mapping itself: five branches, one empty cell (mutual DT–NB coupling), one dormant mechanism (tree-depth weighting) whose revival is only defensible under the mutual premise, one unclaimed controlled experiment (order ablation), and one calibrated evaluation package — collectively, the evidence base for the project's four claimed contributions (C1–C4).

## References

[1] D. M. Farid, L. Zhang, C. M. Rahman, M. A. Hossain, R. Strachan, "Hybrid decision tree and naïve Bayes classifiers for multi-class classification tasks," Expert Systems with Applications, 2014.
[2] M. Hall, "A decision tree-based attribute weighting filter for Naive Bayes," Knowledge-Based Systems, 2007.
[3] C. E. Brodley, M. A. Friedl, "Identifying mislabeled training data," Journal of Artificial Intelligence Research, 1999.
[4] T.-T. Wong, N.-Y. Yang, G.-H. Chen, "Hybrid classification algorithms based on instance filtering," Information Sciences, 2020.
[5] R. Kohavi, "Scaling up the accuracy of naive-Bayes classifiers: a decision-tree hybrid," KDD, 1996.
[6] N. Friedman, D. Geiger, M. Goldszmidt, "Bayesian network classifiers," Machine Learning, 1997.
[7] L.-M. Wang, X.-L. Li, C.-H. Cao, S.-M. Yuan, "Combining decision tree and Naive Bayes for classification," Knowledge-Based Systems, 2006.
[8] M. Hall, E. Frank, "Combining naive Bayes and decision tables," FLAIRS, 2008.
[9] A. Hassan, D. M. Farid, "Naïve Bayesian tree for multi-class classification tasks," ICCIT, 2025.
[10] N. Sourov, F. H. Chowdhury, A. S. M. Redowan, et al., "Ensemble machine learning for multi-class classification tasks," ICCIT, 2023.
[11] S. Saurav, M. A. Mitu, N. A. Ritu, et al., "A new method for learning decision tree classifier," ECCE, 2023.
[12] M. M. Islam, Fatema-Tuj-Jahra, M. K. Hasan, et al., "KNNTree: a new method to ameliorate k-nearest neighbour classification using decision tree," ECCE, 2023.
[13] M. A. Mitu, S. Arefin, S. Saurav, "Pruning-based ensemble tree for multi-class classification," ICEEICT, 2024.
[14] F. Changpetch, P. Pitpeng, S. Hiriote, et al., "Integrating data mining techniques for Naive Bayes classification: applications to medical datasets," Computation, 2021.
[15] A. Latubessy, E. Winarko, A. Musdholifah, U. Kusrohmaniah, "Robust feature management for gaming disorder classification using modified partial instance reduction and fine-tune attribute-weighted naive Bayes," Journal of Artificial Intelligence and Technology, 2025.
[16] H. Zhang, L. Jiang, L. Yu, "Attribute and instance weighted naive Bayes," Pattern Recognition, 2021.
[17] H. Zhang, K. Meng, P. Lv, S. He, M. Xu, "A general dual-view framework for instance weighted naive Bayes," Pattern Recognition, 2026 (online 2025).
[18] H. Tong, et al., "ARRAY: adaptive triple feature-weighted transfer naive Bayes for cross-project defect prediction," Journal of Systems and Software, 2023.
[19] Z. Yang, J. Ren, Z. Zhang, et al., "A new three-way incremental naive Bayes classifier," Electronics, 2023.
[20] L. Yu, S. Gan, Y. Chen, et al., "A novel hybrid approach: instance weighted hidden naive Bayes," Mathematics, 2021.
[21] A. Ahammed, B. Harangi, A. Hajdu, "Hybrid AdaBoost and naive Bayes classifier for supervised learning," CEUR Workshop Proceedings, 2021.
[22] D. L. Wilson, "Asymptotic properties of nearest neighbor rules using edited data," IEEE Transactions on Systems, Man, and Cybernetics, 1972.
[23] G. H. John, "Robust decision trees: removing outliers from databases," KDD, 1995.
[24] D. Gamberger, N. Lavrač, S. Džeroski, "Noise elimination in inductive concept learning: a case study in medical diagnosis," ALT, 1996.
[25] S. Verbaeten, A. Van Assche, "Ensemble methods for noise elimination in classification problems," Multiple Classifier Systems, 2003.
[26] M. R. Smith, T. Martinez, "Improving classification accuracy by identifying and removing instances that should be misclassified," IJCNN, 2011.
[27] B. Frénay, M. Verleysen, "Classification in the presence of label noise: a survey," IEEE Transactions on Neural Networks and Learning Systems, 2014.
[28] I. H. Sarker, M. A. Kabir, A. Colman, J. Han, "An improved naive Bayes classifier-based noise detection technique for classifying user phone call behavior," AusDM, 2017.
[29] C. G. Northcutt, L. Jiang, I. L. Chuang, "Confident learning: estimating uncertainty in dataset labels," Journal of Artificial Intelligence Research, 2021.
[30] B. Zerhari, A. Ait Lahcen, S. Mouline, "MIPCNF: multi-iterative partitioning class noise filter," Journal of Intelligent & Fuzzy Systems, 2019.
[31] M. Samami, A. Akbari, M. Abdar, et al., "A mixed solution-based high agreement filtering method for class noise detection in binary classification," Physica A, 2020.
[32] G. Jiang, et al., "Which is more effective in label noise cleaning, correction or filtering?," AAAI, 2024.
[33] L. P. F. Garcia, J. Lehmann, A. C. P. L. F. de Carvalho, A. C. Lorena, "New label noise injection methods for the evaluation of noise filters," Knowledge-Based Systems, 2019.
[34] J. M. Johnson, T. M. Khoshgoftaar, "A survey on classifying big data with label noise," ACM Computing Surveys, 2022.
[35] S. Szeghalmy, A. Fazekas, "A comparative study on noise filtering of imbalanced data sets," Knowledge-Based Systems, 2024.
[36] Z. Liu, V. S. Sheng, T. Sun, et al., "Learning with imbalanced noisy data by preventing bias in sample selection," IEEE Transactions on Multimedia, 2024.
[37] F. H. Brishti, J. Zhang, N. Mohammed, et al., "Imbalanced classification with label noise: a systematic review and comparative analysis," ICT Express, 2025.
[38] G. Hu, F. Liu, M. Gong, et al., "Learning imbalanced data with beneficial label noise," ICML, 2025.
[39] L. Wilton, G. Ye, "Robust loss functions for training decision trees with noisy labels," AAAI, 2024.
[40] R. Gomes Mantovani, T. Horváth, A. Rossi, et al., "Better trees: an empirical study on hyperparameter tuning of classification decision tree induction algorithms," Data Mining and Knowledge Discovery, 2024.
[41] G. Pio, A. Rivolli, A. C. P. L. F. de Carvalho, L. P. F. Garcia, "Two meta-learning approaches for noise filter algorithm recommendation," Journal of Information and Data Management, 2024.
[42] Q. Zeng, Y. Zhu, X. Zhu, et al., "Improved naive Bayes with mislabeled data," Statistics and Its Interface, 2024.
[43] Y. Zhu, Y. Wang, L. Qin, B. Zhang, "Naive Bayes classifier based on reliability measurement for datasets with noisy labels," Annals of Operations Research, 2023.
[44] M. Delli Veneri, S. Cavuoti, R. Abbruzzese, M. Brescia, "HyCASTLE: a hybrid classification system based on typicality, labels and entropy," Knowledge-Based Systems, 2022.
[45] P. Langley, S. Sage, "Induction of selective Bayesian classifiers," UAI, 1994.
[46] C. A. Ratanamahatana, D. Gunopulos, "Feature selection for the naive Bayesian classifier using decision trees," Applied Artificial Intelligence, 2003.
[47] H. Zhang, S. Sheng, "Learning weighted naive Bayes with accurate ranking," ICDM, 2004.
[48] N. A. Zaidi, J. Cerquides, M. J. Carman, G. I. Webb, "Alleviating naive Bayes attribute independence assumption by attribute weighting," JMLR, 2013.
[49] L. Jiang, C. Li, S. Wang, L. Zhang, "Deep feature weighting for naive Bayes and its application to text classification," Engineering Applications of Artificial Intelligence, 2016.
[50] L. Jiang, L. Zhang, C. Li, J. Wu, "A correlation-based feature weighting filter for naive Bayes," IEEE TKDE, 2019.
[51] M. Kalra, V. Kumar, M. Kaur, S. Ahmed Idris, Ş. Öztürk, "Attribute weighted naive Bayes classifier," Computers, Materials & Continua, 2022.
[52] H. Zhang, L. Jiang, "Fine tuning attribute weighted naive Bayes," Neurocomputing, 2022.
[53] H. Zhang, L. Jiang, et al., "Attribute augmented and weighted naive Bayes," Science China Information Sciences, 2022.
[54] H. Zhang, L. Jiang, W. Zhang, et al., "Multi-view attribute weighted naive Bayes," IEEE TKDE, 2023.
[55] G.-L. Ou, Y. He, P. Fournier-Viger, et al., "A novel multi-source weighted naive Bayes classifier," Information Sciences, 2025.
[56] S. Wu, H. Zhang, K. Meng, et al., "Learning representations from perturbation: a novel matrix-view weighting framework for naive Bayes," UAI, PMLR, 2026.
[57] W. Xu, L. Jiang, L. Yu, "An attribute value frequency-based instance weighting filter for naive Bayes," Journal of Experimental & Theoretical Artificial Intelligence, 2019.
[58] X. Zhu, X. Wu, "Class noise vs. attribute noise: a quantitative study," Artificial Intelligence Review, 2004.
[59] D. Fragoudis, G. Tsoumakas, I. Vlahavas, "Integrating feature and instance selection for text classification," IJCAI, 2005.
[60] C.-F. Tsai, W. Eberle, C.-Y. Chu, "Genetic algorithms in feature and instance selection," Knowledge-Based Systems, 2013.
[61] J. Derrac, I. Triguero, S. García, F. Herrera, "Enhancing evolutionary instance selection algorithms by means of fuzzy rough set based feature selection," Information Sciences, 2012.
[62] N. García-Pedrajas, A. de Haro-García, J. Pérez-Rodríguez, "A scalable memetic algorithm for simultaneous instance and feature selection," Evolutionary Computation, 2014.
[63] N. García-Pedrajas, J. A. Romero del Castillo, G. Cerruela-García, "SI(FS)2: fast simultaneous instance and feature selection for datasets with many features," Pattern Recognition, 2021.
[64] Y. Villuendas-Rey, C. C. Tusell-Rey, O. Camacho-Nieto, "Simultaneous instance and attribute selection for noise filtering," Applied Sciences, 2024.
[65] M. Kusy, R. Zajdel, "New data reduction algorithms based on the fusion of instance and feature selection," Knowledge-Based Systems, 2024.
[66] W. Fan, K. Liu, H. Liu, et al., "Feature and instance joint selection: a reinforcement learning perspective," arXiv, 2022.
[67] G. Feng, G. F. A. Yeo, I. Hudson, et al., "Sequential feature selection and instance selection using SpFSR and SpFixedIS," ICAIBD, 2023.
[68] J. A. Sáez, J. Luengo, J. Stefanowski, F. Herrera, "SMOTE-IPF: addressing the noisy and borderline examples problem in imbalanced classification by a re-sampling method with filtering," Information Sciences, 2015.
[69] S. Varma, R. Simon, "Bias in error estimation when using cross-validation for model selection," BMC Bioinformatics, 2006.
[70] G. C. Cawley, N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," JMLR, 2010.
[71] S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, 2023.
[72] A. Demircioğlu, "Measuring the bias of incorrect application of feature selection when using cross-validation in radiomics," Insights into Imaging, 2021.
[73] M. A. Bouke, A. Abdullah, "An empirical study of pattern leakage impact during data preprocessing on machine learning-based intrusion detection models reliability," Expert Systems with Applications, 2023.
[74] S. Roth, "Which leakage types matter? A quantitative landscape across 2,047 benchmark datasets," arXiv, 2026 (preprint).
[75] C. Nadeau, Y. Bengio, "Inference for the generalization error," Machine Learning, 2003.
[76] J. Demšar, "Statistical comparisons of classifiers over multiple data sets," JMLR, 2006.
[77] S. García, F. Herrera, "An extension on 'statistical comparisons of classifiers over multiple data sets' for all pairwise comparisons," JMLR, 2008.
[78] A. Benavoli, G. Corani, J. Demšar, M. Zaffalon, "Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis," JMLR, 2017.

---

## Verification appendix (not part of the survey proper)

- **Audit record: `verification-ledger-2026-09-17.md`** in this folder — per-source status, the full correction list, and the seed-paper extraction record. Read it before citing anything here.
- All references in the list passed at least metadata verification against Crossref, arXiv, PubMed, publisher pages, or direct PDF extraction, except ROFS [64] (metadata-only; full text HTTP 403) and Pio et al. [41] (page verified; no Crossref DOI) — claims for these are restricted to title/abstract level. Roth [74] is a non-peer-reviewed preprint.
- **Corrections applied after the retrieval pass** (details in the ledger, §A): [16] corrected to Zhang, Jiang & Yu (the Sep-13 review's DOI `…107725` does not exist; correct is `…107674`); [29] corrected to Northcutt, Jiang & Chuang; [51] corrected to Kalra et al. (previously misattributed as "Foo et al."); [42] corrected to 2024; [71] Kapoor DOI corrected to `10.1016/j.patter.2023.100804` (the earlier `…100794` resolves to a different paper); [75] correct title "Inference for the generalization error"; [26] note the 2016 AIIR follow-up (Smith & Martinez) verified as the majority-voting robustness paper; several author lists corrected ([19], [20], [21], [38], [40], [44], [56], [57], [62], [63], [65]).
- Directly re-verified at source: [15] (publisher page — no tree; mPIR + FTAWNB; gaming-disorder only; +1.28%/+1.4% — do not generalize), [32] (Crossref + AAAI metadata: Theorem 5 correction vs filtering, over-cleaning caveat), [72] (PubMed-confirmed: up to 0.15 AUC-ROC / 0.29 AUC-F1 / 0.17 accuracy bias), and the seed PDF itself (algorithms extracted verbatim; **the paper never states whether filtering/selection runs inside or outside the CV folds** — the leaky protocol is underspecified, not stated, and C1's two-variant audit tests exactly that).
- Deliberately excluded under the grey-zone rule: a 2022 ProQuest-only attribute-weighted NB hit; a 2025 Big Data Mining and Analytics tree-guidance framework (metadata insufficient); "dashed technique" terminology (no indexed source); a Sci Rep 2024 oversampling-leakage follow-up (snippet only).
- Method degradations: search agents ran serially (platform concurrency limit); Semantic Scholar API and DBLP were unreachable during the audit (verification re-routed through Crossref, arXiv API, PubMed, publisher pages, and PDF extraction); one verification sub-agent failed on rate limits and its task was absorbed manually; ROFS full text inaccessible (403).
