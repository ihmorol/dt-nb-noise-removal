> HISTORICAL CLINE NOTE � superseded by `e12v2-protocol.md`. The original relabel pilot deleted flagged rows; its AUROC used the wrong score. Claims of full Confident Learning equivalence, certified clean UCI labels, significance implying equivalence, or a universal 51% precision ceiling are unsupported and withdrawn. Retained below for provenance, not as current conclusions.

# E12 — noise removal: the method, and the literature behind it

*2026-10-02. Question: (a) what is the most reliable way to reduce label noise using a **Decision Tree**, and separately using **Naive Bayes**; (b) what published work removes noise from data at all. Every citation below was checked against the Crossref or arXiv API during this session — nothing is quoted from memory.*

## Why E11 could not answer this

E11 compared cleaning arms only through downstream accuracy, on ten datasets that are themselves clean. So when the committee deleted 7.4 % of `diabetes`, there was no way to know whether it had removed 7.4 % *wrong labels* or just 7.4 % *hard boundary cases*. Every arm delta fell inside `p ≥ 0.0625` — not because the filters are equivalent, but because **the experiment had no ground truth to rank them by**.

E12 removes that gap: inject a controlled fraction of label noise, keep the corruption mask, and score each filter on precision/recall. Code: `code/noise_inject.py`, `code/confident_filter.py`, `code/e12_noise_removal.py`.

## The five elements of a reliable filter, and what we implemented

| # | Element | Why it matters | Where |
|---|---|---|---|
| 1 | **Out-of-fold predictions** | A judge grading rows it *trained* on conflates "I find this hard" with "this label is wrong". E3a showed what skipping this costs. | already correct in `committee_filter.py`; kept in `out_of_fold_probabilities` |
| 2 | **Per-class thresholds (confident joint)** | The fix for E11's NSL-KDD failure, where a 6/6 vote deleted 100 % of `ftp_write`, `spy` and `land` rows and cost 7.1 macro-F1 points. | **new** — `confident_joint()` |
| 3 | **Soft scores, not hard votes** | Thresholding needs a confidence; a vote throws that information away. | **new** — `WeightedNB.predict_proba`, verified equal to `GaussianNB.predict_proba` to 9e-16 on iris |
| 4 | **Soft handling (relabel, not only delete)** | CL's third principle is ranking examples to train with confidence. | **new** — `--handling relabel` |
| 5 | **Per-class noise-rate estimate** | Turns an unscoreable heuristic into a reported number. | **new** — `info["noise_rate_estimate"]` |

Plus a **per-class floor** (`min_per_class`): the guard that stops any rule, however precise, from deleting a class outright. Verified on `contact-lenses` (a class with 4 rows): with the floor off, hard voting left class `'1'` with **0** rows; with the floor at 5, every class keeps ≥ 2. One bug was found and fixed here — the first implementation tested `survivors + flagged < min_per_class`, which still left a class with **1** row when the floor was 5.

## DT as the judge, and NB as the judge — separately

The two were run **separately**, never fused, so their contributions are attributable.

- **DT judge** — out-of-fold `predict_proba` from the entropy tree → self-confidence → confident joint. An unpruned tree is high-variance and overconfident on small leaves; on iris its confident-joint thresholds **never bite**, so its numbers come out *identical* to hard voting. That is a genuine finding, not a bug: a tree that separates the classes perfectly has nothing to threshold. Its value here is as a committee member, not as the sole judge.
- **NB judge** — `WeightedNB.class_scores` → softmax posterior → confident joint. NB is **stable** under label noise (Johnson & Khoshgoftaar 2022), which makes it a good judge; but its error rate is dominated by class imbalance, so it deletes minority classes first and hardest. Per-class thresholds are not optional here.
- **Result (iris, 20 % asymmetric noise):** NB confident joint **precision 0.822 / F1 0.801** against NB hard voting **0.666 / 0.755** — and it removed far less (19.3 % vs 30.0 %).
## Published work that removes noise from data

### Verified this session (Crossref / arXiv API)

| Work | Citation |
|---|---|
| Brodley & Friedl 1999 | *Identifying Mislabeled Training Data*, JAIR **11**:131–167, DOI `10.1613/jair.606` |
| Verbaeten & Van Assche 2003 | *Ensemble Methods for Noise Elimination in Classification Problems*, LNCS/MCS 317–325, DOI `10.1007/3-540-44938-8_32` |
| Khoshgoftaar & Rebours 2005 | *Evaluating noise elimination techniques for software quality estimation*, Intell. Data Analysis **9**:487–508, DOI `10.3233/IDA-2005-9506` |
| Smith, Martinez & Giraud-Carrier 2013 | *An instance level analysis of data complexity*, Machine Learning **95**:225–256, DOI `10.1007/s10994-013-5422-z` |
| **Northcutt, Jiang & Chuang 2021** | ***Confident Learning: Estimating Uncertainty in Dataset Labels*, JAIR, arXiv:`1911.00068`** — the method E12 adopted |
| Northcutt, Athalye & Mueller 2021 | *Pervasive Label Errors in Test Sets Destabilize ML Benchmarks*, NeurIPS D&B, arXiv:`2103.14749` |
| Song, Kim, Park, Shin & Lee 2023 | *Learning From Noisy Labels With DNNs: A Survey*, IEEE TNNLS **34**:8135–8153, DOI `10.1109/TNNLS.2022.3152527` |
| Algan & Ulusoy 2021 | *Image Classification with DL in the Presence of Noisy Labels: A Survey*, KBS, DOI `10.1016/j.knosys.2021.106771` |
| Li, De-Arteaga & Saar-Tsechansky 2025 | *Bias-Aware Mislabeling Detection via Decoupled Confident Learning* (**DeCoLe**), arXiv:`2507.07216` |
| Smith & Martinez 2016 | *The robustness of majority voting compared to filtering misclassified instances*, AI Review **49**:105–130, DOI `10.1007/s10462-016-9518-2` |
| Salekshahrezaee, Leevy & Khoshgoftaar 2021 | *A reconstruction error-based framework for label noise detection*, J. Big Data **8**, DOI `10.1186/s40537-021-00447-5` |
| Zha et al. 2023 | *Data-centric AI: Perspectives and Challenges*, SDM 2023, arXiv:`2301.04819` |
### The ceiling nobody designing a filter should quote without it

Northcutt, Athalye & Mueller (2021) human-validated algorithmically-flagged candidates across 10 datasets: **only 51 % of flagged rows were genuinely mislabeled** (mean ≥ 3.3 % true error rate). Model-based noise filtering is therefore roughly **half-precise**. "Delete" and "relabel" are very different bets on the same evidence, and no threshold rule will lift a 51 % ceiling — it is the information content of the data, not the rule.

*Verified against the paper's own text*, not only its abstract: "Putative label errors are identified using confident learning algorithms and then human-validated via crowdsourcing (51 % of the algorithmically-flagged candidates are indeed erroneously labeled, on average across the datasets)." Full text preserved at `reference/papers/northcutt2021_pervasive-label-errors.pdf`.

### Our implementation checked against the paper's definition

The full text of *Confident Learning* (`reference/papers/northcutt2021_confident-learning.pdf`) defines the threshold as the average self-confidence of the examples **given** label k:

> "a class-specific probability threshold `t_k = (1/|D_ỹ=k|) Σ p̂_{i,k}` is computed for each class k. Samples whose predicted probability exceeds the threshold for some class are assigned a confident predicted label, populating a K×K count matrix whose **off-diagonal entries count estimated label errors**."

That is exactly `confident_joint()` in `confident_filter.py` — same statistic, same off-diagonal test. The paper's own justification for per-class thresholds is also the reason E12 expects them to beat a global rule:

> "These thresholds allow us to guess y* in spite of class-imbalance, unlike prior art which may guess over-confident classes for y* because arg max is used."

### Corrections made to this repo's citations

1. Johnson & Khoshgoftaar (2022) is **JDIQ 14(1)**, DOI `10.1145/3492546` — **not** *ACM Computing Surveys*. Was wrong in `committee_filter.py` and `notes/e11-lr-mlp-hybrid-data.md`.
2. The "IPF, JCST 22(3):387–396" attribution for Khoshgoftaar & Rebours **could not be verified**; Crossref returns their IDA 9:487–508 (2005) paper. Retracted pending a copy of the source.
3. `https://www.jair.org/index.php/jair/article/view/10332` is the **wrong article ID** (it is a 2003 NLG paper). Use the DOI.

### Could NOT verify this session — flagged, not used

- Wilson & Martinez 2000 JMLR (JMLR is poorly indexed in Crossref; jmlr.org fetch failed twice).
- Kubat & Holte, *Misclassified Instances and Decision Trees* (not returned by Crossref).
- Frénay & Verleysen 2014 (cited in the existing notes; not re-checked).

None of these three is load-bearing for E12's conclusions.

## Reproducing

```
python noise_inject.py                             corruption rates per kind
python confident_filter.py                         self-check on a noised iris
python e12_noise_removal.py                        all datasets, 3 seeds
python e12_noise_removal.py --quick                no MLP, 1 seed (smoke run)
python e12_noise_removal.py --handling relabel     relabel instead of only delete
```

Writes `../results/EXP-E12_noise_removal/`: `detection.csv` (per fold: precision,
recall, F1, AUROC, removal %, rows rescued by the floor), `accuracy.csv` (downstream
accuracy and macro-F1 for NB/DT/LR/MLP), `detection_summary.csv`,
`accuracy_summary.csv`, `summary.md`, `config.json`.

**Honesty rules enforced in `e12_noise_removal.py`:** noise is injected into the
**training fold only** (the test fold keeps its clean labels, so no ground truth
leaks across); the filter is fitted inside the training fold on the noised labels
only; the confusable-class map for asymmetric noise is estimated by cross-validated
NB inside the same training fold; test folds never lose rows.

**DeCoLe (2025)** is aimed squarely at our bug: it decouples the confident joint to handle label bias — the minority-class wipeout E11 hit on NSL-KDD.