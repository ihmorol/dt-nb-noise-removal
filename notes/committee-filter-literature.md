# Making the NB noise judge reliable — the literature, and what we built

*2026-10-02, for E11. The question: Farid (2014) Algorithm 1 deletes every training row its single NB judge misclassifies. Farid's own feedback (and the literature) says that is not reliable. What do the papers say makes a filter reliable, and what did we implement?*

## Why a single NB judge is unreliable

1. **It grades rows it was trained on.** Algorithm 1 fits NB on the training data and then classifies that same data. A misclassification there can be model bias, a class overlap, or a genuinely hard region — not a wrong label. The judge conflates "I find this row hard" with "this row is wrong".
2. **A single learner's errors are systematic.** Brodley & Friedl (1999, §1–2) make this the central point: one algorithm deletes the rows *it* finds difficult, so the "noise" it reports is partly just its own inductive bias. In their experiments single-algorithm filters were the weakest design; committees of different algorithms were markedly better.
3. **NB specifically is too eager.** Its independence assumption makes it misclassify whole overlap regions, so Algorithm 1's deletion rate tracks the judge's error rate — our E9p pilot measured exactly that (glass: NB error ≈ 46 %, 47 % of rows deleted, tree accuracy 68 → 51). Johnson & Khoshgoftaar (2022) survey the label-noise-in-big-data literature and conclude NB is the most *stable* learner under noise — which makes it a good committee member, not a sole judge.

## What the literature says makes a filter reliable

| Mechanism | Source | What it changes |
|---|---|---|
| Cross-validated committee of *different* learners | Brodley & Friedl 1999 (JAIR 11:131–167, DOI 10.1613/jair.606) | No judge ever scores a row it trained on; heterogeneous learners cancel each other's biases |
| Consensus vs majority voting | Brodley & Friedl 1999, §3 | Consensus (delete only when *every* judge disagrees) = most conservative, highest precision; majority = higher recall |
| Iterate the partitioned voting | Khoshgoftaar & Rebours 2007 (JCST 22(3):387–396) — the Iterative-Partitioning Filter (IPF) | A single random partitioning can fluke a vote; repeating the k-fold vote and summing votes averages the luck out |
| Ensembles of filters | Verbaeten & Van Assche 2003 (MCS) | Same direction: multiple filter views instead of one |
| Framing survey | Frénay & Verleysen 2014 (IEEE TNNLS 25(5):845–869) | Label-noise robustness is a data problem; filter design is a separate axis from classifier choice |

## What we implemented (committee_filter.py)

A **cross-validated DT+NB committee with repeated voting and a consensus rule**:

- 3 repeats × 3-fold stratified splits inside the training fold;
- each split fits two judges on the other folds — an entropy decision tree and the pipeline's mixed-likelihood NB — and both predict the held-out fold, so **no judge ever scores a row it trained on**;
- every row collects up to 6 votes (2 judges × 3 repeats);
- **consensus rule (default): a row is noise only with 6/6 votes** — both judges misclassified it in every repeat. Majority (≥ 4/6) is implemented as the sensitivity variant;
- safety rule: never delete a whole class (same rule as Algorithm 1's implementation);
- diagnostics recorded: per-row votes, removal %, per-class removal rates.

This is exactly the upgrade Farid's critique demands: NB's misclassification is no longer *sufficient* to delete a row — it must be confirmed by an independent learner that never saw the row, across repeated random splits. Compared with Algorithm 1, the filter trades recall (it deletes fewer rows) for precision (what it deletes is far more likely to be actual noise).

## Relation to the experiment registry

This is the E5a design (consensus/majority committee vs single-NB deletion), here carried by LR + a deep MLP as the downstream classifiers instead of NB/DT, combined with Alg 2's attribute selection and weighting (E10's W1-style weighting). If E11 shows the combination lifts LR/MLP, E5a/E5b/E5c still have value as the NB/DT-native variants.
