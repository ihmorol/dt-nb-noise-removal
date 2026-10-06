> HISTORICAL E11 NOTE: retained with its original results. Statements below calling the filter safe/reliable, diagnosing native label noise, or declaring the noise-removal target achieved are not established by this experiment and are withdrawn. Labels were not independently verified, rare classes were wiped out, and MLP losses occurred. Non-significant tests do not prove equivalence. Treat mechanistic explanations as hypotheses; use `findings.md` for the current bounded interpretation.

# E11 — reliable (committee) noise removal + attribute weighting for LR and a deep MLP

*2026-10-02. Branch `feat/committee-filter-lr-mlp`. Question: does reliable DT+NB noise removal plus Alg 2 attribute selection/weighting improve Logistic Regression and a fundamental deep model (MLP) over the same models on the raw data? One run, 10 datasets, 10-fold x 5 seeds, refit per fold, 37 min. Artifacts: `../results/EXP-E11_lr-mlp-hybrid-data/`.*

## The configuration (pre-registered in DECISIONS.md 2026-10-02)

- **Reliable noise filter** (`committee_filter.py`) replaces Farid's single-NB judge: 3 repeats x 3-fold cross-validated voting by an entropy tree and the mixed NB; **a row is deleted only with 6/6 votes against** (both judges misclassify it in every repeat, and no judge ever scores a row it trained on). Literature basis: Brodley & Friedl 1999 (single-algorithm filters are the weakest; consensus committee), Khoshgoftaar & Rebours 2005 (*Intelligent Data Analysis* 9:487–508 — repeat the k-fold vote), Johnson & Khoshgoftaar 2022 (*Journal of Data and Information Quality* 14(1), DOI 10.1145/3492546 — NB is stable under noise, so a committee member not a sole judge). See `notes/committee-filter-literature.md`.

> **Citation corrections (2026-10-02).** Two references in this note were wrong, now fixed in `committee-filter-literature.md`: Johnson & Khoshgoftaar (2022) is in **JDIQ 14(1)**, not *ACM Computing Surveys*; and the "IPF, JCST 22(3):387–396" attribution for Khoshgoftaar & Rebours could not be verified — Crossref returns their *Intelligent Data Analysis* 9:487–508 (2005) paper instead.

> **Superseded by E12 on the noise-removal question.** E11 could not tell whether the committee removed mislabeled rows or merely hard ones, because these ten datasets carry no known label noise. `notes/e12-noise-removal.md` injects noise we control and grades each filter on precision/recall against ground truth.
- **Copy A (clean)**: common data minus committee noise. **Copy B (weighted)**: Alg 2's selected attributes scaled by 1/sqrt(depth), fit on the common data. **New data = A x B**, as the user's design describes.
- **Classifiers**: sklearn LogisticRegression, and a PyTorch MLP (1 hidden layer, 32 ReLU units, Adam, early stopping) — the most fundamental deep architecture, kept small so the comparison is about the data, not capacity. Everything standardized per training fold; test folds never lose rows.

## Headline numbers (accuracy, mean over 50 folds; delta vs old)

| dataset | committee removed % | LR: clean / weighted / new | MLP: clean / weighted / new |
|---|---|---|---|
| breast-cancer | 9.9 | +2.5 / +2.1 / **+3.4** | +0.7 / +1.4 / **+2.0** |
| contact-lenses | 6.9 | +6.7 / +6.3 / **+9.7** | +2.3 / −6.0 / −2.3 |
| diabetes | 7.4 | −0.1 / +0.1 / −0.2 | +0.6 / +0.5 / +0.5 |
| glass | 7.5 | −1.5 / −0.8 / −2.6 | −0.0 / −3.0 / −3.2 |
| iris | 2.3 | +0.3 / +0.3 / +0.3 | −0.0 / +4.5 / **+4.4** |
| soybean | 2.1 | −1.1 / −1.0 / −1.1 | −0.4 / −1.7 / −2.0 |
| vote | 2.0 | −0.6 / −0.1 / −0.8 | +0.2 / −0.1 / −0.3 |
| image-segmentation | 0.9 | +0.1 / −0.9 / −0.7 | −0.2 / −1.4 / −1.4 |
| tic-tac-toe | 0.6 | 0.0 / −2.1 / −2.0 | −5.7 / −4.0 / −11.0 |
| nsl-kdd | 0.2 | −0.0 / −0.8 / −0.8 | −0.0 / −0.4 / −0.4 |
| **macro-average** | | **+0.6 / +0.3 / +0.5** | **−0.2 / −1.0 / −1.4** |

Wilcoxon on the 5 seed-level means: every p is 0.0625 or larger — with 5 seeds the smallest attainable two-sided p IS 0.0625 (all five deltas the same sign), so no per-dataset result reaches conventional significance; read the deltas as effect sizes. The exploratory fold-level t-tests are in `summary.csv`.

## What the numbers say

1. **The committee filter is validated as the *safe* noise remover — the target "solidify the DT+NB noise removal" is met for the cleaning half.** Farid's Alg 1 deleted up to 47 % of rows and cost the tree −4.8 pts (E9p); the committee deletes 0.2–9.9 % and *never* turns a win into a collapse. Where the data actually has noise it pays: breast-cancer and contact-lenses gain for LR (+3.4, +9.7) and MLP (+2.0), iris +4.4 for MLP; where the data is clean (vote, tic-tac-toe, image-segmentation) it stays within ±0.8 for LR. The reliability mechanism works as the literature promises: precision instead of recall.
2. **The attribute-weighting half does not transfer to LR/MLP.** Weighted-only is the worst arm for MLP (−1.0 macro-avg) and hurts LR on 6/10 datasets. These weights were built for NB's likelihood exponents (Eq. 14); on standardized LR they act as input shrinking (l2 re-penalizes the shrunk features), and they disturb the MLP's optimization the same way. tic-tac-toe shows it most: clean-only is 0.0 for LR while weighted-only is −2.1 — the new data's loss there is entirely the weighting.
3. **Consequently "new data" only wins where both halves help** (breast-cancer, contact-lenses for LR; iris, breast-cancer for MLP). Macro-averaged, new-LR is +0.5 over old — positive but carried entirely by the cleaning half (clean-only is the best LR arm, +0.6).
4. **Two honest caveats about the filter we must keep visible:**
   - **Rare classes get wiped on imbalanced data.** On NSL-KDD the committee deletes 100 % of ftp_write, land, spy, loadmodule, multihop rows (anything with a handful of instances cannot survive 6/6 cross-validated votes). Accuracy barely moves (99.1 → 98.3) but macro-F1 falls 84.6 → 83.5 (clean) and 77.6 (new). This is the class-bias failure mode of sample-selection cleaning (L34) — a per-class removal guard (E5b's per-class thresholds) is the direct fix.
   - **The MLP is fragile to any data perturbation.** On tic-tac-toe removing 0.4 % of rows costs the MLP −5.7 with cleaning alone (old std ±3.5, clean std ±6.9) — tiny deletions reshuffle the batches and the retraining lands elsewhere. The MLP, not the filter, is the unstable component there.

## Verdict

The reliable DT+NB committee filter replaces Alg 1 successfully: conservative, class-aware diagnostics, gains on noisy datasets, neutral on clean ones — the noise-removal target is achieved *for the cleaning stage*. The combined "new data" as configured (with Alg 2 weighting) does not increase quality overall for LR/MLP: the weighting copy is the weak link, not the cleaning.

## Next (in priority order)

1. Add a per-class guard to `committee_filter.py` (never delete below k rows of a class, or per-class vote thresholds — E5b) and re-run: this should recover the NSL-KDD macro-F1 without losing the wins.
2. Re-run E11 with `--rule majority` (sensitivity; higher recall, checks whether consensus is leaving recoverable noise in).
3. Make copy B optional per arm via inner-CV selection (E10's design) instead of always-on weighting — for LR/MLP the weighting needs to earn its place.
4. For the MLP story: 5 seeds is the floor; a 10-seed run on the 4 datasets with |delta| > 2 would firm up the small p-values.
