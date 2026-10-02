# E12v2 locked takeover protocol

## Scope and prior knowledge

Cline E12 pilot was already inspected. This is a prospective extension after
pilot exposure, not a blind original preregistration. Historical relabeling
and AUROC are invalid and will not be pooled with new results.

## Hypothesis and fixed strategy

Cross-fitted NB and entropy CART propose an alternative label separately.
The candidate corrects a row only if both clear their own alternative-class
mean-self-confidence threshold and both propose the same alternative.
Otherwise preserve its label. Protect at least min(5, original class count)
rows per observed class from departure. No tuning of thresholds from results.
This is a conservative agreement heuristic inspired by ensemble filtering
and probability thresholds, not a full Confident Learning implementation or
a claim of methodological novelty.

## Evaluation

All ten locally available Farid datasets. Three outer seeds (0,1,2), stratified
10-fold outer evaluation; five inner folds for all OOF judges, one repeat.
Singleton observed classes are protected from cleaning. Every test row remains.
Reference training labels and corruption masks are never given to filters.
Original labels are reference labels, not certified clean ground truth.
One-hot category vocabulary uses the existing loader without label fitting;
LR and MLP scaling is fitted on the treated training data only.

Rates 0, 0.10, 0.20, 0.40; symmetric random-other-class flips and independent
sorted-class cyclic pairflip (not NB-generated). These are synthetic stress
conditions, not necessarily real-world noise. Actual realized rates reported.
The NB-confusable asymmetric mechanism remains exploratory only.

Arms: no cleaning; NB/DT hard voting; NB/DT class thresholds; paired consensus;
dual confident same-label agreement; original-reference-label training.
The latter is a descriptive reference, not an attainable oracle or upper bound.
Main treatment: relabel EVERY flagged row without deleting rows for all arms.
Deletion is a separate sensitivity run at rates 0 and .2.
Classifiers: mixed NB, entropy CART, standardized LR; MLP evaluated as a separate
transfer check with fixed inherited architecture, no parameter optimization.
No cherry-picked dataset subsets in the main report.

Primary endpoint: dual-agreement relabel macro-F1 minus no-cleaning macro-F1,
averaged equally over classifiers NB/DT/LR and both noise mechanisms at .20,
then one paired value per dataset. Two-sided Wilcoxon at .05 across datasets,
not folds; one primary hypothesis. Accuracy, detection precision/recall/F1,
false-positive rate, post-treatment corruption, class harms and other arms
are descriptive secondary analyses. Report all datasets and both mechanisms.
Zero-added-noise harm is a required safety outcome, not an optimization target.
Any adaptive second hypothesis must get a new protocol before execution and
remain explicitly exploratory because these datasets have now been inspected.

## Stop condition

Finish the fixed main matrix, deletion sensitivity, and MLP transfer check;
verify row completeness and independent review; record supported and negative
findings. Do not iterate indefinitely until a positive p-value appears.

## Sources

- Brodley & Friedl, 1999: https://arxiv.org/abs/1106.0219
- Northcutt, Jiang & Chuang, 2021: https://arxiv.org/abs/1911.00068

No theorem of improvement follows from agreement: correlated model errors
and weak minority-class support remain material limitations.
