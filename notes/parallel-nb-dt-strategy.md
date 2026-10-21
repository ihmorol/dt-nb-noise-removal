# A three-classifier parallel NB–DT strategy

Date: 9 October 2026. Status: evidence synthesis and proposed development protocol. No new cleaning method has been trained or shown superior.

## Recommendation and scope

Use exactly three final classifiers: Logistic Regression (LR), Decision Tree (DT), and Naive Bayes (NB). MLP is excluded from the new study. Its historical code and outputs remain available for reproducibility. The current E12 runner already defaults to NB, DT, and LR, so removing historical MLP support is unnecessary.

I recommend testing **NB-led, DT-checked deletion in parallel with tree-based feature selection and weighting**. Both branches start from the same unchanged training partition. NB nominates suspicious rows; DT can veto their removal. Separately, several tree fits propose feature scores. Combine the retained rows with selected columns and carry the weights as metadata.

The main change from our old parallel experiment is the evidence used for each decision: out-of-fold disagreement, capped removal within each class, several trees instead of one, and separate tests of selection and weighting. These changes are hypotheses, not proven improvements.

The output is one training dataset plus feature metadata. Each classifier interprets the weights appropriately. There is no universal multiplication of CSV columns that supplies equivalent feature weighting to all three learners.

## What was analyzed

The audit script [connect_runs.py](audit-artifacts/connect_runs.py) reads the saved experiment outputs and writes descriptive tables under [parallel-strategy](audit-artifacts/parallel-strategy). It does not train models or alter results. Input SHA-256 hashes are recorded in [source_hashes.json](audit-artifacts/parallel-strategy/source_hashes.json) and checked again after analysis.

Coverage includes E9 pipeline metrics and every saved CSV sweep, E3a protocol results, both R1 replication summaries, faithful E9, E5, the LR portion of E11, and the three-classifier E12v2 main and deletion runs. The invalid original E12 pilot and E12 smoke run are not efficacy evidence. Historical MLP outcomes do not enter this new synthesis.

For E12v2, the audit recomputed accuracy and macro-F1 from raw confusion matrices, pooling folds within each seed and then averaging seeds. It reproduced the saved dataset summaries for all 57,600 main records and 9,600 deletion records. The deletion/correction comparison below uses only the shared seed 0, shared noise mechanisms, and shared rates. Their untreated baseline scores agree in all 120 shared dataset/mechanism/rate/classifier conditions. Detection summaries were read from saved dataset-level files, not reconstructed from their raw records in this audit.

All new contrasts here are exploratory. We did not pool numerical outcomes across implementation tracks or run new significance tests. Descriptive differences do not identify causes.

## Patterns that survive comparison across the runs

### Hard NB decisions are the main concern

In the saved final sklearn pilot, plain DT accuracy was 85.16%, NB-filtered DT 80.58%, and the sequential combinations 80.52% and 79.54%. The faithful E9 track tells a similar story with different models and original columns: plain DT 86.33%, NB-filtered DT 81.03%, parallel DT 81.05%.

E3a and R1 show much higher scores when cleaning happens before CV. That changes information access and the population tested, so it cannot rescue the cleaning claim. The source output, rather than older narrative summaries, is used for these figures.

In E12v2 at 20% added noise, NB-only threshold correction had mean detection precision 0.467 and recall 0.719. The dual rule had precision 0.640 and recall 0.464. The NB-only rule changed more rows but increased reference-label error from 19.89% to 26.97%; the dual rule reduced it to 16.64%. These are dataset-equal averages against injected/reference labels, not verified natural-noise rates.

**Inference:** NB is useful for producing suspicion scores, but the saved evidence does not support letting it delete or correct every disputed row by itself. A second model should check its nominations.

### Tree feature treatment helped NB in one track, not universally

| Faithful E9 treatment | Final NB accuracy | Final DT accuracy |
|---|---:|---:|
| No treatment | 75.93 | 86.33 |
| Attributes only | 77.35 | 86.19 |
| Parallel rows and attributes | 76.39 | 81.05 |

Adding plain-NB deletion reduced the feature-only NB mean. The parallel tree was much worse than the attributes-only tree.

However, in the sklearn/one-hot pilot, attribute-only NB scored 76.45% versus plain NB's 77.82%. Therefore, we cannot claim that the depth-weight formula generally helps NB. Representation, likelihood, learner and settings differ across the tracks.

**Inference:** feature treatment is worth testing for NB, but the original weighting rule should be a comparator, not an accepted optimum.

### Combining operations often failed to add value

In E11, using only LR and giving datasets equal weight:

| Treatment | Accuracy change, points | Macro-F1 change, points |
|---|---:|---:|
| Clean rows only | +0.617 | +1.224 |
| Selected/scaled features only | +0.317 | -0.206 |
| Both | +0.505 | +0.401 |

The combination exceeded clean-only accuracy on two datasets, tied on one, and was lower on seven. Its feature operation removed columns and scaled inputs together. The result does not separate the two effects.

**Inference:** row treatment, column removal, and feature weighting need separate arms. A full combination should earn its place by improving on its components.

### The final classifier changes the answer

Recomputed E12v2 dual-correction effects, in macro-F1 percentage points:

| Added noise | DT | LR | NB |
|---|---:|---:|---:|
| 0% | +0.252 | +0.972 | -0.174 |
| 10% | +4.295 | +2.345 | +1.579 |
| 20% | +5.641 | +0.786 | +1.134 |
| 40% | +7.911 | -0.203 | +1.295 |

At 20%, LR accuracy was nearly unchanged (-0.022 points) and NB accuracy fell 0.797 points, despite positive macro-F1 means. At that rate, Tic-tac-toe LR lost 11.52 macro-F1 points, while its DT gained 0.93. Glass lost with all three. Contact Lenses DT lost 5.31 while LR and NB gained.

**Inference:** choosing a treatment by the average across classifiers can hide serious harm to one classifier. One common transformed dataset may not improve all three. Per-classifier reporting and a shared-data fallback are necessary.

### Deletion deserves a fair test against correction

The following is a new descriptive comparison using matched seed-0 conditions at 20% noise. Each entry is macro-F1 change from the matching no-treatment baseline:

| Rule and action | DT | LR | NB |
|---|---:|---:|---:|
| Committee disagreement, delete | +8.961 | +1.643 | +2.264 |
| Committee disagreement, correct | +6.612 | -0.567 | +0.019 |
| Dual confident agreement, delete | +6.257 | +1.012 | +1.217 |
| Dual confident agreement, correct | +6.200 | +0.086 | +0.407 |

For deletion, two judges can disagree with the recorded label while proposing different alternatives. Correction needs a replacement. Our committee and dual rules differ in both confidence conditions and replacement agreement, so this table does not isolate why their outcomes differ.

This supports testing deletion, but does not prove it wins: only one deletion seed is available. At zero added noise, committee deletion still removed about 7.98% of reference-labeled rows. Better mean downstream scores do not prove those rows were mislabeled.

**Recommendation:** make capped disagreement-based deletion the new candidate, with existing committee deletion and dual correction as comparators. This updates the earlier guide's preference for correction after the matched comparison and the user's revised scope.

## Proposed method, step by step

### Start with an unchanged training partition

Write the data as \(D=\{(x_i,\widetilde y_i)\}_{i=1}^n\), where the recorded label \(\widetilde y_i\) may be wrong. Use a small, fixed number of inner judge folds, initially three for development. Fit one NB and one DT per fold on its complementary rows. Handle small classes conservatively; unscorable rows remain.

Each row gets probabilities from judges that did not train on it. Use the existing NB likelihood and tree settings first to avoid changing every part at once. Keep all learned preprocessing inside the training partition.

### Branch A: NB proposes removals, DT checks them

For judge \(g\), define the alternative-versus-recorded-label margin:

\[
m_i^{(g)}=\max_{c\ne\widetilde y_i}p_i^{(g)}(c)-p_i^{(g)}(\widetilde y_i).
\]

A row is eligible only when \(m_i^{NB}>0\) and \(m_i^{DT}>0\): both prefer some other class. Ties are retained. This is an NB-led filter with a DT check, not an NB-only filter. Different alternatives are allowed because the operation is deletion. The existing same-alternative dual rule remains a comparator.

Within each recorded class, rank eligible rows by NB margin, largest first. The margin is a ranking score, not a calibrated probability of a wrong label. Fix a stable row-ID tie-break; do not select ties using outcomes.

For class \(c\), allow at most

\[
b_c=\min\left(\lfloor r n_c\rfloor,\max(0,n_c-5)\right)
\]

deletions, taking fewer if fewer rows are eligible. Here \(r\) is a maximum removal fraction, not an estimated noise rate. Classes of five or fewer lose no rows. The initial development grid can use \(r\in\{0,0.05,0.15\}\). Those values are proposed settings, not values proven best by the saved runs. They must be fixed before fresh confirmation.

The cap limits damage by count but cannot guarantee class recall. It also limits noise removal at high corruption, which must be reported rather than hidden.

### Branch B: select and weight original feature groups

Use the same fixed collection of DT judge fits to derive feature scores, before any row deletion. A categorical attribute's dummy columns form one group. Do not count dummy columns as independent original attributes.

For group \(j\), define

\[
q_j=\frac1K\sum_{k=1}^{K}
\begin{cases}
1/\sqrt{d_{kj}},&\text{tree }k\text{ uses group }j,\\
0,&\text{otherwise},
\end{cases}
\]

where \(d_{kj}\) is the shallowest depth of any column from that group, with root depth 1. This averages the existing depth heuristic across several trees. The trees overlap in training data; these are not independent votes and \(q_j\) is not an importance probability.

Use either all original groups or the top \(\lceil0.75d\rceil\) groups, with a fixed tie rule. If all scores are zero, keep all groups and unit weights. The 75% budget refers to all original groups, not only those with nonzero scores. It is a modest proposed reduction, not a proven threshold. High-cardinality and correlated-group bias remains possible.

For retained groups, test either unit weights or shrunken depth weights:

\[
w_j=0.5+0.5q_j.
\]

This puts every retained group weight between 0.5 and 1. It deliberately avoids extreme suppression while we test the idea. The shrinkage is a proposed heuristic; older weighting failures do not prove this fixes them.

Both branches use the original training partition. The feature scores are not refitted after deleting rows. That preserves the parallel design and keeps it distinct from a sequential strategy.

### Combine into one dataset and a weight record

If \(R\) contains retained rows and \(S\) contains selected columns,

\[
X'=X[R,S],\qquad y'=\widetilde y[R].
\]

No labels are changed by the proposed deletion candidate. Export feature names/group mapping and \(w\) separately. Preserve original row IDs and record why each row was removed. Test examples lose the same columns but never lose rows.

For model column \(\ell\), broadcast its original group's weight as \(v_\ell=w_{g(\ell)}\). Thus every dummy column from a retained categorical attribute receives that group's weight. This does not remove bias from a group contributing several dummy-column likelihood terms.

The final learners receive the same retained rows, labels and original feature groups, with these distinct weight meanings:

| Learner | How to use the weights |
|---|---|
| NB | Use likelihood exponents: \(\log P(c)+\sum_\ell v_\ell\log p(x_\ell\mid c)\). Use the existing weighted-NB mechanism. |
| DT | Use the selected columns without multiplying by weights. A positive rescaling preserves column order and the possible split partitions in exact arithmetic. |
| LR | First standardize using training data, then optionally multiply each model column by \(v_\ell\). Keep a unit-weight comparator and match regularization settings. Do not standardize again afterward. |

For LR, writing \(\beta_\ell=v_\ell\theta_\ell\), an L2 penalty on the fitted coefficients \(\theta\) becomes

\[
\sum_\ell\theta_\ell^2=\sum_\ell\frac{\beta_\ell^2}{v_\ell^2}.
\]

Thus smaller weights impose stronger penalties on the corresponding effective coefficients. This is a modeling choice, not generic data cleaning. Multiplying columns before a second standardization would cancel positive weights. Multiplying Gaussian NB inputs is also not equivalent to likelihood weighting because its fitted means and variances scale with the data.

## Choose one common candidate without hiding classifier harm

Three row caps times two feature budgets times two weight modes give 12 candidate configurations. Include the original-data, unit-weight configuration. DT has only six distinct candidates because the weight mode does not alter its inputs. The grid has at most 12 NB, 12 LR and 6 DT distinct final fits per validation partition before data-dependent duplicate removal, plus six judge fits for three-fold judging. Extra comparator arms and outer refits add work. This is a fit count, not a runtime guarantee.

Select one common row mask rule and feature/weight rule using development-only validation, subject to individual learner checks. A defensible pilot objective is the worst of the three macro-F1 improvements:

\[
J(\lambda)=\min_{a\in\{LR,DT,NB\}}
\left[\widehat{F1}_{macro}(a,\lambda)-\widehat{F1}_{macro}(a,0)\right].
\]

Choose the candidate with the largest \(J\) only after it passes prespecified per-learner accuracy and per-class recall tolerances. Use fewer features or lower measured cost as a tie-break. If no candidate provides credible useful improvement under the development rule, select the original data. Do not call the maximum over candidates an unbiased final result.

These screens are empirical unless independent trusted labels and valid uncertainty bounds support certification. Small-class checks may be too uncertain to authorize treatment. Thresholds and the uncertainty procedure still need to be set in the new protocol; this document is not claiming that protocol is already locked.

For honest selection, every inner validation split must rebuild preprocessing, out-of-fold judges, feature scores and candidate masks using only its inner training portion. Reusing outer-training judge predictions would leak inner-validation labels. Cache judges across candidates only within the same training partition. Refit the chosen rule on outer-training data and evaluate once on untouched outer-test rows. That refit is valid for nested empirical evaluation; it is not an automatic extension of a frozen-model audit certificate.

If the common candidate falls back frequently, report that. A later model-specific treatment study is possible, but it would no longer answer the present goal of one common new dataset.

## Comparisons needed to explain the outcome

- Original data, with no deletion, selection or weighting.
- NB-only deletion with the same caps, to test whether DT checking adds value.
- Capped NB–DT deletion only.
- Tree selection only, with unit weights.
- Tree weighting only on all features, with the per-learner meanings above.
- Full parallel combination, with and without weighting.
- Existing E12 committee deletion and dual correction, under matching new folds and budgets.

The 12-candidate family contains several component comparisons already; reuse those fits. External published comparators such as proper Confident Learning remain needed before a publication claim, but adding them is a separate implementation step. Give all comparators equal access to labels and tuning resources.

Report accuracy and macro-F1 for each final classifier separately, all class recalls, removed-row counts, true/false removals under controlled corruption, attribute-group counts, feature-score variability, total runtime and peak memory. Report losing datasets and zero-added-noise results. Keep detection truth hidden from the algorithm.

## Efficient execution and the next decision

First implement only this bounded prototype and run correctness checks, followed by a small development matrix covering zero and 20% added noise. Include known difficult datasets rather than selecting only favorable ones. The current ten datasets are development evidence because their outcomes informed the design.

Then decide whether the combined arm adds value beyond its components. Expand to more seeds and fresh datasets only if that question has a useful answer. Full new runs must use separate output directories and recorded source/config hashes.

For larger data, chunk NB/tree scoring and avoid keeping full repeated probability matrices where cached margins suffice. Group scoring can reuse the fitted DT judges. Cap candidate count and measure all judge, selection and final-training costs. Final training and nested validation still require substantial work; three-fold judging alone does not establish scalability. Bounded sampling or histogram trees would be later approximations requiring their own validation.

Do not launch a huge search for the best score on the current outputs. Stop or simplify if the feature branch harms LR, if DT checking adds no value, or if the common dataset does not improve the weakest-performing learner enough to justify the work.

The established findings are conditional: aggressive NB treatment often hurts; DT checking can reduce that harm; feature treatment helped faithful NB but not every NB representation; and the same treatment has different effects on LR, DT and NB. The proposed capped, averaged and shrunken variant remains untested.

## Evidence files

- [E9 means](audit-artifacts/parallel-strategy/e9_means.csv)
- [E5 means](audit-artifacts/parallel-strategy/e5_means.csv)
- [E11 LR effects](audit-artifacts/parallel-strategy/e11_lr_effects.csv)
- [E12 main recomputed effects](audit-artifacts/parallel-strategy/EXP-E12v2_main_effects.csv)
- [E12 classifier/dataset effects](audit-artifacts/parallel-strategy/EXP-E12v2_main_dual_by_dataset.csv)
- [Matched deletion/correction effects](audit-artifacts/parallel-strategy/matched_seed0_action_means.csv)
- [Saved detection summaries aggregated by rate](audit-artifacts/parallel-strategy/EXP-E12v2_main_detection.csv)

No saved research outputs or historical model support were removed. The only new executable code is the descriptive audit helper. The proposed algorithm, new experiments, best-result claim, true-label safety and novelty remain unverified.
