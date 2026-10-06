# Project defense guide

Use this as one reading sheet for the whole project. The conference paper covers the original NB/C4.5 comparison. The later noise-correction study is separate. Everything is now available on the GitHub `main` branch.

## The project in one minute

“I studied whether we can improve classification by changing the training data before learning. I started with Farid's two methods: use Naive Bayes to remove rows it predicts incorrectly, and use a decision tree to select and weight useful attributes. I implemented both, combined them in both orders, and compared them with plain classifiers. I fitted cleaning inside each training fold and kept every test row. The combinations did not improve mean tree accuracy. I then tested committees, softer handling, and label correction. In the later study, I added known label errors to training data so I could measure detection as well as classification. Agreement-based correction helped decision trees on average, but the main test did not establish an overall improvement. My next step would be a separate study of that tree-specific benefit and the harm to data with no added errors.”

## Basic terms and why we used them

- **Row / instance:** one example. **Attribute / feature:** one input column. **Label:** the class to predict.
- **NB:** estimates each class's probability from its prior frequency and the input values. It treats features as independent within a class. It is simple, but its mistakes can reflect a weak assumption rather than a wrong label.
- **Decision tree:** asks feature questions until it reaches a class prediction. C4.5 is implemented by Weka J48; sklearn CART is a different tree implementation.
- **LR:** Logistic Regression, a simple linear classifier. It checks whether a treatment helps beyond NB and trees.
- **MLP:** a small neural network, here one hidden layer with 32 ReLU units. It checks limited transfer to another model, not modern deep-learning performance in general.
- **Judge versus final classifier:** the judge decides which training rows to change; the final classifier learns from the treated data and predicts test rows. They have different jobs.
- **Arm:** one method being compared, such as NA followed by NB. **Baseline:** the same classifier without cleaning.
- **Accuracy:** percentage of correct predictions. **Macro-F1:** calculate F1 for each class, then average classes equally. It can reveal rare-class losses hidden by high accuracy.
- **Percentage points (pp):** 86% to 81% is a fall of 5 points. It is not a 5% relative decrease.

Our question is whether cleaning helps the final classifier. A smaller dataset, fewer attributes, or better-looking training accuracy alone cannot answer it.

## The original methods and our combination

**Algorithm 1, N: filter rows.** Fit NB on the training data, predict those same rows, and remove rows whose predictions disagree with their supplied labels. Then fit the final classifier again. These predictions are in-sample. We call rejected rows *suspected* errors, not proven errors. Our safeguard cancels deletion if fewer than two classes would remain; it does not protect every small class.

**Algorithm 2, A: select and weight attributes.** Fit a tree and keep attributes that it tests. An attribute's weight is `1 / sqrt(its smallest tree depth)`, with root depth 1. Depths 1, 4, and 9 give weights 1, 0.5, and about 0.33. Untested attributes are dropped. Weighted NB multiplies each attribute's log-likelihood contribution by its weight; it does not simply multiply the raw input value. A final tree uses selected columns without NB's weights.

**Algorithm 3: combine the steps.** NA means filter rows, then select attributes from the remaining data. AN means select attributes first, then filter with weighted NB. In AN, the same weights reach the judge and the final NB; they are not refitted after deletion. This weight passing is called *support*. Each order ends in NB or a tree. The comparison therefore includes the implemented order and the plain-versus-weighted judge, not a pure order-only effect.

The conference comparison has ten methods: two plain classifiers; N and A each ending in NB or a tree; and NA and AN each ending in NB or a tree. The single-step reference methods tell us whether a change comes from cleaning or simply choosing a different final classifier.

**Parallel control, E3b.** Run N and A independently on the same original training fold, then intersect their retained rows and columns. It tests whether chaining the steps matters. These two additional arms are saved in E9 results but excluded from the conference paper's ten-method comparison.

**Empty-tree handling.** In the faithful original-column implementation, standalone A--NB uses class priors if no attribute is selected. Combined paths keep current columns with unit weights. The earlier CART pipeline also keeps all columns in that case. No empty-selection safeguard occurred in the stored E9 conference runs.

## How one honest evaluation works

1. Split the data into ten folds, roughly keeping class proportions. Use nine folds for training and one for testing.
2. Fit the judge, attribute selector, weights, scaling when needed, and final classifier using training rows only.
3. Apply the selected columns to the test fold. Keep **every test row** and do not use its labels for cleaning.
4. Predict, record accuracy and macro-F1, and record removed rows, columns, and within-class removal rates.
5. Rotate the test fold and repeat with fixed random seeds. All compared methods share the splits.

**Protocol B** follows this procedure and is the main evaluation. **Protocol A** cleans the complete dataset first, then cross-validates on the cleaned sample. It sees future test labels and, when filtering, removes future test cases. Its higher scores mix label access with a changed evaluation population. **Protocol C**, used in the early diagnostic work, fits and scores on the same full data; it is a training-score diagnostic, not evidence of generalization. None identifies exactly which procedure the 2014 paper used.

E9 averages its 100 fold scores per dataset, then weights the ten datasets equally. Its macro-F1 class set can vary between folds. E12v2 instead combines test-fold confusion matrices within each seed before calculating scores, then averages seeds. This avoids averaging very small test-fold F1 scores. The two studies have different learners, data representations, and scoring conventions; their numbers must not be pooled.

The datasets are Breast cancer, Contact lenses, Diabetes, Glass, Iris, Soybean, Vote, Image segmentation, Tic-tac-toe, and NSL-KDD. Their sizes range from 24 to 25,192 rows. Contact lenses has classes with four or five examples; NSL-KDD includes a singleton class. Ten folds can therefore lack a class in training or testing. Repeated splits cannot create missing examples.

## What we implemented and tested along the way

| Work | Why we did it | What it tells us |
|---|---|---|
| Data checks and early E9 pipeline | Build loaders and try the two orders with sklearn CART and one-hot inputs | Filtering reduced mean tree accuracy; pruning, likelihood, and support settings were explored. These are pilot results. |
| E3a protocol diagnostic | Compare cleaning before CV, within folds, and full-data scoring | Before-CV filtering gave much higher scores. This does not isolate leakage from sample selection. |
| R1 replication attempt | Move closer to the original learner and attribute representation | Weka J48 plus textbook NB reproduced some baselines closely, but exact original scores remain unresolved. Three seeds; also an unpruned-tree sensitivity. |
| E9f conference study | Compare the original methods, both orders, and reference methods fairly | Original columns, pruned J48 3.8.6, ten seeds and ten folds. Main results below. |
| E3b parallel control | Compare independent selection with sequential cleaning | Parallel--DT reached 81.05%, versus 81.21% for NA and 80.89% for AN. It did not recover plain C4.5's 86.33%. |
| E5 treatments | Replace hard deletion with soft weights, majority deletion, or correction | DT accuracy: soft 82.16%, committee 85.77%, correction 85.12%; plain tree 86.33%. Less damage than single-NB deletion is not a win over no cleaning. |
| E11 transfer | Test original, cleaned, weighted, and combined data with LR/MLP | Combined accuracy changed LR +0.50 points and MLP -1.38. Rare classes could disappear; safety was not established. |
| Original E12 pilot | Add known training-label errors and test probability-based rules | Correction mistakenly deleted its flagged rows, and its AUROC score was unsuitable. Retained for history, excluded from current claims. |
| E12v2 | Repair the pilot; evaluate fixed correction, deletion sensitivity, and MLP transfer | Main study completed; conditional gains, but no established overall superiority. Details below. |

Some old registry statuses are stale: later runs cover earlier baseline, combination, deletion-diagnostic, and test questions. Stronger published/learned attribute-weight comparisons (E8/E10) and broad natural-noise validation remain unfinished. Earlier statements calling filters safe or methods equivalent are not supported by the current evidence.

## Conference results and how to defend them

All means give datasets equal weight under Protocol B.

| Method | Accuracy % | Macro-F1 % |
|---|---:|---:|
| Plain NB | 75.93 | 69.80 |
| Plain C4.5 | 86.33 | 81.92 |
| N--NB | 75.42 | 69.08 |
| N--DT | 81.03 | 75.70 |
| A--NB | 77.35 | 71.73 |
| A--DT | 86.19 | 81.84 |
| NA--NB | 75.63 | 69.60 |
| NA--DT | 81.21 | 75.76 |
| AN--NB | 76.35 | 69.91 |
| AN--DT | 80.89 | 73.93 |

“The combined tree methods lost about five accuracy points against plain C4.5. Both combined NB methods trailed attribute weighting alone. Combining the steps did not produce a better mean result.”

We used eight two-sided Wilcoxon tests across ten dataset averages, then Holm adjustment because testing several comparisons increases the chance of a false finding. None had adjusted p below 0.05. AN--DT versus plain C4.5 had raw p=0.014 but adjusted p=0.109. These tests were added during revision. Lower observed means do not prove universal harm; nonsignificance does not prove equality. Overlapping folds and seeds are not independent datasets.

N--DT rose from 81.03% under B to 94.52% under A, a 13.50-point difference calculated before rounding. It is a score change, not proven improvement. In Breast cancer, plain NB removed 50.63% of recurrence rows versus 14.51% of non-recurrence rows; weighted NB removed 58.97% versus 11.72%. These within-class rates show uneven deletion, not confirmed wrong labels.

Exact replication remains unresolved. NSL-KDD C4.5 scored 99.47%, versus 71.11% reported in 2014. Our Glass has six observed classes, NSL-KDD 22, and segmentation 2,310 rows. We cannot claim identical original files and splits.

## Why the later methods were different

**E5:** three judges, textbook NB, J48, and standardized sklearn LR, predicted their own training rows. Soft treatment used NB's probability of the supplied label as a row weight. Majority deletion removed rows disputed by at least two judges. Correction changed a label when at least two proposed the same alternative. Weka Logistic initially timed out, so the final implementation used sklearn LR. Raw exploratory tests favored committee deletion over hard deletion (+4.74 accuracy points, p=0.0078), but committee accuracy remained 0.56 points below the plain tree. These results do not prove equality or safety.

**E11:** NB and CART judged training rows out of fold, with three repetitions of three-fold judging. Consensus required six votes against a label. Copy A retained rows; copy B selected and depth-weighted features from the common training data; combined data used both. Inputs were standardized before weights were applied to LR/MLP. That differs from NB likelihood weighting. Selection and scaling changed together, so the losses cannot identify which was responsible. The MLP used 32 hidden units, Adam, and early stopping. Exported full-data CSVs are inspection examples, not evaluated fold inputs.

The committee could eliminate small classes. Retaining two classes overall did not protect every class. Also, no independent annotations told us whether rejected labels were actually wrong. We therefore moved to controlled corruption.

## E12v2: what happens and what we obtained

Change some labels in the outer **training fold only**; keep test labels unchanged. Symmetric noise chooses a random different class. Pairflip changes each class to a fixed next class. Try 0%, 10%, 20%, and 40% added noise. Those mechanisms coincide for binary data. The older NB-derived asymmetric mechanism is exploratory, not part of the fixed main study.

Use five-fold out-of-fold NB and CART judges within corrupted training data. A class's confidence threshold is the mean probability assigned to rows carrying that label. **Dual correction** changes a label only when both judges disagree with it, clear their proposed class thresholds, and suggest the same replacement. Otherwise retain it. Keep all rows, protect singletons, and prevent departures below `min(5, original class size)`.

The known corruption mask and original reference training labels go only to evaluation, never to the filters. Precision measures how many flagged rows really were injected errors; recall measures how many injected errors were found. Detection alone cannot guarantee correct replacement labels or better final classification.

Eight arms compare no treatment, NB/DT hard-disagreement correction, NB/DT threshold correction, committee correction, dual correction, and original-reference-label training. Committee correction can use the average probability vector even when judges suggest different replacements; dual correction requires agreement. Original-label training is a reference, not a guaranteed upper bound. The rule is a threshold heuristic inspired by earlier work, not full Confident Learning or demonstrated methodological novelty.

The main run used ten datasets, three seeds, ten outer folds, two mechanisms, four rates, and NB/DT/LR finals: 57,600 classification and 16,800 detection records. Scores pool outer-fold confusion matrices within each seed. The primary endpoint was dual-correction minus no-cleaning macro-F1 at 20% noise, averaged over classifiers and mechanisms, with one value per dataset. The fixed protocol followed exposure to the earlier pilot, so it was not a blind first study.

| Result | What I can say |
|---|---|
| Main macro-F1 +2.5205 points; 8/10 datasets positive; p=0.083984375 | Overall improvement was not established at 0.05. |
| DT +5.642; NB +1.134; LR +0.786 macro-F1 points | These descriptive gains suggest studying the final tree separately. |
| Glass -1.938; Tic-tac-toe -4.024 points | Report these losses, not only favorable datasets. |
| Zero added noise: 5.44% of reference labels changed; accuracy -0.094, macro-F1 +0.350 points | Not harmless: Glass, Segmentation, Soybean, and Tic-tac-toe lost macro-F1. Original labels are unverified references. |
| Separate deletion check: +2.829 macro-F1 points at 20% | One-seed sensitivity, not proof that deletion is better. |
| MLP check: +1.434 at 20%, -0.162 at zero added noise | Iris/Diabetes/Vote, one seed; Diabetes lost at 20%. No broad neural-network claim. |

## Implementation map and checks

| Files in `code/` | Job |
|---|---|
| `data.py`, `nb.py`, `faithful_nb.py` | Load original or one-hot data; implement the two NB paths |
| `algorithm1.py`, `algorithm2.py`, `pipeline.py`, `main.py` | Filter/select, pass weights, preserve test rows; run early CART experiments |
| `weka_utils.py`, `replicate_farid.py`, `e9_farid.py` | Write ARFF data, call Java/Weka, read tree depths and scores; run R1/E9 |
| `leakage_check.py`, `e5_strategy.py` | Protocol diagnostics and alternative row treatments |
| `committee_filter.py`, `deep_mlp.py`, `e11_lr_mlp_hybrid.py` | Repeated committee judging and LR/MLP transfer |
| `noise_inject.py`, `confident_filter.py`, `e12_noise_removal.py` | Add errors, make decisions, apply correction/deletion, measure E12 |

Weka needs Java and the saved jars. Python dependencies are in `requirements.txt`; the loader needs pandas below 3. Optional MLP runs need PyTorch. Model/data settings and seeds are saved with results. E9 refuses an existing nonempty output directory; other historical runners need isolated outputs to avoid overwriting evidence.

Checks cover training/test separation, correction retaining rows, class floors, finite probabilities, score consistency, and safe E9 outputs. Selected E9 folds were reproduced with Weka, but the full experiment was not rerun after later numerical safeguards. Exact E12 source-byte hashes now flag merged version/line-ending differences. Keep the recorded run versions and hashes; do not claim every stored result was generated by the current source bytes.

Further limits: nominal vocabularies come from complete files, so unseen-category deployment was not tested. FaithfulNB is not identical to Weka NB on all numeric/mixed data. Synthetic corruption is known; original label correctness is not. Small-class protection does not create enough examples to learn a rare class.

## Supervisor questions and next steps

**Why did filtering hurt?** NB disagreement can remove valid boundary cases and minority examples. The records support that concern, but do not prove one mechanism caused every loss.

**Why did order not solve it?** Both orders still depend on the judge's decisions. Our means and adjusted tests give no clear order advantage under these settings.

**Did we prove the original paper leaked?** No. Our protocol gap mixes label access and sample selection and cannot identify its exact procedure.

**What is our contribution?** The implemented comparisons, saved evidence, protocol analysis, class-removal records, and bounded correction evaluation. Algorithm 3 combines existing steps; we have not proved novelty or overall superiority. A negative result still answers whether this combination helped under the tested conditions.

**Why not increase the network size?** That changes a separate question. The issue may be treatment decisions rather than model capacity. Use fixed settings and repeated training seeds before attributing MLP losses to one cause.

**What should I improve first?** I recommend a separately specified study of the final tree's benefit and harm at zero added noise. Compare no cleaning, the current dual rule, and a proposed stricter rule. Protect small classes and report every dataset's losses. The DT finding is a lead, not a confirmed general effect.

**How can I choose better settings fairly?** Define the main outcome and safety limits before inspecting new results. Choose thresholds or model settings only within training data, using an inner validation split or inner CV; evaluate them on untouched outer folds. Add independent datasets and, where available, human-checked wrong-label examples. Do not retune these same outputs until p passes 0.05.

**What about attribute weighting?** Test selection-only, scaling-only, and their combination separately, then compare with plain classifiers. Stronger learned weights remain proposed work. NB weights and LR/MLP input scaling are different treatments.

**What would count as a better result?** Better held-out predictions under a fixed protocol, acceptable zero-added-noise and class-specific harm, and support on independent data. Do not remove hard test rows, omit losing datasets, or clean before CV. No suggested change guarantees improvement.

## Files to show if asked for evidence

The four-page paper is `paper/manuscript/conference_final.pdf`; the broader later study is `to_human/e12-report.html`. All fold scores, settings, removal records, and inspection datasets are under `results/`. Prior drafts remain in `archive/branch-snapshots/`.

Sources for this guide: [E9 means](../results/EXP-E9_farid/metrics.csv), [conference paired tests](../paper/manuscript/tables/paired_tests.csv), [E11 summaries](../results/EXP-E11_lr-mlp-hybrid-data/summary.csv), [E12 dataset changes](../results/EXP-E12v2_main/primary_by_dataset.csv), [fixed protocol](e12v2-protocol.md), [implementation audit](e12v2-audit.md), and [current findings](../findings.md). These take priority over stronger claims in historical notes.
