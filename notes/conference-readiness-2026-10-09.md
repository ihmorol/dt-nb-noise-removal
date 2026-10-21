# Conference readiness assessment

Assessment date: 9 October 2026, Bangladesh time.

I think this project can support a conference submission as a bounded empirical evaluation of supervised filtering. The current evidence does not establish a superior classifier, verified natural label-error detection, or exact reproduction of the 2014 scores. The recommendation is to revise the E9 paper for ICEEICT 2027, rather than launch a new research direction to meet the nearest deadline. Acceptance remains a reviewer decision.

## Repository coverage

The review covered all 12 local named branches, all nine actual remote branch heads, nine linked worktrees, the detached Kilo worktree, Cline development checkpoints, six Codex turn-diff tree refs, and six archived ZIPs. Remote heads were checked live and origin was fetched without checking out, merging, or changing manuscript files.

Local main is `aaae637`; fetched origin/main is `fac6cbc`. Local main is three commits behind. The new commits add self-service documentation and a full-results manuscript, then revise its language. They add no new model experiments. All other named branches are ancestors of origin/main. Their commits have been incorporated into its history, although individual files were reconciled or superseded rather than retained verbatim.

| Local branch | Head | Commits behind origin/main | Branch-only commits |
|---|---|---:|---:|
| main | aaae637 | 3 | 0 |
| cline/e1e3f | 81ba1a2 | 32 | 0 |
| cline/e33d3 | 4c39231 | 33 | 0 |
| dev/backup-conference-analysis-2026-10-06 | 68ef2b0 | 31 | 0 |
| dev/backup-conference-final-2026-10-06 | 5a9b9d9 | 31 | 0 |
| dev/backup-project-audit-2026-10-06 | 34beb54 | 31 | 0 |
| dev/conference-ai-detection-removal-4b2194 | 3d827c0 | 31 | 0 |
| dev/e12-research-takeover | bd3e8ab | 21 | 0 |
| docs/ieee-manuscript-literature-review | 7ebb4ef | 37 | 0 |
| feat/committee-filter-lr-mlp | ebec8a3 | 32 | 0 |
| refactor/deepen-fold-cleaner | 8f76105 | 35 | 0 |
| refactor/simplify-pipeline-code | 7ebb4ef | 37 | 0 |

All nine worktrees were clean at inspection. The detached `.kilo/worktrees/mildly-airboat` head `e87a543` is also an ancestor, seven commits behind origin/main. The auxiliary named worktrees match their corresponding remote branches.

The archives preserve E11, an E12 pilot, earlier conference versions, alternate prose, and the manuscript before four-page shortening. Hidden Cline checkpoints contain early E12 development superseded by the later implementation. The Codex tree refs contain no unique filenames; their contents represent older versions. No additional unpublished experimental result was found in the audited sources. Checkpoint inspection was structural and selective, not an exhaustive rereading of every historical line.

## Scientific status

E9 is the strongest coherent paper now. It evaluates original-column Weka J48 and textbook FaithfulNB on ten datasets, with ten seeds and ten folds per seed. The stored evidence has 12 Protocol B arms and ten Protocol A arms. The short paper reports ten B arms.

| E9 method | Mean accuracy, percent |
|---|---:|
| Plain C4.5 | 86.3294 |
| Instances first, then attributes, final C4.5 | 81.2080 |
| Attributes first, then instances, final C4.5 | 80.8920 |
| Plain NB | 75.9284 |
| Attribute weighting alone, final NB | 77.3490 |
| Instances first, then attributes, final NB | 75.6327 |
| Attributes first, then instances, final NB | 76.3462 |

None of the short paper's eight dataset-level accuracy comparisons is significant after Holm adjustment. The minimum adjusted p-value is 0.109375. These results do not establish equality between methods. They support reporting lower tree means, lack of established combination benefit, and changes in deletion behavior.

Before-CV filtering changes both information access and the population evaluated. Its large score difference cannot isolate a causal leakage effect or establish the protocol used by the earlier paper. NB rejection means disagreement with a judge; E9 has no verified wrong-label annotations.

E12v2 is a separate synthetic training-label correction study using sklearn learners and one-hot representations. Independent recomputation from its raw confusion matrices reproduces a primary mean improvement of 2.520496 macro-F1 percentage points, with eight positive and two negative datasets, and two-sided Wilcoxon p=0.083984375. Overall superiority is not established at 0.05. Its DT-specific descriptive improvement is not a replacement primary hypothesis. Deletion uses one seed and MLP covers three datasets with one seed. E12 should remain a distinct later paper or extension.

## Manuscript issues to resolve

Use the short `paper/manuscript/main.tex` as the starting point for a submission revision. The newer `main_full.tex` exists on origin/main but is not in this local checkout. Its broader tables can inform the revision after their reporting is reconciled.

The full version contains four material problems:

- It interprets nonsignificant comparisons as making no measurable difference. Replace this with no statistically established advantage under the tested settings.
- It says no arrangement beat the plain classifiers despite positive descriptive NB differences. Distinguish numerical improvement from statistically established benefit.
- It assigns the before-CV score increase to specific causal mechanisms. The experiment changes several conditions together; report a protocol contrast.
- Its table has 16 contrasts with two outcomes each, and the generator adjusts all 32 p-values together. The abstract says 16 comparisons; the generator docstring says a family of 24. Reconcile the prose, code documentation, tables, and chosen family. The short paper's eight accuracy tests are a different explicit scope, not interchangeable p-values.

Aggregate resemblance to the 2014 means does not establish a sound exact reproduction. NSL-KDD has 99.47% baseline C4.5 accuracy versus the earlier reported 71.11%, and dataset sizes/class counts differ. FaithfulNB is not identical to Weka NB on mixed/numeric data. Describe the actual implementation and disclose the unresolved differences.

## Improvements by priority

For the nearest deadline:

- State the contribution as an empirical comparison of inherited filtering and weighting steps, their implemented orders, and class-specific removal. A possible title is “Evaluating Instance Filtering and Attribute Weighting in Decision Tree and Naive Bayes Hybrids.”
- Correct the claim problems above and retain explicit limitations. Do not present Algorithm 3 as established methodological novelty solely because two existing steps are composed.
- Keep E9 and E12 separate. Select and document one statistical family for the submission; do not select it to obtain a smaller p-value. Existing dataset-level inference is preferable to treating repeated folds as independent samples.
- Use the available space for contribution positioning, removal diagnostics, and reproducibility. There is no requirement to fill all six pages.
- Obtain coauthor agreement on scientific interpretation, author order, and the submission version; anonymize the review PDF and adapt it to the target rules.

The most useful bounded additional analysis is a fixed-class, pooled-prediction macro-F1 sensitivity check for E9. Current E9 F1 averages fold scores whose class sets can vary. Its stored fold-score files cannot reconstruct pooled F1, so new predictions or confusion matrices are needed in a separate output directory. This is an improvement to reporting, not an automatic invalidation of the disclosed accuracy study. If compute threatens the deadline, retain accuracy as the primary endpoint and clearly qualify macro-F1.

For a stronger later research contribution, prioritize a fixed held-out population to separate protocol effects, a plain-versus-weighted judge ablation if making an order-only claim, and fresh datasets for any hypothesis developed from E12. A new-method label-cleaning claim would benefit from a published cleaning comparator under matched folds and treatment semantics. Confident Learning is an established distinct method, not the name of the current threshold heuristic: [primary paper](https://arxiv.org/abs/1911.00068).

Do not repeatedly tune these benchmarks until a significant result appears. More split seeds do not turn ten datasets into more independent datasets. Do not add a deep model solely to make the paper look more advanced.

## Near-term conference target

ICEEICT 2027 at MIST, Dhaka is my recommended near-term target. The official site lists 25 October 2026 as the final extended paper deadline and 28–30 January 2027 as the event dates. Its AI and Data Analytics track explicitly includes machine learning. IEEE Xplore inclusion is conditional, and technical co-sponsorship is still marked pending approval. [Official dates and status](https://iceeict.mist.ac.bd/), [scope](https://iceeict.mist.ac.bd/authors/call-for-papers/).

Initial submissions require an anonymous IEEE double-column A4 PDF, at most six pages including references, at most 3 MB, with embedded fonts and no Type 3 fonts. The site also specifies LLM-use disclosure and author approval. Disclosure must reflect actual assistance; do not copy a claim that assistance was solely non-intellectual unless that is true. [Submission rules](https://iceeict.mist.ac.bd/authors/paper-submission/).

The existing `conference_final.pdf` is four pages, 225,426 bytes, Letter size, and contains author names. Its listed fonts are embedded and no Type 3 font was listed. It needs A4 and anonymity changes, followed by rebuilding and visual inspection. It is not currently a compliant ICEEICT review submission.

QPAIN 2027 is a possible later backup: its official announcement lists a 30 November 2026 deadline, 8–10 April 2027 event in Chattogram, and four-to-six-page anonymous submissions. The same page says its submission portal is opening soon, so active submission availability is not verified. [Official announcement](https://qpain.org/announcements). Submit to one venue at a time.

## Proposed schedule

| Dates, October 2026 | Deliverable |
|---|---|
| 9–11 | Agree with the supervisor on E9 scope, contribution, target, and hypothesis family |
| 12–16 | Revise scientific claims and reconcile tables; complete a bounded sensitivity analysis if feasible |
| 17–20 | Build the anonymous A4 PDF, verify tables and citations, inspect every page |
| 21–23 | Coauthor review and venue compliance check |
| 24 | Submit the approved version, allowing a day before the advertised final deadline |

This schedule is a recommendation, not a claim that the revisions or submission have occurred. If the contribution remains unclear at supervisor review, defer rather than submit inflated claims.

## Fresh verification in this assessment

- Pipeline checks passed for Glass and Tic-tac-toe.
- E12 treatment, held-out-label isolation, and metric regression checks passed.
- Six audit pytest regressions passed.
- FaithfulNB validation passed its stated checks. The reported 7,998 NSL-KDD disagreements with Weka remain visible and are not an equivalence pass.
- E9 validation recomputed all 220 summary rows from 22,000 raw fold rows, checked coverage and duplicates, and freshly reproduced all twelve arms for seed 0/fold 0 on Iris, Contact lenses, and Glass. Result-file hashes were unchanged. The validation report write was disabled for this read-only check.
- Independent E12 primary inference was recomputed from saved raw confusion matrices.
- Existing PDF page count, dimensions, size, font types, and embedding were inspected.

No model code, paper source, saved experiment results, branches, or commits were changed. Only this assessment file was added; remote tracking refs were refreshed. Full experiment reruns, a fresh PDF compilation/visual review, exhaustive current-literature novelty verification, submission portal login, and conference approval confirmation were not performed. Those limits prevent a final submission-ready or guaranteed-publication claim.
