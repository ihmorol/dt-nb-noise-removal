# Project audit, 5 October 2026

## Verdict

The E9f evidence supports a supervisor review of a controlled DT–NB combination and evaluation-protocol study. It does not establish improved classification, detection of true label errors, or exact replication of the 2014 results. Conference-specific submission readiness remains open because the venue, page limit, anonymity policy, and deadline have not been supplied.

The current checkout started at detached commit `54b23e1`, with an older manuscript and a stale aspirational README. A newer uncommitted manuscript existed in worktree `3f30`. Its source was brought into this checkout, its tables regenerated from this checkout's results, and its numerical conclusions independently checked. Other worktrees were left intact.

## Evidence and fresh checks

- E9f: 220 summary rows and 22,000 raw fold rows; ten datasets, twelve B arms and ten A arms, 100 folds per dataset/arm/protocol. All A/B summaries recompute from the raw records within 1e-9. No duplicate B dataset/arm/seed/fold records; scores are finite and bounded. No recorded filtering/selection safeguards triggered.
- Fresh real Weka reproduction: seed 0, fold 0 on Iris, Contact lenses, and Glass. All twelve arms reproduce both stored accuracy and macro-F1 within 1e-9. This is a bounded reproduction, not a rerun of all experiments.
- Existing `check_pipeline.py`: eight checks pass. Existing `verify_faithful.py`: all stated checks pass. Numeric agreement with GaussianNB is tested on Iris; exact Weka NB agreement is tested on the four all-nominal datasets. This does not establish equality with Weka NB on mixed/numeric data. NSL-KDD training predictions differ on 7,998 rows, making this distinction essential.
- Python syntax compilation passes for code and manuscript scripts. Runtime versions, source/data hashes, and verification logs are in `notes/audit-artifacts/`. No evidence of static type-checker coverage is claimed.
- Eight two-sided dataset-level signed-rank comparisons recomputed using exact permutation enumeration and Holm adjustment. No adjusted p-value falls below 0.05. Raw AN–DT versus plain DT p=0.013671875 becomes 0.109375 after adjustment. Statistical analysis is explicitly post hoc.
- Breast-cancer removal claims reproduce from 100 folds: plain judge removes 50.6288% recurrence and 14.5054% no-recurrence; weighted judge removes 58.9672% and 11.7195%.
- Existing experiment files retain identical SHA-256 hashes after validation.
- Full project LaTeX build succeeds using local MiKTeX with resolved citations/references and no overfull-box warnings. PDF has five pages and seven bibliography entries. Every page was rendered and inspected; text, algorithm and tables are readable. The fifth page contains references with unused space. Venue-specific final layout remains open.
- The native editor's compiler returns `Unable to find standard directories for platform`. The source remains open and editable. The existing local multi-file build is verified; the native preview is not.

## Findings resolved for review

The earlier manuscript had a methods-only abstract and almost no results discussion. The imported, regenerated version reports actual results, eight adjusted tests, class-specific deletion, rare-class limitations, vocabulary handling, and the difference between protocol sensitivity and a causal leakage estimate. Algorithm 3 now has explicit equations and numbered pseudocode.

The README's promised gains, nested CV, and state of “not started” were inconsistent with the code and results. It now states current evidence and separates historical plans from completed work. The older pilot paper, roadmap, notes and experiment tracker retain historical claims; they are not the current authority for the conference paper.

## Scientific limits that remain

1. NB misclassification is a disagreement signal, not verified noise. There are no known wrong-label annotations in E9f.
2. Protocol A changes both label access and the evaluated row population. Its score gap cannot isolate leakage or determine the original paper's protocol.
3. The benchmark is small and heterogeneous. Contact lenses has 24 rows; several datasets have fewer than ten instances in a class, and NSL-KDD has a singleton. Some folds lack classes. Repeated CV does not create new samples.
4. Category vocabularies and class headers come from the full files. Supervised estimates use training rows; unknown-category deployment is not evaluated.
5. Mean fold accuracy and fold macro-F1 differ from pooled predictions. Macro-F1 uses the union of true/predicted classes in each fold, so class sets can differ. The paper states this convention.
6. NSL-KDD C4.5 is 99.47%, versus 71.11% in the original study. Segmentation uses 2,310 rows instead of the reported 1,500; Glass has six observed classes and NSL-KDD 22. Exact original data/split reproduction remains unresolved.
7. Order comparisons bundle the implemented order with a plain versus weighted judge. They do not isolate an abstract order-only causal effect.
8. Nonsignificance is not equivalence. No universal improvement or universal harm claim is supported.
9. The E9f runner now refuses nonempty output directories and supports `--out-dir`. Other historical runners require isolated outputs.
10. Historical requirements are not fully pinned. This audit saves current versions; it does not prove the original run environment was identical.

## Latest project state beyond this checkout

The newer `dev/e12-research-takeover` branch is at `bd3e8ab` in worktree `70b9`. Its tracked checkout is clean. The primary `feat/committee-filter-lr-mlp` checkout is older at `4c39231` and contains two untracked diagram files.

An independent audit in this session ran the newer branch's pipeline checks, E12 checks, syntax compilation and stored-output validators for main/deletion/MLP. Recalculation from raw confusion matrices reproduced its primary mean gain of 2.5205 macro-F1 points and Wilcoxon p=0.083984375, with eight positive and two negative datasets. Overall superiority is not established at 0.05. This is three-seed synthetic training-label-noise correction using NB/DT/LR, not the E9f conference experiment. DT-only descriptive benefit is +5.642 points; Glass and Tic-tac-toe are harmed at the primary setting. Deletion is a one-seed sensitivity and MLP a three-dataset one-seed exploration. Original labels are reference labels, not proven clean labels. Natural-noise detection remains unanswered.

Do not combine E12's pooled metrics, one-hot/sklearn learners, synthetic noise or correction semantics with E9f's original-column J48 results. The supervisor briefing explains both tracks and recommends preserving the conference paper's coherent E9f scope.

## Reference checks and limits

Fresh primary-source checks confirmed the Farid paper's identity and separate hybrid methods, Demšar's dataset-level comparison framework, Cawley–Talbot's evaluation-bias paper, and the fourth edition's book authors/date. Some DOI pages were inaccessible to the browser, so no claim of fresh verification of every bibliography field is made. Prior ledger annotations are historical.

- Farid publisher page: https://www.sciencedirect.com/science/article/pii/S0957417413007100
- Demšar primary paper: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
- Cawley–Talbot primary page: https://www.jmlr.org/beta/papers/v11/cawley10a.html
- Witten et al. publisher page: https://shop.elsevier.com/books/data-mining/witten/978-0-12-804291-5

## Deliverables and stop condition

Use `paper/manuscript/main.tex` and its rebuilt PDF for review, `notes/supervisor-code-guide.md` for code study, and `notes/supervisor-meeting-2026-10-06.md` for the meeting. The source archive contains the actual multi-file manuscript. Audit artifacts record checks rather than claiming a flaw-free or fully rerun project. No commit, push, merge or submission has been performed.

## Code fixes and final validation

Six targeted regressions now cover constant numeric columns (weighted/unweighted), standalone empty-tree NB fallback, optimized-Python matrix integrity, and safe output refusal. All six pass. NB has a positive variance floor and rejects nonfinite scores. Standalone Algorithm 2 uses priors if no attribute is selected. The matrix integrity guard raises an explicit exception.

Existing checks and isolated three-dataset reproduction pass after these edits. The stored results retain their original byte hashes. The initial regression exercised the unsafe default pilot; its four changed files were restored byte-for-byte from the identical preserved snapshot and checked against pre-run hashes. Regression tests now use temporary directories. The complete experiment was not rerun after fixes. Stored evidence remains from commit 54b23e1; no recorded empty-tree event occurred. Original and final code hashes are preserved separately.

Combined-versus-single-step differences are explicitly descriptive means. Inferential claims cover only the eight baseline/order comparisons. Per-dataset macro-F1 is available in `paper/manuscript/tables/supplementary_dataset_metrics.csv`. Audit environment pins and jar/data/source hashes identify the current verification environment, without inferring the historical runtime.
