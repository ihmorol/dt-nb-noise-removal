# Manuscript evidence map and blueprint

Date: 2026-10-03. Mode: evidence-grounded IEEE conference draft; no named venue or page limit.

## Work plan

1. Audit current Weka experiment artifacts and methods; verify relevant primary literature. Complete.
2. Recompute aggregates, exploratory tests and removal diagnostics; write research synthesis and blueprint. Complete.
3. Revise the existing manuscript in place with standard IEEE layout. Complete.
4. Independently review citations and scientific claims, polish, and check the built-in compiler. Complete at source level; compilation/render blocked by compiler runtime error.

## Frozen research questions

RQ1: Do the two specific DT/NB cleaning compositions improve on matched uncleaned and single-step classifiers?
RQ2: Does ordering change predictive performance or what gets removed, relative to independent composition?
RQ3: How does cleaning before versus within cross-validation change scores, and what can this contrast establish?

Audience: supervisor and reviewers of an empirical classification-method evaluation. Scope: the saved Weka/original-column experiment, not the older sklearn pilot, and a focused literature synthesis rather than an exhaustive systematic review.

## Evidence map

| ID | Source | Level | Supports | Cannot support | Planned use | Risk |
|---|---|---|---|---|---|---|
| E1 | results/EXP-E9_farid/metrics.csv and per_fold.csv | full data | 12 B arms, 10 A cleaning arms; classification means | unrecorded model runtime, external deployment performance | Results | saved runs, not rerun now |
| E2 | results/EXP-E9_farid/removals.csv | full data | training removal fractions, per-class loss, attribute retention | actual corruption labels, causal mediation | Results/Discussion | duplicate event counts across classes must be deduplicated |
| E3 | code/e9_farid.py, faithful_nb.py, weka_utils.py | full implementation | judge fitting, weights, NB likelihood, J48 options, guards, parallel design | exact fidelity to unpublished 2014 software | Methods | docstring/config empty-tree statement differs from code |
| E4 | code/data.py and pipeline.py | full implementation | schema, missing token treatment, folds, rare-class boundary | fully inductive vocabulary handling | Setup/Limitations | full-file nominal vocabularies |
| E5 | paper/manuscript/analysis.json and scripts/make_results_table.py | derived data | seed sensitivity, permutation signed-rank tests, joint Holm family | prospective preregistration, equivalence, population-level inference | Results | post-hoc exploratory analysis |
| L1 | Farid 2014 primary publisher/institution records and local full PDF | full text | two independent algorithms and original empirical setting | proof of original cleaning placement, exact reproduction | Intro/Related work | underspecified original software and protocol |
| L2 | Hall 2007 publisher and author working paper | full text | depth weights, bagged unpruned trees | novelty of weighting or exact match to single pruned tree | Related work | distinguish versions and implementations |
| L3 | Tsai 2013 publisher abstract | abstract | selection priorities studied using genetic algorithms | this DT/NB composition's outcomes, first-ever ordering claim | Related work | no universal absence claim |
| L4 | Brodley/Friedl 1999 and John 1995 primary PDFs | full text | valid hard cases can be discarded, different filtering procedures | actual reason for every observed error here | Discussion | conditional external evidence |
| L5 | Northcutt 2021 and Jiang 2024 primary records | abstract/full record | alternative approaches to uncertainty/correction/filtering | superiority to them without comparison | Related work | contextual only |
| L6 | Demircioglu 2021, Cawley/Talbot 2010, Kapoor/Narayanan 2023 | primary text/abstract | preprocessing leakage and evaluation concerns | isolated leakage magnitude from our population-changing A/B contrast | Intro/Discussion | external effect sizes not transferred |
| L7 | Demsar 2006 primary JMLR paper | full text | paired dataset-level inference and multiple-comparison safeguards | 100 independent CV observations, equivalence from nonsignificance | Setup | selected ten-dataset suite |
| L8 | Witten et al. 2016 book and Tavallaee et al. 2009 | primary record | Weka/C4.5 and NSL-KDD context | official held-out NSL-KDD test evaluation here | Setup | exact local config comes from code |

## Chapter blueprint

| Section | Role and judgment | Evidence | Open boundary |
|---|---|---|---|
| Abstract | bounded empirical protocol/order audit with numerical findings | E1-E5 | conference fit and render pending |
| Introduction | state three questions, no novel-weighting or improvement premise | L1-L7 | no first-ever claim |
| Related work | three themes: composition, disagreement filtering, evaluation | L1-L7 | no direct modern-method comparisons |
| Methods | operational rules and guards, weighted AN judge, independent P | E3 | latent empty-selection asymmetry disclosed |
| Setup | files, schema, shared splits, union-class F1, exploratory test family | E4-E5,L7-L8 | rare classes and global vocabularies |
| Results | full suite + per-dataset metrics, statistical uncertainty, A/B, diagnostics | E1-E5 | causal and corruption identification excluded |
| Discussion | interactions do not imply accuracy gains; protocol changes task | E1-E5,L4-L6 | mechanisms remain hypotheses |
| Limitations | exact scope and reproducibility checks | E1-E5 | no fresh full-model rerun |
| Conclusion | answer RQ1-RQ3; fixed-population and known-noise follow-ups | E1-E5 | no equivalence or general filtering verdict |

## Contribution trace

- Controlled composition comparison: Methods and classification/parallel tables; supported by saved data.
- Evaluation protocol contrast: Setup and A/B table; descriptive, not a causal leakage estimate.
- Removal diagnostics: N/A method records and removal table; no verified noise labels.

## Submission boundaries

Keep the existing author block, but confirm coauthor approval and author order with the supervisor before submission. A named conference is needed for page limit, anonymity, disclosure and template-specific checks. Built-in compiler currently fails with “Unable to find standard directories for platform”; no PDF/render verification or replacement PDF was produced.
