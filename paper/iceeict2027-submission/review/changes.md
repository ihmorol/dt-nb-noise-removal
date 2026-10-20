# Revision record

This separate E9-based draft preserves the existing manuscripts and experiment outputs.

- Anonymous A4 IEEEtran, ordinary 10-point body text and unchanged template margins.
- Abstract and body revised together using the supplied final Lorenz paper as a style reference: first-person plural, direct explanations, defined symbols and bounded interpretation.
- All twelve E9 arms, including the independent parallel reference. E5/E11/E12v2 and the proposed new cleaner remain separate.
- Five displayed tables and two vector figures generated from saved records. A removal table is also included among the checking outputs.
- Original eight exploratory accuracy tests and Holm family retained. Parallel comparisons remain descriptive. No new training or significance-guided selection.
- Dataset sizes and rare classes, fold-specific macro-F1, known category vocabularies, unequal removal and the unresolved NSL-KDD replication discrepancy remain explicit.
- A/B pairing corrected: attribute-only arms retain shared folds; filtering can change the row pool and splits.
- Combined pseudocode numbered Algorithm 3, preserving inherited Algorithm 1/2 terminology.
- Embedded Times New Roman plot labels of at least 8.16 points after placement, with colorblind-safe blue/orange colors and explicit denominators.
- Ten cited references; available DOI values displayed. Citation verification limits recorded separately.
- Truthful AI-assistance acknowledgment, without asserting non-intellectual-only use or completed author approval.

Independent scientific review reproduced means, seed SDs, protocol gaps, exact tests, removal rates and attribute retention. Independent visual review inspected every page and found no clipping, overlap, unreadable labels or identity disclosure. Final figure-font adjustments were inspected again.

One underfull vertical-box warning remains from IEEE float placement, with no visible defect. There are no overfull boxes or unresolved citations/references. Terminal and native desktop compilation both succeeded. The delivered PDF is the terminal-produced file.

## Audit revision, 19 October 2026

Every number in the text, tables and figures was recomputed independently from `results/EXP-E9_farid` (means, seed SDs, exact Wilcoxon p, Holm, A/B gaps, class removal, attribute retention, safeguard and empty-selection counts); all matched. A fresh refit of seed 0/fold 0 and seed 9/fold 9 on all ten datasets reproduced all twelve arms exactly. All branches are merged into main; no unreported result affects this scope. Text changes only, no new training and no table or figure changes:

- Datasets: disclosed the third known file deviation (image segmentation 2,310 rows here vs 1,500 in the 2014 study).
- Tests: stated that zero differences are dropped, so N--DT and NA--DT vs DT use nine datasets (contact lenses ties).
- Discussion: replaced the vague "similar aggregate means elsewhere" sentence with the 2014 table averages (Alg 1 88.43, Alg 2 83.42, C4.5 83.52, NB 77.33, from `results/EXP-R1_replication/replication.csv`) against our Protocol B means; Alg 1 is 5.30 pp below C4.5 instead of above it, Alg 2 is +1.42 pp over NB instead of +6.09 pp.
- Softened "show why" to "indicate that"; removed three redundant sentences to keep five pages.

## Six pages with per-dataset line charts, 19 October 2026

At the authors' request, page 6 is filled with per-dataset line charts in the style of the 2014 paper's Figs. 3 and 4, each followed by a short explanation. An earlier expansion (macro-F1 table, deletion table, scatter and dot plot; commit 5dd7394) was withdrawn. All figures come from `scripts/generate_evidence.py`, use the 2014 dataset order, fit no models, and every quoted per-dataset number is asserted in the generator.

- Fig. 1: C4.5, Algorithm 1, NB and Algorithm 2 per dataset (Protocol B). The 2014 tables show both hybrids above their base classifiers on all ten datasets; here Algorithm 1 is below C4.5 on six (up to -17.75 pp on glass) and Algorithm 2 is above NB on seven.
- Fig. 3: the three combined designs against the uncleaned classifier, final NB and final C4.5 panels. On the four datasets with large tree losses (5.93 to 17.94 pp), the designs differ by at most 2.60 pp.
- Fig. 5: Algorithms 1 and 2 under Protocols B and A beside the 2014 values. The reported Algorithm 1 value lies between our protocols on seven datasets; the reported Algorithm 2 value exceeds both on eight.
- Colours: Okabe-Ito blue, vermillion, green, pink, with distinct markers and dash styles because two pairs sit in the colour-vision-deficiency floor band.

This is a completed conference draft under the E9 scope. Author approval, screening, CMT checks and eligibility under the conference LLM restriction remain outstanding.
