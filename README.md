# DT–NB research: current state

The Cline work has been integrated into `dev/e12-research-takeover`.
E12v2's fixed evaluation is complete: ten datasets, three seeds, ten outer
folds, two synthetic noise mechanisms, four noise levels, and NB/DT/LR finals.
The separate deletion and small MLP-transfer checks are also complete.

**Result:** conservative NB+DT correction gains 2.52 macro-F1 points at 20%
added noise on average, but the primary Wilcoxon test is **p=0.084**. This does
not establish overall improvement. Gains concentrate in the final decision
tree (+5.64 points); glass and tic-tac-toe lose performance. Original labels
are unverified references, so synthetic-noise findings do not establish
real-world mislabel detection or methodological novelty.

## Main artifacts

- [Human report](to_human/e12-report.html), [result figure](to_human/e12-results.pdf)
- [Findings](findings.md), [locked protocol](notes/e12v2-protocol.md)
- [Research state](research-state.yaml), [decision timeline](research-log.md)
- [Independent implementation audit](notes/e12v2-audit.md)
- [Main raw results](results/EXP-E12v2_main/)
- [Deletion sensitivity](results/EXP-E12v2_delete/)
- [Exploratory MLP transfer](results/EXP-E12v2_mlp/)
- [Code and run commands](code/README.md)

## Strategy and verification

Each training row is scored by out-of-fold NB and CART judges. Correct a
label only when both judges clear their per-class confidence thresholds and
propose the same replacement. Preserve all rows during correction and protect
small classes. The filters never see reference training labels or corruption
masks. Every test row is scored. Metrics pool outer-fold predictions per seed;
inference uses datasets, not folds, as independent paired units.

The inherited E12 relabel and AUROC pilot was invalid. It remains archived
for provenance and is excluded from the new analysis. All new run domains,
row counts, code hashes and dataset fingerprints have been validated.
Rare-class CV warnings are retained and documented; no convergence or numeric
failures were observed.

## Earlier research

E9/E5: combining hard-deletion and attribute stages did not beat the plain
tree; pre-CV cleaning inflated filter-based scores. Committee handling was
less damaging than single-NB deletion. E11 added committee cleaning with
LR/MLP finals. Historical decisions and experiment records remain in
`DECISIONS.md` and `trackers/experiments.md`. The sklearn/CART pipeline and
Weka/J48 replication are distinct implementations; do not conflate them.
