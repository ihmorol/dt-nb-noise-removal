# Literature review (draft, ≤1,200 words)

*Sources: `paper/manuscript/references.bib` (20 entries, all re-verified 2026-09-25 against
Crossref/JMLR/primary PDFs). F1–F4 refer to this paper's findings: (F1) fitting the NB
instance filter before CV inflates accuracy ~13–15 points, attribute selection before CV
barely matters; (F2) the seed's hybrid gains do not replicate when everything is refit
per fold, while plain C4.5/NB baselines do; (F3) a single-model NB filter deletes up to
~47% of training rows and hurts a downstream tree; (F4) composing the two algorithms in
either order gives only small differences.*

## DT–NB hybrids and the seed

Farid et al. [1] is the seed itself: Algorithm 1 fits a naive Bayes model, deletes every
training instance it misclassifies, and builds a C4.5-style tree on what remains;
Algorithm 2 builds a tree first, converts each attribute's minimum test depth into a
1/√d weight, and folds it into naive Bayes as an exponent. It reports large 10-fold
gains but never states whether the filter or weights are fit inside or outside each
fold — the exact ambiguity this replication resolves, and the source of F1 and F2. Kohavi's NBTree [10] is the canonical DT–NB
architecture — a tree whose leaves hold local naive Bayes models rather than a
sequential filter-then-tree or tree-then-weight pipeline — and sets the bar for what a
genuinely architectural hybrid gain looks like, which is the gain F2 finds missing once
leakage is removed. Hall's tree-based attribute-weighting filter [2] derives almost the
same 1/√d-style formula as the seed's Algorithm 2, independently and earlier, but the
seed never cites it; because Hall's filter is a mild, monotone reweighting rather than a
row-deletion step, its existence corroborates F1's finding that attribute selection
barely moves accuracy compared with instance filtering. Quinlan's C4.5 book [15] and the
Witten–Frank–Hall–Pal data-mining text [16] are the tool references for the tree
induction and Weka-style baselines used as the "plain C4.5/NB" comparators in F2. Wong,
Yang, and Chen [4] build a later instance-filtering hybrid that outperforms the seed's
original design on benchmark accuracy, showing that filter-based hybrids can carry a
real, non-leaked advantage under some designs — a qualifier on F2 and F4, since it
implies the seed's specific composition, not hybridization in general, is what fails to
replicate. Zhang, Jiang, Zhang, and Li's multi-view attribute-weighted naive Bayes [11]
replaces the seed's single tree-derived weight with weights aggregated across many
random trees; its existence shows the field moved away from the seed's exact "one tree
feeds one naive Bayes" design, which frames why no later paper repeats or defends the
original composition that this paper re-examines.

## Instance filtering and attribute weighting

Brodley and Friedl [3] is the classical audit of noise-filtering designs: a filter that
uses a single classifier to judge and discard its own training errors — exactly
Algorithm 1's design — is shown to be the most aggressive and least reliable filter
compared with consensus or majority-vote filters across several learners. This is the
direct explanation for F3: a lone naive Bayes model, with no second opinion, has no
check against over-deleting rows it happens to fit poorly. John's robust decision trees
[19] is the earlier ancestor of that same "delete what the model gets wrong" loop, and
John's own experiments already showed it degrading accuracy on domains with little
genuine label noise (e.g., Tic-Tac-Toe) — the earliest demonstration of the failure mode
F3 re-observes decades later on the seed's ten-dataset benchmark. Tsai, Eberle, and Chu
[13] use a genetic-algorithm search to show that the relative priority given to feature
selection versus instance selection changes which combination wins, i.e., that ordering
two preprocessing knobs is not innocuous in general — direct precedent for testing
order, which motivates F4's ablation even though this paper's own order effect turns out
small. Sáez, Luengo, Stefanowski, and Herrera's SMOTE-IPF [14] is a sharper example of
the same principle: filtering before resampling strips borderline minority instances
that resampling would otherwise have used, producing a large order-dependent swing under
class imbalance. Read against F4, SMOTE-IPF shows order effects can be large elsewhere,
so this paper's finding that the seed's two operations compose almost symmetrically is a
substantive (not merely default) result rather than an assumption confirmed.

## Evaluation leakage and statistics

Cawley and Talbot [5] is the canonical general result behind F1: any selection step —
feature selection, tuning, or by extension an instance filter or attribute weight —
fit outside the evaluation loop rather than inside each fold produces an optimistically
biased error estimate; nested cross-validation is the prescribed fix. Kapoor and Narayanan [6] situate this as one instance of a broader,
modern "leakage and reproducibility crisis" across ML-based science, giving F1's specific
preprocessing leakage a named category and a citable umbrella. Demircioğlu [7] supplies
the closest published magnitude analog to F1: in radiomics, applying feature selection
before rather than inside cross-validation inflates AUC-ROC by up to 0.15 and accuracy
by up to 0.17, with the bias growing as the samples-per-feature ratio shrinks — a
different domain from this paper's tabular benchmarks, but the same mechanism and a
similar order of magnitude to the ~13–15 point inflation F1 reports. Bouke and Abdullah
[8] show the same preprocessing-leakage effect concretely on NSL-KDD and UNSW-NB15
intrusion-detection data; because NSL-KDD is one of the seed's own ten benchmark
datasets, this is the most directly relevant prior demonstration that F1's leakage
mechanism operates on data this paper actually reuses. Jiang, Zhang, Bai, Wang, and Meng
[9] show that, for cleaning label noise, correction outperforms filtering at the dataset
level because filtering tends to "overclean" — relevant context for F3, since it frames
the seed's pure-deletion Algorithm 1 as a comparatively blunt instrument next to
correction-based alternatives, without itself being a leakage result. Northcutt, Jiang,
and Chuang's confident learning [12] offers the principled counterpart to Algorithm 1: a
confidence-thresholded, noise-rate-aware method for identifying which labels to prune,
in contrast to Algorithm 1's single-pass, single-model deletion rule — again context for
why F3's ~47% deletion rate is extreme rather than typical of principled filters. Demšar
[17] supplies the multi-dataset statistical comparison protocol (Wilcoxon signed-rank
and Friedman/Nemenyi) used to judge whether the gaps behind F2 and F4 are
statistically distinguishable rather than noise across the ten benchmark datasets.
Pedregosa et al.'s scikit-learn [18] is the toolkit reference for the `Pipeline`/
`cross_val_score` machinery used to implement the strictly within-fold refitting that
produces the F1 and F2 comparisons. Tavallaee, Bagheri, Lu, and Ghorbani [20] introduce
NSL-KDD itself, the cleaned KDD Cup 99 successor that is one of the seed's ten datasets
and the same dataset on which Bouke and Abdullah [8] independently demonstrate
preprocessing leakage.

## Gap this paper fills

- No prior work quantifies, for Farid et al.'s specific hybrid (NB-filter-then-tree;
  tree-then-weighted-NB), how much of its reported advantage over baselines was leakage
  from fitting the filter or the attribute weights before rather than inside
  cross-validation — the general leakage literature [5], [6], [7] is domain- or
  method-general, never applied to this hybrid.
- The leakage-magnitude literature that exists is either in a different domain
  (radiomics [7]) or on a different task family (intrusion detection [8]); none refits a
  DT–NB hybrid end-to-end per fold and separately isolates instance-filter leakage from
  attribute-weighting leakage the way F1 does.
- No study composes the seed's two algorithms in both orders under one controlled,
  leakage-free protocol on the seed's own ten-dataset benchmark to check whether
  composition order matters at all once leakage is removed — the order-effect
  precedents [13], [14] exist for other filter/selector pairs, not for this one, which is
  what F4 supplies.
