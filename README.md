# Research Workspace — Mutual DT–NB Hybrid (leakage-safe, confidence-aware)

**Working title:** *A Leakage-Safe, Confidence-Aware Mutual Hybridization of Decision Trees and Naïve Bayes Classifiers*

## Pitch (3 sentences)

Farid et al. (2014) proposed two *separate* DT–NB hybrids: NB deletes misclassified training instances before tree induction (Alg. 1), and a tree depth-weights attributes for NB (Alg. 2). No published work combines them into one mutually-reinforcing system, none addresses the cross-validation leakage created by filtering/selection before CV, and none replaces hard deletion with confidence-aware handling. This project delivers all three, evaluated on the seed paper's own 10 UCI datasets plus a leakage-safe protocol.

## Contributions (the publishable unit)

- **C1 — Replication + leakage audit.** Faithful reimplementation of both hybrids; quantify how much of the reported +4.8/+9.4 point gains comes from filtering/selection done before 10-fold CV (leakage) vs. genuinely.
- **C2 — Mutual hybrid.** One system: NB filters instances → tree grows on cleaned data → tree returns depth-based weights → weighted NB uses the same tree's structure. Show it beats each one-directional hybrid.
- **C3 — Confidence-aware instance handling.** Replace hard deletion with (a) committee/thresholded filters, (b) soft instance reweighting; report what is removed and the class distribution of removals (nobody reports this).
- **C4 — Honest evaluation.** Nested CV, repeated seeds, CIs, significance tests; Hall (2007) and WANBIA as weighting baselines; Wong et al. (2020) as the filtering baseline.

## Existing assets (do not duplicate — link)

- Seed-paper study notes: `../paper-study-notes/01-hybrid-DT-NB.md`
- Verified literature review: `../reference/literature-review-dt-nb-hybrid.md`
- Three-paper analysis + gaps: `../reference/source-paper-analysis.md`
- Prior related-work notes: `../reference/related-work-notes.md`
- All 410 citing papers: `../tmp/farid2014_citations.json`

## Status snapshot

| Phase | Status |
|---|---|
| 0 — Setup (env, data) | done |
| 1 — Replication + leakage audit | done (R1, E3a) |
| 2 — Mutual hybrid + confidence-aware filter | done (E9, E11) |
| 2b — **Which filter actually finds noise? (E12)** | **running** — `notes/e12-noise-removal.md` |
| 3 — Robustness (repeats, CIs, baselines) | not started |
| 4 — Writing + submission | not started |

Rule: update the status table and `trackers/experiments.md` at the end of every session.

## E12 in one paragraph

E11 compared noise filters only through downstream accuracy, on ten datasets that
are themselves clean — so it could never tell whether a filter removed *mislabeled*
rows or merely *hard* ones, and every delta landed inside `p >= 0.0625`. E12 fixes
that: `noise_inject.py` corrupts a controlled fraction of the **training fold's**
labels and keeps the mask, so every filter is graded on precision/recall against
ground truth. `confident_filter.py` runs DT and NB as judges **separately**, using
per-class thresholds (the confident joint) instead of E11's vote counting, plus a
per-class floor so no rule can empty a class. `e12_noise_removal.py` reports both
the detection quality and the downstream effect.
