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
| 0 — Setup (env, data) | not started |
| 1 — Replication + leakage audit | not started |
| 2 — Mutual hybrid + confidence-aware filter | not started |
| 3 — Robustness (repeats, CIs, baselines) | not started |
| 4 — Writing + submission | not started |

Rule: update the status table and `trackers/experiments.md` at the end of every session.
