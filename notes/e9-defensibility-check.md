# Is E9 defensible? Verdict on the refined two-combination pipeline

*2026-09-18. Requested check: after narrowing E9 to the two combinations (C1: instances → attributes; C2: attributes → instances, boxes = Farid's Alg 1 and Alg 2) plus reference arms, is this pipeline defensible and strong enough to forward? Evidence base: the falsification review (`phase1-goal-critique.md`), the pilots (`pilots/`), and the audited literature (`verification-ledger-2026-09-17.md`).*

## Verdict

**Yes, forward with it — with four conditions kept in place.** The refined design is stronger than the earlier 14-system draft for the purpose of approaching Prof. Farid: it is a crisp, two-sided question about his own two algorithms, the gap is verified, every claim is attributable (reference arms), and the protocol is stricter than the seed's. The honest risk is not validity but **effect size**: the pilot says the noise-removal stage is near-neutral on clean data and order effects are small — so the paper's strength must come from the comparison, the audit, and the diagnostics, not from a large headline accuracy jump.

## What makes it strong (the case to forward)

1. **Novelty verified.** Five search perspectives plus all 410 papers citing the seed: no work runs these two orders of these two algorithms under one protocol. Farid's own 2010 predecessor did a *different* joint design (dedupe + ML relabeling + info-gain splitting); his group then split the idea into the two 2014 algorithms and moved to NBTree. The loop his 2014 paper left open is still open.
2. **The question is two-sided and pre-registered.** "Instances first or attributes first?" with predictions (order effect ~0.2–1.5 points, pilot +0.73; noise stage earns its keep only on noisy datasets) and a pre-registered noisy/clean subgroup split. A null result is still a finding; a positive result is a clean story.
3. **Lineage fit.** C1 ending in DT is Farid's Algorithm 1 extended; C2 ending in weighted NB (after Part 2) is Algorithm 2 with an added filter. The project completes the base paper's own two branches.
4. **Attribution built in.** No-cleaning baselines + single-stage arms let any gain be assigned to the right cleaner, so reviewers cannot say "the gain is just attribute selection."
5. **Protocol stronger than the seed's.** Refit per fold (leakage-safe), paired folds, 10-fold × 10 seeds, macro-F1 mandatory, significance battery planned; the leakage audit itself is contribution C1.
6. **Diagnostics are unique.** Nobody in this line reports what the removals actually delete (per-class removal rates, survivor counts, weight shifts). That figure is publishable even if accuracy deltas are small.
7. **Feasible.** Hours per experiment on the measured device (Ryzen 5 5600G, sklearn stack already installed).

## Residual risks (honest) and the mitigations already in the design

| Risk | Evidence | Mitigation |
|---|---|---|
| N stage near-inert on clean data | Pilot: N-arms ≈ −0.16 pts mean on 4 bundled datasets; John 1995 (degradation on error-free domains); AAAI 2024 (over-cleaning) | Noisy/clean subgroup pre-registered (A3); faithful arm; report per dataset |
| Order effect small or not significant | Pilot +0.73 mean difference, 3 seeds only | 10×10 + E7 battery; tie-group declaration if not significant (A5) |
| "Just chaining two known algorithms" | — | Frame as controlled comparison + audit + diagnostics; Part 2 adds the improvement step |
| Hard removal only (no weighting in Part 1) | Hall 2007: weighting ≥ selection on most data | `1/√d` weights are kept, not dropped: Part 2 runs Farid's single-tree form + the published shortlist on the winner |
| Benchmark saturation (NSL-KDD) | Farid 2010's own tables (99 %+ DR) | Macro-F1, subgroup analysis, relative interpretation |
| Single-judge NB filter fragility | Brodley & Friedl 1999 (single-model filter = weakest) | Faithful arm only in Part 1; committee/confidence variants are E5 (stated future work) |

## Conditions (keep these or the defense weakens)

1. **Reference arms stay in every table** — they are the attribution.
2. **Pre-registered rules honored** — subgroup split and winner declaration.
3. **Fail-fast gate (A4)** — run the grid on 2 real datasets (one clean, one noisy) before the full run.
4. **Diagnostics logged per fold** — removal counts, per-class rates, survivor counts. This is the paper's unique figure.

## What would change the verdict

- If the fail-fast gate shows the N stage deleting < 1 % or > 50 % of rows on real datasets, revisit the stage definitions before the full run.
- If E7 shows everything inside noise, the paper becomes an audit + diagnostics paper (still publishable; the claim changes from "combination X wins" to "cleaning order does not matter under a leakage-safe protocol, and here is what the filters delete").

## Bottom line

Forward. For the meeting with Prof. Farid: lead with the lineage (his two algorithms, the loop his paper left open), the verified gap, and the honest protocol point; bring the diagram and the risk table so every objection has a prepared answer. The companion document is `meeting-brief-prof-farid.md`.
