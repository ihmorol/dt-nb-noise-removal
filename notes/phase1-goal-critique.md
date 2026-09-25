# Phase-1 goal under fire: a falsification-first review of E9

*2026-09-18. Mandate from the user: analyze the Phase-1 goal (E9 — the two-slot combination sweep), brainstorm it critically, try to disprove it, then find the evidence that supports it. Everything documented so it can be read straight through. Companion: `phase1-code-explained.md` (the Python side). Frozen goal: `goal-parts-1-2.md`. Evidence discipline: only audited sources (see `verification-ledger-2026-09-17.md`) plus new pilot experiments run today (`pilots/pilot_phase1_output.txt`).*

## 0. Verdict in one box

**Verdict: Accept with Revisions (do not skip E9; amend it first).** The sweep itself is sound and still unclaimed. The pilots show the *attribute* half of E9 is the live ingredient and the *noise-removal* half is approximately inert on clean data — which does not break the goal but changes what we must predict and measure. Three pre-start amendments are proposed in §7; none changes the user's pipeline idea, all make its conclusions harder to attack.

## 1. What E9 actually claims (so it can be attacked)

Restated as testable hypotheses — if E9 runs and these are vague, no result can refute anything:

- **H1 (mechanics).** Both preprocessing operations are implementable and do something measurable: DT attribute removal leaves a non-trivial fraction of attributes unused; NB noise removal deletes a non-trivial fraction of training instances.
- **H2 (benefit).** At least one composition improves accuracy over the matching no-preprocessing baseline (NB or DT) across the 10 datasets.
- **H3 (interaction).** Composition *order* matters: `AN` and `NA` differ beyond noise; the two operations interact (the mutual-hybrid premise).
- **H4 (the mutual claim).** A two-stage composition beats the best single-stage composition on average.
- **H5 (selection).** A global winner can be declared by the frozen rank rule with statistical support.
- **H6 (robustness).** Any gains are not an artifact of one dataset, one seed, or one surprising fold.

## 2. How this review was done

Three evidence layers, strongest to weakest:

1. **Audited literature** — the verified corpus from the deep-research wave (`deep-research-mutual-hybrid.md`, corrections in the ledger): John 1995, Brodley & Friedl 1999, Hall 2007, SBC 2003, García-Pedrajas 2014, Tsai 2013, Sáez 2015, AAAI 2024 (Jiang), Liu 2024, Wong 2020, Latubessy 2025, Zhang/Jiang/Yu 2021, and the seed + Farid 2010 extractions.
2. **Primary-source extracts** — the seed's Alg 1/Alg 2 verbatim (including the protocol underspecification) and Farid 2010's full results tables.
3. **New pilots run today** — `pilots/pilot_phase1_check.py` on four *bundled* sklearn datasets (iris, wine, breast_cancer, digits). **Scope warning: these are not the seed's 10 datasets and not results** — they test mechanics (does A do anything? how much does N delete? which signs appear?) cheaply and honestly. All pilot numbers below are labeled as such.

Severity tags (pre-submission-reviewer convention): **CRITICAL** = breaks the experiment; **MAJOR** = reviewers will ask; **MINOR** = polish.

## 3. Falsification round — eight attacks, with what survived

### F1. "The attribute-removal stage is a no-op: trees test everything." — REFUTED (by pilot)

The worry: sklearn's default tree grows until leaves are pure, so on small UCI-style data it might split on every attribute, and "remove untested attributes" would delete nothing — making the A stage decorative.

Pilot Check 1 (trees grown on full data, attributes actually used):

| Dataset | Attrs | default tree | ccp_alpha=0.01 | removed by A (default / 0.01) |
|---|---|---|---|---|
| iris | 4 | 3/4 | 3/4 | 25% / 25% |
| wine | 13 | 6/13 | 6/13 | 54% / 54% |
| breast_cancer | 30 | 12/30 | 9/30 | 60% / 70% |
| digits | 64 | 43/64 | 28/64 | 33% / 56% |

The stage is not a no-op — trees leave 25–70% of attributes unused because other attributes separate the data first. **But** the survivor count swings hard with the pruning setting (breast_cancer 12→9→5 as `ccp_alpha` rises 0→0.02), so "which tree" is an experimental knob, not a detail. See amendment A1.

### F2. "The noise-removal stage does nothing on clean data — and the benchmark is clean." — SUPPORTED (pilot + literature)

Pilot Check 2: the NB judge deletes 1.4% (wine) to 14.1% (digits) of each training fold — plenty of deletions, actually. But the *effect* of those deletions on accuracy (Check 3, mean over 4 bundled datasets, 3 seeds):

| Arm (vs its own baseline) | iris | wine | breast_cancer | digits | mean Δ |
|---|---|---|---|---|---|
| `N→NB` | 0.00 | −0.37 | −0.71 | +0.44 | **−0.16** |
| `N→DT` | +0.66 | −1.49 | −0.30 | −1.93 | **−0.77** |
| `NN→NB` | 0.00 | −0.37 | −0.76 | +0.44 | **−0.17** |

So on clean bundled data the noise stage is neutral-to-harmful, and the second noise pass changes nothing beyond the first (NN ≈ N — convergence, as the goal's edge-case note predicted). This matches the literature: John 1995 found deletion statistically indistinguishable on most datasets and *degrading* on error-free domains (his Tic-Tac-Toe/segment); AAAI 2024 proves filtering over-cleans relative to correction; Brodley & Friedl showed single-model filters are the weakest design. **Consequence:** H2 as stated ("at least one composition improves … across the 10 datasets") is too weak to be interesting and too strong to be safe — the noise mechanism should be expected to earn its keep only on genuinely noisy datasets. The seed's own noisy ones (tic-tac-toe, breast-cancer-286, and the seed's claimed gains on diabetes/NSL-KDD) are where the test lives. This becomes amendment A3 (pre-registered noisy-vs-clean subgroup analysis) and prediction P3 in §6.

### F3. "Attribute removal hurts as often as it helps — weighting is the better tool." — PARTIALLY SUPPORTED, and that is fine for Part 2

Pilot Check 3: `A→NB` = +0.66 (iris), **−1.87** (wine), +1.05 (breast_cancer), **+3.36** (digits); mean +0.80. The sign depends on the dataset, exactly as the literature warns: Hall 2007 found graded weighting (his 1/√d) beat selection-only (weights→0/1) on 12 of 14 datasets; Ratanamahatana & Gunopulos' SBC selection still beat plain NB. The pilot's split has an obvious reading: digits (64 features, much redundancy) gains hugely; wine (13 features, NB already at 97.6%) loses signal. **Consequence:** H2's single global sign is unlikely; per-dataset reporting (already in the goal) plus a redundancy-aware explanation are required. And this is precisely the case that makes **Part 2 well-aimed**: if hard removal is sign-unstable, graded weighting applied to the winning composition is the natural next test — the two-part program's design is internally justified.

### F4. "No interaction — the two stages are additive; the mutual premise is empty." — PARTIALLY REFUTED (order effects appear), fully refuted only on noisy data

Pilot Check 3, paired orders (`AN` vs `NA`, same folds/seeds):

| | iris | wine | breast_cancer | digits | mean |
|---|---|---|---|---|---|
| `AN→NB` | 95.78 | 95.89 | 94.61 | 84.01 | −0.16 |
| `NA→NB` | 95.11 | 95.92 | 93.44 | 82.90 | −0.89 |
| difference | +0.67 | −0.03 | +1.17 | +1.11 | **+0.73 in favor of A-first** |

`AN` beats `NA` on three of four datasets, by up to 1.17 points — order matters, so the operations are not independent. This is consistent with the data-reduction school's claim that instance and feature selection are "interwoven" (García-Pedrajas 2014), Tsai 2013's priority effects, and Sáez 2015's ordering effect under imbalance. **But** note what the order comparison also shows: `AA→NB` ≈ `A→NB` (second A adds ≤0.4), and `AN/NA` do not beat `A→NB` on these clean data at all. So the *mutual* claim (H4) is currently carried only by the order contrast, not by a two-stage-vs-one-stage win. On noisy datasets that could flip; that is exactly what E9-then-E3b must measure. **Status: H3 has direct pilot support; H4 is unproven and now flagged as the hypothesis most at risk.**

### F5. "Nothing will be statistically significant on 10 datasets." — UNRESOLVED, but the effect sizes now have a scale

Pilot deltas range roughly −2 to +3.5 points, dataset-dependent, from a 3-seed run on four bundled datasets — not a significance claim. The 10×10 design plus the E7 battery is the mitigation, but two honest points: (a) the seed's own reported deltas (+3.89 to +7.55 on its noisy datasets) came from an *underspecified protocol we believe leaked* — the leakage-safe deltas should be expected to be smaller (Demircioğlu-scale inflation was up to +0.15 AUC in small-sample settings); (b) with 14 systems, "no difference at the top" is a plausible outcome. Mitigation: amendment A5 (declare a winner only if it beats the runner-up with significance; otherwise report a tie group) and the pre-registered predictions below.

### F6. "The winner is selection noise — 14 systems, one dataset set." — REAL RISK, already mitigated, sharpened

The frozen rank rule (mean Friedman rank, tie-break macro-F1/std) plus the E7 significance battery covers most of it. Amendment A5 adds the "significant or tie-group" declaration rule so the sweep cannot crown a noise champion. Also note the pilot's per-dataset flip-flops (wine vs digits) predict the likely honest finding: *no single system wins everywhere* — the deliverable should be a per-dataset table with the rank summary, not a coronation. That is still a publishable result (it is the controlled comparison nobody has run).

### F7. "Implementation choices drive the result more than the composition." — REAL RISK (pilot-confirmed)

Pilot Check 1 shows pruning swings A's aggressiveness (breast_cancer 12→5 survivors across ccp settings); the NB variant and the numeric encoding will move results too. Mitigation (amendment A1): pin the tree configuration and NB variant *in the shared config before running*, report survivor-count distributions, and note that sensitivity to these knobs is future work, not hidden tuning. The seed's own kill-list already accepts "comparison under one consistent protocol" as the promise — same spirit.

### F8. "The benchmark can't show anything: saturation." — PARTIALLY SUPPORTED

NSL-KDD sits under 99%+ DR in Farid 2010's own tables; the seed reports Alg 1 at 88.3% average. Absolute numbers on the saturated datasets carry little information; differences on iris-like data are fractions of a point. Mitigation: macro-F1 (already mandatory), per-dataset reporting, the noisy subgroup analysis (A3), and interpreting relative order not absolute deltas. This is a presentation problem, not a validity problem.

## 4. Fatal-flaws audit (idea-evaluator format, early gate)

| # | Flaw candidate | Severity | Why it is not fatal / defense |
|---|---|---|---|
| 1 | **Data-refuted core?** Does existing data show the mechanism beaten? | Not triggered | No existing data covers the two-stage DT–NB composition under a leakage-safe protocol; the pilot's negatives apply to the noise stage on clean data only, and the goal survives as the controlled comparison (H1/H3 remain live). |
| 2 | **Null-result risk on the headline (H4, mutual beats single-stage)** | MAJOR | Pre-register H4 as the at-risk hypothesis; the paper's contribution already includes the audit/diagnostics, and a null H4 with a clear order effect is still a finding (C1/C2 honesty). |
| 3 | **Attribution: composition vs knobs** | MAJOR | Amend with pinned config + survivor/removal diagnostics (A1/A2). |
| 4 | **Multiple comparisons / winner's curse** | MINOR after mitigation | Rank rule + A5 significance-or-tie declaration. |

No CRITICAL flaw. Proceed with revisions.

## 5. Steel-man: the evidence that supports running E9

1. **Every published partial combination improved something.** Latubessy 2025 (instance reduction + attribute-weighted NB): +1.28%/+1.4%. Zhang, Jiang & Yu 2021 (attribute + instance weighting, one framework): "significantly outperform NB and all state-of-the-art competitors." Xu, Jiang & Yu 2019 (attribute-derived instance weights): significant over NB on 36 datasets. Farid 2010 (both operations in one loop): KDD99 U2R detection 49.2% (ID3) / 64.0% (NB) → 99.2%. Wong 2020 (keep + route instead of delete): beat Farid's own hybrid on 20 datasets. The direction the user proposes is the direction every partial result points.
2. **The attribute-removal mechanism is live in the pilot** — digits +3.4 points for `A→NB` with 56% of attributes removed; breast_cancer +1.5 for `AA→NB`. This is not a dead mechanism; it is dataset-conditional, which is a finding.
3. **The noise mechanism has a literature-backed home** — Brodley & Friedl: filtering helps up to ~30% injected noise; Smith & Martinez: principled deletion improves many downstream classifiers; the seed's own noisy datasets are in the benchmark.
4. **The controlled comparison itself is unclaimed** — five search perspectives found no work that runs the DT-removal × NB-removal composition grid under one protocol. Even a fully negative result establishes something nobody has established.
5. **The pilot already produced a sharp publishable question:** "does the noise-removal stage pay for itself on a standard benchmark, and if not, what dataset property predicts when it does?" E9 answers it.

## 6. Pre-registered predictions (falsifiable; to be checked when E9 runs)

- **P1** — A stage removes ≥ 15% of attributes on at least 6 of the 10 datasets (pilot range 25–70% on bundled data), varying with the pinned pruning.
- **P2** — N removal rate per fold lands in 1–15% for most datasets (pilot 1.4–14.1%); on NSL-KDD expect the high end.
- **P3** — N-arms are within ±0.5 points of their baseline on the "clean" datasets (iris, glass, image-segmentation, soybean, vote, contact-lenses), and improve accuracy on the noisy ones (tic-tac-toe, breast-cancer-286, diabetes) — the subgroup split is the test of the mechanism.
- **P4** — `AN` ≠ `NA` with a mean difference of 0.2–1.5 points in favor of A-first (pilot: +0.73 mean).
- **P5** — No system wins on all 10 datasets; the rank summary shows a leading group rather than a single champion.
- **P6** — The second application of either primitive (`AA`, `NN`) adds ≤ 0.5 points over the single application (pilot: saturation observed).

## 7. Proposed amendments to the frozen goal (pre-start; none changes the user's design)

- **A1 (pin the knobs).** In the shared config before any E9 run: tree = entropy, pinned `ccp_alpha` (recommend 0.01 as primary, with a one-dataset sensitivity check), NB = the E1 canonical variant, encoding fixed. Report survivor-count and removal-rate distributions. Reason: pilot F1/F7.
- **A2 (single-class safety).** In the N stage, if removing all misclassified instances would leave fewer than two classes in the training fold, skip the removal for that fold and log it. Reason: crash-proofing; the pilot uses this rule.
- **A3 (pre-registered subgroup analysis).** Split the 10 datasets into noisy vs clean *before* running (documented rationale per dataset), and report the grid separately for the subgroups. This is the mechanism test for the N stage (F2/P3).
- **A4 (fail-fast gate).** After E0–E2, run E9's grid on 2 datasets (one clean, one noisy) before the full 10-dataset run; verify A survivor counts and N removal rates look like P1/P2; only then launch the full run.
- **A5 (winner declaration).** Declare a champion only if it significantly beats the runner-up under the E7 battery; otherwise report a tie group. Reason: F5/F6.

## 8. What this means for Part 2 (short)

If the winner is an A-arm (the pilot's hint), Part 2 is aimed exactly right: replace hard selection with published weighting (Hall/CFW/FTAWNB/WANBIA/MAWNB) on the winning composition and test whether the wine-type losses disappear while the digits-type gains stay. If the winner turns out to contain the N stage, the noisy-subgroup result (A3) carries its claim — and Part 2 must be evaluated per subgroup, not just globally.

## 9. Where everything lives

- Pilot script + raw output: `pilots/pilot_phase1_check.py`, `pilots/pilot_phase1_output.txt` (re-runnable in ~2 minutes; bundled datasets only).
- Code explanation: `phase1-code-explained.md`.
- Frozen goal + these amendments: `goal-parts-1-2.md` (amendment block appended 2026-09-18).
- Evidence base: `deep-research-mutual-hybrid.md` + `verification-ledger-2026-09-17.md`.

*Wording note per the hedge ladder: "refuted/supported" apply to the pilot's narrow scope (mechanics, bundled datasets); all accuracy statements are directional signs, not results. No claim here exceeds what the evidence can carry.*
