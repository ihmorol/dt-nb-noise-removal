# Goal: the two-part experiment program (set 2026-09-18)

*This is the frozen goal definition for the next stage of the project. It converts the research strategy (`strategy-mutual-hybrid.md`) into the user's two-part program: (1) sweep combinations of noise-instance removal and attribute removal, pick the best; (2) take the best published weighting strategy, plug it into the winning pipeline, and re-test accuracy. Registered as E9/E10 in `../trackers/experiments.md`. Do not change the rules below mid-run — that is the point of freezing them.*

## Goal statement

**Part 1 (E9).** Build a preprocessing pipeline of the form
`Data → [DT attribute-removal | NB noise-instance removal] → [same choice again] → New Data → [NB | DT] classifier`
— i.e., two configurable preprocessing slots, each performing either tree-based attribute removal or NB-based noise-instance removal, followed by a final classifier. Run **all combinations** under one protocol and compare accuracy.

**Part 2 (E10).** Take the Part-1 winning composition, plug in the **best published attribute-weighting strategy** (shortlist in §3), and test whether accuracy improves further — with the "best" chosen by a pre-registered, leakage-safe rule.

**Success criteria.** P1: the 14-system matrix (§1.2) runs end-to-end from scripts on all 10 datasets, the winner is declared by the frozen rule (§1.4), and the diagnostics (§1.5) are logged. P2: the weighting shortlist is applied to the winner, the final method is selected by the frozen inner-CV rule (§3.2), and the accuracy comparison vs the Part-1 winner is reported with significance under E7.

## 1. Part 1 — the combination sweep (E9)

### 1.1 The two primitives (frozen definitions — boxes bound to Farid's algorithms)

- **A — DT attribute removal.** Fit an entropy decision tree (`criterion='entropy'`, fixed hyperparameters in the run config) on the current fold's training data. Remove every attribute **not tested anywhere in the tree** (the seed's "Wi = 0" selection rule from Alg. 2 — the soft-weight analog is E10's job). Log the number of surviving attributes per fold.
- **N — NB noise-instance removal.** Fit the project's canonical NB on the current fold's training data (same NB as E1's baseline; fixed in the shared config), classify the training instances, and **remove every misclassified instance** (seed Alg. 1, seed-faithful hard deletion). Log removal counts and per-class removal rates.
- **Both boxes are Farid's own algorithms** (user scoping, 2026-09-18): the N box is Algorithm 1's NB noise filter; the A box is Algorithm 2's tree-based attribute selection in hard-removal form. The `1/√d` weights are **NOT dropped — they are kept for Part 2**, where Farid's exact single-tree form runs alongside the published weighting shortlist (Hall's bagged `1/√d` as closest prior).
- **Cumulative semantics.** Each stage transforms "New Data" for the next stage: N's NB judge is fit on the *current* attribute set (so after A, the judge sees fewer attributes — that interaction is the experiment), and A's tree is fit on the *current* instance set (cleaned by a previous N). Attributes removed earlier stay removed; instances removed earlier stay removed. Recorded per fold.
- **Classifier.** One of the project's canonical NB or entropy DT, trained on the post-preprocessing data (with surviving attributes only).

### 1.2 The system set (refined 2026-09-18 — two orders, boxes bound to Farid's algorithms)

**Core (the two combinations — the headline comparison):**

| ID | Pipeline | Meaning |
|---|---|---|
| C1-NB / C1-DT | N→A: NB noise filter → DT attribute selection → final NB or DT | Combination 1: instances → attributes |
| C2-NB / C2-DT | A→N: DT attribute selection → NB noise filter → final NB or DT | Combination 2: attributes → instances |

**Reference arms (kept for attribution, not part of the headline comparison):**

| ID | Pipeline |
|---|---|
| B-NB, B-DT | no cleaning (E1 baselines) |
| N→NB, N→DT, A→NB, A→DT | single-stage arms |

**Removed from the earlier 14-system draft:** `AA` and `NN` (second application saturated in the pilot — prediction P6 — and they are not part of the two combinations). Fold-level logs stay separate from the outer metrics.

### 1.3 Protocol (identical for every system — the E3a/B rule)

- Everything (both stages **and** the classifier) is refit **inside each training fold**; the test fold is never transformed with information from itself.
- Stratified 10-fold CV × 10 seeds (0–9), same folds for all systems (paired comparison).
- Metrics: **accuracy (primary, as requested)**, macro-F1 (mandatory secondary — the removal steps are known to be class-biased; see the survey), AUROC where the dataset permits.
- Diagnostics per fold: instances removed (count, fraction, per-class rates), attributes surviving after each A, stage run-times.

### 1.4 Winner rule (pre-registered — no post-hoc metric switching)

Winner = highest **mean Friedman rank on accuracy across the 10 datasets** (per-dataset mean over the 100 folds), tie-break by macro-F1 rank, then by lower std. Significance vs baselines/other systems via the E7 battery (Wilcoxon + corrected resampled t; Friedman/Holm across systems). Report the full matrix regardless of which system wins.

### 1.5 Edge cases (logged, not silently handled)

- A removes everything or the tree tests no attribute → stage is a no-op for that fold; log it.
- N removes > 50% of a fold's training data, or leaves a single class → run it anyway (Part 1 has no cap), log loudly; caps/thresholds are E5's territory.
- NN's second pass removes ~nothing (convergence) → expected; record it as the finding.

### 1.6 Out of scope for Part 1 (keeps the grid interpretable)

Confidence-aware/thresholded/soft filtering (E5), relabeling (E5d), iteration M2 (E4), and all attribute *weighting* (E10). Part 1 uses the seed-faithful hard variants only; the grid stays 14 cells.

## 2. Part 2 — the weighting upgrade (E10)

### 2.1 Shortlist — published attribute-weighting strategies (all verified in the ledger)

| ID | Strategy | What it does | Feasibility |
|---|---|---|---|
| W1 | **Hall 2007** bagged depth weights | 10 bagged unpruned trees on 50% subsamples; `w_j = mean 1/√d` (root = depth 1), 0 if absent; exponent in NB product. Farid's Alg-2 single-tree `1/√d` runs alongside as the exact seed form (kept from the base paper) | sklearn-only; **mandatory — closest prior** |
| W2 | **CFW** (Jiang et al. 2019, TKDE) | Correlation-based feature weights | sklearn-only |
| W3 | **FTAWNB** (Zhang & Jiang 2022) | Fine-tuned gain-ratio weights | sklearn-only |
| W4 | **WANBIA** (Zaidi et al. 2013) | Weights learned by minimizing negative conditional log-likelihood | scipy optimizer; ~50 lines |
| W5 | **MAWNB** (Zhang et al. 2023, TKDE) | Multi-view weights derived from random-tree views | reimplementation risk — stretch; include only if W1–W4 land cleanly |

### 2.2 Selection rule (pre-registered, leakage-safe)

- Candidate systems = **Part-1 winner composition + each W_i** with a weighted-NB final classifier (weighting strategies are NB-specific; if the Part-1 winner ended with DT, Part 2 swaps the final classifier to weighted NB — a defined transformation, stated in the results).
- The weight vectors are learned **inside each training fold** (W1–W5 are all fit-on-train procedures by construction).
- "Best" is decided by **inner-CV accuracy on the training folds only** (nested selection), averaged across datasets; the outer test set is touched exactly once for the final report of the selected system.
- Report the full outer-test ablation table for all W_i regardless of the selection outcome — the selection cannot hide a loser.
- Metrics: accuracy primary (as requested); log-loss / Brier secondary (published weighting gains are often in probability quality, not accuracy — Hall's own result).

### 2.3 Comparisons that make Part 2 interpretable

Final algorithm (winner+W*) vs: (a) Part-1 winner with plain NB final — isolates the weighting's contribution; (b) E8's standalone weighting baselines (plain NB+W_i, no preprocessing) — isolates the preprocessing×weighting interaction; (c) the seed's Alg-2 single-tree weighting — the "is Hall-plus-cleaning better than Farid's formula" question.

## 3. Deliverables & mapping

| Item | Registry | Deliverable |
|---|---|---|
| Part-1 sweep | **E9** | `results/EXP-E9_combination-sweep/`: `config.json`, `metrics.csv` (dataset×system×seed), `diagnostics.csv` (removals/survivors), `summary.md` (≤5 lines answering "which composition wins"), figures |
| Part-2 weighting | **E10** | `results/EXP-E10_weighting-upgrade/`: same shape + `selection.json` documenting the inner-CV choice |
| Relationship to existing registry | E4 | E9's winning two-stage cell becomes the "mutual hybrid" E4 is re-framed around (E4 adds the M2 iteration row only); E2's Play-Tennis unit tests must pass before E9 runs — they are the faithfulness gate for both primitives |

**Order of execution:** E0 → E1 → E2 (unit tests) → **E9 (Part 1)** → **E10 (Part 2)** → E3a/E3b audit and order-ablation rows reuse E9's machinery; E5/E7/E8 then extend the winner. Part 1's system table is also the evidence for contribution C2; Part 2 covers the C4 baseline question for the Attribute-2 lineage.

## Amendments (2026-09-18, pre-start — from the falsification review)

*The critique (`phase1-goal-critique.md`, with pilot evidence in `pilots/`) proposed five amendments. All are adopted before any E9 run; the core design is unchanged.*

- **A1 — Pin the knobs.** Tree = entropy with `ccp_alpha` fixed in the config (primary 0.01; a one-dataset sensitivity check is documented follow-up). Canonical NB and encoding = E1's choices. Report survivor-count and removal-rate distributions. *(Pilot: pruning swings attribute survival from 12→5 on a 30-attribute dataset.)*
- **A2 — Single-class safety.** N stage: if removing all misclassified rows would leave fewer than two classes in the fold, skip the removal for that fold and log it.
- **A3 — Pre-registered subgroup analysis.** Classify the 10 datasets as noisy vs clean before running (rationale documented per dataset); report the grid for both subgroups. This is the mechanism test for the N stage.
- **A4 — Fail-fast gate.** After E2, run the grid on 2 datasets (one clean, one noisy) and check survivor/removal rates against predictions P1–P2 before launching the full run.
- **A5 — Winner declaration.** Declare a champion only if it significantly beats the runner-up under the E7 battery; otherwise report a tie group.

- **R1 (2026-09-18, user refinement).** The two boxes are bound to Farid's two algorithms (N box = Alg 1's NB noise filter; A box = Alg 2's attribute selection, hard-removal form). The headline comparison is the two orders: **C1 = instances → attributes** and **C2 = attributes → instances**, each with an NB or DT final classifier. `AA`/`NN` dropped; single-stage and no-cleaning arms retained as reference arms. Diagram: `notes/diagrams/e9-pipeline.pdf` (source `e9-pipeline.tex`).

*Pre-registered predictions P1–P6 and the falsification analysis live in `phase1-goal-critique.md` §6.*
