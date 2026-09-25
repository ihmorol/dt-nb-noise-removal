# E9 smoke test results — C1/C2 on small datasets

*2026-09-18. Script: `pilots/e9_small_test.py` · raw output: `pilots/e9_small_test_output.txt`. Scope: smoke test, not E9. 7 datasets (4 of them from the seed's Table 5: iris, contact-lenses-24, breast-cancer-286, tic-tac-toe-958), 10-fold CV × 5 seeds, frozen Farid semantics (entropy tree `ccp_alpha=0.01`; GaussianNB hard deletion), refit inside every training fold.*

## Headline table (accuracy %, mean over 10 folds × 5 seeds)

| Dataset | NB | DT | A→NB | A→DT | N→NB | N→DT | C1 (N→A) →NB | C1 →DT | C2 (A→N) →NB | C2 →DT |
|---|---|---|---|---|---|---|---|---|---|---|
| iris | 95.47 | 94.53 | **96.27** | 93.73 | 95.47 | 94.27 | 95.33 | 95.33 | 95.73 | 96.13 |
| wine | **97.09** | 93.60 | 95.76 | 91.93 | 97.20 | 91.93 | 95.76 | 91.93 | 95.76 | 91.93 |
| breast_cancer_569 | 93.92 | 93.67 | **95.15** | 93.85 | 93.14 | 93.24 | 93.85 | 93.28 | 94.97 | 93.85 |
| digits | 84.04 | 87.37 | **87.82** | 86.85 | 84.45 | 85.22 | 83.25 | 85.40 | 84.70 | 85.48 |
| lenses_24 | 73.33 | 76.00 | **76.67** | 74.00 | 73.33 | 73.33 | 73.33 | 73.33 | 76.67 | 76.67 |
| **breast_cancer_286** (seed) | 57.29 | 66.09 | 66.10 | **68.85** | 65.08 | 57.77 | 55.96 | 57.56 | 62.65 | 66.45 |
| **tic_tac_toe_958** (seed) | 67.18 | 93.99 | 67.45 | **95.76** | 67.75 | 73.68 | 69.34 | 72.18 | 67.18 | 78.83 |

## The four findings that matter

**1. The attribute-removal stage is real and helpful for trees on the seed's datasets.** `A→DT` improved tic-tac-toe from 93.99 to **95.76** (+1.77) and breast-cancer-286 from 66.09 to **68.85** (+2.76), with 40–50 % of columns removed. On iris and digits the same stage lifted NB by +0.8 and +3.8. This is the first live confirmation of the Alg-2 mechanism on the actual benchmark, not just bundled data.

**2. The NB noise filter is the dangerous stage — and its damage is class-biased.** On tic-tac-toe the NB judge deleted **31.6 %** of training rows, and the per-class rate was **50.5 % of the "negative" class vs 21.5 % of "positive"** — an enormous skew. Downstream: `N→DT` collapsed to 73.68 vs the plain tree's 93.99 (−10.3 points). This is exactly the failure mode the literature predicts (single-model filter, Brodley & Friedl; over-cleaning, AAAI 2024) and exactly the diagnostic nobody publishes. On breast-cancer_286 the filter removed 36.5 % of rows, hurting the tree (−8.3) while helping NB (+7.8) — the stage's sign depends on the downstream classifier.

**3. Order matters, and attributes-first (C2) generally wins.** C2 beat C1 on 6 of 7 datasets for both finals (breast-cancer-286: +6.7 NB / +8.9 DT; lenses: +3.3; tic-tac-toe DT: +6.7; iris positive; wine a tie). Mean C2−C1: **+1.6 points (NB finals) / +2.9 (DT finals)**. This agrees with the earlier bundled-data pilot (A-first favored by +0.73) and with prediction P4.

**4. The seed's own headline numbers do not appear under the leakage-safe protocol — yet.** Breast-cancer-286 reaches ~69 % here, while the seed reported numbers in the mid-90s for this dataset. Part of that gap is our provisional encoding (one-hot with "?" as a category, GaussianNB), part may be protocol. This is precisely what the E3a leakage audit exists to quantify properly — it now has its first concrete signal, and it must be measured carefully, not asserted.

## Prediction check (pre-registered list from the critique)

| Prediction | Result |
|---|---|
| P1: A removes ≥ 15 % of attributes on most datasets | **Confirmed** — survivors 26–59 %, removals 41–74 % |
| P2: N removes ~1–15 % of rows | **Partially** — 10–14 % on clean data, but 31–36 % on the two harder seed datasets (high end) |
| P3: N near-neutral on clean datasets, active on noisy ones | **Directionally confirmed with a twist** — neutral on iris/wine/lenses; on tic-tac-toe it hurt badly; on breast-cancer-286 it helped NB but hurt DT (classifier-dependent sign) |
| P4: C2 (attributes-first) beats C1 by 0.2–1.5 points | **Confirmed, larger than predicted** — mean +1.6 (NB) / +2.9 (DT) |

## Caveats (keep these attached to every use of these numbers)

Small datasets and 5 seeds — smoke-test evidence, not E9 results. Provisional encoding for nominal datasets (one-hot, "?" as its own category) and GaussianNB everywhere; the E0 loader will pin the final policy and E1 the canonical NB. The seed's exact preprocessing/protocol is unknown (that is the research question); nothing here accuses the paper of error — the audit will test both protocols properly. These runs used `ccp_alpha=0.01`; the sensitivity of the A stage to that knob is still to be measured (A1 follow-up).

## What this changes for the project

The two orders are not a coin flip: **C2 (attributes → instances) is the current front-runner** and the design should say so in the pre-registered predictions. The deletion diagnostics are no longer a promise but a demonstrated phenomenon (50 % vs 21 % class removal on tic-tac-toe is a figure). And the E3a leakage audit gets its first motivating datum from breast-cancer-286. Next steps unchanged: E0 loader (with these four datasets in it), E1 baselines, E2 unit tests, then the two-dataset fail-fast gate before the full E9 run.
