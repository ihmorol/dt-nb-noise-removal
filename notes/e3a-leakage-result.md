# E3a — how much of the gap to Farid (2014) is protocol?

*2026-09-18. Script: `code/leakage_check.py` · raw output and CSV:
`results/EXP-E3a_leakage_check/`. Same algorithms, same data, same encoding, same
classifiers — only the moment at which the stages are fitted changes. 10 datasets,
10-fold CV × 3 seeds, `ccp_alpha=0.01`, mixed-likelihood NB.*

| Arm | B: refit per fold (honest) | A: stages fitted once before CV | A − B |
|---|---|---|---|
| baseline NB / DT | 81.74 | 81.74 | **+0.00** (nothing can leak) |
| **Alg 1 (Farid's filter)** | 81.03 | **96.16** | **+15.13** |
| **Alg 2 (Farid's selection)** | 76.44 | 76.89 | **+0.45** |
| Path 1 (N→A) | 80.83 | 96.81 | +15.97 |
| Path 2 (A→N) | 81.03 | 98.04 | +17.01 |

A third protocol (stages *and* classifier fitted on all data) reaches 100 % on the trees —
not a protocol anyone would report, kept only as an upper bound.

## What this says

1. **The instance filter's number is mostly a protocol artifact.** Fitting the filter once
   on the whole dataset raises Algorithm 1 by 15 points on average, and then this code
   matches or beats the paper's reported Algorithm 1 on 9 of 10 datasets (iris 98.63 vs
   98.66; tic-tac-toe 96.72 vs 88.10; NSL-KDD 99.90 vs 81.92). So the honest numbers are
   not a bug — they are what the method does when the test folds are not pre-cleaned.

2. **The mechanism, in one line of maths.** The leaky protocol deletes the set H of rows
   the judge gets wrong from the *whole* dataset, so the test folds become D \ H. Expected
   inflation ≈ (|H|/|D|) · (error on H − error on D\H). H is exactly the region where the
   recorded label disagrees with the NB posterior (boundary, rare and noisy rows), so
   error on H is far larger than on D \ H, and inflation must be double-digit whenever
   |H|/|D| is not tiny. Our |H|/|D| is 4–47 %.

3. **Attribute selection barely leaks (+0.45)** — because its honest effect is also ~0. So
   the Alg-2 gap is *not* protocol; see §5 of `e9-pipeline-results.md` for candidates
   (one-hot vs nominal attribute space is the first thing to test).

## The honest-protocol numbers, for the record

- The filter costs the tree 4.8 points on average (Alg1→DT 80.98 vs plain DT 85.76).
- Its deletion is class-biased: worst class-share shift 1.0–24.5 points across datasets
  (tic-tac-toe: 55 % of `negative` deleted vs 18 % of `positive`, so the minority share
  drops 34.7 % → 22.4 %).
- Across the 10 datasets, tree damage correlates with the class-share shift (−0.67), the
  deletion volume (−0.59) and the judge's own error rate (−0.57).
- The filter pays off only where the tree is weaker than the judge (breast-cancer +5.5,
  diabetes +4.1) and hurts where the tree is already strong or the judge is weak
  (tic-tac-toe −18.2, glass −17.3, NSL-KDD −13.6).
