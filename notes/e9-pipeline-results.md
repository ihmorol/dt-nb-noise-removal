# The refactored pipeline: results, verification, interpretation

*2026-09-18. Code: `code/data.py`, `code/algorithm1.py`, `code/algorithm2.py`, `code/main.py`.
Run: `python code/main.py --alpha=0.0 --nb=mixed --support=on --out=metrics_final_alpha0_mixed.csv`.
Raw output: `results/EXP-E9_pipeline/run_final_alpha0_mixed.txt`. Sweeps: `results/EXP-E9_pipeline/sweep_*.{csv,txt}`.
10 datasets (the seed's Table 5, all downloaded — see `data/README.md`), stratified 10-fold CV × 5 seeds,
everything refit inside the training fold.*

## 1. What the pipeline does

```
load_data(name)  ->  X, y
  Path 1:  Alg 1 (NB deletes instances) -> Alg 2 (tree deletes attributes) -> new data -> NB, DT
  Path 2:  Alg 2 (tree deletes attributes) -> Alg 1 (NB deletes instances) -> new data -> NB, DT
```

The two algorithms are the paper's; each carries one small extension so they can support
each other: Alg 2 returns its tree-depth weights `W_j = 1/√d_j` and Alg 1 uses them as its
noise judge (and the final NB uses them too), while Alg 2's tree is grown on Alg 1's cleaned
instances. Reference arms (no cleaning, single stage) run in the same folds, so any
difference is attributable.

## 2. Verification (before trusting any number)

| Check | Result |
|---|---|
| `WeightedNB(weights=None)` vs `sklearn.GaussianNB` | identical predictions, max log-probability difference 1.1e-13 |
| Alg 2 weights against a hand-read iris tree | root attribute (petal width, depth 1) → 1.0; depth 3 → 0.577; depth 4 → 0.5; untested → 0. Correct |
| Internal consistency | single-stage `Alg1` and path `C1` delete the same % of rows on every dataset (they must — same judge, same data); `C2` differs, which is the interaction being measured |
| Play-Tennis (paper Table 3) | Gaussian-on-one-hot judge deletes 5/14 rows, **all from one class (56 % of Yes)**, and misses the contradictory instance; the Weka-style Bernoulli judge deletes 1/14. The mixed likelihood is the faithful reading |
| Farid Table 11 vs its own text | the printed Alg-2 column averages to **83.42**, while the paper's text claims **86.7** (+9.4). The Alg-1 claim (+4.8 over C4.5) *does* match its table (88.43 vs 83.52). Verify against the PDF before citing the Alg-2 headline |

## 3. Headline results — final configuration (accuracy %, macro-F1 in brackets)

| Cell | Accuracy | Macro-F1 |
|---|---|---|
| C1 N→A →NB | 76.15 | 70.62 |
| C1 N→A →DT | 80.92 | 75.78 |
| **C2 A→N →NB** | **77.73** | 71.44 |
| **C2 A→N →DT** | **81.04** | 74.64 |
| baseline NB | 77.69 | 72.84 |
| baseline DT | **85.76** | **82.02** |
| Alg 1 alone →NB / →DT | 76.38 / 80.98 | 72.05 / 76.07 |
| Alg 2 alone →NB / →DT | 77.79 / 85.81 | 71.95 / 82.08 |

Per dataset, C2→DT wins 4 of 10; C2→NB, C1→NB and C1→DT win 2 each.

## 4. Against Farid (2014) Tables 8–11, same 10 datasets

| Classifier | Farid table mean | This run | Delta |
|---|---|---|---|
| C4.5 | 83.52 | 85.76 | +2.24 |
| Algorithm 1 (hybrid DT) | 88.43 | 80.98 | **−7.45** |
| NB | 77.33 | 77.69 | +0.36 |
| Algorithm 2 (hybrid NB) | 83.42 | 77.79 | **−5.63** |

Per-dataset: the plain NB reproduces almost exactly (NSL-KDD 76.35 vs 76.27; vote 90.11 vs
90.11; breast-cancer 70.73 vs 71.67) once the NB handles nominal attributes the way Weka
does. C4.5 is also close (better on NSL-KDD and tic-tac-toe, worse on breast-cancer). The
**two hybrids are where the replication fails**: Alg 1 loses 7.5 points and Alg 2 5.6.

## 5. Interpretation — four findings

1. **The attribute stage is real but nearly free of consequence.** Alg 2 removes 26 % of
   attributes on average (up to 63 % on NSL-KDD) yet `Alg2→DT` = 85.81 vs a plain tree's
   85.76: a tree simply re-derives the splits it needs from the surviving attributes. For
   NB the selection is a wash (77.79 vs 77.69).
2. **The instance filter is the dangerous stage, and its damage tracks the judge's own
   error rate.** It deletes 20.7 % of training rows on average (4 % iris → 47 % glass).
   Where the judge is weak, its "noise" is mostly signal: glass (NB = 46 % accurate, deletes
   47 % of rows) collapses the tree from 68.0 to 51.4; tic-tac-toe (NB = 69 %, deletes 30 %)
   from 94.0 to 74.3; image-segmentation from 96.8 to 89.7. Where deletion is mild (iris 4 %,
   vote 7 %, soybean 8 %) the tree is unharmed or slightly better. This is the
   Brodley–Friedl / correction-beats-filtering evidence showing up in the seed's own method.
3. **Order matters and attributes-first (C2) is the better order**: +1.6 points for NB and
   +0.1 for DT on the means, and C2 wins most head-to-head cells — consistent with the
   earlier pilot (+1.6 / +2.9).
4. **The mutual interaction is real and invisible in accuracy.** The same Alg-2 stage keeps
   74 % of attributes on raw data but only 46 % after Alg-1 cleaning (breast-cancer 81 % →
   20 %; tic-tac-toe 91 % → 63 %; vote 40 % → 10 %): cleaned instances yield a smaller tree,
   so the two algorithms genuinely feed each other. It just does not pay off in accuracy.

**Bottom line:** under a leakage-safe protocol with the same algorithms, neither of the
seed's reported gains reproduces. The chained paths (81.0 best) do not beat the plain tree
(85.8); the honest story is the diagnostics and the audit, not a headline gain — which is
exactly what `e9-defensibility-check.md` predicted for the effect size.

## 6. What was tuned, and what is "optimal"

Marginal changes, all inside the theme (same two algorithms, same two paths, NB/DT finals),
3 seeds × 10 datasets:

| Configuration | 4 headline cells (mean) | Farid 4 arms (mean) |
|---|---|---|
| **α=0.0, mixed NB, weights on** | **78.96** | **80.58** |
| α=0.005, mixed NB, weights on | 78.55 | 80.44 |
| α=0.01, mixed NB, weights **off** | 78.31 | 80.53 |
| α=0.01, mixed NB, weights on | 78.03 | 80.24 |
| α=0.02, mixed NB, weights on | 77.67 | 80.21 |
| α=0.01, Gaussian NB, weights on | 76.49 | 78.87 |

- **NB likelihood is the one lever with a real effect** (+1.5 core, +1.7 Farid arms; NSL-KDD
  NB 48.5 → 76.4, breast-cancer 57.4 → 70.7). Binary/one-hot columns get a Bernoulli
  likelihood, the rest Gaussian — that is what the paper's Weka NB does.
- **Tree pruning (α)** is monotone but small: less pruning is better. α = 0 keeps 74 % of
  attributes instead of 55 % and is best on both summary columns, but the spread across
  α ∈ {0, 0.005, 0.01, 0.02} is ~1.3 points — treat as noise, not a finding.
- **Handing Alg 2's weights to the NB judge/final NB** (the "support" extension) changes the
  mean by ≈0.3 points — no measurable effect. The paper's `1/√d` weighting is a wash here.
- Chosen configuration: **α = 0.0, mixed likelihood, weights on**; the best *cell* inside the
  theme is **C2 A→N →DT** (81.04).

## 7. Caveats

Smoke-test grade: 5 seeds (the frozen protocol wants 10), no significance testing yet (E7),
one-hot encoding for nominal attributes (Alg 2 selects among dummies, not among the original
attributes as Weka did), sklearn CART ≠ Weka J48, image-segmentation is the full 2310-row UCI
set (Table 5 says 1500) and KDDTrain+_20Percent carries 22 of the 23 labels. The Farid
comparison assumes their protocol is 10-fold CV over the same data, which the leakage audit
(E3a) still has to test properly.

## 8. The three data versions and the removal reports (added 2026-09-18)

Every run now writes, under `results/EXP-E9_pipeline/`:

| File | What it holds |
|---|---|
| `versions/<dataset>/main.csv` | the data as loaded, one row per original row id |
| `versions/<dataset>/path1__Alg1_then_Alg2.csv` | new data after Path 1 (fewer rows **and** columns) |
| `versions/<dataset>/path2__Alg2_then_Alg1.csv` | new data after Path 2 |
| `versions/<dataset>/removals.txt` | per algorithm: counts, per-class removal rates, the removed row ids, the removed attribute names |
| `stages.csv` | the same information for the cross-validated arms, averaged over the folds |
| `versions_removals.csv` | the whole-dataset pass in one machine-readable table |

The versions come from one pass over the whole dataset (so a removal list and a file can be
compared directly); the metrics above are computed inside the CV folds, where the same steps
are refit on each training fold. Example, tic-tac-toe: Path 1 removes 297 of 958 instances
(31.0 %, and 55.4 % of them from the `negative` class) and then 11 of 27 attributes; Path 2
removes 5 attributes first and then 282 instances (29.4 %), leaving 676 × 22.

The code was re-split into one-purpose files the same day (`nb.py`, `pipeline.py`,
`versions.py`, `tables.py`, thin `main.py`); re-running the identical configuration produced
byte-identical `metrics.csv` and `stages.csv`, so nothing about the numbers changed.

Figures for the paper draft: `results/EXP-E9_pipeline/figures/` holds recreations of the
seed's Figures 2 and 3 (C4.5 vs Alg 1; NB vs Alg 2) with our Path-2 hybrid added as a third
bar, plus an all-classifiers chart. Generated by `code/figures.py`; captions and the
protocol caveat are in that folder's README.

## 9. Next steps

1. E3a leakage audit on this machinery: fit the stages once on the full dataset, then CV —
   measure the inflation against these numbers.
2. E7 significance battery (Wilcoxon + corrected resampled t across the 10 datasets) before
   claiming C2 > C1 or mixed > Gaussian.
3. The filter fix belongs to E5 (threshold/committee/soft weights) — finding 2 is its
   motivation, and glass alone justifies it.
