# Strategy — the mutual hybrid, implementable on this device

*2026-09-17. Companion to `deep-research-mutual-hybrid.md` (the evidence base; bracketed numbers below cite its reference list). Feasibility targets: your laptop, Python 3.13 + scikit-learn, solo ≈10 h/week, datasets ≤ 25,192 rows (NSL-KDD).*

## 0. The one-paragraph strategy

Build the mutual hybrid as a **grid, not a single algorithm**: an instance-handling stage (F) feeds a tree that feeds an attribute-weighted NB (W), every stage refit inside every CV fold. The evidence says the winning arm is unlikely to be the seed's hard deletion — it is likely a **committee/thresholded, confidence-aware filter** (survey §5: single-model filters are the weakest design [3]; filtering over-cleans relative to correction [32]; deletion biases minority classes [36]–[38]) combined with **tree-depth weights computed on cleaned data** — a condition no weighting study has ever tested (survey §6, §10.3). The leakage audit and the order ablation (filter-first vs weight-first vs one-pass) are the two cheap, unclaimed micro-contributions that make the paper defensible even if the final accuracy deltas are small.

## 1. The method family (what to actually implement)

### 1.1 Instance-handling arms (F) — all use NB posteriors, none use a bare misclassification flag

| Arm | Rule | Evidence anchor |
|---|---|---|
| **F0** none | train on raw data | baseline |
| **F1** seed Alg 1 | delete every instance NB misclassifies | the seed [1] — the arm to beat |
| **F2** committee | consensus / majority vote of NB + entropy-tree + 1-NN; delete only on agreement | Brodley & Friedl design [3]; ensemble-filter argument [25] |
| **F3** confidence threshold | delete only if NB posterior for the instance's label < τ; **per-class τ** tuned on inner folds | confident learning's confidence framing [29]; Sarker's threshold [28]; imbalance correction [36] |
| **F4** soft reweight | keep all instances; weight each by NB posterior confidence (or 1−p_noise); feed as sample_weight to tree and NB | AAAI-2024 correction>filtering [32]; boosting analog [21] |
| **F5** relabel | if posterior for *another* class > τ_hi, relabel instead of delete; delete only the residue | graded remove/relabel [31]; correction beats deletion [32] |

F2/F3/F4/F5 are all < 100 lines of sklearn each. The per-class threshold in F3 and the class-distribution diagnostics in §4 are the direct answer to the class-bias evidence [36]–[38].

### 1.2 Attribute-handling arms (W) — the tree decides, three ways

| Arm | Rule | Evidence anchor |
|---|---|---|
| **W0** none | plain NB on the (filtered) data | baseline |
| **W1** seed Alg 2 | single entropy tree; `w_j = 1/√(min depth of attr j)`, 0 if absent; exponent in NB product | the seed [1] |
| **W2** Hall-style | 10 bagged trees on 50% subsamples; average the `1/√d` weights | **closest prior — must cite and beat [2]** |
| **W3** hard removal ("your suggestion") | keep only attributes in the tree's top-k levels (k=3); delete the rest | SBC [46]; the selection-only variant of Alg 2 |
| **W4** (stretch) MI-based weights | conditional-MI weights on the same filtered data | the active 2019–2026 family [50], [54] — shows the depth scheme is chosen, not defaulted |

W1 vs W2 tells you whether the seed's single-tree shortcut costs anything; W3 vs W1 answers *your* question (hard removal vs graded weighting) — Hall's own result favors graded weighting [2], so W3 is the hypothesis arm, not the favorite.

### 1.3 The mutual systems under test

- **M1 (one-pass mutual):** F arm → tree grows on cleaned data → tree yields W weights → weighted NB. The paper's main method.
- **M2 (iterated, optional):** M1's weighted NB re-filters once more (its posteriors are better calibrated); check help vs over-cleaning.
- **Order ablation (new experiment — no controlled order study was retrieved; see survey §7 and the ledger):** F→W (M1) vs W→F (tree+weights first on raw data, then filter for NB) vs one-pass-simultaneous (F3+F4 jointly) vs F0W0. Same folds, same seeds.

## 2. The leakage-safe protocol (non-negotiable, and itself contribution C1)

1. **Everything inside the pipeline.** Filter, tree, weights, NB — one sklearn `Pipeline`; nothing is fit on data that any fold's test set can see. Variant A (paper-style): run the whole F/W transformation once on the full dataset, then 10-fold CV on the transformed data. Variant B (correct): the same pipeline refit inside every training fold. The A−B delta is the leakage audit (survey §8: expect a *measured* gap; radiomics-scale inflation of +0.15 AUC [72] need not appear on low-dimensional UCI data [74] — a small gap is still a result, NSL-KDD is where it should show [73]).
2. **10×10 repeated stratified CV**, seeds 0–9 in the config; report mean ± std per dataset.
3. **Metrics:** accuracy, macro-F1, AUROC (one-vs-rest macro), Brier. Macro metrics are mandatory — the class-bias evidence [36]–[38] hides in plain accuracy.
4. **Significance (E7):** Wilcoxon signed-rank across the 10 datasets on per-dataset means (Demšar's protocol [76]) + corrected resampled t-test for the repeated-seed variance (Nadeau–Bengio correction [75]); Friedman + Holm across *all* systems [77]; optionally baycomp-style posterior probabilities [78].
5. **No hyperparameter search on test folds.** Per-class thresholds (F3) and k (W3) tune on a nested inner split or are fixed a priori; document which.

## 3. Device feasibility (why this is a laptop project — measured 2026-09-17)

Measured profile: AMD Ryzen 5 5600G (6 cores / 12 threads), 16 GB RAM (≈2.6 GB free while working — close heavy apps during the 10×10 runs or the parallel jobs will page), Windows 11, Python 3.13.2, and numpy 2.2.4 / scipy 1.16.2 / pandas 2.3.3 / scikit-learn 1.7.2 / pytest 9.0.2 already installed (Phase-0 package step effectively satisfied; still create the venv for reproducibility). No GPU needed — nothing in the plan is deep learning.

- Largest dataset is NSL-KDD (25,192 × 41, 23 classes). NB fits in well under a second; entropy trees similar. 1-NN in F2 is the only nontrivial fit — use `sklearn.neighbors` with `algorithm='kd_tree'`; fine at 25k rows.
- Budget: ~25 systems (F×W grid + baselines) × 10 datasets × 100 folds ≈ 25k pipeline fits, each sub-second except NSL-KDD. With `joblib(n_jobs=-1)` on 12 threads this is **hours per experiment, not days**; E-experiments stay under the 10 h/week budget.
- Storage per your `results/README.md` convention: one `EXP-<id>_<name>/` per experiment — `config.json` (all seeds), `metrics.csv` (row per dataset×system×protocol×seed), `summary.md` (≤5 lines), `figures/` 300 dpi png. Nothing else.

## 4. Diagnostics as a first-class output (contribution C3's figure)

Per dataset and system: % instances removed, **per-class removal rate vs class frequency** (the imbalance-bias figure — the thing nobody in this line reports), per-class accuracy delta vs F0, and the tree's weight vector *before vs after filtering* (directly tests the mutual premise "cleaning changes attribute relevance" — survey §10.3, currently untested in the literature).

## 5. Mapping onto your existing registry (E0–E8)

| Registry | Change |
|---|---|
| E0–E2 | unchanged (loaders, baselines, faithful Alg 1/Alg 2 + Play-Tennis unit tests) |
| **E3a** | leakage audit as §2.1 (variant A vs B) — unchanged in spirit |
| **E3b (new)** | order ablation: F→W vs W→F vs simultaneous vs none, refit per fold |
| E4 | M1 mutual; add **M2 iterated** as an optional row |
| E5a–c | committee (F2), per-class threshold (F3), soft reweight (F4) — **add E5d: relabel arm (F5)** |
| E6 | deletion diagnostics as §4, incl. weight-shift analysis |
| E7 | significance battery as §2.4 |
| E8 | baselines: Wong-style 3-model routing [4] (sketch), **Hall 2007 W2** [2] (mandatory, it is the closest prior), **AIWNB-style attribute+instance weighted NB without a tree** [16] (the "both knobs, no tree" contrast that pins down what the tree adds), MAWNB-style MI weights [54] (stretch) |

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Reviewer: "Alg 2 = Hall 2007" | cite [2] on page 1 of Methods; W2 arm *is* Hall; contribution framed as the mutual premise + audit + confidence-aware handling, not the weight formula |
| CART ≠ C4.5 | state as limitation; entropy tree + honest protocol is the comparison promise (your DECISIONS.md already fixed this) |
| Over-cleaning on error-free datasets (Tic-Tac-Toe per John [23]) | expected; E6 diagnostics turn the failure into a finding (per-class evidence) |
| Synthetic-noise experiments flatter filters | if you inject noise at all, use complexity-based injection, not uniform random [33] — or skip synthetic and use the real UCI benchmark only |
| Leakage gap turns out small | frame as measurement with CI, not headline; NSL-KDD is the likely showpiece [73] |
| NB filter single-judge fragility | never ship F1 alone as "the method"; committee/confidence arms are the method, F1 is the baseline |
| NSL-KDD gotchas (41 mixed-type attrs, 23 classes, `KDDTrain+_20Percent`) | loader asserts Table 5 shape (E0); document type split for Gaussian/Categorical NB |
| Scope creep (M2 iteration, W4 MI weights, deep baselines) | each is one registry row; do not start before the previous phase's exit criteria (your ROADMAP rule) |

## 7. Execution order (fits the ROADMAP weeks)

1. **Week 1 (Phase 0):** venv, `git init` in `code/` (commit before/after every run), download the 10 datasets, loader + Table-5 shape tests (E0).
2. **Weeks 2–4 (Phase 1):** E1 baselines; E2 faithful Alg 1/Alg 2 with Play-Tennis unit tests.
3. **Weeks 4–6:** E3a leakage audit (variant A/B) **and E3b order ablation** — both reuse the same pipeline machinery, so build `pipeline.py` once: `F-transformer → tree → weight-transformer → weighted NB`.
4. **Weeks 6–10 (Phase 2):** E4 (M1, M2), E5a–d, E6 diagnostics. The paper's three figures come from here (mutual-vs-one-directional, confidence-vs-hard, class-bias of removal).
5. **Weeks 10–14 (Phase 3):** E7 stats battery, E8 baselines, 10×10 seeds everywhere.
6. **Weeks 12–16 (Phase 4):** write results-first per `paper/outline.md`; cite Hall [2], Wong [4], Latubessy [15], Zhang [16], [17] as the positioning set from the first draft.

## 8. First three files to write in `code/` (concrete starting point)

```
code/
├── src/
│   ├── filters.py        # F0–F5 as sklearn transformers (fit learns the judge; transform drops/reweights/relabels)
│   ├── weighting.py      # W1–W3: tree-depth weights + WeightedNB (predict_proba exponentiates by w_j)
│   └── pipeline.py       # make_system(F_arm, W_arm) -> Pipeline; variant A/B switch for the audit
└── tests/
    └── test_play_tennis.py   # 14-instance table; asserts Alg 1/Alg 2 reproduce the paper's Tables 2–4
```

One thin `run_E<i>.py` per registry row, fixed seeds in config, outputs to `../results/` — exactly your `code/README.md` rules, now with the survey's evidence behind every design choice.
