# Code ↔ pipeline mapping

*2026-09-18. How the working code maps to the E9 pipeline diagram (`diagrams/e9-pipeline.pdf`). One rule of reading: **every box in the diagram is one function or one list entry in the code.** The reference implementation is the smoke test `pilots/e9_small_test.py` (working, ran on 7 datasets); the final project will move these pieces into `code/src/` without changing the logic.*

## 1. The master mapping (diagram box → code)

| Pipeline element (diagram) | Code | File:line | What it does |
|---|---|---|---|
| **Training fold (90 %)** | `StratifiedKFold(10, ...)` then `X[tr], y[tr], X[te], y[te]` | `e9_small_test.py:62-64` | Slices one fold. Stratified, so rare classes appear in every fold. |
| **N box — NB noise filter (Alg 1)** | `fit_transform_N(X_train, y_train)` | `:51-58` | Fits GaussianNB on the fold's training rows → marks rows it misclassifies → drops them. Returns the kept fraction (diagnostic). Safety: if dropping would wipe a class, the stage is skipped for that fold. |
| **A box — DT attribute selection (Alg 2, hard form)** | `fit_transform_A(X_train, y_train, X_test)` + `tested_attributes(tree)` | `:41-49` + `:36-39` | Fits an entropy tree (`ccp_alpha=0.01`) → reads `tree_.feature` ("which column does each node test?") → keeps the tested columns, drops the rest. Slices **train and test** with the same column list. If no column survives: no-op. |
| **Box order: C1 vs C2** | the `stages` list: `["N","A"]` = C1, `["A","N"]` = C2 | `SYSTEMS :84-95`, loop `:65-72` | Runs the boxes in the given order on the same fold, feeding each stage the previous stage's output. |
| **New data: surviving rows × columns** | the variables `Xa, ya` after the stage loop | `:65-72` | What is left: rows not deleted by N, columns not dropped by A. |
| **Classifier: NB or DT** | `final` parameter → `GaussianNB()` or `DecisionTreeClassifier(criterion="entropy")` | `:73-75` | Fits only on New data. |
| **Predict held-out test fold** | `clf.predict(Xt)` | `:76` | `Xt` has had only *column* drops applied (A); test rows are never deleted. |
| **Result: accuracy · macro-F1** | `accuracy_score(y[te], pred)`, `f1_score(..., average="macro")` | `:77-78` | Per fold; averaged per seed, then across seeds (`:86-88`). |
| **Diagnostics: removal %, survivors %** | `remove_rates`, `survive_rates` lists | `:61, :66-71` | Logged per stage per fold, averaged at the end. |
| **Per-class removal (the E6 figure)** | `per_class_removal(X, y)` | `:137-146` | Fraction removed per class (fold-averaged) — the class-bias check. |
| **Rule: refit per fold (no leakage)** | everything sits inside the `for tr, te` loop | `:63-78` | Nothing is fitted outside a training fold. |
| **Rule: paired folds across systems** | the same `seed` drives every system's split | `run_system :60`, `main :158-165` | All systems see identical folds → fair, paired comparison. |
| **Baselines / single-stage reference arms** | `[]`, `["A"]`, `["N"]` entries in `SYSTEMS` | `:84-95` | No-cleaning and one-box arms for attribution. |

## 2. The two combinations, in code order

- **C1 — instances → attributes:** `stages = ["N", "A"]` → call order per fold: `fit_transform_N` → `fit_transform_A` → classifier fit → predict.
- **C2 — attributes → instances:** `stages = ["A", "N"]` → call order per fold: `fit_transform_A` → `fit_transform_N` → classifier fit → predict.

The `SYSTEMS` list is the whole system set in one screen:

```python
SYSTEMS = [
    ("baseline NB",     [],         "NB"),
    ("baseline DT",     [],         "DT"),
    ("single A->NB",    ["A"],      "NB"),   ("single A->DT", ["A"], "DT"),
    ("single N->NB",    ["N"],      "NB"),   ("single N->DT", ["N"], "DT"),
    ("C1 N->A ->NB",    ["N", "A"], "NB"),   ("C1 N->A ->DT", ["N", "A"], "DT"),
    ("C2 A->N ->NB",    ["A", "N"], "NB"),   ("C2 A->N ->DT", ["A", "N"], "DT"),
]
```

The loop `for s in stages:` (`:65`) is literally the "two boxes, any order" of the diagram.

## 3. One fold traced in numbers (iris-style)

| Step | Code line | Value in this example |
|---|---|---|
| Split | `:63` | 135 train rows, 15 test rows, 4 columns |
| N (Alg 1) | `:51-58` | NB gets ~4 % wrong → drops ~5 rows → 130 rows remain |
| A (Alg 2) | `:41-49` | Tree tests 3 of 4 columns → keeps 3, drops 1 (train **and** test lose it) |
| New data | `:65-72` | 130 rows × 3 columns |
| Fit classifier | `:75` | NB or DT trained on those 130 × 3 |
| Predict | `:76` | 15 test rows × 3 columns → labels |
| Score | `:77-78` | accuracy and macro-F1 for this fold |
| Repeat | `:60` | 10 folds × 5 seeds, then averaged |

## 4. Where the rules live (checklist)

| Rule | Enforced at |
|---|---|
| No fit outside the training fold | everything inside `:63-78` |
| Test rows never deleted | N is called only on `(Xa, ya)`; `Xt` is untouched by N (`:65-72`) |
| Same columns dropped from train and test | `fit_transform_A` returns both slices from one `keep` list (`:41-49`) |
| Never wipe a class | safety check in `fit_transform_N` (`:56-58`) |
| Identical folds for all systems | same `seed` for every `run_system` call (`:158-165`) |
| Diagnostics always logged | `remove_rates` / `survive_rates` (`:61, :66-71`) |

## 5. From smoke test to the real project tree

| Final file (from `code/README.md` plan) | Will contain | Smoke-test equivalent |
|---|---|---|
| `code/src/data.py` | loaders for the 10 seed datasets + Table-5 shape asserts | `load_lenses`, `load_breast_cancer_286`, `load_tic_tac_toe`, `dataset_table` (`:98-135`) |
| `code/src/filters.py` | the N box (noise filter; later confidence-aware variants) | `fit_transform_N` (`:51-58`) |
| `code/src/weighting.py` | the A box (column selection) **and, in Part 2, the `1/√d` weighted NB** | `fit_transform_A` + `tested_attributes` (`:36-49`) |
| `code/src/evaluate.py` | CV loop, seeds, metrics, significance helpers | `run_system` + `main` loops (`:60-95, :149-170`) |
| `code/run_E9.py` | the one script per experiment, writes results | `e9_small_test.py` as a whole |
| `results/EXP-E9_.../metrics.csv` | one row per dataset × system × seed | printed table + `e9_small_test_output.txt` |
| Part 2 (E10) | weighted NB replaces the `clf` slot; weights computed inside the fold | new `WeightedNB` class (sketch in `phase1-code-explained.md` §9) |

The refactor is mechanical: cut the functions out, give them the file names above, add the loader asserts and the results-writing. The logic — and the mapping to the diagram — does not change.
