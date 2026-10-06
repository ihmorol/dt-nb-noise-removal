# Supervisor code guide

This guide explains the code that exists in this checkout. It is written for a reader who understands basic machine learning but may be new to this repository. The conference paper's authoritative experiment is **E9f**, implemented in [`code/e9_farid.py`](../code/e9_farid.py). It uses Weka J48, `FaithfulNB`, and the original dataset columns. The older sklearn E9 pipeline is useful engineering and sensitivity evidence, but its results must not be substituted for the conference table because it uses a different tree implementation and a different attribute representation.

The shortest accurate description of the project is:

1. Algorithm 1, called **N**, trains a Naive Bayes judge on a training set and removes the training rows that the judge misclassifies.
2. Algorithm 2, called **A**, trains a decision tree, keeps the attributes the tree tests, and gives each kept attribute a weight based on its shallowest tree depth.
3. Algorithm 3 runs both steps in a sequence. `N -> A` filters rows before selecting attributes; `A -> N` selects and weights attributes before the weighted NB judge filters rows.
4. Protocol B fits every cleaning step and final classifier inside each training fold. This is the main evaluation.
5. Protocol A cleans the complete dataset before cross-validation. It is a protocol-sensitivity comparison whose test population can differ from Protocol B.

## 1. Which code answers which question

| Experiment | Entry point | Data representation | Tree | NB | Purpose | Conference status |
|---|---|---|---|---|---|---|
| E9f | [`code/e9_farid.py`](../code/e9_farid.py) | Original columns | Weka J48 3.8.6 | [`FaithfulNB`](../code/faithful_nb.py) | Compare single steps, both Algorithm 3 orders, and a parallel control under Protocols A and B | Authoritative experiment for the current conference paper |
| sklearn E9 | [`code/main.py`](../code/main.py) | Nominal columns expanded to one-hot columns | sklearn entropy CART | [`WeightedNB`](../code/nb.py) | Earlier implementation and configuration study of the two orders | Supporting development evidence; different computational world |
| R1 | [`code/replicate_farid.py`](../code/replicate_farid.py) | Original columns | Weka J48 3.8.6 | `FaithfulNB` | Compare the four Farid systems with reported Farid values under Protocols A and B | Replication and protocol context, not the Algorithm 3 conference table |
| E5 | [`code/e5_strategy.py`](../code/e5_strategy.py) | Original columns | Weka J48 3.8.6 | `FaithfulNB` | Compare soft weighting, correction, and committee deletion with hard deletion references | Separate strategy study; not part of the current conference claim |
| E3a | [`code/leakage_check.py`](../code/leakage_check.py) | One-hot pipeline data | sklearn entropy CART | `WeightedNB` | Show how results change across Protocols B, A, and C | Diagnostic evidence only |

I think the safest supervisor explanation is to start with E9f, then use R1 to explain replication, the sklearn pipeline to explain why the implementation was rebuilt in the paper's computational setting, and E5 only if asked about future alternatives to hard deletion.

The repository also has newer E12v2 research on branch `dev/e12-research-takeover`, documented there in `notes/e12v2-protocol.md`. It is outside this checkout and outside the current conference scope. Do not merge its design or results into the E9f explanation without checking out that branch and auditing it separately.

## 2. Vocabulary used by the code

- **Judge:** the NB model inside Algorithm 1. It decides which training rows survive.
- **Final classifier:** the NB or DT trained after cleaning. It is distinct from the judge.
- **Step:** `N` or `A`.
- **Path:** an ordered tuple of steps, such as `("N", "A")`.
- **Arm:** a path plus a final classifier, such as `C1 N->A->DT`.
- **Support:** passing Algorithm 2's weights into the NB judge and final NB.
- **Fold cleaner:** code that learns cleaning decisions from one training fold.
- **Original columns:** one variable per source attribute. A nominal source attribute remains one nominal variable.
- **One-hot columns:** one binary column per nominal value. A source attribute may become many model columns.

The distinction between a judge and a final classifier is essential. In `Alg1->DT`, NB judges the training rows, then J48 is the evaluated classifier. A bad test score does not mean the NB judge itself was scored as the final model.

## 3. The mathematical pieces

### 3.1 Plain and weighted Naive Bayes

For a class `c` and row `x`, both NB implementations compare a log score of this form:

```text
score(c | x) = log P(c) + sum_j W_j log P(x_j | c)
```

The predicted class is the class with the largest score. Plain NB uses `W_j = 1` for every attribute. Algorithm 2 supplies non-uniform weights.

For a nominal attribute with `K_j` possible values, `FaithfulNB` uses add-one smoothing:

```text
P(x_j = v | c) = (count(x_j = v, class = c) + 1) / (N_c + K_j)
```

For a numeric attribute, it uses a Gaussian likelihood with a positive variance floor to avoid division by zero, including when every numeric column is constant:

```text
log P(x_j | c) = -0.5 [log(2 pi variance_jc)
                       + (x_j - mean_jc)^2 / variance_jc]
```

[`code/faithful_nb.py`](../code/faithful_nb.py) implements these rules on original attributes. [`code/nb.py`](../code/nb.py) implements the earlier pipeline model on one-hot data. In its `mixed` mode, binary columns use Bernoulli probabilities and other columns use Gaussians.

### 3.2 Algorithm 1: instance filtering

Algorithm 1 performs these operations on training data only:

```text
fit NB judge on training rows
predict the labels of those same training rows
keep row i when prediction_i == observed_label_i
train the final classifier on the kept rows
```

Small example:

```text
observed labels:  [A, A, B, B, B, A]
judge predicts:   [A, B, B, B, A, A]
keep mask:        [T, F, T, T, F, T]
surviving row ids:[0, 2, 3, 5]
```

Rows 1 and 4 are deleted from the training fold. Test rows are never passed through this deletion rule under Protocol B.

The code adds a safety rule that is not a claim from the seed algorithm: if deletion would leave fewer than two classes, it cancels the deletion. E9f implements this in `filter_step()` in [`code/e9_farid.py`](../code/e9_farid.py). The sklearn path implements it in [`code/algorithm1.py`](../code/algorithm1.py).

The code records rows as "misclassified by the NB judge." Calling every such row true label noise would be too strong because a correct but difficult boundary case can also be misclassified.

### 3.3 Algorithm 2: attribute selection and weighting

Algorithm 2 grows a decision tree and finds the smallest depth at which each attribute is tested. The root has depth 1. For attribute `j`:

```text
W_j = 1 / sqrt(d_j)   if the tree tests attribute j
W_j = 0               if the tree never tests attribute j
```

Small example:

| Attribute | Smallest tree depth | Weight |
|---|---:|---:|
| age | 1 | 1.000 |
| glucose | 2 | 0.707 |
| mass | 4 | 0.500 |
| identifier | not tested | 0.000 and dropped |

E9f obtains the depths by parsing J48's printed tree with `tree_depths()` in [`code/weka_utils.py`](../code/weka_utils.py). The sklearn path walks the fitted `DecisionTreeClassifier.tree_` arrays in [`code/algorithm2.py`](../code/algorithm2.py).

The same formula therefore has two different attribute spaces:

- E9f, R1, and E5 weight original source attributes.
- sklearn E9 weights one-hot model columns. A single nominal source attribute may contribute several separate binary columns.

This is the main reason their numerical results are not interchangeable.

### 3.4 Algorithm 3: the two sequential orders

`C1 N->A` runs Algorithm 1 first:

```text
raw training fold
  -> plain NB judge removes misclassified training rows
  -> tree grows on the surviving rows
  -> tree selects and weights attributes
  -> final NB or DT fits on surviving rows and selected columns
```

`C2 A->N` runs Algorithm 2 first:

```text
raw training fold
  -> tree selects and weights attributes
  -> weighted NB judge evaluates rows in that selected space
  -> misclassified training rows are removed
  -> final NB or DT fits on surviving rows and selected columns
```

The order can change both the rows and columns because the second step sees the output of the first. That dependence is the interaction being studied. The code does not prove that either order must improve predictive performance.

### 3.5 The parallel control

E9f also evaluates a parallel arm:

```text
raw training fold -> plain NB selects rows
raw training fold -> J48 selects columns and weights
intersection of selected rows and selected columns -> final NB or DT
```

Neither cleaning step sees the other's output. Comparing a sequential arm with this parallel arm helps describe whether the sequential interaction changes the outcome. It does not by itself establish a causal mechanism.

## 4. E9f, the authoritative conference path

### 4.1 Entry point and dependencies

[`code/e9_farid.py`](../code/e9_farid.py) imports:

- `available()` and `load_original()` from [`code/data.py`](../code/data.py)
- `FaithfulNB` from [`code/faithful_nb.py`](../code/faithful_nb.py)
- `make_folds()` from [`code/pipeline.py`](../code/pipeline.py)
- ARFF, J48, tree parsing, and name-cleaning helpers from [`code/weka_utils.py`](../code/weka_utils.py)
- CSV writing from [`code/tables.py`](../code/tables.py)

The Python process writes temporary ARFF files. `run_weka()` starts Java with the bundled jars in [`code/lib/`](../code/lib/). J48's test accuracy, printed tree, and final confusion matrix are parsed back into Python.

### 4.2 Current arms

The current `ARMS` list contains **12**, despite the older module text saying ten:

| Arm | Rows used | Columns used | Final model |
|---|---|---|---|
| `baseline->NB` | raw fold | all | FaithfulNB |
| `baseline->DT` | raw fold | all | J48 |
| `Alg1->NB` | plain-NB survivors | all | FaithfulNB |
| `Alg1->DT` | plain-NB survivors | all | J48 |
| `Alg2->NB` | raw fold | J48-selected | weighted FaithfulNB |
| `Alg2->DT` | raw fold | J48-selected | J48 |
| `C1 N->A->NB` | plain-NB survivors | tree selected after N | weighted FaithfulNB |
| `C1 N->A->DT` | plain-NB survivors | tree selected after N | J48 |
| `C2 A->N->NB` | weighted-NB survivors after A | tree selected before N | weighted FaithfulNB |
| `C2 A->N->DT` | weighted-NB survivors after A | tree selected before N | J48 |
| `parallel->NB` | plain-NB survivors from raw fold | tree selected from raw fold | weighted FaithfulNB |
| `parallel->DT` | plain-NB survivors from raw fold | tree selected from raw fold | J48 |

### 4.3 One Protocol B fold, function by function

`main()` loads `X`, `y`, and metadata with `load_original()`. For every seed and fold, it calls `run_fold()`.

Inside `run_fold()`:

1. `_j48(..., "b1", ...)` trains J48 on the raw training fold and scores `baseline->DT` on the untouched test fold.
2. `tree_depths()` turns the printed tree into `{attribute_name: minimum_depth}`.
3. `select_attributes()` maps those names back to original column indices and computes weights.
4. `_nb()` plus `nb_scores()` scores `baseline->NB` and `Alg2->NB`.
5. A separate J48 trained on selected columns scores `Alg2->DT`.
6. `filter_step()` uses weighted NB on selected columns to produce the `C2 A->N` row set. The two C2 finals are then scored.
7. Another `filter_step()` uses plain NB on all columns to produce the Algorithm 1 row set. `Alg1->NB` and `Alg1->DT` are scored.
8. A tree on the Algorithm 1 survivors supplies the C1 columns and weights. The two C1 finals are scored.
9. The already computed plain-NB rows and raw-tree columns are intersected for the two parallel finals.

The code reuses a tree result when Algorithm 1 removes no rows. With the parallel arm present, the current implementation performs up to **six** scored J48 calls per fold, although a stale comment says five.

### 4.4 What reaches the test fold

For every Protocol B arm:

- row deletion is learned only from `train` and affects training rows only;
- column selection is learned only from `train`;
- the same selected columns are applied to `test`;
- all original test rows receive predictions;
- `y[test]` is used only after prediction to compute metrics.

This is the key leakage-safe contract.

### 4.5 Protocol A in E9f

`run_protocol_a()` first fits the cleaning steps on the full dataset. It then cross-validates only the final classifier over the already cleaned row pool and fixed column set.

For an Algorithm 1 arm, rows deleted by the full-data judge are absent from both training and test folds. Protocol A therefore evaluates a selected population. Protocol B evaluates all held-out cases. Their score difference combines several changes and must not be described as a pure causal leakage estimate.

The full-data parallel files under [`results/EXP-E9_farid/versions/`](../results/EXP-E9_farid/versions/) are inspection artifacts. They are not the source of Protocol B scores.

### 4.6 Scoring and output

Accuracy is:

```text
accuracy = 100 * total correct predictions / total test predictions
```

For each class, macro-F1 uses:

```text
precision_c = TP_c / predicted_as_c
recall_c    = TP_c / actually_c
F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
macro-F1    = mean of F1_c over classes present in truth or predictions
```

Zero divisions contribute zero. For J48, `checked_scores()` recomputes accuracy from Weka's confusion matrix and raises `ValueError` if it differs from Weka's printed accuracy by more than 0.001 percentage points. This check still runs when Python is started with optimization enabled.

E9f writes:

- [`results/EXP-E9_farid/config.json`](../results/EXP-E9_farid/config.json): exact run settings
- [`results/EXP-E9_farid/metrics.csv`](../results/EXP-E9_farid/metrics.csv): mean accuracy and macro-F1 by dataset, arm, and protocol
- [`results/EXP-E9_farid/per_fold.csv`](../results/EXP-E9_farid/per_fold.csv): every fold score
- [`results/EXP-E9_farid/removals.csv`](../results/EXP-E9_farid/removals.csv): step-level row and attribute counts plus class-specific removal rates
- `results/EXP-E9_farid/versions/<dataset>/parallel.csv`: one full-data parallel pass for inspection
- `results/EXP-E9_farid/versions/<dataset>/removals.txt`: what that inspection pass removed

## 5. Dataset loading

[`code/data.py`](../code/data.py) defines the ten dataset specifications, their label columns, dropped identifier columns, and the dimensions reported in Farid's table.

`read_table(name)` reads source files and separates attributes from labels. `load_original(name)` then returns:

```text
X       numeric NumPy matrix
y       string class labels
meta    names, nominal flags, category levels, class count, expected shape
```

Nominal values are stored as integer category codes in `X`, but `meta["levels"]` preserves the mapping back to text. Those codes are meaningful only with their metadata.

`load_data(name)` starts from `load_original()` and one-hot encodes nominal attributes. It is used by the sklearn pipeline and E3a. The categorical vocabulary is constructed before cross-validation from the complete dataset. This exposes the set of category levels, though it does not use class labels. It is another reason to describe E9f as the conference-authoritative implementation.

`available()` includes a dataset when all listed raw files exist. Iris has no external file and is loaded from sklearn, so its empty file list passes this check.

## 6. FaithfulNB and Weka helpers

### 6.1 FaithfulNB

[`code/faithful_nb.py`](../code/faithful_nb.py) keeps nominal variables intact. Its main methods are:

- `fit(X, y, sample_weight=None)`: learns class priors, nominal log tables, and numeric Gaussian parameters;
- `class_scores(X)`: returns one log score per row and class;
- `predict(X)`: selects the largest score;
- `accuracy(X, y)`: computes percentage accuracy.

E5 uses `sample_weight`. Weighted nominal counts can become fractional, and weighted numeric means and variances use the same row weights.

The numeric variance term is at least `1e-12`, and `class_scores()` raises `ValueError` if any score is nonfinite. These guards prevent constant numeric columns from silently producing NaN predictions.

### 6.2 Weka bridge

[`code/weka_utils.py`](../code/weka_utils.py) contains the boundary between Python and Java:

- `clean_name()` converts names to the restricted form written into ARFF and parsed from trees;
- `write_arff()` writes original columns, class labels, and optional instance weights;
- `run_weka()` runs the Java command with a ten-minute timeout;
- `run_j48()` trains and tests J48 with `-C 0.25 -M 2` unless `unpruned=True`;
- `confusion_matrix()` parses the last printed Weka matrix;
- `tree_depths()` finds each tested attribute's minimum depth;
- `weka_predicted()` extracts row-level predictions;
- `naive_bayes_mistakes()` compares Weka NB predictions with labels.

Temporary ARFFs are removed automatically when each script's `TemporaryDirectory` closes.

## 7. The sklearn E9 pipeline

[`code/main.py`](../code/main.py) is the older/general pipeline entry point. It calls `load_data()`, so nominal variables become one-hot columns. Its decision tree is sklearn's entropy `DecisionTreeClassifier`, which is CART-style binary splitting rather than Weka J48/C4.5.

### 7.1 Core objects

[`code/pipeline.py`](../code/pipeline.py) defines:

- `Settings`: pruning strength `alpha`, whether weights support NB, and NB likelihood;
- `Cleaned`: cleaned training arrays plus original row indices, original column indices, weights, and removal records;
- `Cleaned.apply_to_test()`: applies selected columns to test data while retaining all test rows;
- `make_folds()`: shuffled stratified 10-fold splits controlled by a seed;
- `clean()`: applies an ordered sequence of `N` and `A` steps;
- `make_classifier()`: creates `WeightedNB` or sklearn DT;
- `run_fold()`: cleans one training fold, fits a final model, and predicts its test fold;
- `run_arm()`: repeats one arm across seeds and folds and averages metrics and removal records.

`clean()` maintains two coordinate systems. `X` and `y` shrink as steps run, while `rows` and `columns` remember which original training-fold positions remain. This is why a later removal record can still identify original rows and columns.

### 7.2 The sklearn algorithm modules

[`code/algorithm1.py`](../code/algorithm1.py) fits `WeightedNB`, returns a Boolean keep mask, and applies the two-class safety rule.

[`code/algorithm2.py`](../code/algorithm2.py) fits an entropy tree with `ccp_alpha`, walks the tree to find shallowest test depths, and returns kept columns plus weights. If the tree is a single leaf, this pipeline keeps every column at weight 1 so later code never receives a zero-width matrix.

[`code/nb.py`](../code/nb.py) provides the pipeline's NB. `mixed` mode treats every observed 0/1 column as Bernoulli. This includes one-hot indicators and any numeric column whose training-fold values happen to be only 0 and 1.

### 7.3 Output and inspection traces

[`code/versions.py`](../code/versions.py) runs each path once on the whole dataset and saves human-readable before/after tables. Those files explain what was removed; they are not cross-validated evaluation results.

The pipeline outputs live in [`results/EXP-E9_pipeline/`](../results/EXP-E9_pipeline/). Files with different names may represent different settings or historical runs. Always read each CSV's setting columns or its paired log before comparison.

## 8. R1 replication path

[`code/replicate_farid.py`](../code/replicate_farid.py) evaluates four systems:

- `C4.5`: J48 on all original columns;
- `NB`: plain `FaithfulNB`;
- `Alg1`: plain `FaithfulNB` judges training rows, then J48 fits survivors;
- `Alg2`: J48 supplies original-column depths, then weighted `FaithfulNB` predicts.

Protocol B refits everything inside each fold. Protocol A fits Algorithm 1 or 2 once on all data, then cross-validates the final classifier. The CSV includes the code's score, Farid's reported score, and their difference. A difference is replication evidence, not proof of why the numbers differ.

[`code/verify_faithful.py`](../code/verify_faithful.py) checks four implementation anchors before R1:

1. numeric `FaithfulNB` predictions equal sklearn `GaussianNB` on iris;
2. `FaithfulNB` and Weka NB agree on all-nominal datasets;
3. the J48 tree parser finds iris `petal_width` at root depth;
4. the Play-Tennis judge deletes one training row.

## 9. E5 graded-treatment path

E5 is implemented in [`code/e5_strategy.py`](../code/e5_strategy.py). It uses Protocol B only and fits three judges on each raw training fold:

- plain `FaithfulNB`;
- pruned J48;
- sklearn `LogisticRegression`, after `StandardScaler` is fit on the training fold.

The treatments are:

### Soft weighting

For training row `i` with observed label `y_i`:

```text
q_i = P_NB(y_i | x_i)
```

No row is deleted. `q_i` becomes a fractional count in `FaithfulNB` or an ARFF instance weight for J48.

Example: if the judge assigns probabilities `{A: 0.7, B: 0.3}` and the observed label is `B`, the row receives weight `0.3`.

### Committee deletion

The row is deleted only when at least two of the three judges misclassify it. The final evaluated model for this treatment is J48.

### Correction

The label changes to `c'` when at least two judges predict the same `c'` and `c'` differs from the observed label. The row remains in the training fold.

### Attribute support

The raw-fold J48 tree supplies Algorithm 2 weights for `attr->NB` and `softattr->NB`. The latter combines soft row weights with tree-derived attribute weights.

E5 writes aggregate metrics, fold metrics, per-class treatment counts, and judge diagnostics under [`results/EXP-E5_strategy/`](../results/EXP-E5_strategy/). Its hard-deletion reference comes from matching E9f folds rather than recomputing a duplicate hard arm.

E5 can test treatment strategies. It does not establish that NB errors are true noise, that a corrected label is ground truth, or that judge independence holds.

## 10. Evaluation protocols and what can be claimed

### Protocol B: refit per fold

For each seed and fold:

```text
split original dataset into train and test
fit cleaner on train only
transform train according to learned rows and columns
transform test with learned columns only
fit final classifier on cleaned train
predict every test row
score predictions against test labels
```

This estimates performance on held-out cases from the original dataset population, subject to the usual limits of repeated cross-validation.

### Protocol A: clean before CV

```text
fit cleaner once on the complete labeled dataset
remove selected rows and columns
cross-validate final classifier on the cleaned dataset
```

The cleaner has seen information from rows that later appear in test folds. For Algorithm 1 arms, removed rows are also absent from evaluation. Use Protocol A to show protocol sensitivity, not as the main generalization estimate.

### Protocol C: fit and score on all data

E3a's Protocol C cleans, fits, and scores on the same complete dataset. It is an intentionally optimistic diagnostic and not a generalization estimate.

### Repeated folds

`make_folds(y, seed)` uses `StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)`. Ten seeds produce 100 fold scores per dataset and arm. Folds from different seeds overlap, so the 100 scores should not be described as 100 independent datasets.

## 11. Verification commands

Run these commands from the repository root in Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
java -version
Set-Location code
..\.venv\Scripts\python.exe data.py
..\.venv\Scripts\python.exe check_pipeline.py
..\.venv\Scripts\python.exe verify_faithful.py
..\.venv\Scripts\python.exe -m pytest -q test_audit_regressions.py
```

The Java-dependent verification should pass before running R1, E9f, or E5.

Small E9f smoke run:

```powershell
Set-Location code
..\.venv\Scripts\python.exe e9_farid.py iris --seeds 1 --out-dir ..\results\EXP-E9_farid_smoke
```

The output path must be absent or empty. Choose a new path for each smoke run. The script refuses to overwrite a nonempty directory, including the committed conference result directory.

Full conference E9f run:

```powershell
Set-Location code
..\.venv\Scripts\python.exe e9_farid.py --out-dir ..\results\EXP-E9_farid_rerun
```

Use a new empty name for the rerun. After verification, compare it with the committed [`results/EXP-E9_farid/`](../results/EXP-E9_farid/) artifacts before any deliberate promotion.

Other experiment entry points:

```powershell
..\.venv\Scripts\python.exe replicate_farid.py
..\.venv\Scripts\python.exe main.py
..\.venv\Scripts\python.exe leakage_check.py
..\.venv\Scripts\python.exe e5_strategy.py
```

Use the command recorded in each experiment log and inspect its generated `config.json` before treating existing outputs as reproducible evidence.

## 12. What the automated checks establish

[`code/check_pipeline.py`](../code/check_pipeline.py) checks the sklearn fold-cleaner contract on glass and tic-tac-toe:

- changing test labels does not change predictions;
- every test row gets a prediction;
- selected training columns are applied to test data;
- unknown step and classifier names raise `ValueError`.

These are focused invariants. They do not validate every dataset, the Java bridge, statistical tests, paper tables, or the scientific interpretation.

[`code/verify_faithful.py`](../code/verify_faithful.py) validates selected NB and J48 anchors. It does not prove exact equivalence between `FaithfulNB` and Weka NB for mixed numeric-nominal datasets.

The E9f accuracy assertion checks parser consistency. It does not independently validate Weka's classifier.

## 13. Current limitations and audit cautions

1. **E9f comments contain stale counts.** The executable `ARMS` list has 12 arms, not ten. The parallel arm makes the maximum six scored J48 calls per Protocol B fold, not five. Use the arm list and executed functions as the source of truth.

2. **Empty-tree behavior is an explicit edge rule.** When J48 tests no attribute, standalone `Alg2->NB` uses no attribute evidence and therefore predicts from class priors. A chained path keeps every column at weight 1 so later steps receive a usable matrix. Protocol B computes separate standalone and chained selections to preserve this distinction.

3. **Class vocabularies are dataset-level metadata.** ARFF headers and nominal level mappings are created from the complete dataset. This keeps train and test schemas compatible, but a strict deployment simulation would derive vocabularies from training data and define unknown-category handling.

4. **Algorithm 1 detects disagreement, not verified noise.** A row can be hard, rare, mislabeled, or outside NB's assumptions. Removal rates should be described as judge-disagreement rates.

5. **Protocol A changes the evaluated population for row-filtering arms.** Its differences from Protocol B combine information reuse, row selection, and changed folds.

6. **The two E9 implementations are different estimators.** Weka J48 versus sklearn CART and original columns versus one-hot columns can each change trees, depths, selected attributes, NB likelihoods, and final predictions.

7. **The code has safety rules beyond the paper description.** It cancels one-class deletion. Chained empty trees keep all columns at weight 1. These rules should be disclosed because they affect edge cases.

8. **Small classes make 10-fold estimates unstable.** The code suppresses sklearn's warning for contact-lenses to preserve the paper's ten-fold protocol. Some test folds can have very few members of a class, which makes fold macro-F1 volatile.

9. **Repeated-CV fold scores are dependent.** Statistical procedures must respect pairing and repeated observations. Reading `n_folds = 100` as 100 independent samples would overstate evidence.

10. **E5 judges can share errors.** Majority arguments that assume independent judge errors do not automatically apply to three models trained on the same fold.

11. **Output protection differs by entry point.** E9f accepts `--out-dir` and refuses to write into a nonempty directory. R1, sklearn E9, E3a, and E5 still use fixed result paths, so run their pilots in a disposable checkout when preserved outputs matter.

12. **The old root README is stale.** [`README.md`](../README.md) still labels phases as not started, while code and result artifacts exist. Use [`code/README.md`](../code/README.md), the experiment configs, the current manuscript, and the actual scripts for operational status.

## 14. A concise supervisor walkthrough

Open these files in this order:

1. [`code/e9_farid.py`](../code/e9_farid.py): show `ARMS`, then `run_fold()`, then `run_protocol_a()`.
2. [`code/faithful_nb.py`](../code/faithful_nb.py): show nominal smoothing, Gaussian likelihoods, attribute weights, and prediction.
3. [`code/weka_utils.py`](../code/weka_utils.py): show ARFF generation, J48 invocation, tree-depth parsing, and the confusion-matrix check.
4. [`code/data.py`](../code/data.py): show original-column loading and contrast it with one-hot `load_data()`.
5. [`code/pipeline.py`](../code/pipeline.py): show the general fold-cleaner contract and `apply_to_test()`.
6. [`results/EXP-E9_farid/config.json`](../results/EXP-E9_farid/config.json): establish the settings of the committed conference run.
7. [`results/EXP-E9_farid/per_fold.csv`](../results/EXP-E9_farid/per_fold.csv) and [`results/EXP-E9_farid/removals.csv`](../results/EXP-E9_farid/removals.csv): connect aggregate claims to fold-level evidence and class-specific deletion behavior.

The defensible final explanation is that the project implements Farid's two cleaning directions in both sequential orders, evaluates them inside training folds, and records how rows and attributes change. The code supports comparison and protocol-sensitivity claims. Performance improvement, true-noise identification, causal explanations, and broad generalization require separate evidence and should not be inferred from the algorithm names.
