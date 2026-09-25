# The Phase-1 algorithm in Python, explained simply

*2026-09-18. Companion to `phase1-goal-critique.md` (the red-team review) and `goal-parts-1-2.md` (the frozen goal). This document explains, in plain words, what the Python code for E9 will look like and why each part exists. No fancy tricks — plain functions, numpy slicing, and the standard scikit-learn calls everyone uses. The pilot in `pilots/pilot_phase1_check.py` already runs this exact shape on bundled datasets; read it side by side with this doc.*

## 1. What the code has to do, in one paragraph

We have a table of data: rows = examples, columns = attributes (features), plus one label column. The pipeline has three parts: **cleaner 1** (either "drop useless columns" or "drop wrong-looking rows"), **cleaner 2** (the other one — the two orders are the comparison), and a **classifier** (Naive Bayes or Decision Tree) that learns from what's left and predicts the test rows. The code tries the two cleaning orders (instances→attributes and attributes→instances) plus the reference arms (single-stage and no-cleaning), repeats the whole thing 10 times with different train/test splits and 10 times with different random seeds, and records the accuracy. That's it.

```
raw data ──▶ [cleaner 1: A or N] ──▶ [cleaner 2: A or N] ──▶ [classifier: NB or DT] ──▶ predictions
                 (can also be skipped in the single-stage systems)
```

- **A cleaner** = "tree drops columns". Fit a small decision tree; any attribute the tree never uses gets deleted. (This is Farid's Algorithm 2 selection rule: weight 0 means gone.)
- **N cleaner** = "NB drops rows". Fit a Naive Bayes on the training rows; any training row NB gets wrong gets deleted. (This is Farid's Algorithm 1.)

Both cleaners are fit **inside each training fold** and never see the test fold. More on why in §7.

## 2. Words you'll see, in plain terms

| Word | Plain meaning |
|---|---|
| **fit** | Learn from data. `nb.fit(X, y)` = "study these examples and their answers". |
| **transform** | Apply what was learned: change the data (drop columns, drop rows). |
| **fold** | One train/test split of the data. 10-fold CV = 10 splits, each using 90% for training and 10% for testing. |
| **seed** | A number that fixes all randomness, so the same code gives the same answer every run. |
| **leakage** | Accidentally letting test information influence training. The #1 way to fake good results. |
| **accuracy** | Fraction of test rows predicted correctly. |
| **macro-F1** | Like accuracy, but every class counts equally no matter how rare. Needed because row-deletion can quietly wreck rare classes. |
| **X, y** | The usual names: `X` = table of features, `y` = the answers (labels). |
| **column / attribute / feature** | The same thing: one variable in the table. |

## 3. Cleaner 1: dropping columns the tree doesn't use (the A stage)

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def tested_attributes(tree):
    """Which columns did this fitted tree actually split on?
    sklearn stores the tree as arrays; tree_.feature[node] is the column
    used at that node, and -2 means 'this node is a leaf, no question asked'."""
    features = tree.tree_.feature
    return np.unique(features[features >= 0])


class AttributeRemover:
    """A cleaner that deletes attributes the tree never uses."""

    def __init__(self, ccp_alpha=0.01):
        self.ccp_alpha = ccp_alpha      # how much the tree is pruned (pinned in config)

    def fit(self, X_train, y_train):
        self.tree_ = DecisionTreeClassifier(
            criterion="entropy",        # the C4.5-style splitting rule from the seed paper
            ccp_alpha=self.ccp_alpha,   # pruning: smaller tree, fewer attributes survive
            random_state=0,
        ).fit(X_train, y_train)
        self.keep_ = tested_attributes(self.tree_)    # column numbers to keep
        return self

    def transform(self, X):
        """Delete every other column. Applies to train AND test rows alike
        (columns are a property of the whole dataset, so the same columns go)."""
        if len(self.keep_) == 0:
            return X                     # safety: a tree that used nothing removes nothing
        return X[:, self.keep_]          # numpy slicing: keep these columns only
```

Line by line, in words:

1. We fit a decision tree on the current training data. The tree asks yes/no questions about columns to separate the classes.
2. `tree_.feature` is the tree's "question list": each node records which column it asked about. Leaf nodes record −2 (no question).
3. `tested_attributes` collects the unique column numbers with a question. Those are the "important" columns by the seed paper's definition.
4. `transform` keeps only those columns — `X[:, self.keep_]` is numpy's way of saying "all rows, these columns".
5. The safety line: if the tree used zero columns, we keep everything instead of producing an empty table. This case gets logged (the goal's edge-case rule).

**Why this is not "feature selection" by another name:** we never score columns in advance. The tree decides which columns are worth asking about, and the rest are dropped. That's exactly the seed's Algorithm 2 idea, minus the weights (the weights come in Part 2).

## 4. Cleaner 2: dropping rows NB gets wrong (the N stage)

```python
from sklearn.naive_bayes import GaussianNB


class NoiseRemover:
    """A cleaner that deletes training rows the NB classifier gets wrong."""

    def __init__(self, nb=None):
        self.nb = nb or GaussianNB()     # the project's canonical NB (pinned in config)

    def fit(self, X_train, y_train):
        self.nb_ = self.nb.fit(X_train, y_train)
        self.correct_ = self.nb_.predict(X_train) == y_train   # True = keep this row
        return self

    def transform(self, X, y):
        """Delete the wrong-looking rows. NOTE: labels come along, so the
        method needs y as well as X — unlike the column cleaner."""
        keep = self.correct_
        # Safety: never delete rows if it would remove an entire class.
        if len(np.unique(y[keep])) < 2:
            return X, y                  # skip the removal for this fold (logged)
        return X[keep], y[keep]
```

Line by line:

1. Fit NB on the training rows.
2. Ask NB to predict those same training rows. Rows where the prediction differs from the true label are "the ones NB got wrong" — the seed paper calls these noisy.
3. `transform` keeps every other row: `X[keep]` is "these rows, all columns". The labels must be sliced identically, or row i's features would be paired with row j's answer — a silent disaster.
4. The safety check: if the wrong rows include a whole (usually rare) class, removing them all can leave a one-class training set, which no classifier can learn from. In that case we skip the removal for this fold, and the log records it. (This is amendment A2 in the goal.)

**One important detail:** the N cleaner is fitted on the *current* data of the fold. If an A cleaner ran before it, NB sees fewer columns — so its mistakes, and therefore its deletions, are different. That difference is the interaction E9 exists to measure.

## 5. The two classifiers

```python
nb_classifier   = GaussianNB()
dt_classifier   = DecisionTreeClassifier(criterion="entropy", random_state=0)
```

- **NB** = Gaussian Naive Bayes: assumes each column is a bell curve per class and multiplies the evidence. Simple, fast, the "NB" of the seed paper.
- **DT** = decision tree with entropy splitting (the C4.5-style rule, implemented as CART — a documented limitation of replicating the seed exactly).

A note on the NB choice: our datasets mix numeric and categorical columns. The E0 loader will fix one encoding (one-hot for categorical columns) and one NB variant (start with GaussianNB on the encoded table; a mixed-likelihood NB is a robustness variant, not the default). The important rule is consistency: **the same NB is used as cleaner and as final classifier in all 14 systems**, or the comparison means nothing.

## 6. Putting the blocks together: one function, 14 systems

```python
def run_one_fold(X_train, y_train, X_test, y_test, stages, final):
    """Run one system on one fold. `stages` is like ['A','N'] or [] or ['N']."""
    Xa, ya, Xt = X_train, y_train, X_test

    for stage in stages:
        if stage == "A":
            rem = AttributeRemover(ccp_alpha=0.01).fit(Xa, ya)
            Xa, Xt = rem.transform(Xa), rem.transform(Xt)   # same columns on test
        elif stage == "N":
            rem = NoiseRemover().fit(Xa, ya)
            Xa, ya = rem.transform(Xa, ya)                  # test rows are NOT removed

    clf = GaussianNB() if final == "NB" else DecisionTreeClassifier(
        criterion="entropy", random_state=0)
    clf.fit(Xa, ya)                                         # learn from what's left
    predictions = clf.predict(Xt)
    return (predictions == y_test).mean()                   # accuracy for this fold
```

Read it as a recipe:

- Start with the fold's training data and test data.
- Apply each cleaning stage in order. **A** drops columns (train and test must drop the *same* columns — otherwise the test table has different columns than the model expects). **N** drops rows, and only from the training side; test rows are never deleted, because at prediction time you don't get to throw away inputs you don't like.
- Train the final classifier on what survived.
- Predict the test fold and score it.

The systems are just this function called with different `stages` and `final` (the two core orders, plus reference arms for attribution):

| System | stages | final |
|---|---|---|
| **C1** (instances → attributes) | `['N','A']` | NB / DT |
| **C2** (attributes → instances) | `['A','N']` | NB / DT |
| Reference: single-stage | `['A']` or `['N']` | NB / DT |
| Reference: baselines | `[]` | NB / DT |

That's the whole trick: **one function, a list of stage letters, a final classifier name.** No frameworks, no inheritance trees — plain loops.

## 7. The experiment loop: cross-validation and seeds

```python
from sklearn.model_selection import StratifiedKFold

SYSTEMS = [([], "NB"), ([], "DT"), (["A"], "NB"), (["A"], "DT"), (["N"], "NB"), (["N"], "DT"),
           (["A","N"], "NB"), (["A","N"], "DT"), (["N","A"], "NB"), (["N","A"], "DT")]
# 4 core (the two orders) + 4 single-stage + 2 baselines = 10 systems

for seed in range(10):                          # 10 different shuffles
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed).split(X, y)
    for train_idx, test_idx in folds:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        for stages, final in SYSTEMS:           # same fold for every system (paired comparison)
            acc = run_one_fold(X_train, y_train, X_test, y_test, stages, final)
            results.append((dataset, "".join(stages) or "none", final, seed, acc))
```

Why it looks like this:

- **Stratified** 10-fold = each split keeps the class proportions, so rare classes appear in every fold. Important for datasets like soybean (19 classes).
- **The same folds for all 14 systems.** Comparisons are paired: system A and system B see identical data, so differences are about the method, not luck. This matters later for the significance tests.
- **10 seeds** = repeat the whole 10-fold cycle with different shuffles, so the result doesn't depend on one lucky split.
- **`fit` inside the loop.** Everything — tree, NB cleaner, final classifier — is learned from `X_train` only. This is the leakage rule: if any of it were fitted on `X` before splitting, the test fold's answers would have influenced training, and every number would be fake.

Reality check with the pilot numbers: iris has 150 rows, so each fold has ~135 training rows. Check 2 showed NB gets ~4% of them wrong → ~5 rows deleted → ~130 remain for the classifier, tested on the other ~15 rows. That's the entire mechanism, applied 100 times per system per dataset.

## 8. What the code writes out

Per the project's results convention, one folder per experiment:

- `config.json` — every fixed choice: tree settings, NB variant, encoding, seeds, the 14 systems. Written first, before running.
- `metrics.csv` — one row per (dataset, system, seed): accuracy, macro-F1, and whatever else the E7 battery needs. ~10 datasets × 14 systems × 10 seeds = 1,400 rows; the fold-level mean happens in the summary step.
- `diagnostics.csv` — one row per (dataset, system, fold): how many columns each A stage kept, how many rows each N stage deleted (and their class distribution). This is the evidence for predictions P1–P6 and the class-bias figure.
- `summary.md` — five lines answering the experiment's one question: which composition wins, by the frozen rank rule.

## 9. Part 2 preview: the weighted NB (why "weights" are just multipliers)

Part 2 keeps the winning cleaning pipeline and swaps the final classifier for a **weighted NB**: each attribute's evidence gets multiplied by an importance weight before the votes are added up.

```python
class WeightedNB:
    """Naive Bayes where each column's vote counts more or less (weights)."""

    def fit(self, X, y, weights=None):
        self.classes_ = np.unique(y)
        self.w_ = np.ones(X.shape[1]) if weights is None else np.asarray(weights)
        self.means_, self.vars_, self.priors_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.means_.append(Xc.mean(axis=0))          # per-column average for this class
            self.vars_.append(Xc.var(axis=0) + 1e-9)     # per-column spread
            self.priors_.append(len(Xc) / len(X))        # how common this class is
        return self

    def predict(self, X):
        # Standard trick: work in logs. log(a*b) = log(a) + log(b),
        # so multiplying probabilities becomes adding their logs.
        scores = np.zeros((len(X), len(self.classes_)))
        for k, c in enumerate(self.classes_):
            log_p = -0.5 * (np.log(2 * np.pi * self.vars_[k])
                            + (X - self.means_[k]) ** 2 / self.vars_[k])   # per column
            scores[:, k] = np.log(self.priors_[k]) + (log_p * self.w_).sum(axis=1)
        return self.classes_[scores.argmax(axis=1)]
```

In words: every column gives each class a score (how typical is this value for that class). The weight multiplies that score — importance 2 means the column's opinion counts double; importance 0.1 means it barely counts. Then the scores are summed per class, and the class with the highest total wins. Setting every weight to 1 gives ordinary Gaussian NB. The published strategies (Hall's `1/√depth`, correlation weights, fine-tuned weights, learned weights) differ only in **where the numbers in `self.w_` come from** — and in Part 2 they will be computed inside each training fold, never on the test set.

## 10. Common mistakes checklist (the ways this silently breaks)

1. **Fitting before splitting.** Any `fit` outside the fold loop = leakage = fake results.
2. **Deleting test rows.** N removes training rows only. Test rows are sacred.
3. **Forgetting to move the labels with the rows.** `X[keep]` and `y[keep]` always together.
4. **Columns out of sync between train and test.** Apply the same `keep_` column list to both.
5. **Wiping a class.** The A2 safety rule; log it when it triggers.
6. **Different folds per system.** Same `train_idx`/`test_idx` for all 14, or the comparison is unpaired.
7. **Reporting accuracy alone.** Add macro-F1; the deletions can quietly hurt small classes.
8. **Unpinned randomness.** Seeds and tree parameters fixed in `config.json` before running.
9. **Assuming numeric.** GaussianNB needs numbers — one-hot encode categorical columns in the loader, once, identically for every system.
10. **Reading a win from one dataset.** The pilot already shows sign flips (wine vs digits); conclusions come from the whole table and the rank rule, not from the best cell.

## 11. Glossary (repeat, so this document stands alone)

- **Attribute** — a column of the table.
- **Instance / example / row** — one data point.
- **NB** — Naive Bayes, a fast probabilistic classifier.
- **DT** — Decision Tree, a flowchart classifier that asks column questions.
- **Alg 1 / Alg 2** — the seed paper's two hybrids: NB-deletes-rows (Alg 1), tree-selects-columns (Alg 2).
- **Fold / CV** — one train-test split / the whole 10-split routine.
- **Refit per fold** — relearn the cleaners and classifier inside each training split (the anti-leakage rule).
- **Leakage** — test information sneaking into training; the thing that makes results look better than they are.
- **Macro-F1** — class-balanced correctness score.
- **Seed** — the number that makes randomness repeatable.
- **Weighted NB** — NB where each column's vote is multiplied by an importance weight (Part 2).
