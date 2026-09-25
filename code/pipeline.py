"""The pipeline: run the algorithms in some order, then train and test a classifier.

    Path 1:  Alg 1 (instances) -> Alg 2 (attributes) -> NB or DT
    Path 2:  Alg 2 (attributes) -> Alg 1 (instances) -> NB or DT

A path is written as steps: "N" is Algorithm 1 (noise filter), "A" is
Algorithm 2 (attribute selection). ("A", "N") is Path 2, () is no cleaning.

Rules that keep the results honest:
    * everything is fitted on the training fold only
    * Algorithm 2 drops columns from the training AND the test fold,
      Algorithm 1 drops rows from the training fold only
    * every arm uses the same folds, so arms can be compared fold by fold
"""
import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from algorithm1 import algorithm1
from algorithm2 import algorithm2
from nb import WeightedNB

N_SPLITS = 10

# contact-lenses has a class with only 4 rows, so sklearn warns that 10 folds is
# more than that class has. We keep 10 folds, like the paper.
warnings.filterwarnings("ignore", message="The least populated class")


def make_folds(y, seed):
    """The 10 (train rows, test rows) pairs for one seed. Same seed, same folds."""
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def clean(X, y, steps, alpha, support, likelihood):
    """Run the algorithms in the order given by steps.

    support=True hands Algorithm 2's weights on to Algorithm 1's judge and to the
    final NB. Returns (X, y, weights, rows, columns, log):
        rows, columns   which original rows and columns are left
        log             one dictionary per step, saying what it removed
    """
    rows = np.arange(len(y))
    columns = np.arange(X.shape[1])
    weights = None
    log = []

    for step in steps:
        before = {"rows_before": len(y), "attributes_before": X.shape[1]}
        if step == "A":
            keep, new_weights = algorithm2(X, y, alpha)
            entry = {"method": "Alg2", "removed_columns": np.delete(columns, keep),
                     "skipped": False}
            X, columns = X[:, keep], columns[keep]
            if support:
                weights = new_weights
        else:
            keep, skipped = algorithm1(X, y, weights, likelihood)
            entry = {"method": "Alg1", "removed_rows": rows[~keep], "skipped": skipped,
                     "judge": "plain NB" if weights is None else "weighted NB",
                     "per_class": {c: float(np.mean(~keep[y == c])) for c in np.unique(y)}}
            X, y, rows = X[keep], y[keep], rows[keep]

        entry.update(before)
        entry.update({"rows_after": len(y), "attributes_after": X.shape[1]})
        log.append(entry)

    return X, y, weights, rows, columns, log


def make_classifier(final, weights, likelihood):
    if final == "NB":
        return WeightedNB(weights, likelihood)
    return DecisionTreeClassifier(criterion="entropy", random_state=0)


def run_fold(X, y, train, test, steps, final, alpha, support, likelihood):
    """One arm on one fold. Returns (accuracy %, macro-F1 %, stage log)."""
    X_train, y_train, weights, _, columns, log = clean(X[train], y[train], steps,
                                                       alpha, support, likelihood)
    X_test = X[test][:, columns]

    model = make_classifier(final, weights, likelihood).fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = 100 * np.mean(predicted == y[test])
    macro_f1 = 100 * f1_score(y[test], predicted, average="macro", zero_division=0)
    return accuracy, macro_f1, log


def run_arm(X, y, steps, final, seeds, alpha, support, likelihood):
    """Cross-validate one arm: 10 folds for every seed. Returns a result dictionary."""
    accuracies, macro_f1s, logs = [], [], []
    for seed in seeds:
        for train, test in make_folds(y, seed):
            accuracy, macro_f1, log = run_fold(X, y, train, test, steps, final,
                                               alpha, support, likelihood)
            accuracies.append(accuracy)
            macro_f1s.append(macro_f1)
            logs.append(log)

    stages = average_stages(logs)
    result = {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1": float(np.mean(macro_f1s)),
        "stages": stages,
        "instances_removed_pct": 0.0,
        "attributes_kept_pct": 100.0,
    }
    if stages:
        first, last = stages[0], stages[-1]
        result["instances_removed_pct"] = 100 * (1 - last["rows_after"] / first["rows_before"])
        result["attributes_kept_pct"] = 100 * last["attributes_after"] / X.shape[1]
    return result


def average_stages(logs):
    """Average the fold logs: one summary per step of the path."""
    stages = []
    for i in range(len(logs[0])):
        step_logs = [log[i] for log in logs]
        stage = {"stage": i + 1, "method": step_logs[0]["method"]}
        for key in ("rows_before", "rows_after", "attributes_before", "attributes_after",
                    "skipped"):
            stage[key] = float(np.mean([entry[key] for entry in step_logs]))
        stage["rows_removed_pct"] = 100 * (1 - stage["rows_after"] / stage["rows_before"])
        stage["attributes_removed_pct"] = 100 * (1 - stage["attributes_after"]
                                                 / stage["attributes_before"])
        stage["skipped_pct"] = 100 * stage.pop("skipped")
        stages.append(stage)
    return stages


def describe_path(stages):
    """One line: what each algorithm removed, and the size of the new data."""
    parts = []
    for stage in stages:
        if stage["method"] == "Alg1":
            parts.append(f"Alg 1 removed {stage['rows_removed_pct']:.1f}% of the rows")
        else:
            parts.append(f"Alg 2 removed {stage['attributes_removed_pct']:.1f}% of the attributes")
    last = stages[-1]
    parts.append(f"new data {last['rows_after']:.0f} rows x {last['attributes_after']:.0f} attributes")
    return " -> ".join(parts)
