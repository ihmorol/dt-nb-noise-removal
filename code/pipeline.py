"""The pipeline: run the algorithms in some order, then train and test a classifier.

    Path 1:  Alg 1 (instances) -> Alg 2 (attributes) -> NB or DT
    Path 2:  Alg 2 (attributes) -> Alg 1 (instances) -> NB or DT

A path is written as steps: "N" is Algorithm 1 (noise filter), "A" is
Algorithm 2 (attribute selection). ("A", "N") is Path 2, () is no cleaning.

Rules that keep the results honest, all enforced in this file:
    * the algorithms and the classifier see the training fold only
    * Algorithm 2 drops columns from the training AND the test fold
      (Cleaned.apply_to_test), Algorithm 1 drops training rows only
    * every arm uses the same folds, so arms can be compared fold by fold
check_pipeline.py tests these rules.
"""
import warnings
from dataclasses import dataclass

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


@dataclass
class Settings:
    alpha: float = 0.0            # Algorithm 2 tree pruning (sklearn ccp_alpha)
    support: bool = True          # hand Algorithm 2's weights to the judge and the final NB
    likelihood: str = "mixed"     # NB likelihood, see nb.py


@dataclass
class Cleaned:
    """A training fold after the algorithms ran.

    rows, columns   the original rows and columns that are left
    weights         Algorithm 2's weights for those columns, or None
    removals        one record per step, all with the same keys
    """
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    removals: list

    def apply_to_test(self, X_test):
        """The test fold loses the same columns; its rows are never removed."""
        return X_test[:, self.columns]


def make_folds(y, seed):
    """The 10 (train rows, test rows) pairs for one seed. Same seed, same folds."""
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def clean(X, y, steps, settings):
    """Run the algorithms on a training fold, in the order given by steps."""
    rows = np.arange(len(y))
    columns = np.arange(X.shape[1])
    weights = None
    removals = []

    for step in steps:
        record = {"rows_before": len(y), "attributes_before": X.shape[1],
                  "removed_rows": np.array([], dtype=int),
                  "removed_columns": np.array([], dtype=int),
                  "skipped": False, "judge": None, "per_class": {}}
        if step == "A":
            keep, new_weights = algorithm2(X, y, settings.alpha)
            record["method"] = "Alg2"
            record["removed_columns"] = np.delete(columns, keep)
            X, columns = X[:, keep], columns[keep]
            if settings.support:
                weights = new_weights
        elif step == "N":
            keep, skipped = algorithm1(X, y, weights, settings.likelihood)
            record["method"] = "Alg1"
            record["removed_rows"] = rows[~keep]
            record["skipped"] = skipped
            record["judge"] = "plain NB" if weights is None else "weighted NB"
            record["per_class"] = {c: float(np.mean(~keep[y == c])) for c in np.unique(y)}
            X, y, rows = X[keep], y[keep], rows[keep]
        else:
            raise ValueError(f"unknown step {step!r}; use 'N' (Alg 1) or 'A' (Alg 2)")

        record["rows_after"] = len(y)
        record["attributes_after"] = X.shape[1]
        removals.append(record)

    return Cleaned(X, y, weights, rows, columns, removals)


def make_classifier(final, weights, settings):
    if final == "NB":
        return WeightedNB(weights, settings.likelihood)
    if final == "DT":
        return DecisionTreeClassifier(criterion="entropy", random_state=0)
    raise ValueError(f"unknown final classifier {final!r}; use 'NB' or 'DT'")


def run_fold(X, y, train, test, steps, final, settings):
    """One arm on one fold. Returns (predicted labels for the test rows, removals)."""
    cleaned = clean(X[train], y[train], steps, settings)
    model = make_classifier(final, cleaned.weights, settings).fit(cleaned.X, cleaned.y)
    return model.predict(cleaned.apply_to_test(X[test])), cleaned.removals


def run_arm(X, y, steps, final, seeds, settings):
    """Cross-validate one arm: 10 folds for every seed. Returns a result dictionary."""
    accuracies, macro_f1s, removals = [], [], []
    for seed in seeds:
        for train, test in make_folds(y, seed):
            predicted, fold_removals = run_fold(X, y, train, test, steps, final, settings)
            accuracies.append(100 * np.mean(predicted == y[test]))
            macro_f1s.append(100 * f1_score(y[test], predicted, average="macro",
                                            zero_division=0))
            removals.append(fold_removals)

    stages = average_stages(removals)
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


def average_stages(removals):
    """Average the fold removals: one summary per step of the path."""
    stages = []
    for i in range(len(removals[0])):
        records = [fold[i] for fold in removals]
        stage = {"stage": i + 1, "method": records[0]["method"]}
        for key in ("rows_before", "rows_after", "attributes_before", "attributes_after",
                    "skipped"):
            stage[key] = float(np.mean([record[key] for record in records]))
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
