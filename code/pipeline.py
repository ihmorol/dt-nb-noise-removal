"""The pipeline itself: the two paths, the classifiers, and the cross-validation.

    Path 1:  Alg 1 (instances)  ->  Alg 2 (attributes)  ->  new data  ->  NB, DT
    Path 2:  Alg 2 (attributes) ->  Alg 1 (instances)   ->  new data  ->  NB, DT

Rules the code keeps:
    * every stage and the classifier are fitted inside the training fold only
    * attributes are dropped from the training and the test rows alike (they are a
      property of the table), instances are dropped from the training rows only
    * all arms run on the same folds, so the comparison is paired
"""

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from algorithm1 import farid_algorithm1
from algorithm2 import CCP_ALPHA, farid_algorithm2
from nb import WeightedNB

N_SPLITS = 10


def classify(X_train, y_train, X_test, y_test, final, weights=None, likelihood="gaussian"):
    """Fit one of the two classifiers on the new data and score the held-out fold."""
    if final == "NB":
        model = WeightedNB(weights=weights, likelihood=likelihood)
    else:
        model = DecisionTreeClassifier(criterion="entropy", random_state=0)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction) * 100
    macro_f1 = f1_score(y_test, prediction, average="macro", zero_division=0) * 100
    return accuracy, macro_f1


def run_fold(X, y, train_rows, test_rows, steps, final, alpha=CCP_ALPHA, support=True,
             likelihood="gaussian"):
    """Run one arm on one fold, and record what each algorithm removed.

    steps is the path, for example ("N", "A") for Alg 1 then Alg 2, or () for no
    cleaning. Returns (accuracy, macro_f1, stage_log).
    """
    X_train = X[train_rows]
    y_train = y[train_rows]
    X_test = X[test_rows]

    weights = None          # weights only exist once Algorithm 2 has run
    stage_log = []

    for position, step in enumerate(steps):
        if step == "A":
            # Algorithm 2: keep the attributes the tree tests, drop the rest
            attributes_before = X_train.shape[1]
            X_train, info = farid_algorithm2(X_train, y_train, ccp_alpha=alpha)
            X_test = X_test[:, info["keep"]]
            if support:
                weights = info["weights"]
            stage_log.append({
                "stage": position + 1, "method": "Alg2",
                "rows_before": len(y_train), "rows_after": len(y_train),
                "attributes_before": attributes_before,
                "attributes_after": X_train.shape[1], "skipped": False,
            })
        else:
            # Algorithm 1: drop the training instances the NB judge gets wrong
            rows_before = len(y_train)
            attributes_before = X_train.shape[1]
            X_train, y_train, info = farid_algorithm1(X_train, y_train, weights=weights,
                                                      likelihood=likelihood)
            stage_log.append({
                "stage": position + 1, "method": "Alg1",
                "rows_before": rows_before, "rows_after": len(y_train),
                "attributes_before": attributes_before,
                "attributes_after": X_train.shape[1], "skipped": info["skipped"],
            })

    accuracy, macro_f1 = classify(X_train, y_train, X_test, y[test_rows], final,
                                  weights, likelihood)
    return accuracy, macro_f1, stage_log


def average_stages(stage_logs, number_of_steps):
    """Average the per-fold logs, so each algorithm gets one line per arm."""
    stages = []
    for i in range(number_of_steps):
        entries = [log[i] for log in stage_logs]

        rows_before = float(np.mean([e["rows_before"] for e in entries]))
        rows_after = float(np.mean([e["rows_after"] for e in entries]))
        attributes_before = float(np.mean([e["attributes_before"] for e in entries]))
        attributes_after = float(np.mean([e["attributes_after"] for e in entries]))

        rows_removed_pct = 100 * (1 - rows_after / rows_before) if rows_before else 0.0
        attributes_removed_pct = (100 * (1 - attributes_after / attributes_before)
                                  if attributes_before else 0.0)

        stages.append({
            "stage": i + 1,
            "method": entries[0]["method"],
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed_pct": rows_removed_pct,
            "attributes_before": attributes_before,
            "attributes_after": attributes_after,
            "attributes_removed_pct": attributes_removed_pct,
            "skipped_pct": 100 * float(np.mean([e["skipped"] for e in entries])),
        })
    return stages


def run_arm(X, y, steps, final, seeds, alpha=CCP_ALPHA, support=True, likelihood="gaussian"):
    """Cross-validate one arm over the seeds. Every arm uses the same folds.

    Returns a result dictionary with the mean accuracy, its spread, macro-F1, how much
    was removed end to end, and the per-algorithm stage log.
    """
    accuracies = []
    macro_f1s = []
    stage_logs = []

    for seed in seeds:
        # contact-lenses has a class with 4 rows, so sklearn warns about 10 folds.
        # The run goes ahead with 10 folds anyway, exactly as the paper did.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The least populated class")
            folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                         random_state=seed).split(X, y))

        for train_rows, test_rows in folds:
            accuracy, macro_f1, stage_log = run_fold(X, y, train_rows, test_rows, steps,
                                                     final, alpha, support, likelihood)
            accuracies.append(accuracy)
            macro_f1s.append(macro_f1)
            stage_logs.append(stage_log)

    stages = average_stages(stage_logs, len(steps))
    result = {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1": float(np.mean(macro_f1s)),
        "stages": stages,
    }
    if stages:
        result["instances_removed_pct"] = 100 * (1 - stages[-1]["rows_after"]
                                                 / stages[0]["rows_before"])
        result["attributes_kept_pct"] = 100 * stages[-1]["attributes_after"] / X.shape[1]
    else:
        result["instances_removed_pct"] = 0.0
        result["attributes_kept_pct"] = 100.0
    return result


def describe_path(stages):
    """One readable line: what each algorithm removed, and how big the new data is."""
    parts = []
    for stage in stages:
        if stage["method"] == "Alg1":
            parts.append(f"Alg 1 removed {stage['rows_removed_pct']:.1f}% of the rows")
        else:
            parts.append(f"Alg 2 removed {stage['attributes_removed_pct']:.1f}% of the attributes")
    last = stages[-1]
    parts.append(f"new data {last['rows_after']:.0f} rows x {last['attributes_after']:.0f} attributes")
    return " -> ".join(parts)
