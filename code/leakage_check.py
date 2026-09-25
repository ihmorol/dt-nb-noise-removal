"""E3a -- the leakage check: how much of the gap to Farid (2014) is protocol?

Protocol B (what main.py does): every stage is refit inside each training fold. The test
fold never influences anything, so the accuracy is an honest estimate.

Protocol A (paper-style): the noise filter and the attribute selection are fitted ONCE on
the whole dataset, their effect is applied to the whole dataset, and only then is the
classifier cross-validated. Two things leak:
    * the instances the NB filter deletes are chosen using every row's label, and the
      deleted rows disappear from the test folds too -- so the test folds keep only the
      rows the judge already agreed with;
    * the tree that selects attributes has seen the test rows.

Protocol C (upper bound): like A, but the classifier is also fitted on the whole dataset
instead of per fold. Nothing is honest any more; it bounds how much a fully leaky setup
could inflate a number.

Running all three on the same data, stages and classifiers isolates the protocol effect
from every other difference (encoding, tree engine, NB likelihood).

Usage: python leakage_check.py                all datasets, 3 seeds
       python leakage_check.py iris glass     a subset
Writes ../results/EXP-E3a_leakage_check/leakage.csv and prints the table.
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithm1 import farid_algorithm1                                # noqa: E402
from algorithm2 import CCP_ALPHA, farid_algorithm2                     # noqa: E402
from data import FARID_2014, available, load_data                      # noqa: E402
from pipeline import N_SPLITS, classify, run_arm                       # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E3a_leakage_check"
N_SEEDS = 3

# (arm, steps, final classifier, the paper's number for that classifier)
ARMS = [
    ("baseline", (), "NB", "NB"),
    ("baseline", (), "DT", "DT"),
    ("Alg1", ("N",), "DT", "Alg1"),          # Farid's Algorithm 1
    ("Alg2", ("A",), "NB", "Alg2"),          # Farid's Algorithm 2
    ("C1 N->A", ("N", "A"), "DT", None),     # our Path 1
    ("C2 A->N", ("A", "N"), "DT", None),     # our Path 2
]


def apply_stages_once(X, y, steps, alpha, support, likelihood):
    """Run the algorithms once over the whole dataset (protocol A)."""
    X_new, y_new = X.copy(), y.copy()
    weights = None
    for step in steps:
        if step == "A":
            X_new, info = farid_algorithm2(X_new, y_new, ccp_alpha=alpha)
            weights = info["weights"] if support else None
        else:
            X_new, y_new, info = farid_algorithm1(X_new, y_new, weights=weights,
                                                  likelihood=likelihood)
    return X_new, y_new, weights


def run_arm_before_cv(X, y, steps, final, seeds, alpha, support, likelihood):
    """Protocol A: transform the whole dataset once, then cross-validate the classifier."""
    X_new, y_new, weights = apply_stages_once(X, y, steps, alpha, support, likelihood)

    accuracies = []
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The least populated class")
            folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=seed).split(X_new, y_new)
            for train_rows, test_rows in folds:
                accuracy, _ = classify(X_new[train_rows], y_new[train_rows],
                                       X_new[test_rows], y_new[test_rows],
                                       final, weights, likelihood)
                accuracies.append(accuracy)
    return float(np.mean(accuracies)), X_new.shape, y_new.shape


def run_arm_all_data(X, y, steps, final, seeds, alpha, support, likelihood):
    """Protocol C: stages and classifier all fitted once on the whole dataset."""
    from pipeline import WeightedNB            # the same NB class the pipeline uses
    from sklearn.tree import DecisionTreeClassifier

    X_new, y_new, weights = apply_stages_once(X, y, steps, alpha, support, likelihood)

    if final == "NB":
        model = WeightedNB(weights=weights, likelihood=likelihood).fit(X_new, y_new)
    else:
        model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X_new, y_new)

    accuracies = []
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The least populated class")
            folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=seed).split(X_new, y_new)
            for _, test_rows in folds:
                accuracies.append(float((model.predict(X_new[test_rows]) == y_new[test_rows]).mean()) * 100)
    return float(np.mean(accuracies))


def main(names, alpha=CCP_ALPHA, support=True, likelihood="mixed"):
    seeds = range(N_SEEDS)
    print("=" * 126)
    print("E3a leakage check -- same algorithms, same data, three protocols")
    print(f"B: stages refit inside each training fold (honest) | A: stages fitted once on all "
          f"data | C: stages AND classifier fitted on all data\n"
          f"{N_SPLITS}-fold CV x {N_SEEDS} seeds, ccp_alpha={alpha}, NB likelihood={likelihood}")
    print("=" * 126)
    print(f"{'dataset':<20}{'arm':<10}{'final':<5}{'B honest':>10}{'A leaky':>10}{'C worst':>9}"
          f"{'A - B':>8}{'paper':>8}{'A - paper':>11}{'B - paper':>11}")

    rows = []
    for name in names:
        X, y, _ = load_data(name)
        for arm, steps, final, farid_key in ARMS:
            honest = run_arm(X, y, steps, final, seeds, alpha, support, likelihood)["accuracy"]
            leaky, _, _ = run_arm_before_cv(X, y, steps, final, seeds, alpha, support, likelihood)
            worst = run_arm_all_data(X, y, steps, final, seeds, alpha, support, likelihood)
            paper = FARID_2014[name][farid_key] if farid_key else float("nan")
            print(f"{name:<20}{arm:<10}{final:<5}{honest:>10.2f}{leaky:>10.2f}{worst:>9.2f}"
                  f"{leaky - honest:>+8.2f}{paper:>8.2f}{leaky - paper:>+11.2f}"
                  f"{honest - paper:>+11.2f}")
            rows.append([name, arm, final, f"{honest:.2f}", f"{leaky:.2f}", f"{worst:.2f}",
                         f"{leaky - honest:+.2f}",
                         f"{paper:.2f}" if farid_key else "",
                         f"{leaky - paper:+.2f}" if farid_key else "",
                         f"{honest - paper:+.2f}" if farid_key else ""])
        print("-" * 126)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "leakage.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "arm", "final", "accuracy_refit_per_fold",
                         "accuracy_before_cv", "accuracy_classifier_on_all_data",
                         "delta_leaky_minus_honest",
                         "farid_reported", "delta_leaky_vs_paper", "delta_honest_vs_paper"])
        writer.writerows(rows)
    print(f"\n[saved] {RESULTS / 'leakage.csv'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or available())
