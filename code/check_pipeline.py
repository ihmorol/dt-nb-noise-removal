"""Checks for the pipeline's honesty rules. Prints PASS or FAIL for each.

    P1  the test fold's labels never reach a fit: changing them changes no prediction
    P2  Algorithm 1 removes training rows only: every test row gets a prediction
    P3  Algorithm 2's column choice is applied to the test fold
    P4  an unknown step or final classifier raises ValueError

Usage: python check_pipeline.py      (a few seconds, no Java needed)
"""
import numpy as np

from algorithm2 import algorithm2
from data import load_data
from pipeline import Settings, clean, make_classifier, make_folds, run_fold

PATHS = [(), ("N",), ("A",), ("N", "A"), ("A", "N")]
SETTINGS = Settings(alpha=0.0, support=True, likelihood="mixed")


def result(passed):
    return "PASS" if passed else "FAIL"


def check_test_labels_unused(X, y, train, test):
    poisoned = y.copy()
    poisoned[test] = y[test][::-1]                  # scramble the test labels
    passed = True
    for steps in PATHS:
        for final in ("NB", "DT"):
            honest, _ = run_fold(X, y, train, test, steps, final, SETTINGS)
            changed, _ = run_fold(X, poisoned, train, test, steps, final, SETTINGS)
            passed = passed and np.array_equal(honest, changed)
    return passed


def check_test_rows_kept(X, y, train, test):
    passed = True
    for steps in PATHS:
        predicted, _ = run_fold(X, y, train, test, steps, "DT", SETTINGS)
        cleaned = clean(X[train], y[train], steps, SETTINGS)
        passed = (passed and len(predicted) == len(test)
                  and set(cleaned.rows) <= set(range(len(train))))
    return passed


def check_test_columns(X, y, train, test):
    cleaned = clean(X[train], y[train], ("A",), SETTINGS)
    keep, _ = algorithm2(X[train], y[train], SETTINGS.alpha)
    passed = np.array_equal(cleaned.columns, keep)
    passed = passed and np.array_equal(cleaned.apply_to_test(X[test]), X[test][:, keep])

    cleaned = clean(X[train], y[train], ("N", "A"), SETTINGS)
    return passed and cleaned.apply_to_test(X[test]).shape == (len(test), len(cleaned.columns))


def check_unknown_names(X, y, train):
    passed = True
    for bad_call in (lambda: clean(X[train], y[train], ("X",), SETTINGS),
                     lambda: make_classifier("SVM", None, SETTINGS)):
        try:
            bad_call()
            passed = False
        except ValueError:
            pass
    return passed


def main():
    checks = []
    for name in ("glass", "tic-tac-toe"):           # numeric data, and one-hot data
        X, y, _ = load_data(name)
        train, test = make_folds(y, seed=0)[0]
        checks += [
            ("P1", name, check_test_labels_unused(X, y, train, test)),
            ("P2", name, check_test_rows_kept(X, y, train, test)),
            ("P3", name, check_test_columns(X, y, train, test)),
            ("P4", name, check_unknown_names(X, y, train)),
        ]
    for check, name, passed in checks:
        print(f"{check}  {name:<12} {result(passed)}")
    print("\noverall:", "ALL PASS" if all(p for _, _, p in checks) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
