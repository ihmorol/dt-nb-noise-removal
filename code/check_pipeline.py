"""Checks for the pipeline's honesty rules. Prints PASS or FAIL for each.

    P1  the test fold's labels never reach a fit: changing them changes no prediction
    P2  Algorithm 1 removes training rows only: every test row gets a prediction
    P3  Algorithm 2's column choice is applied to the test fold
    P4  an unknown step or final classifier raises ValueError
    C1  a noise filter never sees a test label: scrambling them changes no deletion
    C2  a noise filter is deterministic and returns a mask of the right length
    C3  the per-class floor never lets a filter empty a class
    C4  noise injection changes only the labels it was handed, and only some of them
    C5  an unknown judge or filter method raises ValueError

Usage: python check_pipeline.py      (a few seconds, no Java needed)
"""
import numpy as np

from algorithm2 import algorithm2
from confident_filter import confident_filter
from data import load_data
from noise_inject import inject
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


# --- E12: the noise filters have their own honesty rules ---------------------
#   C1  a filter never sees a test label: scrambling them changes no deletion
#   C2  the filter is deterministic and shapes match its input
#   C3  the per-class floor never lets a filter empty a class
#   C4  noise injection leaves the rows it was not given alone
#   C5  an unknown judge or method raises ValueError
E12_ARMS = [("DT", "hard_vote"), ("NB", "hard_vote"),
            ("DT", "confident_joint"), ("NB", "confident_joint")]


def check_filter_ignores_test_labels(X, y, train, test):
    """C1: the filter's keep-mask must not depend on the test labels."""
    passed = True
    poisoned = y.copy()
    poisoned[test] = y[test][::-1]
    for judge, method in E12_ARMS:
        honest, _ = confident_filter(X[train], y[train], judge=judge,
                                     method=method, seed=0)
        changed, _ = confident_filter(X[train], poisoned[train], judge=judge,
                                      method=method, seed=0)
        passed = passed and np.array_equal(honest, changed)
    return passed


def check_filter_deterministic(X, y, train):
    """C2: same input, same seed, same mask - and the mask fits its input."""
    passed = True
    for judge, method in E12_ARMS:
        first, _ = confident_filter(X[train], y[train], judge=judge,
                                    method=method, seed=0)
        again, _ = confident_filter(X[train], y[train], judge=judge,
                                    method=method, seed=0)
        passed = passed and np.array_equal(first, again)
        passed = passed and len(first) == len(y[train])
    return passed


def check_class_floor(X, y):
    """C3: with a floor on, no class may be left empty."""
    y_noisy, _ = inject(X, y, 0.30, kind="asymmetric", seed=0)
    passed = True
    for judge, method in E12_ARMS:
        keep, _ = confident_filter(X, y_noisy, judge=judge, method=method,
                                   min_per_class=5, seed=0)
        for c in np.unique(y_noisy):
            size = int((y_noisy == c).sum())
            kept = int((y_noisy[keep] == c).sum())
            if kept < min(5, size):
                passed = False
    return passed


def check_injection_is_local(X, y, train, test):
    """C4: inject() must change only the labels it was handed, and only some."""
    original = y.copy()
    labels = y[train]
    noised, mask = inject(X[train], labels, 0.30, kind="asymmetric", seed=0)
    unchanged = labels[~mask]
    return (len(mask) == len(labels)
            and np.array_equal(noised[~mask], unchanged)
            and np.array_equal(y, original)
            and not np.array_equal(noised, labels))


def check_filter_bad_names(X, y):
    """C5: unknown judge / method must raise, not silently misbehave."""
    passed = True
    for bad in (lambda: confident_filter(X, y, judge="SVM"),
                lambda: confident_filter(X, y, method="magic")):
        try:
            bad()
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
            ("C1", name, check_filter_ignores_test_labels(X, y, train, test)),
            ("C2", name, check_filter_deterministic(X, y, train)),
            ("C3", name, check_class_floor(X, y)),
            ("C4", name, check_injection_is_local(X, y, train, test)),
            ("C5", name, check_filter_bad_names(X, y)),
        ]
    for check, name, passed in checks:
        print(f"{check}  {name:<12} {result(passed)}")
    print("\noverall:", "ALL PASS" if all(p for _, _, p in checks) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
