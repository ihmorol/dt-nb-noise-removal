"""E3a leakage check: how much of the gap to Farid (2014) comes from the protocol?

The same algorithms on the same data, under three protocols:
    B  honest: every step is refit inside each training fold (what main.py does)
    A  paper-style: the algorithms run ONCE on the whole dataset, then only the
       classifier is cross-validated. Two things leak: the NB filter picks rows
       using every label (and the removed rows vanish from the test folds too),
       and the attribute-selecting tree has seen the test rows.
    C  worst case: algorithms and classifier all fitted on the whole dataset and
       scored on that same data. It shows how high a fully leaky number can go.

Usage:
    python leakage_check.py                all datasets, 3 seeds
    python leakage_check.py iris glass
Writes ../results/EXP-E3a_leakage_check/leakage.csv
"""
import sys
from pathlib import Path

import numpy as np

from data import FARID_2014, available, load_data
from pipeline import Settings, clean, make_classifier, make_folds, run_arm
from tables import write_csv

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E3a_leakage_check"
SEEDS = range(3)
SETTINGS = Settings(alpha=0.01, support=True, likelihood="mixed")

# (arm, steps, final classifier, key of the paper's number or None)
ARMS = [
    ("baseline", (), "NB", "NB"),
    ("baseline", (), "DT", "DT"),
    ("Alg1", ("N",), "DT", "Alg1"),
    ("Alg2", ("A",), "NB", "Alg2"),
    ("C1 N->A", ("N", "A"), "DT", None),
    ("C2 A->N", ("A", "N"), "DT", None),
]


def protocol_a(X, y, steps, final):
    """Clean the whole dataset once, then cross-validate only the classifier."""
    cleaned = clean(X, y, steps, SETTINGS)
    X, y = cleaned.X, cleaned.y
    accuracies = []
    for seed in SEEDS:
        for train, test in make_folds(y, seed):
            model = make_classifier(final, cleaned.weights, SETTINGS).fit(X[train], y[train])
            accuracies.append(100 * np.mean(model.predict(X[test]) == y[test]))
    return float(np.mean(accuracies))


def protocol_c(X, y, steps, final):
    """Clean and fit on the whole dataset, then score on that same data."""
    cleaned = clean(X, y, steps, SETTINGS)
    model = make_classifier(final, cleaned.weights, SETTINGS).fit(cleaned.X, cleaned.y)
    return 100 * float(np.mean(model.predict(cleaned.X) == cleaned.y))


def main(names):
    print("=" * 110)
    print("E3a leakage check: B = refit per fold (honest), A = algorithms fitted once on "
          "all data, C = everything fitted and scored on all data")
    print(f"10-fold CV x {len(SEEDS)} seeds, ccp_alpha={SETTINGS.alpha}, "
          f"NB likelihood={SETTINGS.likelihood}")
    print("=" * 110)
    print(f"{'dataset':<20}{'arm':<10}{'final':<6}{'B honest':>9}{'A leaky':>9}{'C worst':>9}"
          f"{'A - B':>8}{'paper':>8}{'A - paper':>11}{'B - paper':>11}")

    rows = []
    for name in names:
        X, y, _ = load_data(name)
        for arm, steps, final, paper_key in ARMS:
            b = run_arm(X, y, steps, final, SEEDS, SETTINGS)["accuracy"]
            a = protocol_a(X, y, steps, final)
            c = protocol_c(X, y, steps, final)
            paper = FARID_2014[name][paper_key] if paper_key else np.nan
            print(f"{name:<20}{arm:<10}{final:<6}{b:>9.2f}{a:>9.2f}{c:>9.2f}{a - b:>+8.2f}"
                  f"{paper:>8.2f}{a - paper:>+11.2f}{b - paper:>+11.2f}")
            no_paper = paper_key is None
            rows.append([name, arm, final, f"{b:.2f}", f"{a:.2f}", f"{c:.2f}", f"{a - b:+.2f}",
                         "" if no_paper else f"{paper:.2f}",
                         "" if no_paper else f"{a - paper:+.2f}",
                         "" if no_paper else f"{b - paper:+.2f}"])
        print("-" * 110)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "leakage.csv",
              ["dataset", "arm", "final", "accuracy_refit_per_fold", "accuracy_before_cv",
               "accuracy_classifier_on_all_data", "delta_leaky_minus_honest",
               "farid_reported", "delta_leaky_vs_paper", "delta_honest_vs_paper"], rows)
    print(f"\n[saved] {RESULTS / 'leakage.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:] or available())
