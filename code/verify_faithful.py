"""Checks to run before replicate_farid.py. Prints PASS or FAIL for each.

    V1  FaithfulNB gives the same predictions as sklearn's GaussianNB on numeric data
    V2  FaithfulNB gives the same predictions as Weka's NaiveBayes on every dataset
        whose attributes are all nominal (numeric datasets are only reported:
        Weka adjusts its Gaussian in ways the paper's own NB does not)
    V3  the J48 tree reader finds petal_width at depth 1 in the iris tree
    V4  on the paper's Play-Tennis table, Algorithm 1's judge deletes 1 of 14 rows

Usage: python verify_faithful.py      (needs Java)
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from data import available, load_original
from faithful_nb import FaithfulNB
from weka_utils import naive_bayes_mistakes, run_j48, tree_depths, write_arff

PLAY_TENNIS = pd.DataFrame({
    "outlook": ["sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast",
                "sunny", "sunny", "rain", "sunny", "overcast", "overcast", "rain"],
    "temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild",
                    "cool", "mild", "mild", "mild", "hot", "mild"],
    "humidity": ["high", "high", "high", "high", "normal", "normal", "normal", "high",
                 "normal", "normal", "normal", "high", "normal", "high"],
    "wind": ["weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak",
             "weak", "weak", "strong", "strong", "weak", "strong"],
    "play": ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
             "yes", "yes", "yes", "yes", "yes", "no"],
})


def result(passed):
    return "PASS" if passed else "FAIL"


def check_numeric_nb():
    X, y, meta = load_original("iris")
    mine = FaithfulNB(meta["nominal"], meta["levels"]).fit(X, y).predict(X)
    sklearn = GaussianNB().fit(X, y).predict(X)
    passed = bool((mine == sklearn).all())
    print(f"V1  FaithfulNB == GaussianNB on iris: {result(passed)}")
    return passed


def check_weka_nb(folder):
    print("V2  FaithfulNB vs Weka NaiveBayes, predicting the training file itself:")
    print(f"    {'dataset':<22}{'rows':>7}{'disagree':>10}{'ours wrong':>12}{'weka wrong':>12}")
    passed = True
    for name in available():
        X, y, meta = load_original(name)
        ours_wrong = FaithfulNB(meta["nominal"], meta["levels"]).fit(X, y).predict(X) != y

        file = folder / f"{name}.arff"
        write_arff(file, X, y, meta, sorted(set(y)))
        weka_wrong = np.array(naive_bayes_mistakes(file, file))
        disagree = int((ours_wrong != weka_wrong).sum())

        verdict = ""
        if meta["nominal"].all():                  # all nominal: must agree exactly
            verdict = result(disagree == 0)
            passed = passed and disagree == 0
        print(f"    {name:<22}{len(y):>7}{disagree:>10}{int(ours_wrong.sum()):>12}"
              f"{int(weka_wrong.sum()):>12}  {verdict}")
    print(f"    all-nominal datasets agree exactly: {result(passed)}")
    return passed


def check_tree_reader(folder):
    X, y, meta = load_original("iris")
    file = folder / "iris.arff"
    write_arff(file, X, y, meta, sorted(set(y)))
    depths = tree_depths(run_j48(file, file)[1])
    passed = depths.get("petal_width") == 1
    print(f"V3  J48 iris tree depths {depths}: {result(passed)}")
    return passed


def check_play_tennis(folder):
    table = PLAY_TENNIS.drop(columns="play")
    columns = [pd.Categorical(table[c]) for c in table.columns]
    X = np.column_stack([c.codes for c in columns]).astype(float)
    y = PLAY_TENNIS["play"].to_numpy()
    meta = {"names": list(table.columns), "nominal": np.ones(X.shape[1], dtype=bool),
            "levels": [list(c.categories) for c in columns]}

    wrong = FaithfulNB(meta["nominal"], meta["levels"]).fit(X, y).predict(X) != y
    deleted = [int(i) + 1 for i in np.flatnonzero(wrong)]
    passed = len(deleted) == 1
    print(f"V4  Play-Tennis judge deletes {len(deleted)}/14 rows {deleted}: {result(passed)}")

    file = folder / "play_tennis.arff"
    write_arff(file, X, y, meta, sorted(set(y)))
    print(f"    J48 Play-Tennis tree depths: {tree_depths(run_j48(file, file)[1])}")
    return passed


def main():
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        checks = [check_numeric_nb(), check_weka_nb(folder), check_tree_reader(folder),
                  check_play_tennis(folder)]
    print("\noverall:", "ALL PASS" if all(checks) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
