"""Pilot checks for Phase 1 (E9) — mechanical falsification tests.

Datasets: sklearn bundled only (illustrative, NOT the seed's 10 UCI datasets).
Purposes:
  1) Attribute-removal (A) no-op risk: how many columns does the tree actually
     test, as a function of pruning? If ~all attributes are tested, an
     "untested attributes" removal step deletes nothing and the A-stage is a
     no-op on typical small UCI-style data.
  2) Noise-removal (N): how much of a training fold does the NB judge actually
     delete (in-fold resubstitution error of GaussianNB)?
  3) Preliminary accuracy signs for the 14-system grid (10 folds x 3 seeds).

Run:  python notes/pilots/pilot_phase1_check.py | tee notes/pilots/pilot_phase1_output.txt
"""
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

DATA = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}
ALPHA = 0.01  # pruning example used in Check 3


def tested_attrs(tree):
    """Column indices the fitted tree actually splits on (sklearn: -2 = leaf)."""
    f = tree.tree_.feature
    return np.unique(f[f >= 0])


print("=" * 78)
print("CHECK 1 - attribute-removal (A): how many attributes does the tree test?")
print("=" * 78)
for name, loader in DATA.items():
    X, y = loader(return_X_y=True)
    n = X.shape[1]
    line = f"{name:14s} n_attr={n:3d} | default: "
    t = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    line += f"{len(tested_attrs(t)):3d}/{n} tested"
    for a in (0.001, 0.005, 0.01, 0.02):
        t2 = DecisionTreeClassifier(criterion="entropy", ccp_alpha=a, random_state=0).fit(X, y)
        line += f" | ccp={a}: {len(tested_attrs(t2)):3d}/{n}"
    print(line)

print()
print("=" * 78)
print("CHECK 2 - noise-removal (N): in-fold NB removal rate (resubstitution)")
print("=" * 78)
for name, loader in DATA.items():
    X, y = loader(return_X_y=True)
    skf = StratifiedKFold(10, shuffle=True, random_state=0)
    rates = []
    for tr, _ in skf.split(X, y):
        nb = GaussianNB().fit(X[tr], y[tr])
        rates.append((nb.predict(X[tr]) != y[tr]).mean())
    print(f"{name:14s} removal per fold: mean {np.mean(rates)*100:5.1f}%  "
          f"(min {min(rates)*100:.1f}%, max {max(rates)*100:.1f}%)")

print()
print("=" * 78)
print("CHECK 3 - preliminary accuracy signs, 10-fold x 3 seeds (accuracy %)")
print("=" * 78)


def run_system(X, y, stages, final, seeds=(0, 1, 2)):
    accs = []
    for seed in seeds:
        skf = StratifiedKFold(10, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            Xa, ya, Xte = X[tr], y[tr], X[te]
            for s in stages:
                if s == "A":
                    rem = DecisionTreeClassifier(
                        criterion="entropy", ccp_alpha=ALPHA, random_state=0
                    ).fit(Xa, ya)
                    keep = tested_attrs(rem)
                    if len(keep) == 0:
                        continue  # edge case: stage is a no-op, logged in E9
                    Xa, Xte = Xa[:, keep], Xte[:, keep]
                elif s == "N":
                    nb = GaussianNB().fit(Xa, ya)
                    m = nb.predict(Xa) == ya
                    if len(np.unique(ya[m])) >= 2:
                        Xa, ya = Xa[m], ya[m]
                    # else: removal would wipe a class -> skip this fold
            clf = (
                GaussianNB()
                if final == "NB"
                else DecisionTreeClassifier(criterion="entropy", random_state=0)
            )
            clf.fit(Xa, ya)
            accs.append(clf.score(Xte, y[te]))
    return np.mean(accs) * 100


SYSTEMS = [
    ("NB      ", [], "NB"), ("DT      ", [], "DT"),
    ("A->NB   ", ["A"], "NB"), ("N->NB   ", ["N"], "NB"),
    ("AA->NB  ", ["A", "A"], "NB"), ("AN->NB  ", ["A", "N"], "NB"),
    ("NA->NB  ", ["N", "A"], "NB"), ("NN->NB  ", ["N", "N"], "NB"),
    ("A->DT   ", ["A"], "DT"), ("N->DT   ", ["N"], "DT"),
    ("AA->DT  ", ["A", "A"], "DT"), ("AN->DT  ", ["A", "N"], "DT"),
    ("NA->DT  ", ["N", "A"], "DT"), ("NN->DT  ", ["N", "N"], "DT"),
]

for name, loader in DATA.items():
    X, y = loader(return_X_y=True)
    print(f"-- {name} --")
    for label, stages, final in SYSTEMS:
        print(f"   {label} {run_system(X, y, stages, final):6.2f}")
