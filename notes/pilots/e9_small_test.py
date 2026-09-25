"""E9 smoke test on small datasets — the refined C1/C2 design with Farid semantics.

Runs the frozen system set on small datasets (bundled + three real seed datasets):
  baselines          NB, DT
  single-stage       A->NB, A->DT, N->NB, N->DT
  core (two orders)  C1: N->A (instances then attributes) x {NB, DT}
                     C2: A->N (attributes then instances) x {NB, DT}

Frozen semantics
  A (Alg 2, hard form): entropy tree, ccp_alpha=0.01; drop columns the tree never tests
                        from BOTH train and test; if none kept, no-op.
  N (Alg 1):            GaussianNB drops misclassified training rows; skip if a class
                        would vanish; test rows never touched.
  Classifier:           GaussianNB or entropy DecisionTree.
  Protocol:             stratified 10-fold x 5 seeds, identical folds across systems;
                        everything refit inside the training fold.

Output: notes/pilots/e9_small_test_output.txt
"""
import io
import sys
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

CCP_ALPHA = 0.01
SEEDS = range(5)
N_SPLITS = 10


def tested_attributes(tree):
    f = tree.tree_.feature
    return np.unique(f[f >= 0])


def fit_transform_A(X_train, y_train, X_test):
    """A stage: fit tree on train, drop untested columns from train and test."""
    tree = DecisionTreeClassifier(criterion="entropy", ccp_alpha=CCP_ALPHA, random_state=0)
    tree.fit(X_train, y_train)
    keep = tested_attributes(tree)
    if len(keep) == 0:
        return X_train, X_test, 1.0  # no-op, survives fraction = 1.0
    return X_train[:, keep], X_test[:, keep], len(keep) / X_train.shape[1]


def fit_transform_N(X_train, y_train):
    """N stage: fit NB on train, drop misclassified training rows (safety: never wipe a class)."""
    nb = GaussianNB().fit(X_train, y_train)
    keep = nb.predict(X_train) == y_train
    if len(np.unique(y_train[keep])) >= 2:
        return X_train[keep], y_train[keep], float(keep.mean())
    return X_train, y_train, 0.0  # skipped for this fold


def run_system(X, y, stages, final, seed):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    accs, f1s, remove_rates, survive_rates = [], [], [], []
    for tr, te in skf.split(X, y):
        Xa, ya, Xt = X[tr], y[tr], X[te]
        for s in stages:
            if s == "A":
                Xa, Xt, surv = fit_transform_A(Xa, ya, Xt)
                survive_rates.append(surv)
            elif s == "N":
                n_before = len(ya)
                Xa, ya, kept = fit_transform_N(Xa, ya)
                remove_rates.append(1.0 - kept if kept > 0 else 0.0)
        clf = GaussianNB() if final == "NB" else DecisionTreeClassifier(
            criterion="entropy", random_state=0)
        clf.fit(Xa, ya)
        pred = clf.predict(Xt)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return (float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)),
            float(np.mean(remove_rates)) if remove_rates else float("nan"),
            float(np.mean(survive_rates)) if survive_rates else float("nan"))


SYSTEMS = [
    ("baseline      NB    ", [], "NB"),
    ("baseline      DT    ", [], "DT"),
    ("single        A->NB ", ["A"], "NB"),
    ("single        A->DT ", ["A"], "DT"),
    ("single        N->NB ", ["N"], "NB"),
    ("single        N->DT ", ["N"], "DT"),
    ("C1 N->A      ->NB   ", ["N", "A"], "NB"),
    ("C1 N->A      ->DT   ", ["N", "A"], "DT"),
    ("C2 A->N      ->NB   ", ["A", "N"], "NB"),
    ("C2 A->N      ->DT   ", ["A", "N"], "DT"),
]


def load_lenses():
    df = pd.read_csv("data/lenses.data", sep=r"\s+", header=None, engine="python")
    X = df.iloc[:, 1:-1].astype(str).values
    y = df.iloc[:, -1].astype(str).values
    return X, y


def load_breast_cancer_286():
    df = pd.read_csv("data/breast-cancer.data", header=None, na_filter=False)
    y = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].astype(str).values
    return X, y


def load_tic_tac_toe():
    df = pd.read_csv("data/tic-tac-toe.data", header=None, na_filter=False)
    y = df.iloc[:, -1].astype(str).values
    X = df.iloc[:, :-1].astype(str).values
    return X, y


def one_hot(X):
    return pd.get_dummies(pd.DataFrame(X)).values.astype(float)


def dataset_table():
    out = []
    for name, loader in [("iris", load_iris), ("wine", load_wine),
                         ("breast_cancer_569", load_breast_cancer), ("digits", load_digits)]:
        X, y = loader(return_X_y=True)
        out.append((name, X.astype(float), y))
    for name, loader in [("lenses_24", load_lenses),
                         ("breast_cancer_286", load_breast_cancer_286),
                         ("tic_tac_toe_958", load_tic_tac_toe)]:
        X, y = loader()
        out.append((name, one_hot(X), y))
    return out


def per_class_removal(X, y, seed=0):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    rows = {}
    for tr, _ in skf.split(X, y):
        nb = GaussianNB().fit(X[tr], y[tr])
        removed = nb.predict(X[tr]) != y[tr]
        for c in np.unique(y[tr]):
            m = y[tr] == c
            rows.setdefault(c, []).append(removed[m].mean())
    return {c: float(np.mean(v)) for c, v in rows.items()}


def main():
    print("=" * 100)
    print("E9 smoke test — C1/C2 with Farid semantics — 10-fold CV x 5 seeds (accuracy / macro-F1)")
    print("=" * 100)
    summary = {}
    for name, X, y in dataset_table():
        print(f"\n--- {name}: {X.shape[0]} rows x {X.shape[1]} attrs, {len(np.unique(y))} classes ---")
        print("system                 acc_mean  acc_std   macroF1   N_removed%  A_kept%")
        results = {}
        for label, stages, final in SYSTEMS:
            accs = []
            f1s = []
            rems = []
            survs = []
            for seed in SEEDS:
                acc, std, f1, rem, surv = run_system(X, y, stages, final, seed)
                accs.append(acc)
                f1s.append(f1)
                if not np.isnan(rem):
                    rems.append(rem)
                if not np.isnan(surv):
                    survs.append(surv)
            acc, f1 = float(np.mean(accs)), float(np.mean(f1s))
            rem = float(np.mean(rems)) * 100 if rems else float("nan")
            surv = float(np.mean(survs)) * 100 if survs else float("nan")
            results[label] = (acc, f1)
            print(f"{label}  {acc*100:7.2f}  {np.std(accs)*100:7.2f}   {f1*100:7.2f}"
                  f"   {rem:8.2f}    {surv:8.2f}")
        summary[name] = results
        if name in ("breast_cancer_286", "tic_tac_toe_958"):
            pcr = per_class_removal(X, y)
            line = ", ".join(f"{c}: {v*100:.1f}%" for c, v in sorted(pcr.items()))
            print(f"   per-class NB removal rate (fold-avg): {line}")

    print("\n" + "=" * 100)
    print("HEADLINES")
    print("=" * 100)
    for name, results in summary.items():
        base_nb = results["baseline      NB    "][0]
        base_dt = results["baseline      DT    "][0]
        c1_nb = results["C1 N->A      ->NB   "][0]
        c1_dt = results["C1 N->A      ->DT   "][0]
        c2_nb = results["C2 A->N      ->NB   "][0]
        c2_dt = results["C2 A->N      ->DT   "][0]
        best = max(results.items(), key=lambda kv: kv[1][0])
        print(f"{name:18s} NB {base_nb*100:6.2f} | DT {base_dt*100:6.2f} | "
              f"C1: {c1_nb*100:6.2f}/{c1_dt*100:6.2f} | C2: {c2_nb*100:6.2f}/{c2_dt*100:6.2f} | "
              f"best: {best[0].strip()} {best[1][0]*100:.2f}")
    print("\nC1-vs-C2 (same final): positive = C2 better")
    for name, results in summary.items():
        for fin, key1, key2 in [("NB", "C1 N->A      ->NB   ", "C2 A->N      ->NB   "),
                                ("DT", "C1 N->A      ->DT   ", "C2 A->N      ->DT   ")]:
            d = (results[key2][0] - results[key1][0]) * 100
            print(f"   {name:18s} {fin}: {d:+6.2f}")


if __name__ == "__main__":
    out = io.StringIO()
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = Tee(sys.stdout, out)
    main()
    sys.stdout = sys.__stdout__
    with open("e9_small_test_output.txt", "w", encoding="utf-8") as fh:
        fh.write(out.getvalue())
    print("\n[saved] e9_small_test_output.txt")
