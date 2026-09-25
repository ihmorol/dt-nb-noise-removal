"""R1 -- replication of Farid et al. (2014), Tables 8-11: C4.5, Algorithm 1,
Naive Bayes, Algorithm 2 on their ten datasets.

What is faithful here (every choice fixed a priori, none tuned against results):
    * tree engine   Weka J48 3.8.6, pruned, Weka defaults (-C 0.25 -M 2) -- J48 IS
                    C4.5; it splits nominal attributes one branch per value, so the
                    "attribute" of Algorithm 2 is one ARFF column, exactly as in the
                    paper. Verified in verify_faithful.py (iris tree parse).
    * NB            FaithfulNB: the paper's own Java NB is a textbook NB -- categorical
                    (add-one) on nominal attributes, Gaussian on numeric. Verified
                    == sklearn GaussianNB on numeric and == Weka NB on nominal data.
    * attribute space  the ORIGINAL columns (no one-hot), as the paper's tables have it
    * Algorithm 1   faithful judge = FaithfulNB self-classification of the training
                    rows; delete every misclassified row; J48 on the remainder.
    * Algorithm 2   J48 on the training rows; W_j = 1/sqrt(min depth) for tested
                    attributes, 0 otherwise; FaithfulNB with those exponents
                    (Eq. 14) on the kept attributes.

Protocols (both reported -- the paper never states which it used; E3a showed this
choice is worth ~15 points to Algorithm 1):
    B  refit per fold (leakage-safe): stages AND classifier refit inside every
       training fold; the test fold is never touched by any fit.
    A  paper-apparent: the filter / the attribute selection are fitted ONCE on the
       whole dataset, applied to it, and only the classifier is cross-validated.
       (The baselines have no stages, so A == B for them.)

Usage: python replicate_farid.py --seeds 3 iris glass
       python replicate_farid.py                  all datasets, 3 seeds
       python replicate_farid.py --unpruned       J48 -U sensitivity (declared)
Writes results/EXP-R1_replication/{replication.csv, config.json}.
"""

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import FARID_2014, available, load_original                       # noqa: E402
from faithful_nb import FaithfulNB                                          # noqa: E402
from weka_utils import parse_tree, run_j48, sanitize, write_arff            # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-R1_replication"
TMP = RESULTS / "tmp"
N_SPLITS = 10

# the four paper arms, in the paper's own table order; FARID_2014 stores the
# C4.5 column under the key "DT"
ARMS = ["C4.5", "NB", "Alg1", "Alg2"]
PAPER_KEY = {"C4.5": "DT", "NB": "NB", "Alg1": "Alg1", "Alg2": "Alg2"}


def parse_args(argv):
    seeds = list(range(3))
    names, rest = [], list(argv)
    unpruned = False
    while rest:
        arg = rest.pop(0)
        if arg == "--seeds":
            seeds = list(range(int(rest.pop(0))))
        elif arg == "--unpruned":
            unpruned = True
        else:
            names.append(arg)
    return seeds, (names or available()), unpruned


def folds_for(y, seeds):
    """The same stratified fold plans the rest of the project uses."""
    out = []
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The least populated class")
            splits = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=seed).split(np.zeros(len(y)), y)
            out.append(list(splits))
    return out


def write_fold(name, tag, X, y, rows, meta, class_levels):
    """Write one ARFF holding the given rows; returns the path.

    class_levels are the WHOLE-dataset class values, so every file of a dataset
    carries the same header (a training slice missing a class still declares it).
    """
    path = TMP / f"{name}_{tag}.arff"
    write_arff(path, X[rows], y[rows], meta["names"], meta["nominal"],
               meta["levels"], relation=f"{name}_{tag}",
               class_levels=class_levels)
    return path


def reduced_view(X, meta, keep):
    """Slice columns down to the tree-tested ones, with matching mask/levels."""
    keep = np.asarray(sorted(keep), dtype=int)
    return X[:, keep], meta["nominal"][keep], [meta["levels"][j] for j in keep]


def run_dataset(name, seeds, unpruned):
    """Both protocols for the four arms on one dataset. Returns result rows."""
    X, y, meta = load_original(name)
    name_to_idx = {sanitize(n): j for j, n in enumerate(meta["names"])}
    class_levels = sorted(set(str(v) for v in y))
    paper = FARID_2014[name]
    n = len(y)
    print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes "
          f"({int(meta['nominal'].sum())} nominal), {meta['n_classes']} classes ---")

    accs = {arm: {"B": [], "A": []} for arm in ARMS}

    # ---- protocol A stage fits happen ONCE, on the whole dataset -----------------
    arff_all = write_fold(name, "all", X, y, np.arange(n), meta, class_levels)

    # Alg1-A: the judge deletes from the WHOLE dataset
    judge = FaithfulNB(meta["nominal"], meta["levels"]).fit(X, y)
    keep_rows = np.flatnonzero(judge.predict(X) == y)
    arff_a1 = write_fold(name, "a1filtered", X, y, keep_rows, meta, class_levels)
    plan_a1 = folds_for(y[keep_rows], seeds)        # folds recomputed on the filter

    # Alg2-A folds over the FULL data: only the columns were pre-selected.
    # (Reusing Alg1's filtered-row plan here would silently chain the two
    # algorithms -- the bug the tic-tac-toe smoke test exposed.)
    plan_full = folds_for(y, seeds)

    # Alg2-A: the tree selects on the WHOLE dataset
    _, tree_lines_a = run_j48(arff_all, arff_all, unpruned)
    tested_a = parse_tree(tree_lines_a)
    keep_a = sorted(name_to_idx[a] for a in tested_a)
    weights_a = np.array([1.0 / np.sqrt(tested_a[sanitize(meta["names"][j])]) for j in keep_a]) \
        if keep_a else np.array([])
    print(f"    Alg2-A tree tests {len(keep_a)}/{X.shape[1]} attributes; "
          f"Alg1-A deletes {n - len(keep_rows)} of {n} rows")

    # ---- protocol B: everything refit inside each fold ---------------------------
    for seed_folds in folds_for(y, seeds):
        for train_rows, test_rows in seed_folds:
            arff_train = write_fold(name, "train", X, y, train_rows, meta, class_levels)
            arff_test = write_fold(name, "test", X, y, test_rows, meta, class_levels)

            # one J48 call serves both C4.5 (accuracy) and Alg2 (the tree)
            accuracy_c45, tree_lines = run_j48(arff_train, arff_test, unpruned)
            accs["C4.5"]["B"].append(accuracy_c45)

            tested_b = parse_tree(tree_lines)
            keep_b = sorted(name_to_idx[a] for a in tested_b)
            if keep_b:
                Xb, nominal_b, levels_b = reduced_view(X, meta, keep_b)
                w = np.array([1.0 / np.sqrt(tested_b[sanitize(meta["names"][j])])
                              for j in keep_b])
                model = FaithfulNB(nominal_b, levels_b,
                                   weights=w).fit(Xb[train_rows], y[train_rows])
                accs["Alg2"]["B"].append(model.score(Xb[test_rows], y[test_rows]) * 100)
            else:
                accs["Alg2"]["B"].append(np.nan)    # degenerate: no attribute tested

            model = FaithfulNB(meta["nominal"], meta["levels"]).fit(X[train_rows],
                                                                    y[train_rows])
            accs["NB"]["B"].append(model.score(X[test_rows], y[test_rows]) * 100)

            # Alg1: judge on the training rows, delete, tree on the remainder
            judge = FaithfulNB(meta["nominal"], meta["levels"]).fit(X[train_rows],
                                                                    y[train_rows])
            wrong = judge.predict(X[train_rows]) != y[train_rows]
            if np.unique(y[train_rows][~wrong]).size < 2:
                accs["Alg1"]["B"].append(accuracy_c45)   # safety rule: skip filter
            else:
                kept = train_rows[~wrong]
                arff_f = write_fold(name, "filtered", X, y, kept, meta, class_levels)
                accs["Alg1"]["B"].append(run_j48(arff_f, arff_test, unpruned)[0])

    # ---- protocol A: stages once, classifier cross-validated ---------------------
    # Alg1-A: CV the tree on the pre-filtered population
    for seed_folds in plan_a1:
        for train_rows, test_rows in seed_folds:
            arff_train = write_fold(name, "a1train", X, y, keep_rows[train_rows], meta, class_levels)
            arff_test = write_fold(name, "a1test", X, y, keep_rows[test_rows], meta, class_levels)
            accs["Alg1"]["A"].append(run_j48(arff_train, arff_test, unpruned)[0])

    # Alg2-A: CV the weighted NB on the pre-selected columns, full population
    Xa, nominal_a, levels_a = reduced_view(X, meta, keep_a) if keep_a else (X, meta["nominal"], meta["levels"])
    for seed_folds in plan_full:
        for train_rows, test_rows in seed_folds:
            if keep_a:
                model = FaithfulNB(nominal_a, levels_a,
                                   weights=weights_a).fit(Xa[train_rows], y[train_rows])
                accs["Alg2"]["A"].append(model.score(Xa[test_rows], y[test_rows]) * 100)
            else:
                accs["Alg2"]["A"].append(np.nan)

    # ---- collect and print --------------------------------------------------------
    rows = []
    print(f"    {'arm':<6}{'B honest':>10}{'A paper-style':>14}{'paper':>9}")
    for arm in ARMS:
        b = float(np.nanmean(accs[arm]["B"]))
        entries = [{"dataset": name, "arm": arm, "protocol": "B refit-per-fold",
                    "accuracy": round(b, 2), "paper": paper[PAPER_KEY[arm]],
                    "delta": round(b - paper[PAPER_KEY[arm]], 2)}]
        if arm in ("Alg1", "Alg2"):                 # baselines have no stages
            a = float(np.nanmean(accs[arm]["A"]))
            entries.append({"dataset": name, "arm": arm, "protocol": "A before-CV",
                            "accuracy": round(a, 2), "paper": paper[PAPER_KEY[arm]],
                            "delta": round(a - paper[PAPER_KEY[arm]], 2)})
        for entry in entries:
            rows.append(entry)
            print(f"    {entry['arm']:<6}{entry['accuracy']:>10.2f}"
                  f"{entry['accuracy'] if entry['protocol'] == 'A before-CV' else '':>14}"
                  f"{entry['paper']:>9.2f}")
    return rows


def main():
    seeds, names, unpruned = parse_args(sys.argv[1:])
    TMP.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    config = {"seeds": seeds, "n_splits": N_SPLITS,
              "j48": "Weka 3.8.6 " + ("unpruned -M 2" if unpruned
                                      else "pruned, defaults -C 0.25 -M 2"),
              "nb": "FaithfulNB (categorical add-one nominal, Gaussian numeric)",
              "attribute_space": "original columns, no one-hot",
              "protocol_B": "stages and classifier refit inside each training fold",
              "protocol_A": "stages fitted once on the whole dataset before CV"}
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows = []
    for name in names:
        all_rows += run_dataset(name, seeds, unpruned)
        for stale in TMP.glob(f"{sanitize(name)}_*.arff"):
            stale.unlink()

    with open(RESULTS / "replication.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "arm", "protocol",
                                               "accuracy", "paper", "delta"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[saved] {RESULTS / 'replication.csv'}")


if __name__ == "__main__":
    main()
