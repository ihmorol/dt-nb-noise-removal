"""R1: replicate Farid et al. (2014) Tables 8-11 with the paper's own setup.

    C4.5   Weka J48, pruned, Weka's default options
    NB     FaithfulNB (textbook NB on the original attribute columns)
    Alg 1  FaithfulNB classifies the training rows, the wrong ones are deleted,
           J48 is trained on the rest
    Alg 2  J48 is trained, every tested attribute gets W = 1/sqrt(smallest depth),
           the rest get W = 0, then the weighted FaithfulNB (Eq. 14) classifies

All choices were fixed before any result was seen (see DECISIONS.md, 2026-09-19).
The paper does not say which protocol it used, so both are reported:
    B  honest: Algorithm 1/2 and the classifier are refit inside every training fold
    A  paper-style: Algorithm 1/2 run once on the whole dataset, then only the
       classifier is cross-validated (C4.5 and NB have no algorithm step, so A = B)

Usage:
    python replicate_farid.py                    all datasets, 3 seeds
    python replicate_farid.py iris glass --seeds 10
    python replicate_farid.py --unpruned         J48 -U sensitivity run
Writes ../results/EXP-R1_replication/replication.csv and config.json
(replication_unpruned.csv and config_unpruned.json with --unpruned).
Needs Java; the Weka jars are in code/lib/.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from data import FARID_2014, available, load_original
from faithful_nb import FaithfulNB
from pipeline import N_SPLITS, make_folds
from tables import write_csv
from weka_utils import clean_name, run_j48, tree_depths, write_arff

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-R1_replication"
PAPER_KEY = {"C4.5": "DT", "NB": "NB", "Alg1": "Alg1", "Alg2": "Alg2"}


def weighted_nb_accuracy(X, y, meta, depths, train, test):
    """Algorithm 2's classifier: FaithfulNB on the tested attributes, W = 1/sqrt(depth).

    If the tree tests no attribute, every W is 0 and NB predicts from the class
    priors alone, which is what Eq. (14) gives in that case.
    """
    column_of = {clean_name(name): j for j, name in enumerate(meta["names"])}
    columns = sorted(column_of[name] for name in depths)
    weights = [1 / np.sqrt(depths[clean_name(meta["names"][j])]) for j in columns]

    nb = FaithfulNB(meta["nominal"][columns], [meta["levels"][j] for j in columns], weights)
    nb.fit(X[train][:, columns], y[train])
    return nb.accuracy(X[test][:, columns], y[test])


def run_dataset(name, seeds, unpruned, folder):
    X, y, meta = load_original(name)
    classes = sorted(set(y))
    print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes "
          f"({int(meta['nominal'].sum())} nominal), {meta['n_classes']} classes ---")

    def arff(tag, rows):
        path = folder / f"{tag}.arff"
        write_arff(path, X[rows], y[rows], meta, classes)
        return path

    honest = {"C4.5": [], "NB": [], "Alg1": [], "Alg2": []}
    paper_style = {"Alg1": [], "Alg2": []}

    # Protocol B: everything is refit inside each training fold
    for seed in seeds:
        for train, test in make_folds(y, seed):
            train_file, test_file = arff("train", train), arff("test", test)

            # one J48 run gives both the C4.5 score and Algorithm 2's tree
            accuracy, tree = run_j48(train_file, test_file, unpruned)
            honest["C4.5"].append(accuracy)
            honest["Alg2"].append(weighted_nb_accuracy(X, y, meta, tree_depths(tree),
                                                       train, test))
            nb = FaithfulNB(meta["nominal"], meta["levels"]).fit(X[train], y[train])
            honest["NB"].append(nb.accuracy(X[test], y[test]))

            # Algorithm 1: the same NB judges the training rows, the wrong ones go
            keep = nb.predict(X[train]) == y[train]
            if len(np.unique(y[train][keep])) < 2:
                # our safety rule: never delete down to one class; the tree is then plain C4.5
                honest["Alg1"].append(accuracy)
            else:
                honest["Alg1"].append(run_j48(arff("filtered", train[keep]), test_file,
                                              unpruned)[0])

    # Protocol A: Algorithm 1 and 2 run once on the whole dataset
    all_file = arff("all", np.arange(len(y)))
    depths = tree_depths(run_j48(all_file, all_file, unpruned)[1])
    judge = FaithfulNB(meta["nominal"], meta["levels"]).fit(X, y)
    kept_rows = np.flatnonzero(judge.predict(X) == y)
    print(f"    paper-style: the tree tests {len(depths)}/{X.shape[1]} attributes, "
          f"the NB filter deletes {len(y) - len(kept_rows)} of {len(y)} rows")

    for seed in seeds:
        for train, test in make_folds(y[kept_rows], seed):
            train_file = arff("a1train", kept_rows[train])
            test_file = arff("a1test", kept_rows[test])
            paper_style["Alg1"].append(run_j48(train_file, test_file, unpruned)[0])
        for train, test in make_folds(y, seed):
            paper_style["Alg2"].append(weighted_nb_accuracy(X, y, meta, depths, train, test))

    # the table
    print(f"    {'arm':<6}{'B honest':>10}{'A paper-style':>15}{'paper':>9}")
    rows = []
    for arm, scores in honest.items():
        paper = FARID_2014[name][PAPER_KEY[arm]]
        b = float(np.mean(scores))
        rows.append([name, arm, "B refit-per-fold", round(b, 2), paper, round(b - paper, 2)])
        a_text = ""
        if arm in paper_style:
            a = float(np.mean(paper_style[arm]))
            rows.append([name, arm, "A before-CV", round(a, 2), paper, round(a - paper, 2)])
            a_text = f"{a:.2f}"
        print(f"    {arm:<6}{b:>10.2f}{a_text:>15}{paper:>9.2f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Replicate Farid (2014) Tables 8-11")
    parser.add_argument("datasets", nargs="*", help="default: every dataset on disk")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--unpruned", action="store_true", help="run J48 with -U")
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    suffix = "_unpruned" if args.unpruned else ""

    RESULTS.mkdir(parents=True, exist_ok=True)
    config = {"seeds": seeds, "n_splits": N_SPLITS,
              "j48": "Weka 3.8.6 " + ("unpruned -U -M 2" if args.unpruned
                                      else "pruned, defaults -C 0.25 -M 2"),
              "nb": "FaithfulNB (add-one counts for nominal, Gaussian for numeric)",
              "attribute_space": "original columns, no one-hot",
              "protocol_B": "Algorithm 1/2 and classifier refit inside each training fold",
              "protocol_A": "Algorithm 1/2 fitted once on the whole dataset before CV"}
    (RESULTS / f"config{suffix}.json").write_text(json.dumps(config, indent=2),
                                                  encoding="utf-8")

    rows = []
    with tempfile.TemporaryDirectory() as folder:          # ARFF files, deleted at the end
        for name in args.datasets or available():
            rows += run_dataset(name, seeds, args.unpruned, Path(folder))

    out = RESULTS / f"replication{suffix}.csv"
    write_csv(out, ["dataset", "arm", "protocol", "accuracy", "paper", "delta"], rows)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
