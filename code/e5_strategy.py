"""E5: graded noise handling -- can treating suspected noise more gently than
Farid's hard deletion give a better classifier, on both the instance and the
attribute side?

Three independent judges are fitted on the raw training fold (never on test):
    FaithfulNB (the seed paper's judge), J48 (pruned defaults), Weka Logistic.
Each treatment then decides what happens to a training row:
    hard      delete every row the NB judge misclassifies (Farid's Algorithm 1;
              not rerun here -- identical folds and engines make E9f's
              per-fold numbers directly comparable)
    soft      no deletion; every row keeps influence w_i = P_judge(y_i|x_i)
              (fractional counts for NB, ARFF instance weights for J48)
    correct   relabel a row to c' iff at least two of the three judges predict
              the same c' != y_i; nothing is deleted or reweighted
    committee delete a row iff at least two of the three judges misclassify it
              (Brodley & Friedl's majority partnership)

Arms (protocol B, the honest one; 10 seeds x stratified 10-fold):
    DT finals (J48 on treated rows, all columns):   soft, correct, committee
    NB finals (FaithfulNB, fractional counts):      baseline, soft, attr,
                                                    softattr, correct
        attr = Algorithm 2's 1/sqrt(min-depth) weights from the raw-fold tree,
        so the NB block is the 2x2 of {instance treatment x attribute weights}
        plus the correction arm.

Mathematical backing, pre-registered in DECISIONS.md (2026-09-28) before any
run: hard deletion removes exactly the rows the judge finds hardest (the
boundary mass, whose labels are usually correct); posterior weighting is the
plug-in of the label-quality posterior P(correct|x,y) (importance-reweighting
for label noise, Liu & Tao 2015; the self-confidence term of confident
learning, Northcutt et al. 2021); majority voting suppresses false deletions
quadratically under judge-error independence (Condorcet; measured by
Brodley & Friedl 1999); correction keeps the row's real feature information
and replaces a label that is more probably wrong than right when the judges
agree (Jiang et al. 2024: correction beats filtering at dataset level).

Success criteria fixed in advance: primary endpoint is soft vs hard on DT
finals (paired over the same folds); secondary: every arm vs the no-cleaning
baselines (E9f's, same folds). Beating hard deletion is the claim the
literature supports; beating the baseline on mostly-clean datasets is NOT
promised by any theory -- where it happens (genuine label noise), it is a
per-dataset finding, not an average one.

Usage (from code/):
    python e5_strategy.py                    all datasets, 10 seeds
    python e5_strategy.py iris --seeds 1
Writes results/EXP-E5_strategy/{config.json, metrics.csv, per_fold.csv,
treatments.csv, judges.csv}. Needs Java; the Weka jars are in code/lib/.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import available, load_original
from e9_farid import checked_scores, scores_from_predictions, select_attributes
from faithful_nb import FaithfulNB
from pipeline import N_SPLITS, make_folds
from tables import write_csv
from weka_utils import J48_OPTIONS, run_j48, tree_depths, weka_predicted, write_arff

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E5_strategy"

NB_ARMS = ["baseline->NB", "soft->NB", "attr->NB", "softattr->NB", "correct->NB"]
DT_ARMS = ["soft->DT", "correct->DT", "committee->DT"]
ARMS = NB_ARMS + DT_ARMS
J48 = "weka.classifiers.trees.J48"


def logistic_predictions(X_train, y_train):
    """The third judge: a discriminative-linear model, standardized on the
    training fold. (Weka's Logistic was unusable here: it fits unscaled
    features, and NSL-KDD's byte-count ranges defeat both its default and a
    ridge of 1.0 -- two 600-second timeouts. Same model family, sklearn
    implementation, deterministic.)"""
    scaled = StandardScaler().fit_transform(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaled, y_train)
    return model.predict(scaled)


# ----------------------------------------------------------------- arms ---
def nb_fit_predict(X, y, meta, train, test, columns, attr_weights, y_train,
                   sample_weight):
    """Fit FaithfulNB on the treated training rows, score on the test fold."""
    nb = FaithfulNB([meta["nominal"][j] for j in columns],
                    [meta["levels"][j] for j in columns], attr_weights)
    nb.fit(X[train][:, columns], y_train, sample_weight=sample_weight)
    return scores_from_predictions(y[test], nb.predict(X[test][:, columns]))


def j48_fit_score(folder, tag, X, y, meta, classes, train, test, y_train, weights):
    """J48 on the treated training rows (optional labels/weights), scored on the fold."""
    train_file = folder / f"{tag}.arff"
    write_arff(train_file, X[train], y_train, meta, classes, weights=weights)
    test_file = folder / f"{tag}_t.arff"
    write_arff(test_file, X[test], y[test], meta, classes)
    accuracy, _, matrix = run_j48(train_file, test_file)
    return checked_scores(accuracy, matrix)


# ------------------------------------------------------------ per fold ----
def run_fold(folder, X, y, meta, classes, train, test):
    """All E5 arms on one training fold.

    Returns ({arm: (accuracy, macro-F1)}, treatments rows, judges row).
    """
    all_columns = list(range(X.shape[1]))
    ytr = y[train]

    # the three judges, all fitted on the raw training fold only
    train_file = folder / "tr.arff"
    write_arff(train_file, X[train], ytr, meta, classes)
    _, tree_all, _ = run_j48(train_file, train_file)          # the tree itself
    depths_all = tree_depths(tree_all)
    cols_all, w_all, empty_all = select_attributes(depths_all, all_columns,
                                                   meta, chained=True)

    judge = FaithfulNB([meta["nominal"][j] for j in all_columns],
                       [meta["levels"][j] for j in all_columns], None)
    judge.fit(X[train], ytr)
    log_post = judge.class_scores(X[train])
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    class_column = {c: k for k, c in enumerate(judge.classes_)}
    w_soft = post[np.arange(len(train)), [class_column[c] for c in ytr]]

    p_nb = np.asarray(judge.predict(X[train]), dtype=object)
    p_tree = np.asarray(weka_predicted(J48, train_file, train_file, J48_OPTIONS),
                        dtype=object)
    p_log = np.asarray(logistic_predictions(X[train], ytr), dtype=object)
    assert len(p_tree) == len(p_log) == len(train), "judge predictions lost rows"

    wrong = np.array([p_nb != ytr, p_tree != ytr, p_log != ytr], dtype=int)
    delete_committee = wrong.sum(axis=0) >= 2

    relabel = np.full(len(train), "", dtype=object)
    for a, b in [(p_nb, p_tree), (p_nb, p_log), (p_tree, p_log)]:
        agree = (a == b) & (a != ytr)
        relabel[agree] = a[agree]
    y_correct = np.where(relabel == "", ytr, relabel)

    scores, treatments = {}, []
    scores["baseline->NB"] = nb_fit_predict(X, y, meta, train, test, all_columns,
                                            None, ytr, None)
    scores["soft->NB"] = nb_fit_predict(X, y, meta, train, test, all_columns,
                                        None, ytr, w_soft)
    scores["attr->NB"] = nb_fit_predict(X, y, meta, train, test, cols_all,
                                        w_all, ytr, None)
    scores["softattr->NB"] = nb_fit_predict(X, y, meta, train, test, cols_all,
                                            w_all, ytr, w_soft)
    scores["correct->NB"] = nb_fit_predict(X, y, meta, train, test, all_columns,
                                           None, y_correct, None)
    scores["soft->DT"] = j48_fit_score(folder, "sdt", X, y, meta, classes, train,
                                       test, ytr, w_soft)
    scores["correct->DT"] = j48_fit_score(folder, "cdt", X, y, meta, classes, train,
                                          test, y_correct, None)
    keep = train[~delete_committee]
    scores["committee->DT"] = j48_fit_score(folder, "mdt", X, y, meta, classes,
                                            keep, test, y[keep], None)

    for c in np.unique(ytr):
        in_c = ytr == c
        treatments.append([c, float(np.mean(in_c)),
                           int(np.sum(in_c & delete_committee)),
                           int(np.sum(in_c & (relabel != ""))),
                           int(np.sum(in_c & (w_soft < 0.5))),
                           float(np.mean(w_soft[in_c]))])
    judges = [float(np.mean(wrong[0])), float(np.mean(wrong[1])),
              float(np.mean(wrong[2])), len(depths_all) / len(all_columns),
              int(delete_committee.sum()), float(np.mean(w_soft)),
              int(np.sum(w_soft < 0.5))]
    return scores, treatments, judges


# ------------------------------------------------------------------ main --
def main():
    parser = argparse.ArgumentParser(description="E5: graded noise handling")
    parser.add_argument("datasets", nargs="*", help="default: every dataset on disk")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    RESULTS.mkdir(parents=True, exist_ok=True)
    config = {
        "seeds": seeds, "n_splits": N_SPLITS,
        "judges": "FaithfulNB (plain), J48 pruned defaults, sklearn "
                  "LogisticRegression (C=1, standardized on the training fold); "
                  "all fitted on the raw training fold only",
        "soft": "w_i = P_judge(y_i|x_i); no deletion; fractional counts / instance weights",
        "correct": "relabel to c' iff >= 2 of 3 judges predict the same c' != y_i",
        "committee": "delete iff >= 2 of 3 judges misclassify (majority partnership)",
        "attr": "Algorithm 2's 1/sqrt(min-depth) weights from the raw-fold J48 tree",
        "hard_reference": "E9f's Alg1->DT and baseline arms (same folds, same engines)",
        "protocol": "B refit-per-fold only",
        "endpoints": "primary: soft->DT vs hard->DT paired over folds; secondary: "
                     "each arm vs baseline on the same folds",
    }
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics, per_fold_rows, treatment_rows, judge_rows = [], [], [], []
    with tempfile.TemporaryDirectory() as folder:
        for name in args.datasets or available():
            X, y, meta = load_original(name)
            classes = sorted(set(y))
            print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes, "
                  f"{meta['n_classes']} classes ---", flush=True)

            per_arm = {}
            for seed in seeds:
                for i, (train, test) in enumerate(make_folds(y, seed)):
                    scores, treatments, judges = run_fold(Path(folder), X, y, meta,
                                                          classes, train, test)
                    for arm, (acc, f1) in scores.items():
                        per_arm.setdefault(arm, []).append((acc, f1))
                        per_fold_rows.append([name, seed, i, arm, acc, f1])
                    for c, share, deleted, relabelled, low, mean_w in treatments:
                        treatment_rows.append([name, seed, i, c, share, deleted,
                                               relabelled, low, mean_w])
                    judge_rows.append([name, seed, i] + judges)

            print(f"    {'arm':<15}{'accuracy':>10}{'macro-F1':>10}")
            for arm in ARMS:
                if arm not in per_arm:
                    continue
                acc = float(np.mean([s[0] for s in per_arm[arm]]))
                f1 = float(np.mean([s[1] for s in per_arm[arm]]))
                metrics.append([name, arm, acc, f1, len(per_arm[arm])])
                print(f"    {arm:<15}{acc:>10.2f}{f1:>10.2f}")

            write_csv(RESULTS / "metrics.csv",
                      ["dataset", "arm", "accuracy", "macro_f1", "n_folds"], metrics)
            write_csv(RESULTS / "per_fold.csv",
                      ["dataset", "seed", "fold", "arm", "accuracy", "macro_f1"],
                      per_fold_rows)
            write_csv(RESULTS / "treatments.csv",
                      ["dataset", "seed", "fold", "class", "class_share", "deleted",
                       "relabelled", "low_weight", "mean_weight"], treatment_rows)
            write_csv(RESULTS / "judges.csv",
                      ["dataset", "seed", "fold", "nb_error", "j48_error",
                       "logistic_error", "attributes_kept", "committee_deleted",
                       "soft_mean_weight", "soft_low_weight"], judge_rows)

    print("\n[saved] " + str(RESULTS))


if __name__ == "__main__":
    main()
