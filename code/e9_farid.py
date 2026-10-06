"""E9 in the paper's own world: both orders of Farid's two algorithms, with
Weka J48 and the textbook NB on the original attribute columns. This is the
experiment the manuscript's Table II describes; the earlier E9 pilot (main.py)
used sklearn CART/one-hot, a different attribute space.

Ten arms, named <path>-><final>; "N" = Algorithm 1 (NB deletes instances),
"A" = Algorithm 2 (a tree selects and weights attributes):

    baseline->NB, baseline->DT          no cleaning
    Alg1->DT                            the paper's Algorithm 1
    Alg2->NB                            the paper's Algorithm 2
    Alg1->NB, Alg2->DT                  references (one step, other classifier)
    C1 N->A->NB, C1 N->A->DT            instances first, then attributes
    C2 A->N->NB, C2 A->N->DT            attributes first, then instances
    parallel->NB, parallel->DT          both algorithms on the SAME raw fold,
                                        independently: the rows the plain-NB
                                        judge keeps x the columns the tree
                                        tests; no step sees the other's output,
                                        so the C1/C2-vs-parallel difference is
                                        pure interaction

Both protocols are run for every cleaning arm:
    B  refit-per-fold (main): every step and the classifier refit in the fold
    A  before-CV: the cleaning runs once on the whole dataset, then only the
       final classifier is cross-validated on the cleaned data
Baselines have no cleaning step, so A = B and they are reported once.

Choices fixed before any result (DECISIONS.md, 2026-09-28):
    * J48 pruned, Weka defaults. One J48 run serves several arms whenever it
      is grown on the same rows and columns: the full-fold tree is the C4.5
      score AND Algorithm 2's step AND the C2 paths' first step; the tree
      grown after plain-NB filtering is the Alg1 score AND the C1 paths'
      second step. Five J48 runs per fold in total.
    * support is always on in attributes-first paths: Algorithm 2's weights go
      to the NB judge and the final NB, as the manuscript describes. The
      parallel arm's judge stays PLAIN (independence is the point of the arm)
      while its final NB uses the tree's weights, like every attributes-first
      final NB.
    * the run also writes the parallel pass's cleaned dataset to
      versions/<dataset>/parallel.csv (rows x attributes the full-data pass
      kept, original columns) with a removals report. That artifact comes from
      cleaning on the whole dataset, so it is for inspection only — the
      evaluation numbers are the CV arms.
    * if Algorithm 2's tree tests no attribute: the standalone Alg2 arm keeps
      R1's reading (Eq. 14 with all W = 0, NB on the priors); a chained path
      cannot continue with zero columns, so it keeps every column at W = 1.
    * if Algorithm 1's filter would leave fewer than two classes, nothing is
      deleted (same rule as replicate_farid.py and the manuscript).
    * macro-F1 uses sklearn's convention (average over the classes present in
      the fold, 0 where a class is never predicted) so it is comparable with
      the sklearn pipeline: from Weka's confusion matrix for J48 arms, from
      predictions for NB arms. The matrix-derived accuracy is checked against
      Weka's own summary on every scored run.

Usage (from code/):
    python e9_farid.py                    all datasets, 10 seeds (the paper's protocol)
    python e9_farid.py iris glass --seeds 1
Writes results/EXP-E9_farid/{config.json, metrics.csv, per_fold.csv, removals.csv}.
Needs Java; the Weka jars are in code/lib/.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from data import available, load_original
from faithful_nb import FaithfulNB
from pipeline import N_SPLITS, make_folds
from tables import write_csv
from weka_utils import clean_name, run_j48, tree_depths, write_arff

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E9_farid"
ARMS = ["baseline->NB", "baseline->DT", "Alg1->NB", "Alg1->DT", "Alg2->NB",
        "Alg2->DT", "C1 N->A->NB", "C1 N->A->DT", "C2 A->N->NB", "C2 A->N->DT",
        "parallel->NB", "parallel->DT"]


# --------------------------------------------------------------- scoring --
def scores_from_matrix(matrix):
    """(accuracy %, macro-F1 %) from a confusion matrix (rows = actual).

    Macro-F1 averages the F1 of the classes present in the fold's truth or
    predictions, with 0 where a class is never predicted or never correct —
    sklearn's f1_score(average="macro", zero_division=0) convention.
    """
    m = np.asarray(matrix, dtype=float)
    actual, predicted = m.sum(axis=1), m.sum(axis=0)
    hit = np.diag(m)
    precision = np.divide(hit, predicted, out=np.zeros_like(hit), where=predicted > 0)
    recall = np.divide(hit, actual, out=np.zeros_like(hit), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(hit), where=(precision + recall) > 0)
    present = (actual > 0) | (predicted > 0)
    return 100 * hit.sum() / m.sum(), 100 * f1[present].mean()


def scores_from_predictions(y_true, y_pred):
    """The same two scores from label lists (the NB arms never touch Weka)."""
    classes = sorted(set(y_true) | set(y_pred))
    index = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)))
    np.add.at(m, ([index[t] for t in y_true], [index[p] for p in y_pred]), 1)
    return scores_from_matrix(m)


def checked_scores(accuracy, matrix):
    """Matrix-derived scores, guarded against parsing drift: Weka's own summary
    (four decimals) and the matrix must agree on the accuracy."""
    scores = scores_from_matrix(matrix)
    if not np.isfinite(accuracy) or not np.isfinite(scores).all() or abs(scores[0] - accuracy) > 0.001:
        raise ValueError(f"confusion matrix disagrees with Weka's summary: {scores[0]} vs {accuracy}")
    return scores


# ----------------------------------------------------------- the steps ----
def select_attributes(depths, columns, meta, chained):
    """Algorithm 2's output on the current column set: (kept, weights, empty).

    A tree that tests none of the available columns leaves the standalone Alg2
    arm with Eq. 14 at all-zero weights (NB on the priors, R1's reading); a
    chained path needs columns to continue, so it keeps them all at weight 1.
    """
    column_of = {clean_name(meta["names"][j]): j for j in columns}
    tested = sorted(column_of[clean_name(name)] for name in depths
                    if clean_name(name) in column_of)
    if not tested:
        if chained:
            return list(columns), np.ones(len(columns)), True
        return [], np.zeros(0), True
    weights = np.array([1 / np.sqrt(depths[clean_name(meta["names"][j])])
                        for j in tested])
    return tested, weights, False


def filter_step(X, y, rows, columns, weights, meta):
    """Algorithm 1: delete the training rows the (weighted) NB judge gets wrong."""
    judge = FaithfulNB([meta["nominal"][j] for j in columns],
                       [meta["levels"][j] for j in columns], weights)
    keep = judge.fit(X[rows][:, columns], y[rows]).predict(X[rows][:, columns]) == y[rows]
    skipped = len(np.unique(y[rows][keep])) < 2
    if skipped:                        # our rule: never delete down to one class
        keep = np.ones(len(rows), dtype=bool)
    record = {"method": "Alg1", "judge": "weighted NB" if weights is not None else "plain NB",
              "skipped": skipped, "rows_before": len(rows), "rows_after": int(keep.sum()),
              "attributes_before": len(columns), "attributes_after": len(columns)}
    record["per_class"] = {c: [float(np.mean(y[rows] == c)),
                               float(np.mean(~keep[y[rows] == c]))]
                           for c in np.unique(y[rows])}
    return rows[keep], record


def attribute_record(columns, kept, empty):
    """The removal record of one Algorithm 2 step (the tree itself lives in J48's output)."""
    return {"method": "Alg2", "judge": "", "skipped": empty, "rows_before": "",
            "rows_after": "", "attributes_before": len(columns),
            "attributes_after": len(kept), "per_class": {}}


def _write(folder, tag, X, y, meta, classes, rows, columns):
    path = folder / f"{tag}.arff"
    write_arff(path, X[rows], y[rows], meta, classes, columns)
    return path


def _j48(folder, tag, X, y, meta, classes, train, test, columns):
    """One scored J48 run: train vs test ARFFs of the given columns."""
    accuracy, tree, matrix = run_j48(_write(folder, tag, X, y, meta, classes, train, columns),
                                     _write(folder, tag + "_t", X, y, meta, classes,
                                            test, columns))
    return checked_scores(accuracy, matrix), tree_depths(tree)


def _nb(meta, columns, weights):
    return FaithfulNB([meta["nominal"][j] for j in columns],
                      [meta["levels"][j] for j in columns], weights)


# --------------------------------------------------------- protocol B ----
def run_fold(folder, X, y, meta, classes, train, test):
    """All ten arms on one training fold.

    Returns ({arm: (accuracy, macro-F1)}, [removal records]). Each record is
    one cleaning event, named by its stage; the arm-to-stage mapping is in the
    module docstring.
    """
    all_columns = list(range(X.shape[1]))

    def nb_scores(rows, columns, weights):
        nb = _nb(meta, columns, weights).fit(X[rows][:, columns], y[rows])
        return scores_from_predictions(y[test], nb.predict(X[test][:, columns]))

    scores, records = {}, []

    # one tree on the full training fold: the C4.5 score, Alg2's step, C2's first step
    scores["baseline->DT"], depths_all = _j48(folder, "b1", X, y, meta, classes,
                                              train, test, all_columns)
    cols_all, w_all, empty_all = select_attributes(depths_all, all_columns, meta,
                                                   chained=True)
    records.append(dict(attribute_record(all_columns, cols_all, empty_all),
                        stage="A on the raw fold"))
    scores["baseline->NB"] = nb_scores(train, all_columns, None)
    standalone_cols, standalone_weights, _ = select_attributes(
        depths_all, all_columns, meta, chained=False)
    scores["Alg2->NB"] = nb_scores(train, standalone_cols, standalone_weights)
    scores["Alg2->DT"], _ = _j48(folder, "b5", X, y, meta, classes, train, test, cols_all)
    rows_an, rec_an = filter_step(X, y, train, cols_all, w_all, meta)
    records.append(dict(rec_an, stage="N weighted (after A)"))
    scores["C2 A->N->NB"] = nb_scores(rows_an, cols_all, w_all)
    scores["C2 A->N->DT"], _ = _j48(folder, "b4", X, y, meta, classes, rows_an, test,
                                    cols_all)

    # one tree on the plain-NB-filtered fold: the Alg1 score, C1's second step
    rows_na, rec_na = filter_step(X, y, train, all_columns, None, meta)
    records.append(dict(rec_na, stage="N plain"))
    scores["Alg1->NB"] = nb_scores(rows_na, all_columns, None)
    if np.array_equal(rows_na, train):      # nothing was deleted: same tree as b1
        scores["Alg1->DT"], depths_na = scores["baseline->DT"], depths_all
    else:
        scores["Alg1->DT"], depths_na = _j48(folder, "b2", X, y, meta, classes,
                                             rows_na, test, all_columns)
    cols_na, w_na, empty_na = select_attributes(depths_na, all_columns, meta, chained=True)
    records.append(dict(attribute_record(all_columns, cols_na, empty_na),
                        stage="A after plain N"))
    scores["C1 N->A->NB"] = nb_scores(rows_na, cols_na, w_na)
    scores["C1 N->A->DT"], _ = _j48(folder, "b3", X, y, meta, classes, rows_na, test,
                                    cols_na)

    # parallel: the same raw fold, both algorithms independently. Both inputs
    # already exist (rows_na from the plain judge, cols_all/w_all from b1), so
    # the only new computation is one J48 run on the intersection.
    scores["parallel->NB"] = nb_scores(rows_na, cols_all, w_all)
    scores["parallel->DT"], _ = _j48(folder, "b6", X, y, meta, classes, rows_na, test,
                                     cols_all)
    return scores, records


# --------------------------------------------------------- protocol A ----
def run_protocol_a(folder, name, X, y, meta, classes, seeds, baseline_dt_folds):
    """Clean once on the whole dataset, then CV only the final classifier.

    Also writes the parallel pass's cleaned dataset and removal report to
    versions/<name>/. Returns ({arm: [(accuracy, macro-F1) per fold]},
    [removal records]).
    """
    all_columns = list(range(X.shape[1]))
    everything = np.arange(len(y))
    records = []

    rows_na, rec_na = filter_step(X, y, everything, all_columns, None, meta)
    all_file = _write(folder, "a_all", X, y, meta, classes, everything, all_columns)
    depths_all = tree_depths(run_j48(all_file, all_file)[1])
    cols_all, w_all, empty_all = select_attributes(depths_all, all_columns, meta,
                                                   chained=True)
    rows_an, rec_an = filter_step(X, y, everything, cols_all, w_all, meta)
    na_file = _write(folder, "a_na", X, y, meta, classes, rows_na, all_columns)
    depths_na = tree_depths(run_j48(na_file, na_file)[1])
    cols_na, w_na, _ = select_attributes(depths_na, all_columns, meta, chained=True)
    records += [dict(rec_na, stage="N plain on the full data"),
                dict(attribute_record(all_columns, cols_all, empty_all),
                     stage="A on the full data"),
                dict(rec_an, stage="N weighted on the full data"),
                dict(attribute_record(all_columns, cols_na, False),
                     stage="A after plain N on the full data")]
    print(f"    protocol A: the filter deletes {len(y) - len(rows_na)} rows plain, "
          f"{len(y) - len(rows_an)} weighted; the full-data trees test "
          f"{len(depths_all)}/{len(all_columns)} and {len(depths_na)}/"
          f"{len(all_columns)} attributes")

    def cv(row_pool, columns, weights, tag, tree):
        """The final classifier, refit per fold over the cleaned row pool."""
        out = []
        for seed in seeds:
            for train, test in make_folds(y[row_pool], seed):
                train, test = row_pool[train], row_pool[test]
                if tree:
                    out.append(_j48(folder, tag, X, y, meta, classes, train, test,
                                    columns)[0])
                else:
                    nb = _nb(meta, columns, weights).fit(X[train][:, columns], y[train])
                    out.append(scores_from_predictions(
                        y[test], nb.predict(X[test][:, columns])))
        return out

    def priors():
        """Eq. 14 with all W = 0: predict each training fold's majority class."""
        out = []
        for seed in seeds:
            for train, test in make_folds(y, seed):
                values, counts = np.unique(y[train], return_counts=True)
                out.append(scores_from_predictions(y[test],
                                                   [values[counts.argmax()]] * len(test)))
        return out

    arms = {
        "Alg1->DT": cv(rows_na, all_columns, None, "a_alg1", True),
        "Alg1->NB": cv(rows_na, all_columns, None, "", False),
        "Alg2->NB": cv(everything, cols_all, w_all, "", False) if not empty_all
                    else priors(),
        "Alg2->DT": cv(everything, cols_all, None, "a_alg2dt", True),
        "C1 N->A->NB": cv(rows_na, cols_na, w_na, "", False),
        "C1 N->A->DT": cv(rows_na, cols_na, None, "a_c1dt", True),
        "C2 A->N->NB": cv(rows_an, cols_all, w_all, "", False),
        "C2 A->N->DT": cv(rows_an, cols_all, None, "a_c2dt", True),
        "parallel->NB": cv(rows_na, cols_all, w_all, "", False),
        "parallel->DT": cv(rows_na, cols_all, None, "a_pdt", True),
    }
    if cols_all == all_columns:
        # the full-data tree kept every column: protocol A's Alg2->DT folds are
        # the same J48 runs as protocol B's baseline tree folds, and the
        # parallel->DT folds are the same runs as protocol A's Alg1->DT folds
        arms["Alg2->DT"] = baseline_dt_folds
        arms["parallel->DT"] = arms["Alg1->DT"]
    _save_versions(name, X, y, meta, rows_na, cols_all, depths_all)
    return arms, records


def _save_versions(name, X, y, meta, rows_kept, columns, depths):
    """The new dataset from one full-data parallel pass, plus what was removed.

    rows_kept are the rows the plain-NB judge did not delete, columns the
    attributes the tree tested. This is the leaky full-data pass: an inspection
    artifact, never evaluation data.
    """
    out = RESULTS / "versions" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "parallel.csv", "w", encoding="utf-8") as f:
        f.write(",".join(str(meta["names"][j]) for j in columns) + ",class\n")
        for row, label in zip(X[rows_kept], y[rows_kept]):
            cells = [meta["levels"][j][int(row[j])] if meta["nominal"][j]
                     else f"{row[j]:.10g}" for j in columns]
            f.write(",".join(cells) + f",{label}\n")
    removed_rows = np.setdiff1d(np.arange(len(y)), rows_kept)
    removed_cols = np.setdiff1d(np.arange(X.shape[1]), np.asarray(columns))
    with open(out / "removals.txt", "w", encoding="utf-8") as f:
        f.write(f"parallel full-data pass: the plain-NB judge deleted "
                f"{len(removed_rows)}/{len(y)} rows, the tree kept "
                f"{len(columns)}/{X.shape[1]} attributes\n")
        f.write("rows deleted per class: " + json.dumps(
            {str(c): int(np.sum(y[removed_rows] == c)) for c in np.unique(y[removed_rows])})
            + "\n")
        f.write("attributes removed: " + ", ".join(str(meta["names"][j])
                                                   for j in removed_cols) + "\n")
        f.write("tree tests (min depth): " +
                json.dumps({k: int(v) for k, v in sorted(depths.items())}) + "\n")


# -------------------------------------------------------------- output ---
def removal_rows(name, protocol, seed, fold, records):
    """One CSV row per cleaning event, plus one row per class for filter steps."""
    rows = []
    for record in records:
        base = [name, protocol, seed, fold, record["stage"], record["method"],
                record["judge"], record["skipped"], record["rows_before"],
                record["rows_after"], record["attributes_before"],
                record["attributes_after"]]
        for c, (share, rate) in record["per_class"].items():
            rows.append(base + [c, share, rate])
        if not record["per_class"]:
            rows.append(base + ["", "", ""])
    return rows


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description="E9 in the J48/FaithfulNB world")
    parser.add_argument("datasets", nargs="*", help="default: every dataset on disk")
    parser.add_argument("--seeds", type=int, default=10,
                        help="CV repeats (default 10, the paper's protocol)")
    parser.add_argument("--out-dir", type=Path, help="new empty result directory for this run")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.out_dir is not None:
        RESULTS = args.out_dir.resolve()
    if RESULTS.exists() and any(RESULTS.iterdir()):
        parser.error("refusing to overwrite existing results; use --out-dir with a new empty directory")
    seeds = list(range(args.seeds))

    RESULTS.mkdir(parents=True, exist_ok=True)
    config = {"seeds": seeds, "n_splits": N_SPLITS,
              "j48": "Weka 3.8.6 pruned, defaults -C 0.25 -M 2",
              "nb": "FaithfulNB (add-one counts for nominal, Gaussian for numeric)",
              "attribute_space": "original columns, no one-hot",
              "support": "Algorithm 2's weights go to the NB judge and the final NB "
                         "in attributes-first paths (always on, as the manuscript "
                         "describes); the parallel arm's judge stays plain",
              "parallel": "both algorithms on the same raw fold, independently: "
                          "rows kept by the plain-NB judge x attributes tested by "
                          "the tree; final NB uses the tree's weights; the "
                          "full-data pass also writes versions/<dataset>/parallel.csv",
              "safety": "Alg1 never deletes down to one class; a chained path whose "
                        "tree tests no attribute keeps every column at W = 1; the "
                        "standalone Alg2 arm follows R1 (priors-only)",
              "macro_f1": "sklearn convention: classes present in the fold, "
                          "0-division = 0",
              "protocol_B": "every step refit inside each training fold",
              "protocol_A": "cleaning once on the whole dataset, then only the final "
                            "classifier cross-validated on the cleaned data"}
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics, per_fold_rows, all_removal_rows = [], [], []
    with tempfile.TemporaryDirectory() as folder:
        for name in args.datasets or available():
            X, y, meta = load_original(name)
            classes = sorted(set(y))
            print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes, "
                  f"{meta['n_classes']} classes ---", flush=True)

            b_scores, a_scores = {}, {}
            for seed in seeds:
                for i, (train, test) in enumerate(make_folds(y, seed)):
                    scores, records = run_fold(Path(folder), X, y, meta, classes,
                                               train, test)
                    for arm, (acc, f1) in scores.items():
                        b_scores.setdefault(arm, []).append((acc, f1))
                        per_fold_rows.append([name, "B refit-per-fold", seed, i, arm,
                                              acc, f1])
                    all_removal_rows += removal_rows(name, "B refit-per-fold", seed, i,
                                                     records)

            a_folds, a_records = run_protocol_a(Path(folder), name, X, y, meta,
                                                classes, seeds,
                                                b_scores["baseline->DT"])
            a_scores = {arm: folds for arm, folds in a_folds.items() if folds}
            for arm, folds in a_scores.items():
                per_fold_rows += [[name, "A before-CV", "", "", arm, acc, f1]
                                  for acc, f1 in folds]
            all_removal_rows += removal_rows(name, "A before-CV", "all", "all",
                                             a_records)

            print(f"    {'arm':<14}{'accuracy':>10}{'macro-F1':>10}{'A accuracy':>12}")
            for arm in ARMS:
                if arm not in b_scores:
                    continue
                b = b_scores[arm]
                acc, f1 = float(np.mean([s[0] for s in b])), float(np.mean([s[1] for s in b]))
                metrics.append([name, arm, "B refit-per-fold", acc, f1, len(b)])
                a_text = " " * 12
                if arm in a_scores:
                    a = a_scores[arm]
                    a_acc = float(np.mean([s[0] for s in a]))
                    a_text = f"{a_acc:>12.2f}"
                    metrics.append([name, arm, "A before-CV", a_acc,
                                    float(np.mean([s[1] for s in a])), len(a)])
                print(f"    {arm:<14}{acc:>10.2f}{f1:>10.2f}{a_text}")

            # incremental write: a long run stays useful if it is stopped early
            write_csv(RESULTS / "metrics.csv",
                      ["dataset", "arm", "protocol", "accuracy", "macro_f1", "n_folds"],
                      metrics)
            write_csv(RESULTS / "per_fold.csv",
                      ["dataset", "protocol", "seed", "fold", "arm", "accuracy",
                       "macro_f1"], per_fold_rows)
            write_csv(RESULTS / "removals.csv",
                      ["dataset", "protocol", "seed", "fold", "stage", "method",
                       "judge", "skipped", "rows_before", "rows_after",
                       "attributes_before", "attributes_after", "class",
                       "class_share", "class_removal_rate"], all_removal_rows)

    print("\n[saved] " + str(RESULTS))


if __name__ == "__main__":
    main()
