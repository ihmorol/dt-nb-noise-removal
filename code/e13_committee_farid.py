"""E13: Farid's two algorithms composed with E5's majority committee in place
of Algorithm 1's single-NB judge.

E9f: every composition of Algorithm 1 and Algorithm 2 lands about 5 points
under plain C4.5, and the damage comes from the single-NB hard deletion. E5:
the 3-judge committee matches C4.5 on its own. E13 runs E9f's protocol-B fold
(e9_farid.run_fold, with the committee passed as its filter_fn) and asks
whether the compositions then keep C4.5's accuracy.

Twelve arms, named <path>-><final>; "M" = the committee filter (majority),
"A" = Algorithm 2 (a tree selects and weights attributes):

    baseline->NB, baseline->DT          no cleaning (anchored to E9f)
    committee->NB, committee->DT        committee only (DT anchored to E5)
    Alg2->NB, Alg2->DT                  Algorithm 2 only (anchored to E9f)
    C1 M->A->NB, C1 M->A->DT            committee on the raw fold, then
                                        Algorithm 2's tree on the kept rows
    C2 A->M->NB, C2 A->M->DT            Algorithm 2 on the raw fold, then the
                                        committee on its columns
    parallel-M->NB, parallel-M->DT      rows the raw-fold committee keeps x
                                        columns Algorithm 2 keeps on the raw
                                        fold; neither step sees the other

Choices fixed before any result (DECISIONS.md, 2026-10-20):
    * the committee is E5's, unchanged: FaithfulNB, J48 pruned defaults and
      e5_strategy.logistic_predictions, each fitted on the current training
      rows and predicting those same rows; a row is deleted iff at least two
      of the three misclassify it.
    * attributes-first (C2): all three judges see only Algorithm 2's columns,
      and the NB judge also takes its weights. J48 and the logistic judge run
      unweighted (J48 has no attribute weights; standardization cancels any
      positive column weight). The parallel arm's judges stay plain.
    * E9f's safeguards: a deletion that would leave fewer than two classes
      deletes nothing (recorded as skipped); a chained Algorithm 2 whose tree
      tests nothing keeps every column at W = 1. No per-class floor (it would
      break pairing with E5).
    * protocol B only, on E9f's folds (make_folds, seeds 0-9 x 10-fold), so
      every fold pairs with E9f's and E5's saved ones. E9f's hard-deletion
      arms are read from its per_fold.csv, not recomputed.
    * --analyze: the determinism anchors first (exact per-fold equality with
      E9f and E5, or no analysis), then the primary test, the Holm family and
      the exploratory rows, all exact Wilcoxon over the 10 dataset means.

Usage (from code/):
    python e13_committee_farid.py                       all datasets, 10 seeds
    python e13_committee_farid.py nsl-kdd --seeds 1 --out-dir <new dir>
    python e13_committee_farid.py --analyze             tests on the saved run
Writes results/EXP-E13_committee_farid/{config.json, metrics.csv, per_fold.csv,
removals.csv, judges.csv}; --analyze adds tests.csv. Needs Java.
"""
import argparse
import json
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon
from sklearn.exceptions import ConvergenceWarning

import e9_farid as e9
from data import available, load_original
from e5_strategy import J48, logistic_predictions
from faithful_nb import FaithfulNB
from pipeline import N_SPLITS, make_folds
from tables import write_csv
from weka_utils import J48_OPTIONS, weka_predicted, write_arff

ROOT = Path(__file__).resolve().parent.parent / "results"
RESULTS = ROOT / "EXP-E13_committee_farid"
PROTOCOL = "B refit-per-fold"
# E9f's arm and stage names -> E13's (the committee "M" replaces Algorithm 1's "N")
RENAME = {"Alg1->NB": "committee->NB", "Alg1->DT": "committee->DT",
          "C1 N->A->NB": "C1 M->A->NB", "C1 N->A->DT": "C1 M->A->DT",
          "C2 A->N->NB": "C2 A->M->NB", "C2 A->N->DT": "C2 A->M->DT",
          "parallel->NB": "parallel-M->NB", "parallel->DT": "parallel-M->DT",
          "N weighted (after A)": "M weighted (after A)", "N plain": "M plain",
          "A after plain N": "A after plain M"}
ARMS = [RENAME.get(arm, arm) for arm in e9.ARMS]

DELTA, ALPHA = 1.0, 0.025          # non-inferiority margin (accuracy points), one-sided alpha
E9F, E5 = "E9f ", ROOT / "EXP-E5_strategy" / "per_fold.csv"
# (family, id, left arm, right arm, margin); "E9f " arms are E9f's saved hard-deletion arms
CONFIRMATORY = [
    ("primary", "P", "C2 A->M->DT", "baseline->DT", DELTA),
    ("holm", "H1", "C1 M->A->DT", E9F + "C1 N->A->DT", 0.0),
    ("holm", "H2", "C2 A->M->DT", E9F + "C2 A->N->DT", 0.0),
    ("holm", "H3", "parallel-M->DT", E9F + "parallel->DT", 0.0),
    ("holm", "H4", "C1 M->A->DT", "baseline->DT", DELTA),
    ("holm", "H5", "parallel-M->DT", "baseline->DT", DELTA),
    # the two checks the registration ran on E5's saved committee->DT (p = 0.080, 0.0039)
    ("proxy", "R1", "committee->DT", "baseline->DT", DELTA),
    ("proxy", "R2", "committee->DT", E9F + "Alg1->DT", 0.0),
]
NB_HARD = {"committee->NB": "Alg1->NB", "C1 M->A->NB": "C1 N->A->NB",
           "C2 A->M->NB": "C2 A->N->NB", "parallel-M->NB": "parallel->NB"}
EXPLORATORY = (
    [(arm, right) for arm, hard in NB_HARD.items()
     for right in ("baseline->NB", "Alg2->NB", E9F + hard)]
    + [("C2 A->M->DT", "C1 M->A->DT"), ("C2 A->M->NB", "C1 M->A->NB")]
    + [(arm, "committee->DT") for arm in ("C1 M->A->DT", "C2 A->M->DT", "parallel-M->DT")])


# ------------------------------------------------------------ the filter --
def committee_filter(folder, classes):
    """E5's committee with filter_step's signature and record keys, so
    e9.run_fold runs it where Algorithm 1's judge was. Records also carry the
    three judges' error rates and the logistic ConvergenceWarning count."""
    path = Path(folder) / "judge.arff"

    def step(X, y, rows, columns, weights, meta):
        yr, Xr = y[rows], X[rows][:, columns]
        nb = FaithfulNB([meta["nominal"][j] for j in columns],
                        [meta["levels"][j] for j in columns], weights)
        p_nb = np.asarray(nb.fit(Xr, yr).predict(Xr), dtype=object)
        write_arff(path, X[rows], yr, meta, classes, columns)
        p_tree = np.asarray(weka_predicted(J48, path, path, J48_OPTIONS), dtype=object)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            p_log = np.asarray(logistic_predictions(Xr, yr), dtype=object)
        unconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        for w in caught:                   # anything else is shown as usual
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if not len(p_tree) == len(p_log) == len(rows):
            raise RuntimeError("judge predictions lost rows")

        wrong = np.array([p_nb != yr, p_tree != yr, p_log != yr], dtype=int)
        keep = wrong.sum(axis=0) < 2
        skipped = len(np.unique(yr[keep])) < 2
        if skipped:                        # E9f's rule: never delete down to one class
            keep = np.ones(len(rows), dtype=bool)
        record = {"method": "committee",
                  "judge": "weighted committee" if weights is not None else "plain committee",
                  "skipped": skipped, "rows_before": len(rows), "rows_after": int(keep.sum()),
                  "attributes_before": len(columns), "attributes_after": len(columns),
                  "per_class": {c: [float(np.mean(yr == c)), float(np.mean(~keep[yr == c]))]
                                for c in np.unique(yr)},
                  "judge_errors": [float(e) for e in wrong.mean(axis=1)],
                  "unconverged": unconverged}
        return rows[keep], record
    return step


# ------------------------------------------------------------------ run ---
def run(datasets, seeds):
    config = {"seeds": seeds, "n_splits": N_SPLITS, "protocol": PROTOCOL + " only",
              "j48": "Weka 3.8.6 pruned, defaults -C 0.25 -M 2",
              "nb": "FaithfulNB (add-one counts for nominal, Gaussian for numeric)",
              "attribute_space": "original columns, no one-hot",
              "committee": "E5's: FaithfulNB, J48 pruned defaults, sklearn LogisticRegression "
                           "(C=1, lbfgs, max_iter 1000, StandardScaler on the judged rows); "
                           "resubstitution on the current training rows; delete iff >= 2 "
                           "of 3 misclassify",
              "support": "C2: all judges on Algorithm 2's columns, NB judge weighted, J48 and "
                         "logistic unweighted; attributes-first and parallel final NBs use "
                         "Algorithm 2's weights; final DTs unweighted J48; parallel judges plain",
              "safety": "a deletion leaving fewer than 2 classes deletes nothing; a chained "
                        "Algorithm 2 whose tree tests nothing keeps every column at W = 1; "
                        "standalone Alg2->NB follows R1 (priors-only); no per-class floor",
              "folds": "pipeline.make_folds, the same folds as E9f and E5",
              "hard_reference": "E9f's per_fold.csv (protocol B), read at --analyze",
              "macro_f1": "sklearn convention: classes present in the fold, 0-division = 0"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics, per_fold_rows, removal_rows, judge_rows = [], [], [], []
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        for name in datasets:
            started = time.perf_counter()
            X, y, meta = load_original(name)
            classes = sorted(set(y))
            print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes, "
                  f"{meta['n_classes']} classes ---", flush=True)
            committee = committee_filter(folder, classes)
            per_arm = {}
            for seed in seeds:
                for i, (train, test) in enumerate(make_folds(y, seed)):
                    scores, records = e9.run_fold(folder, X, y, meta, classes, train,
                                                  test, filter_fn=committee)
                    records = [dict(r, stage=RENAME.get(r["stage"], r["stage"]))
                               for r in records]
                    for arm, (acc, f1) in scores.items():
                        arm = RENAME.get(arm, arm)
                        per_arm.setdefault(arm, []).append((acc, f1))
                        per_fold_rows.append([name, PROTOCOL, seed, i, arm, acc, f1])
                    removal_rows += e9.removal_rows(name, PROTOCOL, seed, i, records)
                    judge_rows += [[name, seed, i, r["stage"], r["judge"], r["rows_before"],
                                    r["attributes_before"], *r["judge_errors"],
                                    r["rows_before"] - r["rows_after"], r["skipped"],
                                    r["unconverged"]]
                                   for r in records if r["method"] == "committee"]

            print(f"    {'arm':<16}{'accuracy':>10}{'macro-F1':>10}")
            for arm in ARMS:
                acc = float(np.mean([s[0] for s in per_arm[arm]]))
                f1 = float(np.mean([s[1] for s in per_arm[arm]]))
                metrics.append([name, arm, PROTOCOL, acc, f1, len(per_arm[arm])])
                print(f"    {arm:<16}{acc:>10.2f}{f1:>10.2f}")
            mine = [r for r in judge_rows if r[0] == name]
            for stage in ("M plain", "M weighted (after A)"):
                calls = [r for r in mine if r[3] == stage]
                print(f"    {stage}: deletes {np.mean([r[10] / r[5] for r in calls]):.1%} "
                      f"of rows, {sum(r[11] for r in calls)} skips")
            print(f"    logistic ConvergenceWarnings: {sum(r[12] for r in mine)}; "
                  f"{time.perf_counter() - started:.0f} s", flush=True)

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
                       "class_share", "class_removal_rate"], removal_rows)
            write_csv(RESULTS / "judges.csv",
                      ["dataset", "seed", "fold", "stage", "judge", "rows_before",
                       "columns", "nb_error", "j48_error", "logistic_error", "deleted",
                       "skipped", "logistic_unconverged"], judge_rows)
    print("\n[saved] " + str(RESULTS))


# -------------------------------------------------------------- analyze ---
def read_folds(path):
    """A per_fold.csv with exact doubles (round_trip) and integer seed/fold keys."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if "protocol" in frame:
        frame = frame[frame.protocol == PROTOCOL]
    return frame.astype({"seed": int, "fold": int})


def check_anchors(e13, e9f, e5):
    """The registration's determinism anchors: exact per-fold accuracy and
    macro-F1 on every one of the reference's folds, or no analysis."""
    key = ["dataset", "seed", "fold"]
    anchors = [(arm, e9f, "E9f") for arm in ("baseline->NB", "baseline->DT",
                                             "Alg2->NB", "Alg2->DT")]
    for arm, ref, ref_name in anchors + [("committee->DT", e5, "E5")]:
        mine = e13[e13.arm == arm].set_index(key)[["accuracy", "macro_f1"]].sort_index()
        theirs = ref[ref.arm == arm].set_index(key)[["accuracy", "macro_f1"]].sort_index()
        same = mine.index.equals(theirs.index) and (mine.to_numpy() == theirs.to_numpy()).all()
        print(f"anchor {arm:<14} = {ref_name} on {len(theirs)} folds: "
              f"{'PASS' if same else 'FAIL'}")
        if not same:
            raise SystemExit(f"anchor {arm} failed: stop, no analysis (DECISIONS.md 2026-10-20)")
    if set(e13.arm) != set(ARMS) or e13.groupby("arm").size().nunique() != 1:
        raise SystemExit("incomplete run: every arm needs the anchors' folds")


def paired_test(d, alternative):
    """Exact Wilcoxon signed-rank over the dataset differences (manuscript convention)."""
    if np.all(d == 0):
        return np.nan, 1.0
    test = wilcoxon(d, zero_method="wilcox", alternative=alternative,
                    method=PermutationMethod(n_resamples=np.inf))
    return float(test.statistic), float(test.pvalue)


def test_row(means, family, test_id, metric, left, right, margin, alternative):
    d = (means[metric][left] - means[metric][right]).to_numpy()
    statistic, p = paired_test(d + margin, alternative)
    return {"family": family, "id": test_id, "metric": metric, "left": left,
            "right": right, "margin": margin, "alternative": alternative,
            "n_datasets": len(d), "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)), "statistic": statistic, "p_raw": p,
            "p_holm": np.nan, "alpha": np.nan, "verdict": ""}


def analyze(results):
    e13 = read_folds(results / "per_fold.csv")
    e9f = read_folds(ROOT / "EXP-E9_farid" / "per_fold.csv")
    check_anchors(e13, e9f, read_folds(E5))

    folds = pd.concat([e13, e9f.assign(arm=E9F + e9f.arm)])
    means = {metric: folds.pivot_table(index="dataset", columns="arm", values=metric,
                                       aggfunc="mean")
             for metric in ("accuracy", "macro_f1")}

    rows = [test_row(means, family, test_id, "accuracy", left, right, margin, "greater")
            for family, test_id, left, right, margin in CONFIRMATORY]
    rows[0].update(alpha=ALPHA, verdict="pass" if rows[0]["p_raw"] <= ALPHA else "fail")
    holm = [row for row in rows if row["family"] == "holm"]
    raw = np.array([row["p_raw"] for row in holm])
    order = np.argsort(raw)
    adjusted = np.empty(len(raw))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        raw[order] * np.arange(len(raw), 0, -1)))
    for row, p in zip(holm, adjusted):
        row.update(p_holm=float(p), alpha=ALPHA, verdict="pass" if p <= ALPHA else "fail")
    # exploratory: two-sided, no margin, uncorrected; macro-F1 for every comparison
    rows += [test_row(means, "exploratory", f"E{k}", "accuracy", left, right, 0.0,
                      "two-sided") for k, (left, right) in enumerate(EXPLORATORY, 1)]
    pairs = [(r["id"], r["left"], r["right"]) for r in rows]
    rows += [test_row(means, "exploratory", test_id + "-F1", "macro_f1", left, right,
                      0.0, "two-sided") for test_id, left, right in pairs]

    tests = pd.DataFrame(rows)
    tests.to_csv(results / "tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(tests[tests.metric == "accuracy"][["id", "left", "right", "margin",
                                                 "mean_delta", "p_raw", "p_holm",
                                                 "verdict"]].to_string(index=False))
    print("\n[saved] " + str(results / "tests.csv"))


def main():
    global RESULTS
    parser = argparse.ArgumentParser(description="E13: E5's committee composed with "
                                                 "Algorithm 2 in the J48/FaithfulNB world")
    parser.add_argument("datasets", nargs="*", help="default: every dataset on disk")
    parser.add_argument("--seeds", type=int, default=10,
                        help="CV repeats (default 10, the paper's protocol)")
    parser.add_argument("--out-dir", type=Path,
                        help="new empty result directory for this run (with --analyze: "
                             "the run to analyze)")
    parser.add_argument("--analyze", action="store_true",
                        help="check the anchors and write tests.csv for a finished run")
    args = parser.parse_args()
    if args.out_dir is not None:
        RESULTS = args.out_dir.resolve()
    if args.analyze:
        return analyze(RESULTS)
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        parser.error("refusing to overwrite existing results; use --out-dir with a new empty directory")
    run(args.datasets or available(), list(range(args.seeds)))


if __name__ == "__main__":
    main()
