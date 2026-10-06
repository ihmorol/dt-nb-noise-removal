"""E12: which noise filter actually finds the noise? DT and NB, compared separately.

The question E11 could not answer. E11 compared cleaning arms only through
downstream accuracy on ten datasets that are themselves clean, so "did the filter
remove the mislabeled rows or just the hard ones?" had no answer, and every delta
came back inside the noise. Here we corrupt a controlled fraction of the labels
OURSELVES (noise_inject.py), keep the corruption mask, and can therefore grade
each filter on precision and recall against ground truth.

    python e12_noise_removal.py                     all datasets, 3 seeds
    python e12_noise_removal.py --datasets iris glass --seeds 2
    python e12_noise_removal.py --rates 0.1 0.2 --kinds symmetric
    python e12_noise_removal.py --handling relabel
    python e12_noise_removal.py --quick             no MLP, 1 seed

Honesty rules, all enforced here:

  * noise is injected into the TRAINING FOLD ONLY. The test fold keeps its clean
    labels, so the accuracy we report is "how well does this training set teach",
    and no ground truth leaks across.
  * the filter is fitted inside the training fold, on the noised labels only.
  * the confusable-class map for asymmetric noise is estimated by cross-validated
    NB inside the same training fold.
  * the test fold never loses rows and never has its labels touched.

Arms, reported separately so DT and NB can be told apart:

    none                      no cleaning - the baseline to beat
    NB hard_vote              Farid Algorithm 1's rule, out-of-fold
    DT hard_vote              the same rule with a tree as judge
    DT confident_joint        per-class thresholds (Confident Learning)
    NB confident_joint        per-class thresholds
    committee consensus       E11's committee, as the reference point

Writes ../results/EXP-E12_noise_removal/: detection.csv (precision, recall, F1,
AUROC per dataset/kind/rate/arm), accuracy.csv (downstream accuracy and macro-F1
for every classifier), summary.md, config.json.
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from confident_filter import confident_filter
from data import available, load_data
from deep_mlp import DeepMLP
from nb import WeightedNB
from noise_inject import confusable_map, inject
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", message="The least populated class")

SEEDS = [0, 1, 2]
RATES = [0.0, 0.05, 0.10, 0.20, 0.40]
KINDS = ["symmetric", "asymmetric"]
OUT = Path(__file__).resolve().parent.parent / "results" / "EXP-E12_noise_removal"

ARMS = [
    ("none", None, None),
    ("NB hard_vote", "NB", "hard_vote"),
    ("DT hard_vote", "DT", "hard_vote"),
    ("DT confident_joint", "DT", "confident_joint"),
    ("NB confident_joint", "NB", "confident_joint"),
    ("committee consensus", "committee", "consensus"),
]


def make_classifier(name, seed):
    if name == "NB":
        return WeightedNB(likelihood="mixed")
    if name == "DT":
        return DecisionTreeClassifier(criterion="entropy", random_state=seed)
    if name == "LR":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "MLP":
        return DeepMLP(seed=seed)
    raise ValueError(name)
def detect_scores(keep, info, is_noisy):
    """Grade one filter against the known corruption. Returns a metrics dict.

    precision  of the rows it deleted, how many were really mislabeled
    recall     of the truly mislabeled rows, how many it caught
    auroc      ranking quality, using the judge's confidence as the score; free
               of any threshold, so it separates "the judge knows" from "the rule
               knew where to cut"
    """
    removed = ~keep
    hits = int((removed & is_noisy).sum())
    precision = hits / removed.sum() if removed.sum() else float("nan")
    recall = hits / is_noisy.sum() if is_noisy.sum() else float("nan")
    f1 = (0.0 if not (precision and recall)
          else 2 * precision * recall / (precision + recall))

    score = (np.asarray(info["confidence"], dtype=float) if info
             else np.zeros(len(is_noisy)))
    auroc = float("nan")
    if is_noisy.any() and (~is_noisy).any():
        try:
            auroc = float(roc_auc_score(is_noisy.astype(int), score))
        except ValueError:
            auroc = float("nan")
    return {
        "removed_pct": 100.0 * float(removed.mean()),
        "precision": precision, "recall": recall, "f1": f1, "auroc": auroc,
        "true_noise_pct": 100.0 * float(is_noisy.mean()),
    }


def run_fold(X, y_clean, train, test, kind, rate, seed, fold_seed, handling,
             min_per_class, classifiers):
    """One fold: inject noise into the training rows, run every arm, and score
    both the detection and the downstream classifier."""
    X_train, y_train = X[train], y_clean[train]

    mapping = None
    if kind == "asymmetric" and rate > 0:
        mapping = confusable_map(X_train, y_train, seed=fold_seed)

    # noise goes into the TRAINING FOLD ONLY; y_clean[test] stays pristine
    y_noisy, is_noisy = inject(X_train, y_train, rate, kind=kind,
                               seed=fold_seed, mapping=mapping)

    detection, accuracy = [], []
    for arm, judge, method in ARMS:
        if method is None:
            keep = np.ones(len(y_noisy), dtype=bool)
            info = None
            train_labels = y_noisy
        else:
            # E11's committee ran 3 repeats x 3 folds; keep that setting so the
            # "committee consensus" arm really is the E11 filter being compared
            repeats, splits = (3, 3) if method == "consensus" else (1, 5)
            keep, info = confident_filter(X_train, y_noisy, judge=judge,
                                          method=method, repeats=repeats,
                                          n_splits=splits,
                                          min_per_class=min_per_class,
                                          seed=fold_seed)
            train_labels = y_noisy[keep]
            if handling == "relabel" and method == "confident_joint":
                # trust the judge's own answer on the rows it disputes, but only
                # where it was confident enough to clear that class's threshold
                predicted = np.asarray(info["predicted"])
                thresholds = np.array([info["thresholds"][c]
                                       for c in info["classes"]], dtype=float)
                chosen = np.array([list(info["classes"]).index(p)
                                   for p in predicted])
                disputed = predicted != y_noisy
                confident = np.asarray(info["confidence"]) >= thresholds[chosen]
                switch = (disputed & confident)[keep]
                train_labels = np.where(switch, predicted[keep], y_noisy[keep])

        row = {"arm": arm, "judge": judge, "method": method, "kind": kind,
               "rate": rate, "handling": handling, "fold_seed": fold_seed}
        detection.append({**row, **detect_scores(keep, info, is_noisy),
                          "rescued_by_floor": info["rescued_by_floor"] if info else 0})

        for name in classifiers:
            model = make_classifier(name, seed).fit(X_train[keep], train_labels)
            predicted_labels = model.predict(X[test])
            accuracy.append({
                **row, "classifier": name,
                "accuracy": 100 * float(np.mean(predicted_labels == y_clean[test])),
                "macro_f1": 100 * float(f1_score(y_clean[test], predicted_labels,
                                                 average="macro", zero_division=0)),
            })
    return detection, accuracy
def run_dataset(name, seeds, rates, kinds, handling, min_per_class, classifiers,
                n_splits=10):
    X, y, _ = load_data(name)
    det_rows, acc_rows = [], []

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(y)), y))
        for kind in kinds:
            for rate in rates:
                for fold, (train, test) in enumerate(folds):
                    det, acc = run_fold(X, y, train, test, kind, rate, seed,
                                        seed * 1000 + fold, handling,
                                        min_per_class, classifiers)
                    for row in det + acc:
                        row.update({"dataset": name, "seed": seed, "fold": fold})
                    det_rows += det
                    acc_rows += acc
    print(f"  {name}: {len(det_rows)} detection rows, {len(acc_rows)} accuracy rows",
          flush=True)
    return det_rows, acc_rows


def summarize_detection(detection):
    """Mean precision / recall / F1 / AUROC per kind, rate and arm."""
    if detection.empty:
        return pd.DataFrame()
    return detection.groupby(["kind", "rate", "arm"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), auroc=("auroc", "mean"),
        removed_pct=("removed_pct", "mean"),
        true_noise_pct=("true_noise_pct", "mean"),
        rescued=("rescued_by_floor", "mean"),
    ).reset_index()


def summarize_accuracy(accuracy):
    """Mean accuracy per kind, rate, arm and classifier, plus delta vs 'none'."""
    if accuracy.empty:
        return pd.DataFrame()
    means = (accuracy.groupby(["kind", "rate", "arm", "classifier"])
             .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                  acc_std=("accuracy", "std"))
             .reset_index())
    base = means[means["arm"] == "none"].set_index(["kind", "rate", "classifier"])
    means = means.join(base[["accuracy"]].rename(
        columns={"accuracy": "accuracy_none"}), on=["kind", "rate", "classifier"])
    means["delta_vs_none"] = means["accuracy"] - means["accuracy_none"]
    return means


def write_report(detection, accuracy, out):
    """The human-readable summary: detection and accuracy, macro over datasets."""
    lines = ["# E12 — which filter finds the noise? (DT and NB, separately)",
             "", "Generated " + datetime.now().strftime("%Y-%m-%d %H:%M"), ""]

    summary = summarize_detection(detection)
    if not summary.empty:
        for kind in summary["kind"].unique():
            block = summary[summary["kind"] == kind]
            pivot = block.pivot_table(index="rate", columns="arm",
                                      values=["precision", "recall", "f1"])
            lines += [f"## Detection — {kind} noise", "",
                      "```", pivot.round(3).to_string(), "```", ""]

    acc = summarize_accuracy(accuracy)
    if not acc.empty:
        for classifier in acc["classifier"].unique():
            block = acc[acc["classifier"] == classifier]
            pivot = block.pivot_table(index=["kind", "rate"], columns="arm",
                                      values="delta_vs_none")
            lines += [f"## Accuracy delta vs no cleaning — {classifier} (points)",
                      "", "```", pivot.round(2).to_string(), "```", ""]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=available())
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--rates", type=float, nargs="*", default=RATES)
    parser.add_argument("--kinds", nargs="*", choices=KINDS, default=KINDS)
    parser.add_argument("--handling", choices=["delete", "relabel"], default="delete")
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--quick", action="store_true",
                        help="one seed, no MLP - a fast smoke run")
    args = parser.parse_args()

    seeds = SEEDS[:args.seeds]
    classifiers = ["NB", "DT", "LR"] if args.quick else ["NB", "DT", "LR", "MLP"]

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_det, all_acc = [], []

    for i, name in enumerate(args.datasets, 1):
        print(f"[{i}/{len(args.datasets)}] {name}", flush=True)
        det, acc = run_dataset(name, seeds, args.rates, args.kinds,
                               args.handling, args.min_per_class, classifiers)
        all_det += det
        all_acc += acc

    detection = pd.DataFrame(all_det)
    accuracy = pd.DataFrame(all_acc)
    detection.to_csv(OUT / "detection.csv", index=False)
    accuracy.to_csv(OUT / "accuracy.csv", index=False)
    summarize_detection(detection).to_csv(OUT / "detection_summary.csv", index=False)
    summarize_accuracy(accuracy).to_csv(OUT / "accuracy_summary.csv", index=False)

    with open(OUT / "config.json", "w") as fh:
        json.dump({
            "experiment": "E12",
            "question": "which noise filter actually finds injected label noise, "
                        "DT vs NB, and does removing it improve accuracy?",
            "datasets": args.datasets, "seeds": seeds, "rates": args.rates,
            "kinds": args.kinds, "handling": args.handling,
            "min_per_class": args.min_per_class, "classifiers": classifiers,
            "arms": [a for a, _, _ in ARMS], "n_splits": 10,
            "sklearn": sklearn.__version__,
            "started": datetime.fromtimestamp(started).isoformat(),
            "minutes": round((time.time() - started) / 60, 1),
        }, fh, indent=2)

    print("\n" + write_report(detection, accuracy, OUT))
    print(f"\nwritten to {OUT} in {round((time.time() - started) / 60, 1)} min")


if __name__ == "__main__":
    main()