"""E11: does reliable noise removal (DT+NB committee) plus attribute weighting
improve Logistic Regression and a deep MLP, against the same models on the raw data?

Per training fold (refit-per-fold protocol, the honest one used everywhere here):

    old       all rows, all attributes                        (the common copy)
    clean     copy A: the committee-filtered rows, all attributes
    weighted  copy B: the attributes Alg 2 keeps, scaled by its 1/sqrt(depth) weights
    new       copy A combined with copy B: filtered rows x selected, weighted attributes

The committee filter (committee_filter.py) replaces Farid's single-NB judge:
a row is deleted only when BOTH the tree and NB, cross-validated so neither ever
scores a row it trained on, misclassify it in every repeat. Alg 2 (algorithm2.py)
is the attribute selector, fit on the common copy exactly as the user's design
describes (both copies are derived from the same data, then combined).

Classifiers: Logistic Regression (sklearn) and the fundamental deep model, a
PyTorch multilayer perceptron (deep_mlp.py). Every model sees standardized
features (statistics from the training fold only). Test folds never lose rows;
they only lose the columns the selected variants lose, with the same weights.

Run from inside code/:

    python e11_lr_mlp_hybrid.py                              all 10 datasets, 5 seeds
    python e11_lr_mlp_hybrid.py --datasets iris glass        subset
    python e11_lr_mlp_hybrid.py --seeds 3                    smoke run
    python e11_lr_mlp_hybrid.py --rule majority              looser voting rule

Writes ../results/EXP-E11_lr-mlp-hybrid-data/: metrics.csv (per fold),
summary.csv (per dataset/classifier/variant, with deltas and paired tests vs old),
filter_stats.csv (what the committee deleted), versions/<dataset>/ (one full-data
old/new pair for inspection - NOT the evaluation protocol), config.json.
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
import torch
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from algorithm2 import algorithm2
from committee_filter import committee_filter
from data import available, load_data
from deep_mlp import DeepMLP
from pipeline import make_folds

warnings.filterwarnings("ignore", message="The least populated class")

SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ("old", "clean", "weighted", "new")
OUT = Path(__file__).resolve().parent.parent / "results" / "EXP-E11_lr-mlp-hybrid-data"


def make_classifier(name, seed):
    if name == "LR":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "MLP":
        return DeepMLP(seed=seed)
    raise ValueError(name)


def run_fold(X, y, train, test, seed, rule, alpha):
    """Build the four data variants from this fold's training copy, and score
    every classifier on every variant. Returns (metric rows, filter info)."""
    X_train_raw, y_train = X[train], y[train]

    # copy A: rows the DT+NB committee calls noise; copy B: Alg 2's attributes
    keep_rows, finfo = committee_filter(X_train_raw, y_train, rule=rule, seed=seed)
    keep_cols, weights = algorithm2(X_train_raw, y_train, alpha)

    # one standardizer for everything, fitted on the training fold only
    mu, sd = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    sd[sd == 0] = 1.0
    train_std = (X_train_raw - mu) / sd
    test_std = (X[test] - mu) / sd          # test rows are never removed

    variants = {
        "old":      (train_std, y_train, test_std, 100.0, 100.0),
        "clean":    (train_std[keep_rows], y_train[keep_rows], test_std,
                     100 * np.mean(keep_rows), 100.0),
        "weighted": (train_std[:, keep_cols] * weights, y_train,
                     test_std[:, keep_cols] * weights,
                     100.0, 100 * len(keep_cols) / X.shape[1]),
        "new":      (train_std[keep_rows][:, keep_cols] * weights,
                     y_train[keep_rows],
                     test_std[:, keep_cols] * weights,
                     100 * np.mean(keep_rows), 100 * len(keep_cols) / X.shape[1]),
    }

    rows = []
    for classifier in ("LR", "MLP"):
        for variant in VARIANTS:
            Xv, yv, Xv_test, rows_pct, cols_pct = variants[variant]
            model = make_classifier(classifier, seed).fit(Xv, yv)
            predicted = model.predict(Xv_test)
            rows.append({
                "classifier": classifier, "variant": variant,
                "accuracy": 100 * np.mean(predicted == y[test]),
                "macro_f1": 100 * f1_score(y[test], predicted, average="macro",
                                           zero_division=0),
                "rows_kept_pct": rows_pct, "cols_kept_pct": cols_pct,
            })
    return rows, finfo


def run_dataset(name, seeds, rule, alpha):
    X, y, meta = load_data(name)
    metric_rows, filter_rows = [], []

    for seed in seeds:
        first_row = len(metric_rows)
        for fold, (train, test) in enumerate(make_folds(y, seed)):
            rows, finfo = run_fold(X, y, train, test, seed, rule, alpha)
            for row in rows:
                row.update({"dataset": name, "seed": seed, "fold": fold})
                metric_rows.append(row)
            histogram = np.bincount(finfo["votes"],
                                    minlength=finfo["max_votes"] + 1)
            filter_rows.append({
                "dataset": name, "seed": seed, "fold": fold, "rule": rule,
                "n_rows": len(y), "removed_pct": finfo["removed_pct"],
                "skipped": finfo["skipped"],
                "votes_hist": " ".join(str(int(v)) for v in histogram),
                "per_class_removed": json.dumps(finfo["per_class"]),
            })
        seed_rows = pd.DataFrame(metric_rows[first_row:])
        means = seed_rows.groupby(["variant", "classifier"])["accuracy"].mean()
        line = "  ".join(f"{v}-{c} {means[v, c]:.2f}"
                         for v in VARIANTS for c in ("LR", "MLP"))
        print(f"  {name} seed {seed} mean acc: {line}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics, seeds)
    return metrics, pd.DataFrame(filter_rows), summary, meta


def summarize(metrics, seeds):
    """Per (dataset, classifier, variant): mean +/- std, delta and tests vs old."""
    parts = []
    for (dataset, classifier), group in metrics.groupby(["dataset", "classifier"]):
        old = group[group["variant"] == "old"].set_index(["seed", "fold"])
        for variant in VARIANTS:
            part = group[group["variant"] == variant].set_index(["seed", "fold"])
            joined = old.join(part, rsuffix="_v", how="outer")
            delta = joined["accuracy_v"] - joined["accuracy"]
            seed_means_new = part["accuracy"].groupby("seed").mean()
            seed_means_old = old["accuracy"].groupby("seed").mean()
            parts.append({
                "dataset": dataset, "classifier": classifier, "variant": variant,
                "acc_mean": part["accuracy"].mean(),
                "acc_std": part["accuracy"].std(),
                "f1_mean": part["macro_f1"].mean(),
                "delta_acc_vs_old": delta.mean(),
                "rows_kept_pct": part["rows_kept_pct"].mean(),
                "cols_kept_pct": part["cols_kept_pct"].mean(),
                "wilcoxon_p_vs_old": paired_wilcoxon(seed_means_new, seed_means_old),
                "fold_ttest_p_vs_old": paired_ttest(joined["accuracy_v"],
                                                    joined["accuracy"]),
            })
    return pd.DataFrame(parts)


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank on the per-seed means (n = len(seeds), small)."""
    diff = (a - b).dropna()
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    try:
        return float(wilcoxon(diff).pvalue)
    except ValueError:
        return float("nan")


def paired_ttest(a, b):
    """Paired t over the fold scores - many pairs but not independent, so this
    is the exploratory test; the seed-level Wilcoxon is the honest one."""
    diff = (a - b).dropna()
    if len(diff) < 2 or np.allclose(diff, 0):
        return 1.0
    return float(ttest_rel(a.loc[diff.index], b.loc[diff.index]).pvalue)


def write_versions(name, meta, rule, seed, alpha):
    """One full-data pass per dataset so 'old' vs 'new' is a file you can open.
    Inspection artifact only - the evaluation never touches these."""
    X, y, _ = load_data(name)
    keep_rows, finfo = committee_filter(X, y, rule=rule, seed=seed)
    keep_cols, weights = algorithm2(X, y, alpha)

    folder = OUT / "versions" / name
    folder.mkdir(parents=True, exist_ok=True)
    columns = meta.get("columns", [f"a{i}" for i in range(X.shape[1])])
    old = pd.DataFrame(X, columns=columns)
    old.insert(0, "class", y)
    new = old.loc[keep_rows, ["class"] + [columns[j] for j in keep_cols]].copy()
    for j, weight in zip(keep_cols, weights):
        new[columns[j]] = new[columns[j]] * weight
    old.to_csv(folder / "old.csv", index=False)
    new.to_csv(folder / "new.csv", index=False)
    pd.DataFrame({"row": finfo["removed_rows"], "class": y[finfo["removed_rows"]],
                  "votes_against": finfo["votes"][finfo["removed_rows"]]}
                 ).to_csv(folder / "removed_rows.csv", index=False)
    full_weights = np.zeros(X.shape[1])
    full_weights[keep_cols] = weights
    pd.DataFrame({"attribute": columns, "weight": full_weights,
                  "kept": [j in set(keep_cols) for j in range(X.shape[1])]}
                 ).to_csv(folder / "attribute_weights.csv", index=False)
    return finfo["removed_pct"], 100 * len(keep_cols) / X.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=available())
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--rule", choices=["consensus", "majority"],
                        default="consensus")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Alg 2 tree pruning (ccp_alpha)")
    args = parser.parse_args()
    seeds = SEEDS[:args.seeds]

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_metrics, all_filters, all_summary = [], [], []

    for i, name in enumerate(args.datasets, 1):
        print(f"[{i}/{len(args.datasets)}] {name}", flush=True)
        metrics, filters, summary, meta = run_dataset(name, seeds, args.rule,
                                                      args.alpha)
        removed_pct, kept_cols_pct = write_versions(name, meta, args.rule,
                                                    seeds[0], args.alpha)
        all_metrics.append(metrics)
        all_filters.append(filters)
        all_summary.append(summary)
        print(f"  full-data pass: committee removed {removed_pct:.1f}% of rows, "
              f"Alg 2 kept {kept_cols_pct:.1f}% of attributes", flush=True)

    metrics = pd.concat(all_metrics, ignore_index=True)
    filters = pd.concat(all_filters, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    filters.to_csv(OUT / "filter_stats.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)

    with open(OUT / "config.json", "w") as f:
        json.dump({
            "experiment": "E11",
            "question": "does DT+NB committee noise removal + Alg 2 attribute "
                        "weighting improve LR and an MLP over the raw data?",
            "datasets": args.datasets, "seeds": seeds, "rule": args.rule,
            "committee_splits": 3, "committee_repeats": 3, "alpha": args.alpha,
            "lr": {"max_iter": 2000}, "mlp": DeepMLP().__dict__,
            "sklearn": sklearn.__version__, "torch": torch.__version__,
            "n_folds": 10, "started": datetime.fromtimestamp(started).isoformat(),
            "minutes": round((time.time() - started) / 60, 1),
        }, f, indent=2)

    pivot = summary.pivot_table(index=["dataset", "classifier"],
                                columns="variant", values="acc_mean")
    print("\nAccuracy, old vs new (mean over folds):")
    print(pivot.round(2).to_string())
    print(f"\nwritten to {OUT} in {round((time.time() - started) / 60, 1)} min")


if __name__ == "__main__":
    main()
