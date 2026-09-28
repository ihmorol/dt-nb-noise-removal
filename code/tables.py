"""The tables main.py prints, and the CSV files it writes."""
import csv

import numpy as np

from data import FARID_2014
from pipeline import N_SPLITS

# the two paths x the two final classifiers
HEADLINE = ["C1 N->A->NB", "C1 N->A->DT", "C2 A->N->NB", "C2 A->N->DT"]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def print_arm_header():
    print(f"{'arm':<16}{'final':<5}{'accuracy':>9}{'std':>7}{'macroF1':>9}"
          f"{'rows removed':>14}{'attrs kept':>12}")


def print_arm_row(arm, final, result, seconds):
    print(f"{arm:<16}{final:<5}{result['accuracy']:>9.2f}{result['accuracy_std']:>7.2f}"
          f"{result['macro_f1']:>9.2f}{result['instances_removed_pct']:>13.1f}%"
          f"{result['attributes_kept_pct']:>11.1f}%   ({seconds:.0f}s)")


def print_headline(all_results):
    """Both paths with both final classifiers, one line per dataset."""
    print("\n" + "=" * 96)
    print("THE FOUR RESULTS: accuracy %, macro-F1 in brackets")
    print("=" * 96)
    print(f"{'dataset':<20}" + "".join(f"{key:>19}" for key in HEADLINE))
    for name, results in all_results.items():
        cells = [f"{results[key]['accuracy']:>12.2f} ({results[key]['macro_f1']:4.1f})"
                 for key in HEADLINE]
        print(f"{name:<20}" + "".join(cells))


def print_farid_comparison(all_results):
    """Our four replication arms next to Farid's Tables 8-11."""
    print("\n" + "=" * 71)
    print("COMPARISON WITH FARID (2014) TABLES 8-11: 10-fold CV accuracy %")
    print("=" * 71)
    print(f"{'dataset':<20}{'classifier':<20}{'Farid 2014':>11}{'this run':>11}{'delta':>9}")

    pairs = [("C4.5", "DT", "baseline->DT"), ("Alg 1 (hybrid DT)", "Alg1", "Alg1->DT"),
             ("NB", "NB", "baseline->NB"), ("Alg 2 (hybrid NB)", "Alg2", "Alg2->NB")]
    paper_values, our_values = [], []
    for name, results in all_results.items():
        for label, paper_key, our_key in pairs:
            paper = FARID_2014[name][paper_key]
            ours = results[our_key]["accuracy"]
            print(f"{name:<20}{label:<20}{paper:>11.2f}{ours:>11.2f}{ours - paper:>+9.2f}")
            paper_values.append(paper)
            our_values.append(ours)
        print("-" * 71)

    paper_mean, our_mean = np.mean(paper_values), np.mean(our_values)
    print(f"{'MEAN over all cells':<40}{paper_mean:>11.2f}{our_mean:>11.2f}"
          f"{our_mean - paper_mean:>+9.2f}")


def metrics_rows(all_results, seeds, settings):
    """One row per dataset, arm and final classifier (for metrics.csv)."""
    rows = []
    for name, results in all_results.items():
        for key, r in results.items():
            arm, final = key.rsplit("->", 1)
            rows.append([name, arm, final, f"{r['accuracy']:.4f}", f"{r['accuracy_std']:.4f}",
                         f"{r['macro_f1']:.4f}", f"{r['instances_removed_pct']:.2f}",
                         f"{r['attributes_kept_pct']:.2f}", seeds, N_SPLITS,
                         settings.alpha, settings.likelihood, settings.support])
    return rows


METRICS_HEADER = ["dataset", "arm", "final", "accuracy", "accuracy_std", "macro_f1",
                  "instances_removed_pct", "attributes_kept_pct", "seeds", "n_splits",
                  "ccp_alpha", "nb_likelihood", "support"]


def stage_rows(name, results):
    """One row per arm and algorithm step, averaged over the folds (for stages.csv)."""
    rows = []
    for key, result in results.items():
        arm, final = key.rsplit("->", 1)
        for s in result["stages"]:
            rows.append([name, arm, final, s["stage"], s["method"],
                         f"{s['rows_before']:.1f}", f"{s['rows_after']:.1f}",
                         f"{s['rows_removed_pct']:.2f}", f"{s['attributes_before']:.1f}",
                         f"{s['attributes_after']:.1f}", f"{s['attributes_removed_pct']:.2f}",
                         f"{s['skipped_pct']:.1f}"])
    return rows


STAGES_HEADER = ["dataset", "arm", "final", "stage", "algorithm", "rows_before", "rows_after",
                 "rows_removed_pct", "attrs_before", "attrs_after", "attrs_removed_pct",
                 "skipped_pct"]

REMOVALS_HEADER = ["dataset", "path", "step", "algorithm", "removed", "removed_pct", "detail"]
