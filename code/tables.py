"""The tables a run prints, and the CSV files it leaves behind."""

import csv

import numpy as np

from data import FARID_2014

# the four results the pipeline is about: two paths x two final classifiers
HEADLINE_ARMS = [
    ("C1 N->A", "NB"), ("C1 N->A", "DT"),
    ("C2 A->N", "NB"), ("C2 A->N", "DT"),
]


def print_arm_header():
    print(f"{'arm':<16}{'final':<5}{'accuracy':>9}{'std':>7}{'macroF1':>9}"
          f"{'rows removed':>14}{'attrs kept':>12}")


def print_arm_row(path, final, result, seconds):
    print(f"{path:<16}{final:<5}{result['accuracy']:>9.2f}{result['accuracy_std']:>7.2f}"
          f"{result['macro_f1']:>9.2f}{result['instances_removed_pct']:>13.1f}%"
          f"{result['attributes_kept_pct']:>11.1f}%   ({seconds:.0f}s)")


def print_headline(all_results):
    """The four results side by side: which path, which final classifier."""
    print("\n" + "=" * 96)
    print("THE FOUR RESULTS -- both paths, both classifiers (accuracy %, macro-F1 in brackets)")
    print("=" * 96)
    print(f"{'dataset':<20}{'Path 1 N->A ->NB':>19}{'Path 1 N->A ->DT':>19}"
          f"{'Path 2 A->N ->NB':>19}{'Path 2 A->N ->DT':>19}")
    for name, results in all_results.items():
        line = ""
        for path, final in HEADLINE_ARMS:
            cell = results[f"{path}->{final}"]
            line += f"{cell['accuracy']:>14.2f} ({cell['macro_f1']:.1f})"
        print(f"{name:<20}{line}")


def print_farid_comparison(all_results):
    """Our four replication arms next to Farid's Tables 8-11 numbers."""
    print("\n" + "=" * 96)
    print("COMPARISON WITH FARID (2014) TABLES 8-11 -- accuracy % on 10-fold CV")
    print("=" * 96)
    print(f"{'dataset':<20}{'classifier':<20}{'Farid 2014':>11}{'this run':>11}{'delta':>9}")

    farid_values = []
    our_values = []
    for name, results in all_results.items():
        farid = FARID_2014.get(name)
        if farid is None:
            continue
        pairs = [
            ("C4.5", farid["DT"], results["baseline->DT"]["accuracy"]),
            ("Alg 1 (hybrid DT)", farid["Alg1"], results["Alg1->DT"]["accuracy"]),
            ("NB", farid["NB"], results["baseline->NB"]["accuracy"]),
            ("Alg 2 (hybrid NB)", farid["Alg2"], results["Alg2->NB"]["accuracy"]),
        ]
        for label, paper_value, our_value in pairs:
            print(f"{name:<20}{label:<20}{paper_value:>11.2f}{our_value:>11.2f}"
                  f"{our_value - paper_value:>+9.2f}")
            farid_values.append(paper_value)
            our_values.append(our_value)
        print("-" * 71)

    if farid_values:
        print(f"{'MEAN over all cells':<40}{np.mean(farid_values):>11.2f}"
              f"{np.mean(our_values):>11.2f}"
              f"{np.mean(our_values) - np.mean(farid_values):>+9.2f}")


def stage_rows_for(dataset, results):
    """One row per algorithm per arm: what it removed, averaged over the folds."""
    rows = []
    for key, result in results.items():
        path, final = key.rsplit("->", 1)
        for stage in result["stages"]:
            rows.append([
                dataset, path, final, stage["stage"], stage["method"],
                f"{stage['rows_before']:.1f}", f"{stage['rows_after']:.1f}",
                f"{stage['rows_removed_pct']:.2f}",
                f"{stage['attributes_before']:.1f}", f"{stage['attributes_after']:.1f}",
                f"{stage['attributes_removed_pct']:.2f}", f"{stage['skipped_pct']:.1f}",
            ])
    return rows


def write_metrics(folder, filename, all_results, settings):
    """One row per dataset, arm and final classifier."""
    with open(folder / filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "arm", "final", "accuracy", "accuracy_std", "macro_f1",
                         "instances_removed_pct", "attributes_kept_pct",
                         "seeds", "n_splits", "ccp_alpha", "nb_likelihood", "support"])
        for dataset, results in all_results.items():
            for key, result in results.items():
                path, final = key.rsplit("->", 1)
                writer.writerow([dataset, path, final,
                                 f"{result['accuracy']:.4f}", f"{result['accuracy_std']:.4f}",
                                 f"{result['macro_f1']:.4f}",
                                 f"{result['instances_removed_pct']:.2f}",
                                 f"{result['attributes_kept_pct']:.2f}",
                                 settings["seeds"], settings["n_splits"],
                                 settings["alpha"], settings["nb"], settings["support"]])


def write_stages(folder, stage_rows):
    """One row per dataset, arm, final classifier and algorithm stage."""
    with open(folder / "stages.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "arm", "final", "stage", "algorithm", "rows_before",
                         "rows_after", "rows_removed_pct", "attrs_before", "attrs_after",
                         "attrs_removed_pct", "skipped_pct"])
        writer.writerows(stage_rows)


def write_removals(folder, removal_rows):
    """What the whole-dataset pass removed: counts, percentages and names."""
    with open(folder / "versions_removals.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "path", "step", "algorithm", "removed", "removed_pct",
                         "detail"])
        writer.writerows(removal_rows)
