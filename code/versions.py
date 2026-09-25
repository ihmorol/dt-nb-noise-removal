"""Save the data before and after each path, and what each algorithm removed.

This is one pass over the WHOLE dataset (no cross-validation). It shows what the
algorithms do; the scores come from the cross-validated run, not from these files.

    versions/<dataset>/main.csv                    the data as loaded
    versions/<dataset>/path1__Alg1_then_Alg2.csv   the data after Path 1
    versions/<dataset>/path2__Alg2_then_Alg1.csv   the data after Path 2
    versions/<dataset>/removals.txt                what was removed, step by step

Every row keeps its original row id, so the removed ids can be checked in the files.
"""
import numpy as np
import pandas as pd

from data import load_data
from pipeline import clean

PATHS = {"path1__Alg1_then_Alg2": ("N", "A"), "path2__Alg2_then_Alg1": ("A", "N")}


def save_table(folder, filename, X, y, column_names, row_ids):
    """Write one data version. Tables over 5000 rows are gzipped."""
    path = folder / (filename + (".csv.gz" if len(y) > 5000 else ".csv"))
    table = pd.DataFrame(X, columns=column_names, index=pd.Index(row_ids, name="row_id"))
    table["class"] = y
    table.to_csv(path)
    return path


def short_list(items, limit=40):
    """The first `limit` items, then how many more there are."""
    items = [str(item) for item in items]
    if not items:
        return "none"
    text = ", ".join(items[:limit])
    if len(items) > limit:
        text += f" ... (+{len(items) - limit} more)"
    return text


def trace_dataset(name, results_folder, alpha, support, likelihood):
    """Write the data versions and removals.txt for one dataset.

    Returns (report lines, rows for versions_removals.csv).
    """
    X, y, meta = load_data(name)
    names = np.array(meta["columns"])
    folder = results_folder / "versions" / name
    folder.mkdir(parents=True, exist_ok=True)

    main_file = save_table(folder, "main", X, y, names, np.arange(len(y)))
    report = [f"dataset: {name}",
              f"main data: {X.shape[0]} rows x {X.shape[1]} attributes -> {main_file.name}",
              "(one pass over the whole dataset, no cross-validation)", ""]
    csv_rows = []

    for tag, steps in PATHS.items():
        X_new, y_new, _, rows, columns, log = clean(X, y, steps, alpha, support, likelihood)
        report.append(f"=== {tag} ===")

        for number, step in enumerate(log, start=1):
            if step["method"] == "Alg1":
                removed = len(step["removed_rows"])
                pct = 100 * removed / step["rows_before"]
                per_class = ", ".join(f"{c}: {100 * rate:.1f}%"
                                      for c, rate in sorted(step["per_class"].items()))
                report += [f"  step {number}  Alg 1, NB noise filter (judge: {step['judge']})",
                           f"           rows {step['rows_before']} -> {step['rows_after']}, "
                           f"removed {removed} ({pct:.1f}%)",
                           f"           removed per class: {per_class}"]
                if step["skipped"]:
                    report.append("           SKIPPED: fewer than two classes would be left")
                report.append(f"           removed row ids: {short_list(step['removed_rows'])}")
                detail = f"rows {step['rows_before']}->{step['rows_after']}"
            else:
                removed_names = names[step["removed_columns"]]
                removed = len(removed_names)
                pct = 100 * removed / step["attributes_before"]
                report += [f"  step {number}  Alg 2, tree attribute selection (ccp_alpha={alpha})",
                           f"           attributes {step['attributes_before']} -> "
                           f"{step['attributes_after']}, removed {removed} ({pct:.1f}%)",
                           f"           removed attributes: {short_list(removed_names, 1000)}"]
                detail = "removed: " + short_list(removed_names, 1000)
            csv_rows.append([name, tag, number, step["method"], removed, f"{pct:.2f}", detail])

        saved = save_table(folder, tag, X_new, y_new, names[columns], rows)
        report += [f"  new data: {X_new.shape[0]} rows x {X_new.shape[1]} attributes "
                   f"-> {saved.name}", ""]

    (folder / "removals.txt").write_text("\n".join(report), encoding="utf-8")
    return report, csv_rows
