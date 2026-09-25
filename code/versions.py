"""The three data versions of a run, and what each algorithm removed.

For every dataset, one pass over the whole table (no cross-validation) writes

    versions/<dataset>/main.csv                    the data as loaded
    versions/<dataset>/path1__Alg1_then_Alg2.csv   new data after Path 1
    versions/<dataset>/path2__Alg2_then_Alg1.csv   new data after Path 2
    versions/<dataset>/removals.txt                what was removed, step by step

Every row keeps its original row id, so the removed row ids in the report can be
checked against the files. The metrics themselves are cross-validated per fold, so the
versions are there to show the mechanics, not to compute the scores.
"""

import numpy as np
import pandas as pd

from algorithm1 import farid_algorithm1
from algorithm2 import CCP_ALPHA, farid_algorithm2
from data import load_data

PLAIN_CSV_MAX_ROWS = 5000        # bigger tables are written as .csv.gz instead

# (name of the folder and file, the algorithms in the order they run)
PATHS = [
    ("path1__Alg1_then_Alg2", ("N", "A")),
    ("path2__Alg2_then_Alg1", ("A", "N")),
]


def save_table(folder, filename, X, y, columns, row_ids):
    """Write one data version and return the path that was written."""
    path = folder / filename
    if len(X) > PLAIN_CSV_MAX_ROWS:
        path = path.with_suffix(".csv.gz")

    table = pd.DataFrame(X, columns=columns)
    table["class"] = y
    table.index = row_ids
    table.index.name = "row_id"
    table.to_csv(path)
    return path


def id_list(ids, limit=40):
    """The first few ids plus a count, so the report stays readable."""
    ids = list(ids)
    if not ids:
        return "none"
    text = ", ".join(str(int(i)) for i in ids[:limit])
    if len(ids) > limit:
        text += f" ... (+{len(ids) - limit} more)"
    return text


def trace_dataset(name, results_folder, alpha=CCP_ALPHA, support=True,
                  likelihood="gaussian"):
    """Write the three data versions and the removal report for one dataset.

    Returns (report_lines, removal_rows); removal_rows feed versions_removals.csv.
    """
    X, y, meta = load_data(name)
    columns = list(meta["columns"])
    folder = results_folder / "versions" / name
    folder.mkdir(parents=True, exist_ok=True)

    main_file = save_table(folder, "main.csv", X, y, columns, np.arange(len(y)))
    report = [
        f"dataset: {name}",
        f"main data: {X.shape[0]} rows x {X.shape[1]} attributes -> {main_file.name}",
        "(one pass over the whole dataset, no cross-validation)",
        "",
    ]
    removal_rows = []

    for tag, steps in PATHS:
        X_new = X.copy()
        y_new = y.copy()
        column_names = list(columns)
        row_ids = np.arange(len(y))
        weights = None

        report.append(f"=== {tag} ===")

        for position, step in enumerate(steps):
            if step == "N":
                # ---- Algorithm 1: the NB judge deletes instances ---------------
                rows_before = len(y_new)
                X_new, y_new, info = farid_algorithm1(X_new, y_new, weights=weights,
                                                      likelihood=likelihood)

                removed_ids = row_ids[~info["keep"]]
                row_ids = row_ids[info["keep"]]

                report.append(f"  step {position + 1}  Alg 1 -- NB noise filter "
                              f"(judge: {info['judge']})")
                report.append(f"           rows {rows_before} -> {len(y_new)}, removed "
                              f"{info['removed']} ({info['removed_rate'] * 100:.1f}%)")
                class_rates = ", ".join(f"{c}: {r * 100:.1f}%"
                                        for c, r in sorted(info["per_class"].items()))
                report.append(f"           removed per class: {class_rates}")
                if info["skipped"]:
                    report.append("           SKIPPED: removing them would wipe a class")
                report.append(f"           removed row ids: {id_list(removed_ids)}")

                removal_rows.append([name, tag, position + 1, "Alg1", info["removed"],
                                     f"{info['removed_rate'] * 100:.2f}",
                                     f"rows {rows_before}->{len(y_new)}"])
            else:
                # ---- Algorithm 2: the tree deletes attributes ------------------
                attributes_before = len(column_names)
                X_new, info = farid_algorithm2(X_new, y_new, ccp_alpha=alpha)

                kept = set(int(k) for k in info["keep"])
                removed_names = [c for j, c in enumerate(column_names) if j not in kept]
                column_names = [column_names[j] for j in info["keep"]]
                if support:
                    weights = info["weights"]
                removed_pct = 100 * len(removed_names) / attributes_before

                report.append(f"  step {position + 1}  Alg 2 -- tree attribute selection "
                              f"(ccp_alpha={alpha})")
                report.append(f"           attributes {attributes_before} -> "
                              f"{len(column_names)}, removed {len(removed_names)} "
                              f"({removed_pct:.1f}%)")
                report.append("           removed attributes: "
                              + (", ".join(removed_names) if removed_names else "none"))

                removal_rows.append([name, tag, position + 1, "Alg2", len(removed_names),
                                     f"{removed_pct:.2f}",
                                     "removed: " + (", ".join(removed_names)
                                                    if removed_names else "none")])

        saved = save_table(folder, tag + ".csv", X_new, y_new, column_names, row_ids)
        report.append(f"  new data: {X_new.shape[0]} rows x {X_new.shape[1]} attributes "
                      f"-> {saved.name}")
        report.append("")

    (folder / "removals.txt").write_text("\n".join(report), encoding="utf-8")
    return report, removal_rows
