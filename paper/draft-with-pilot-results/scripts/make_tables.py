"""Build every LaTeX table and scripts/numbers.txt from the results CSVs.

Every number in the manuscript comes from here or from make_figures.py. No number
is hand-typed into the .tex files. Run from paper/manuscript/:

    python scripts/make_tables.py

Reads:
    results/EXP-R1_replication/replication.csv, replication_unpruned.csv
    results/EXP-E3a_leakage_check/leakage.csv
    results/EXP-E9_pipeline/metrics_final_alpha0_mixed.csv, stages.csv
Writes:
    tables/*.tex   (booktabs)
    scripts/numbers.txt   (every number quoted in the prose, "key: value")
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
TABLES = Path(__file__).resolve().parents[1] / "tables"
NUMBERS = Path(__file__).resolve().parent / "numbers.txt"

r1 = pd.read_csv(RESULTS / "EXP-R1_replication" / "replication.csv")
r1_unpruned = pd.read_csv(RESULTS / "EXP-R1_replication" / "replication_unpruned.csv")
e3a = pd.read_csv(RESULTS / "EXP-E3a_leakage_check" / "leakage.csv")
e9 = pd.read_csv(RESULTS / "EXP-E9_pipeline" / "metrics_final_alpha0_mixed.csv")
stages = pd.read_csv(RESULTS / "EXP-E9_pipeline" / "stages.csv")

DATASETS = sorted(r1["dataset"].unique())
numbers = {}


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


# ---------------------------------------------------------------- Wilcoxon --
def paired_wilcoxon(x, y, datasets_x, datasets_y):
    """x, y indexed the same way by dataset; returns (stat, p, median diff, n)."""
    dx = dict(zip(datasets_x, x))
    dy = dict(zip(datasets_y, y))
    common = sorted(set(dx) & set(dy))
    a = np.array([dx[d] for d in common])
    b = np.array([dy[d] for d in common])
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0, 0.0, len(common)
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p), float(np.median(diff)), len(common)


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values, same order."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


tests = []  # (label, stat, p, median_diff, n)

# R1: protocol A (before-CV) vs B (refit-per-fold), Alg1 and Alg2
for arm in ["Alg1", "Alg2"]:
    a = r1[(r1.arm == arm) & (r1.protocol == "A before-CV")]
    b = r1[(r1.arm == arm) & (r1.protocol == "B refit-per-fold")]
    stat, p, med, n = paired_wilcoxon(a.accuracy.values, b.accuracy.values,
                                       a.dataset.values, b.dataset.values)
    tests.append((f"R1 protocol A vs B, {arm}", stat, p, med, n))

# E3a: protocol A (before-CV, one-hot/CART) vs B (refit-per-fold), Alg1 and Alg2
for arm in ["Alg1", "Alg2"]:
    sub = e3a[e3a.arm == arm]
    stat, p, med, n = paired_wilcoxon(sub.accuracy_before_cv.values,
                                       sub.accuracy_refit_per_fold.values,
                                       sub.dataset.values, sub.dataset.values)
    tests.append((f"E3a protocol A vs B, {arm}", stat, p, med, n))

# R1, protocol B (honest): Alg1 vs C4.5, Alg2 vs NB
b_alg1 = r1[(r1.arm == "Alg1") & (r1.protocol == "B refit-per-fold")]
b_c45 = r1[(r1.arm == "C4.5") & (r1.protocol == "B refit-per-fold")]
stat, p, med, n = paired_wilcoxon(b_alg1.accuracy.values, b_c45.accuracy.values,
                                   b_alg1.dataset.values, b_c45.dataset.values)
tests.append(("R1 honest Alg1 vs C4.5", stat, p, med, n))

b_alg2 = r1[(r1.arm == "Alg2") & (r1.protocol == "B refit-per-fold")]
b_nb = r1[(r1.arm == "NB") & (r1.protocol == "B refit-per-fold")]
stat, p, med, n = paired_wilcoxon(b_alg2.accuracy.values, b_nb.accuracy.values,
                                   b_alg2.dataset.values, b_nb.dataset.values)
tests.append(("R1 honest Alg2 vs NB", stat, p, med, n))

# E9: Path 2 (C2 A->N) vs Path 1 (C1 N->A), per final classifier
for final in ["DT", "NB"]:
    p2 = e9[(e9.arm == "C2 A->N") & (e9.final == final)]
    p1 = e9[(e9.arm == "C1 N->A") & (e9.final == final)]
    stat, p, med, n = paired_wilcoxon(p2.accuracy.values, p1.accuracy.values,
                                       p2.dataset.values, p1.dataset.values)
    tests.append((f"E9 Path 2 vs Path 1, final {final}", stat, p, med, n))

pvals = np.array([t[2] for t in tests])
adj = holm(pvals)

with open(TABLES / "table_tests.tex", "w") as f:
    f.write("% Wilcoxon signed-rank tests over the 10 datasets, Holm-adjusted within this "
            "family of 8 (n=10, low power; Demsar 2006).\n")
    f.write("\\begin{table}[t]\n\\caption{Wilcoxon signed-rank tests over the 10 datasets "
            "(n=10 pairs unless noted; two-sided; Holm-adjusted p within this table).}\n"
            "\\label{tab:tests}\n\\centering\n\\small\n"
            "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Comparison & $W$ & $p$ & med.\\ diff.\\ & Holm $p$ \\\\\n\\midrule\n")
    for (label, stat, p, med, n), padj in zip(tests, adj):
        n_note = f"{label} ($n$={n})" if n != 10 else label
        f.write(f"{esc(n_note)} & {stat:.1f} & {p:.3f} & {med:+.2f} & {padj:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

for (label, stat, p, med, n), padj in zip(tests, adj):
    key = label.lower().replace(" ", "_").replace(",", "").replace(".", "")
    numbers[f"test__{key}__W"] = f"{stat:.1f}"
    numbers[f"test__{key}__p"] = f"{p:.4f}"
    numbers[f"test__{key}__median_diff"] = f"{med:+.2f}"
    numbers[f"test__{key}__holm_p"] = f"{padj:.4f}"
    numbers[f"test__{key}__n"] = str(n)

# ---------------------------------------------------------- Table: R1 -------
b = r1[r1.protocol == "B refit-per-fold"].set_index(["dataset", "arm"])["accuracy"]
a = r1[r1.protocol == "A before-CV"].set_index(["dataset", "arm"])["accuracy"]
paper = r1.drop_duplicates(["dataset", "arm"]).set_index(["dataset", "arm"])["paper"]

with open(TABLES / "table_r1.tex", "w") as f:
    f.write("% R1: Weka J48 (pruned) / FaithfulNB replication, protocol B honest, "
            "10-fold CV x 3 seeds.\n")
    f.write("\\begin{table*}[t]\n\\caption{R1 replication (protocol B, refit-per-fold): "
            "C4.5 and NB baselines and the two hybrids, against Farid et al.\\ (2014).}\n"
            "\\label{tab:r1}\n\\centering\\small\n"
            "\\begin{tabular}{l" + "r" * 8 + "}\n\\toprule\n"
            "& \\multicolumn{2}{c}{C4.5} & \\multicolumn{2}{c}{NB} "
            "& \\multicolumn{2}{c}{Alg.\\ 1} & \\multicolumn{2}{c}{Alg.\\ 2} \\\\\n"
            "Dataset & ours & paper & ours & paper & ours & paper & ours & paper \\\\\n"
            "\\midrule\n")
    for d in DATASETS:
        row = [esc(d)]
        for arm in ["C4.5", "NB", "Alg1", "Alg2"]:
            row.append(f"{b[(d, arm)]:.2f}")
            row.append(f"{paper[(d, arm)]:.2f}")
        f.write(" & ".join(row) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

for arm in ["C4.5", "NB", "Alg1", "Alg2"]:
    vals_b = b.xs(arm, level="arm")
    vals_p = paper.xs(arm, level="arm")
    numbers[f"r1_mean_B_{arm}"] = f"{vals_b.mean():.2f}"
    numbers[f"r1_mean_paper_{arm}"] = f"{vals_p.mean():.2f}"
    numbers[f"r1_mean_delta_{arm}"] = f"{(vals_b - vals_p).mean():+.2f}"
numbers["r1_nsl_kdd_C4.5_ours"] = f"{b[('nsl-kdd', 'C4.5')]:.2f}"
numbers["r1_nsl_kdd_C4.5_paper"] = f"{paper[('nsl-kdd', 'C4.5')]:.2f}"
numbers["r1_max_abs_delta_C4.5_excl_nslkdd"] = (
    f"{(b.xs('C4.5', level='arm').drop('nsl-kdd') - paper.xs('C4.5', level='arm').drop('nsl-kdd')).abs().max():.2f}")
numbers["r1_max_abs_delta_NB"] = f"{(b.xs('NB', level='arm') - paper.xs('NB', level='arm')).abs().max():.2f}"
numbers["r1_C4.5_within_1.4_excl_nslkdd"] = str(
    int(((b.xs('C4.5', level='arm').drop('nsl-kdd') - paper.xs('C4.5', level='arm').drop('nsl-kdd')).abs() <= 1.4).sum())) + "/9"
numbers["r1_NB_within_2"] = str(
    int(((b.xs('NB', level='arm') - paper.xs('NB', level='arm')).abs() <= 2.0).sum())) + "/10"
alg1_delta = b.xs('Alg1', level='arm') - paper.xs('Alg1', level='arm')
numbers["r1_alg1_worst_dataset"] = alg1_delta.idxmin()
numbers["r1_alg1_worst_delta"] = f"{alg1_delta.min():+.2f}"
alg2_delta = b.xs('Alg2', level='arm') - paper.xs('Alg2', level='arm')
numbers["r1_alg2_worst_dataset"] = alg2_delta.idxmin()
numbers["r1_alg2_worst_delta"] = f"{alg2_delta.min():+.2f}"

# unpruned sensitivity: mean C4.5 / Alg1 accuracy, protocol B
ub = r1_unpruned[r1_unpruned.protocol == "B refit-per-fold"]
for arm in ["C4.5", "Alg1"]:
    numbers[f"r1_unpruned_mean_B_{arm}"] = f"{ub[ub.arm == arm].accuracy.mean():.2f}"

# ---------------------------------------------------------- Table: E3a -----
with open(TABLES / "table_e3a.tex", "w") as f:
    f.write("% E3a: sklearn entropy CART / one-hot / mixed-likelihood NB, "
            "protocol B honest vs A before-CV vs C fully leaky.\n")
    f.write("\\begin{table}[t]\n\\caption{E3a leakage check, mean accuracy over 10 datasets "
            "(entropy CART, one-hot attributes, mixed-likelihood NB).}\n"
            "\\label{tab:e3a}\n\\centering\\small\n"
            "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Arm & B honest & A before-CV & C on all data & A$-$B \\\\\n\\midrule\n")
    for arm in ["baseline", "Alg1", "Alg2", "C1 N->A", "C2 A->N"]:
        sub = e3a[e3a.arm == arm]
        if arm == "baseline":
            for final in ["NB", "DT"]:
                s = sub[sub.final == final]
                f.write(f"{esc(arm)} ({final}) & {s.accuracy_refit_per_fold.mean():.2f} & "
                        f"{s.accuracy_before_cv.mean():.2f} & "
                        f"{s.accuracy_classifier_on_all_data.mean():.2f} & "
                        f"{(s.accuracy_before_cv - s.accuracy_refit_per_fold).mean():+.2f} \\\\\n")
        else:
            f.write(f"{esc(arm)} & {sub.accuracy_refit_per_fold.mean():.2f} & "
                    f"{sub.accuracy_before_cv.mean():.2f} & "
                    f"{sub.accuracy_classifier_on_all_data.mean():.2f} & "
                    f"{(sub.accuracy_before_cv - sub.accuracy_refit_per_fold).mean():+.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

for arm in ["Alg1", "Alg2", "C1 N->A", "C2 A->N"]:
    sub = e3a[e3a.arm == arm]
    key = arm.replace(" ", "_").replace(">", "").replace("-", "")
    numbers[f"e3a_mean_B_{key}"] = f"{sub.accuracy_refit_per_fold.mean():.2f}"
    numbers[f"e3a_mean_A_{key}"] = f"{sub.accuracy_before_cv.mean():.2f}"
    numbers[f"e3a_mean_AminusB_{key}"] = f"{(sub.accuracy_before_cv - sub.accuracy_refit_per_fold).mean():+.2f}"

# ---------------------------------------------------------- Table: E9 ------
with open(TABLES / "table_e9.tex", "w") as f:
    f.write("% E9: refit-per-fold, 10-fold CV x 5 seeds, alpha=0, mixed NB, weights on.\n")
    f.write("\\begin{table}[t]\n\\caption{E9 pipeline, mean accuracy over 10 datasets "
            "($\\pm$ across-dataset SD), 10-fold CV $\\times$ 5 seeds.}\n"
            "\\label{tab:e9}\n\\centering\\small\n"
            "\\begin{tabular}{lrr}\n\\toprule\n"
            "Arm & final DT & final NB \\\\\n\\midrule\n")
    label = {"baseline": "baseline (no cleaning)", "Alg1": "Algorithm 1 alone",
              "Alg2": "Algorithm 2 alone", "C1 N->A": "Path 1 (Alg1 $\\to$ Alg2)",
              "C2 A->N": "Path 2 (Alg2 $\\to$ Alg1)"}
    for arm in ["baseline", "Alg1", "Alg2", "C1 N->A", "C2 A->N"]:
        dt = e9[(e9.arm == arm) & (e9.final == "DT")].accuracy
        nb = e9[(e9.arm == arm) & (e9.final == "NB")].accuracy
        f.write(f"{label[arm]} & {dt.mean():.2f} $\\pm$ {dt.std():.2f} & "
                f"{nb.mean():.2f} $\\pm$ {nb.std():.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

for arm in ["baseline", "Alg1", "Alg2", "C1 N->A", "C2 A->N"]:
    key = arm.replace(" ", "_").replace(">", "").replace("-", "")
    for final in ["DT", "NB"]:
        v = e9[(e9.arm == arm) & (e9.final == final)].accuracy
        numbers[f"e9_mean_{key}_{final}"] = f"{v.mean():.2f}"

# rows removed by Alg1 alone, mean over datasets (also used by make_figures.py)
alg1_rows = stages[(stages.arm == "Alg1") & (stages.algorithm == "Alg1")]
numbers["e9_alg1_mean_rows_removed_pct"] = f"{alg1_rows.rows_removed_pct.mean():.2f}"
numbers["e9_alg2_mean_attrs_kept_pct"] = f"{(100 - stages[(stages.arm == 'Alg2') & (stages.algorithm == 'Alg2')].attrs_removed_pct).mean():.2f}"

dt_change = (e9[(e9.arm == "Alg1") & (e9.final == "DT")].set_index("dataset").accuracy
             - e9[(e9.arm == "baseline") & (e9.final == "DT")].set_index("dataset").accuracy)
worst = dt_change.idxmin()
numbers["e9_worst_dt_drop_dataset"] = worst
numbers["e9_worst_dt_drop_value"] = f"{dt_change.min():+.2f}"
best = dt_change.idxmax()
numbers["e9_best_dt_change_dataset"] = best
numbers["e9_best_dt_change_value"] = f"{dt_change.max():+.2f}"

numbers["n_datasets"] = str(len(DATASETS))

with open(NUMBERS, "w") as f:
    for k in sorted(numbers):
        f.write(f"{k}: {numbers[k]}\n")

print(f"wrote {TABLES/'table_tests.tex'}, table_r1.tex, table_e3a.tex, table_e9.tex, "
      f"and {NUMBERS} ({len(numbers)} numbers)")
