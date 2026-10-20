"""Generate submission evidence from saved E9 records; never fit models.

Run from any directory. Seed SD describes ten seed-level benchmark means,
not uncertainty over independent datasets. All plots are descriptive.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE.parents[1] / "results" / "EXP-E9_farid"
REPORTED = HERE.parents[1] / "results" / "EXP-R1_replication" / "replication.csv"  # 2014 table values
TABLES, FIGURES = HERE / "tables", HERE / "figures"
PATHS = ["baseline", "Alg1", "Alg2", "C1 N->A", "C2 A->N", "parallel"]
SHORT = dict(zip(PATHS, ["Base", "N", "A", "NA", "AN", "P"]))
ARMS = [f"{path}->{learner}" for learner in ["NB", "DT"] for path in PATHS]
COMPARISONS = [
    ("N--DT vs. DT", "Alg1->DT", "baseline->DT"),
    ("A--NB vs. NB", "Alg2->NB", "baseline->NB"),
    ("NA--DT vs. DT", "C1 N->A->DT", "baseline->DT"),
    ("AN--DT vs. DT", "C2 A->N->DT", "baseline->DT"),
    ("NA--NB vs. NB", "C1 N->A->NB", "baseline->NB"),
    ("AN--NB vs. NB", "C2 A->N->NB", "baseline->NB"),
    ("AN--DT vs. NA--DT", "C2 A->N->DT", "C1 N->A->DT"),
    ("AN--NB vs. NA--NB", "C2 A->N->NB", "C1 N->A->NB"),
]


def hashes():
    return {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            for name in ["config.json", "metrics.csv", "per_fold.csv", "removals.csv"]}


def table(name, spec, heading, rows):
    slash, newline = chr(92), chr(10)
    text = (slash + "begin{tabular}{" + spec + "}" + newline + slash + "toprule" + newline
            + heading + " " + slash * 2 + newline + slash + "midrule" + newline
            + newline.join(" & ".join(row) + " " + slash * 2 for row in rows)
            + newline + slash + "bottomrule" + newline + slash + "end{tabular}" + newline)
    (TABLES / name).write_text(text, encoding="utf-8")


def label(arm):
    path, learner = arm.rsplit("->", 1)
    return SHORT[path] + "--" + learner


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"Creator": "E9 evidence generator"})
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    before = hashes()
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    folds = pd.read_csv(RESULTS / "per_fold.csv")
    removals = pd.read_csv(RESULTS / "removals.csv")
    assert len(metrics) == 220 and len(folds) == 22000
    assert not folds[folds.protocol == "B refit-per-fold"].duplicated(["dataset", "seed", "fold", "arm"]).any()
    assert folds.groupby(["dataset", "protocol", "arm"]).size().eq(100).all()
    recorded = metrics.set_index(["dataset", "protocol", "arm"])[["accuracy", "macro_f1"]]
    calculated = folds.groupby(["dataset", "protocol", "arm"])[["accuracy", "macro_f1"]].mean()
    np.testing.assert_allclose(recorded, calculated.loc[recorded.index], atol=1e-9, rtol=0)
    b = metrics[metrics.protocol == "B refit-per-fold"]
    assert len(b) == 120 and set(b.arm) == set(ARMS)
    means = b.groupby("arm")[["accuracy", "macro_f1"]].mean().loc[ARMS]
    seed_means = (folds[folds.protocol == "B refit-per-fold"]
                  .groupby(["arm", "seed", "dataset"])[["accuracy", "macro_f1"]].mean()
                  .groupby(["arm", "seed"]).mean())
    sd = seed_means.groupby("arm").std(ddof=1).loc[ARMS]
    summary = means.rename(columns={x: x + "_mean" for x in means.columns}).join(
        sd.rename(columns={x: x + "_seed_sd" for x in sd.columns}))
    summary.to_csv(TABLES / "summary.csv")
    table("table_summary.tex", "lrrrr", r"Arm & Acc. & SD & F1 & SD",
          [[label(arm), f"{means.loc[arm, 'accuracy']:.2f}", f"{sd.loc[arm, 'accuracy']:.2f}",
            f"{means.loc[arm, 'macro_f1']:.2f}", f"{sd.loc[arm, 'macro_f1']:.2f}"] for arm in ARMS])
    acc = b.pivot(index="dataset", columns="arm", values="accuracy")
    names = {"breast-cancer": "Breast cancer", "contact-lenses": "Contact lenses",
             "image-segmentation": "Segmentation", "tic-tac-toe": "Tic-tac-toe", "nsl-kdd": "NSL-KDD"}
    sys.path.insert(0, str(HERE.parents[1] / "code"))
    from data import load_original
    audited_shapes = json.loads((HERE.parents[1] / "notes" / "audit-artifacts" / "e9-validation.json").read_text())["datasets"]
    dataset_shapes = {}
    for ds in acc.index:
        X, y, _ = load_original(ds)
        shape = [len(y), X.shape[1], len(np.unique(y)), int(pd.Series(y).value_counts().min())]
        assert shape == audited_shapes[ds], (ds, shape, audited_shapes[ds])
        dataset_shapes[ds] = shape
    pd.DataFrame.from_dict(dataset_shapes, orient="index", columns=["rows", "attributes", "classes", "min_class"]).to_csv(TABLES / "data.csv")
    table("table_data.tex", "lrrrr", "Dataset & Rows & Attr. & Classes & Min.",
          [[names.get(ds, ds.capitalize())] + list(map(str, shape)) for ds, shape in dataset_shapes.items()])
    heading = (r" & \multicolumn{6}{c}{Final NB} & \multicolumn{6}{c}{Final DT (J48)} \\",
               r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
               "Dataset & " + " & ".join([SHORT[p] for p in PATHS] * 2))
    table("table_datasets.tex", "l" + "r" * 12, chr(10).join(heading),
          [[names.get(ds, ds.capitalize())] + [f"{acc.loc[ds, arm]:.2f}" for arm in ARMS] for ds in acc.index])
    tests = []
    for name, left, right in COMPARISONS:
        delta = acc[left] - acc[right]
        test = wilcoxon(delta.to_numpy(), zero_method="wilcox", alternative="two-sided",
                        method=PermutationMethod(n_resamples=np.inf))
        tests.append(dict(comparison=name, mean_delta_pp=float(delta.mean()),
                          median_delta_pp=float(delta.median()), p_raw=float(test.pvalue), n_datasets=10))
    raw = np.array([row["p_raw"] for row in tests])
    order = np.argsort(raw)
    adjusted = np.empty(len(raw))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(raw[order] * np.arange(len(raw), 0, -1)))
    assert np.isclose(adjusted.min(), .109375) and (adjusted > .05).all()
    for row, p in zip(tests, adjusted):
        row["p_holm"] = float(p)
    pd.DataFrame(tests).to_csv(TABLES / "paired_tests.csv", index=False)
    table("table_tests.tex", "lrrr", r"Contrast & $\Delta$ & $p$ & Holm $p$",
          [[r["comparison"], f"{r['mean_delta_pp']:+.2f}", f"{r['p_raw']:.3f}", f"{r['p_holm']:.3f}"] for r in tests])
    a = metrics[metrics.protocol == "A before-CV"].groupby("arm").accuracy.mean()
    assert len(a) == 10
    protocol_rows = []
    for arm in [x for x in ARMS if not x.startswith("baseline")]:
        gap = float(a[arm] - means.loc[arm, "accuracy"])
        protocol_rows.append(dict(arm=arm, A=float(a[arm]), B=float(means.loc[arm, "accuracy"]), gap=gap))
    pd.DataFrame(protocol_rows).to_csv(TABLES / "protocols.csv", index=False)
    table("table_protocols.tex", "lrrr", r"Arm & A (\%) & B (\%) & A$-$B (pp)",
          [[label(r["arm"]), f"{r['A']:.2f}", f"{r['B']:.2f}", f"{r['gap']:+.2f}"] for r in protocol_rows])
    # N, NA and P reuse the same raw-fold plain-NB removal decision.
    rm = removals[(removals.dataset == "breast-cancer") &
                  (removals.protocol == "B refit-per-fold") & (removals.method == "Alg1")]
    assert set(rm.stage) == {"N plain", "N weighted (after A)"}
    assert rm.groupby(["stage", "class"]).size().eq(100).all()
    assert not rm.duplicated(["stage", "class", "seed", "fold"]).any()
    removal_rows = []
    for path in ["N", "NA", "AN", "P"]:
        stage = "N weighted (after A)" if path == "AN" else "N plain"
        for cls in ["no-recurrence-events", "recurrence-events"]:
            sub = rm[(rm.stage == stage) & (rm["class"] == cls)]
            removed = sub.rows_before * sub.class_share * sub.class_removal_rate
            assert (removed - removed.round()).abs().max() < 1e-8
            removal_rows.append(dict(path=path, label_class=cls,
                                     mean_within_class_removal_percent=float(100 * sub.class_removal_rate.mean()),
                                     folds=len(sub)))
    removal_data = pd.DataFrame(removal_rows)
    removal_data.to_csv(TABLES / "removal.csv", index=False)
    rates = removal_data.pivot(index="path", columns="label_class", values="mean_within_class_removal_percent")
    table("table_removal.tex", "lrr", r"Path & No recurrence (\%) & Recurrence (\%)",
          [[p, f"{rates.loc[p, 'no-recurrence-events']:.2f}", f"{rates.loc[p, 'recurrence-events']:.2f}"] for p in ["N", "NA", "AN", "P"]])
    assert font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8.2,
                         "ytick.labelsize": 9, "legend.fontsize": 8.2, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    changes = []
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3), layout="constrained")
    for ax, learner, color in zip(axes, ["NB", "DT"], ["#0072B2", "#D55E00"]):
        values = []
        for path in PATHS[1:]:
            delta = float(means.loc[f"{path}->{learner}", "accuracy"] - means.loc[f"baseline->{learner}", "accuracy"])
            changes.append(dict(learner=learner, path=SHORT[path], delta_accuracy_pp=delta))
            values.append(delta)
        ax.barh([SHORT[p] for p in PATHS[1:]], values, height=.62, color=color)
        ax.axvline(0, color="black", linewidth=.7)
        ax.invert_yaxis()
        ax.set_xlim(-6.8, 2.5)
        ax.set_xticks([-6, -4, -2, 0, 2])
        ax.set_title("Final " + ("DT (J48)" if learner == "DT" else "NB"))
        ax.set_xlabel("Accuracy change from baseline (pp)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        for y, v in enumerate(values):
            ax.text(v + (.12 if v >= 0 else -.12), y, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.2)
    pd.DataFrame(changes).to_csv(FIGURES / "accuracy_change.csv", index=False)
    save_figure(fig, "accuracy_change")
    fig, ax = plt.subplots(figsize=(3.5, 2.65), layout="constrained")
    y = np.arange(4)
    for offset, cls, title, color in [(-.18, "no-recurrence-events", "No recurrence", "#0072B2"),
                                      (.18, "recurrence-events", "Recurrence", "#D55E00")]:
        v = rates.loc[["N", "NA", "AN", "P"], cls].to_numpy()
        ax.barh(y + offset, v, height=.32, label=title, color=color)
        for yy, value in zip(y + offset, v):
            ax.text(value + .8, yy, f"{value:.1f}", va="center", fontsize=8.2)
    ax.set_yticks(y, ["N", "NA", "AN", "P"])
    ax.invert_yaxis()
    ax.set_xlim(0, 72)
    ax.set_xlabel("Within-class training rows removed (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.2)
    removal_data.to_csv(FIGURES / "class_removal.csv", index=False)
    save_figure(fig, "class_removal")

    # Per-dataset line charts in the 2014 paper's style and dataset order (its Figs. 3-4).
    order = ["breast-cancer", "contact-lenses", "diabetes", "glass", "iris", "soybean", "vote",
             "image-segmentation", "tic-tac-toe", "nsl-kdd"]
    ticks = ["Breast cancer", "Contact lenses", "Diabetes", "Glass", "Iris", "Soybean", "Vote",
             "Image seg.", "Tic-tac-toe", "NSL-KDD"]
    # Okabe-Ito hues; markers and dashes carry identity too (two pairs sit in the CVD floor band).
    style = [("#0072B2", "o", "-"), ("#D55E00", "s", "--"), ("#009E73", "^", "-."), ("#CC79A7", "D", ":")]
    reported = pd.read_csv(REPORTED).drop_duplicates(["dataset", "arm"]).pivot(index="dataset", columns="arm", values="paper")
    a_acc = metrics[metrics.protocol == "A before-CV"].pivot(index="dataset", columns="arm", values="accuracy")
    lines = pd.DataFrame({"C4.5": acc["baseline->DT"], "Algorithm 1": acc["Alg1->DT"],
                          "NB": acc["baseline->NB"], "Algorithm 2": acc["Alg2->NB"],
                          **{f"{SHORT[p]}--{l}": acc[f"{p}->{l}"] for l in ["NB", "DT"] for p in PATHS[3:]},
                          "Alg1 2014": reported["Alg1"], "Alg1 A": a_acc["Alg1->DT"],
                          "Alg2 2014": reported["Alg2"], "Alg2 A": a_acc["Alg2->NB"],
                          "C4.5 2014": reported["C4.5"], "NB 2014": reported["NB"]}).loc[order]
    lines.to_csv(FIGURES / "dataset_lines.csv")

    def panel(ax, columns, labels, title=None):
        x = np.arange(len(order))
        for column, label, (color, marker, dash) in zip(columns, labels, style):
            ax.plot(x, lines[column], color=color, marker=marker, linestyle=dash, linewidth=1.3,
                    markersize=4.2, label=label)
        ax.set_ylim(20, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x, ticks, rotation=40, ha="right", rotation_mode="anchor")
        ax.grid(axis="y", alpha=.3)
        ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=2.6, columnspacing=1)
        if title:
            ax.set_title(title, fontsize=9)

    fig, ax = plt.subplots(figsize=(3.5, 2.75), layout="constrained")
    panel(ax, ["C4.5", "Algorithm 1", "NB", "Algorithm 2"],
          ["C4.5", "Algorithm 1 (N--DT)".replace("--", "–"), "NB", "Algorithm 2 (A--NB)".replace("--", "–")])
    save_figure(fig, "single_steps")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True, layout="constrained")
    for ax, learner, base in zip(axes, ["NB", "DT"], ["NB", "C4.5"]):
        cols = [base] + [f"{SHORT[p]}--{learner}" for p in PATHS[3:]]
        panel(ax, cols, [c.replace("--", "–") for c in cols], "Final " + ("NB" if learner == "NB" else "DT (C4.5)"))
    save_figure(fig, "combined_paths")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True, layout="constrained")
    for ax, alg, base, title in [(axes[0], "Alg1", "C4.5", "Algorithm 1 (final DT)"), (axes[1], "Alg2", "NB", "Algorithm 2 (final NB)")]:
        own = "Algorithm 1" if alg == "Alg1" else "Algorithm 2"
        panel(ax, [own, f"{alg} A", f"{alg} 2014", f"{base} 2014"],
              ["Protocol B", "Protocol A", "2014 table", f"{base}, 2014 table"], title)
    save_figure(fig, "reported_protocols")

    # Every per-dataset statement in the figure explanations, checked against the data.
    assert (lines["Alg1 2014"] > lines["C4.5 2014"]).all() and (lines["Alg2 2014"] > lines["NB 2014"]).all()
    d1, d2 = lines["Algorithm 1"] - lines["C4.5"], lines["Algorithm 2"] - lines["NB"]
    assert (d1 < 0).sum() == 6 and set(d1[d1 > 0].index) == {"diabetes", "iris", "vote"} and d1.max() < 1
    assert (d2 > 0).sum() == 7 and set(d2[d2 < 0].index) == {"iris", "soybean", "nsl-kdd"}
    heavy = ["glass", "image-segmentation", "tic-tac-toe", "nsl-kdd"]
    dt = lines.loc[heavy, [f"{SHORT[p]}--DT" for p in PATHS[3:]]]
    assert (dt.max(axis=1) - dt.min(axis=1)).max() < 2.61
    assert np.isclose(dt.sub(lines.loc[heavy, "C4.5"], axis=0).max().max(), -5.93, atol=.006)
    rounded = lambda s: tuple(np.round([s.min(), s.max()], 2))
    assert (rounded(d1), d2.round(2)[["contact-lenses", "vote", "nsl-kdd"]].tolist()) == ((-17.75, .99), [7.67, 5.18, -4.62])
    assert rounded(dt.sub(lines.loc[heavy, "C4.5"], axis=0).stack()) == (-17.94, -5.93)
    nb = lines[[f"{SHORT[p]}--NB" for p in PATHS[3:]]].sub(lines["NB"], axis=0)
    assert [rounded(nb.loc[ds]) for ds in ["vote", "nsl-kdd", "glass", "contact-lenses"]] == \
        [(5.29, 5.43), (7.11, 8.75), (-15.03, -8.68), (6.33, 10.0)]
    between = (lines["Alg1 2014"] > lines["Algorithm 1"]) & (lines["Alg1 2014"] < lines["Alg1 A"])
    above = (lines["Alg2 2014"] > lines["Algorithm 2"]) & (lines["Alg2 2014"] > lines["Alg2 A"])
    assert between.sum() == 7 and above.sum() == 8
    assert hashes() == before, "Source experiment files changed during generation"
    before["EXP-R1_replication/replication.csv"] = hashlib.sha256(REPORTED.read_bytes()).hexdigest()
    report = dict(dataset_shapes=dataset_shapes, dataset_shape_crosscheck=True, source_sha256=before, model_fits=0, metrics_rows=len(metrics), fold_rows=len(folds),
                  all_saved_fold_means_match=True, statistical_family="eight original exploratory accuracy comparisons",
                  minimum_holm_p=float(adjusted.min()), source_invariance=True,
                  summary_sd="sample SD of ten seed-level equal-dataset means; descriptive, not a confidence interval",
                  removal_denominator="training rows in the same recorded class at each filtering step; mean over 100 folds",
                  shared_masks="N, NA and P share the raw-fold plain-NB mask; AN uses weighted NB after attributes",
                  macro_f1="mean of saved fold macro-F1; fold class sets may vary",
                  figure_font="Times New Roman, 8.2-10 pt at native 3.5-inch width; PDF fonttype 42",
                  checks="row counts, unique fold keys, fold summary equality, exact original test family, source hashes",
                  figure_sizes_inches={"accuracy_change": [3.5, 3.3], "class_removal": [3.5, 2.65], "single_steps": [3.5, 2.75],
                                       "combined_paths": [3.5, 4.6], "reported_protocols": [3.5, 4.6]})
    (TABLES / "evidence-check.json").write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
