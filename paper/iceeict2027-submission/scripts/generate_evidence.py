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
from scipy.stats import PermutationMethod, spearmanr, wilcoxon

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
    f1 = b.pivot(index="dataset", columns="arm", values="macro_f1").loc[acc.index]
    table("table_datasets_f1.tex", "l" + "r" * 12, chr(10).join(heading),
          [[names.get(ds, ds.capitalize())] + [f"{f1.loc[ds, arm]:.2f}" for arm in ARMS] for ds in f1.index])
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
    # N, NA and P share one mask, so the figure shows the two distinct judges only.
    assert np.allclose(rates.loc["N"], rates.loc["NA"]) and np.allclose(rates.loc["N"], rates.loc["P"])
    fig, ax = plt.subplots(figsize=(3.5, 1.6), layout="constrained")
    y = np.arange(2)
    for offset, cls, title, color in [(-.18, "no-recurrence-events", "No recurrence", "#0072B2"),
                                      (.18, "recurrence-events", "Recurrence", "#D55E00")]:
        v = rates.loc[["N", "AN"], cls].to_numpy()
        ax.barh(y + offset, v, height=.32, label=title, color=color)
        for yy, value in zip(y + offset, v):
            ax.text(value + .8, yy, f"{value:.1f}", va="center", fontsize=8.2)
    ax.set_yticks(y, ["Plain judge\n(N, NA, P)", "Supported\njudge (AN)"])
    ax.invert_yaxis()
    ax.set_xlim(0, 72)
    ax.set_xlabel("Within-class training rows removed (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.2)
    removal_data.to_csv(FIGURES / "class_removal.csv", index=False)
    save_figure(fig, "class_removal")

    # Deletion share and attribute retention for every dataset (one record per fold and stage).
    rec = removals[removals.protocol == "B refit-per-fold"]
    filt = rec[rec.method == "Alg1"].drop_duplicates(["dataset", "seed", "fold", "stage"])
    attrs = rec[rec.method == "Alg2"]
    assert filt.groupby(["dataset", "stage"]).size().eq(100).all()
    assert attrs.groupby(["dataset", "stage"]).size().eq(100).all()
    deleted = (100 * (1 - filt.rows_after.astype(float) / filt.rows_before.astype(float))
               ).groupby([filt.dataset, filt.stage]).mean().unstack()
    kept = (100 * attrs.attributes_after / attrs.attributes_before
            ).groupby([attrs.dataset, attrs.stage]).mean().unstack()
    effect = pd.DataFrame({"deleted_plain": deleted["N plain"],
                           "deleted_supported": deleted["N weighted (after A)"],
                           "kept_raw_fold": kept["A on the raw fold"],
                           "kept_after_plain_n": kept["A after plain N"],
                           "n_dt_minus_dt_pp": acc["Alg1->DT"] - acc["baseline->DT"]}).loc[acc.index]
    effect.to_csv(TABLES / "deletion.csv")
    nsl = rec[(rec.dataset == "nsl-kdd") & (rec.stage == "N plain")].groupby("class").class_removal_rate.mean()
    nsl.mul(100).rename("mean_within_class_removal_percent").to_csv(TABLES / "nsl_class_removal.csv")
    assert len(nsl) == 22 and (nsl > .5).sum() == 8 and (nsl < .01).sum() == 9  # counts quoted in the text
    # Descriptive claims quoted in the text about Table VII and the deletion figure.
    assert (effect.kept_after_plain_n < effect.kept_raw_fold).all()
    heavy = effect[effect.deleted_plain > 25]
    assert len(heavy) == 4 and (heavy.n_dt_minus_dt_pp < 0).all()
    assert np.isclose(spearmanr(effect.deleted_plain, effect.n_dt_minus_dt_pp).statistic, -.709, atol=5e-4)
    table("table_deletion.tex", "lrrrr",
          chr(10).join((r" & \multicolumn{2}{c}{Rows removed (\%)} & \multicolumn{2}{c}{Attributes kept (\%)} \\",
                        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                        r"Dataset & Plain & Supported & Raw fold & After N")),
          [[names.get(ds, ds.capitalize())] + [f"{effect.loc[ds, c]:.2f}" for c in effect.columns[:4]]
           for ds in effect.index])
    fig, ax = plt.subplots(figsize=(3.5, 2.45), layout="constrained")
    ax.axhline(0, color="black", linewidth=.7)
    ax.scatter(effect.deleted_plain, effect.n_dt_minus_dt_pp, s=30, color="#D55E00", zorder=3,
               edgecolors="white", linewidths=.8)
    nudge = {"iris": (4, 5), "contact-lenses": (4, -9), "vote": (5, 3), "soybean": (5, -3),
             "diabetes": (5, 2), "breast-cancer": (5, 0), "glass": (-5, 6)}
    for ds, row in effect.iterrows():
        dx, dy = nudge.get(ds, (5, 0))
        ax.annotate(names.get(ds, ds.capitalize()), (row.deleted_plain, row.n_dt_minus_dt_pp),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8.2, va="center",
                    ha="right" if dx < 0 else "left")
    ax.set_xlim(0, 52)
    ax.set_ylim(-19.5, 3)
    ax.set_xlabel("Training rows removed by the plain judge (%)")
    ax.set_ylabel("N--DT minus DT (pp)".replace("--", "–"))
    ax.set_axisbelow(True)
    ax.grid(alpha=.2)
    save_figure(fig, "deletion_effect")

    # The 2014 table values beside both protocols, per dataset (descriptive; files differ).
    reported_raw = pd.read_csv(REPORTED)
    reported = reported_raw.drop_duplicates(["dataset", "arm"]).pivot(index="dataset", columns="arm", values="paper")
    assert set(reported.index) == set(acc.index)
    a_acc = metrics[metrics.protocol == "A before-CV"].pivot(index="dataset", columns="arm", values="accuracy")
    compare = pd.DataFrame({"alg1_reported": reported["Alg1"], "alg1_B": acc["Alg1->DT"], "alg1_A": a_acc["Alg1->DT"],
                            "alg2_reported": reported["Alg2"], "alg2_B": acc["Alg2->NB"], "alg2_A": a_acc["Alg2->NB"]}).loc[acc.index]
    compare.to_csv(FIGURES / "reported_protocols.csv")
    between = (compare.alg1_reported > compare.alg1_B) & (compare.alg1_reported < compare.alg1_A)
    above = (compare.alg2_reported > compare.alg2_B) & (compare.alg2_reported > compare.alg2_A)
    assert between.sum() == 7 and above.sum() == 8  # counts quoted in the Discussion
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.0), sharey=True, layout="constrained")
    y = np.arange(len(compare))
    for ax, alg, title in zip(axes, ["alg1", "alg2"], ["Algorithm 1 (final DT)", "Algorithm 2 (final NB)"]):
        ax.hlines(y, compare[f"{alg}_B"], compare[f"{alg}_A"], color="#BBBBBB", linewidth=1.2, zorder=1)
        for col, color, marker, name in [("B", "#0072B2", "o", "Protocol B"), ("A", "#D55E00", "^", "Protocol A"),
                                         ("reported", "#009E73", "D", "2014 table")]:
            ax.scatter(compare[f"{alg}_{col}"], y, s=28, color=color, marker=marker, zorder=3,
                       edgecolors="white", linewidths=.6, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(40, 102)
        ax.set_xticks([40, 60, 80, 100])
        ax.set_xlabel("Accuracy (%)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(y, [names.get(ds, ds.capitalize()) for ds in compare.index])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False,
               handletextpad=.2, columnspacing=.9)
    save_figure(fig, "reported_protocols")
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
                  figure_sizes_inches={"accuracy_change": [3.5, 3.3], "class_removal": [3.5, 1.6], "deletion_effect": [3.5, 2.45], "reported_protocols": [3.5, 3.0]})
    (TABLES / "evidence-check.json").write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
