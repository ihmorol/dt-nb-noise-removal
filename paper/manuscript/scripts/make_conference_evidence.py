from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE.parents[1] / "results" / "EXP-E9_farid"
TABLES = HERE / "tables"
ARMS = [f"{path}->{final}" for final in ("NB", "DT")
        for path in ("baseline", "Alg1", "Alg2", "C1 N->A", "C2 A->N")]
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


def main():
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    b = metrics[metrics.protocol == "B refit-per-fold"]
    assert not b.duplicated(["dataset", "arm"]).any()
    acc = b.pivot(index="dataset", columns="arm", values="accuracy")
    assert len(acc) == 10 and acc[ARMS].notna().all().all()
    folds = pd.read_csv(RESULTS / "per_fold.csv")
    folds = folds[folds.protocol == "B refit-per-fold"]
    assert not folds.duplicated(["dataset", "arm", "seed", "fold"]).any()
    assert folds.groupby(["dataset", "arm"]).size().eq(100).all()
    recalculated = folds.groupby(["dataset", "arm"])[["accuracy", "macro_f1"]].mean()
    recorded = b.set_index(["dataset", "arm"])[["accuracy", "macro_f1"]]
    assert np.allclose(recalculated.loc[recorded.index], recorded, atol=1e-9)

    names = {"breast-cancer": "Breast cancer", "contact-lenses": "Contact lenses",
             "image-segmentation": "Segmentation", "tic-tac-toe": "Tic-tac-toe",
             "nsl-kdd": "NSL-KDD"}
    body = "\n".join(names.get(ds, ds.capitalize()) + " & " +
                     " & ".join(f"{acc.loc[ds, arm]:.2f}" for arm in ARMS) + r" \\"
                     for ds in acc.index)
    table = (r"\begin{tabular}{lrrrrrrrrrr}" + "\n" + r"\toprule" + "\n" +
             r" & \multicolumn{5}{c}{Final NB} & \multicolumn{5}{c}{Final C4.5} \\" + "\n" +
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}" + "\n" +
             r"Dataset & Base & N & A & NA & AN & Base & N & A & NA & AN \\" + "\n" +
             r"\midrule" + "\n" + body + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    TABLES.mkdir(exist_ok=True)
    (TABLES / "table_conference_datasets.tex").write_text(table, encoding="utf-8")

    rows = []
    for label, left, right in COMPARISONS:
        delta = acc[left] - acc[right]
        test = wilcoxon(delta.to_numpy(), zero_method="wilcox",
                        alternative="two-sided",
                        method=PermutationMethod(n_resamples=np.inf))
        rows.append({"comparison": label, "mean_delta_pp": delta.mean(),
                     "median_delta_pp": delta.median(), "statistic": test.statistic,
                     "p_raw": test.pvalue, "n_datasets": len(delta)})
    raw = np.array([row["p_raw"] for row in rows])
    order = np.argsort(raw)
    adjusted = np.empty(len(raw))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        raw[order] * np.arange(len(raw), 0, -1)))
    for row, p in zip(rows, adjusted):
        row["p_holm"] = p
    pd.DataFrame(rows).to_csv(TABLES / "paired_tests.csv", index=False)
    body = "\n".join(f"{row['comparison']} & {row['mean_delta_pp']:+.2f} & "
                     f"{row['p_raw']:.3f} & {row['p_holm']:.3f}" + r" \\"
                     for row in rows)
    (TABLES / "table_conference_tests.tex").write_text(
        r"\begin{tabular}{lrrr}" + "\n" + r"\toprule" + "\n" +
        r"Comparison & $\Delta$ (pp) & $p$ & Holm $p$ \\" + "\n" +
        r"\midrule" + "\n" + body + "\n" + r"\bottomrule" + "\n" +
        r"\end{tabular}" + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Stored means match all protocol-B fold records; conference tables saved.")


if __name__ == "__main__":
    main()
