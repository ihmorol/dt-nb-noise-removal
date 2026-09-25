"""Farid's Figures 2 and 3, with our Path 2 hybrid added as an extra bar.

Grey bars are the numbers the paper reports (its protocol is not stated).
Orange bars are ours, with every step refit inside each training fold.
The two sources use different protocols; the leakage check (E3a) measures that gap.

Usage:
    python figures.py                  reads metrics_final_alpha0_mixed.csv
    python figures.py other.csv
Writes ../results/EXP-E9_pipeline/figures/fig2..fig4 (.png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save files only, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from data import FARID_2014

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E9_pipeline"
FIGURES = RESULTS / "figures"

# Table 5 order, like the paper's charts
DATASETS = ["breast-cancer", "contact-lenses", "diabetes", "glass", "iris",
            "soybean", "vote", "image-segmentation", "tic-tac-toe", "nsl-kdd"]
LABELS = ["Breast\ncancer", "Contact\nlenses", "Diabetes", "Glass", "Iris",
          "Soybean", "Vote", "Image\nseg.", "Tic-Tac-\nToe", "NSL-\nKDD"]


def paper(key):
    return [FARID_2014[d][key] for d in DATASETS]


def bar_chart(filename, title, bars, note):
    """bars: list of (legend label, one value per dataset, colour)."""
    width = 0.8 / len(bars)
    figure, axes = plt.subplots(figsize=(13, 6))
    for i, (label, values, colour) in enumerate(bars):
        x = [d + (i - (len(bars) - 1) / 2) * width for d in range(len(DATASETS))]
        drawn = axes.bar(x, values, width, label=label, color=colour, edgecolor="white")
        axes.bar_label(drawn, fmt="%.1f", fontsize=6.5, padding=1, rotation=90)

    axes.set_xticks(range(len(DATASETS)))
    axes.set_xticklabels(LABELS, fontsize=9)
    axes.set_ylabel("Classification accuracy (%)")
    axes.set_ylim(0, 115)                   # room for the value labels
    axes.set_title(title, fontsize=12)
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)
    axes.legend(loc="upper center", ncol=len(bars), fontsize=9, frameon=False,
                bbox_to_anchor=(0.5, -0.10))
    figure.text(0.5, 0.005, note, ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / filename, dpi=300)
    plt.close(figure)
    print(f"[saved] {FIGURES / filename}")


def main(metrics_file):
    table = pd.read_csv(metrics_file)
    path2 = table[table["arm"] == "C2 A->N"]
    ours_dt = [path2[(path2["final"] == "DT") & (path2["dataset"] == d)]["accuracy"].item()
               for d in DATASETS]
    ours_nb = [path2[(path2["final"] == "NB") & (path2["dataset"] == d)]["accuracy"].item()
               for d in DATASETS]
    note = (f"Grey: accuracy reported by Farid (2014). Orange: this work, Path 2 "
            f"(Alg 2 -> Alg 1), every step refit per fold, 10-fold CV x "
            f"{table['seeds'].iloc[0]} seeds.")

    bar_chart("fig2_hybrid_dt.png", "Hybrid DT: C4.5, Farid's Algorithm 1, and our Path 2 -> DT",
              [("C4.5 (Farid 2014)", paper("DT"), "#7f7f7f"),
               ("Algorithm 1 (Farid 2014)", paper("Alg1"), "#4d4d4d"),
               ("Ours: Alg 2 -> Alg 1 -> DT", ours_dt, "#d95f02")], note)
    bar_chart("fig3_hybrid_nb.png", "Hybrid NB: NB, Farid's Algorithm 2, and our Path 2 -> NB",
              [("Naive Bayes (Farid 2014)", paper("NB"), "#7f7f7f"),
               ("Algorithm 2 (Farid 2014)", paper("Alg2"), "#4d4d4d"),
               ("Ours: Alg 2 -> Alg 1 -> NB", ours_nb, "#d95f02")], note)
    bar_chart("fig4_all_classifiers.png", "All classifiers on the ten datasets",
              [("C4.5 (Farid 2014)", paper("DT"), "#7f7f7f"),
               ("Alg 1 (Farid 2014)", paper("Alg1"), "#4d4d4d"),
               ("NB (Farid 2014)", paper("NB"), "#c7c7c7"),
               ("Alg 2 (Farid 2014)", paper("Alg2"), "#1b1b1b"),
               ("Ours -> DT", ours_dt, "#d95f02"),
               ("Ours -> NB", ours_nb, "#fdb863")], note)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "metrics_final_alpha0_mixed.csv"
    main(RESULTS / name)
