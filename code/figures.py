"""The paper's Figures 2 and 3, with our hybrid added as a third bar.

Farid's Figure 2 compares C4.5 with his hybrid DT (Algorithm 1) on the ten datasets.
Farid's Figure 3 compares plain NB with his hybrid NB (Algorithm 2). Both charts are
recreated here from the paper's own tables (data.py -> FARID_2014), and our mutual
hybrid -- Path 2, the attributes-first order -- is added next to them.

Read the colours honestly: the first two bars of each chart are the numbers the paper
reports (its protocol is unstated); the third bar is measured here with everything
refit inside each training fold (10-fold CV x 5 seeds). The two sources are not
protocol-matched, which is exactly what the leakage audit (E3a) has to quantify.

Usage: python figures.py            reads metrics_final_alpha0_mixed.csv
       python figures.py OTHER.csv  reads another results file
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no window, just files
import matplotlib.pyplot as plt             # noqa: E402
import pandas as pd                         # noqa: E402

from data import FARID_2014                 # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E9_pipeline"
FIGURES = RESULTS / "figures"

# Table 5 order, so the charts look like the paper's
DATASETS = ["breast-cancer", "contact-lenses", "diabetes", "glass", "iris",
            "soybean", "vote", "image-segmentation", "tic-tac-toe", "nsl-kdd"]
SHORT_NAMES = ["Breast\ncancer", "Contact\nlenses", "Diabetes", "Glass", "Iris",
               "Soybean", "Vote", "Image\nseg.", "Tic-Tac-\nToe", "NSL-\nKDD"]

PAPER_COLOR = "#7f7f7f"          # the paper's numbers
HYBRID_COLOR = "#d95f02"         # our hybrid
OFFSET = 0.26                    # bar width used by grouped_bars


def our_accuracy(metrics_file, arm, final):
    """Per-dataset accuracy of one arm, read from a results file."""
    table = pd.read_csv(metrics_file)
    rows = table[(table["arm"] == arm) & (table["final"] == final)]
    return dict(zip(rows["dataset"], rows["accuracy"]))


def grouped_bars(filename, title, series, note):
    """Draw one grouped bar chart: dataset on x, accuracy % on y."""
    positions = range(len(DATASETS))
    width = 0.8 / len(series)

    figure, axes = plt.subplots(figsize=(13, 6))
    for index, (label, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        bars = axes.bar([p + offset for p in positions], values, width,
                        label=label, color=color, edgecolor="white", linewidth=0.4)
        axes.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1, rotation=90)

    axes.set_xticks(list(positions))
    axes.set_xticklabels(SHORT_NAMES, fontsize=9)
    axes.set_ylabel("Classification accuracy (%)")
    axes.set_ylim(0, 115)                    # room for the value labels
    axes.set_title(title, fontsize=12)
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)
    axes.legend(loc="upper center", ncol=len(series), fontsize=9, frameon=False,
                bbox_to_anchor=(0.5, -0.10))
    figure.text(0.5, 0.005, note, ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / filename, dpi=300)
    plt.close(figure)
    print(f"[saved] {FIGURES / filename}")


def main(metrics_file):
    hybrid_dt = our_accuracy(metrics_file, "C2 A->N", "DT")
    hybrid_nb = our_accuracy(metrics_file, "C2 A->N", "NB")

    figure_two(hybrid_dt)
    figure_three(hybrid_nb)
    figure_four(hybrid_dt, hybrid_nb)


def figure_two(hybrid_dt):
    """Farid's Figure 2: C4.5 vs his hybrid DT, plus our hybrid."""
    grouped_bars(
        "fig2_hybrid_dt.png",
        "Hybrid DT: C4.5, Farid's Algorithm 1, and our mutual hybrid (Path 2, DT final)",
        [("C4.5 (Farid 2014)", [FARID_2014[d]["DT"] for d in DATASETS], PAPER_COLOR),
         ("Hybrid DT, Algorithm 1 (Farid 2014)", [FARID_2014[d]["Alg1"] for d in DATASETS], "#4d4d4d"),
         ("Our hybrid: Alg 2 -> Alg 1 -> DT", [hybrid_dt[d] for d in DATASETS], HYBRID_COLOR)],
        "Grey bars are the paper's reported 10-fold accuracy; the orange bar is this work "
        "(leakage-safe 10-fold CV x 5 seeds, entropy tree).",
    )


def figure_three(hybrid_nb):
    """Farid's Figure 3: plain NB vs his hybrid NB, plus our hybrid."""
    grouped_bars(
        "fig3_hybrid_nb.png",
        "Hybrid NB: Naive Bayes, Farid's Algorithm 2, and our mutual hybrid (Path 2, NB final)",
        [("Naive Bayes (Farid 2014)", [FARID_2014[d]["NB"] for d in DATASETS], PAPER_COLOR),
         ("Hybrid NB, Algorithm 2 (Farid 2014)", [FARID_2014[d]["Alg2"] for d in DATASETS], "#4d4d4d"),
         ("Our hybrid: Alg 2 -> Alg 1 -> NB", [hybrid_nb[d] for d in DATASETS], HYBRID_COLOR)],
        "Grey bars are the paper's reported 10-fold accuracy; the orange bar is this work "
        "(leakage-safe 10-fold CV x 5 seeds, mixed-likelihood NB).",
    )


def figure_four(hybrid_dt, hybrid_nb):
    """All four paper classifiers next to our hybrid, both final classifiers."""
    grouped_bars(
        "fig4_all_classifiers.png",
        "All classifiers on the ten datasets, with our mutual hybrid added",
        [("C4.5 (Farid 2014)", [FARID_2014[d]["DT"] for d in DATASETS], "#7f7f7f"),
         ("Hybrid DT, Alg 1 (Farid 2014)", [FARID_2014[d]["Alg1"] for d in DATASETS], "#4d4d4d"),
         ("NB (Farid 2014)", [FARID_2014[d]["NB"] for d in DATASETS], "#c7c7c7"),
         ("Hybrid NB, Alg 2 (Farid 2014)", [FARID_2014[d]["Alg2"] for d in DATASETS], "#1b1b1b"),
         ("Our hybrid -> DT", [hybrid_dt[d] for d in DATASETS], HYBRID_COLOR),
         ("Our hybrid -> NB", [hybrid_nb[d] for d in DATASETS], "#fdb863")],
        "Grey bars are Farid's four reported classifiers; the two orange bars are this work "
        "(Path 2: Alg 2 -> Alg 1, leakage-safe 10-fold CV x 5 seeds).",
    )


if __name__ == "__main__":
    metrics = sys.argv[1] if len(sys.argv) > 1 else "metrics_final_alpha0_mixed.csv"
    main(RESULTS / metrics if not Path(metrics).is_absolute() else Path(metrics))
