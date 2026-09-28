"""Build the two figures (vector PDF) from the results CSVs. Run from paper/manuscript/:

    python scripts/make_figures.py

fig1: Farid-reported vs protocol B (honest) vs protocol A (before-CV) for Alg.1/Alg.2, per
      dataset (R1, results/EXP-R1_replication/replication.csv).
fig2: rows removed by Algorithm 1 alone vs the accuracy change of the downstream tree
      (E9, results/EXP-E9_pipeline/{stages,metrics_final_alpha0_mixed}.csv).
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})

SHORT = {"breast-cancer": "breast", "contact-lenses": "lenses", "diabetes": "diabetes",
          "glass": "glass", "image-segmentation": "imgseg", "iris": "iris",
          "nsl-kdd": "nsl-kdd", "soybean": "soybean", "tic-tac-toe": "t-t-t", "vote": "vote"}

# -------------------------------------------------------------------- fig 1 --
r1 = pd.read_csv(RESULTS / "EXP-R1_replication" / "replication.csv")
datasets = sorted(r1["dataset"].unique())
x = range(len(datasets))
w = 0.26

fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
for ax, arm, title in zip(axes, ["Alg1", "Alg2"], ["Algorithm 1 (hybrid DT)", "Algorithm 2 (hybrid NB)"]):
    sub = r1[r1.arm == arm].set_index("dataset")
    b = sub[sub.protocol == "B refit-per-fold"]["accuracy"].reindex(datasets)
    a = sub[sub.protocol == "A before-CV"]["accuracy"].reindex(datasets)
    paper = sub.drop_duplicates().groupby("dataset")["paper"].first().reindex(datasets)
    ax.bar([i - w for i in x], paper, width=w, label="Farid (2014)", color="0.75",
           edgecolor="black", hatch="//")
    ax.bar([i for i in x], b, width=w, label="protocol B (honest)", color="0.4",
           edgecolor="black")
    ax.bar([i + w for i in x], a, width=w, label="protocol A (before-CV)", color="0.85",
           edgecolor="black", hatch="..")
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("accuracy (\\%)" if False else "accuracy (%)")
    ax.set_ylim(0, 105)
axes[0].legend(loc="lower left", ncol=1, frameon=False)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([SHORT[d] for d in datasets], rotation=60, ha="right")
fig.tight_layout()
fig.savefig(FIGURES / "fig1_r1_protocols.pdf")
plt.close(fig)

# -------------------------------------------------------------------- fig 2 --
stages = pd.read_csv(RESULTS / "EXP-E9_pipeline" / "stages.csv")
metrics = pd.read_csv(RESULTS / "EXP-E9_pipeline" / "metrics_final_alpha0_mixed.csv")

removed = (stages[(stages.arm == "Alg1") & (stages.algorithm == "Alg1") & (stages.final == "DT")]
           .set_index("dataset")["rows_removed_pct"])
dt_alg1 = metrics[(metrics.arm == "Alg1") & (metrics.final == "DT")].set_index("dataset")["accuracy"]
dt_base = metrics[(metrics.arm == "baseline") & (metrics.final == "DT")].set_index("dataset")["accuracy"]
change = (dt_alg1 - dt_base).reindex(datasets)
removed = removed.reindex(datasets)

fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.axhline(0, color="0.6", linewidth=0.8)
ax.scatter(removed, change, color="black", s=18, zorder=3)
for d in datasets:
    ax.annotate(SHORT[d], (removed[d], change[d]), fontsize=6,
                xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("rows removed by Algorithm 1 (%)")
ax.set_ylabel("accuracy change of the tree (pts)")
fig.tight_layout()
fig.savefig(FIGURES / "fig2_alg1_removal_vs_tree.pdf")
plt.close(fig)

print(f"wrote {FIGURES/'fig1_r1_protocols.pdf'} and {FIGURES/'fig2_alg1_removal_vs_tree.pdf'}")
