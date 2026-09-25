# Figures — the paper's Figures 2 and 3 with our hybrid added

Source: `code/figures.py` (reads `../metrics_final_alpha0_mixed.csv` and `code/data.py`'s
`FARID_2014`). Regenerate with `python figures.py` from `code/`.

| File | What it is |
|---|---|
| `fig2_hybrid_dt.png` | Paper Figure 2 recreated (C4.5 vs Algorithm 1), plus our hybrid with the DT final classifier |
| `fig3_hybrid_nb.png` | Paper Figure 3 recreated (NB vs Algorithm 2), plus our hybrid with the NB final classifier |
| `fig4_all_classifiers.png` | All four paper classifiers plus both of our hybrid variants |

**Suggestion for the caption.** "Recreations of Figures 2 and 3 of Farid et al. (2014) on
the same ten datasets, with this work's mutual hybrid added. Grey bars: the accuracies the
original paper reports (protocol unstated). Orange bars: this work, Path 2 (Algorithm 2 then
Algorithm 1), everything refit inside each training fold, 10-fold CV × 5 seeds, entropy tree
with `ccp_alpha = 0`, mixed-likelihood Naive Bayes. The two sources are not protocol-matched;
the leakage audit quantifies that difference."

**Choice of bar.** The orange bar is Path 2 (attributes first, then instances) because it
wins the order comparison in `notes/e9-pipeline-results.md`; the other order and the
reference arms are in `../metrics_final_alpha0_mixed.csv`. To plot a different arm, change
the `arm`/`final` pair in `figures.py`'s `main()`.

**Verified programmatically** (bar heights equal to the source values, 10 datasets × the
expected series, no text outside the canvas, 300 dpi, non-blank). Not yet eyeballed — the
visual acceptance pass could not be run in this session.
