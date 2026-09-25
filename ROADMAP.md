# Roadmap — one phase at a time, solo-friendly

Estimates assume part-time solo work (≈10 h/week). Do not start a phase before the previous phase's exit criteria are met.

## Phase 0 — Environment + data (week 1)

- [ ] Python env: `python -m venv .venv`, install scikit-learn, pandas, numpy, matplotlib, pytest
- [ ] Download the 10 UCI datasets (see `data/README.md` for sources and gotchas)
- [ ] Data loader + per-dataset sanity stats (rows, attrs, classes == paper's Table 5)
- [ ] `git init` in `research/code/`; every experiment reproducible from a script

**Exit:** loading all 10 datasets reproduces Table 5's shape exactly.

## Phase 1 — Replication + leakage audit (weeks 2–6)

- [ ] E1: baselines — plain C4.5 (entropy tree) and Gaussian/categorical NB on all 10 datasets, 10-fold CV
- [ ] E2: reimplement Alg. 1 (NB filter → tree) and Alg. 2 (tree → 1/√d weights → NB); unit-test on the paper's Play-Tennis table
- [ ] E3: the audit — run each hybrid both ways: (a) filter/fit on FULL data before CV [paper's apparent protocol], (b) refit inside each training fold [correct]; report the gap
- [ ] Verify my numbers land in the right neighborhood of Tables 7–11 before claiming anything

**Exit:** a results table showing hybrid-vs-baseline deltas under both protocols; the leakage effect quantified. This alone is a defensible short-paper result.

## Phase 2 — The new method (weeks 6–10)

- [ ] E4: mutual hybrid (filter → tree → weights → weighted NB), refit per fold
- [ ] E5: confidence-aware variants: (a) consensus/majority committee filter, (b) per-class threshold, (c) soft reweighting instead of delete
- [ ] Log deletion diagnostics: % removed per dataset, class distribution of removed instances

**Exit:** E4 beats both one-directional hybrids; E5 beats hard deletion at least on noisy datasets (Tic-Tac-Toe, Breast Cancer).

## Phase 3 — Robustness + baselines (weeks 10–14)

- [ ] 10 repetitions × 10-fold CV with different seeds; mean ± std, 95% CIs
- [ ] Statistical tests: corrected paired t-test or Wilcoxon signed-rank across datasets
- [ ] Baselines: Hall 2007 (bagged-tree weighting), WANBIA-style learned weights, Wong-style 3-model routing (at least sketch-level)
- [ ] Optional strengthening: 1–2 modern datasets (e.g., OpenML-CC18 subset)

**Exit:** final results matrix, all numbers reproducible from one script.

## Phase 4 — Writing + submission (weeks 12–16+, overlaps Phase 3)

- [ ] Paper skeleton per `paper/outline.md`; results-first writing order (Methods → Results → Intro → Related → Discussion)
- [ ] Internal review pass (advisor / teacher — see note in README)
- [ ] Format to target journal; submit; start `trackers/writing-progress.md` revision log

**Teacher note:** the seed papers' author is likely your course teacher. An early conversation about this project (and possible co-authorship) costs nothing and de-risks everything — especially Hall (2007) precedent for Alg. 2.
