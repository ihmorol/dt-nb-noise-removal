# Experiment registry (one row per experiment; append, never delete)

Format: ID | date | question | config (data/protocol/seed) | result summary | verdict | artifacts

Rules:
- Every experiment must answer ONE question.
- Nothing counts until it runs end-to-end from a script committed in `code/`.
- `artifacts` points to `results/EXP-<id>_<shortname>/`.

| ID | Date | Question | Config | Result | Verdict | Artifacts |
|---|---|---|---|---|---|---|
| E0 | planned | Do my data loaders reproduce the seed's Table 5 shapes? | 10 UCI datasets, loader sanity check | — | — | — |
| E1 | planned | What do plain C4.5-equivalent + NB score on all 10 (my protocol)? | sklearn entropy tree + NB, 10-fold CV ×10 seeds | — | — | — |
| E2a | planned | Does re-implemented Alg. 1 beat C4.5, and by how much? | NB-filter→tree, refit per fold | — | — | — |
| E2b | planned | Does re-implemented Alg. 2 beat NB, and by how much? | tree→1/√d weights→NB, refit per fold | — | — | — |
| E3a | 2026-09-18 | How much do E2 gains inflate when filter/selection is fit BEFORE CV (paper's apparent protocol) vs refit per fold? | both protocols + an all-data upper bound, same data/stages/classifiers, 10 datasets, 10-fold × 3 seeds | Filtering before CV adds **+15.1** points to Alg 1 (+16 to +17 to the chained paths) and only **+0.5** to Alg 2; under the leaky protocol this code matches or beats the paper's Alg 1 on 9/10 datasets (iris 98.63 vs 98.66), so the honest-vs-paper gap is mostly protocol, not implementation | done (pilot seeds) — write-up in `notes/e3a-leakage-result.md` | `results/EXP-E3a_leakage_check/` |
| E3b | planned | Does ORDER matter: instance-filter→weight vs weight→instance-filter vs one-pass vs none? | all orders refit per fold, same seeds (evidence: L48/L49 — see notes/strategy-mutual-hybrid.md) | — | — | — |
| E4 | planned | Does the MUTUAL hybrid (filter→tree→weights→weighted NB) beat E2a and E2b? | refit per fold | — | — | — |
| E5a | planned | Does a consensus/majority committee filter beat single-NB deletion? | C3 variant | — | — | — |
| E5b | planned | Does per-class thresholded deletion beat hard deletion? | C3 variant | — | — | — |
| E5c | planned | Does soft reweighting of suspect instances beat deletion? | C3 variant | — | — | — |
| E5d | planned | Does relabeling weak-noise instances (deleting only the residue) beat hard deletion? | relabel arm; evidence L31/L33 (correction > filtering) | — | — | — |
| E6 | planned | What exactly gets deleted? (% removed, class distribution) | deletion diagnostics on E2a/E4 | — | — | — |
| E7 | planned | Do gains survive statistical testing? | Wilcoxon / corrected paired t over 10 datasets ×10 seeds | — | — | — |
| E8 | planned | Do E4/E5 still win vs Hall-2007-style weighting and a learned-weight baseline? | baseline block | — | — | — |
| E9 | goal P1 | Does instances→attributes (C1) or attributes→instances (C2) clean better, and with which final classifier? | 2 orders × {NB, DT} core + reference arms (baselines, single-stage); boxes = Farid Alg 1/Alg 2; refit per fold, 10×10 CV; winner = pre-registered mean-rank rule | — | — | — |
| E10 | goal P2 | Does the best published attribute-weighting (W1 Hall / W2 CFW / W3 FTAWNB / W4 WANBIA / W5 MAWNB) improve the E9 winner's accuracy? | winner composition + weighted-NB final; inner-CV selection inside training folds; full outer ablation | — | — | — |
| E9p | 2026-09-18 | Pipeline pilot: with the refactored 4-file code, which order/config of Farid's two algorithms works best on the 10 seed datasets? | 10 datasets (all downloaded), 10-fold CV × 3–5 seeds, refit per fold; knobs = tree α {0, 0.005, 0.01, 0.02} × NB likelihood {Gaussian, mixed} × Alg-2 weights on/off | Attribute stage ≈ neutral (Alg2→DT 85.81 vs plain 85.76); instance filter costs the tree ≈4.8 pts (Alg1→DT 80.98 vs 85.76) and its damage tracks the judge's error rate (glass: NB 46 %, 47 % deleted, tree 68→51); attributes-first C2 > C1; best cell C2→DT 81.04; ≠ Farid's reported Alg1 88.4 / Alg2 83.4 (table) — the paper's text claims 86.7 for Alg 2 while its own table averages 83.42 | pilot, not E9: 5 seeds, no significance tests | `results/EXP-E9_pipeline/` (metrics_final_alpha0_mixed.csv, sweep_*.csv), write-up `notes/e9-pipeline-results.md` |
| R1 | 2026-09-19 | Do Farid's Tables 8–11 reproduce with the paper's OWN setup — Weka J48, original nominal attribute space, textbook NB — under the honest protocol vs the paper-apparent (leaky) one? | 10 datasets, stratified 10-fold × 3 seeds; C4.5 = J48 pruned (Weka defaults), NB = FaithfulNB, Alg1 = FaithfulNB judge + J48 on the remainder, Alg2 = J48 min-depth weights + FaithfulNB (Eq. 14); protocol A = stages fitted once on the full dataset, B = refit per fold; all choices fixed a priori (see DECISIONS.md 2026-09-19) | **Baselines replicate**: C4.5 within ±1.4 on 7/10 (diabetes exact, tic-tac-toe +0.01, glass +0.06, breast-cancer −0.21; NSL-KDD unresolvable: ours 99.47 vs paper 71.11); NB within ±2 on 8/10. **The hybrids do not**: Alg1 honest 81.02 vs paper 88.43, leaky 94.44 (the paper's number sits between the two protocols); Alg2 77.16 honest / 78.14 leaky vs 83.42 — no protocol reaches it. J48 confirms E3a: the filter's gain is protocol, not method | done (pilot 3 seeds; unpruned sensitivity run + 10-seed final to follow) | `results/EXP-R1_replication/`, code `code/replicate_farid.py` + `code/faithful_nb.py` + `code/weka_utils.py`, verification `code/verify_faithful.py` |

## Next up

E0 → E1 → E2 (unit tests green), then the two-part goal: **E9 (combination sweep) → E10 (weighting upgrade)** — full design in `../notes/goal-parts-1-2.md`. E3a/E3b reuse E9's machinery.
The refactored pipeline (`code/main.py`) now covers the E9 core cells plus reference arms; E9p's write-up (`../notes/e9-pipeline-results.md`) says what to fix before the full 10-seed E9 run: the mixed NB likelihood is mandatory for nominal data, and E3a should measure how much of Farid's Alg-1/Alg-2 gain is protocol.
