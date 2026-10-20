# ICEEICT 2027 submission audit (19 October 2026)

Scope: `paper/iceeict2027-submission/` and its source ZIP (identical before editing), checked against every saved result, branch and decision in this repository.

## Verification

- Every number in the paper's text, five tables and two figures was recomputed independently from `results/EXP-E9_farid` (means, seed SDs, exact Wilcoxon p, Holm, Protocol A-B gaps, breast-cancer class removal, glass/vote attribute retention, safeguard and empty-selection counts). All match.
- A fresh refit with the committed code reproduced all twelve arms exactly on seed 0/fold 0 and seed 9/fold 9 for all ten datasets (240 scores).
- Every local and remote branch is already merged into main; no branch holds an unreported result that bears on the E9 scope.
- Rebuilt PDF: 5 A4 pages, 270,361 bytes, all fonts embedded, no Type 3, empty author metadata, no author names or affiliations in decompressed streams, no overfull boxes or unresolved references.

## How the paper connects to the other experiments

| Experiment | Relation to the paper | Consistent? |
|---|---|---|
| R1 replication | Source of the 2014 table averages and the NSL-KDD 99.47 vs 71.11 discrepancy | Yes; now reported in the Discussion |
| E3a, E9p (sklearn pilots) | Same pattern in a different representation: filter costs the tree about 5 pp; before-CV inflates filter arms by about 15 pp | Yes; correctly not pooled |
| E3b (pre-registered parallel arm) | The paper's P design; its prediction (P close to NA) held | Yes; reported descriptively |
| E5 (same folds and engines) | Majority committee deletion gives C4.5 85.77 (vs plain 86.33, p=0.65; vs single-NB deletion p=0.0078). Supports the paper's reading that the harm comes from single-judge hard deletion | Yes; excluded from the paper by design |
| E11, E12v2 | Different learners, one-hot data, injected noise | Correctly excluded |

`trackers/experiments.md` (E9f row) still calls AN--DT vs C4.5 "significantly worse" at uncorrected p=0.014. The paper correctly reports Holm p=0.109. Do not quote the registry wording to reviewers.

## Changes made (text only; no tables, figures or results changed)

See `paper/iceeict2027-submission/review/changes.md`. Disclosed the image-segmentation row-count deviation, stated the Wilcoxon zero-difference handling, added the 2014 table averages next to our Protocol B means, softened one causal phrase and removed three redundant sentences to stay at five pages.

## Six-page expansion (same day)

The paper now uses all six pages. Page 6 is filled with three per-dataset line charts in the style and dataset order of the 2014 paper's Figs. 3 and 4, each with a short explanation: single steps against their base classifiers (Fig. 1), the three combined designs (Fig. 3), and both protocols beside the 2014 values (Fig. 5). All figures are generated from saved results by `scripts/generate_evidence.py`, which asserts every per-dataset number quoted in the text. A first expansion with extra tables, a scatter plot and a dot plot (commit 5dd7394) was withdrawn at the authors' request. Details in `paper/iceeict2027-submission/review/changes.md`.

## Are more runs required?

No. Every claim is supported by saved, reproducible results. Optional, not needed for this submission: pooled fixed-class macro-F1 (needs an hours-long E9 rerun; macro-F1 is already descriptive), a matched plain-vs-weighted judge ablation, and a fixed-population A/B contrast. The paper already lists the last two as future work.

## Remaining author-side blockers

1. LLM policy. The acknowledgment truthfully says LLMs assisted with drafting. The team's 18 October check recorded that ICEEICT permits only non-intellectual LLM use. Authors must rewrite the prose themselves or otherwise satisfy that rule, then make the acknowledgment match what actually happened.
2. Author approval of the interpretation, final wording and author order.
3. Similarity screening (under 30 percent) and CMT metadata.
4. Re-check the deadline on the official site; the last recorded check (9 October) listed 25 October 2026.
