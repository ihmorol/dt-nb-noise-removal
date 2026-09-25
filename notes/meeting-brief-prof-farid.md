# Meeting brief: approaching Prof. Farid

*2026-09-18. Prepared package for the conversation about this project. Read it, adjust it to your voice, and bring the printed one-pager below plus the pipeline diagram (`diagrams/e9-pipeline.pdf`).*

## First, the mechanics

I cannot email, message, or talk to Prof. Farid — I have no channel to people outside this session. What I can do is prepare everything so your approach lands. This document has: a draft email, the one-pager to hand him, his likely questions with prepared answers, and the framing rules that make the difference between "interesting, keep me posted" and "let's work on this together".

## Draft email (short, respectful — attach the one-pager)

> **Subject:** Extending your 2014 DT–NB hybrids: a verified gap and a 20-minute request
>
> Dear Professor Farid,
>
> I am [your name], [course/semester, roll number]. I have been studying your 2014 ESWA paper "Hybrid decision tree and naïve Bayes classifiers for multi-class classification tasks" and your 2010 IJNSA predecessor on combining Naive Bayes and decision trees for intrusion detection.
>
> Working through the literature, I found that your two 2014 algorithms — NB-based instance removal (Alg. 1) and tree-based attribute selection (Alg. 2) — have never been run together in one pipeline in either order, under one protocol; I checked all 410 papers citing yours and did targeted searches besides. I have built a leakage-safe experiment plan for exactly that comparison on your ten UCI datasets, with preliminary evidence on where each step helps and where it is neutral.
>
> May I have 20 minutes to show you the plan and ask two protocol details I could not reconstruct from the paper (numeric-attribute handling, and whether the filtering ran inside or outside the CV folds — I plan to test both ways and report the difference)? I would value your view on whether this is worth developing further.
>
> Respectfully,
> [name · contact]

## The one-pager to hand him

**Completing the loop: both orders of your two 2014 hybrids, under a leakage-safe protocol**

- **The gap (verified):** no published work combines Alg. 1 (NB instance removal) and Alg. 2 (tree attribute selection) into one pipeline. Your 2014 paper proposed them separately; your 2010 paper's joint design differed (dedupe + ML relabeling + information-gain splitting); your later work moved to NBTree-style hybrids and ensembles.
- **What I propose:** two orders, one protocol — **C1: instances → attributes** (NB filter, then tree selection) vs **C2: attributes → instances** (tree selection, then NB filter), each with an NB or DT final classifier; plus no-cleaning and single-stage reference arms so any gain is attributable.
- **What is already done:** verified literature review (all 410 citing papers scanned); leakage-safe protocol design (refit per fold, 10-fold × 10 seeds, paired folds, significance tests); pilots on bundled data showing the attribute stage removes 25–70 % of columns and the instance stage is near-neutral on clean data (hypothesis: it pays only on noisy datasets); implementation plan in Python/sklearn, hours per run.
- **Part 2 (kept, not dropped):** apply the published weighting strategies — including your single-tree `1/√d` from Alg. 2 — to the winning order and test whether weighting beats hard removal.
- **My ask:** supervision/feedback; the two protocol details; and if you find it interesting, your involvement on whatever terms you prefer.

## If he asks (prepared answers)

1. **"This is just my two algorithms chained."** → The chain is unclaimed — verified across all 410 citing papers; the contribution is the *first controlled comparison* of the two orders, the leakage audit of this family, and the removal diagnostics nobody has published. C1 ending in a tree is your Alg. 1 extended; C2 completes Alg. 2's line with a filter.
2. **"My paper already shows the gains."** → We replicate your algorithms faithfully first (unit tests against your Play-Tennis tables). The audit then checks whether the gains hold when the filter is refit inside each fold. If they do, your results are confirmed under a stricter protocol — a win for both of us.
3. **"Deleting instances is known to be risky."** → Agreed — that is documented in the literature (single-model filters are the weakest design, and correction beats filtering). That is exactly why the faithful hard-deletion arm stays and confidence-aware variants are the planned extension, not the default.
4. **"What is publishable here?"** → First controlled order comparison of your two algorithms; the leakage audit; per-class deletion diagnostics (what the filters actually remove — nobody reports this); and Part 2's weighting upgrade. There is enough for a paper even if the accuracy differences turn out small.
5. **"Why these datasets and this protocol?"** → Your Table 5's ten datasets for comparability; nested/refit-per-fold cross-validation for correctness; everything else fixed in a config before running.
6. **"What do you need from me?"** → Twenty minutes now; the protocol details; feedback on the design; supervision if it develops. Co-authorship is his call, not a request you lead with.

## Framing rules

- **Do** frame everything as extending his work; bring artifacts (diagram, one-pager, pilot numbers); ask small, concrete questions — protocol ambiguities are genuine and easy for him to answer.
- **Do not** say "your paper leaked". Say: "the protocol is underspecified — I test both ways and report the difference."
- **Do not** overclaim: the pilots are mechanics on bundled data, not results; say so plainly.
- **Do not** ask for co-authorship up front. State that his involvement is his call; let the work earn it.
- **Keep it 20 minutes:** 2 min lineage, 5 min gap + design, 5 min what is done, 5 min his questions, 3 min the ask.

## After the meeting

- Write his answers into `DECISIONS.md` (protocol details, feedback, suggested datasets/baselines) the same day.
- Send a short follow-up email within 24 h summarizing what was agreed.
- If he declines involvement: the project stands on its own — the gap is verified and the plan is evidence-based. Proceed with E0 and keep citing him properly.

## Why this is hard to refuse (the honest version)

Not because of pressure — because of evidence and fit: it is **his own paper's open loop**, the gap is **verified**, the protocol is **stricter than the original**, the ask is **small** (20 minutes and two questions), and the artifacts are **already built** (literature audit, design, diagram, pilot evidence, runnable plan). That combination is what makes a supervisor say yes.
