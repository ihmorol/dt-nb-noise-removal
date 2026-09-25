# Idea log (append freely; prune quarterly; promote the best to README contributions)

## Active

1. **Mutual hybrid (E4).** NB filter → tree → depth weights → weighted NB, one loop. Optional: iterate twice (NB re-filters using weighted probabilities). Check: does iteration help or over-clean?
2. **Leakage audit (E3a).** Quantify inflated gains from pre-CV filtering. Sellable even if the new method wins narrowly — an honest audit + corrected protocol is a contribution by itself.
3. **Confidence-aware filtering (E5).** Replace "NB disagreed → delete" with committee consensus, per-class thresholds, or soft weights. Hypothesis: hard deletion hurts rare classes most; check via E6 deletion diagnostics.
4. **Deletion diagnostics (E6).** Nobody in this line reports what the filter removes. A figure: % removed per class vs. class frequency, per dataset.

## Parking lot (don't start before Phase 2 done)

- Same idea under concept drift (streams): re-filter per chunk; compare with the 2013 Farid ensemble. Big scope — park.
- Calibration: does attribute weighting improve NB probability calibration (log-loss/Brier)? Cheap add-on analysis, maybe Phase 3.
- Dynamic feature sets: the seed's future work — too broad alone.

## Kill list (decided against, with reason)

- Matching Farid's exact Table numbers — protocol under-specified (seeds, numeric handling, root depth) and CART≠C4.5; comparison-under-one-protocol is the goal instead.
