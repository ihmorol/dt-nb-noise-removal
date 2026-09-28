# Domain terms

Words the code, notes and paper use with one fixed meaning.

- **Algorithm 1 (Alg 1, step "N")**: Farid (2014)'s noise filter. The **judge** (a Naive Bayes model) classifies the training rows and every row it gets wrong is deleted.
- **Algorithm 2 (Alg 2, step "A")**: Farid (2014)'s attribute step. A decision tree is grown; attributes it never tests are dropped, the rest get weight 1/sqrt(smallest depth).
- **Judge**: the NB inside Algorithm 1. Plain NB, or weighted NB when Algorithm 2 ran first and **support** is on.
- **Support**: handing Algorithm 2's weights to the judge and to the final NB.
- **Path**: an order of the two algorithms. Path 1 = Alg 1 then Alg 2; Path 2 = Alg 2 then Alg 1.
- **Arm**: a path (or no cleaning, or one algorithm alone) plus a final classifier (NB or DT). Reference arms exist to attribute gains.
- **Fold cleaner**: `pipeline.clean()`; runs a path on a training fold only and returns a `Cleaned` result. `Cleaned.apply_to_test()` is the only way a test fold is prepared: it loses the same columns, never rows.
- **Removal record**: one per step of a path, same keys for both algorithms (rows/attributes before and after, removed rows, removed columns, skipped, judge, per-class removal rates).
- **Protocol B (refit-per-fold)**: every step and the classifier are fitted inside each training fold. The honest estimate.
- **Protocol A (fit-before-CV)**: the algorithms run once on the whole dataset, then only the classifier is cross-validated. Leaks test labels into the cleaning.
- **Protocol C**: algorithms and classifier fitted and scored on the whole dataset. Upper bound on leakage.
- **R1 world vs pipeline world**: R1 (`replicate_farid.py`) uses Weka J48 and FaithfulNB on the original columns; the pipeline uses sklearn CART and WeightedNB on one-hot columns.
