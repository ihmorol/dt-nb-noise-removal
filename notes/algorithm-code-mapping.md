# Algorithm ↔ code, line by line

*2026-09-18. The listings below are transcribed from the publisher PDF's text layer
(`reference/Farid_2014_hybrid_DT_NB_multiclass.pdf`, pp. 1941–1942). Verdict column:
**match** = implemented as written; **split** = implemented, but in another file because
the pipeline composes steps; **deviation** = deliberate difference, explained below.*

## Algorithm 1 — Decision tree induction

| Step (paper) | Code | Verdict |
|---|---|---|
| 1–3: for each class find the prior probabilities P(C_i) | `WeightedNB.fit` (nb.py:28), called by `farid_algorithm1` (algorithm1.py:34) | match |
| 4–6: for each attribute value find the class-conditional probabilities P(A_ij\|C_i) | same `fit` (Gaussian for numeric, Bernoulli for 0/1 columns) | match |
| 7–8: for each training instance find the posterior P(C_i\|x_i) | `WeightedNB.predict` → `joint_log_likelihood` (nb.py:74) | match |
| 9–11: if x_i is misclassified, remove x_i from D | `wrong = judge.predict(X) != y` (algorithm1.py:35) → `X[~wrong], y[~wrong]` (algorithm1.py:53) | match |
| 12: end for (the filter) | the return of `farid_algorithm1` | match |
| 13: T = ∅ | — | split: this is the final classifier slot, `classify` (pipeline.py:27) |
| 14–16: best splitting attribute; root node; arcs | `DecisionTreeClassifier(criterion="entropy")` (pipeline.py:32), fitted on the cleaned data | split |
| 17–24: recurse `DTBuild(D)` for each arc, leaf at the stopping point | the same estimator (sklearn CART), **not** Weka J48 | split + deviation (CART ≠ C4.5, documented limitation) |
| 25: end for | — | — |

**How to read the "split".** Farid's Algorithm 1 is one object: *filter → tree*. In the code
the filter is `farid_algorithm1` and the tree is whatever `final` classifier the arm asks for.
For the arm `Alg1→DT` that reproduces Algorithm 1 exactly (filter, then an entropy tree on
what survived). For the reference arm `Alg1→NB` no tree is grown — the paper never did that;
that arm exists only to attribute a gain to the filter alone.

## Algorithm 2 — Naïve Bayes classifier

| Step (paper) | Code | Verdict |
|---|---|---|
| 1–13: build the decision tree T on D (root, arcs, recurse `DTBuild`) | `tree.fit(X, y)` in `farid_algorithm2` (algorithm2.py:58) | match (CART, not J48) |
| 14: for each attribute A_i ∈ D | `attribute_weights(tree, n_attributes)` (algorithm2.py:23, called at :60) | match |
| 15–16: if A_i is not tested in T → W_i = 0 | `depth[attribute] == 0` → the weight stays 0 (algorithm2.py:41–45) | match |
| 17–18: else d = minimum depth of A_i in T, W_i = 1/√d | breadth-first walk recording each node's depth; smallest depth kept per attribute; `weights[tested] = 1.0 / np.sqrt(depth[tested])` (algorithm2.py:45) | match, **with one choice the paper leaves open**: root = depth 1, so the first-split attribute gets weight 1. Root = 0 would make 1/√d undefined at the root |
| 19–20: end for | — | — |
| 21–23: for each class find the prior probabilities P(C_i) | `WeightedNB.fit` (nb.py:28) | match |
| 24–26: for each attribute with W_i ≠ 0 and each of its values, find P(A_ij\|C_i)^W_i | only the kept columns reach the NB (`return X[:, keep]`, algorithm2.py:73); the exponent is applied in `joint_log_likelihood`: `(log_p * self.weights).sum(axis=1)` (nb.py:79) — that is Π P(A_j\|C)^W_j, Eq. (14) | match |
| 27–28: end for | — | — |
| 29–31: for each instance find the posterior P(C_i\|x_i) | `predict` = argmax of the weighted joint log-likelihood (nb.py:82) | match |

## The deviations, all deliberate

1. **Attributes are removed from the table, not only zero-weighted** (algorithm2.py:73).
   For the NB product this is *identical*, because a zero weight makes the term p⁰ = 1. It
   differs for the **DT** final classifier, which Farid never ran after Alg 2 — the reduced
   feature space is a real change there, and removal is what the pipeline was asked for.
2. **Alg 1's judge can be the weighted NB** (`weights` argument, algorithm1.py:25 and :34).
   As published, Algorithm 1's judge is plain NB. This is the mutual-support extension: on
   Path 2, the tree that Alg 2 grew decides which attributes the filter looks at.
   `weights=None` (Path 1) is the faithful form.
3. **Safety rule in Alg 1** (algorithm1.py:46–51): if deleting every misclassified instance
   would leave a single class, nothing is deleted and `info["skipped"]` records it. The paper
   has no such guard; without it the fold has no trainable data at all.
4. **Likelihood for 0/1 columns** (`likelihood="mixed"`, nb.py:32–37 and :63–67): Bernoulli for
   0/1 columns, Gaussian elsewhere. The paper says probabilities are computed "even if it is
   numeric" but never specifies the nominal case; its Weka NB uses frequency counts for nominal
   attributes. Gaussian-on-one-hot was measurably the wrong reading (Play-Tennis below).
5. **Tree engine**: sklearn CART with entropy instead of Weka J48, and `ccp_alpha` in place of
   J48's confidence factor. Recorded in `DECISIONS.md` from the start.

## Checks that were run against the code

| Check | Result |
|---|---|
| `WeightedNB(weights=None)` vs `sklearn.GaussianNB` | identical predictions, max log-probability difference 1.1e-13 |
| Alg 2 weights against a hand-read iris tree | root attribute → 1.0, depth 3 → 0.577, depth 4 → 0.5, untested → 0 | 
| Play-Tennis (paper Table 3), Alg 1 | Gaussian-on-one-hot judge deletes 5/14 rows, all from one class (56 % of Yes); the mixed judge deletes 1/14 — the faithful reading |
| Play-Tennis, Alg 2 | keeps 6 of the 8 one-hot columns, weights 1.0 / 0.707 / 0.577 / 0.5 |
| Internal consistency | the single-stage `Alg1` arm and Path 1 delete the same rows on every dataset (same judge, same data); Path 2 differs, which is the interaction being measured |

## What is in the code but not in the paper (and why)

- 10-fold CV with everything refit inside each training fold (`run_fold`, pipeline.py:41): the
  paper says "10-fold cross validation" but never says whether the filter and the tree were
  refit per fold. This is the leakage-safe reading; the difference is what E3a will measure.
- Reference arms and the Farid comparison table (`ARMS`, main.py:39; `print_farid_comparison`,
  tables.py): attribution scaffolding, not part of either algorithm.
- The three data versions and the removal log (`trace_dataset`, versions.py:56): reporting only.
