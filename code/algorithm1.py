"""Farid (2014) Algorithm 1 -- hybrid decision tree.

The published steps (p. 1941):
    1-6    find the class priors P(C) and the class-conditional probabilities P(A_ij|C)
    7-12   for each training instance find the posterior P(C|x) and delete the
           instance if Naive Bayes gets it wrong
    13-25  grow a C4.5-style tree on what is left

Steps 1-12 are this function. Steps 13-25 are the final tree classifier of the arm,
which the pipeline fits on the data this function returns.

Two optional arguments go slightly beyond the paper, so that the two algorithms can
support each other:
    weights     the weights that Algorithm 2 produced. With weights, the judge is the
                weighted NB on the tree-selected attributes instead of plain NB.
    likelihood  how the judge treats nominal attributes, see nb.py
weights=None is the faithful form of the paper's Algorithm 1.
"""

import numpy as np

from nb import WeightedNB


def farid_algorithm1(X, y, weights=None, likelihood="gaussian"):
    """Delete every training instance that the Naive Bayes judge misclassifies.

    X, y  the current training data (already reduced by Algorithm 2 if it ran first)

    Returns the cleaned data and an info dictionary with the removal count, the
    removal rate, per-class removal rates, which rows survived, and whether the
    safety rule below skipped the removal.
    """
    judge = WeightedNB(weights=weights, likelihood=likelihood).fit(X, y)
    wrong = judge.predict(X) != y

    info = {
        "removed": int(wrong.sum()),
        "removed_rate": float(wrong.mean()),
        "per_class": {c: float(wrong[y == c].mean()) for c in np.unique(y)},
        "judge": "weighted NB (Alg 2 weights)" if weights is not None else "plain NB",
        "keep": ~wrong,
        "skipped": False,
    }

    # Safety rule that the paper does not have: if deleting every misclassified
    # instance would remove a whole class, delete nothing this time. Log it.
    if np.unique(y[~wrong]).size < 2:
        info["skipped"] = True
        info["keep"] = np.ones(len(y), dtype=bool)
        return X, y, info

    return X[~wrong], y[~wrong], info
