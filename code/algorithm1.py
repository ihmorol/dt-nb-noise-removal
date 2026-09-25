"""Farid (2014) Algorithm 1: Naive Bayes removes noisy training instances.

Paper steps 1-12: learn NB on the training data, classify every training
instance, and delete the ones NB gets wrong. Steps 13-25 grow a tree on what is
left; in this code that tree is the final classifier of the pipeline.

With weights=None this is the paper's Algorithm 1. With the weights from
Algorithm 2, the judge is the weighted NB instead of plain NB.
"""
import numpy as np

from nb import WeightedNB


def algorithm1(X, y, weights=None, likelihood="gaussian"):
    """Return (keep, skipped): keep[i] is True for the instances NB classifies right.

    Our own safety rule, not in the paper: if the deletion would leave fewer than
    two classes, nothing is deleted and skipped is True.
    """
    judge = WeightedNB(weights, likelihood).fit(X, y)
    keep = judge.predict(X) == y

    if len(np.unique(y[keep])) < 2:
        return np.ones(len(y), dtype=bool), True
    return keep, False
