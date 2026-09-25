"""Farid (2014) Algorithm 2: a decision tree selects and weights the attributes.

Paper steps 1-20: build a tree on the training data. An attribute the tree never
tests gets weight 0 (it is dropped). A tested attribute gets W = 1/sqrt(d), where
d is the smallest depth at which the tree tests it. The root is depth 1.
Steps 21-31 are the weighted NB (nb.py) that uses these weights.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def tree_weights(tree, n_columns):
    """W_j = 1/sqrt(smallest depth where the tree tests column j), 0 if never tested."""
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    tested_column = tree.tree_.feature

    smallest_depth = {}
    nodes = [(0, 1)]                        # (node number, depth); node 0 is the root
    while nodes:
        node, depth = nodes.pop()
        if left[node] == -1:                # a leaf tests nothing
            continue
        column = tested_column[node]
        smallest_depth[column] = min(depth, smallest_depth.get(column, depth))
        nodes.append((left[node], depth + 1))
        nodes.append((right[node], depth + 1))

    weights = np.zeros(n_columns)
    for column, depth in smallest_depth.items():
        weights[column] = 1 / np.sqrt(depth)
    return weights


def algorithm2(X, y, alpha):
    """Return (keep, weights): the columns the tree tests and their weights.

    alpha is the tree's pruning strength (sklearn ccp_alpha); 0 means no pruning.
    If the tree is a single leaf, every column is kept with weight 1, so the
    data never ends up with no columns (our own safety rule).
    """
    tree = DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha, random_state=0)
    weights = tree_weights(tree.fit(X, y), X.shape[1])

    keep = np.flatnonzero(weights > 0)
    if len(keep) == 0:
        return np.arange(X.shape[1]), np.ones(X.shape[1])
    return keep, weights[keep]
