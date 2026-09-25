"""Farid (2014) Algorithm 2 -- hybrid Naive Bayes.

The published steps (pp. 1941-1942):
    1-13   build a decision tree T on the training data
    14-20  for each attribute: W = 0 if the tree never tests it, otherwise
           W = 1/sqrt(d), where d is the smallest depth at which the tree tests it
    21-23  find the class priors P(C)
    24-28  find P(A_ij|C)^W only for the attributes with W != 0
    29-31  take the class with the highest posterior

The code returns the reduced table (attributes the tree never tests are dropped, which
for the NB product is the same as W = 0 because p^0 = 1) together with the weights.
Handing those weights on is how Algorithm 2 supports Algorithm 1's noise judge, and how
the final NB classifier gets the paper's weighted form (Eq. 14).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

CCP_ALPHA = 0.01     # pruning: 0 = unpruned tree, larger = fewer attributes survive


def attribute_weights(tree, n_attributes):
    """Steps 14-20: W_j = 1/sqrt(min depth of attribute j), 0 if never tested.

    The root counts as depth 1, so the attribute asked first gets weight 1. The paper
    does not say where depth starts; starting at 0 would make 1/sqrt(d) undefined.
    """
    depth = np.zeros(n_attributes, dtype=int)      # 0 means "not tested in the tree"

    # walk the tree breadth first, remembering how deep each node sits
    queue = [(0, 1)]                               # (node id, depth of that node)
    while queue:
        node, node_depth = queue.pop(0)
        if tree.tree_.children_left[node] == -1:   # a leaf asks no question
            continue
        attribute = tree.tree_.feature[node]
        if depth[attribute] == 0 or node_depth < depth[attribute]:
            depth[attribute] = node_depth
        queue.append((tree.tree_.children_left[node], node_depth + 1))
        queue.append((tree.tree_.children_right[node], node_depth + 1))

    weights = np.zeros(n_attributes)
    tested = depth > 0
    weights[tested] = 1.0 / np.sqrt(depth[tested])
    return weights


def farid_algorithm2(X, y, ccp_alpha=CCP_ALPHA, seed=0):
    """Steps 1-13 (the tree), then the attribute selection and weighting of steps 14-20.

    Returns the data with only the tested attributes, and an info dictionary:
        keep        column indices that survived (slice the test fold with the same list)
        weights     the 1/sqrt(d) weights of those columns
        n_kept, n_removed, tree
    """
    tree = DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha, random_state=seed)
    tree.fit(X, y)

    weights = attribute_weights(tree, X.shape[1])
    keep = np.flatnonzero(weights > 0)
    if keep.size == 0:                 # safety: an empty table would break everything
        keep = np.arange(X.shape[1])
        weights = np.ones(X.shape[1])

    info = {
        "keep": keep,
        "weights": weights[keep],
        "n_kept": int(keep.size),
        "n_removed": int(X.shape[1] - keep.size),
        "tree": tree,
    }
    return X[:, keep], info
