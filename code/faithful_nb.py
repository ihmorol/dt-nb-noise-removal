"""Weka-style Naive Bayes on the ORIGINAL attribute space.

The pipeline's WeightedNB (nb.py) works on one-hot columns. That is measurably the
wrong reading for replication: Weka's NB (and the paper's own Java NB, which follows
the same textbook) treats a nominal attribute as ONE categorical variable -- a single
frequency table per class -- not as independent 0/1 columns. This module is that
classifier, so the replication can run the paper's algorithms in the paper's own
attribute space.

Input convention:
    X                float ndarray. NOMINAL columns hold integer level codes
                     (the code of the value in the dataset's level list); NUMERIC
                     columns hold their value.
    nominal_mask     bool per column -- which columns are nominal
    levels           one list of level strings per nominal column (defines k_j)

Likelihoods (the two Weka estimators):
    nominal  DiscreteEstimator:  P(v|C) = (count(v,C) + 1) / (n_C + k_j)   [add-one]
    numeric  NormalEstimator:    Gaussian per class, variance smoothed like sklearn

weights   the paper's Eq. (14):  P(x|C) = P(C) * prod_j P(A_j|C)^W_j.
          None -> all exponents 1, i.e. plain NB.

With no nominal columns and weights=None this reproduces sklearn.GaussianNB exactly
(same epsilon construction as the pipeline's WeightedNB) -- that identity is checked
in verify_faithful.py.
"""

import numpy as np


class FaithfulNB:
    """Naive Bayes over mixed nominal/numeric attributes, with per-attribute exponents."""

    def __init__(self, nominal_mask, levels, weights=None, var_smoothing=1e-9):
        self.nominal_mask = np.asarray(nominal_mask, dtype=bool)
        self.levels = levels                       # k_j per nominal column
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        numeric = ~self.nominal_mask

        # epsilon exactly as sklearn's GaussianNB / the pipeline's WeightedNB compute it
        if numeric.any():
            self.epsilon_ = self.var_smoothing * np.var(X[:, numeric], axis=0).max()
        else:
            self.epsilon_ = 0.0

        self.priors_ = []
        self.means_ = []
        self.variances_ = []
        self.log_prob_tables_ = []                 # per class: list of (k_j,) log-prob arrays
        for c in self.classes_:
            Xc = X[y == c]
            n_c = len(Xc)
            self.priors_.append(n_c / len(X))

            means = np.zeros(X.shape[1])
            variances = np.zeros(X.shape[1])
            tables = []
            for j in range(X.shape[1]):
                if self.nominal_mask[j]:
                    k_j = len(self.levels[j])
                    counts = np.bincount(Xc[:, j].astype(int), minlength=k_j)
                    p = (counts + 1.0) / (n_c + k_j)          # Weka's add-one estimator
                    tables.append(np.log(p))
                else:
                    means[j] = Xc[:, j].mean()
                    variances[j] = Xc[:, j].var() + self.epsilon_
                    tables.append(None)
            self.means_.append(means)
            self.variances_.append(variances)
            self.log_prob_tables_.append(tables)

        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        return self

    def joint_log_likelihood(self, X):
        """log P(C) + sum_j W_j * log P(A_j = x_j | C), one column per class."""
        X = np.asarray(X, dtype=float)
        scores = np.zeros((len(X), len(self.classes_)))
        for k, c in enumerate(self.classes_):
            total = np.full(len(X), np.log(self.priors_[k]))
            tables = self.log_prob_tables_[k]
            for j in range(X.shape[1]):
                if self.nominal_mask[j]:
                    total += self.weights[j] * tables[j][X[:, j].astype(int)]
                else:
                    mean, var = self.means_[k][j], self.variances_[k][j]
                    log_gaussian = -0.5 * (np.log(2 * np.pi * var)
                                           + (X[:, j] - mean) ** 2 / var)
                    total += self.weights[j] * log_gaussian
            scores[:, k] = total
        return scores

    def predict(self, X):
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]

    def predict_proba(self, X):
        scores = self.joint_log_likelihood(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())
