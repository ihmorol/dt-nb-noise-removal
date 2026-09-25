"""The Naive Bayes model that both of Farid's algorithms use.

The same class is used in three places, so the project has one NB everywhere:
    * Algorithm 1's judge   -- which training instances does NB get wrong?
    * Algorithm 2's model   -- Eq. (14), P(x|C) = P(C) * product of P(A_j|C)^W_j
    * the final NB classifier of a pipeline arm

weights=None  -> all exponents are 1, which is plain Naive Bayes
likelihood    -> "gaussian": every attribute is a bell curve per class
                 "mixed"   : 0/1 attributes (one-hot columns) use the Bernoulli
                             formula, the rest stay Gaussian. This is how Weka's
                             NB treats nominal vs numeric attributes, and Weka's
                             NB is what the paper ran.
"""

import numpy as np


class WeightedNB:
    """Naive Bayes where every attribute has an exponent (weight)."""

    def __init__(self, weights=None, likelihood="gaussian", var_smoothing=1e-9, alpha=1.0):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.likelihood = likelihood
        self.var_smoothing = var_smoothing
        self.alpha = alpha                            # Laplace smoothing for 0/1 columns

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # find the 0/1 columns (only the "mixed" likelihood needs them)
        self.binary_ = np.zeros(X.shape[1], dtype=bool)
        if self.likelihood == "mixed":
            for j in range(X.shape[1]):
                values = np.unique(X[:, j])
                if np.all(np.isin(values, (0.0, 1.0))):
                    self.binary_[j] = True

        # smooth the variances the same way sklearn's GaussianNB does
        epsilon = self.var_smoothing * np.var(X, axis=0).max()

        self.means_ = []
        self.variances_ = []
        self.priors_ = []
        self.binary_p_ = []
        for c in self.classes_:
            Xc = X[y == c]
            self.means_.append(Xc.mean(axis=0))
            self.variances_.append(Xc.var(axis=0) + epsilon)
            self.priors_.append(len(Xc) / len(X))
            if self.binary_.any():
                counts = Xc[:, self.binary_].sum(axis=0)
                self.binary_p_.append((counts + self.alpha) / (len(Xc) + 2 * self.alpha))

        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        return self

    def _log_likelihood(self, X, class_index):
        """log P(attribute_j = x_j | class) for every attribute j, one row per instance."""
        log_p = np.empty(X.shape)

        if self.binary_.any():
            p = self.binary_p_[class_index]
            Xb = X[:, self.binary_]
            log_p[:, self.binary_] = Xb * np.log(p) + (1 - Xb) * np.log(1 - p)

        mean = self.means_[class_index]
        variance = self.variances_[class_index]
        gaussian = -0.5 * (np.log(2 * np.pi * variance) + (X - mean) ** 2 / variance)
        log_p[:, ~self.binary_] = gaussian[:, ~self.binary_]
        return log_p

    def joint_log_likelihood(self, X):
        """log P(C) + sum of (weight * log P(attribute | C)), one column per class."""
        scores = np.zeros((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            log_p = self._log_likelihood(X, k)
            scores[:, k] = np.log(self.priors_[k]) + (log_p * self.weights).sum(axis=1)
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
