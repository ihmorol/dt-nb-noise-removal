"""The Naive Bayes model of the pipeline (one-hot data).

It is used as Algorithm 1's judge, as Algorithm 2's classifier, and as the final NB.

    weights     one exponent per attribute, Farid's Eq. (14):
                P(x|C) = P(C) * product of P(A_j|C) ^ W_j
                None means every exponent is 1, which is plain Naive Bayes.
                (a column scored 0 by algorithm2 is not kept at all - see
                algorithm2 - so W_j = 0 does not have to be handled here.)
    likelihood  "gaussian": every column is a bell curve per class.
                "mixed":    0/1 columns use the Bernoulli formula, the rest stay
                            Gaussian (closer to how Weka treats nominal attributes).
"""
import numpy as np


class WeightedNB:
    def __init__(self, weights=None, likelihood="gaussian"):
        self.weights = weights
        self.likelihood = likelihood

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_columns = X.shape[1]
        self.weights_ = np.ones(n_columns) if self.weights is None else np.asarray(self.weights)

        # which columns only hold 0 and 1 (only the "mixed" likelihood uses this)
        if self.likelihood == "mixed":
            self.binary_ = np.all((X == 0) | (X == 1), axis=0)
        else:
            self.binary_ = np.zeros(n_columns, dtype=bool)

        # a tiny extra variance so no column has variance 0 (same as sklearn's GaussianNB)
        epsilon = 1e-9 * np.var(X, axis=0).max()

        self.priors_, self.means_, self.variances_, self.p_one_ = [], [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_.append(len(Xc) / len(X))
            self.means_.append(Xc.mean(axis=0))
            self.variances_.append(Xc.var(axis=0) + epsilon)
            # P(column = 1 | class), with add-one (Laplace) smoothing
            self.p_one_.append((Xc[:, self.binary_].sum(axis=0) + 1) / (len(Xc) + 2))
        return self

    def class_scores(self, X):
        """log P(C) + sum over columns of W_j * log P(x_j | C). One column per class."""
        scores = np.zeros((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            mean, variance = self.means_[k], self.variances_[k]
            log_p = -0.5 * (np.log(2 * np.pi * variance) + (X - mean) ** 2 / variance)

            p = self.p_one_[k]
            Xb = X[:, self.binary_]
            log_p[:, self.binary_] = Xb * np.log(p) + (1 - Xb) * np.log(1 - p)

            scores[:, k] = np.log(self.priors_[k]) + (log_p * self.weights_).sum(axis=1)
        return scores

    def predict(self, X):
        return self.classes_[self.class_scores(X).argmax(axis=1)]

    def predict_proba(self, X):
        """P(C | x) for every class: softmax over the class scores.

        Bayes' rule - class_scores already holds log P(C) + sum_j W_j log P(x_j|C),
        so normalizing those exponentials gives the posterior. Needed by every
        threshold-based filter (E12's confident joint), which thresholds a
        confidence rather than counting hard votes.
        """
        scores = self.class_scores(X)
        scores = scores - scores.max(axis=1, keepdims=True)   # keep exp() finite
        posterior = np.exp(scores)
        return posterior / posterior.sum(axis=1, keepdims=True)
