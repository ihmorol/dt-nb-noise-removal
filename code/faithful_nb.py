"""The Naive Bayes of the replication (R1), on the original attribute columns.

The paper's NB is a textbook NB: a nominal attribute is ONE variable with one
frequency table per class, not a set of 0/1 columns. This class does that.

    nominal  column holds the position of its value in levels[j]
             P(value | C) = (count(value, C) + 1) / (n_C + number of values)
    numeric  Gaussian per class, variance smoothed like sklearn's GaussianNB
    weights  Farid's Eq. (14) exponents; None means plain NB

verify_faithful.py checks that this equals sklearn's GaussianNB on numeric data
and Weka's NaiveBayes on nominal data.
"""
import numpy as np


class FaithfulNB:
    def __init__(self, nominal, levels, weights=None):
        self.nominal = np.asarray(nominal, dtype=bool)
        self.levels = levels
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.weights_ = np.ones(X.shape[1]) if self.weights is None else np.asarray(self.weights)

        numeric = ~self.nominal
        epsilon = 1e-9 * np.var(X[:, numeric], axis=0).max() if numeric.any() else 0.0

        self.priors_, self.means_, self.variances_, self.log_tables_ = [], [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_.append(len(Xc) / len(X))
            means, variances, tables = {}, {}, {}
            for j in range(X.shape[1]):
                if self.nominal[j]:
                    n_values = len(self.levels[j])
                    counts = np.bincount(Xc[:, j].astype(int), minlength=n_values)
                    tables[j] = np.log((counts + 1) / (len(Xc) + n_values))
                else:
                    means[j] = Xc[:, j].mean()
                    variances[j] = Xc[:, j].var() + epsilon
            self.means_.append(means)
            self.variances_.append(variances)
            self.log_tables_.append(tables)
        return self

    def class_scores(self, X):
        """log P(C) + sum over attributes of W_j * log P(x_j | C). One column per class."""
        scores = np.zeros((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            total = np.full(len(X), np.log(self.priors_[k]))
            for j in range(X.shape[1]):
                if self.nominal[j]:
                    log_p = self.log_tables_[k][j][X[:, j].astype(int)]
                else:
                    mean, variance = self.means_[k][j], self.variances_[k][j]
                    log_p = -0.5 * (np.log(2 * np.pi * variance) + (X[:, j] - mean) ** 2 / variance)
                total += self.weights_[j] * log_p
            scores[:, k] = total
        return scores

    def predict(self, X):
        return self.classes_[self.class_scores(X).argmax(axis=1)]

    def accuracy(self, X, y):
        return 100 * float((self.predict(X) == y).mean())
