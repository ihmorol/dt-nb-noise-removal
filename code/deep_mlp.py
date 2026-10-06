"""The deep-learning model: a multilayer perceptron in PyTorch.

The most fundamental deep network there is - fully connected layers, ReLU
nonlinearity, trained by backpropagation with Adam - the architecture deep
learning starts from. Kept deliberately small (one hidden layer of 32 units)
so the comparison stays about the DATA (old vs new), not about model capacity:
a huge network could memorize noise and muddy the attribution.

One hidden layer is enough for validity: an MLP with a single hidden layer and
a nonlinearity is a universal approximator (Hornik 1989), and this is the same
shape every deep-learning textbook builds first.

Features must be standardized before fit (the experiment does that for every
model, so LR and MLP see the same matrix). Labels may be any strings/numbers.
"""
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit

torch.set_num_threads(4)


class DeepMLP:
    def __init__(self, hidden=32, lr=1e-3, max_epochs=150, batch_size=256,
                 patience=10, weight_decay=1e-4, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.classes_ = np.unique(y)
        targets = np.searchsorted(self.classes_, y)          # labels -> 0..C-1

        # a small stratified validation split for early stopping; on tiny data
        # (a class with <5 rows) it is not worth it, so we just run the epochs
        val_X = val_y = None
        if len(y) >= 60 and np.min(np.bincount(targets)) >= 5:
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                                           random_state=self.seed)
            train_idx, val_idx = next(split.split(X, targets))
            val_X, val_y = X[val_idx], targets[val_idx]
        else:
            train_idx = np.arange(len(y))
        train_X = torch.from_numpy(X[train_idx])
        train_y = torch.from_numpy(targets[train_idx])

        d, n_classes = X.shape[1], len(self.classes_)
        self.net_ = nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(),
                                  nn.Linear(self.hidden, n_classes))
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_val, best_state, waited = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net_.train()
            order = torch.randperm(len(train_X))
            for start in range(0, len(train_X), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net_(train_X[batch]), train_y[batch])
                loss.backward()
                optimizer.step()

            self.net_.eval()
            with torch.no_grad():
                if val_X is not None:
                    watched = loss_fn(self.net_(torch.from_numpy(val_X)),
                                      torch.from_numpy(val_y)).item()
                else:
                    watched = loss.item()             # plateau of the train loss
            if watched < best_val - 1e-5:
                best_val, waited = watched, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.net_.eval()
        with torch.no_grad():
            picked = self.net_(X).argmax(axis=1).numpy()
        return self.classes_[picked]
