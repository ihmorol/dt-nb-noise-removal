"""Inject label noise we KNOW about, so a noise filter can actually be scored.

Why this file exists: the ten UCI datasets have no verified row-level noise
ground truth, so "did the filter remove the noise or just the hard cases?" has
no answer on them. Here we corrupt a controlled fraction of labels ourselves
and keep the corruption mask, so a filter can be graded against known injected
corruption instead of treating the original labels as proven clean.

Three controlled corruption mechanisms:

  symmetric    flip a row's label to a uniformly random OTHER class. Independent
               of the class and of the row. Easy to detect - the sanity check.

  pairflip     flip each sorted class to the next sorted class. This standard,
               fixed class-dependent mechanism is independent of a fitted model.

  asymmetric   flip to the class an NB judge confuses with the source class most.
               This is deliberately data-derived and NB-biased; it must not be
               presented as an independent or generally realistic noise source.

The confusable-class map is estimated by cross-validated NB *inside the data it is
estimated from*, so nothing about the test fold leaks in. Call inject() on one
training fold at a time - never on a whole dataset before splitting.

Usage: python noise_inject.py        show the corruption rates per kind
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from nb import WeightedNB


def confusable_map(X, y, seed=0, n_splits=3):
    """class -> the other class an out-of-fold NB confuses it with most often.

    Estimated by cross-validation so the map never sees a row's own prediction
    from a model that trained on it. Ties and self-confusions are excluded, so
    every returned value is a DIFFERENT class.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("X must be a non-empty 2D array matching 1D y")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("X must contain only finite numeric values")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {c: c for c in classes}

    votes = {c: {} for c in classes}
    effective_splits = min(n_splits, int(counts.min()))
    if effective_splits < 2:
        return {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True,
                               random_state=seed)
    for rest, held_out in splitter.split(X, y):
        model = WeightedNB(likelihood="mixed").fit(X[rest], y[rest])
        predicted = model.predict(X[held_out])
        for actual, guess in zip(y[held_out], predicted):
            if actual != guess:
                votes[actual][guess] = votes[actual].get(guess, 0) + 1

    mapping = {}
    for c in classes:
        others = votes[c]
        mapping[c] = max(others, key=others.get) if others else classes[0]
        if mapping[c] == c:                      # cannot happen, but be certain
            mapping[c] = classes[1] if len(classes) > 1 else c
    return mapping


def inject(X, y, rate, kind="symmetric", seed=0, mapping=None):
    """Corrupt `rate` of the labels. Returns (y_noisy, is_noisy).

    is_noisy[i] is True exactly where we changed the label - that is the ground
    truth every filter in E12 is graded against.

    ``rate`` must be between 0 and 0.5 inclusive. Invalid experiment settings
    raise instead of being silently clipped.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("X must be a non-empty 2D array matching 1D y")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("X must contain only finite numeric values")
    if (np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all()) or any(
            value is None or value != value for value in y):
        raise ValueError("y must contain only finite, non-missing labels")
    if kind not in ("symmetric", "asymmetric", "pairflip"):
        raise ValueError("kind must be 'symmetric', 'asymmetric', or 'pairflip'")
    if not np.isscalar(rate):
        raise ValueError("rate must be a finite number between 0 and 0.5")
    try:
        rate = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("rate must be a finite number between 0 and 0.5") from exc
    if not np.isfinite(rate):
        raise ValueError("rate must be a finite number between 0 and 0.5")
    if not 0.0 <= rate <= 0.5:
        raise ValueError("rate must be between 0 and 0.5")
    rng = np.random.default_rng(seed)
    y_noisy = np.array(y, copy=True)
    is_noisy = np.zeros(len(y), dtype=bool)
    if rate == 0:
        return y_noisy, is_noisy

    classes = np.unique(y)
    if len(classes) < 2:
        return y_noisy, is_noisy
    if kind == "asymmetric" and mapping is None:
        mapping = confusable_map(X, y, seed=seed)
    elif kind == "pairflip":
        mapping = {c: classes[(i + 1) % len(classes)]
                   for i, c in enumerate(classes)}
    elif mapping is not None:
        raise ValueError("mapping is only valid for kind='asymmetric'")
    if mapping is not None:
        for source, target in mapping.items():
            if source not in classes or target not in classes or source == target:
                raise ValueError("mapping must map known classes to different known classes")

    # Bernoulli sampling gives an expected rate. It does not guarantee a cap per
    # class, so reports must use the observed corruption mask rather than claim
    # that every class retained a majority.
    chosen = np.flatnonzero(rng.random(len(y)) < rate)
    for i in chosen:
        current = y_noisy[i]
        others = classes[classes != current]
        if kind in ("asymmetric", "pairflip"):
            target = mapping.get(current, others[0])
            if target == current:                        # map degenerated
                target = others[0]
        else:
            target = others[rng.integers(len(others))]
        y_noisy[i] = target
        is_noisy[i] = True
    return y_noisy, is_noisy


if __name__ == "__main__":
    from collections import Counter

    from data import available, load_data

    for name in ("iris", "glass", "contact-lenses"):
        if name not in available():
            continue
        X, y, _ = load_data(name)
        sizes = Counter(y)
        print(f"\n{name}: {len(y)} rows, {len(sizes)} classes, "
              f"smallest class {min(sizes.values())} rows")
        for kind in ("symmetric", "pairflip", "asymmetric"):
            for rate in (0.05, 0.10, 0.20, 0.40):
                y_noisy, is_noisy = inject(X, y, rate, kind=kind, seed=0)
                observed = 100 * is_noisy.mean()
                # how many corrupted rows would a per-class rule even be able to
                # see: the smallest class's share of the corruption
                moved = Counter(y_noisy[is_noisy])
                print(f"  {kind:<11} rate {rate:.2f} -> {observed:5.1f}% of rows "
                      f"corrupted; most common new label "
                      f"{moved.most_common(1)[0][0] if moved else '-'}")
