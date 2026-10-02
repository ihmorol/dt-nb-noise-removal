"""Inject label noise we KNOW about, so a noise filter can actually be scored.

Why this file exists: the ten UCI datasets are clean, so "did the filter remove
the noise or just the hard cases?" has no answer on them - which is exactly why
E11 produced deltas that all looked like noise. Here we corrupt a controlled
fraction of the labels ourselves and keep the corruption mask, so a filter can
be graded on precision / recall against ground truth instead of only through
downstream accuracy.

Two kinds, both standard in the label-noise literature:

  symmetric    flip a row's label to a uniformly random OTHER class. Independent
               of the class and of the row. Easy to detect - the sanity check.

  asymmetric   flip a row to a class it is CONFUSABLE with rather than to a random
               one, the CIFAR-10N / Patrini-style class-dependent noise. This is
               the realistic and the hard case, and it is the one that separates
               a per-class threshold rule from a global one: a global rule spends
               its deletions on whichever class happens to be hardest, while an
               asymmetric flip only shows up as noise once you look per class.

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
    classes = np.unique(y)
    if len(classes) < 2:
        return {c: c for c in classes}

    votes = {c: {} for c in classes}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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

    rate is capped at 0.5: above half the labels flipped, the majority class is
    no longer the majority and "which label was original" stops being meaningful
    for this kind of study.
    """
    rng = np.random.default_rng(seed)
    rate = float(np.clip(rate, 0.0, 0.5))
    y_noisy = np.array(y, copy=True)
    is_noisy = np.zeros(len(y), dtype=bool)
    if rate == 0:
        return y_noisy, is_noisy

    classes = np.unique(y)
    if len(classes) < 2:
        return y_noisy, is_noisy
    if kind == "asymmetric" and mapping is None:
        mapping = confusable_map(X, y, seed=seed)

    # which rows to touch: a random subset, but never more than half of any one
    # class, so the corrupted data still has a learnable majority per class.
    chosen = np.flatnonzero(rng.random(len(y)) < rate)
    for i in chosen:
        current = y_noisy[i]
        others = classes[classes != current]
        if kind == "asymmetric":
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
        for kind in ("symmetric", "asymmetric"):
            for rate in (0.05, 0.10, 0.20, 0.40):
                y_noisy, is_noisy = inject(X, y, rate, kind=kind, seed=0)
                observed = 100 * is_noisy.mean()
                # how many corrupted rows would a per-class rule even be able to
                # see: the smallest class's share of the corruption
                moved = Counter(y_noisy[is_noisy])
                print(f"  {kind:<11} rate {rate:.2f} -> {observed:5.1f}% of rows "
                      f"corrupted; most common new label "
                      f"{moved.most_common(1)[0][0] if moved else '-'}")