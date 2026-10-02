"""E12's noise filters: DT and NB as judges, compared separately.

The problem with Farid's Algorithm 1 judge (a single NB that deletes what it
gets wrong on its own training data):

  * it grades rows it was TRAINED on, so a wrong prediction there can be model
    bias or a genuinely hard region, not a wrong label;
  * a single learner's errors are systematic - it deletes the rows IT finds hard,
    not the rows that are actually mislabeled (Brodley & Friedl 1999);
  * a single global rule spends its deletions on whichever class happens to be
    hardest. On NSL-KDD the E11 committee deleted 100% of the ftp_write, spy and
    land rows and cost 7.1 macro-F1 points (notes/e11-lr-mlp-hybrid-data.md).

Three methods, run with one judge at a time so DT and NB are compared separately:

  method="hard_vote"      delete every row the judge gets wrong. This is
                          Algorithm 1's rule, the baseline to beat.
  method="consensus"      E11's committee: delete only when every judge of every
                          repeat disagrees. High precision, and it is what wipes
                          out rare classes.
  method="confident_joint"  Confident Learning (Northcutt, Jiang & Chuang, JAIR
                          2021, arXiv:1911.00068). Per-class thresholds instead
                          of votes: class j gets its OWN threshold, the mean
                          self-confidence of the rows GIVEN label j. A row is
                          noise when the judge disagrees AND is confident enough
                          about its own answer to clear that class's bar.

Why per-class thresholds are the point. A row counts as noise only if the judge
believes in its disagreement: argmax != y AND p[argmax] >= t[argmax]. The
threshold is per class precisely so that a hard class, whose rows all get low
confidence, is not held to a bar it can never clear - which is exactly what
deleted those KDD minority classes. It also yields a noise-rate estimate per
class, which a vote count cannot.

All judges are cross-validated: a row is only ever scored by a model that did
not train on it. E3a showed what skipping that costs.

    python confident_filter.py        self-check on a noised iris
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from nb import WeightedNB


def _judge(name, seed=0):
    """The judge: one of the two learners committee_filter.py uses."""
    if name == "DT":
        return DecisionTreeClassifier(criterion="entropy", random_state=seed)
    if name == "NB":
        return WeightedNB(likelihood="mixed")
    raise ValueError(f"unknown judge {name!r}; use 'DT' or 'NB'")


def out_of_fold_probabilities(X, y, judge, n_splits=5, repeats=1, seed=0):
    """Average P(class | x) over `repeats` x `n_splits` cross-validated fits.

    Every row is predicted only by models that never saw it. Returns
    (classes, probabilities) with the probability columns in `classes` order.
    """
    classes = np.unique(y)
    order = {c: i for i, c in enumerate(classes)}
    total = np.zeros((len(y), len(classes)))
    counts = np.zeros(len(y))

    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed + repeat)
        for rest, held_out in splitter.split(X, y):
            model = _judge(judge, seed).fit(X[rest], y[rest])
            posterior = model.predict_proba(X[held_out])

            # a judge fitted on a subset may not know every class; put its
            # columns back into the global class order, zeros where it is blind
            block = np.zeros((len(held_out), len(classes)))
            for k, c in enumerate(model.classes_):
                block[:, order[c]] = posterior[:, k]
            total[held_out] += block
            counts[held_out] += 1

    if not np.all(counts > 0):
        raise RuntimeError("confident filter: some rows were never predicted")
    return classes, total / counts[:, None]
def confident_joint(classes, probabilities, y):
    """Confident Learning's off-diagonal test. Returns (noise_mask, thresholds).

    t_j = mean P(j | x) over the rows GIVEN label j. A row is noise when the
    judge's argmax disagrees with its label and the judge's confidence in that
    argmax clears the threshold of the class it argued for.
    """
    order = {c: i for i, c in enumerate(classes)}
    given = np.array([order[v] for v in y])
    predicted = probabilities.argmax(axis=1)
    best = probabilities.max(axis=1)

    thresholds = np.ones(len(classes))
    for i, c in enumerate(classes):
        rows = given == i
        if rows.any():
            thresholds[i] = probabilities[rows, i].mean()

    noise = (predicted != given) & (best >= thresholds[predicted])
    return noise, thresholds


def _enforce_class_floor(noise, y, min_per_class):
    """Never delete a class below min_per_class rows. This is the E11 fix.

    Returns (noise, rescued) where rescued counts the rows the floor put back.
    A non-zero count is the guard actively saving a rare class, and it is logged
    so the paper can report how often it had to intervene.
    """
    rescued = 0
    for c in np.unique(y):
        rows = np.flatnonzero(noise & (y == c))
        survivors = int((~noise & (y == c)).sum())
        if survivors < min_per_class and len(rows):
            put_back = min(min_per_class - survivors, len(rows))
            rescued += put_back
            noise[rows[:put_back]] = False
    return noise, rescued


def _committee_votes(X, y, n_splits, repeats, rule, seed):
    """E11's committee: DT and NB together, repeated k-fold votes."""
    votes = np.zeros(len(y), dtype=int)
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed + repeat)
        for rest, held_out in splitter.split(X, y):
            for judge in ("DT", "NB"):
                model = _judge(judge, seed).fit(X[rest], y[rest])
                votes[held_out] += (model.predict(X[held_out]) != y[held_out])
    max_votes = 2 * repeats
    if rule == "consensus":
        noise = votes >= max_votes
    elif rule == "majority":
        noise = votes > max_votes / 2
    else:
        raise ValueError(f"unknown rule {rule!r}; use 'consensus' or 'majority'")
    return noise, votes, max_votes


def confident_filter(X, y, judge="DT", method="confident_joint", n_splits=5,
                     repeats=1, rule="consensus", min_per_class=5, seed=0):
    """Mark the noisy training rows. Returns (keep, info).

    judge            "DT", "NB", or "committee" (E11's pair) - one at a time, so
                     DT and NB can be compared separately
    method           "confident_joint" (per-class thresholds, the new method),
                     "hard_vote" (delete what the judge gets wrong = Algorithm 1),
                     or "consensus" (E11's committee of both judges)
    rule             for method="consensus" only: "consensus" or "majority"
    min_per_class    per-class floor; 0 disables the guard

    keep[i] is True for the rows that stay. info carries the diagnostics:
    votes, per-class thresholds, the estimated noise rate per class, the removal
    percentage, per-class removal rates, and the judge's proposed new labels.
    """
    # the committee has no single judge, so its out-of-fold probabilities (used
    # only for diagnostics and the AUROC score) come from the DT member
    proba_judge = "DT" if judge == "committee" else judge
    classes, probabilities = out_of_fold_probabilities(X, y, proba_judge,
                                                      n_splits=n_splits,
                                                      repeats=repeats, seed=seed)
    order = {c: i for i, c in enumerate(classes)}
    given = np.array([order[v] for v in y])
    predicted = probabilities.argmax(axis=1)
    best = probabilities.max(axis=1)

    if method == "hard_vote":
        # Algorithm 1's rule, but scored out-of-fold instead of on the rows the
        # judge trained on. The vote count is the disagreement over repeats.
        votes = np.zeros(len(y), dtype=int)
        for repeat in range(repeats):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed + repeat)
            for rest, held_out in splitter.split(X, y):
                model = _judge(judge, seed).fit(X[rest], y[rest])
                votes[held_out] += (model.predict(X[held_out]) != y[held_out])
        noise = (votes > 0) if repeats == 1 else (
            votes >= repeats if rule == "consensus" else votes > repeats / 2)
        max_votes = repeats
        thresholds = np.ones(len(classes))
    elif method == "confident_joint":
        if judge == "committee":
            raise ValueError("the confident joint needs a single judge; use "
                             "judge='DT' or 'NB'")
        noise, thresholds = confident_joint(classes, probabilities, y)
        # a coarse confidence, only so every method reports the same histogram
        max_votes = 10
        votes = np.rint(best * max_votes).astype(int)
    elif method == "consensus":
        noise, votes, max_votes = _committee_votes(X, y, n_splits, repeats,
                                                   rule, seed)
        thresholds = np.ones(len(classes))
    else:
        raise ValueError(f"unknown method {method!r}; use 'confident_joint', "
                         f"'hard_vote' or 'consensus'")

    noise = np.asarray(noise, dtype=bool).copy()
    before_floor = int(noise.sum())
    noise, rescued = _enforce_class_floor(noise, y, min_per_class)

    # the confident joint's own noise estimate per class: the off-diagonal mass,
    # i.e. the share of class j's rows the judge disputes
    disputed = noise & (predicted != given)
    noise_rate = {}
    for c in classes:
        rows = given == order[c]
        noise_rate[c] = float(disputed[rows].mean()) if rows.any() else 0.0

    info = {
        "judge": judge, "method": method, "rule": rule,
        "repeats": repeats, "n_splits": n_splits, "max_votes": max_votes,
        "votes": votes, "probabilities": probabilities, "classes": classes,
        "thresholds": dict(zip(classes, np.round(thresholds, 4))),
        "predicted": classes[predicted], "confidence": best,
        "removed_rows": np.flatnonzero(noise),
        "removed_pct": 100.0 * float(noise.mean()),
        "removed_before_floor": before_floor,
        "rescued_by_floor": rescued,
        "per_class": {c: float(np.mean(noise[y == c])) for c in np.unique(y)},
        "noise_rate_estimate": noise_rate,
        "relabelled": classes[predicted],
    }
    return ~noise, info


if __name__ == "__main__":
    from data import load_data
    from noise_inject import inject

    X, y, _ = load_data("iris")
    y_noisy, is_noisy = inject(X, y, 0.20, kind="asymmetric", seed=0)
    print(f"iris, 20% asymmetric noise: {100 * is_noisy.mean():.1f}% of rows "
          f"corrupted\n")
    header = (f"{'judge':<5}{'method':<17}{'removed%':>9}{'precision':>11}"
              f"{'recall':>8}{'F1':>8}{'rescued':>9}")
    print(header)
    for judge in ("DT", "NB"):
        for method in ("hard_vote", "confident_joint"):
            keep, info = confident_filter(X, y_noisy, judge=judge, method=method,
                                          min_per_class=5, seed=0)
            removed = ~keep
            hits = int((removed & is_noisy).sum())
            precision = hits / max(1, removed.sum())
            recall = hits / max(1, is_noisy.sum())
            f1 = 0.0 if precision + recall == 0 else \
                2 * precision * recall / (precision + recall)
            print(f"{judge:<5}{method:<17}{info['removed_pct']:>8.1f}%"
                  f"{precision:>11.3f}{recall:>8.3f}{f1:>8.3f}"
                  f"{info['rescued_by_floor']:>9}")

    _, info = confident_filter(X, y_noisy, judge="DT", method="confident_joint",
                               min_per_class=5, seed=0)
    print("\nDT confident_joint, estimated noise rate per class "
          "(vs the truth):")
    truth = {c: round(float(is_noisy[y_noisy == c].mean()), 3)
             for c in np.unique(y_noisy)}
    for c, rate in info["noise_rate_estimate"].items():
        print(f"  class {c}: estimated {rate:.3f}   actual {truth[c]:.3f}")