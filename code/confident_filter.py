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

Four methods, including single-judge baselines and two DT+NB committee rules:

  method="hard_vote"      delete every row the judge gets wrong. This is
                          Algorithm 1's rule, the baseline to beat.
  method="consensus"      E11's committee: delete only when every judge of every
                          repeat disagrees. High precision, and it is what wipes
                          out rare classes.
  method="confident_joint"  A per-class confidence-threshold heuristic inspired
                          by Confident Learning. It is not the full calibrated
                          confident-joint estimator from cleanlab.
  method="dual_agreement"   DT and NB must independently clear their per-class
                          thresholds and agree on the same alternative label.

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


def _validate_inputs(X, y, n_splits, repeats):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("X must be a non-empty 2D array matching 1D y")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("X must contain only finite numeric values")
    if (np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all()) or any(
            value is None or value != value for value in y):
        raise ValueError("y must contain only finite, non-missing labels")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be an integer >= 1")
    return X, y


def _normalise_rows(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("posterior must be a non-empty 2D array")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("judge returned a non-finite or negative posterior")
    totals = probabilities.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("judge returned a posterior with zero total probability")
    return probabilities / totals


def _oof_result(X, y, judge, n_splits=5, repeats=1, seed=0, cache=None):
    X, y = _validate_inputs(X, y, n_splits, repeats)
    _judge(judge, seed)  # validate before consulting a caller-owned cache
    key = ("oof", id(X), id(y), judge, n_splits, repeats, seed)
    if cache is not None and key in cache:
        return cache[key]

    classes, class_counts = np.unique(y, return_counts=True)
    order = {c: i for i, c in enumerate(classes)}
    eligible = np.isin(y, classes[class_counts >= 2])
    eligible_rows = np.flatnonzero(eligible)
    total = np.zeros((len(y), len(classes)))
    counts = np.zeros(len(y), dtype=int)
    disagreements = np.zeros(len(y), dtype=int)

    if len(eligible_rows):
        eligible_counts = np.unique(y[eligible], return_counts=True)[1]
        effective_splits = min(n_splits, int(eligible_counts.min()))
        if effective_splits >= 2:
            for repeat in range(repeats):
                splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True,
                                           random_state=seed + repeat)
                for train_rel, held_rel in splitter.split(X[eligible], y[eligible]):
                    held_out = eligible_rows[held_rel]
                    # Singleton classes remain in training for other rows, but
                    # their own row is never scored in-sample.
                    rest = np.concatenate((eligible_rows[train_rel],
                                           np.flatnonzero(~eligible)))
                    model = _judge(judge, seed + repeat).fit(X[rest], y[rest])
                    posterior = _normalise_rows(model.predict_proba(X[held_out]))
                    block = np.zeros((len(held_out), len(classes)))
                    for k, c in enumerate(model.classes_):
                        block[:, order[c]] = posterior[:, k]
                    total[held_out] += block
                    counts[held_out] += 1
                    disagreements[held_out] += model.predict(X[held_out]) != y[held_out]

    scorable = counts > 0
    probabilities = np.zeros_like(total)
    probabilities[scorable] = total[scorable] / counts[scorable, None]
    # A protected row receives an observed-label one-hot vector solely to keep
    # the returned posterior finite and normalized; scorable=False prevents use.
    for row in np.flatnonzero(~scorable):
        probabilities[row, order[y[row]]] = 1.0
    probabilities = _normalise_rows(probabilities)
    result = (classes, probabilities, scorable, disagreements)
    if cache is not None:
        cache[key] = result
    return result


def out_of_fold_probabilities(X, y, judge, n_splits=5, repeats=1, seed=0,
                              cache=None, return_scorable=False):
    """Average P(class | x) over `repeats` x `n_splits` cross-validated fits.

    Every row is predicted only by models that never saw it. Returns
    (classes, probabilities) with the probability columns in `classes` order.
    """
    classes, probabilities, scorable, _ = _oof_result(
        X, y, judge, n_splits, repeats, seed, cache)
    result = (classes, probabilities, scorable) if return_scorable else (
        classes, probabilities)
    return result


def confident_joint(classes, probabilities, y):
    """Per-class off-diagonal threshold heuristic.

    This keeps the historical ``confident_joint`` experiment label, but is not
    a full calibrated Confident Learning joint estimate.

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


def _enforce_class_floor(noise, y, min_per_class, suspicion):
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
            restore = rows[np.argsort(suspicion[rows], kind="stable")[:put_back]]
            noise[restore] = False
    return noise, rescued


def _committee_votes(X, y, n_splits, repeats, rule, seed, cache):
    """E11's committee: DT and NB together, repeated k-fold votes."""
    dt = _oof_result(X, y, "DT", n_splits, repeats, seed, cache)
    nb = _oof_result(X, y, "NB", n_splits, repeats, seed, cache)
    votes = dt[3] + nb[3]
    scorable = dt[2] & nb[2]
    max_votes = 2 * repeats
    if rule == "consensus":
        noise = scorable & (votes >= max_votes)
    elif rule == "majority":
        noise = scorable & (votes > max_votes / 2)
    else:
        raise ValueError(f"unknown rule {rule!r}; use 'consensus' or 'majority'")
    return noise, votes, max_votes, dt, nb


def confident_filter(X, y, judge="DT", method="confident_joint", n_splits=5,
                     repeats=1, rule="consensus", min_per_class=5, seed=0,
                     cache=None):
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
    X, y = _validate_inputs(X, y, n_splits, repeats)
    if not isinstance(min_per_class, (int, np.integer)) or min_per_class < 0:
        raise ValueError("min_per_class must be a non-negative integer")
    if judge not in ("DT", "NB", "committee"):
        raise ValueError("unknown judge; use 'DT', 'NB', or 'committee'")
    if rule not in ("consensus", "majority"):
        raise ValueError("unknown rule; use 'consensus' or 'majority'")

    proba_judge = "DT" if judge == "committee" else judge
    classes, probabilities, scorable, disagreements = _oof_result(
        X, y, proba_judge, n_splits, repeats, seed, cache)
    order = {c: i for i, c in enumerate(classes)}
    given = np.array([order[v] for v in y])
    predicted = probabilities.argmax(axis=1)
    best = probabilities.max(axis=1)
    observed_probability = probabilities[np.arange(len(y)), given]
    suspicion = 1.0 - observed_probability
    proposed = classes[predicted]

    if method == "hard_vote":
        # Algorithm 1's rule, but scored out-of-fold instead of on the rows the
        # judge trained on. The vote count is the disagreement over repeats.
        if judge == "committee":
            raise ValueError("hard_vote needs judge='DT' or 'NB'")
        votes = disagreements
        noise = (votes > 0) if repeats == 1 else (
            votes >= repeats if rule == "consensus" else votes > repeats / 2)
        noise &= scorable
        max_votes = repeats
        thresholds = np.ones(len(classes))
    elif method == "confident_joint":
        if judge == "committee":
            raise ValueError("the confident joint needs a single judge; use "
                             "judge='DT' or 'NB'")
        noise, thresholds = confident_joint(classes, probabilities, y)
        noise &= scorable
        # a coarse confidence, only so every method reports the same histogram
        max_votes = 10
        votes = np.rint(best * max_votes).astype(int)
    elif method == "consensus":
        noise, votes, max_votes, dt, nb = _committee_votes(
            X, y, n_splits, repeats, rule, seed, cache)
        probabilities = _normalise_rows((dt[1] + nb[1]) / 2.0)
        scorable = dt[2] & nb[2]
        observed_probability = probabilities[np.arange(len(y)), given]
        suspicion = 1.0 - observed_probability
        predicted = probabilities.argmax(axis=1)
        best = probabilities.max(axis=1)
        proposed = classes[predicted]
        thresholds = np.ones(len(classes))
    elif method == "dual_agreement":
        if judge != "committee":
            raise ValueError("dual_agreement needs judge='committee'")
        dt = _oof_result(X, y, "DT", n_splits, repeats, seed, cache)
        nb = _oof_result(X, y, "NB", n_splits, repeats, seed, cache)
        dt_noise, dt_thresholds = confident_joint(classes, dt[1], y)
        nb_noise, nb_thresholds = confident_joint(classes, nb[1], y)
        dt_pred, nb_pred = dt[1].argmax(axis=1), nb[1].argmax(axis=1)
        scorable = dt[2] & nb[2]
        noise = scorable & dt_noise & nb_noise & (dt_pred == nb_pred)
        proposed = classes[dt_pred]
        probabilities = _normalise_rows((dt[1] + nb[1]) / 2.0)
        observed_probability = probabilities[np.arange(len(y)), given]
        suspicion = 1.0 - observed_probability
        predicted = probabilities.argmax(axis=1)
        best = probabilities.max(axis=1)
        thresholds = (dt_thresholds + nb_thresholds) / 2.0
        votes = dt[3] + nb[3]
        max_votes = 2 * repeats
    else:
        raise ValueError(f"unknown method {method!r}; use 'confident_joint', "
                         f"'hard_vote', 'consensus', or 'dual_agreement'")

    noise = np.asarray(noise, dtype=bool).copy()
    before_floor = int(noise.sum())
    noise, rescued = _enforce_class_floor(noise, y, min_per_class, suspicion)

    # Historical key retained for result compatibility. This is the post-floor
    # flagged disagreement rate, not a calibrated estimate of true noise.
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
        "observed_label_probability": observed_probability,
        "suspicion": suspicion, "suspicion_score": suspicion,
        "scorable": scorable,
        "removed_rows": np.flatnonzero(noise),
        "removed_pct": 100.0 * float(noise.mean()),
        "removed_before_floor": before_floor,
        "rescued_by_floor": rescued,
        "per_class": {c: float(np.mean(noise[y == c])) for c in np.unique(y)},
        "noise_rate_estimate": noise_rate,
        "relabelled": proposed,
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

    # Rare classes cannot be honestly OOF-scored. They stay protected while all
    # returned probability rows remain finite and normalized.
    X_tiny = np.arange(14, dtype=float).reshape(7, 2)
    y_tiny = np.array([0, 0, 0, 1, 1, 1, 2])
    keep, info = confident_filter(X_tiny, y_tiny, judge="committee",
                                  method="dual_agreement", min_per_class=1)
    assert keep[-1] and not info["scorable"][-1]
    assert np.isfinite(info["probabilities"]).all()
    assert np.allclose(info["probabilities"].sum(axis=1), 1.0)
