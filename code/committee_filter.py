"""The reliable noise filter: a cross-validated committee of DT and NB.

Why not Farid's Algorithm 1 judge (a single NB that deletes what it gets wrong):

  * it classifies rows it was TRAINED on, so a wrong prediction can be model
    bias or a genuinely hard region of the data, not a wrong label;
  * a single learner's errors are systematic - it removes the rows IT finds
    hard, not the rows that are actually mislabeled. Brodley & Friedl (1999,
    JAIR 11) tested exactly this and found single-algorithm filters are the
    weakest design.

What the literature says makes a filter reliable:

  * a committee of DIFFERENT learners votes per row, cross-validated so no
    judge ever scores a row it trained on (Brodley & Friedl 1999);
  * the voting is repeated over several random partitionings and the votes are
    summed, which averages out the luck of one split (Iterative-Partitioning
    Filter, Khoshgoftaar & Rebours 2007, JCST 22(3):387-396);
  * the CONSENSUS rule - delete only when every judge disagrees in every
    repeat - is the most conservative and the hardest to fool; the MAJORITY
    rule is the looser variant (Brodley & Friedl 1999, section 3).

So a row is "noise" here only when BOTH the decision tree AND Naive Bayes,
trained without that row, misclassify it - in every repeat.

NB stays in the committee rather than being thrown out: Johnson & Khoshgoftaar
(2022, ACM Computing Surveys) report NB as the most stable learner under label
noise - a good committee member, a bad sole judge.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from nb import WeightedNB


def committee_filter(X, y, rule="consensus", n_splits=3, repeats=3, seed=0):
    """Vote on every row: is it noise?  Returns (keep, info).

    keep    keep[i] is True for the rows that stay
    info    vote diagnostics: votes per row, removal rate, per-class rates

    Each repeat splits the rows into n_splits stratified parts. Both judges are
    fitted on the other parts only, and predict the held-out part. Every row is
    held out exactly once per repeat and gets 2 votes (one per judge), so a row
    collects at most 2*repeats votes.

    rule = "consensus":  noise iff ALL votes are against it (max_votes/2 per repeat
                         means both judges, every repeat)
    rule = "majority":   noise iff MORE than half the votes are against it
    """
    n = len(y)
    votes = np.zeros(n, dtype=int)       # votes against the row
    possible = np.zeros(n, dtype=int)    # votes the row could have received

    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed + repeat)
        for rest, held_out in splitter.split(X, y):   # split gives (train, test)
            tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
            tree.fit(X[rest], y[rest])
            nb = WeightedNB(likelihood="mixed").fit(X[rest], y[rest])

            votes[held_out] += (tree.predict(X[held_out]) != y[held_out])
            votes[held_out] += (nb.predict(X[held_out]) != y[held_out])
            possible[held_out] += 2

    if not np.all(possible == 2 * repeats):
        raise RuntimeError("committee filter: some rows were never voted on")

    max_votes = 2 * repeats
    if rule == "consensus":
        noise = votes >= max_votes
    elif rule == "majority":
        noise = votes > max_votes / 2
    else:
        raise ValueError(f"unknown rule {rule!r}; use 'consensus' or 'majority'")

    # safety rule: never delete a whole class
    skipped = len(np.unique(y[~noise])) < 2
    if skipped:
        noise = np.zeros(n, dtype=bool)

    per_class = {c: float(np.mean(noise[y == c])) for c in np.unique(y)}
    info = {
        "rule": rule,
        "repeats": repeats,
        "max_votes": max_votes,
        "votes": votes,
        "removed_rows": np.flatnonzero(noise),
        "removed_pct": 100.0 * np.mean(noise),
        "per_class": per_class,
        "skipped": skipped,
    }
    return ~noise, info
