# Code

## Files

```
code/
├── data.py             load the ten datasets (original columns or one-hot)
├── nb.py               the pipeline's Naive Bayes (plain or weighted, Eq. 14)
├── algorithm1.py       Farid Algorithm 1: NB deletes the training rows it gets wrong
├── algorithm2.py       Farid Algorithm 2: a tree picks and weights the attributes
├── pipeline.py         run the algorithms in order, then cross-validate NB or DT
├── versions.py         save the data before/after each path and what was removed
├── tables.py           printed tables and CSV writing
├── main.py             E9: run every arm on every dataset
├── leakage_check.py    E3a: honest vs paper-style vs fully leaky protocol
├── faithful_nb.py      R1: textbook NB on the original (nominal) columns
├── weka_utils.py       R1+E9f: write ARFF, run Weka J48 and NaiveBayes,
│                       parse the tree, the accuracy and the confusion matrix
├── replicate_farid.py  R1: replicate Farid's Tables 8-11 with J48
├── e9_farid.py         E9f: the manuscript's Table II — both orders of the two
│                       algorithms in this same J48/FaithfulNB world, protocols
│                       A+B, accuracy + macro-F1 + per-class removals
├── e5_strategy.py      E5: graded noise handling — soft posterior weighting,
│                       committee correction/deletion vs Farid's hard deletion
├── check_pipeline.py   checks the leakage rules of pipeline.py (no Java)
├── verify_faithful.py  R1: checks to run before replicate_farid.py
├── figures.py          the paper's Figures 2 and 3 with our hybrid added
└── lib/                Weka 3.8.6 jars (R1 needs Java)
```

## How to run

Run from inside `code/` (first: `python -m venv .venv && .venv/bin/pip install -r ../requirements.txt`; `pandas<3` is load-bearing, see the file):

```
python data.py                     dataset shapes vs Farid's Table 5
python check_pipeline.py           pipeline leakage rules (all must PASS)
python main.py                     E9, all datasets, final settings (5 seeds, alpha 0, mixed NB)
python main.py iris glass --seeds 3
python main.py --alpha 0.01 --nb gaussian --out metrics_alpha0.01.csv
python leakage_check.py            E3a
python verify_faithful.py          R1 checks (all must PASS)
python replicate_farid.py          R1, pruned J48, 3 seeds
python replicate_farid.py --unpruned
python e9_farid.py                 E9f, all datasets, 10 seeds (the manuscript's Table II)
python e9_farid.py iris --seeds 1  E9f pilot
python e5_strategy.py              E5, all datasets, 10 seeds (graded noise handling)
python e5_strategy.py iris --seeds 1
python figures.py
```

`main.py` writes into `../results/EXP-E9_pipeline/`:

| File | Contents |
|---|---|
| `metrics*.csv` | accuracy, macro-F1 and removal % per dataset, arm and classifier, plus the settings used |
| `stages.csv` | per algorithm step: rows/attributes before and after, averaged over the folds |
| `versions_removals.csv` | what the one pass over the whole dataset removed |
| `versions/<dataset>/` | the data before and after each path, and `removals.txt` |

## Rules

1. Every experiment is one script run that writes into `../results/EXP-<id>_<name>/`.
2. Seeds are fixed and written into the output files.
3. Everything (filter, attribute selection, classifier) is fitted on the training fold only.
4. Commit before and after every experiment run.
