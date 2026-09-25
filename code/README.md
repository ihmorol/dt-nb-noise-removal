# Code conventions (keep it boring and reproducible)

## Layout

```
code/
├── data.py         # data selection and structuring: load_data(name) -> X, y, meta
├── nb.py           # the Naive Bayes model both algorithms share (plain or weighted)
├── algorithm1.py   # Farid (2014) Algorithm 1 -- NB deletes the instances it gets wrong
├── algorithm2.py   # Farid (2014) Algorithm 2 -- tree selects and weights attributes
├── pipeline.py     # the two paths, the classifiers, the cross-validation
├── versions.py     # the three data versions of a run + what each algorithm removed
├── tables.py       # the printed tables and the CSV files
└── main.py         # run configuration, argument parsing, the run loop
```

Each file holds one thing, and none of them is long. `nb.py` is separate because both
algorithms need the same NB (Algorithm 1 as its judge, Algorithm 2 as its classifier).

## How to run

```
python main.py                       # every dataset found under ../data/raw
python main.py iris glass            # a subset
python main.py --seeds 3             # fewer CV repeats
python main.py --alpha 0.0 --nb mixed --support on
python main.py --no-trace            # skip the three-data-version dump
```

What one run writes into `../results/EXP-E9_pipeline/`:

| File | Contents |
|---|---|
| `metrics.csv` | accuracy, macro-F1 and end-to-end removal per dataset × arm × classifier |
| `stages.csv` | per algorithm stage: rows/attributes before and after, what it removed |
| `versions_removals.csv` | what the whole-dataset pass removed, step by step |
| `versions/<dataset>/main.csv` | the data as loaded |
| `versions/<dataset>/path1__Alg1_then_Alg2.csv` | new data after Path 1 |
| `versions/<dataset>/path2__Alg2_then_Alg1.csv` | new data after Path 2 |
| `versions/<dataset>/removals.txt` | removed row ids, removed attribute names, per-class rates |

Every row of a version file keeps its original row id, so the removal lists can be
checked against the files.

## Rules (small, non-negotiable)

1. Every experiment = one run that writes its outputs to `../results/EXP-<id>_<name>/` and prints nothing else needed by hand.
2. Fixed seeds in config at the top of each script; seeds are part of the experiment config.
3. Pure functions; models keep the sklearn API (`fit`/`predict`) so the CV plumbing stays generic.
4. Unit-test Alg. 1 + Alg. 2 on the paper's 14-instance Play-Tennis table before touching UCI data (the paper's own Tables 2–4 give you the numbers).
5. `git init` here; commit before and after every experiment run.
