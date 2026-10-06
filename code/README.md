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
├── weka_utils.py       R1: write ARFF, run Weka J48 and NaiveBayes
├── replicate_farid.py  R1: replicate Farid's Tables 8-11 with J48
├── check_pipeline.py   checks the leakage rules of pipeline.py (no Java)
├── verify_faithful.py  R1: checks to run before replicate_farid.py
├── figures.py          the paper's Figures 2 and 3 with our hybrid added
├── committee_filter.py E11: the reliable noise filter - cross-validated DT+NB committee
├── deep_mlp.py         E11: the fundamental deep model - a small PyTorch MLP
├── e11_lr_mlp_hybrid.py E11: old vs clean vs weighted vs new data, LR + MLP
├── noise_inject.py     E12: inject KNOWN label noise, keep the corruption mask
├── confident_filter.py E12: DT and NB judges, per-class confident-joint thresholds
├── e12_noise_removal.py E12: which filter finds the noise? graded against ground truth
└── lib/                Weka 3.8.6 jars (R1 needs Java)
```

## How to run

Run from inside `code/`:

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
python e11_lr_mlp_hybrid.py        E11, all datasets, 5 seeds (LR + MLP)
python e11_lr_mlp_hybrid.py --datasets iris glass --seeds 3   smoke run
python e11_lr_mlp_hybrid.py --rule majority                   looser voting
python noise_inject.py             E12: show the corruption rate per noise kind
python confident_filter.py         E12: self-check on a noised iris
python e12_noise_removal.py        E12: all datasets, 3 seeds, MLP included
python e12_noise_removal.py --quick                        one seed, no MLP
python e12_noise_removal.py --datasets iris glass --rates 0.1 0.2
python e12_noise_removal.py --handling relabel              relabel instead of delete
python figures.py
```

`main.py` writes into `../results/EXP-E9_pipeline/`:

| File | Contents |
|---|---|
| `metrics*.csv` | accuracy, macro-F1 and removal % per dataset, arm and classifier, plus the settings used |
| `stages.csv` | per algorithm step: rows/attributes before and after, averaged over the folds |
| `versions_removals.csv` | what the one pass over the whole dataset removed |
| `versions/<dataset>/` | the data before and after each path, and `removals.txt` |

`e11_lr_mlp_hybrid.py` writes into `../results/EXP-E11_lr-mlp-hybrid-data/`:

| File | Contents |
|---|---|
| `metrics.csv` | accuracy and macro-F1 per dataset, classifier, variant, seed and fold |
| `summary.csv` | mean +/- std per variant, delta vs old, Wilcoxon (seed means) and exploratory fold t-test |
| `filter_stats.csv` | what the committee deleted per fold: removal %, vote histogram, per-class rates |
| `versions/<dataset>/` | one full-data old.csv / new.csv pair for inspection (not the evaluation) |
| `config.json` | every setting of the run, incl. library versions |

`e12_noise_removal.py` writes into `../results/EXP-E12_noise_removal/`:

| File | Contents |
|---|---|
| `detection.csv` | per fold and arm: precision, recall, F1, AUROC against the injected noise, removal %, rows rescued by the per-class floor |
| `accuracy.csv` | per fold, arm and classifier: accuracy and macro-F1 on the CLEAN test fold |
| `detection_summary.csv` | the same averaged over datasets, per kind x rate x arm |
| `accuracy_summary.csv` | accuracy per arm with `delta_vs_none` — the number the paper quotes |
| `summary.md` | the printed detection and accuracy tables |
| `config.json` | every setting of the run |

E12 arms: `none` (no cleaning), `NB hard_vote`, `DT hard_vote`, `DT confident_joint`,
`NB confident_joint`, `committee consensus` (E11's filter, as the reference point).

## Rules

1. Every experiment is one script run that writes into `../results/EXP-<id>_<name>/`.
2. Seeds are fixed and written into the output files.
3. Everything (filter, attribute selection, classifier) is fitted on the training fold only.
4. Commit before and after every experiment run.
