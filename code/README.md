# Research code

## Core implementations

- `data.py`: the ten datasets, original columns or one-hot view.
- `nb.py`: plain/attribute-weighted mixed or Gaussian NB.
- `algorithm1.py`, `algorithm2.py`, `pipeline.py`: Farid stages and fold-safe paths.
- `faithful_nb.py`, `weka_utils.py`: original-column NB/J48 replication.
- `main.py`, `leakage_check.py`: sklearn path and leakage audit.
- `replicate_farid.py`, `e9_farid.py`, `e5_strategy.py`: replication, combined
  orders and graded handling in the J48 world.
- `committee_filter.py`, `e11_lr_mlp_hybrid.py`, `deep_mlp.py`: inherited E11.
- `noise_inject.py`, `confident_filter.py`, `e12_noise_removal.py`: E12v2.

The sklearn/CART and Weka/J48 experiments use different implementations and
attribute spaces. Do not mix their scores or claim exact replication.

## Checks

Run from this directory:

```sh
python data.py
python check_pipeline.py
python check_e12.py
python verify_faithful.py
```

The first three need no Java. Weka checks/runs need Java and the jars in `lib/`.
Python dependencies are in `../requirements.txt`; pandas must stay below 3.
Optional MLP runs use the already-used PyTorch runtime.

## E12v2

Choose a fresh output directory; existing outputs are never overwritten.

```sh
python e12_noise_removal.py --out ../results/EXP-E12v2_new
python e12_noise_removal.py --seeds 1 --rates 0 .2 --handling delete --out ../results/EXP-E12v2_delete_new
python e12_noise_removal.py --datasets iris diabetes vote --seeds 1 --rates 0 .2 --classifiers MLP --out ../results/EXP-E12v2_mlp_new
```

Defaults: three seeds, ten outer folds, symmetric/pairflip corruption at
0/.1/.2/.4, correction, NB/DT/LR finals. NB and DT judges use out-of-fold
probabilities. Dual correction requires the same confident alternative from
both, protects small classes and retains every row. No corruption mask or
reference training label reaches a judge.

The completed main run took 146 minutes on this host. Original labels are
unverified references. The threshold arm is a heuristic, not full calibrated
Confident Learning. Inherited E12 pilot results are excluded due to bugs.

Outputs include per-fold metrics/confusion matrices, pooled dataset-seed
metrics, summaries, source/data fingerprints and configuration. The locked
protocol is `../notes/e12v2-protocol.md`. To validate the committed runs and
regenerate the report, run `python ../figures/gen_fig_e12.py`.

## Earlier runs

```sh
python main.py
python leakage_check.py
python replicate_farid.py
python e9_farid.py
python e5_strategy.py
python e11_lr_mlp_hybrid.py --help
```

Everything used for honest evaluation is fitted inside training folds. Saved
full-data cleaned versions are inspection artifacts, not evaluation data.
