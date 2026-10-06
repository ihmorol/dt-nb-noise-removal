"""E12v2: paired noise detection and downstream learning.

Reference labels are not verified clean labels. Corruption is injected only
inside training folds. Filters see neither masks nor original labels. Each
output directory is exclusive; completed pilots are never overwritten.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from confident_filter import confident_filter
from data import available, load_data
from nb import WeightedNB
from noise_inject import inject

ARMS = [
    ('none', None, None),
    ('NB hard_vote', 'NB', 'hard_vote'),
    ('DT hard_vote', 'DT', 'hard_vote'),
    ('NB threshold', 'NB', 'confident_joint'),
    ('DT threshold', 'DT', 'confident_joint'),
    ('committee consensus', 'committee', 'consensus'),
    ('dual agreement', 'committee', 'dual_agreement'),
    ('reference labels', None, None),
]


def make_classifier(name, seed):
    if name == 'NB':
        return WeightedNB(likelihood='mixed')
    if name == 'DT':
        return DecisionTreeClassifier(criterion='entropy', random_state=seed)
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == 'MLP':
        from deep_mlp import DeepMLP
        return make_pipeline(StandardScaler(), DeepMLP(seed=seed))
    raise ValueError(name)


def apply_handling(X, y, keep, info, handling):
    """Use exactly the flag mask; correction keeps every row."""
    if handling == 'delete':
        return X[keep], y[keep]
    if handling != 'relabel':
        raise ValueError(handling)
    labels = y.copy()
    labels[~keep] = np.asarray(info['predicted'])[~keep]
    return X, labels


def detect_scores(keep, info, is_noisy):
    flagged = ~keep
    tp = int((flagged & is_noisy).sum())
    fp = int((flagged & ~is_noisy).sum())
    fn = int((~flagged & is_noisy).sum())
    tn = int((~flagged & ~is_noisy).sum())
    auroc = float('nan')
    if info is not None and is_noisy.any() and (~is_noisy).any():
        auroc = float(roc_auc_score(is_noisy, info['suspicion']))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else float('nan'),
                recall=tp/(tp+fn) if tp+fn else float('nan'),
                f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else float('nan'),
                auroc=auroc, flagged_pct=100*flagged.mean(),
                false_positive_rate=fp/(fp+tn) if fp+tn else float('nan'),
                true_noise_pct=100*is_noisy.mean())


def run_fold(X, y_reference, train, test, kind, rate, seed, fold_seed,
             handling, min_per_class, classifiers):
    X_train, y_train = X[train], y_reference[train]
    noisy, mask = inject(X_train, y_train, rate, kind=kind, seed=fold_seed)
    labels = np.unique(y_reference)
    detection, accuracy, cache = [], [], {}
    for arm, judge, method in ARMS:
        info = None
        keep = np.ones(len(noisy), dtype=bool)
        fit_X, fit_y = X_train, noisy
        if arm == 'reference labels':
            fit_y = y_train
        elif method:
            keep, info = confident_filter(X_train, noisy, judge=judge, method=method,
                repeats=1, n_splits=5, min_per_class=min_per_class,
                seed=fold_seed, cache=cache)
            fit_X, fit_y = apply_handling(X_train, noisy, keep, info, handling)
        row = dict(arm=arm, kind=kind, rate=rate, handling=handling, fold_seed=fold_seed)
        if arm != 'reference labels':
            det = detect_scores(keep, info, mask)
            if handling == 'relabel' and method:
                post_errors = int((fit_y != y_train).sum())
                corrected = int((mask & (fit_y == y_train)).sum())
                newly_wrong = int((~mask & (fit_y != y_train)).sum())
            else:
                post_errors = int(mask[keep].sum())
                corrected, newly_wrong = 0, 0
            detection.append(dict(**row, **det, post_noise_pct=100*post_errors/len(fit_y),
                corrected=corrected, newly_wrong=newly_wrong, rows_after=len(fit_y),
                rescued_by_floor=info['rescued_by_floor'] if info else 0))
        for name in classifiers:
            model = make_classifier(name, seed).fit(fit_X, fit_y)
            prediction = model.predict(X[test])
            assert len(prediction) == len(test)
            accuracy.append(dict(**row, classifier=name,
                confusion=json.dumps(confusion_matrix(y_reference[test],prediction,labels=labels).tolist()),
                accuracy=100*np.mean(prediction == y_reference[test]),
                macro_f1=100*f1_score(y_reference[test], prediction, labels=labels,
                                     average='macro', zero_division=0)))
    return detection, accuracy


def summarize(detection, accuracy, out):
    # Equal weighting across datasets, not across rows or dataset sizes.
    det_groups = ['dataset', 'kind', 'rate', 'handling', 'arm']
    det_rows = []
    for key, block in detection.groupby(det_groups+['seed']):
        counts = block[['tp','fp','fn','tn']].sum()
        tp,fp,fn,tn = (int(counts[k]) for k in ['tp','fp','fn','tn'])
        values = block.mean(numeric_only=True).to_dict()
        values.update(tp=tp,fp=fp,fn=fn,tn=tn,
            precision=tp/(tp+fp) if tp+fp else float('nan'),
            recall=tp/(tp+fn) if tp+fn else float('nan'),
            f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else float('nan'),
            false_positive_rate=fp/(fp+tn) if fp+tn else float('nan'))
        det_rows.append(dict(zip(det_groups+['seed'],key),**values))
    det_seed = pd.DataFrame(det_rows)
    det_seed.to_csv(out/'detection_by_seed.csv',index=False)
    det = det_seed.groupby(det_groups).mean(numeric_only=True).reset_index()
    det.to_csv(out/'detection_by_dataset.csv', index=False)
    det.groupby(['kind','rate','handling','arm']).mean(numeric_only=True).to_csv(out/'detection_summary.csv')
    groups = ['dataset','kind','rate','handling','arm','classifier']
    seed_rows = []
    for key, block in accuracy.groupby(groups+['seed']):
        matrix = np.sum([np.asarray(json.loads(v)) for v in block.confusion], axis=0)
        tp = np.diag(matrix)
        denominator = matrix.sum(axis=0)+matrix.sum(axis=1)
        f1 = np.divide(2*tp,denominator,out=np.zeros(len(tp),float),where=denominator>0)
        seed_rows.append(dict(zip(groups+['seed'],key), accuracy=100*tp.sum()/matrix.sum(), macro_f1=100*f1.mean()))
    pooled = pd.DataFrame(seed_rows)
    pooled.to_csv(out/'accuracy_by_seed.csv',index=False)
    means = pooled.groupby(groups)[['accuracy','macro_f1']].mean().reset_index()
    base = means[means.arm == 'none'].drop(columns='arm')
    means = means.merge(base, on=['dataset','kind','rate','handling','classifier'], suffixes=('', '_none'), validate='many_to_one')
    for metric in ['accuracy','macro_f1']:
        means['delta_'+metric] = means[metric]-means[metric+'_none']
    means.to_csv(out/'accuracy_by_dataset.csv', index=False)
    summary = means.groupby(['kind','rate','handling','arm','classifier'])[['accuracy','macro_f1','delta_accuracy','delta_macro_f1']].mean().reset_index()
    summary.to_csv(out/'accuracy_summary.csv', index=False)
    # Predefined primary endpoint: average 20% noise types and classifiers,
    # one paired value per dataset, dual relabel vs no cleaning.
    primary = means[(means.arm == 'dual agreement') & (means.rate == .2)]
    primary = primary.groupby('dataset')[['delta_accuracy','delta_macro_f1']].mean()
    primary.to_csv(out/'primary_by_dataset.csv')
    text = ['# E12v2 results', '', 'Reference labels are unverified; detection is against injected corruption only.',
            'No novelty claim. Full thresholded Confident Learning is not implemented.',
            'Primary metrics pool confusion matrices across outer folds per seed, using all dataset classes.', '',
            '## Primary: dual agreement vs no cleaning at 20% added noise', '', primary.round(3).to_string(), '']
    eligible = (set(accuracy.handling) == {'relabel'}
        and set(accuracy.kind) == {'symmetric','pairflip'}
        and set(accuracy.classifier) == {'NB','DT','LR'}
        and set(accuracy.dataset) == set(available())
        and set(accuracy.seed) == {0,1,2}
        and set(accuracy.rate) == {0,.1,.2,.4})
    if eligible and len(primary) >= 5:
        delta = primary.delta_macro_f1.to_numpy()
        p = 1.0 if np.all(delta == 0) else float(wilcoxon(delta).pvalue)
        text += [f'Dataset-level mean macro-F1 delta: {delta.mean():.3f} points; two-sided Wilcoxon p={p:.6g}.',
                 'Accuracy and all other arm comparisons are descriptive secondary results.']
    else:
        text += ['Descriptive run only: this configuration does not satisfy the locked main protocol.']
    text += ['', '## All downstream means', '', summary.round(3).to_string(index=False)]
    (out/'summary.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=available())
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--rates', type=float, nargs='+', default=[0, .1, .2, .4])
    parser.add_argument('--kinds', nargs='+', choices=['symmetric','pairflip','asymmetric'], default=['symmetric','pairflip'])
    parser.add_argument('--handling', choices=['delete','relabel'], default='relabel')
    parser.add_argument('--min-per-class', type=int, default=5)
    parser.add_argument('--classifiers', nargs='+', choices=['NB','DT','LR','MLP'], default=['NB','DT','LR'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or not args.datasets or not args.rates:
        parser.error('positive seeds and nonempty datasets/rates required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['out'] = str(out)
    config['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    config['code_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    config['versions'] = {'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__}
    config['datasets_loaded'] = {}
    config['status'] = 'running'
    (out/'config.json').write_text(json.dumps(config,indent=2))
    start = time.time()
    all_det, all_acc = [], []
    for dataset in args.datasets:
        X, y, _ = load_data(dataset)
        config['datasets_loaded'][dataset] = dict(rows=len(y),features=X.shape[1],classes=len(np.unique(y)),
            sha256=hashlib.sha256(X.tobytes()+'|'.join(y).encode()).hexdigest())
        (out/'config.json').write_text(json.dumps(config,indent=2))
        for seed in range(args.seeds):
            splitter = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
            folds = list(splitter.split(X,y))
            for kind in args.kinds:
                for rate in args.rates:
                    for fold,(train,test) in enumerate(folds):
                        det,acc = run_fold(X,y,train,test,kind,rate,seed,seed*1000+fold,
                            args.handling,args.min_per_class,args.classifiers)
                        for row in det+acc:
                            row.update(dataset=dataset,seed=seed,fold=fold)
                        all_det.extend(det)
                        all_acc.extend(acc)
                    print(f'{dataset} seed={seed} {kind} rate={rate} complete',flush=True)
            pd.DataFrame(all_det).to_csv(out/'detection.csv',index=False)
            pd.DataFrame(all_acc).to_csv(out/'accuracy.csv',index=False)
    summarize(pd.DataFrame(all_det),pd.DataFrame(all_acc),out)
    config.update(status='complete',minutes=(time.time()-start)/60,
                  detection_rows=len(all_det),accuracy_rows=len(all_acc))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(f'COMPLETE {out} in {config["minutes"]:.2f} min',flush=True)


if __name__ == '__main__':
    main()
