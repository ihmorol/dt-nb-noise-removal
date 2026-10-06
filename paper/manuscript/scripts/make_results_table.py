"""Rebuild all manuscript tables and the exploratory audit from saved E9 data.

Run from the repository root: python paper/manuscript/scripts/make_results_table.py
No models are rerun. Checks fail on incomplete or inconsistent saved results.
"""
import json
import hashlib
import sys

import scipy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE.parents[1] / 'results' / 'EXP-E9_farid'
TABLES = HERE / 'tables'
ARMS = ['baseline->NB', 'baseline->DT', 'Alg1->NB', 'Alg1->DT',
        'Alg2->NB', 'Alg2->DT', 'C1 N->A->NB', 'C1 N->A->DT',
        'C2 A->N->NB', 'C2 A->N->DT', 'parallel->NB', 'parallel->DT']
LABEL = dict(zip(ARMS, ['NB', 'C4.5', 'N--NB', 'N--DT', 'A--NB', 'A--DT',
                       'NA--NB', 'NA--DT', 'AN--NB', 'AN--DT', 'P--NB', 'P--DT']))
CONTRASTS = [('N--DT vs DT', 'Alg1->DT', 'baseline->DT'),
             ('A--NB vs NB', 'Alg2->NB', 'baseline->NB'),
             ('NA--NB vs NB', 'C1 N->A->NB', 'baseline->NB'),
             ('AN--NB vs NB', 'C2 A->N->NB', 'baseline->NB'),
             ('NA--DT vs DT', 'C1 N->A->DT', 'baseline->DT'),
             ('AN--DT vs DT', 'C2 A->N->DT', 'baseline->DT'),
             ('AN--NB vs NA--NB', 'C2 A->N->NB', 'C1 N->A->NB'),
             ('AN--DT vs NA--DT', 'C2 A->N->DT', 'C1 N->A->DT'),
             ('NA--NB vs P--NB', 'C1 N->A->NB', 'parallel->NB'),
             ('AN--NB vs P--NB', 'C2 A->N->NB', 'parallel->NB'),
             ('NA--DT vs P--DT', 'C1 N->A->DT', 'parallel->DT'),
             ('AN--DT vs P--DT', 'C2 A->N->DT', 'parallel->DT')]


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted


def write_table(name, columns, heading, rows, header):
    TABLES.mkdir(exist_ok=True)
    slash = chr(92)
    text = header + slash + 'begin{tabular}{' + columns + '}\n' + slash + 'toprule\n'
    text += heading + ' ' + slash*2 + '\n' + slash + 'midrule\n'
    text += '\n'.join(' & '.join(row) + ' ' + slash*2 for row in rows)
    text += '\n' + slash + 'bottomrule\n' + slash + 'end{tabular}\n'
    (TABLES / name).write_text(text, encoding='utf-8')
    print('[saved]', name)


def main():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    metrics = pd.read_csv(RESULTS / 'metrics.csv')
    folds = pd.read_csv(RESULTS / 'per_fold.csv')
    removal = pd.read_csv(RESULTS / 'removals.csv')
    config = json.loads((RESULTS / 'config.json').read_text())
    b = metrics[metrics.protocol == 'B refit-per-fold']
    bf = folds[folds.protocol == 'B refit-per-fold']
    datasets = sorted(b.dataset.unique())
    assert len(datasets) == 10 and set(b.arm) == set(ARMS)
    assert not metrics.duplicated(['dataset', 'protocol', 'arm']).any()
    assert not bf.duplicated(['dataset', 'seed', 'fold', 'arm']).any()
    assert set(bf.seed) == set(config['seeds'])
    assert set(bf.fold) == set(range(config['n_splits']))
    assert (b.n_folds == 100).all()
    assert (bf.groupby(['dataset', 'arm']).size() == 100).all()
    assert len(bf) == len(datasets) * len(ARMS) * 100
    rebuilt = bf.groupby(['dataset', 'arm'])[['accuracy', 'macro_f1']].mean()
    stored = b.set_index(['dataset', 'arm'])[['accuracy', 'macro_f1']].sort_index()
    np.testing.assert_allclose(rebuilt.sort_index(), stored, atol=1e-9, rtol=0)
    af = folds[folds.protocol == 'A before-CV']
    a = metrics[metrics.protocol == 'A before-CV']
    assert len(af) == 10000 and len(a) == 100 and (a.n_folds == 100).all()
    np.testing.assert_allclose(
        af.groupby(['dataset', 'arm'])[['accuracy', 'macro_f1']].mean().sort_index(),
        a.set_index(['dataset', 'arm'])[['accuracy', 'macro_f1']].sort_index(),
        atol=1e-9, rtol=0)
    header = '% Generated from EXP-E9_farid; 10 datasets, 10 seeds, 10 folds.\n'
    means = b.groupby('arm')[['accuracy', 'macro_f1']].mean()
    # Variation of the equal-dataset mean over ten split seeds, not a standard error.
    seedmeans = bf.groupby(['seed', 'dataset', 'arm'])[['accuracy', 'macro_f1']].mean()
    seed_sd = seedmeans.groupby(['seed', 'arm']).mean().groupby('arm').std(ddof=1)
    rows = []
    for arm in ARMS:
        cells = [f'{means.loc[arm,m]:.2f} $\\pm$ {seed_sd.loc[arm,m]:.2f}'
                 for m in ['accuracy', 'macro_f1']]
        if arm == 'baseline->DT':
            cells = ['\\textbf{' + c + '}' for c in cells]
        rows.append([LABEL[arm]] + cells)
    write_table('table_results.tex', 'lrr', 'Method & Accuracy (\\%) & Macro-F1 (\\%)', rows, header)
    for metric, name in [('accuracy', 'table_results_datasets.tex'),
                         ('macro_f1', 'table_f1_datasets.tex')]:
        pivot = b.pivot(index='dataset', columns='arm', values=metric)
        # Separate panels avoid unreadably scaling a 13-column table.
        panels = []
        for final in ['NB', 'DT']:
            arms = [arm for arm in ARMS if arm.endswith('->' + final)]
            panel_rows = [[ds] + [f'{pivot.loc[ds,arm]:.2f}' for arm in arms]
                          for ds in datasets]
            panel_name = name.replace('.tex', '_' + final + '.tex')
            write_table(panel_name, 'l' + 'r'*len(arms),
                        'Dataset & ' + ' & '.join(LABEL[arm] for arm in arms),
                        panel_rows, header)
            panels.append('\\input{tables/' + panel_name[:-4] + '}')
        (TABLES / name).write_text(header + '\n\\par\\medskip\n'.join(panels) + '\n', encoding='utf-8')
    amean = a.groupby('arm').accuracy.mean()
    write_table('table_results_A.tex', 'lrrr', 'Method & B (\\%) & A (\\%) & A$-$B (pp)',
                [[LABEL[arm], f'{means.loc[arm,"accuracy"]:.2f}', f'{amean[arm]:.2f}',
                  f'{amean[arm]-means.loc[arm,"accuracy"]:+.2f}']
                 for arm in ARMS if arm in amean.index], header)
    tests = []
    for label, first, second in CONTRASTS:
        for metric in ['accuracy', 'macro_f1']:
            pivot = b.pivot(index='dataset', columns='arm', values=metric)
            diff = (pivot[first] - pivot[second]).round(10)
            p = (float(wilcoxon(diff, zero_method='wilcox', alternative='two-sided',
                  method=PermutationMethod(n_resamples=np.inf)).pvalue)
                 if (diff != 0).any() else 1.0)
            tests.append(dict(comparison=label, first=first, second=second, metric=metric,
                              delta=float(diff.mean()), p=p,
                              wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                              ties=int((diff == 0).sum())))
    for test, adjusted in zip(tests, holm([t['p'] for t in tests])):
        test['p_holm'] = float(adjusted)
    stats_rows = []
    for i, (label, _, _) in enumerate(CONTRASTS):
        cells = [label]
        for test in tests[2*i:2*i+2]:
            cells += [f'{test["delta"]:+.2f}', f'{test["p"]:.3f}', f'{test["p_holm"]:.3f}']
        stats_rows.append(cells)
    write_table('table_tests.tex', 'lrrrrrr',
                'Comparison & $\\Delta$Acc & $p$ & $p_{H}$ & $\\Delta$F1 & $p$ & $p_{H}$',
                stats_rows, header)
    rb = removal[removal.protocol == 'B refit-per-fold']
    events = rb.drop_duplicates(['dataset', 'seed', 'fold', 'stage'])
    assert len(events) == 4000 and not events.skipped.any()
    rates = {}
    for stage, group in events.groupby('stage'):
        values = (100*(1-group.rows_after/group.rows_before) if stage.startswith('N')
                  else 100*group.attributes_after/group.attributes_before)
        rates[stage] = values.groupby(group.dataset).mean()
    write_table('table_removals.tex', 'lrrrr',
                'Dataset & N del. & AN del. & A kept & NA kept',
                [[ds] + [f'{rates[stage][ds]:.2f}' for stage in
                  ['N plain', 'N weighted (after A)', 'A on the raw fold', 'A after plain N']]
                 for ds in datasets], header)
    breast = rb[(rb.dataset == 'breast-cancer') & (rb.method == 'Alg1')].groupby(
        ['stage', 'class'])[['class_share', 'class_removal_rate']].mean()
    summary = dict(analysis_runtime={'python':sys.version.split()[0],
                                     'numpy':np.__version__, 'pandas':pd.__version__,
                                     'scipy':scipy.__version__},
                   source_sha256={name:hashlib.sha256((RESULTS/name).read_bytes()).hexdigest()
                                  for name in ['metrics.csv','per_fold.csv','removals.csv','config.json']},
                   coverage={'datasets':10,'B_fold_scores':len(bf),'A_fold_scores':len(af),
                            'B_cleaning_events':len(events)},
                   means=means.to_dict(orient='index'), seed_sd=seed_sd.to_dict(orient='index'),
                   tests=tests, test_family_size=len(tests),
                   diagnostic_dataset_means={k:v.to_dict() for k,v in rates.items()},
                   breast_class_rates=[dict(stage=s, label=c, **row.to_dict())
                                       for (s,c),row in breast.iterrows()],
                   complete_class_deletion_records=int((rb.class_removal_rate == 1).sum()),
                   skipped_B_events=int(events.skipped.sum()))
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Keep the editor source standalone: inline generated tables, including panels.
    import re
    manuscript = HERE / 'main.tex'
    source = manuscript.read_text(encoding='utf-8')
    def expand_table(name):
        text = (TABLES / name).read_text(encoding='utf-8')
        return re.sub(r'\\input\{tables/([^}]+)\}',
                      lambda match: expand_table(match.group(1) + '.tex'), text)
    for name in ['table_results.tex', 'table_results_datasets.tex',
                 'table_f1_datasets.tex', 'table_tests.tex',
                 'table_results_A.tex', 'table_removals.tex']:
        start = '% BEGIN GENERATED ' + name
        end = '% END GENERATED ' + name
        pattern = re.escape(start) + r'.*?' + re.escape(end)
        assert len(re.findall(pattern, source, flags=re.S)) == 1
        source = re.sub(pattern, lambda match: start + '\n' + expand_table(name) + end,
                        source, flags=re.S)
    manuscript.write_text(source, encoding='utf-8')
    print('[verified] all saved means match fold records; Holm family:', len(tests))
    print('[tests] smallest raw/adjusted p:', min(t['p'] for t in tests),
          min(t['p_holm'] for t in tests))


if __name__ == '__main__':
    main()
