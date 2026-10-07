"""Generate the plain-language full-results tables and figure for main_full.tex.

Same source data and the same statistical conventions as make_results_table.py
(EXP-E9_farid, 12 arms, exact-permutation Wilcoxon, joint Holm family of 24),
but every method name is written out in words -- no letter codes -- and the
before-CV protocol is called what it is. Also renders the accuracy-change
figure with all twelve settings. main.tex is not touched.

Run from the repository root: python paper/manuscript/scripts/make_full_tables.py
No models are rerun. Checks fail on incomplete or inconsistent saved results.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE.parents[1] / 'results' / 'EXP-E9_farid'
TABLES = HERE / 'tables'
FIGURES = HERE / 'figures'
ARMS = ['baseline->NB', 'baseline->DT', 'Alg1->NB', 'Alg1->DT',
        'Alg2->NB', 'Alg2->DT', 'C1 N->A->NB', 'C1 N->A->DT',
        'C2 A->N->NB', 'C2 A->N->DT', 'parallel->NB', 'parallel->DT']
# Full plain names used in the summary and protocols tables.
LABEL = {
    'baseline->NB': 'Plain NB (no cleaning)',
    'baseline->DT': 'Plain C4.5 (no cleaning)',
    'Alg1->NB': 'Remove instances only $\\to$ NB',
    'Alg1->DT': 'Remove instances only $\\to$ C4.5',
    'Alg2->NB': 'Weight attributes only $\\to$ NB',
    'Alg2->DT': 'Select attributes only $\\to$ C4.5',
    'C1 N->A->NB': 'Instances first, then attributes $\\to$ NB',
    'C1 N->A->DT': 'Instances first, then attributes $\\to$ C4.5',
    'C2 A->N->NB': 'Attributes first, then instances $\\to$ NB',
    'C2 A->N->DT': 'Attributes first, then instances $\\to$ C4.5',
    'parallel->NB': 'Both steps separately, combined $\\to$ NB',
    'parallel->DT': 'Both steps separately, combined $\\to$ C4.5',
}
# Compact plain names for the wide per-dataset panels (explained in the caption).
SHORT = {
    'baseline->NB': 'No cleaning', 'baseline->DT': 'No cleaning',
    'Alg1->NB': 'Instances only', 'Alg1->DT': 'Instances only',
    'Alg2->NB': 'Attributes only', 'Alg2->DT': 'Attributes only',
    'C1 N->A->NB': 'Instances first', 'C1 N->A->DT': 'Instances first',
    'C2 A->N->NB': 'Attributes first', 'C2 A->N->DT': 'Attributes first',
    'parallel->NB': 'Both separately', 'parallel->DT': 'Both separately',
}
# The complete paired dataset-level comparison family for the full version:
# every cleaning setting against its plain classifier, the two order swaps,
# and the four sequential-versus-parallel contrasts (accuracy and macro-F1 each).
CONTRASTS = [
    ('Remove instances only vs plain C4.5', 'Alg1->DT', 'baseline->DT'),
    ('Select attributes only vs plain C4.5', 'Alg2->DT', 'baseline->DT'),
    ('Instances first vs plain C4.5', 'C1 N->A->DT', 'baseline->DT'),
    ('Attributes first vs plain C4.5', 'C2 A->N->DT', 'baseline->DT'),
    ('Both separately vs plain C4.5', 'parallel->DT', 'baseline->DT'),
    ('Attributes first vs instances first (C4.5)', 'C2 A->N->DT', 'C1 N->A->DT'),
    ('Instances first vs both separately (C4.5)', 'C1 N->A->DT', 'parallel->DT'),
    ('Attributes first vs both separately (C4.5)', 'C2 A->N->DT', 'parallel->DT'),
    ('Remove instances only vs plain NB', 'Alg1->NB', 'baseline->NB'),
    ('Weight attributes only vs plain NB', 'Alg2->NB', 'baseline->NB'),
    ('Instances first vs plain NB', 'C1 N->A->NB', 'baseline->NB'),
    ('Attributes first vs plain NB', 'C2 A->N->NB', 'baseline->NB'),
    ('Both separately vs plain NB', 'parallel->NB', 'baseline->NB'),
    ('Attributes first vs instances first (NB)', 'C2 A->N->NB', 'C1 N->A->NB'),
    ('Instances first vs both separately (NB)', 'C1 N->A->NB', 'parallel->NB'),
    ('Attributes first vs both separately (NB)', 'C2 A->N->NB', 'parallel->NB'),
]


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
    header = '% Generated from EXP-E9_farid; 10 datasets, 10 seeds, 10 folds; plain-language labels.\n'
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
    write_table('table_full_summary.tex', 'lrr',
                'Method & Accuracy (\\%) & Macro-F1 (\\%)', rows, header)
    for metric, name in [('accuracy', 'table_full_datasets_acc.tex'),
                         ('macro_f1', 'table_full_datasets_f1.tex')]:
        pivot = b.pivot(index='dataset', columns='arm', values=metric)
        for final in ['NB', 'DT']:
            arms = [arm for arm in ARMS if arm.endswith('->' + final)]
            panel_rows = [[ds] + [f'{pivot.loc[ds,arm]:.2f}' for arm in arms]
                          for ds in datasets]
            panel_name = name.replace('.tex', '_' + final + '.tex')
            write_table(panel_name, 'l' + 'r'*len(arms),
                        'Dataset & ' + ' & '.join(SHORT[arm] for arm in arms),
                        panel_rows, header)
    amean = a.groupby('arm').accuracy.mean()
    write_table('table_full_protocols.tex', 'lrrrr',
                'Design & Final & Inside folds (\\%) & Before CV (\\%) & Difference (pp)',
                [[SHORT[arm], 'NB' if arm.endswith('NB') else 'C4.5',
                  f'{means.loc[arm,"accuracy"]:.2f}', f'{amean[arm]:.2f}',
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
    write_table('table_full_tests.tex', 'lrrrrrr',
                'Comparison & Acc.\\ diff.\\ (pp) & $p$ & Holm $p$ & F1 diff.\\ (pp) & $p$ & Holm $p$',
                stats_rows, header)
    rb = removal[removal.protocol == 'B refit-per-fold']
    events = rb.drop_duplicates(['dataset', 'seed', 'fold', 'stage'])
    assert len(events) == 4000 and not events.skipped.any()
    rates = {}
    for stage, group in events.groupby('stage'):
        values = (100*(1-group.rows_after/group.rows_before) if stage.startswith('N')
                  else 100*group.attributes_after/group.attributes_before)
        rates[stage] = values.groupby(group.dataset).mean()
    write_table('table_full_removals.tex', 'lrrrr',
                'Dataset & \\makecell{Deleted,\\\\plain filter} & '
                '\\makecell{Deleted,\\\\weighted filter} & '
                '\\makecell{Kept,\\\\raw fold} & '
                '\\makecell{Kept,\\\\after deletion}',
                [[ds] + [f'{rates[stage][ds]:.2f}' for stage in
                  ['N plain', 'N weighted (after A)', 'A on the raw fold', 'A after plain N']]
                 for ds in datasets], header)

    # Figure: accuracy change from the corresponding plain classifier, all twelve settings.
    FIGURES.mkdir(exist_ok=True)
    pivot = b.pivot(index='dataset', columns='arm', values='accuracy')
    designs = [('Alg1', 'Remove instances\nonly'),
               ('Alg2', 'Weight attributes\nonly'),
               ('C1 N->A', 'Instances\nfirst'),
               ('C2 A->N', 'Attributes\nfirst'),
               ('parallel', 'Both separately,\ncombined')]
    y = np.arange(len(designs))
    height = 0.36
    figure, axes = plt.subplots(figsize=(4.6, 2.9))
    for offset, final, base, face, hatch in [
            (height/2, 'NB', 'baseline->NB', 'white', '///'),
            (-height/2, 'DT', 'baseline->DT', '0.35', '')]:
        values = [(pivot[arm + '->' + final] - pivot[base]).mean() for arm, _ in designs]
        bars = axes.barh(y + offset, values, height=height, facecolor=face,
                         hatch=hatch, edgecolor='black', linewidth=0.6)
        for bar, value in zip(bars, values):
            # Long bars carry their label inside (white on the dark bars);
            # short bars label outside their far end so nothing is clipped.
            if abs(value) >= 1.0:
                axes.text(value + 0.08, bar.get_y() + bar.get_height()/2,
                          f'{value:+.2f}', va='center', ha='left',
                          color='white' if not hatch else 'black', fontsize=7)
            else:
                x, ha = (value + 0.08, 'left') if value >= 0 else (value - 0.08, 'right')
                axes.text(x, bar.get_y() + bar.get_height()/2,
                          f'{value:+.2f}', va='center', ha=ha, color='black', fontsize=7)
    axes.axvline(0, color='black', linewidth=0.8)
    axes.set_xlim(-5.9, 2.4)
    axes.set_yticks(y)
    axes.set_yticklabels([name for _, name in designs], fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel('Accuracy change from the plain classifier (pp)', fontsize=8)
    axes.tick_params(axis='x', labelsize=8)
    axes.legend(['Final NB', 'Final C4.5'], fontsize=8, loc='upper right', frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / 'accuracy_change_full.pdf')
    print('[saved] figures/accuracy_change_full.pdf')

    print('[verified] all stored means match fold records; Holm family:', len(tests))
    print('[tests] smallest raw/adjusted p:', min(t['p'] for t in tests),
          min(t['p_holm'] for t in tests))


if __name__ == '__main__':
    main()
