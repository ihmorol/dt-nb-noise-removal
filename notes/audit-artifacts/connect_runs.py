import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'notes' / 'audit-artifacts' / 'parallel-strategy'
SOURCES = {}


def read(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def scores(matrix):
    matrix = np.asarray(matrix)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * matrix.diagonal(), denominator,
                   out=np.zeros(len(matrix), dtype=float), where=denominator > 0)
    return 100 * matrix.trace() / matrix.sum(), 100 * f1.mean()


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


def e12(run):
    raw = read(f'results/{run}/accuracy.csv')
    assert set(raw.classifier) == {'NB', 'DT', 'LR'}
    keys = ['dataset', 'kind', 'rate', 'handling', 'arm', 'classifier']
    assert not raw.duplicated(keys + ['seed', 'fold']).any()
    rows = []
    for key, block in raw.groupby(keys + ['seed']):
        assert len(block) == 10
        matrices = [np.asarray(json.loads(v)) for v in block.confusion]
        acc, f1 = scores(np.sum(matrices, axis=0))
        rows.append(dict(zip(keys + ['seed'], key), accuracy=acc, macro_f1=f1))
    means = pd.DataFrame(rows).groupby(keys)[['accuracy', 'macro_f1']].mean().reset_index()
    base_keys = ['dataset', 'kind', 'rate', 'handling', 'classifier']
    base = means[means.arm == 'none'].drop(columns='arm')
    means = means.merge(base, on=base_keys, suffixes=('', '_base'), validate='many_to_one')
    for metric in ['accuracy', 'macro_f1']:
        means['delta_' + metric] = means[metric] - means[metric + '_base']
    saved = read(f'results/{run}/accuracy_by_dataset.csv')
    check = means.merge(saved, on=keys, suffixes=('_fresh', '_saved'), validate='one_to_one')
    assert len(check) == len(means) == len(saved)
    for metric in ['accuracy', 'macro_f1', 'delta_accuracy', 'delta_macro_f1']:
        assert np.allclose(check[metric + '_fresh'], check[metric + '_saved'], atol=1e-9, rtol=0)
    save(means, run + '_recomputed.csv')
    summary = means.groupby(['rate', 'arm', 'classifier'])[['delta_accuracy', 'delta_macro_f1']].mean().reset_index()
    save(summary, run + '_effects.csv')
    dual = means[means.arm == 'dual agreement'].groupby(['dataset', 'rate', 'classifier'])[['delta_accuracy', 'delta_macro_f1']].mean().reset_index()
    save(dual, run + '_dual_by_dataset.csv')
    det = read(f'results/{run}/detection_by_dataset.csv')
    save(det.groupby(['rate', 'arm'])[['precision', 'recall', 'flagged_pct', 'false_positive_rate', 'true_noise_pct', 'post_noise_pct']].mean().reset_index(), run + '_detection.csv')
    return raw, means, summary


def main():
    assert scores([[2, 0], [0, 3]]) == (100.0, 100.0)
    assert scores([[0, 2], [3, 0]]) == (0.0, 0.0)
    OUT.mkdir(exist_ok=True)
    e9 = read('results/EXP-E9_farid/metrics.csv')
    b = e9[e9.protocol == 'B refit-per-fold']
    assert len(b) == 120 and set(b.n_folds) == {100}
    save(b.groupby('arm')[['accuracy', 'macro_f1']].mean().reset_index(), 'e9_means.csv')
    e5 = read('results/EXP-E5_strategy/metrics.csv')
    save(e5.groupby('arm')[['accuracy', 'macro_f1']].mean().reset_index(), 'e5_means.csv')
    lr = read('results/EXP-E11_lr-mlp-hybrid-data/metrics.csv')
    lr = lr[lr.classifier == 'LR'].groupby(['dataset', 'variant'])[['accuracy', 'macro_f1', 'rows_kept_pct', 'cols_kept_pct']].mean().reset_index()
    lr = lr.merge(lr[lr.variant == 'old'][['dataset', 'accuracy', 'macro_f1']], on='dataset', suffixes=('', '_base'), validate='many_to_one')
    lr['delta_accuracy'] = lr.accuracy - lr.accuracy_base
    lr['delta_macro_f1'] = lr.macro_f1 - lr.macro_f1_base
    save(lr, 'e11_lr_by_dataset.csv')
    save(lr.groupby('variant')[['delta_accuracy', 'delta_macro_f1', 'rows_kept_pct', 'cols_kept_pct']].mean().reset_index(), 'e11_lr_effects.csv')
    for filename in sorted((ROOT / 'results/EXP-E9_pipeline').glob('*.csv')):
        if filename.name.startswith(('metrics_', 'sweep_')):
            frame = read(filename.relative_to(ROOT).as_posix())
            save(frame.groupby(['arm', 'final']).mean(numeric_only=True).reset_index(), 'pilot_' + filename.name)
    leak = read('results/EXP-E3a_leakage_check/leakage.csv')
    save(leak.groupby(['arm', 'final']).mean(numeric_only=True).reset_index(), 'e3a_means.csv')
    for filename in ['replication.csv', 'replication_unpruned.csv']:
        rep = read('results/EXP-R1_replication/' + filename)
        save(rep.groupby(['arm', 'protocol']).mean(numeric_only=True).reset_index(), 'r1_' + filename)
    main_raw, main_means, effects = e12('EXP-E12v2_main')
    delete_raw, _, _ = e12('EXP-E12v2_delete')
    assert len(main_raw) == 57600 and len(delete_raw) == 9600
    # Match deletion's single seed, mechanisms and rates before comparing actions.
    matched = []
    for action, raw in [('relabel', main_raw), ('delete', delete_raw)]:
        raw = raw[(raw.seed == 0) & raw.rate.isin([0, .2])]
        keys = ['dataset', 'kind', 'rate', 'arm', 'classifier']
        rows = []
        for key, block in raw.groupby(keys):
            acc, f1 = scores(np.sum([np.asarray(json.loads(v)) for v in block.confusion], axis=0))
            rows.append(dict(zip(keys, key), accuracy=acc, macro_f1=f1))
        frame = pd.DataFrame(rows)
        base = frame[frame.arm == 'none'].drop(columns='arm')
        frame = frame.merge(base, on=['dataset', 'kind', 'rate', 'classifier'], suffixes=('', '_base'), validate='many_to_one')
        frame['delta_macro_f1'] = frame.macro_f1 - frame.macro_f1_base
        frame['action'] = action
        matched.append(frame)
    matched = pd.concat(matched)
    baseline = matched[matched.arm == 'none']
    baseline_keys = ['dataset', 'kind', 'rate', 'classifier']
    paired = baseline[baseline.action == 'relabel'].merge(baseline[baseline.action == 'delete'], on=baseline_keys, suffixes=('_relabel', '_delete'), validate='one_to_one')
    assert len(paired) == 120
    assert np.allclose(paired[['accuracy_relabel', 'macro_f1_relabel']], paired[['accuracy_delete', 'macro_f1_delete']], atol=1e-9, rtol=0)
    save(matched, 'matched_seed0_actions.csv')
    save(matched.groupby(['action', 'rate', 'arm', 'classifier']).delta_macro_f1.mean().reset_index(), 'matched_seed0_action_means.csv')
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in SOURCES.items())
    (OUT / 'source_hashes.json').write_text(json.dumps(SOURCES, indent=2) + '\n', encoding='utf-8')
    primary = main_means[(main_means.rate == .2) & (main_means.arm == 'dual agreement')].groupby('dataset').delta_macro_f1.mean()
    assert np.isclose(primary.mean(), 2.520495599552765, atol=1e-6)
    print('Fresh E12 aggregation matches saved summaries; source hashes unchanged.')
    print(effects[effects.arm.isin(['dual agreement', 'NB threshold', 'committee consensus'])].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
