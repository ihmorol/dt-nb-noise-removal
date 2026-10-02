"""Validate locked E12 outputs and build a reproducible research report."""
import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from data import available,load_data
from e12_noise_removal import ARMS


def validate(name):
    out = ROOT/'results'/('EXP-E12v2_'+name)
    cfg = json.loads((out/'config.json').read_text())
    assert cfg['status'] == 'complete', f'{name} not complete'
    expected = {
        'main':dict(datasets=available(),seeds=3,rates=[0,.1,.2,.4],kinds=['symmetric','pairflip'],handling='relabel',classifiers=['NB','DT','LR']),
        'delete':dict(datasets=available(),seeds=1,rates=[0,.2],kinds=['symmetric','pairflip'],handling='delete',classifiers=['NB','DT','LR']),
        'mlp':dict(datasets=['iris','diabetes','vote'],seeds=1,rates=[0,.2],kinds=['symmetric','pairflip'],handling='relabel',classifiers=['MLP'])}[name]
    expected['min_per_class'] = 5
    for key,value in expected.items():
        assert cfg[key] == value,(name,key,cfg[key],value)
    det = pd.read_csv(out/'detection.csv')
    acc = pd.read_csv(out/'accuracy.csv')
    keys = ['dataset','seed','fold','kind','rate','arm','handling']
    assert cfg['detection_rows'] == len(det) and cfg['accuracy_rows'] == len(acc)
    for frame in [det,acc]:
        assert set(frame.dataset) == set(cfg['datasets'])
        assert set(frame.seed) == set(range(cfg['seeds']))
        assert set(frame.fold) == set(range(10))
        assert set(frame.kind) == set(cfg['kinds'])
        assert set(frame.rate) == set(cfg['rates'])
        assert set(frame.handling) == {cfg['handling']}
    assert set(acc.arm) == {a for a,_,_ in ARMS}
    assert set(det.arm) == {a for a,_,_ in ARMS if a != 'reference labels'}
    assert set(acc.classifier) == set(cfg['classifiers'])
    assert not det.duplicated(keys).any()
    assert not acc.duplicated(keys+['classifier']).any()
    conditions = len(cfg['datasets'])*cfg['seeds']*10*len(cfg['kinds'])*len(cfg['rates'])
    assert len(det) == conditions*7, (name,len(det),conditions*7)
    assert len(acc) == conditions*8*len(cfg['classifiers'])
    assert np.isfinite(acc[['accuracy','macro_f1']]).all().all()
    for dataset,block in acc.groupby('dataset'):
        assert len(block) == cfg['seeds']*10*len(cfg['kinds'])*len(cfg['rates'])*8*len(cfg['classifiers'])
    for file,digest in cfg['code_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/file).read_bytes()).hexdigest() == digest, file
    for dataset in cfg['datasets']:
        X,y,_ = load_data(dataset)
        actual = dict(rows=len(y),features=X.shape[1],classes=len(np.unique(y)),
            sha256=hashlib.sha256(X.tobytes()+'|'.join(y).encode()).hexdigest())
        assert cfg['datasets_loaded'][dataset] == actual,(name,dataset)
    if cfg['handling'] == 'relabel':
        # Outer training sizes vary by at most one; every treatment agrees with none.
        none = det[det.arm == 'none'][keys[:-2]+['handling','rows_after']]
        check = det.merge(none,on=['dataset','seed','fold','kind','rate','handling'],suffixes=('','_none'))
        assert (check.rows_after == check.rows_after_none).all()
    else:
        none = det[det.arm == 'none'][['dataset','seed','fold','kind','rate','handling','rows_after']]
        check = det.merge(none,on=['dataset','seed','fold','kind','rate','handling'],suffixes=('','_none'))
        assert (check.rows_after == check.rows_after_none-check.tp-check.fp).all()
    return cfg


def main():
    configs = {name:validate(name) for name in ['main','delete','mlp']}
    out = ROOT/'results/EXP-E12v2_main'
    means = pd.read_csv(out/'accuracy_by_dataset.csv')
    primary = pd.read_csv(out/'primary_by_dataset.csv')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'savefig.dpi':300})
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    values = primary.delta_macro_f1.to_numpy()
    axes[0].barh(primary.dataset,values,color=['#0072B2' if v>=0 else '#D55E00' for v in values])
    axes[0].axvline(0,color='#555',lw=.8)
    axes[0].set_xlabel('Macro-F1 change vs no cleaning (points)')
    axes[0].set_title('Dual agreement, 20% added noise')
    axes[0].invert_yaxis()
    for arm,color,marker in [('dual agreement','#0072B2','o'),('NB threshold','#009E73','s'),
            ('committee consensus','#CC79A7','^'),('DT hard_vote','#D55E00','D')]:
        block = means[means.arm == arm].groupby('rate').delta_macro_f1.mean()
        axes[1].plot(100*block.index,block.values,label=arm,color=color,marker=marker)
    axes[1].axhline(0,color='#555',lw=.8)
    axes[1].set_xlabel('Added training-label noise (%)')
    axes[1].set_ylabel('Macro-F1 change vs no cleaning (points)')
    axes[1].set_title('Equal means across datasets, models and types')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(ROOT/'to_human/e12-results.png',dpi=300)
    fig.savefig(ROOT/'to_human/e12-results.pdf')
    summary = (out/'summary.md').read_text()
    table = means[(means.arm == 'dual agreement') & (means.rate.isin([0,.2]))]
    table = table.groupby(['dataset','rate'])[['delta_accuracy','delta_macro_f1']].mean().round(3)
    p_value = float(wilcoxon(primary.delta_macro_f1).pvalue)
    endpoint = f'<p><strong>Primary result: {primary.delta_macro_f1.mean():+.2f} macro-F1 points; p={p_value:.3f}. This does not establish overall improvement at the locked 0.05 threshold.</strong></p>'
    notes = '''<p>Corrections preserve rows. Only known injected corruptions are detection ground truth;
original labels are unverified references. No novelty claim or guarantee of real-world noise improvement.
The primary is fixed; other comparisons and MLP transfer are descriptive. All ten datasets retained.
No-cleaning and reference-label training are separate controls. Singleton protection may reduce recall.</p>'''
    body = '<!doctype html><html><meta charset="utf-8"><title>DT–NB noise research</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17212b}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;font-size:13px}</style><h1>Can NB and DT improve learning by correcting noise?</h1>'
    body += endpoint+notes+'<img src="e12-results.png" alt="Dataset gains and losses; effects by noise rate"><h2>Dataset outcomes</h2>'+table.to_html()
    det = pd.read_csv(out/'detection_by_dataset.csv')
    noise = det[det.arm == 'dual agreement'].groupby('rate')[['true_noise_pct','post_noise_pct','precision','recall','false_positive_rate']].mean().round(3)
    body += '<h2>Injected corruption and residual reference-label errors</h2>'+noise.to_html()
    body += '<p>Zero-added-noise correction changes roughly 5.44% of reference labels. Original labels are unverified; this is a reference-label disagreement rate, not proof of genuine new noise. At zero added noise, Glass, image segmentation, soybean and tic-tac-toe lose macro-F1; Glass and tic-tac-toe have the largest losses. Mean gains are concentrated in the DT classifier; LR/NB are mixed.</p>'
    body += '<h2>Protocol and completeness</h2><p>All three locked runs passed row-count, duplicate, finite-score and code/data-hash checks. Both relabel runs retained every training row; deletion passed expected row-removal checks.</p>'
    body += '<pre>'+html.escape(json.dumps({k:{f:c[f] for f in ['datasets','seeds','rates','classifiers','handling','minutes']} for k,c in configs.items()},indent=2))+'</pre>'
    for run,title in [('delete','Deletion sensitivity: descriptive'),('mlp','MLP transfer: exploratory, three datasets and one seed')]:
        secondary = pd.read_csv(ROOT/'results'/('EXP-E12v2_'+run)/'accuracy_by_dataset.csv')
        secondary = secondary[secondary.arm == 'dual agreement'].groupby(['dataset','rate'])[['delta_accuracy','delta_macro_f1']].mean().round(3)
        body += '<h2>'+title+'</h2>'+secondary.to_html()
    body += '<p>MLP transfer is heterogeneous: gains on Iris/Vote, loss on Diabetes; zero-added-noise mean is negative. It does not establish broad improvement or harmlessness.</p>'
    body += '<h2>Full main results</h2><pre>'+html.escape(summary)+'</pre></html>'
    (ROOT/'to_human/e12-report.html').write_text(body,encoding='utf-8')
    print('All three runs validated; report and PDF/PNG generated.')


if __name__ == '__main__':
    main()
