from pathlib import Path
import hashlib
import json
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from data import available, load_original
from pipeline import make_folds
from e9_farid import run_fold

RESULTS = ROOT / 'results/EXP-E9_farid'
files = ['config.json', 'metrics.csv', 'per_fold.csv', 'removals.csv']
before = {n: hashlib.sha256((RESULTS/n).read_bytes()).hexdigest() for n in files}
metrics = pd.read_csv(RESULTS/'metrics.csv')
folds = pd.read_csv(RESULTS/'per_fold.csv')
records = pd.read_csv(RESULTS/'removals.csv')
config = json.loads((RESULTS/'config.json').read_text())
assert config['seeds'] == list(range(10)) and config['n_splits'] == 10
assert len(metrics) == 220 and len(folds) == 22000
assert not metrics.duplicated(['dataset','protocol','arm']).any()
assert metrics.n_folds.eq(100).all()
assert folds.groupby(['dataset','protocol','arm']).size().eq(100).all()
assert np.isfinite(folds[['accuracy','macro_f1']]).all().all()
assert folds[['accuracy','macro_f1']].ge(0).all().all()
assert folds[['accuracy','macro_f1']].le(100).all().all()
means = folds.groupby(['dataset','protocol','arm'])[['accuracy','macro_f1']].mean()
stored = metrics.set_index(['dataset','protocol','arm'])[['accuracy','macro_f1']]
assert np.allclose(means.loc[stored.index],stored,atol=1e-9,rtol=0)
b = folds[folds.protocol == 'B refit-per-fold']
assert not b.duplicated(['dataset','arm','seed','fold']).any()
assert records.skipped.eq(False).all()
shapes = {}
for name in available():
    X,y,meta = load_original(name)
    assert np.isfinite(X).all()
    shapes[name] = [len(y),X.shape[1],len(set(y)),int(pd.Series(y).value_counts().min())]
print('Dataset [rows, attributes, classes, smallest class]:',shapes)
with tempfile.TemporaryDirectory() as tmp:
    for name in ['iris','contact-lenses','glass']:
        X,y,meta = load_original(name)
        train,test = make_folds(y,0)[0]
        scores,_ = run_fold(Path(tmp),X,y,meta,sorted(set(y)),train,test)
        saved = b[(b.dataset == name)&(b.seed == 0)&(b.fold == 0)].set_index('arm')
        for arm,score in scores.items():
            assert np.allclose(score,saved.loc[arm,['accuracy','macro_f1']].astype(float),atol=1e-9,rtol=0), (name,arm,score)
        print(name,'seed 0 fold 0: all 12 arms reproduce stored scores')
after = {n: hashlib.sha256((RESULTS/n).read_bytes()).hexdigest() for n in files}
assert before == after
report = {'datasets':shapes,'metrics_rows':len(metrics),'fold_rows':len(folds),'source_results_sha256':before,'fresh_fold_reproduction':['iris','contact-lenses','glass'],'full_experiment_rerun':False}
(ROOT/'notes/audit-artifacts/e9-validation.json').write_text(json.dumps(report,indent=2))
print('PASS: all A/B summaries match raw folds; no recorded safeguards; result hashes unchanged')
