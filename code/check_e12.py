"""Runnable regression guards for actual E12 treatment and metrics."""
import numpy as np
import pandas as pd
from e12_noise_removal import apply_handling, detect_scores, run_fold
from data import load_data
from nb import WeightedNB
from sklearn.model_selection import StratifiedKFold


def main():
    X = np.arange(12).reshape(6,2)
    y = np.array(['a','b','a','b','a','b'])
    keep = np.array([True,False,True,True,True,True])
    info = {'predicted': np.array(['a','a','a','b','a','b']),
            'suspicion': np.array([0,.9,0,0,0,0])}
    corrected_X, corrected_y = apply_handling(X,y,keep,info,'relabel')
    assert np.array_equal(corrected_X,X), 'correction must retain every row'
    assert corrected_y[1] == 'a' and np.array_equal(corrected_y[keep],y[keep])
    assert y[1] == 'b', 'must not mutate caller labels'
    deleted_X,deleted_y = apply_handling(X,y,keep,info,'delete')
    assert len(deleted_y) == 5 and np.array_equal(deleted_X,X[keep])
    scores = detect_scores(keep,info,~keep)
    assert scores['f1'] == 1 and scores['auroc'] == 1
    clean = detect_scores(np.ones(6,bool),info,np.zeros(6,bool))
    assert np.isnan(clean['recall']) and np.isnan(clean['f1'])
    assert clean['false_positive_rate'] == 0
    X,y,_ = load_data('iris')
    train,test = next(StratifiedKFold(10,shuffle=True,random_state=0).split(X,y))
    det,acc = run_fold(X,y,train,test,'symmetric',.2,0,0,'relabel',5,['DT'])
    assert all(row['rows_after'] == len(train) for row in det)
    assert len(acc) == 8 and len(det) == 7
    # Changing held-out labels changes evaluation, never filter decisions.
    poisoned = y.copy()
    poisoned[test] = np.roll(poisoned[test],1)
    changed,_ = run_fold(X,poisoned,train,test,'symmetric',.2,0,0,'relabel',5,['DT'])
    assert pd.DataFrame(det).equals(pd.DataFrame(changed)), 'test labels reached training or filtering'
    constant = WeightedNB(likelihood='mixed').fit(np.full((6,2),2.0),np.array([0,1,0,1,0,1]))
    posterior = constant.predict_proba(np.full((2,2),3.0))
    assert np.isfinite(posterior).all() and np.allclose(posterior.sum(axis=1),1)
    print('E12 regression checks: ALL PASS')


if __name__ == '__main__':
    main()
