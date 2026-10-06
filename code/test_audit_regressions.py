import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from faithful_nb import FaithfulNB
import e9_farid as e9


@pytest.mark.parametrize('weighted', [False, True])
def test_constant_numeric_scores_are_finite(weighted):
    X = np.ones((6, 1))
    y = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
    sample_weight = np.ones(6) if weighted else None
    model = FaithfulNB([False], [None]).fit(X, y, sample_weight)
    assert np.isfinite(model.class_scores(X)).all()
    assert (model.predict(X) == 'a').all()


def test_leaf_tree_standalone_nb_uses_priors(monkeypatch, tmp_path):
    X = np.array([[0], [0], [0], [0], [1], [1]], dtype=float)
    y = np.array(['a', 'a', 'a', 'a', 'b', 'b'])
    meta = {'names': ['feature'], 'nominal': [True], 'levels': [['zero', 'one']]}
    monkeypatch.setattr(e9, '_j48', lambda *args: ((100.0, 100.0), {}))
    rows = np.arange(6)
    scores, _ = e9.run_fold(tmp_path, X, y, meta, ['a', 'b'], rows, rows)
    assert scores['baseline->NB'][0] == 100.0
    assert scores['Alg2->NB'][0] == pytest.approx(100 * 4 / 6)


@pytest.mark.parametrize("accuracy", ["99", "float('nan')"])
def test_matrix_integrity_survives_optimized_python(accuracy):
    code = f"from e9_farid import checked_scores; checked_scores({accuracy}, [[1,0],[0,1]])"
    p = subprocess.run([sys.executable, '-O', '-c', code], cwd=Path(__file__).parent,
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert 'confusion matrix disagrees' in p.stderr


def test_pilot_does_not_overwrite_existing_results(monkeypatch, tmp_path):
    output = tmp_path / 'results'
    output.mkdir()
    sentinel = output / 'config.json'
    sentinel.write_text('preserve')
    monkeypatch.setattr(e9, 'RESULTS', output)
    monkeypatch.setattr(sys, 'argv', ['e9_farid.py', 'iris', '--seeds', '1'])
    with pytest.raises(SystemExit):
        e9.main()
    assert sentinel.read_text() == 'preserve'
