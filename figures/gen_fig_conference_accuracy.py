"""Plot Protocol B accuracy changes from the saved E9 dataset averages."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'paper' / 'manuscript' / 'figures'
data = pd.read_csv(ROOT / 'results' / 'EXP-E9_farid' / 'metrics.csv')
data = data[data.protocol == 'B refit-per-fold']
assert not data.duplicated(['dataset', 'arm']).any()
assert data.groupby('arm').dataset.nunique().eq(10).all()
means = data.groupby('arm').accuracy.mean()
methods = ['N', 'A', 'NA', 'AN']
arms = ['Alg1->{final}', 'Alg2->{final}', 'C1 N->A->{final}', 'C2 A->N->{final}']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
fig, ax = plt.subplots(figsize=(3.45, 2.25))
y = np.arange(len(methods))
for final, offset, color, hatch, label in [('NB', -.16, 'white', '///', 'Final NB'),
                                         ('DT', .16, '0.35', None, 'Final C4.5')]:
    changes = [means[arm.format(final=final)] - means[f'baseline->{final}'] for arm in arms]
    bars = ax.barh(y + offset, changes, height=.28, color=color, edgecolor='black',
                   linewidth=.6, hatch=hatch, label=label)
    for bar, value in zip(bars, changes):
        ax.text(value + (.13 if value >= 0 else -.13), bar.get_y() + bar.get_height()/2,
                f'{value:+.2f}', va='center', ha='left' if value >= 0 else 'right', fontsize=8)
ax.axvline(0, color='black', linewidth=.8)
ax.set_yticks(y, methods)
ax.invert_yaxis()
ax.set_xlim(-7, 2.6)
ax.set_xticks([-6, -4, -2, 0, 2])
ax.set_xlabel('Accuracy change from plain classifier (pp)', fontsize=8.5)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color='0.88', linewidth=.5)
ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=2, frameon=False, fontsize=8)
fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.80)
OUTPUT.mkdir(exist_ok=True)
for suffix in ['pdf', 'png']:
    fig.savefig(OUTPUT / f'conference_accuracy_change.{suffix}', dpi=300)
print('Saved conference accuracy figure; NA/AN tree changes:',
      [round(means[a.format(final='DT')] - means['baseline->DT'], 2) for a in arms[2:]])
