# Разведочная находка: курение и оценки. Запуск: python replication/figure_smoking.py [путь к data_clean.pkl]
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data_clean.pkl'))
OUT = os.path.join(HERE, 'figures', '12_smoking_and_marks.png')
SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'savefig.facecolor': SURF, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.titlecolor': INK})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, w, b in zip(axs, '34', ['−0,46', '−0,34']):
    r = D['in_w3'] if w == '3' else D['in_w4'] & ~D['tech_4w']; e = D[r]
    g = e.groupby(f'mark_{w}')[f'smoke_{w}']; v = g.mean().reindex([1, 2, 3, 4]) * 100; n = g.count().reindex([1, 2, 3, 4])
    x = np.arange(4)
    ax.bar(x, v.values, width=0.62, color=RAMP, edgecolor=SURF, linewidth=2)
    for i, val in enumerate(v.values):
        ax.text(i, val + 0.8, f'{val:.0f}%', ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold')
    ax.set_xticks(x, [f'{l}\nn={k}' for l, k in zip(['В основном\nтройки', 'В основном\nчетвёрки', '4 и 5', 'Только\nпятёрки'], n.values)], fontsize=9)
    ax.set_title(f'{w}-я волна'); ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0); ax.set_ylim(0, 48)
    ax.yaxis.set_major_formatter(lambda y, _: f'{y:.0f}%')
axs[0].set_ylabel('Курят')
fig.tight_layout(rect=[0, 0.09, 1, 0.9])
fig.suptitle('Чем ниже оценки, тем чаще студенты курят', x=0.01, y=0.995, ha='left', va='top', fontsize=14, fontweight='bold', color=INK)
fig.text(0.01, 0.005, 'При учёте пола, направления, работы, достатка, спорта, сна и алкоголя курящие учатся хуже на 0,46 и 0,34 балла (p < 0,002).\n'
         'Курение в 3-й волне не предсказывает изменения оценок к 4-й: это устойчивая черта, а не воздействие. '
         'Данные: очищенный массив (cleaning.ipynb), анкеты-сбои исключены.', ha='left', va='bottom', fontsize=9, color=MUTED)
fig.savefig(OUT, dpi=160, bbox_inches='tight'); print('saved', OUT)
