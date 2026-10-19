# Графики к replication_report.md. Запуск: python replication/figures_replication.py [путь к data_clean.pkl]
import sys, io, os, contextlib, runpy
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.formula.api as smf

sys.argv = ['analysis_replication.py'] + sys.argv[1:2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(os.path.join(os.path.dirname(__file__), 'analysis_replication.py'))
W = G['W']

# ---- tokens (reference palette, light) ----
SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
WORK_C = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']          # ordinal blue: нет → полный день
WAVE_C = {3: '#2a78d6', 4: '#eb6834'}                           # categorical slots 1, 2
CAT3 = ['#2a78d6', '#eb6834', '#1baf7a']
WORK_KEYS = ['0_нет', '1_подработки', '2_неполный', '3_полный']
WORK_LAB = ['Не работает', 'Приработки', 'Неполный\nдень', 'Полный\nдень']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'savefig.facecolor': SURF, 'axes.edgecolor': BASE, 'axes.linewidth': 1, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 12, 'axes.titlecolor': INK,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False, 'legend.labelcolor': INK2,
})

def finish(fig, name, title, note=None):
    h = fig.get_size_inches()[1]
    top = 1 - 0.55 / h; bottom = (0.45 / h) if note else 0
    fig.tight_layout(rect=[0, bottom, 1, top])
    fig.suptitle(title, x=0.01, y=0.995, ha='left', va='top', fontsize=14, fontweight='bold', color=INK)
    note = (note + ' ' if note else '') + 'Данные: очищенный массив (cleaning.ipynb), анкеты-сбои 4-й волны исключены.'
    if note:
        fig.text(0.01, 0.005, note, ha='left', va='bottom', fontsize=9, color=MUTED, wrap=True)
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)

def bars(ax, vals, labels=WORK_LAB, colors=WORK_C, fmt='{:.0f}%', ymax=None, ns=None):
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=colors, edgecolor=SURF, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.15', fc=SURF, ec='none'))
    ax.set_xticks(x, labels if ns is None else [f'{l}\nn={n}' for l, n in zip(labels, ns)], fontsize=9)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0)
    if ymax: ax.set_ylim(0, ymax)

def share(d, col, by='work'):
    g = d[[by, col]].dropna().groupby(by)[col]
    return (g.mean() * 100).reindex(WORK_KEYS), g.size().reindex(WORK_KEYS)

# 01 — мысли об отчислении
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, w in zip(axs, (3, 4)):
    v, n = share(W[w], 'drop'); bars(ax, v.values, ymax=70, ns=n.values)
    ax.set_title(f'{w}-я волна'); ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}%')
axs[0].set_ylabel('Задумывались об отчислении')
finish(fig, '01_dropout_by_employment.png', 'Работающие полный день чаще задумываются об отчислении',
       'Доля ответивших «да, думал(а)» или «да, и намерен(а)». χ²: 3-я волна p = 0,007; 4-я волна p = 0,16.')

# 02 — отношения шансов (логит с контролями и оценками)
fig, ax = plt.subplots(figsize=(9, 4.4))
labels = ['Приработки', 'Неполный день', 'Полный день']
for k, w in enumerate((3, 4)):
    m = W[w][['drop', 'work', 'sex', 'funding', 'major', 'ses', 'mark']].dropna()
    mod = smf.logit('drop ~ C(work) + C(sex) + C(funding) + C(major) + ses + mark', data=m).fit(disp=0)
    ci = np.exp(mod.conf_int())
    for i, key in enumerate(WORK_KEYS[1:]):
        p = f'C(work)[T.{key}]'; y = i + (0.15 if w == 3 else -0.15)
        ax.plot([ci.loc[p, 0], ci.loc[p, 1]], [y, y], color=WAVE_C[w], lw=2, solid_capstyle='round')
        ax.scatter(np.exp(mod.params[p]), y, s=70, color=WAVE_C[w], edgecolor=SURF, linewidth=2, zorder=3,
                   label=f'{w}-я волна (n = {int(mod.nobs)})' if i == 0 else None)
        ax.text(ci.loc[p, 1] * 1.04, y, f'{np.exp(mod.params[p]):.2f}'.replace('.', ','), va='center', fontsize=9, color=INK2)
ax.axvline(1, color=INK2, lw=1)
ax.text(1.02, 2.55, 'нет связи (OR = 1)', fontsize=9, color=MUTED)
ax.set_xscale('log'); ax.set_xticks([0.5, 1, 2, 4, 8], ['0,5', '1', '2', '4', '8']); ax.minorticks_off()
ax.set_yticks(range(3), labels); ax.set_ylim(-0.6, 2.8); ax.grid(axis='y', visible=False)
ax.set_xlabel('Отношение шансов задуматься об отчислении (против неработающих), 95% ДИ')
ax.legend(loc='lower right')
finish(fig, '02_dropout_odds_ratios.png', 'Работа полный день: шансы задуматься об отчислении выше, но эффект неустойчив',
       'Логистическая регрессия; контроль: пол, форма обучения, направление, материальное положение семьи, оценки. Без поправки на множественность: после поправки Холма (analysis.ipynb) связь незначима.')

# 03 — причина «трудности совмещения» среди задумывавшихся
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, w in zip(axs, (3, 4)):
    v, n = share(W[w], 'drop_workreason'); bars(ax, v.values, ymax=80, ns=n.values)
    ax.set_title(f'{w}-я волна'); ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}%')
axs[0].set_ylabel('Назвали «трудности совмещения\nучёбы и работы»')
finish(fig, '03_dropout_reason_combining.png', 'Для работающих совмещение — частая причина мыслей об отчислении',
       'Задумывавшиеся об отчислении и отметившие хотя бы одну причину. Различия с неработающими p ≤ 0,001 (логистическая регрессия).')

# 04 — соответствие специальности → «помогает учиться»
d4 = W[4][W[4].works == 1]
m = d4[['work_match', 'helps']].dropna()
v = m.groupby('work_match').helps.apply(lambda s: (s >= 3).mean() * 100); n = m.groupby('work_match').size()
fig, ax = plt.subplots(figsize=(8.5, 4.6))
bars(ax, v.values, labels=['Совсем не\nсоответствует', 'Скорее не\nсоответствует', 'Скорее\nсоответствует', 'Полностью\nсоответствует'],
     colors=WORK_C, ymax=100, ns=n.values)
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}%')
ax.set_ylabel('Согласны: «опыт работы помогает\nусваивать учебный материал»')
ax.set_xlabel('Насколько работа соответствует специальности')
finish(fig, '04_fit_and_benefit.png', 'Работа по специальности помогает учиться',
       'Работающие студенты, 4-я волна. ρ Спирмена = 0,49, p < 0,001.')

# 05 — порог нагрузки
d4b = d4.copy(); d4b['band'] = pd.cut(d4b.hours, [-1, 10, 20, 30, 168], labels=['0–10', '11–20', '21–30', '31+'])
fig, ax = plt.subplots(figsize=(9, 4.8))
series = [('helps', '«Работа помогает учиться»'), ('harms', '«Из-за работы страдает успеваемость»'), ('hard', '«Очень трудно совмещать»')]
x = np.arange(4)
for (col, lab), c in zip(series, CAT3):
    s = d4b[['band', col]].dropna().groupby('band', observed=True)[col].apply(lambda v: (v >= 3).mean() * 100)
    ax.plot(x, s.values, color=c, lw=2, marker='o', ms=8, mec=SURF, mew=2, label=lab)
    ax.text(3.12, s.values[-1], f'{s.values[-1]:.0f}%', va='center', fontsize=10, color=INK, fontweight='bold')
ax.axvspan(0.6, 1.4, color=GRID, alpha=0.5, lw=0, zorder=0)
ax.text(1, 72, 'зона баланса', ha='center', fontsize=9, color=INK2)
ax.set_xticks(x, ['0–10', '11–20', '21–30', '31+']); ax.set_xlim(-0.3, 3.5); ax.set_ylim(0, 75)
ax.set_xlabel('Часов работы на прошлой неделе'); ax.set_ylabel('Доля согласных')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}%'); ax.grid(axis='x', visible=False)
ax.legend(loc='lower right', fontsize=10)
finish(fig, '05_hours_and_difficulties.png', 'Больше 20 часов в неделю: жалобы растут, польза — нет',
       'Работающие студенты, 4-я волна, n ≈ 400. Краскел–Уоллис: помогает p = 0,037; страдает p = 0,003; трудно p = 0,001.')

# 06 — профиль мотивов
mot = d4[d4.motive.isin(['Получить опыт', 'Реализовать интерес', 'Иметь дополнительный', 'Содержать себя'])]
order = ['Получить опыт', 'Реализовать интерес', 'Иметь дополнительный', 'Содержать себя']
mlab = ['Получить опыт', 'Реализовать интерес', 'Доп. доход', 'Содержать себя']
cols = [('work_match', 'Соответствие\nспециальности'), ('helps', '«Помогает учиться»'), ('harms', '«Страдает успеваемость»'), ('hard', '«Трудно совмещать»')]
fig, axs = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
for ax, (c, t) in zip(axs, cols):
    v = mot.groupby('motive')[c].mean().reindex(order)
    best = v.idxmax()
    colors = ['#1c5cab' if k == best else '#86b6ef' for k in order]
    ax.barh(np.arange(4), v.values, color=colors, height=0.6, edgecolor=SURF, linewidth=2)
    for i, val in enumerate(v.values):
        ax.text(val + 0.05, i, f'{val:.2f}'.replace('.', ','), va='center', fontsize=10, color=INK)
    ax.set_title(t, fontsize=11); ax.set_xlim(1, 4); ax.set_xticks([1, 2, 3, 4])
    ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
axs[0].set_yticks(np.arange(4), mlab); axs[0].invert_yaxis()
finish(fig, '06_work_motives.png', 'Работа «ради опыта» помогает учиться, работа «чтобы содержать себя» даётся труднее всего',
       'Средние по шкале 1–4, работающие студенты 4-й волны (n ≈ 400). Тёмный столбец — максимум в каждой панели. Все различия p < 0,01.')

# 07 — оценки: нулевой результат
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, w in zip(axs, (3, 4)):
    g = W[w][['work', 'mark']].dropna().groupby('work').mark
    mu, se = g.mean().reindex(WORK_KEYS), (g.std() / np.sqrt(g.size())).reindex(WORK_KEYS)
    x = np.arange(4)
    for i in range(4):
        ax.plot([i, i], [mu.iloc[i] - 1.96 * se.iloc[i], mu.iloc[i] + 1.96 * se.iloc[i]], color=WORK_C[i], lw=2)
        ax.scatter(i, mu.iloc[i], s=80, color=WORK_C[i], edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(i + 0.12, mu.iloc[i], f'{mu.iloc[i]:.2f}'.replace('.', ','), va='center', fontsize=10, color=INK)
    ax.set_xticks(x, WORK_LAB, fontsize=9); ax.set_xlim(-0.5, 3.7); ax.set_ylim(1.8, 3.2)
    ax.set_title(f'{w}-я волна'); ax.grid(axis='x', visible=False)
axs[0].set_yticks([2, 2.5, 3], ['2 — четвёрки', '2,5', '3 — 4 и 5'])
axs[0].set_ylabel('Средняя оценка за сессию')
finish(fig, '07_marks_by_employment.png', 'Оценки работающих и неработающих не различаются',
       'Среднее и 95% ДИ. Шкала: 1 — тройки, 2 — четвёрки, 3 — 4 и 5, 4 — только пятёрки. Краскел–Уоллис: p = 0,22 и p = 0,45.')

# 08 — бюджет времени (3-я волна)
g = W[3][['work', 'pct_study']].dropna().groupby('work').pct_study
fig, ax = plt.subplots(figsize=(8, 4.4))
bars(ax, g.mean().reindex(WORK_KEYS).values, ymax=70, ns=g.size().reindex(WORK_KEYS).values)
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}%'); ax.set_ylabel('Доля времени на учёбу в университете')
finish(fig, '08_study_time.png', 'Работа сокращает время на учёбу на 13–17 п.п.',
       '3-я волна, планируемое распределение времени в неделю (без сна и дороги). Краскел–Уоллис p < 0,001, η² = 0,08.')

# 09 — здоровье
ind = [('sleep_dur', 'Сон в будни, ч', '{:.1f}', (6, 8.2)), ('sport', 'Занимаются спортом', '{:.0f}%', (0, 90)),
       ('smoke', 'Курят', '{:.0f}%', (0, 45)), ('alco', 'Употребляют алкоголь', '{:.0f}%', (0, 80))]
fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
for r, w in enumerate((3, 4)):
    for c, (col, t, fmt, lim) in enumerate(ind):
        ax = axs[r, c]
        g = W[w][['work', col]].dropna().groupby('work')[col].mean().reindex(WORK_KEYS)
        vals = g.values if col == 'sleep_dur' else g.values * 100
        bars(ax, vals, labels=['Нет', 'Прираб.', 'Непол.', 'Полн.'], fmt=fmt.replace('.1f', '.1f'))
        ax.set_ylim(*lim)
        if col == 'sleep_dur':
            for txt in ax.texts: txt.set_text(txt.get_text().replace('.', ','))
            ax.yaxis.set_major_formatter(lambda x, _: f'{x:.1f}'.replace('.', ','))
        else:
            ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}%')
        ax.set_title(f'{t} · {w}-я волна', fontsize=11)
finish(fig, '09_health.png', 'Работающие спят меньше, чаще курят и пьют, но и чаще занимаются спортом',
       'Значимые различия (p < 0,05): сон — 3-я волна; спорт — 4-я волна; курение и алкоголь — обе волны. Ось сна начинается с 6 ч.')

# 10 — матрица Спирмена (4-я волна, работающие)
cols = ['hours', 'mark', 'sat', 'drop', 'sleep_dur', 'work_match', 'helps', 'harms', 'hard']
lab = ['Часы работы', 'Оценки', 'Удовлетворённость', 'Мысли об отчисл.', 'Сон, ч', 'Соответствие спец.', '«Помогает учиться»', '«Страдает успеваемость»', '«Трудно совмещать»']
from scipy import stats
sub = d4[cols]; R = sub.corr(method='spearman').values
P = np.ones_like(R)
for i in range(len(cols)):
    for j in range(len(cols)):
        if i != j:
            mm = sub[[cols[i], cols[j]]].dropna(); P[i, j] = stats.spearmanr(mm.iloc[:, 0], mm.iloc[:, 1])[1]
cmap = LinearSegmentedColormap.from_list('div', ['#b92f2f', '#e34948', '#f0efec', '#3987e5', '#1c5cab'])
mask = np.triu(np.ones_like(R, dtype=bool))
fig, ax = plt.subplots(figsize=(9.5, 8))
n_ = len(cols)
im = ax.imshow(np.ma.array(R, mask=mask)[1:, :-1], cmap=cmap, vmin=-0.6, vmax=0.6)
for i in range(len(cols)):
    for j in range(i):
        star = '***' if P[i, j] < .001 else '**' if P[i, j] < .01 else '*' if P[i, j] < .05 else ''
        txt = f'{R[i, j]:.2f}'.replace('.', ',') + ('\n' + star if star else '')
        ax.text(j, i - 1, txt, ha='center', va='center', fontsize=9, color='white' if abs(R[i, j]) > 0.4 else INK)
ax.set_xticks(range(n_ - 1), lab[:-1], rotation=40, ha='right', fontsize=9.5); ax.set_yticks(range(n_ - 1), lab[1:], fontsize=9.5)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02); cb.outline.set_visible(False); cb.set_label('ρ Спирмена', color=INK2)
finish(fig, '10_spearman_heatmap.png', 'Две оси восприятия работы: «польза» и «цена»',
       'Работающие студенты, 4-я волна, n = 243–440. * p < 0,05, ** p < 0,01, *** p < 0,001.')

# 11 — выбывание
a = W[3].copy(); a['ret'] = a['retained']
rows = [('Не работает', a.work == '0_нет'), ('Приработки', a.work == '1_подработки'), ('Неполный день', a.work == '2_неполный'),
        ('Полный день', a.work == '3_полный'), (None, None),
        ('Тройки', a.mark == 1), ('Четвёрки', a.mark == 2), ('4 и 5', a.mark == 3), ('Только пятёрки', a.mark == 4), (None, None),
        ('Бюджет', a.funding == 'бюджет'), ('Платное', a.funding == 'платное'), ('Целевое', a.funding == 'целевое')]
fig, ax = plt.subplots(figsize=(9, 6.2))
y = 0; yt, yl = [], []
grp_col = {0: WORK_C, 1: WORK_C, 2: ['#86b6ef', '#3987e5', '#1c5cab']}
gi, k = 0, 0
for labr, msk in rows:
    if labr is None: y += 0.6; gi += 1; k = 0; continue
    v = a.loc[msk, 'ret'].mean() * 100; nn = int(msk.sum())
    ax.barh(y, v, color=grp_col[gi][k], height=0.7, edgecolor=SURF, linewidth=2)
    ax.text(v + 1, y, f'{v:.0f}%  (n={nn})', va='center', fontsize=9.5, color=INK, zorder=4, bbox=dict(boxstyle='square,pad=0.2', fc=SURF, ec='none'))
    yt.append(y); yl.append(labr); y += 1; k += 1
ax.axvline(a.ret.mean() * 100, color=INK2, lw=1)
ax.text(a.ret.mean() * 100 + 0.8, -0.9, f'в среднем {a.ret.mean()*100:.0f}%', fontsize=9, color=INK2)
ax.set_yticks(yt, yl); ax.invert_yaxis(); ax.set_xlim(0, 75); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f}%'); ax.set_xlabel('Доля участников 3-й волны с ответом о работе в 4-й')
finish(fig, '11_attrition.png', 'Работающие и слабее успевающие реже дошли до 4-й волны',
       'Группы — по ответам в 3-й волне. Логистическая регрессия: работа OR = 0,69 (p = 0,006), оценки OR = 1,16 (p = 0,035).')
