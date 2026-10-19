# Репликационный анализ: совмещение учёбы и работы (3-я и 4-я волны КИВО).
# Вход: data_clean.pkl из cleaning.ipynb. Запуск: python replication/analysis_replication.py [путь к data_clean.pkl]
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)

# Очищенный массив коллеги (cleaning.ipynb из KIVO_2026): аналитические переменные, статусы, флаги качества
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_clean.pkl')
D = pd.read_pickle(DATA)
in3 = D['in_w3']
in4 = D['in_w4'] & ~D['tech_4w']      # анкеты-сбои 4-й волны (профили A/B) исключены

FMT = {'нет работы': '0_нет', 'подработки': '1_подработки', 'неполный день': '2_неполный', 'полный день': '3_полный'}
MOTIVE = {'оплата обучения': 'Оплачивать обучение', 'содержать себя': 'Содержать себя', 'доп. доход': 'Иметь дополнительный',
          'опыт': 'Получить опыт', 'интерес': 'Реализовать интерес', 'другое': 'Другое'}

def valid(var):
    return D[var].where(D[f'{var}_st'].eq('VALID'))

common = pd.DataFrame({'sex': D['sex_c'].astype(object), 'funding': D['funding_c'].astype(object),
                       'ses': D['ses'], 'major': D['major_c'].astype(object)})

def wave(w):
    s = str(w)
    d = common.copy()
    d['work'] = D[f'fmt_{s}'].astype(object).map(FMT)
    d['works'] = D[f'emp_{s}']
    d['hours'] = D[f'hours_{s}'].where(D[f'emp_{s}'].eq(1))
    d['hours_raw_bad'] = False
    d['mark'] = D[f'mark_{s}']
    d['sat'] = D[f'sat_{s}']
    d['drop'] = D[f'drop_any_{s}']
    edu = D[f'edu3_{s}'].astype(object)
    d['reenroll_same'] = edu.eq('тот же вуз, та же специальность').astype(float).where(edu.notna())
    d['bed'] = 19 + valid(f'sleep1_{s}w')
    d['sleep_dur'] = D[f'sleep_dur_wd_{s}']
    d['jetlag'] = D[f'sjl_{s}']
    d['sport'] = D[f'pe_{s}']
    d['smoke'] = D[f'smoke_{s}']
    d['alco'] = D[f'alco_{s}']
    d['motive'] = D[f'motive_raw_{s}'].astype(object).map(MOTIVE)
    d['sphere'] = D[f'sphere_{s}'].astype(object).replace({'другое': np.nan})
    d['intern'] = D[f'intern_{s}']
    d['drop_workreason'] = D[f'reason7_{s}'].where(D[f'reasons_cmp_{s}'] & D[f'drop_any_{s}'].eq(1))
    # выбывание: «остался» — есть ответ о работе в обеих волнах (как в cleaning.ipynb, раздел 13)
    d['retained'] = (D['emp_3'].notna() & D['emp_4'].notna()).astype(float)
    if w == 3:
        d['pct_study'] = D['alloc_учёба_3'] * 100
        d['pct_work'] = D['alloc_работа_3'] * 100
        d['pe_freq'] = D['pe_freq_3']
    else:
        d['work_match'] = D['fit_4']
        d['helps'] = D['benefit_4']; d['harms'] = D['harm_4']; d['hard'] = D['difficulty_4']
    return d[in3 if w == 3 else in4]

W = {3: wave(3), 4: wave(4)}
for w in W: print('hours valid', w, W[w].hours.notna().sum(), 'median', W[w].hours.median(), W[w].groupby('work').hours.median().to_dict())

def cramer(ct):
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum(); k = min(ct.shape) - 1
    return chi2, p, dof, np.sqrt(chi2 / (n * k)), (exp < 5).mean(), n

def stdres(ct):
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum(); r = ct.sum(1).values[:, None] / n; c = ct.sum(0).values[None, :] / n
    return pd.DataFrame((ct.values - exp) / np.sqrt(exp * (1 - r) * (1 - c)), index=ct.index, columns=ct.columns).round(1)

print('# SAMPLE'); print('w3', in3.sum(), 'w4', in4.sum(), 'both', (in3 & in4).sum())
for w in (3, 4):
    print(f'\n######## WAVE {w}  n={len(W[w])}')
    d = W[w]
    print(d.work.value_counts(normalize=True).sort_index().round(3).to_dict(), 'n=', d.work.notna().sum())

    print('\n## CHI2 / Cramer V')
    pairs = [('work', v) for v in ['sex', 'funding', 'major', 'ses', 'drop', 'reenroll_same', 'sport', 'smoke', 'alco', 'intern']] + \
            [('motive', v) for v in ['sex', 'funding', 'ses', 'major', 'work']] + [('sphere', 'motive'), ('sphere', 'work'), ('sphere', 'sex')]
    rows = []
    for a, b in pairs:
        ct = pd.crosstab(d[a], d[b])
        if ct.shape[0] < 2 or ct.shape[1] < 2: continue
        chi2, p, dof, v, small, n = cramer(ct)
        fisher = stats.fisher_exact(ct.values)[1] if ct.shape == (2, 2) else np.nan
        rows.append(dict(a=a, b=b, n=n, chi2=round(chi2, 1), df=dof, p=p, V=round(v, 3), small_exp=round(small, 2), fisher=fisher))
    r = pd.DataFrame(rows); r['p_fdr'] = multipletests(r.p, method='fdr_bh')[1]
    print(r.assign(p=r.p.map('{:.2g}'.format), p_fdr=r.p_fdr.map('{:.2g}'.format)).to_string(index=False))

    print('\n## Row %, key tables')
    for a, b in [('work', 'drop'), ('work', 'sport'), ('work', 'funding'), ('work', 'ses'), ('motive', 'ses'), ('motive', 'funding'), ('work', 'smoke'), ('work', 'reenroll_same'), ('sphere', 'motive'), ('work', 'intern'), ('work', 'major'), ('work', 'sex')]:
        ct = pd.crosstab(d[a], d[b])
        print(f'\n{a} x {b}  (row %)'); print((pd.crosstab(d[a], d[b], normalize='columns' if a == 'work' and b in ('ses', 'funding', 'major', 'sex') else 'index') * 100).round(1))
        print('std residuals:'); print(stdres(ct))

    print('\n## Kruskal-Wallis: work type -> ordinal/numeric')
    outs = ['mark', 'sat', 'bed', 'sleep_dur', 'jetlag'] + (['pct_study', 'pe_freq'] if w == 3 else [])
    for y in outs:
        sub = d[['work', y]].dropna()
        groups = [g[y].values for _, g in sub.groupby('work')]
        H, p = stats.kruskal(*groups)
        eta2 = (H - len(groups) + 1) / (len(sub) - len(groups))
        means = sub.groupby('work')[y].agg(['mean', 'median', 'count']).round(2)
        print(f'\n{y}: H={H:.2f} p={p:.3g} eta2={eta2:.4f}'); print(means.T)
        if p < .05:
            g = dict(list(sub.groupby('work')[y]))
            for k in g:
                if k != '0_нет':
                    u, pp = stats.mannwhitneyu(g[k], g['0_нет'])
                    print(f'   {k} vs нет: p={pp:.3g} (Bonferroni x3 = {min(pp*3,1):.3g})')
    print('\n## Mann-Whitney works vs not')
    for y in outs:
        sub = d[['works', y]].dropna()
        a_, b_ = sub[sub.works == 1][y], sub[sub.works == 0][y]
        u, p = stats.mannwhitneyu(a_, b_)
        rbc = 1 - 2 * u / (len(a_) * len(b_))
        print(f'{y}: works mean={a_.mean():.2f} not={b_.mean():.2f} p={p:.3g} r_rb={-rbc:.3f}')

    print('\n## Kruskal: motive -> outcomes (workers)')
    for y in ['hours', 'mark', 'sat', 'drop'] + (['helps', 'harms', 'hard', 'work_match'] if w == 4 else []):
        sub = d[['motive', y]].dropna(); sub = sub[sub.motive != 'Другое']
        groups = [g[y].values for _, g in sub.groupby('motive') if len(g) >= 10]
        H, p = stats.kruskal(*groups)
        print(f'{y}: H={H:.2f} p={p:.3g}', sub.groupby('motive')[y].mean().round(2).to_dict())

    print('\n## Spearman matrix (workers)')
    cols = ['hours', 'mark', 'sat', 'drop', 'bed', 'sleep_dur', 'jetlag', 'ses'] + (['pct_work', 'pct_study'] if w == 3 else ['work_match', 'helps', 'harms', 'hard'])
    sub = d[d.works == 1][cols]
    rho = sub.corr(method='spearman').round(2)
    P = pd.DataFrame(index=cols, columns=cols, dtype=float); N = P.copy()
    for i in cols:
        for j in cols:
            m = sub[[i, j]].dropna()
            if i != j and len(m) > 10:
                P.loc[i, j] = stats.spearmanr(m[i], m[j])[1]; N.loc[i, j] = len(m)
    print(rho); print('p:'); print(P.map(lambda x: '' if pd.isna(x) else ('***' if x < .001 else '**' if x < .01 else '*' if x < .05 else 'ns')))
    print('N min/max', N.min().min(), N.max().max())
    if w == 3:
        allsub = d[['pct_work', 'pct_study', 'mark', 'sat', 'drop', 'bed', 'sleep_dur']].dropna()
        print('\nAll w3 respondents pct_work correlations:')
        for y in ['pct_study', 'mark', 'sat', 'drop', 'bed', 'sleep_dur']:
            m = d[['pct_work', y]].dropna(); r_, p_ = stats.spearmanr(m.pct_work, m[y]); print(f'  pct_work~{y}: rho={r_:.3f} p={p_:.3g} n={len(m)}')

    print('\n## Logistic: drop ~ work + controls')
    m = d[['drop', 'work', 'sex', 'funding', 'major', 'ses', 'mark']].dropna()
    for f in ['drop ~ C(work)', 'drop ~ C(work) + C(sex) + C(funding) + C(major) + ses', 'drop ~ C(work) + C(sex) + C(funding) + C(major) + ses + mark']:
        mod = smf.logit(f, data=m).fit(disp=0)
        ci = np.exp(mod.conf_int()); orr = pd.DataFrame({'OR': np.exp(mod.params), 'lo': ci[0], 'hi': ci[1], 'p': mod.pvalues}).round(3)
        print(f'\n{f}  n={int(mod.nobs)} pseudoR2={mod.prsquared:.3f}'); print(orr)

    print('\n## Logistic: drop_workreason (among those who thought) ~ work + controls')
    m = d[['drop_workreason', 'work', 'sex', 'funding', 'ses']].dropna()
    mod = smf.logit('drop_workreason ~ C(work) + C(sex) + C(funding) + ses', data=m).fit(disp=0)
    ci = np.exp(mod.conf_int()); print(f'n={int(mod.nobs)} pseudoR2={mod.prsquared:.3f}')
    print(pd.DataFrame({'OR': np.exp(mod.params), 'lo': ci[0], 'hi': ci[1], 'p': mod.pvalues}).round(3))
    print(m.groupby('work').drop_workreason.agg(['mean', 'count']).round(3))

    print('\n## Logistic: sport ~ works + controls')
    m = d[['sport', 'works', 'sex', 'funding', 'major', 'ses']].dropna()
    mod = smf.logit('sport ~ works + C(sex) + C(funding) + C(major) + ses', data=m).fit(disp=0)
    print(f'n={int(mod.nobs)}', 'OR works=', round(np.exp(mod.params['works']), 3), 'CI', np.exp(mod.conf_int().loc['works']).round(3).tolist(), 'p', round(mod.pvalues['works'], 4))

print('\n######## PANEL (both waves)')
both = in3 & in4
w3, w4 = W[3].loc[both[in3].index[both[in3]]], W[4].loc[both[in4].index[both[in4]]]
p = pd.DataFrame({'w3': w3.works, 'w4': w4.works}).dropna()
ct = pd.crosstab(p.w3, p.w4); print(ct)
b, c = ct.loc[0, 1], ct.loc[1, 0]
print(f'McNemar: начали={b} бросили={c} chi2={(abs(b-c)-1)**2/(b+c):.2f} p={stats.chi2.sf((abs(b-c)-1)**2/(b+c),1):.3g} exact p={stats.binomtest(b, b+c).pvalue:.3g}')
print(pd.crosstab(w3.work, w4.work))
traj = p.apply(lambda r: {(0, 0): 'не работает', (0, 1): 'начал', (1, 0): 'бросил', (1, 1): 'работает в обеих'}[(r.w3, r.w4)], axis=1)
for y in ['mark', 'sat', 'drop', 'bed', 'sleep_dur', 'sport']:
    dd = pd.DataFrame({'traj': traj, 'd': w4.loc[traj.index, y] - w3.loc[traj.index, y]}).dropna()
    H, pp = stats.kruskal(*[g.d.values for _, g in dd.groupby('traj')])
    print(f'Δ{y}: H={H:.2f} p={pp:.3g}', dd.groupby('traj').d.agg(['mean', 'count']).round(2).to_dict())
for y in ['mark', 'sat', 'drop', 'sport']:
    m = pd.DataFrame({'a': w3[y], 'b': w4[y]}).dropna(); r_, pp = stats.spearmanr(m.a, m.b)
    print(f'stability {y}: rho={r_:.3f} n={len(m)}')
# internship in w3 -> work match in w4
m = pd.DataFrame({'intern3': w3.intern, 'match4': w4.work_match}).dropna()
print('intern w3 -> work_match w4:', m.groupby('intern3').match4.agg(['mean', 'count']).round(2).to_dict(), 'MWU p=%.3g' % stats.mannwhitneyu(*[g.match4 for _, g in m.groupby('intern3')])[1])

print('\n######## ATTRITION: w3 participants retained in w4')
a = W[3].copy()
print('retained share', a.retained.mean().round(3))
for v in ['work', 'drop', 'mark', 'sat', 'sex', 'funding']:
    ct = pd.crosstab(a[v], a.retained)
    chi2, p_, dof, V, small, n = cramer(ct)
    print(f'{v}: V={V:.3f} p={p_:.3g}', (a.groupby(v).retained.mean() * 100).round(1).to_dict())
m = a[['retained', 'works', 'drop', 'mark', 'sex', 'funding']].dropna()
mod = smf.logit('retained ~ works + drop + mark + C(sex) + C(funding)', data=m).fit(disp=0)
print(pd.DataFrame({'OR': np.exp(mod.params), 'p': mod.pvalues}).round(3))
