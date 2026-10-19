# Перебор всех пар переменных в каждой волне с поправкой Бенджамини–Хохберга.
# Запуск: python replication/pair_scan.py [путь к data_clean.pkl] -> replication/pair_scan_results.csv
import os, sys, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_colwidth', 40)
HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data_clean.pkl'))

def build(w):
    s = str(w)
    rows = D['in_w3'] if w == 3 else (D['in_w4'] & ~D['tech_4w'])
    num = {'часы': f'hours_{s}', 'оценки': f'mark_{s}', 'удовл': f'sat_{s}', 'сон_будни': f'sleep_dur_wd_{s}',
           'сон_выходные': f'sleep_dur_we_{s}', 'джетлаг': f'sjl_{s}', 'достаток': 'ses',
           'работает': f'emp_{s}', 'отчисл_мысли': f'drop_any_{s}', 'спорт': f'pe_{s}', 'курит': f'smoke_{s}',
           'алкоголь': f'alco_{s}', 'стажировка': f'intern_{s}', 'стаж_оплач': f'intern_paid_{s}', 'ищет_работу': f'intent_{s}'}
    for i, nm in {1: 'прич_вуз', 2: 'прич_направл', 3: 'прич_академ', 4: 'прич_одногр', 5: 'прич_препод', 6: 'прич_экон', 7: 'прич_совмещ'}.items():
        num[nm] = f'reason{i}_{s}'
    if w == 3:
        num.update({'спорт_частота': 'pe_freq_3', 'стаж_соотв': 'intern_fit_3', 'стаж_польза': 'intern_benefit_3',
                    'стаж_вред': 'intern_harm_3', 'стаж_трудно': 'intern_difficulty_3'})
        num.update({f'время_{k}': f'alloc_{k}_3' for k in ['учёба', 'клубы', 'работа', 'волонтёрство', 'курсы', 'остальное']})
    else:
        num.update({'соотв_спец': 'fit_4', 'польза': 'benefit_4', 'вред': 'harm_4', 'трудно': 'difficulty_4', 'стаж_соотв': 'intern_fit_4'})
    cat = {'формат': f'fmt_{s}', 'мотив': f'motive_raw_{s}', 'сфера': f'sphere_{s}', 'повт_выбор': f'edu3_{s}',
           'пол': 'sex_c', 'финанс': 'funding_c', 'направл': 'major_c'}
    X = pd.DataFrame({k: D.loc[rows, v].astype(float) for k, v in num.items()})
    for k, v in cat.items():
        X[k] = D.loc[rows, v].astype(object)
    X['повт_тот_же'] = (X['повт_выбор'] == 'тот же вуз, та же специальность').astype(float).where(X['повт_выбор'].notna())
    return X, set(cat)

res = []
for w in (3, 4):
    X, CAT = build(w)
    for a, b in itertools.combinations(X.columns, 2):
        m = X[[a, b]].dropna()
        if len(m) < 40: continue
        if a not in CAT and b not in CAT:
            if m[a].nunique() < 2 or m[b].nunique() < 2: continue
            r, p = stats.spearmanr(m[a], m[b]); res.append((w, a, b, 'rho', r, abs(r), p, len(m)))
        else:
            ca = m[a] if a in CAT else (pd.qcut(m[a], 4, duplicates='drop') if m[a].nunique() > 5 else m[a])
            cb = m[b] if b in CAT else (pd.qcut(m[b], 4, duplicates='drop') if m[b].nunique() > 5 else m[b])
            ct = pd.crosstab(ca, cb)
            ct = ct.loc[ct.sum(1) >= 5, ct.sum(0) >= 5]
            if min(ct.shape) < 2: continue
            chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
            v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))); res.append((w, a, b, 'V', v, v, p, int(ct.values.sum())))
R = pd.DataFrame(res, columns=['волна', 'a', 'b', 'мера', 'эффект', 'abs', 'p', 'n'])
R['p_fdr'] = multipletests(R.p, method='fdr_bh')[1]
R.to_csv(os.path.join(HERE, 'pair_scan_results.csv'), index=False)
print('всего пар', len(R), 'значимых после FDR', (R.p_fdr < .05).sum())
