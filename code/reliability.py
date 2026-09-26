# -*- coding: utf-8 -*-
import sys, csv
sys.stdout.reconfigure(encoding='utf-8')
from collections import Counter
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

rows = list(csv.DictReader(open('coding_validation_samples.csv', encoding='utf-8-sig')))

def colv(c):
    return [(r.get(c) or '').strip() for r in rows]

c1e, c2e = colv('C1_echo'), colv('C2_echo')
c1f, c2f = colv('C1_function'), colv('C2_function')
mirror = [float(r['mirror']) for r in rows]
cos_ = [float(r['cos_sem']) for r in rows]

def eord(x): return {'0':0,'1':1,'2':2}.get(x, None)
def ord_vec(xs):
    return np.array([eord(x) for x in xs], float)

def linear_weighted_kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    mat = np.zeros((3, 3))
    for x, y in zip(a, b):
        mat[int(x), int(y)] += 1
    p = mat / n
    po = np.trace(p)
    exp = p.sum(0) * p.sum(1)
    pe = exp.sum()
    w = np.abs(np.subtract.outer(np.arange(3), np.arange(3))) / 2.0
    pw = (w * exp).sum()
    kappa_w = 1 - (w * p).sum() / pw if pw else 0.0
    return po, kappa_w

def kappa_report(name, a, b, ordered=False):
    agree = np.mean([x == y for x, y in zip(a, b)])
    k = cohen_kappa_score(a, b)
    line = f'{name}: n={len(a)}  percent agreement={agree:.3f}  Cohen kappa={k:.3f}'
    if ordered:
        po, kw = linear_weighted_kappa(ord_vec(a), ord_vec(b))
        line += f'  (echo ordinal) linear-weighted kappa={kw:.3f}'
    print(line)
    return k, agree

out = []
out.append('# 正式信度集 信度与效度报告（C1–C2，n=92）')
out.append('')
out.append(f'- 样本 n = {len(rows)}（media×sensitive 分层，含 avoid=40 过采样）')
out.append('')
print('C1/C2 均 92 非空。')
k_e, a_e = kappa_report('echo (C1–C2)', c1e, c2e, ordered=True)
k_f, a_f = kappa_report('function (C1–C2)', c1f, c2f, ordered=False)

# effective human echo (mean of C1/C2 ordinal) and its validity vs algorithms
def human_mean_echo():
    ordm = np.nanmean(np.array([ord_vec(c1e), ord_vec(c2e)]), axis=0)
    return ordm
hmean = human_mean_echo()
rho_mir, pmir = spearmanr(hmean, mirror)
rho_cos, pcos = spearmanr(hmean, cos_)
print(f'\nConstruct validity (mean human echo, ordinal):')
print(f'  Spearman(mean_echo, mirror) = {rho_mir:.3f}  p={pmir:.1e}')
print(f'  Spearman(mean_echo, cos_sem) = {rho_cos:.3f}  p={pcos:.1e}')
# per-coder validity
for lbl, c in [('C1', c1e), ('C2', c2e)]:
    ov = ord_vec(c)
    r1, p1 = spearmanr(ov, mirror); r2, p2 = spearmanr(ov, cos_)
    print(f'  {lbl}: Spearman(echo, mirror)={r1:.3f} ; (echo, cos)={r2:.3f}')
    out.append(f'- {lbl} 效度：echo~mirror r={r1:.3f} (p={p1:.2e})；echo~cos r={r2:.3f} (p={p2:.2e})')

out.append('')
out.append('## 结果')
out.append(f'- echo (C1–C2)：一致率 {a_e:.3f}，Cohen κ = {k_e:.3f}'
           f'（有序取线性加权 κ）')
out.append(f'- function (C1–C2)：一致率 {a_f:.3f}，Cohen κ = {k_f:.3f}')
out.append(f'- 效度（平均人工 echo, 有序）与算法：mirror r={rho_mir:.3f} (p={pmir:.2e})；'
           f'cos r={rho_cos:.3f} (p={pcos:.2e})')
# disagreement cells
print('\nC1–C2 echo confusion:')
conf = {}
for x, y in zip(c1e, c2e):
    conf.setdefault((x, y), 0); conf[(x, y)] += 1
for k, v in sorted(conf.items()):
    print('   ', k, v)
print('C1–C2 function disagreements (nonzero off-diagonal):')
for x, y in zip(c1f, c2f):
    if x != y:
        print('   ', x, '->', y)
open('reliability_report.md', 'w', encoding='utf-8').write('\n'.join(out))
print('\nsaved reliability_report.md')
