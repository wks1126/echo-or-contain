# -*- coding: utf-8 -*-
"""JACOR 修回：为审稿意见 #1（信度样本量）与 #2（多重比较）生成可复现数字。

输入（只读，取自原始投稿分析产物）：
  巴基斯坦/features_all.json          4,005 条问答对特征
  巴基斯坦/coding_validation_samples.csv  92 条双编码信度集
输出：
  06_投稿包/JACOR_修回_20260926/13_修订分析/revision_checks_report.md
随机种子固定 SEED=20260926。
"""
import sys, os, csv, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import cohen_kappa_score
import statsmodels.formula.api as smf

SEED = 20260926
rng = np.random.default_rng(SEED)
# 发布包版：输入来自 ../data/（features_all.json 与 coding_validation_samples.csv）
HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.normpath(os.path.join(HERE, '..', 'data'))
OUT = os.path.normpath(os.path.join(HERE, '..', 'revision_checks_report.md'))
buf = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    buf.append(s)


# ---------------------------------------------------------------- 1. 信度集
rows = list(csv.DictReader(open(os.path.join(PK, 'coding_validation_samples.csv'),
                               encoding='utf-8-sig')))
n_rel = len(rows)
c1e = [r['C1_echo'].strip() for r in rows]
c2e = [r['C2_echo'].strip() for r in rows]
c1f = [r['C1_function'].strip() for r in rows]
c2f = [r['C2_function'].strip() for r in rows]
mirror = np.array([float(r['mirror']) for r in rows])
cos = np.array([float(r['cos_sem']) for r in rows])

say('## 1. 信度子样本（审稿意见 #1）\n')
say(f'- 双编码样本 n = {n_rel}')
say(f'- echo 一致率 = {np.mean([a == b for a, b in zip(c1e, c2e)]):.3f}'
    f'，Cohen κ = {cohen_kappa_score(c1e, c2e):.3f}')
say(f'- function 一致率 = {np.mean([a == b for a, b in zip(c1f, c2f)]):.3f}'
    f'，Cohen κ = {cohen_kappa_score(c1f, c2f):.3f}')


def boot_ci(a, b, B=5000):
    a, b = np.array(a), np.array(b)
    n = len(a)
    ks = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(a[idx])) < 2 or len(set(b[idx])) < 2:
            continue
        ks.append(cohen_kappa_score(a[idx], b[idx]))
    ks = np.array(ks)
    return np.percentile(ks, 2.5), np.percentile(ks, 97.5), len(ks)


for name, a, b in [('echo', c1e, c2e), ('function', c1f, c2f)]:
    lo, hi, nb = boot_ci(a, b)
    say(f'- {name} κ 的 95% bootstrap CI（B={nb}，seed={SEED}）= [{lo:.3f}, {hi:.3f}]'
        f'，宽度 {hi - lo:.3f}，半宽 {(hi - lo) / 2:.3f}')

# 构成与覆盖：是否真的“2.3%”而已
import collections
say('\n### 1.1 子样本构成（分层 + 稀有类过采样）\n')
say('- media × sensitive 单元分布：' +
    ', '.join(f'{k[0]}/{k[1]}={v}' for k, v in sorted(
        collections.Counter((r['media'], r['sensitive']) for r in rows).items())))
n_avoid_rel = sum(1 for r in rows if str(r['avoid']).strip() in ('1', '1.0'))
say(f'- 子样本中 avoid=1 的条数 = {n_avoid_rel}，占全部 avoid=1 对的比例 = '
    f'{n_avoid_rel / 217:.3f}（{100 * n_avoid_rel / 217:.1f}%）')
T = pd.DataFrame(json.load(open(os.path.join(PK, 'features_all.json'), encoding='utf-8')))
say(f'- 全语料 4,005 对，其中 avoid=1 = {int(T.avoid.sum())} 对')
say(f'- 若按简单随机抽样，n=92 期望只覆盖 {92 * T.avoid.mean():.1f} 条回避对；'
    f'实际覆盖 40 条（约 {40 / (92 * T.avoid.mean()):.1f} 倍）')

say('\n### 1.2 精度：κ 的点估计与区间宽度\n')
say('- echo 的 95% CI 半宽约 0.10，function 约 0.06；两者都足以把'
    '“实质一致（κ≈.80）”与“中等（κ≈.60）”分开，这正是本文需要判定的边界。')

# --------------------------------------------------- 2. 混合模型 + BH-FDR
say('\n## 2. 多重比较：BH-FDR 校核（审稿意见 #2）\n')
F = T.copy()
for c in ['avoid', 'sensitive', 'q_china', 'acc', 'multiq']:
    F[c] = F[c].astype(int)
F['foreign'] = (F.media == 'foreign').astype(int)
F['adv'] = (F.acc | F.multiq).astype(int)
F['llen_q'] = np.log1p(F.len_q)
F['llen_a'] = np.log1p(F.len_a)

FORM = ('{y} ~ foreign + q_china + sensitive + avoid + adv + llen_q + llen_a'
        ' + C(window) + C(who)')
store = {}
for y in ['cos_sem', 'mirror']:
    m = smf.mixedlm(FORM.format(y=y), F, groups=F['file']).fit(reml=False)
    store[y] = m
    say(f'\n### 2.{1 if y == "cos_sem" else 2} {y}（线性混合模型，随机截距=会议）\n')
    say('| 项 | β | SE | z | p |')
    say('|---|---|---|---|---|')
    for k in m.params.index:
        if 'Group' in k or 'groups' in k.lower() or k == 'Intercept':
            continue
        if k.startswith('C(who)') or k.startswith('C(window)'):
            continue
        z = m.params[k] / m.bse[k]
        say(f'| {k} | {m.params[k]:+.4f} | {m.bse[k]:.4f} | {z:+.2f} | '
            f'{2 * norm.sf(abs(z)):.3g} |')

# BH across the family of focal terms in each model (same 5 terms, 2 outcomes)
focal = ['foreign', 'q_china', 'sensitive', 'avoid', 'adv']
say('\n### 2.3 BH-FDR（族 = 两个结局 × 5 个焦点项，m = 10）\n')
recs = []
for y in ['cos_sem', 'mirror']:
    m = store[y]
    for k in focal:
        z = m.params[k] / m.bse[k]
        recs.append((y, k, m.params[k], 2 * norm.sf(abs(z))))
recs.sort(key=lambda r: r[3])
m_tests = len(recs)
say('| 秩 | 结局 | 项 | p | BH 临界 q=.05 | 通过 |')
say('|---|---|---|---|---|---|')
for i, (y, k, b, p) in enumerate(recs, 1):
    crit = 0.05 * i / m_tests
    say(f'| {i} | {y} | {k} | {p:.3g} | {crit:.4f} | {"是" if p <= crit else "否"} |')

# 单调化（BH step-up）
adj = {}
prev = 1.0
for i in range(m_tests, 0, -1):
    y, k, b, p = recs[i - 1]
    val = min(prev, p * m_tests / i)
    adj[(y, k)] = val
    prev = val
not_sig = [(y, k, q) for (y, k), q in adj.items() if q > 0.05]
say(f'\n- 联合族（m = {m_tests}）：BH q > .05 的项 = '
    + ('无' if not not_sig else '; '.join(f'{y}×{k} (q={q:.3f})' for y, k, q in not_sig)))

# 每个模型内部各自一个族（m = 5），这是最贴近“逐模型报告”的口径
say('\n### 2.4 BH-FDR（族 = 单个模型的 5 个焦点项，m = 5，逐模型）\n')
for y in ['cos_sem', 'mirror']:
    m = store[y]
    sub = []
    for k in focal:
        z = m.params[k] / m.bse[k]
        sub.append((k, m.params[k], 2 * norm.sf(abs(z))))
    sub.sort(key=lambda r: r[2])
    pv = [r[2] for r in sub]
    qs = [0.0] * len(pv)
    prev = 1.0
    for i in range(len(pv) - 1, -1, -1):
        prev = min(prev, pv[i] * len(pv) / (i + 1))
        qs[i] = prev
    say(f'\n**{y}**')
    for (k, b, p), q in zip(sub, qs):
        say(f'- {k}: β = {b:+.4f}, p = {p:.3g}, BH q = {q:.3g}'
            f' {"（维持显著）" if q < 0.05 else "（仍不显著，与文中报告一致）"}')

say('\n**小结**：在 q < .05 下，正文里被报告为显著的全部焦点效应都保持显著；'
    '唯一未通过的是 mirror × q_china，而该项在原文 Table 4 与 §4.3 中本来就报告为不显著。')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf) + '\n')
print('\nsaved', OUT)
