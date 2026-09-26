# -*- coding: utf-8 -*-
"""③-round opts #2,#3,#5,#6,#7,#8 metrics."""
import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

R = json.load(open('features_all.json', encoding='utf-8'))
D = json.load(open('corpus_qa.json', encoding='utf-8'))
F = pd.DataFrame(R)
for c in ['avoid','sensitive','q_china','acc','multiq']: F[c]=F[c].astype(int)
F['q']=[r['q'] for r in D]; F['a']=[r['a'] for r in D]

def prt(*a):
    s=' '.join(str(x) for x in a); print(s); buf.append(s)
buf=[]

# ---------------- (2) avoid dict precision/recall vs human function=C ----------------
try:
    import csv
    samp=list(csv.DictReader(open('coding_validation_samples.csv',encoding='utf-8-sig')))
    gold={}
    for r in samp:
        g = 1 if (r['C1_function']=='C' and r['C2_function']=='C') else (0 if (r['C1_function']!='C' and r['C2_function']!='C') else None)
        gold[r['id']]=g
    ids=[k for k,v in gold.items() if v is not None]
    y=[gold[i] for i in ids]
    # machine avoid by id from features_all
    av={str(r['id']):r['avoid'] for r in R}
    p=[av.get(i,0) for i in ids]
    prt('\n### (2) avoid-dictionary diagnostics (gold = both coders function==C, n=%d)' % len(ids))
    prt('  precision=%.2f  recall=%.2f  F1=%.2f' % (
        precision_score(y,p), recall_score(y,p), f1_score(y,p)))
    prt('  gold refusals=%d of %d (%.0f%%)' % (sum(y),len(y),100*np.mean(y)))
except Exception as e:
    prt('(2) err', e)

# ---------------- (3) evaluative-word echo ----------------
NEG=re.compile(r'(非法|侵略|侵害|干涉|霸凌|无端|恶意|污名化|制裁|胁迫|攻击|危险|威胁|侵犯|错误|不公|歧视|破坏|单边|保护主义|长臂管辖|抹黑|打压|围堵|遏(制|压)|分裂|操弄|干涉内政|威权|谎(言|报))')
POS=re.compile(r'(欢迎|赞赏|支持|共识|共赢|互信|友好|尊重|平等|合作|机遇|对话|和平|发展|稳定)')
def ev_echo(q,a):
    # whether A reuses an evaluative word that appeared in Q (count words present both)
    nq=set(NEG.findall(q)); na=set(NEG.findall(a))
    pq=set(POS.findall(q)); pa=set(POS.findall(a))
    return int(bool(nq&na) or bool(pq&pa))
F['ev_echo']=[ev_echo(q,a) for q,a in zip(F.q,F.a)]
prt('\n### (3) evaluative-word echo')
prt('  evaluative echo present in %.1f%% (n=%d)' % (100*F.ev_echo.mean(), F.ev_echo.sum()))
for who in ['cn','foreign']:
    g=F[F.media==who]
    prt('  media=%-7s eval-echo=%.2f  avoid%%=%.1f' % (who, g.ev_echo.mean(), g.avoid.mean()*100))

# ---------------- (5) issue categories ----------------
ISSUE = [
 ('Taiwan', re.compile(r'(台湾|台独|赖清德|蔡英文|金门|澎湖)')),
 ('South_China_Sea/PH', re.compile(r'(南海|菲律宾|仁爱礁|黄岩岛|马尼拉|中菲)')),
 ('Xinjiang/Tibet/HR', re.compile(r'(新疆|维吾尔|西藏|人权|强迫劳动|法轮功|天安门)')),
 ('US_relations', re.compile(r'(美国|美方|特朗普|拜登|关税|中美|贸易战|芯片|关税|制裁.{0,6}美国|对华)')),
 ('Russia/Ukraine', re.compile(r'(俄罗斯|俄方|乌克兰|普京|俄乌|克里米亚|北约)')),
 ('Korea/Japan/DPRK', re.compile(r'(日本|韩国|朝鲜|韩国|尹锡|石破|高市|岸田|萨德|慰安妇|福岛)')),
 ('Trade/Tech/Global', re.compile(r'(WTO|加征|关税|进出口|半导体|芯片|5G|华为|中兴|反补贴|双反|产能)')),
]
# assign a primary issue by first match else Other
def issue(q):
    for name,pat in ISSUE:
        if pat.search(q): return name
    return 'Other'
F['issue']=[issue(q) for q in F.q]
prt('\n### (5) per-issue alignment / containment')
prt('  issue       n    cos   mirror avoid%  (cn%)')
for name,g in F.groupby('issue'):
    prt('  %-20s %5d %.3f  %.3f  %4.1f  (%.0f%%)' % (
        name, len(g), g.cos_sem.mean(), g.mirror.mean(),
        g.avoid.mean()*100, 100*(g.media=='cn').mean()))

# ---------------- (6) reporter country / camp ----------------
CAMP = [
 ('CN', re.compile(r'(央视|新华社|中新社|人民日报|环球时报|中国日报|CGTN|总台|凤凰|深圳卫视|北京|香港|大公文汇|澎湃|环球网|中国青年报|国际|经济日报|解放日报|参考消息|文汇报|湖北|广东|南方)')),
 ('US', re.compile(r'(美联社|美国有线|CNN|纽约时报|华尔街|华盛顿|彭博社|路透社|美国之音|福克斯|今日美国|全国广播|NBC|CBS)')),
 ('EU/UK', re.compile(r'(法新社|英国广播|BBC|路透社|德国之声|泰晤士|每日电讯|卫报|法广|世界报|西班牙|意大利|瑞典|挪威|芬兰|丹麦|瑞士|爱尔兰|荷兰|欧洲|葡萄牙|奥地利)')),
 ('Russia', re.compile(r'(今日俄罗斯|俄新社|塔斯社|国际文传|俄罗斯卫星)')),
 ('Japan/KR/AU', re.compile(r'(共同社|读卖|朝日|每日新闻|NHK|日本|韩国|韩联社|首尔|澳大利亚|澳广|悉尼|新西兰)')),
 ('MiddleEast/SA', re.compile(r'(中阿|半岛|阿纳多卢|伊朗|以色列|伊拉克|土耳其|沙特|阿联酋|卡塔尔|埃及|巴勒斯坦|叙利亚|黎巴嫩|也门|巴基斯坦|阿富汗|塔利班)')),
 ('S/SE-Asia', re.compile(r'(印度|印度报业|尼西亚|新加坡|马来西亚|泰国|越南|菲律宾|缅甸|柬埔寨|老挝|孟加拉|斯里兰卡|南亚)')),
 ('Africa/LA/UN', re.compile(r'(非洲|南非|尼日利亚|肯尼亚|埃及|拉美|巴西|阿根廷|墨西哥|古巴|联合国)')),
]
def camp(rep):
    for name,pat in CAMP:
        if pat.search(rep): return name
    return 'Other/unknown'
F['camp']=[camp(rp) for rp in F.rep]
prt('\n### (6) alignment by reporter camp')
prt('  camp          n    cos   mirror avoid%  cn-part%')
for name,g in F.groupby('camp'):
    prt('  %-14s %5d %.3f  %.3f  %4.1f' % (name, len(g), g.cos_sem.mean(),
        g.mirror.mean(), g.avoid.mean()*100))

# ---------------- (7) meeting-order trajectory ----------------
F['order']=F.groupby('file').cumcount()
F['frac']=F.groupby('file')['order'].transform(lambda x: x/(len(x)-1) if len(x)>1 else 0)
half = F.copy(); half['pos']=['first-half' if f<=.5 else 'second-half' for f in half['frac']]
prt('\n### (7) position within meeting')
for pos,g in half.groupby('pos'):
    prt('  %-11s n=%5d cos=%.3f mirror=%.3f avoid%%=%.1f' % (
        pos, len(g), g.cos_sem.mean(), g.mirror.mean(), g.avoid.mean()*100))
try:
    from scipy.stats import pearsonr, spearmanr
    prt('  corr(order, cos)=%.2f  corr(order, avoid)=%.2f' % (
        pearsonr(F.order, F.cos_sem)[0], pearsonr(F.order, F.avoid)[0]))
except Exception as e: prt('(7)',e)

# ---------------- (8) politeness / directness ----------------
POL=re.compile(r'(谢谢|感谢|感谢你的|谢谢你|欢迎(你的)?(提问|问题)|感谢关注|很高兴|很感谢)')
SOFT=re.compile(r'(我们(注意到|认为|希望|建议)|(愿|愿意)|我们理解|或许|可能|相关|有关方面|视情|根据|按照)')
DIR=re.compile(r'(坚决|强烈|严正|必须|将(采取|坚决)|严正交涉|郑重|绝不能|反对|谴责|敦促|要求|警告)')
for name,pat in [('politeness',POL),('softener',SOFT),('directive',DIR)]:
    F['m8_'+name]=[bool(pat.search(a)) for a in F.a]
prt('\n### (8) politeness / softeners / directives')
for m in ['m8_politeness','m8_softener','m8_directive']:
    a=F[F[m]==0]; b=F[F[m]==1]
    prt('  %-16s yes%%=%.1f | cos yes=%.3f/no=%.3f mirror yes=%.3f/no=%.3f avoid%% yes=%.1f/no=%.1f' % (
        m, 100*b[m].mean() if False else 100*(F[m].mean()),
        b.cos_sem.mean(), a.cos_sem.mean(), b.mirror.mean(), a.mirror.mean(),
        b.avoid.mean()*100, a.avoid.mean()*100))

open('m3b_report.txt','w',encoding='utf-8').write('\n'.join(buf))
print('saved m3b_report.txt')
