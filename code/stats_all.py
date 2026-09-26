import sys, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats

R = json.load(open('features_all.json', encoding='utf-8'))
B = json.load(open('baseline_all.json'))
F = pd.DataFrame(R)
for c in ['avoid','q_china','sensitive','acc','multiq']:
    F[c]=F[c].astype(int)
F['adv']=F.acc|F.multiq
oc=np.array(B['obs_cos']); sc=np.array(B['shuf_cos'])
om=np.array(B['obs_mirror']); sm=np.array(B['shuf_mirror'])

def cohend(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); na,nb=len(a),len(b)
    sp=math.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    return (a.mean()-b.mean())/sp
def two(a,b):
    if len(a)>1 and len(b)>1:
        t,p=stats.ttest_ind(a,b,equal_var=False); return t,p,cohend(a,b)
    return (None,None,None)
L=[]
def w(s): L.append(s); print(s)

w('===== FULL 4005: observed vs shuffled =====')
for name,o,s in [('cos',oc,sc),('mirror',om,sm)]:
    t,p,d=two(o,s)
    w(f'  {name}: obs={o.mean():.3f} vs shuf={s.mean():.3f}  t={t:.1f} p={p:.1e} d={d:.2f}')

w('\n===== window robustness (⑥) =====')
for win in ['hist','recent']:
    g=F[F.window==win]
    t,p,d=two(g.cos_sem, oc if False else g.cos_sem)  # placeholder skip
    cn=g[g.media=='cn']; fo=g[g.media=='foreign']
    t2,p2,d2=two(cn.cos_sem, fo.cos_sem)
    w(f'  window={win:6s} n={len(g):4d} cos={g.cos_sem.mean():.3f} mirror={g.mirror.mean():.3f} '
      f'| cn n={len(cn)} cos={cn.cos_sem.mean():.3f} vs foreign n={len(fo)} cos={fo.cos_sem.mean():.3f} '
      f'| media-t={t2:.1f} p={p2:.1e} d={d2:.2f} | avoid% cn={cn.avoid.mean()*100:.1f} fo={fo.avoid.mean()*100:.1f}')

w('\n===== speaker robustness (⑥) =====')
for who in sorted(F.who.unique()):
    g=F[F.who==who]; cn=g[g.media=='cn']; fo=g[g.media=='foreign']
    t2,p2,d2=two(cn.cos_sem,fo.cos_sem)
    w(f'  {who:4s} n={len(g):4d} cos={g.cos_sem.mean():.3f} | cn {cn.cos_sem.mean():.3f}({len(cn)}) vs '
      f'foreign {fo.cos_sem.mean():.3f}({len(fo)}) t={t2:.1f} d={d2:.2f} | avoid% cn={cn.avoid.mean()*100:.1f} fo={fo.avoid.mean()*100:.1f}')

w('\n===== groups (full) =====')
for col,lab in [('media',['cn','foreign']),('sensitive',[0,1]),('adv',[0,1]),('avoid',[0,1])]:
    a=F[F[col]==lab[0]]; b=F[F[col]==lab[1]]
    t,p,d=two(a.cos_sem,b.cos_sem); tm,pm,dm=two(a.mirror,b.mirror)
    w(f'  {col}: {lab[0]}(n={len(a)}) cos={a.cos_sem.mean():.3f} mirror={a.mirror.mean():.3f} | '
      f'{lab[1]}(n={len(b)}) cos={b.cos_sem.mean():.3f} mirror={b.mirror.mean():.3f} | cos d={d:+.2f} p={p:.1e} | mirror d={dm:+.2f}')

open('stats_all_report.txt','w',encoding='utf-8').write('\n'.join(L))
print('\nsaved stats_all_report.txt')
