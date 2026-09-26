import sys, json, random, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict
from scipy import stats

R = json.load(open('features_all.json', encoding='utf-8'))
Z = np.load('emb_all.npz'); Qv, Av = Z['Q'], Z['A']
def cos(a,b): return float((a*b).sum())
obs = [r['cos_sem'] for r in R]
random.seed(1)

def paired_shuffle(key_fn, self_excl=True):
    """within file x key, permute answers among same-group pairs (exclude self)."""
    g = defaultdict(list)
    for i, r in enumerate(R):
        g[(r['file'], key_fn(r))].append(i)
    shuf = []
    for (f,k), idx in g.items():
        if len(idx) < 2: continue
        arr = np.array(idx)
        for i in idx:
            pool = arr[arr != i]          # exclude self (same Q-A) to avoid trivial high cos
            if len(pool) == 0: continue
            j = int(np.random.choice(pool))
            shuf.append(cos(Qv[i], Av[j]))
    return np.array(shuf)

obs_arr = np.array(obs)
def cohend(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); na,nb=len(a),len(b)
    sp=math.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    return (a.mean()-b.mean())/sp

print('n obs', len(obs_arr))
for name, fn in [('same-MEDIA within meeting', lambda r: r['media']),
                 ('same-SENSITIVE within meeting', lambda r: r['sensitive']),
                 ('same-MEDIA x SENSITIVE within meeting', lambda r: (r['media'], r['sensitive']))]:
    s = paired_shuffle(fn)
    t, p = stats.ttest_ind(obs_arr, s, equal_var=False)
    print('%-34s n_shuf=%5d  obs=%.3f vs %.3f  t=%.1f p=%.1e d=%.2f'
          % (name, len(s), obs_arr.mean(), s.mean(), t, p, cohend(obs_arr, s)))
