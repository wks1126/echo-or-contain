import sys, re, json, random, time
sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
import numpy as np
import jieba
from collections import defaultdict

R = json.load(open('features_all.json', encoding='utf-8'))
D = json.load(open('corpus_qa.json', encoding='utf-8'))
assert len(R) == len(D)
Z = np.load('emb_all.npz'); Qv, Av = Z['Q'], Z['A']
print('loaded', len(R), 'Qv', Qv.shape)

STOP = set('的了在是和我有就不人都一个上也很到说要对去会能这中下出你自他同可那好而于着并等'.split())
STOP |= set('啊吧呢吗嗯哦呀哈哎喔'.split())
def content_terms(text):
    w = []
    for x in jieba.cut(text):
        x = x.strip()
        if not x or x in STOP: continue
        core = re.sub(r'\W', '', x)
        if len(core) < 2 or core.isdigit(): continue
        w.append(core)
    return set(w)

def cos(a,b): return float((a*b).sum())

# per-row term sets aligned to R order
print('tokenizing...', flush=True)
tq = [content_terms(r['q']) for r in D]
ta = [content_terms(r['a']) for r in D]
print('tokenizing done', flush=True)

g = defaultdict(list)
for i, r in enumerate(R): g[r['file']].append(i)
shuf_cos=[]; shuf_mir=[]
for f, idxs in g.items():
    if len(idxs) < 2: continue
    perm = list(idxs); random.shuffle(perm)
    for k, i in enumerate(idxs):
        j = perm[k]
        shuf_cos.append(cos(Qv[i], Av[j]))
        inter = tq[i] & ta[j]
        shuf_mir.append(len(inter)/len(tq[i]) if tq[i] else 0.0)

json.dump({'obs_cos':[r['cos_sem'] for r in R], 'shuf_cos':shuf_cos,
           'obs_mirror':[r['mirror'] for r in R], 'shuf_mirror':shuf_mir},
          open('baseline_all.json','w'))
import statistics as st
oc=st.mean([r['cos_sem'] for r in R]); sc=st.mean(shuf_cos)
om=st.mean([r['mirror'] for r in R]); sm=st.mean(shuf_mir)
print('FULL n=%d  cos obs=%.3f vs shuf=%.3f | mirror obs=%.3f vs shuf=%.3f'
      % (len(R), oc, sc, om, sm))
print('saved baseline_all.json')
