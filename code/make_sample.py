import sys, re, csv
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
feats = list(csv.DictReader(open('features.csv', encoding='utf-8')))
src = list(csv.DictReader(open('corpus_qa.csv', encoding='utf-8')))
z = np.load('embeddings.npz'); q = z['q']; a = z['a']
n = len(feats)
cos = np.einsum('ij,ij->i', q, a)
aclass = np.array([f['aclass'] for f in feats])
avoid = np.array([int(f['avoid']) for f in feats])
mirror = np.array([float(f['mirror']) for f in feats])

def clean(s, prefix_re): return re.sub(prefix_re, '', s).replace('\n', ' ').strip()
PREQ = re.compile(r'^.*?记者[：:]\s*'); PREA = re.compile(r'^[\u4e00-\u9fff·]{2,4}[：:]\s*')

rng = np.random.default_rng(42)
idx = []
per = 50
for cl in ['CN', 'WEST', 'OTHER']:
    pool = [i for i in range(n) if aclass[i] == cl]
    # over-sample avoid within class when available
    av = [i for i in pool if avoid[i] == 1]
    rng.shuffle(av)
    rng.shuffle(pool)
    take = av[:int(per * 0.3)]
    need = per - len(take)
    rest = [i for i in pool if i not in set(take)]
    take += rest[:need]
    idx.extend(take)
# fill to ~150 evenly, ensure ~>=150
if len(idx) < 150:
    extra = [i for i in range(n) if i not in set(idx)]
    rng.shuffle(extra); idx.extend(extra[:150 - len(idx)])

out = []
for k, i in enumerate(idx[:150]):
    out.append({
        'sample_id': k + 1, 'date': src[i]['date'], 'agency': src[i]['agency'],
        'aclass': feats[i]['aclass'], 'avoid': feats[i]['avoid'],
        'is_followup': src[i]['is_followup'],
        'cos': round(float(cos[i]), 3), 'mirror': round(mirror[i], 3),
        'q_text': clean(src[i]['q_text'], PREQ),
        'a_text': clean(src[i]['a_text'], PREA),
    })
with open('annotation_sample.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('samples', len(out))
from collections import Counter
print('aclass', Counter(o['aclass'] for o in out))
print('avoid=1', sum(1 for o in out if o['avoid'] == '1'))
