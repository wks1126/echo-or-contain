import sys, re, json, random, time
sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)

import jieba, jieba.posseg as pseg
from sentence_transformers import SentenceTransformer

# ---------- load data ----------
D = json.load(open('corpus_qa.json', encoding='utf-8'))
def date_of(fn):
    m = re.search(r't(\d{8})_', fn)
    return m.group(1)[:4] + '-' + m.group(1)[4:6] + '-' + m.group(1)[6:] if m else ''
for r in D:
    r['date'] = date_of(r['file'])

# ---------- stopwords ----------
STOP = set('的了在是和我有就不人都一个上也很到说要对去会能这中下出你自他同可那好而于着并等'.split())
STOP |= set('啊吧呢吗嗯哦呀哈哎喔'.split())
PUN = set('，。？！、；：“”‘’（）《》【】—…·,.!?:;""\'\'()[] \u3000')

def content_terms(text):
    words = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w or w in STOP:
            continue
        core = re.sub(r'\W', '', w)  # keep CJK + alnum
        if len(core) < 2:
            continue
        if core.isdigit():
            continue
        words.append(core)
    return set(words)

# ---------- avoidance / topic / media tags ----------
AVOID = re.compile(r'(没有可以提供的?信息|建议(向|你)?(中方)?(有关)?主管部门询问|不掌握|无可奉告|不予置评|没有可以提供|目前没有|无法确认|不了解有关情况|我们已经多次回答|此前已经多次|请关注|没有更多信息|信息请向|目前不掌握)')
CHINA = re.compile(r'(中国|中方|我国|中国政府|中国外交部)')
CNMEDIA = re.compile(r'(央视|新华社|中新社|人民日报|环球时报|中国日报|总台|CGTN|凤凰卫视|深圳卫视|北京广播|湖北广播|香港|大公文汇|观察者网|文汇报|环球网|澎湃|中国青年报|中国国际广播|经济日报|解放日报|南方日报|广东广播|参考消息|人民日报|法广)')
def media_tag(rep):
    return 'cn' if CNMEDIA.search(rep) else 'foreign'

# ---------- embed ----------
print('loading model...', flush=True)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
def cos(a, b):
    return float((a * b).sum())

# encode in batches for speed
batch = 128
def encode(texts):
    return model.encode(texts, normalize_embeddings=True, batch_size=batch,
                        show_progress_bar=False)

qs = [r['q'] for r in D]
as_ = [r['a'] for r in D]
print('encoding questions...', flush=True)
Qv = encode(qs)
print('encoding answers...', flush=True)
Av = encode(as_)

# ---------- compute per-pair features ----------
rows = []
all_qterms, all_aterms = [], []
for i, r in enumerate(D):
    tq, ta = content_terms(r['q']), content_terms(r['a'])
    all_qterms.append(tq); all_aterms.append(ta)
    inter = tq & ta
    jac = len(inter) / len(tq | ta) if (tq | ta) else 0.0
    mirror = len(inter) / len(tq) if tq else 0.0   # share of Q terms echoed in A
    rows.append({
        'id': i, 'file': r['file'], 'date': r['date'],
        'who': r['who'], 'rep': r['rep'],
        'media': media_tag(r['rep']),
        'len_q': len(r['q']), 'len_a': len(r['a']),
        'cos_sem': cos(Qv[i], Av[i]),
        'jac': jac, 'mirror': mirror,
        'avoid': bool(AVOID.search(r['a'])),
        'q_china': bool(CHINA.search(r['q'])),
        'a_china': bool(CHINA.search(r['a'])),
    })
json.dump(rows, open('features.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------- within-meeting shuffled baseline (one permutation) ----------
# group indices by file
from collections import defaultdict
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[r['file']].append(i)

shuf_cos, shuf_mirror = [], []
for f, idxs in groups.items():
    if len(idxs) < 2:
        continue
    As = [Av[i] for i in idxs]
    perm = list(range(len(idxs)))
    random.shuffle(perm)  # remap answer j -> perm[j]
    for k, i in enumerate(idxs):
        j = idxs[perm[k]]
        shuf_cos.append(cos(Qv[i], Av[j]))
        tq = all_qterms[i]; ta = all_aterms[perm[k]]
        inter = tq & ta
        shuf_mirror.append(len(inter) / len(tq) if tq else 0.0)

obs_cos = [r['cos_sem'] for r in rows]
obs_mirror = [r['mirror'] for r in rows]
import statistics as st
def sm(x): return (sum(x) / len(x), (sum((v - sum(x)/len(x))**2 for v in x)/(len(x)-1))**0.5)
mc, sc = sm(obs_cos); bc, sbc = sm(shuf_cos)
mm, smm = sm(obs_mirror); bm, _ = sm(shuf_mirror)
print('semantic cosine  observed mean=%.4f sd=%.4f | shuffled mean=%.4f' % (mc, sc, bc))
print('lexical mirror   observed mean=%.4f | shuffled mean=%.4f' % (mm, bm))
print('n_obs', len(obs_cos), 'n_shuf', len(shuf_cos))
json.dump({'obs_cos': obs_cos, 'shuf_cos': shuf_cos, 'obs_mirror': obs_mirror,
           'shuf_mirror': shuf_mirror}, open('baseline.json', 'w'))
print('saved features.json & baseline.json')
