import sys, re, json, random, time
sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
import jieba
from sentence_transformers import SentenceTransformer
import numpy as np

D = json.load(open('corpus_qa.json', encoding='utf-8'))
def date_of(fn):
    m = re.search(r't(\d{8})_', fn)
    return '%s-%s-%s' % (m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:]) if m else ''
for r in D:
    r['date'] = date_of(r['file'])

STOP = set('的了在是和我有就不人都一个上也很到说要对去会能这中下出你自他同可那好而于着并等'.split())
STOP |= set('啊吧呢吗嗯哦呀哈哎喔'.split())
PUN = set('，。？！、；：“”‘’（）《》【】—…·,.!?:;"\'()[] \u3000')
def content_terms(text):
    w = []
    for x in jieba.cut(text):
        x = x.strip()
        if not x or x in STOP: continue
        core = re.sub(r'\W', '', x)
        if len(core) < 2 or core.isdigit(): continue
        w.append(core)
    return set(w)

AVOID = re.compile(r'(没有可以提供的?信息|建议(向|你)?(中方)?(有关)?主管部门询问|不掌握|无可奉告|不予置评|'
                   r'没有可以提供|目前没有|无法确认|不了解有关情况|我们已经多次回答|此前已经多次|'
                   r'请关注|没有更多信息|信息请向|目前不掌握|不方便透露)')
CHINA = re.compile(r'(中国|中方|我国|中国政府|中国外交部)')
CNMEDIA = re.compile(r'(央视|新华社|中新社|人民日报|环球时报|中国日报|总台|CGTN|凤凰卫视|深圳卫视|'
                     r'北京广播|湖北广播|香港|大公文汇|观察者网|文汇报|环球网|澎湃|中国青年报|'
                     r'中国国际广播|经济日报|解放日报|南方日报|广东广播|参考消息|法广)')
SENS = re.compile(r'(台湾|西藏|新疆|维吾尔|香港|南海|钓鱼岛|人权|法轮功|天安门|关税|芯片|半导体|'
                  r'稀土|脱钩|间谍|监控|病毒溯源|实验室泄漏|战狼|军事(演习|基地|部署|活动)?|导弹|'
                  r'核(潜艇|武器)?|对台军售|新疆棉|强迫劳动|网络攻击|数据安全|供应链|产业链|气球|'
                  r'窃取|制裁|加征)')
ACC = re.compile(r'(为何|为什么|凭什么|怎么会|如何解释|是否承认|应不应|该不该|有没有计划|'
                 r'是否愿意承诺|是否担心|要作何|作何回应)')

print('loading model...', flush=True)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
def enc(xs, tag):
    out = []
    bs = 64
    for s in range(0, len(xs), bs):
        out.append(model.encode(xs[s:s+bs], normalize_embeddings=True,
                                batch_size=bs, show_progress_bar=False))
        if (s//bs) % 8 == 0:
            print('%s done %d/%d' % (tag, min(s+bs, len(xs)), len(xs)), flush=True)
    return np.concatenate(out)
qs=[r['q'] for r in D]; ans=[r['a'] for r in D]
print('encoding q...', flush=True); Qv=enc(qs, 'q')
print('encoding a...', flush=True); Av=enc(ans, 'a')
print('saving emb_all.npz...', flush=True); np.savez('emb_all.npz', Q=Qv, A=Av)

def cos(a,b): return float((a*b).sum())
rows=[]
for i,r in enumerate(D):
    tq,ta=content_terms(r['q']),content_terms(r['a']); inter=tq&ta
    jac=len(inter)/len(tq|ta) if (tq|ta) else 0.0
    mirror=len(inter)/len(tq) if tq else 0.0
    nq=len(re.findall(r'[？?]', r['q']))
    rows.append({'id':i,'file':r['file'],'date':r['date'],'who':r['who'],'rep':r['rep'],
        'media':'cn' if CNMEDIA.search(r['rep']) else 'foreign',
        'len_q':len(r['q']),'len_a':len(r['a']),
        'cos_sem':cos(Qv[i],Av[i]),'jac':jac,'mirror':mirror,
        'avoid':bool(AVOID.search(r['a'])),'q_china':bool(CHINA.search(r['q'])),
        'sensitive':bool(SENS.search(r['q'])),'acc':bool(ACC.search(r['q'])),
        'nq':nq,'multiq':bool(nq>=2),
        'window':'hist' if r['date']<'2025-09' else 'recent'})
for c in ['avoid','q_china','sensitive','acc','multiq']:
    for r in rows: r[c]=int(r[c])
json.dump(rows, open('features_all.json','w',encoding='utf-8'), ensure_ascii=False)

# within-meeting shuffled baseline
from collections import defaultdict
g=defaultdict(list)
for i,r in enumerate(rows): g[r['file']].append(i)
shuf_cos=[]; shuf_mir=[]
for f,idxs in g.items():
    if len(idxs)<2: continue
    perm=list(idxs); random.shuffle(perm)
    for k,i in enumerate(idxs):
        j=perm[k]
        shuf_cos.append(cos(Qv[i],Av[j]))
        tq=content_terms(rows[i]['q']); ta=content_terms(rows[perm[k]]['a'])
        shuf_mir.append(len(tq&ta)/len(tq) if tq else 0.0)
json.dump({'obs_cos':[r['cos_sem'] for r in rows],'shuf_cos':shuf_cos,
           'obs_mirror':[r['mirror'] for r in rows],'shuf_mirror':shuf_mir},
          open('baseline_all.json','w'))
import statistics as st
print('rows',len(rows),'obs_cos %.3f vs shuf %.3f' % (st.mean([r['cos_sem'] for r in rows]), st.mean(shuf_cos)))
print('mirror %.3f vs shuf %.3f' % (st.mean([r['mirror'] for r in rows]), st.mean(shuf_mir)))
print('windows:', {w:sum(1 for r in rows if r['window']==w) for w in ['hist','recent']})
print('saved features_all.json baseline_all.json emb_all.npz')
