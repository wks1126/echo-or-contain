import os, re, sys, time, csv
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

PREQ = re.compile(r'^.*?记者[：:]\s*'); PREA = re.compile(r'^[\u4e00-\u9fff·]{2,4}[：:]\s*')
rows = list(csv.DictReader(open('corpus_qa.csv', encoding='utf-8')))
qs = [PREQ.sub('', r['q_text']).replace('\n', ' ').strip() for r in rows]
as_ = [PREA.sub('', r['a_text']).replace('\n', ' ').strip() for r in rows]

cands = [
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'moka-ai/m3e-base',
    'DMetaSoul/sbert-chinese-general-v2',
    'shibing624/text2vec-base-chinese',
]
from sentence_transformers import SentenceTransformer
done = None
for name in cands:
    try:
        t0 = time.time()
        print('trying', name, flush=True)
        m = SentenceTransformer(name)
        print('loaded', name, '%.1fs' % (time.time() - t0), flush=True)
        qe = m.encode(qs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        ae = m.encode(as_, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        tag = name.replace('/', '__').replace('-', '_')
        np.savez('embeddings_%s.npz' % tag, q=qe, a=ae)
        print('SAVED embeddings_%s.npz' % tag, qe.shape, flush=True)
        done = name
        break
    except Exception as e:
        print('FAIL', name, repr(e)[:200], flush=True)
print('RESULT_DONE=', done, flush=True)
