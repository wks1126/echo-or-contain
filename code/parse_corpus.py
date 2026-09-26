import sys, re, glob, csv, json
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')
RE_QA = re.compile(r'^(.{1,40}?)记者[：:]')          # reporter question line
RE_NAME = re.compile(r'^([\u4e00-\u9fff]{2,4})[：:]')  # speaker-name start
FOOTER = ('相关附件', '打印', '分享', '网站地图', '【', '字号', '中文', '英文')
SPK = re.compile(r'^[\u4e00-\u9fff·]{2,4}[：:]$')

def lines_of(filepath):
    html = open(filepath, encoding='utf-8', errors='replace').read()
    soup = BeautifulSoup(html, 'lxml')
    wrappers = soup.find_all('div', class_='wrapper')
    if wrappers:
        wrappers = [w for w in wrappers if '记者' in w.get_text()]
    if wrappers:
        wrappers.sort(key=lambda w: w.get_text().count('记者'), reverse=True)
        txt = wrappers[0].get_text('\n')
    else:
        txt = soup.get_text('\n')
    meta = {'title': '', 'pubdate': ''}
    m = soup.find('meta', attrs={'name': 'ArticleTitle'})
    if m: meta['title'] = m.get('content', '')
    m = soup.find('meta', attrs={'name': 'PubDate'})
    if m: meta['pubdate'] = m.get('content', '')
    return meta, txt

def clean_agency(s):
    return s.strip().strip('《》「」').strip('、').strip()

def parse_file(filepath):
    meta, txt = lines_of(filepath)
    raw = [ln.strip() for ln in txt.split('\n')]
    # trim to body: from first reporter/speaker line to footer
    start = end = None
    for i, ln in enumerate(raw):
        if ln and RE_QA.match(ln):
            start = i; break
    if start is None:
        return meta, []
    for i in range(start, len(raw)):
        ln = raw[i]
        if ln.startswith(FOOTER) or (len(ln) <= 4 and ln in ('打印', '相关附件', '分享')):
            end = i; break
    if end is None:
        end = len(raw)
    body = raw[start:end]

    turns = []  # each: {kind:'Q'|'A', who, text}
    for ln in body:
        if not ln:
            continue
        mq = RE_QA.match(ln)
        if mq:
            turns.append({'kind': 'Q', 'who': clean_agency(mq.group(1)), 'text': ln})
            continue
        # else: speaker line or continuation
        mn = RE_NAME.match(ln)
        if turns and turns[-1]['kind'] == 'A' and not mn:
            turns[-1]['text'] += '\n' + ln
            continue
        if mn and turns and turns[-1]['kind'] == 'Q':
            turns.append({'kind': 'A', 'who': mn.group(1), 'text': ln})
            continue
        if turns and turns[-1]['kind'] == 'A':
            turns[-1]['text'] += '\n' + ln
    return meta, turns

rows = []
allfiles = sorted(glob.glob('corpus_html/art_*.html'),
                  key=lambda f: int(re.findall(r'\d+', f)[0]))
total_qa = 0
seq_names = {}
for fp in allfiles:
    seq = int(re.findall(r'\d+', fp)[0])
    meta, turns = parse_file(fp)
    qas = [t for t in turns if t['kind'] == 'Q']
    # pair each Q with the A(s) following until next Q
    qidx = [i for i, t in enumerate(turns) if t['kind'] == 'Q']
    for k, qi in enumerate(qidx):
        q = turns[qi]
        a_end = qidx[k + 1] if k + 1 < len(qidx) else len(turns)
        atexts = [t['text'] for t in turns[qi + 1:a_end] if t['kind'] == 'A']
        a_text = '\n'.join(atexts)
        if not a_text:
            continue
        is_fup = (q['text'].lstrip().startswith('追') or '追问' in q['text'][:12]
                  or ('（追' in q['text']))
        rows.append({
            'seq': seq, 'date': meta['pubdate'], 'title': meta['title'],
            'qa': k + 1, 'agency': q['who'], 'q_text': q['text'],
            'a_text': a_text, 'is_followup': is_fup
        })
        total_qa += 1

with open('corpus_qa.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seq', 'date', 'title', 'qa', 'agency',
                                      'is_followup', 'q_text', 'a_text'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print('files', len(allfiles), 'QA pairs', len(rows))
print('total char(q+a avg) sample:')
from collections import Counter
print('agencies sample:', Counter(r['agency'] for r in rows).most_common(12))
# sample rows
for r in rows[:3]:
    print('---', r['date'], r['agency'], 'fup', r['is_followup'])
    print('Q:', r['q_text'][:80])
    print('A:', r['a_text'][:120])
