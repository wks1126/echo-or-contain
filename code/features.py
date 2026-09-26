import sys, re, csv
sys.stdout.reconfigure(encoding='utf-8')
import jieba
import numpy as np

jieba.setLogLevel(20)

RE_PREFIX_Q = re.compile(r'^.*?记者[：:]\s*')
RE_PREFIX_A = re.compile(r'^[\u4e00-\u9fff·]{2,4}[：:]\s*')

# avoidance / formulaic markers in spokespersons' answers
AVOID = [
    '没有可以提供的信息', '没有可提供的信息', '没有相关信息', '不掌握', '不了解相关情况',
    '没有评论', '无可奉告', '没有进一步信息', '没有更多信息', '不予置评', '不掌握相关情况',
    '建议向中方主管部门询问', '建议向主管部门询问', '建议向有关方面询问', '请你向主管部门了解',
    '没有听说过', '不清楚', '不了解这个情况', '我没有这方面的信息', '没有必要回答',
    '你这个问题问得很具体，我没有更多信息', '具体问题建议向主管部门',
    '为时过早', '目前没有可发布的信息',
]

rows = list(csv.DictReader(open('corpus_qa.csv', encoding='utf-8')))

def tokens(s):
    return [w for w in list(jieba.cut(s))
            if w.strip() and not re.fullmatch(r'[\W_\s\d.，。、；：？！""''（）()《》〈〉·—…、/]+', w)
            and len(w.strip()) >= 1]

# --- agency classification ---
def agency_class(a):
    if not a:
        return 'OTHER'
    a = a.strip()
    CN = ('总台央视', '央视', '新华社', '中新社', '中国日报', '环球时报', '人民日报',
          '中国青年报', '北京日报', '北京青年报', '深圳卫视', '东方卫视', '澎湃新闻',
          '中国新闻社', '中央广播电视总台', '国际锐评', '观察者网', '新闻联播',
          '第一财经', '新京报', '解放日报', '中国之声', '人民网')
    if any(k in a for k in CN):
        return 'CN'
    WEST = ('路透社', '彭博社', '法新社', '美联社', 'BBC', '英国广播公司', 'CNN',
            '美国有线电视新闻网', '华盛顿邮报', '纽约时报', '华尔街日报', '时代周刊',
            '环球', '德国之声', '塔斯社', '共同社', '日本广播协会', 'NHK', '朝日新闻',
            '读卖新闻', '每日新闻', '韩国', '朝鲜日报', '中央日报', '日经', '日本经济新闻',
            '澳大利亚广播公司', '加拿大', '法国', '意大利', '西班牙', '天空新闻',
            '印度', '印度斯坦时报', '法广', '英国', '美国', '澳大利亚', '澳洲广播公司')
    if any(k in a for k in WEST):
        return 'WEST'
    return 'OTHER'

out = []
for r in rows:
    q = RE_PREFIX_Q.sub('', r['q_text']).replace('\n', ' ').strip()
    a = RE_PREFIX_A.sub('', r['a_text']).replace('\n', ' ').strip()
    tq = set(tokens(q)); ta = set(tokens(a))
    inter = tq & ta
    union = tq | ta
    jac = len(inter) / len(union) if union else 0.0
    # answer "mirrors" proportion of the question's lexical tokens
    mirror = len(inter) / len(tq) if tq else 0.0
    av = int(any(k in a for k in AVOID))
    out.append({
        'seq': r['seq'], 'date': r['date'], 'qa': r['qa'],
        'agency': r['agency'], 'aclass': agency_class(r['agency']),
        'is_followup': r['is_followup'],
        'nq': len(tq), 'na': len(ta),
        'jaccard': round(jac, 4), 'mirror': round(mirror, 4),
        'avoid': av,
    })

with open('features.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

print('rows', len(out))
from collections import Counter
print('aclass:', Counter(o['aclass'] for o in out))
print('avoid=1:', sum(o['avoid'] for o in out), 'of', len(out))
print('followup:', Counter(o['is_followup'] for o in out))
print('mean jaccard', round(np.mean([o['jaccard'] for o in out]), 4),
      'mean mirror', round(np.mean([o['mirror'] for o in out]), 4),
      'mean nq', round(np.mean([o['nq'] for o in out]), 1),
      'mean na', round(np.mean([o['na'] for o in out]), 1))
