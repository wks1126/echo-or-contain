import sys, json, random, csv, re
sys.stdout.reconfigure(encoding='utf-8')
random.seed(2026)
R = json.load(open('features_all.json', encoding='utf-8'))
D = json.load(open('corpus_qa.json', encoding='utf-8'))
assert len(R) == len(D)
def date_of(fn):
    m = re.search(r't(\d{8})_', fn)
    return '%s-%s-%s' % (m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:]) if m else ''

# stratified sample: ~balance media x sensitive, ensure avoid included
pools = {}
for i in range(len(R)):
    key = (R[i]['media'], R[i]['sensitive'], int(bool(R[i]['avoid'])))
    pools.setdefault(key, []).append(i)
target_per = 150 // 12 + 1
sel = []
for key, idxs in pools.items():
    k = min(target_per, len(idxs))
    sel.extend(random.sample(idxs, k))
# cap to 150 balanced-ish, guarantee all avoid in
sel = sorted(set(sel))
avoid_ids = [i for i in sel if R[i]['avoid']]
# ensure enough avoid
extra_avoid = [i for i in range(len(R)) if R[i]['avoid'] and i not in sel]
sel = sel + random.sample(extra_avoid, min(len(extra_avoid), max(0, 40 - len(avoid_ids))))
sel = sorted(set(sel))
print('sampled', len(sel), '| avoid in sample', sum(1 for i in sel if R[i]['avoid']))

cols = ['id','date','file','who','media','sensitive','avoid','rep','cos_sem','mirror','Q','A',
        'C1_echo','C1_function','C2_echo','C2_function']
with open('coding_validation_samples.csv','w',encoding='utf-8-sig',newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for i in sel:
        w.writerow({'id':i,'date':date_of(D[i]['file']),'file':D[i]['file'],'who':R[i]['who'],
                    'media':R[i]['media'],'sensitive':R[i]['sensitive'],'avoid':R[i]['avoid'],
                    'rep':R[i]['rep'],'cos_sem':round(R[i]['cos_sem'],3),'mirror':round(R[i]['mirror'],3),
                    'Q':D[i]['q'],'A':D[i]['a'],'C1_echo':'','C1_function':'','C2_echo':'','C2_function':''})
manual = """# 双编码手册（供两位独立编码者）
任务1 —— 回答是否沿用提问的词汇/框架（construct-validity for echo/mirror）
  对每条 Q-A 判断回答在多大程度上"接住"了提问的用词与说法：
  2 = 明显沿用：直接复用提问中的核心词/机构名/表述（如"上合组织…特别代表"照用）
  1 = 部分沿用：用了提问的实体或部分表述，但主体换成官方口径
  0 = 不沿用：几乎不复用提问用词，改用官方套语/重述
  判定依据只限"用词层面是否重复/吸收"，不看同意与否。

任务2 —— 回答的功能（呼应是否提供实质内容）
  A = 提供具体信息/证实   B = 部分/间接回应   C = 拒绝提供（"没有信息/无可奉告/问主管部门"）
  D = 纠正或否定提问预设   E = 反问/施压/质疑提问方   F = 建议或表达立场（无实质新信息）

任务3 —— 回避复标（仅作一致性复核）：该回答是否属于"拒答/回避提供信息"？是/否。

注：media/sensitive/avoid 为机器标注供参考；请独立判断，勿参照以免趋同。完成后我计算
echo(2/1/0)与 mirror、cos_sem 的 Spearman（效度），以及 C1/C2 在 echo 与 function 上的 Cohen's kappa。
"""
open('coding_manual.md','w',encoding='utf-8').write(manual)
print('saved coding_validation_samples.csv + coding_manual.md')
