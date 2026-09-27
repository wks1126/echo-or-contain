# -*- coding: utf-8 -*-
"""从发布代码中抽取全部特征词典，生成 dictionaries.json。

论文承诺公布 "feature dictionaries"。本脚本把它们从源码中抽出，
以免读者需要逐行阅读 compute_features.py / build_all.py / features.py。

用法：  python extract_dictionaries.py
输出：  dictionaries.json
"""
import ast
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, 'code')
if not os.path.isdir(CODE):
    CODE = HERE          # 库内为平铺布局（脚本与 build_all.py / features.py 同目录）

# 来自 build_all.py：中国指向 / 中文媒体 / 涉华敏感议题 / 问责式提问
BUILD_NAMES = {
    'CHINA': 'q_china —— 问句是否含中国指向词',
    'CNMEDIA': 'media —— 机构名单中的中文媒体（用于 media=cn）',
    'SENS': 'sensitive —— 涉华争议议题词典',
    'ACC': 'acc —— 问责式提问框架词典',
}
# 口径溯源：features_all.json 的 avoid 列由 build_all.py 里的 AVOID 正则生成，
# 因此 AVOID 必须从 build_all.py 抽（早期 features.py 里的 AVOID 列表口径不同，仅作遗留留档）。
BUILD_NAMES['AVOID'] = 'avoid —— 拒答/回避套语词典（生成 features_all.json 的正式口径）'
# 早期 features.py 里的拒答词表（口径与 features_all.json 不同）留档，注明勿用于复现
LEGACY = {'AVOID_legacy_features_py':
          ('features.py', 'AVOID',
           '⚠ 早期 features.py 拒答词表：口径与 features_all.json 不同，勿用于复现')}


def consts_from_assigns(path, want):
    tree = ast.parse(open(path, encoding='utf-8').read())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name) or tgt.id not in want:
                continue
            val = node.value
            if isinstance(val, ast.Call):          # re.compile(...)
                val = val.args[0]
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                out[tgt.id] = val.value
            elif isinstance(val, (ast.List, ast.Tuple, ast.Set)):
                out[tgt.id] = [e.value for e in val.elts
                               if isinstance(e, ast.Constant)]
    return out


dicts = {}
b = consts_from_assigns(os.path.join(CODE, 'build_all.py'), set(BUILD_NAMES))
for k, v in b.items():
    dicts[k] = {'role': BUILD_NAMES[k], 'source': 'code/build_all.py',
                'type': 'regex', 'pattern': v,
                'items': sorted(set(re.findall(r'[^()|\\]+', v)))}
for out_key, (src_file, var, role) in LEGACY.items():
    got = consts_from_assigns(os.path.join(CODE, src_file), {var})
    if var in got:
        dicts[out_key] = {'role': role, 'source': 'code/' + src_file,
                          'type': 'literal list', 'items': got[var]}

with open(os.path.join(HERE, 'dictionaries.json'), 'w', encoding='utf-8') as fh:
    json.dump(dicts, fh, ensure_ascii=False, indent=1)
print('wrote dictionaries.json with', len(dicts), 'dictionaries:',
      ', '.join(sorted(dicts)))
for k, v in dicts.items():
    print('  %-8s %s (%d items)' % (k, v['source'], len(v['items'])))
