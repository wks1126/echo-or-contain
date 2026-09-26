import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
F = json.load(open('features_all.json', encoding='utf-8'))
for r in F:
    r['adv'] = int(bool(r['acc']) or bool(r['multiq']))
def sub(col, a):
    x = [r for r in F if r[col] == a]
    return (len(x), round(np.mean([r['cos_sem'] for r in x]), 3),
            round(np.mean([r['mirror'] for r in x]), 3),
            round(np.mean([r['avoid'] for r in x]) * 100, 1),
            round(np.mean([r['sensitive'] for r in x]) * 100, 1))
print('n total', len(F))
for col in ['media', 'sensitive', 'avoid', 'adv', 'q_china']:
    vals = (['cn', 'foreign'] if col == 'media' else [0, 1])
    for a in vals:
        nn, c, m, av, se = sub(col, a)
        print('%-9s %-8s n=%5d cos=%.3f mirror=%.3f avoid%%=%.1f sens%%=%.1f'
              % (col, a, nn, c, m, av, se))
# window & speaker quick
for r in F: pass
for win in ['hist', 'recent']:
    x = [r for r in F if r['window'] == win]
    cn = [r for r in x if r['media'] == 'cn']; fo = [r for r in x if r['media'] == 'foreign']
    print('win %-6s n=%4d cos=%.3f | cn cos=%.3f(n=%d) fo cos=%.3f(n=%d) avoid%% cn=%.1f fo=%.1f'
          % (win, len(x), np.mean([r['cos_sem'] for r in x]),
             np.mean([r['cos_sem'] for r in cn]), len(cn),
             np.mean([r['cos_sem'] for r in fo]), len(fo),
             np.mean([r['avoid'] for r in cn]) * 100, np.mean([r['avoid'] for r in fo]) * 100))
for who in sorted(set(r['who'] for r in F)):
    x = [r for r in F if r['who'] == who]
    cn = [r for r in x if r['media'] == 'cn']; fo = [r for r in x if r['media'] == 'foreign']
    print('who %-4s n=%4d cos=%.3f | cn=%.3f fo=%.3f | avoid%% cn=%.1f fo=%.1f'
          % (who, len(x), np.mean([r['cos_sem'] for r in x]),
             np.mean([r['cos_sem'] for r in cn]), np.mean([r['cos_sem'] for r in fo]),
             np.mean([r['avoid'] for r in cn]) * 100, np.mean([r['avoid'] for r in fo]) * 100))
