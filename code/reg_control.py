# -*- coding: utf-8 -*-
"""Control robustness: do media/sensitive/avoid effects survive meeting-position and reporter-camp controls?"""
import sys, re, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

R = json.load(open('features_all.json', encoding='utf-8'))
F = pd.DataFrame(R)
for c in ['avoid','sensitive','q_china','acc','multiq']: F[c]=F[c].astype(int)
F['foreign']=(F.media=='foreign').astype(int)
F['adv']=(F.acc|F.multiq).astype(int)
F['llen_q']=np.log1p(F.len_q); F['llen_a']=np.log1p(F.len_a)
F['order']=F.groupby('file').cumcount()
F['frac']=F.groupby('file')['order'].transform(lambda x: x/(len(x)-1) if len(x)>1 else 0)

def report(md, tag, y):
    print('== %s: %s ==' % (y, tag))
    for k in md.params.index:
        if k=='Intercept': continue
        print('   %-12s % .4f  z=%6.2f  p=%.3g' % (k, md.params[k], md.tvalues[k], md.pvalues[k]))

# OLS (fast, robust HCO) since meeting var was ~0
fe = ['foreign','q_china','sensitive','avoid','adv','llen_q','llen_a','frac']
f = 'cos_sem ~ ' + ' + '.join(fe) + ' + C(who)'
m1 = smf.ols(f, F).fit(cov_type='HC1')
report(m1,'control frac (cos)', 'cos_sem')
f2 = 'mirror ~ ' + ' + '.join(fe) + ' + C(who)'
m2 = smf.ols(f2, F).fit(cov_type='HC1')
report(m2,'control frac (mirror)','mirror')

# foreign effect before vs after adding frac (partial): just show foreign coef
print('\nforeign coef cos: no-frac %.4f (p=%.2g)  with-frac %.4f (p=%.2g)'
      % (smf.ols('cos_sem ~ foreign+llen_a',F).fit().params['foreign'],
         smf.ols('cos_sem ~ foreign+llen_a',F).fit().pvalues['foreign'],
         m1.params['foreign'], m1.pvalues['foreign']))
