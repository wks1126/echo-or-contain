import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('features_all.json', encoding='utf-8'))
F = pd.DataFrame(R)
for c in ['avoid','sensitive','acc','multiq']: F[c]=F[c].astype(int)
F['fo'] = F.groupby(['file','rep']).cumcount() + 1
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':0.3,'font.family':'DejaVu Sans'})
def desp(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)

# ---- Fig3: mean semantic alignment by media x sensitivity (bar with error), and avoid line ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
def barmeans(ax, col, xlabel, title):
    groups = []
    for media in ['cn','foreign']:
        for s in [0,1]:
            g = F[(F.media==media)&(F[col]==s)].cos_sem
            groups.append((f"{media}\nsens={s}", g.mean(), g.std()/np.sqrt(len(g))))
    labs=[g[0] for g in groups]; means=[g[1] for g in groups]; errs=[g[2] for g in groups]
    ax.bar(labs, means, yerr=errs, capsize=4,
           color=['#4C72B0','#9BB0D8','#C44E52','#E8A0A3'])
    for i,m in enumerate(means): ax.text(i, m+0.01, f"{m:.2f}", ha='center', fontsize=9)
    ax.set_ylabel('Semantic alignment (cos)'); ax.set_title(title); desp(ax)
barmeans(axes[0], 'sensitive', 'media x sensitivity', '(a) By media × sensitivity')
# avoid rate by media/sensitive
ax=axes[1]
for media in ['cn','foreign']:
    for s in [0,1]:
        g=F[(F.media==media)&(F.sensitive==s)]
        ax.bar(f"{media}\nsens={s}", g.avoid.mean()*100, color=('#4C72B0' if media=='cn' else '#C44E52'))
ax.set_ylabel('Evasive answers (%)'); ax.set_title('(b) Avoidance by media × sensitivity'); desp(ax)
plt.tight_layout(); plt.savefig('fig3_interaction.png', dpi=300); plt.close()

# ---- Fig4: forest of model coefficients (cos & mirror) ----
coef = [
 ('foreign', -0.0605, 0.0053, -0.0894, 0.0044, 'foreign media'),
 ('q_china',  0.0552, 0.0063,  0.0009, 0.0053, 'China-focused q'),
 ('sensitive',-0.0218,0.0050, -0.0253,0.0042, 'sensitive topic'),
 ('avoid',   -0.0441, 0.0102, -0.0205,0.0086, 'evasive answer'),
 ('adv',      0.0194, 0.0052,  0.0169,0.0043, 'multi-part q'),
 ('llen_a',   0.1095, 0.0035,  0.0823,0.0030, 'ln(answer len)'),
 ('llen_q',   0.0067, 0.0055, -0.0840,0.0046, 'ln(question len)'),
 ('win_hist',-0.0013, 0.0049,  0.0102,0.0040, 'earlier window'),
]
fig, ax = plt.subplots(figsize=(6.6, 5.0))
y = np.arange(len(coef))[::-1]
h = 0.36
for j,(b,se_idx) in enumerate([(0,1),(2,3)]):
    bv = [c[b] for c in coef]; er=[c[se_idx] for c in coef]
    off = 0 if j==0 else h
    color = '#4C72B0' if j==0 else '#C44E52'
    ax.barh(y+off, bv, height=h, color=color, alpha=.85)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1,color='#4C72B0',label='semantic cosine'),
    plt.Rectangle((0,0),1,1,color='#C44E52',label='lexical echo')],
    loc='lower right', frameon=False)
ax.set_yticks(y); ax.set_yticklabels([c[4] for c in coef])
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Coefficient (mixed-effects model)')
ax.set_title('Predictors of alignment (semantic & lexical)')
ax.set_xlim(-0.16, 0.14); desp(ax)
plt.tight_layout(); plt.savefig('fig4_forest.png', dpi=300); plt.close()

# ---- Fig5: follow-up trajectory ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
f1=F[F.fo==1]; fl=F[F.fo>1]
means=[f1.cos_sem.mean(), fl.cos_sem.mean()]
errs=[f1.cos_sem.std()/np.sqrt(len(f1)), fl.cos_sem.std()/np.sqrt(len(fl))]
ax.bar(['First question\nby an outlet','Follow-up\n(same outlet)'], means, yerr=errs, capsize=5,
       color=['#55A868','#C44E52'])
for i,m in enumerate(means): ax.text(i,m+0.008,f"{m:.2f}",ha='center')
ax.set_ylabel('Semantic alignment (cos)')
ax2=ax.twinx()
ax2.plot(['First','Follow'], [f1.avoid.mean()*100, fl.avoid.mean()*100], 'o-', color='k')
ax2.set_ylabel('Evasive answers (%)', color='k')
ax.set_title('Follow-up: containment persists'); desp(ax); desp(ax2)
plt.tight_layout(); plt.savefig('fig5_followup.png', dpi=300); plt.close()
print('figures saved: fig3_interaction, fig4_forest, fig5_followup')
